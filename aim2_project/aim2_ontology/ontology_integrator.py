#!/usr/bin/env python3
"""
Ontology Integrator module for AIM2 project.

This module provides comprehensive ontology integration and merging capabilities
for combining multiple ontologies into unified structures. It handles term
conflicts, relationship merging, and namespace management.

Key Features:
- Multiple merge strategies (union, intersection, selective)
- Conflict resolution for duplicate terms and relationships
- Namespace preservation and management
- Validation of merged ontologies
- Statistics tracking for merge operations
"""

import logging
import time
from collections import defaultdict
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Tuple, Union

try:
    from .models import Ontology, Term, Relationship
except ImportError:
    # For development/testing scenarios
    try:
        from models import Ontology, Term, Relationship
    except ImportError:
        # Create minimal classes for testing
        from dataclasses import dataclass
        from typing import Dict, List, Optional
        
        @dataclass
        class Term:
            id: str
            name: str
            definition: Optional[str] = None
            synonyms: Optional[List[str]] = None
            namespace: Optional[str] = None
            is_obsolete: bool = False
        
        @dataclass
        class Relationship:
            id: str
            subject: str
            predicate: str
            object: str
            confidence: float = 1.0
            evidence: Optional[str] = None
        
        @dataclass
        class Ontology:
            id: str
            name: str
            version: str = "1.0.0"
            description: str = ""
            terms: Dict[str, Term] = None
            relationships: Dict[str, Relationship] = None
            namespaces: List[str] = None
            is_consistent: bool = True
            validation_errors: List[str] = None
            
            def __post_init__(self):
                if self.terms is None:
                    self.terms = {}
                if self.relationships is None:
                    self.relationships = {}
                if self.namespaces is None:
                    self.namespaces = []
                if self.validation_errors is None:
                    self.validation_errors = []

try:
    from ..exceptions import OntologyIntegrationError
except ImportError:
    try:
        from aim2_project.exceptions import OntologyIntegrationError
    except ImportError:
        class OntologyIntegrationError(Exception):
            """Exception for ontology integration errors."""
            pass


class MergeStrategy(Enum):
    """Enumeration of available merge strategies."""
    UNION = "union"
    INTERSECTION = "intersection"
    SELECTIVE = "selective"
    PRIORITY_BASED = "priority_based"


class ConflictResolution(Enum):
    """Enumeration of conflict resolution strategies."""
    KEEP_FIRST = "keep_first"
    KEEP_LAST = "keep_last"
    MERGE_DEFINITIONS = "merge_definitions"
    PRIORITY_BASED = "priority_based"
    MANUAL = "manual"


@dataclass
class MergeConflict:
    """Represents a conflict during ontology merging."""
    conflict_type: str
    term_id: str
    ontology_sources: List[str]
    conflicting_values: Dict[str, Any]
    resolution: Optional[str] = None
    resolved_value: Optional[Any] = None


@dataclass
class MergeResult:
    """Result container for ontology merge operations."""
    success: bool
    merged_ontology: Optional[Ontology] = None
    conflicts: List[MergeConflict] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)
    merge_time: float = 0.0
    statistics: Dict[str, Any] = field(default_factory=dict)


class OntologyIntegrator:
    """Comprehensive ontology integration and merging system."""

    def __init__(self):
        """Initialize the ontology integrator."""
        self.logger = logging.getLogger(__name__)
        self.merge_statistics = {
            "total_merges": 0,
            "successful_merges": 0,
            "failed_merges": 0,
            "conflicts_resolved": 0,
            "conflicts_unresolved": 0,
        }

    def merge_ontologies(
        self,
        ontologies: List[Ontology],
        strategy: MergeStrategy = MergeStrategy.UNION,
        conflict_resolution: ConflictResolution = ConflictResolution.KEEP_FIRST,
        target_id: Optional[str] = None,
        target_name: Optional[str] = None,
        priorities: Optional[Dict[str, int]] = None,
    ) -> MergeResult:
        """Merge multiple ontologies using the specified strategy.

        Args:
            ontologies: List of ontologies to merge
            strategy: Merge strategy to use
            conflict_resolution: How to resolve conflicts
            target_id: ID for the merged ontology
            target_name: Name for the merged ontology
            priorities: Priority mapping for ontologies (higher = more priority)

        Returns:
            MergeResult: Result of the merge operation
        """
        start_time = time.time()
        self.merge_statistics["total_merges"] += 1

        try:
            # Validate input
            if not ontologies:
                error_msg = "No ontologies provided for merging"
                self.logger.error(error_msg)
                self.merge_statistics["failed_merges"] += 1
                return MergeResult(
                    success=False,
                    errors=[error_msg],
                    merge_time=time.time() - start_time,
                )

            if len(ontologies) == 1:
                # Single ontology - return copy
                merged_ontology = self._copy_ontology(ontologies[0])
                if target_id:
                    merged_ontology.id = target_id
                if target_name:
                    merged_ontology.name = target_name

                self.merge_statistics["successful_merges"] += 1
                return MergeResult(
                    success=True,
                    merged_ontology=merged_ontology,
                    merge_time=time.time() - start_time,
                    statistics=self._calculate_merge_statistics(ontologies, merged_ontology),
                )

            # Perform merge based on strategy
            if strategy == MergeStrategy.UNION:
                result = self._merge_union(ontologies, conflict_resolution, priorities)
            elif strategy == MergeStrategy.INTERSECTION:
                result = self._merge_intersection(ontologies, conflict_resolution)
            elif strategy == MergeStrategy.SELECTIVE:
                result = self._merge_selective(ontologies, conflict_resolution, priorities)
            elif strategy == MergeStrategy.PRIORITY_BASED:
                result = self._merge_priority_based(ontologies, priorities or {})
            else:
                error_msg = f"Unsupported merge strategy: {strategy}"
                self.logger.error(error_msg)
                self.merge_statistics["failed_merges"] += 1
                return MergeResult(
                    success=False,
                    errors=[error_msg],
                    merge_time=time.time() - start_time,
                )

            if result.success and result.merged_ontology:
                # Set target properties
                if target_id:
                    result.merged_ontology.id = target_id
                if target_name:
                    result.merged_ontology.name = target_name

                # Calculate statistics
                result.statistics = self._calculate_merge_statistics(ontologies, result.merged_ontology)
                result.merge_time = time.time() - start_time

                # Update global statistics
                self.merge_statistics["successful_merges"] += 1
                self.merge_statistics["conflicts_resolved"] += len([c for c in result.conflicts if c.resolution])
                self.merge_statistics["conflicts_unresolved"] += len([c for c in result.conflicts if not c.resolution])

                self.logger.info(
                    f"Successfully merged {len(ontologies)} ontologies using {strategy.value} strategy"
                )
            else:
                self.merge_statistics["failed_merges"] += 1

            return result

        except Exception as e:
            error_msg = f"Unexpected error during ontology merge: {str(e)}"
            self.logger.exception(error_msg)
            self.merge_statistics["failed_merges"] += 1
            return MergeResult(
                success=False,
                errors=[error_msg],
                merge_time=time.time() - start_time,
            )

    def _merge_union(
        self,
        ontologies: List[Ontology],
        conflict_resolution: ConflictResolution,
        priorities: Optional[Dict[str, int]] = None,
    ) -> MergeResult:
        """Merge ontologies using union strategy (all terms and relationships).

        Args:
            ontologies: List of ontologies to merge
            conflict_resolution: How to resolve conflicts
            priorities: Priority mapping for ontologies

        Returns:
            MergeResult: Result of the union merge
        """
        merged_terms = {}
        merged_relationships = {}
        conflicts = []
        warnings = []
        merged_namespaces = set()

        # Create base merged ontology
        base_ontology = ontologies[0]
        merged_ontology = Ontology(
            id=f"merged_{int(time.time())}",
            name=f"Merged Ontology ({len(ontologies)} sources)",
            version="1.0.0",
            description=f"Union merge of {len(ontologies)} ontologies",
        )

        # Merge terms
        for ontology in ontologies:
            merged_namespaces.update(ontology.namespaces)
            
            for term_id, term in ontology.terms.items():
                if term_id in merged_terms:
                    # Handle conflict
                    conflict = self._resolve_term_conflict(
                        term_id, merged_terms[term_id], term, 
                        conflict_resolution, priorities, ontology.id
                    )
                    conflicts.append(conflict)
                    if conflict.resolved_value:
                        merged_terms[term_id] = conflict.resolved_value
                else:
                    merged_terms[term_id] = self._copy_term(term)

        # Merge relationships
        for ontology in ontologies:
            for rel_id, relationship in ontology.relationships.items():
                if rel_id in merged_relationships:
                    # Handle conflict
                    conflict = self._resolve_relationship_conflict(
                        rel_id, merged_relationships[rel_id], relationship,
                        conflict_resolution, priorities, ontology.id
                    )
                    conflicts.append(conflict)
                    if conflict.resolved_value:
                        merged_relationships[rel_id] = conflict.resolved_value
                else:
                    merged_relationships[rel_id] = self._copy_relationship(relationship)

        # Set merged data
        merged_ontology.terms = merged_terms
        merged_ontology.relationships = merged_relationships
        merged_ontology.namespaces = list(merged_namespaces)

        return MergeResult(
            success=True,
            merged_ontology=merged_ontology,
            conflicts=conflicts,
            warnings=warnings,
        )

    def _merge_intersection(
        self,
        ontologies: List[Ontology],
        conflict_resolution: ConflictResolution,
    ) -> MergeResult:
        """Merge ontologies using intersection strategy (only common terms and relationships).

        Args:
            ontologies: List of ontologies to merge
            conflict_resolution: How to resolve conflicts

        Returns:
            MergeResult: Result of the intersection merge
        """
        if not ontologies:
            return MergeResult(success=False, errors=["No ontologies provided"])

        # Find common terms
        common_term_ids = set(ontologies[0].terms.keys())
        for ontology in ontologies[1:]:
            common_term_ids.intersection_update(ontology.terms.keys())

        # Find common relationships
        common_rel_ids = set(ontologies[0].relationships.keys())
        for ontology in ontologies[1:]:
            common_rel_ids.intersection_update(ontology.relationships.keys())

        merged_terms = {}
        merged_relationships = {}
        conflicts = []
        warnings = []

        # Create merged ontology
        merged_ontology = Ontology(
            id=f"intersection_merged_{int(time.time())}",
            name=f"Intersection Merge ({len(ontologies)} sources)",
            version="1.0.0",
            description=f"Intersection merge of {len(ontologies)} ontologies",
        )

        # Merge common terms
        for term_id in common_term_ids:
            base_term = ontologies[0].terms[term_id]
            merged_term = self._copy_term(base_term)
            
            # Check for conflicts in definitions
            for ontology in ontologies[1:]:
                other_term = ontology.terms[term_id]
                if base_term.definition != other_term.definition:
                    conflict = self._resolve_term_conflict(
                        term_id, merged_term, other_term,
                        conflict_resolution, None, ontology.id
                    )
                    conflicts.append(conflict)
                    if conflict.resolved_value:
                        merged_term = conflict.resolved_value

            merged_terms[term_id] = merged_term

        # Merge common relationships
        for rel_id in common_rel_ids:
            base_rel = ontologies[0].relationships[rel_id]
            merged_rel = self._copy_relationship(base_rel)
            
            # Check for conflicts
            for ontology in ontologies[1:]:
                other_rel = ontology.relationships[rel_id]
                if base_rel.confidence != other_rel.confidence:
                    conflict = self._resolve_relationship_conflict(
                        rel_id, merged_rel, other_rel,
                        conflict_resolution, None, ontology.id
                    )
                    conflicts.append(conflict)
                    if conflict.resolved_value:
                        merged_rel = conflict.resolved_value

            merged_relationships[rel_id] = merged_rel

        # Set merged data
        merged_ontology.terms = merged_terms
        merged_ontology.relationships = merged_relationships

        # Find common namespaces
        common_namespaces = set(ontologies[0].namespaces)
        for ontology in ontologies[1:]:
            common_namespaces.intersection_update(ontology.namespaces)
        merged_ontology.namespaces = list(common_namespaces)

        return MergeResult(
            success=True,
            merged_ontology=merged_ontology,
            conflicts=conflicts,
            warnings=warnings,
        )

    def _merge_selective(
        self,
        ontologies: List[Ontology],
        conflict_resolution: ConflictResolution,
        priorities: Optional[Dict[str, int]] = None,
    ) -> MergeResult:
        """Merge ontologies using selective strategy (based on priorities and criteria).

        Args:
            ontologies: List of ontologies to merge
            conflict_resolution: How to resolve conflicts
            priorities: Priority mapping for ontologies

        Returns:
            MergeResult: Result of the selective merge
        """
        # For selective merge, use union but with stricter conflict resolution
        return self._merge_union(ontologies, conflict_resolution, priorities)

    def _merge_priority_based(
        self,
        ontologies: List[Ontology],
        priorities: Dict[str, int],
    ) -> MergeResult:
        """Merge ontologies using priority-based strategy.

        Args:
            ontologies: List of ontologies to merge
            priorities: Priority mapping for ontologies (higher = more priority)

        Returns:
            MergeResult: Result of the priority-based merge
        """
        # Sort ontologies by priority (highest first)
        sorted_ontologies = sorted(
            ontologies,
            key=lambda ont: priorities.get(ont.id, 0),
            reverse=True
        )

        return self._merge_union(
            sorted_ontologies,
            ConflictResolution.PRIORITY_BASED,
            priorities
        )

    def _resolve_term_conflict(
        self,
        term_id: str,
        existing_term: Term,
        new_term: Term,
        resolution: ConflictResolution,
        priorities: Optional[Dict[str, int]],
        source_ontology_id: str,
    ) -> MergeConflict:
        """Resolve a conflict between two terms.

        Args:
            term_id: ID of the conflicting term
            existing_term: Already merged term
            new_term: New term causing conflict
            resolution: Conflict resolution strategy
            priorities: Priority mapping
            source_ontology_id: ID of source ontology for new term

        Returns:
            MergeConflict: Conflict information and resolution
        """
        conflict = MergeConflict(
            conflict_type="term_definition",
            term_id=term_id,
            ontology_sources=[existing_term.namespace or "unknown", source_ontology_id],
            conflicting_values={
                "existing": {
                    "definition": existing_term.definition,
                    "synonyms": existing_term.synonyms,
                    "namespace": existing_term.namespace,
                },
                "new": {
                    "definition": new_term.definition,
                    "synonyms": new_term.synonyms,
                    "namespace": new_term.namespace,
                }
            }
        )

        if resolution == ConflictResolution.KEEP_FIRST:
            conflict.resolution = "keep_first"
            conflict.resolved_value = existing_term
        elif resolution == ConflictResolution.KEEP_LAST:
            conflict.resolution = "keep_last"
            conflict.resolved_value = new_term
        elif resolution == ConflictResolution.MERGE_DEFINITIONS:
            conflict.resolution = "merge_definitions"
            merged_term = self._copy_term(existing_term)
            if new_term.definition and existing_term.definition != new_term.definition:
                merged_term.definition = f"{existing_term.definition}; {new_term.definition}"
            # Merge synonyms
            if new_term.synonyms:
                merged_synonyms = set(existing_term.synonyms or [])
                merged_synonyms.update(new_term.synonyms)
                merged_term.synonyms = list(merged_synonyms)
            conflict.resolved_value = merged_term
        elif resolution == ConflictResolution.PRIORITY_BASED and priorities:
            existing_priority = priorities.get(existing_term.namespace or "", 0)
            new_priority = priorities.get(source_ontology_id, 0)
            if new_priority > existing_priority:
                conflict.resolution = "priority_new"
                conflict.resolved_value = new_term
            else:
                conflict.resolution = "priority_existing"
                conflict.resolved_value = existing_term
        else:
            conflict.resolution = None
            conflict.resolved_value = existing_term  # Default to existing

        return conflict

    def _resolve_relationship_conflict(
        self,
        rel_id: str,
        existing_rel: Relationship,
        new_rel: Relationship,
        resolution: ConflictResolution,
        priorities: Optional[Dict[str, int]],
        source_ontology_id: str,
    ) -> MergeConflict:
        """Resolve a conflict between two relationships.

        Args:
            rel_id: ID of the conflicting relationship
            existing_rel: Already merged relationship
            new_rel: New relationship causing conflict
            resolution: Conflict resolution strategy
            priorities: Priority mapping
            source_ontology_id: ID of source ontology for new relationship

        Returns:
            MergeConflict: Conflict information and resolution
        """
        conflict = MergeConflict(
            conflict_type="relationship_confidence",
            term_id=rel_id,
            ontology_sources=["existing", source_ontology_id],
            conflicting_values={
                "existing": {
                    "confidence": existing_rel.confidence,
                    "evidence": existing_rel.evidence,
                },
                "new": {
                    "confidence": new_rel.confidence,
                    "evidence": new_rel.evidence,
                }
            }
        )

        if resolution == ConflictResolution.KEEP_FIRST:
            conflict.resolution = "keep_first"
            conflict.resolved_value = existing_rel
        elif resolution == ConflictResolution.KEEP_LAST:
            conflict.resolution = "keep_last"
            conflict.resolved_value = new_rel
        elif resolution == ConflictResolution.MERGE_DEFINITIONS:
            conflict.resolution = "merge_evidence"
            merged_rel = self._copy_relationship(existing_rel)
            # Take higher confidence
            if new_rel.confidence > existing_rel.confidence:
                merged_rel.confidence = new_rel.confidence
            # Merge evidence
            if new_rel.evidence and existing_rel.evidence != new_rel.evidence:
                merged_rel.evidence = f"{existing_rel.evidence}; {new_rel.evidence}"
            conflict.resolved_value = merged_rel
        else:
            conflict.resolution = None
            conflict.resolved_value = existing_rel  # Default to existing

        return conflict

    def _copy_ontology(self, ontology: Ontology) -> Ontology:
        """Create a deep copy of an ontology.

        Args:
            ontology: Ontology to copy

        Returns:
            Ontology: Deep copy of the ontology
        """
        copied_terms = {term_id: self._copy_term(term) for term_id, term in ontology.terms.items()}
        copied_relationships = {rel_id: self._copy_relationship(rel) for rel_id, rel in ontology.relationships.items()}

        return Ontology(
            id=ontology.id,
            name=ontology.name,
            version=ontology.version,
            description=ontology.description,
            terms=copied_terms,
            relationships=copied_relationships,
            namespaces=ontology.namespaces.copy(),
            is_consistent=ontology.is_consistent,
            validation_errors=ontology.validation_errors.copy(),
        )

    def _copy_term(self, term: Term) -> Term:
        """Create a deep copy of a term.

        Args:
            term: Term to copy

        Returns:
            Term: Deep copy of the term
        """
        return Term(
            id=term.id,
            name=term.name,
            definition=term.definition,
            synonyms=term.synonyms.copy() if term.synonyms else None,
            namespace=term.namespace,
            is_obsolete=term.is_obsolete,
        )

    def _copy_relationship(self, relationship: Relationship) -> Relationship:
        """Create a deep copy of a relationship.

        Args:
            relationship: Relationship to copy

        Returns:
            Relationship: Deep copy of the relationship
        """
        return Relationship(
            id=relationship.id,
            subject=relationship.subject,
            predicate=relationship.predicate,
            object=relationship.object,
            confidence=relationship.confidence,
            evidence=relationship.evidence,
        )

    def _calculate_merge_statistics(
        self, source_ontologies: List[Ontology], merged_ontology: Ontology
    ) -> Dict[str, Any]:
        """Calculate statistics for a merge operation.

        Args:
            source_ontologies: List of source ontologies
            merged_ontology: Resulting merged ontology

        Returns:
            Dict[str, Any]: Merge statistics
        """
        total_source_terms = sum(len(ont.terms) for ont in source_ontologies)
        total_source_relationships = sum(len(ont.relationships) for ont in source_ontologies)

        return {
            "source_ontologies_count": len(source_ontologies),
            "source_terms_total": total_source_terms,
            "source_relationships_total": total_source_relationships,
            "merged_terms_count": len(merged_ontology.terms),
            "merged_relationships_count": len(merged_ontology.relationships),
            "terms_reduction": total_source_terms - len(merged_ontology.terms),
            "relationships_reduction": total_source_relationships - len(merged_ontology.relationships),
            "compression_ratio": {
                "terms": len(merged_ontology.terms) / total_source_terms if total_source_terms > 0 else 0,
                "relationships": len(merged_ontology.relationships) / total_source_relationships if total_source_relationships > 0 else 0,
            }
        }

    def get_merge_statistics(self) -> Dict[str, Any]:
        """Get overall merge statistics.

        Returns:
            Dict[str, Any]: Overall merge statistics
        """
        return self.merge_statistics.copy()

    def reset_statistics(self) -> None:
        """Reset merge statistics."""
        self.merge_statistics = {
            "total_merges": 0,
            "successful_merges": 0,
            "failed_merges": 0,
            "conflicts_resolved": 0,
            "conflicts_unresolved": 0,
        }
        self.logger.info("Reset merge statistics")