#!/usr/bin/env python3
"""
Ontology Trimmer module for AIM2 project.

This module provides comprehensive ontology subset extraction and trimming
capabilities for creating focused, domain-specific ontologies from larger
ontological structures.

Key Features:
- Multiple extraction strategies (term-based, namespace-based, depth-based)
- Relationship preservation and dependency tracking
- Validation of extracted subsets
- Statistics tracking for extraction operations
- Support for custom filtering criteria
"""

import logging
import time
from collections import defaultdict, deque
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Union

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


class ExtractionStrategy(Enum):
    """Enumeration of available extraction strategies."""
    TERM_LIST = "term_list"
    NAMESPACE = "namespace"
    DEPTH_LIMITED = "depth_limited"
    CUSTOM_FILTER = "custom_filter"
    RELATIONSHIP_BASED = "relationship_based"
    HIERARCHICAL = "hierarchical"


class DependencyMode(Enum):
    """Enumeration of dependency inclusion modes."""
    NONE = "none"
    DIRECT = "direct"
    TRANSITIVE = "transitive"
    FULL_CLOSURE = "full_closure"


@dataclass
class ExtractionCriteria:
    """Criteria for ontology subset extraction."""
    strategy: ExtractionStrategy
    term_ids: Optional[List[str]] = None
    namespaces: Optional[List[str]] = None
    max_depth: Optional[int] = None
    custom_filter: Optional[Callable[[Term], bool]] = None
    relationship_types: Optional[List[str]] = None
    root_terms: Optional[List[str]] = None
    dependency_mode: DependencyMode = DependencyMode.DIRECT
    include_obsolete: bool = False
    min_confidence: float = 0.0


@dataclass
class ExtractionResult:
    """Result container for ontology extraction operations."""
    success: bool
    extracted_ontology: Optional[Ontology] = None
    warnings: List[str] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)
    extraction_time: float = 0.0
    statistics: Dict[str, Any] = field(default_factory=dict)
    excluded_terms: List[str] = field(default_factory=list)
    excluded_relationships: List[str] = field(default_factory=list)


class OntologyTrimmer:
    """Comprehensive ontology subset extraction and trimming system."""

    def __init__(self):
        """Initialize the ontology trimmer."""
        self.logger = logging.getLogger(__name__)
        self.extraction_statistics = {
            "total_extractions": 0,
            "successful_extractions": 0,
            "failed_extractions": 0,
            "terms_extracted": 0,
            "relationships_extracted": 0,
        }

    def extract_subset(
        self,
        ontology: Ontology,
        criteria: ExtractionCriteria,
        target_id: Optional[str] = None,
        target_name: Optional[str] = None,
    ) -> ExtractionResult:
        """Extract a subset from an ontology based on specified criteria.

        Args:
            ontology: Source ontology to extract from
            criteria: Extraction criteria and strategy
            target_id: ID for the extracted ontology
            target_name: Name for the extracted ontology

        Returns:
            ExtractionResult: Result of the extraction operation
        """
        start_time = time.time()
        self.extraction_statistics["total_extractions"] += 1

        try:
            # Validate input
            if not ontology or not ontology.terms:
                error_msg = "Invalid or empty ontology provided for extraction"
                self.logger.error(error_msg)
                self.extraction_statistics["failed_extractions"] += 1
                return ExtractionResult(
                    success=False,
                    errors=[error_msg],
                    extraction_time=time.time() - start_time,
                )

            # Perform extraction based on strategy
            if criteria.strategy == ExtractionStrategy.TERM_LIST:
                result = self._extract_by_term_list(ontology, criteria)
            elif criteria.strategy == ExtractionStrategy.NAMESPACE:
                result = self._extract_by_namespace(ontology, criteria)
            elif criteria.strategy == ExtractionStrategy.DEPTH_LIMITED:
                result = self._extract_by_depth(ontology, criteria)
            elif criteria.strategy == ExtractionStrategy.CUSTOM_FILTER:
                result = self._extract_by_custom_filter(ontology, criteria)
            elif criteria.strategy == ExtractionStrategy.RELATIONSHIP_BASED:
                result = self._extract_by_relationships(ontology, criteria)
            elif criteria.strategy == ExtractionStrategy.HIERARCHICAL:
                result = self._extract_hierarchical(ontology, criteria)
            else:
                error_msg = f"Unsupported extraction strategy: {criteria.strategy}"
                self.logger.error(error_msg)
                self.extraction_statistics["failed_extractions"] += 1
                return ExtractionResult(
                    success=False,
                    errors=[error_msg],
                    extraction_time=time.time() - start_time,
                )

            if result.success and result.extracted_ontology:
                # Set target properties
                if target_id:
                    result.extracted_ontology.id = target_id
                if target_name:
                    result.extracted_ontology.name = target_name

                # Calculate statistics
                result.statistics = self._calculate_extraction_statistics(ontology, result.extracted_ontology)
                result.extraction_time = time.time() - start_time

                # Update global statistics
                self.extraction_statistics["successful_extractions"] += 1
                self.extraction_statistics["terms_extracted"] += len(result.extracted_ontology.terms)
                self.extraction_statistics["relationships_extracted"] += len(result.extracted_ontology.relationships)

                self.logger.info(
                    f"Successfully extracted subset using {criteria.strategy.value} strategy: "
                    f"{len(result.extracted_ontology.terms)} terms, "
                    f"{len(result.extracted_ontology.relationships)} relationships"
                )
            else:
                self.extraction_statistics["failed_extractions"] += 1

            return result

        except Exception as e:
            error_msg = f"Unexpected error during ontology extraction: {str(e)}"
            self.logger.exception(error_msg)
            self.extraction_statistics["failed_extractions"] += 1
            return ExtractionResult(
                success=False,
                errors=[error_msg],
                extraction_time=time.time() - start_time,
            )

    def _extract_by_term_list(
        self, ontology: Ontology, criteria: ExtractionCriteria
    ) -> ExtractionResult:
        """Extract subset based on a specific list of term IDs.

        Args:
            ontology: Source ontology
            criteria: Extraction criteria with term_ids

        Returns:
            ExtractionResult: Result of the extraction
        """
        if not criteria.term_ids:
            return ExtractionResult(
                success=False,
                errors=["No term IDs provided for term list extraction"]
            )

        extracted_terms = {}
        excluded_terms = []
        warnings = []

        # Extract specified terms
        for term_id in criteria.term_ids:
            if term_id in ontology.terms:
                term = ontology.terms[term_id]
                if not criteria.include_obsolete and term.is_obsolete:
                    excluded_terms.append(term_id)
                    warnings.append(f"Excluded obsolete term: {term_id}")
                else:
                    extracted_terms[term_id] = self._copy_term(term)
            else:
                excluded_terms.append(term_id)
                warnings.append(f"Term not found in ontology: {term_id}")

        # Handle dependencies
        if criteria.dependency_mode != DependencyMode.NONE:
            additional_terms = self._resolve_dependencies(
                ontology, list(extracted_terms.keys()), criteria.dependency_mode
            )
            for term_id, term in additional_terms.items():
                if term_id not in extracted_terms:
                    extracted_terms[term_id] = term

        # Extract relevant relationships
        extracted_relationships, excluded_relationships = self._extract_relationships(
            ontology, extracted_terms, criteria
        )

        # Create extracted ontology
        extracted_ontology = self._create_extracted_ontology(
            ontology, extracted_terms, extracted_relationships, "term_list"
        )

        return ExtractionResult(
            success=True,
            extracted_ontology=extracted_ontology,
            warnings=warnings,
            excluded_terms=excluded_terms,
            excluded_relationships=excluded_relationships,
        )

    def _extract_by_namespace(
        self, ontology: Ontology, criteria: ExtractionCriteria
    ) -> ExtractionResult:
        """Extract subset based on namespaces.

        Args:
            ontology: Source ontology
            criteria: Extraction criteria with namespaces

        Returns:
            ExtractionResult: Result of the extraction
        """
        if not criteria.namespaces:
            return ExtractionResult(
                success=False,
                errors=["No namespaces provided for namespace extraction"]
            )

        extracted_terms = {}
        excluded_terms = []
        warnings = []

        # Extract terms from specified namespaces
        for term_id, term in ontology.terms.items():
            if term.namespace in criteria.namespaces:
                if not criteria.include_obsolete and term.is_obsolete:
                    excluded_terms.append(term_id)
                    warnings.append(f"Excluded obsolete term: {term_id}")
                else:
                    extracted_terms[term_id] = self._copy_term(term)
            else:
                excluded_terms.append(term_id)

        # Extract relevant relationships
        extracted_relationships, excluded_relationships = self._extract_relationships(
            ontology, extracted_terms, criteria
        )

        # Create extracted ontology
        extracted_ontology = self._create_extracted_ontology(
            ontology, extracted_terms, extracted_relationships, "namespace"
        )
        extracted_ontology.namespaces = criteria.namespaces

        return ExtractionResult(
            success=True,
            extracted_ontology=extracted_ontology,
            warnings=warnings,
            excluded_terms=excluded_terms,
            excluded_relationships=excluded_relationships,
        )

    def _extract_by_depth(
        self, ontology: Ontology, criteria: ExtractionCriteria
    ) -> ExtractionResult:
        """Extract subset based on hierarchical depth from root terms.

        Args:
            ontology: Source ontology
            criteria: Extraction criteria with max_depth and root_terms

        Returns:
            ExtractionResult: Result of the extraction
        """
        if criteria.max_depth is None or criteria.max_depth < 0:
            return ExtractionResult(
                success=False,
                errors=["Invalid max_depth for depth-limited extraction"]
            )

        root_terms = criteria.root_terms or self._find_root_terms(ontology)
        if not root_terms:
            return ExtractionResult(
                success=False,
                errors=["No root terms found or specified for depth-limited extraction"]
            )

        extracted_terms = {}
        excluded_terms = []
        warnings = []

        # Build relationship graph for traversal
        relationship_graph = self._build_relationship_graph(ontology)

        # Traverse from root terms up to max depth
        visited = set()
        queue = deque([(term_id, 0) for term_id in root_terms])

        while queue:
            term_id, depth = queue.popleft()

            if depth > criteria.max_depth or term_id in visited:
                continue

            visited.add(term_id)

            if term_id in ontology.terms:
                term = ontology.terms[term_id]
                if not criteria.include_obsolete and term.is_obsolete:
                    excluded_terms.append(term_id)
                    warnings.append(f"Excluded obsolete term: {term_id}")
                else:
                    extracted_terms[term_id] = self._copy_term(term)

                # Add children to queue
                if depth < criteria.max_depth:
                    for child_id in relationship_graph.get(term_id, []):
                        if child_id not in visited:
                            queue.append((child_id, depth + 1))

        # Extract relevant relationships
        extracted_relationships, excluded_relationships = self._extract_relationships(
            ontology, extracted_terms, criteria
        )

        # Create extracted ontology
        extracted_ontology = self._create_extracted_ontology(
            ontology, extracted_terms, extracted_relationships, "depth_limited"
        )

        return ExtractionResult(
            success=True,
            extracted_ontology=extracted_ontology,
            warnings=warnings,
            excluded_terms=excluded_terms,
            excluded_relationships=excluded_relationships,
        )

    def _extract_by_custom_filter(
        self, ontology: Ontology, criteria: ExtractionCriteria
    ) -> ExtractionResult:
        """Extract subset based on custom filter function.

        Args:
            ontology: Source ontology
            criteria: Extraction criteria with custom_filter

        Returns:
            ExtractionResult: Result of the extraction
        """
        if not criteria.custom_filter:
            return ExtractionResult(
                success=False,
                errors=["No custom filter provided for custom filter extraction"]
            )

        extracted_terms = {}
        excluded_terms = []
        warnings = []

        # Apply custom filter to all terms
        for term_id, term in ontology.terms.items():
            try:
                if criteria.custom_filter(term):
                    if not criteria.include_obsolete and term.is_obsolete:
                        excluded_terms.append(term_id)
                        warnings.append(f"Excluded obsolete term: {term_id}")
                    else:
                        extracted_terms[term_id] = self._copy_term(term)
                else:
                    excluded_terms.append(term_id)
            except Exception as e:
                excluded_terms.append(term_id)
                warnings.append(f"Custom filter error for term {term_id}: {str(e)}")

        # Extract relevant relationships
        extracted_relationships, excluded_relationships = self._extract_relationships(
            ontology, extracted_terms, criteria
        )

        # Create extracted ontology
        extracted_ontology = self._create_extracted_ontology(
            ontology, extracted_terms, extracted_relationships, "custom_filter"
        )

        return ExtractionResult(
            success=True,
            extracted_ontology=extracted_ontology,
            warnings=warnings,
            excluded_terms=excluded_terms,
            excluded_relationships=excluded_relationships,
        )

    def _extract_by_relationships(
        self, ontology: Ontology, criteria: ExtractionCriteria
    ) -> ExtractionResult:
        """Extract subset based on relationship types.

        Args:
            ontology: Source ontology
            criteria: Extraction criteria with relationship_types

        Returns:
            ExtractionResult: Result of the extraction
        """
        if not criteria.relationship_types:
            return ExtractionResult(
                success=False,
                errors=["No relationship types provided for relationship-based extraction"]
            )

        relevant_term_ids = set()
        extracted_relationships = {}
        excluded_relationships = []

        # Find relationships of specified types
        for rel_id, relationship in ontology.relationships.items():
            if relationship.predicate in criteria.relationship_types:
                if relationship.confidence >= criteria.min_confidence:
                    extracted_relationships[rel_id] = self._copy_relationship(relationship)
                    relevant_term_ids.add(relationship.subject)
                    relevant_term_ids.add(relationship.object)
                else:
                    excluded_relationships.append(rel_id)
            else:
                excluded_relationships.append(rel_id)

        # Extract terms involved in relevant relationships
        extracted_terms = {}
        excluded_terms = []
        warnings = []

        for term_id in relevant_term_ids:
            if term_id in ontology.terms:
                term = ontology.terms[term_id]
                if not criteria.include_obsolete and term.is_obsolete:
                    excluded_terms.append(term_id)
                    warnings.append(f"Excluded obsolete term: {term_id}")
                else:
                    extracted_terms[term_id] = self._copy_term(term)

        # Create extracted ontology
        extracted_ontology = self._create_extracted_ontology(
            ontology, extracted_terms, extracted_relationships, "relationship_based"
        )

        return ExtractionResult(
            success=True,
            extracted_ontology=extracted_ontology,
            warnings=warnings,
            excluded_terms=excluded_terms,
            excluded_relationships=excluded_relationships,
        )

    def _extract_hierarchical(
        self, ontology: Ontology, criteria: ExtractionCriteria
    ) -> ExtractionResult:
        """Extract subset maintaining hierarchical structure.

        Args:
            ontology: Source ontology
            criteria: Extraction criteria with root_terms

        Returns:
            ExtractionResult: Result of the extraction
        """
        root_terms = criteria.root_terms or self._find_root_terms(ontology)
        if not root_terms:
            return ExtractionResult(
                success=False,
                errors=["No root terms found or specified for hierarchical extraction"]
            )

        extracted_terms = {}
        excluded_terms = []
        warnings = []

        # Build hierarchical relationships
        hierarchy_graph = self._build_hierarchy_graph(ontology)

        # Extract complete hierarchies from root terms
        visited = set()
        for root_term_id in root_terms:
            self._extract_hierarchy_recursive(
                ontology, root_term_id, hierarchy_graph, extracted_terms,
                excluded_terms, warnings, visited, criteria
            )

        # Extract relevant relationships
        extracted_relationships, excluded_relationships = self._extract_relationships(
            ontology, extracted_terms, criteria
        )

        # Create extracted ontology
        extracted_ontology = self._create_extracted_ontology(
            ontology, extracted_terms, extracted_relationships, "hierarchical"
        )

        return ExtractionResult(
            success=True,
            extracted_ontology=extracted_ontology,
            warnings=warnings,
            excluded_terms=excluded_terms,
            excluded_relationships=excluded_relationships,
        )

    def _resolve_dependencies(
        self, ontology: Ontology, term_ids: List[str], dependency_mode: DependencyMode
    ) -> Dict[str, Term]:
        """Resolve term dependencies based on the specified mode.

        Args:
            ontology: Source ontology
            term_ids: List of term IDs to resolve dependencies for
            dependency_mode: Type of dependency resolution

        Returns:
            Dict[str, Term]: Additional terms to include
        """
        additional_terms = {}

        if dependency_mode == DependencyMode.NONE:
            return additional_terms

        # Build relationship graph
        relationship_graph = self._build_relationship_graph(ontology)

        if dependency_mode == DependencyMode.DIRECT:
            # Include directly related terms
            for term_id in term_ids:
                for related_id in relationship_graph.get(term_id, []):
                    if related_id in ontology.terms and related_id not in term_ids:
                        additional_terms[related_id] = self._copy_term(ontology.terms[related_id])

        elif dependency_mode in [DependencyMode.TRANSITIVE, DependencyMode.FULL_CLOSURE]:
            # Include transitively related terms
            visited = set(term_ids)
            queue = deque(term_ids)

            while queue:
                current_id = queue.popleft()
                for related_id in relationship_graph.get(current_id, []):
                    if related_id not in visited and related_id in ontology.terms:
                        visited.add(related_id)
                        additional_terms[related_id] = self._copy_term(ontology.terms[related_id])
                        queue.append(related_id)

        return additional_terms

    def _extract_relationships(
        self, ontology: Ontology, extracted_terms: Dict[str, Term], criteria: ExtractionCriteria
    ) -> Tuple[Dict[str, Relationship], List[str]]:
        """Extract relationships relevant to the extracted terms.

        Args:
            ontology: Source ontology
            extracted_terms: Dictionary of extracted terms
            criteria: Extraction criteria

        Returns:
            Tuple[Dict[str, Relationship], List[str]]: Extracted relationships and excluded relationship IDs
        """
        extracted_relationships = {}
        excluded_relationships = []

        for rel_id, relationship in ontology.relationships.items():
            # Check if both subject and object are in extracted terms
            if (relationship.subject in extracted_terms and 
                relationship.object in extracted_terms):
                
                # Check confidence threshold
                if relationship.confidence >= criteria.min_confidence:
                    extracted_relationships[rel_id] = self._copy_relationship(relationship)
                else:
                    excluded_relationships.append(rel_id)
            else:
                excluded_relationships.append(rel_id)

        return extracted_relationships, excluded_relationships

    def _build_relationship_graph(self, ontology: Ontology) -> Dict[str, List[str]]:
        """Build a graph of term relationships.

        Args:
            ontology: Source ontology

        Returns:
            Dict[str, List[str]]: Graph mapping term IDs to related term IDs
        """
        graph = defaultdict(list)
        
        for relationship in ontology.relationships.values():
            graph[relationship.subject].append(relationship.object)
            graph[relationship.object].append(relationship.subject)  # Bidirectional

        return dict(graph)

    def _build_hierarchy_graph(self, ontology: Ontology) -> Dict[str, List[str]]:
        """Build a hierarchical graph of term relationships.

        Args:
            ontology: Source ontology

        Returns:
            Dict[str, List[str]]: Graph mapping parent terms to child terms
        """
        hierarchy = defaultdict(list)
        
        # Common hierarchical predicates
        hierarchical_predicates = ["is_a", "part_of", "subclass_of", "child_of"]
        
        for relationship in ontology.relationships.values():
            if relationship.predicate in hierarchical_predicates:
                hierarchy[relationship.object].append(relationship.subject)  # parent -> child

        return dict(hierarchy)

    def _find_root_terms(self, ontology: Ontology) -> List[str]:
        """Find root terms (terms with no parents) in the ontology.

        Args:
            ontology: Source ontology

        Returns:
            List[str]: List of root term IDs
        """
        all_terms = set(ontology.terms.keys())
        child_terms = set()

        # Common hierarchical predicates
        hierarchical_predicates = ["is_a", "part_of", "subclass_of", "child_of"]

        for relationship in ontology.relationships.values():
            if relationship.predicate in hierarchical_predicates:
                child_terms.add(relationship.subject)

        root_terms = all_terms - child_terms
        return list(root_terms)

    def _extract_hierarchy_recursive(
        self, ontology: Ontology, term_id: str, hierarchy_graph: Dict[str, List[str]],
        extracted_terms: Dict[str, Term], excluded_terms: List[str], 
        warnings: List[str], visited: Set[str], criteria: ExtractionCriteria
    ) -> None:
        """Recursively extract hierarchical terms.

        Args:
            ontology: Source ontology
            term_id: Current term ID to process
            hierarchy_graph: Hierarchical relationship graph
            extracted_terms: Dictionary to store extracted terms
            excluded_terms: List to store excluded term IDs
            warnings: List to store warnings
            visited: Set of visited term IDs
            criteria: Extraction criteria
        """
        if term_id in visited or term_id not in ontology.terms:
            return

        visited.add(term_id)
        term = ontology.terms[term_id]

        if not criteria.include_obsolete and term.is_obsolete:
            excluded_terms.append(term_id)
            warnings.append(f"Excluded obsolete term: {term_id}")
        else:
            extracted_terms[term_id] = self._copy_term(term)

        # Recursively process children
        for child_id in hierarchy_graph.get(term_id, []):
            self._extract_hierarchy_recursive(
                ontology, child_id, hierarchy_graph, extracted_terms,
                excluded_terms, warnings, visited, criteria
            )

    def _create_extracted_ontology(
        self, source_ontology: Ontology, extracted_terms: Dict[str, Term],
        extracted_relationships: Dict[str, Relationship], strategy: str
    ) -> Ontology:
        """Create an ontology from extracted terms and relationships.

        Args:
            source_ontology: Original ontology
            extracted_terms: Extracted terms
            extracted_relationships: Extracted relationships
            strategy: Extraction strategy used

        Returns:
            Ontology: New ontology with extracted content
        """
        return Ontology(
            id=f"extracted_{int(time.time())}",
            name=f"Extracted from {source_ontology.name} ({strategy})",
            version=source_ontology.version,
            description=f"Subset extracted from {source_ontology.name} using {strategy} strategy",
            terms=extracted_terms,
            relationships=extracted_relationships,
            namespaces=list(set(term.namespace for term in extracted_terms.values() if term.namespace)),
            is_consistent=True,
            validation_errors=[],
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

    def _calculate_extraction_statistics(
        self, source_ontology: Ontology, extracted_ontology: Ontology
    ) -> Dict[str, Any]:
        """Calculate statistics for an extraction operation.

        Args:
            source_ontology: Original ontology
            extracted_ontology: Extracted ontology

        Returns:
            Dict[str, Any]: Extraction statistics
        """
        source_terms_count = len(source_ontology.terms)
        source_relationships_count = len(source_ontology.relationships)
        extracted_terms_count = len(extracted_ontology.terms)
        extracted_relationships_count = len(extracted_ontology.relationships)

        return {
            "source_terms_count": source_terms_count,
            "source_relationships_count": source_relationships_count,
            "extracted_terms_count": extracted_terms_count,
            "extracted_relationships_count": extracted_relationships_count,
            "terms_reduction": source_terms_count - extracted_terms_count,
            "relationships_reduction": source_relationships_count - extracted_relationships_count,
            "extraction_ratio": {
                "terms": extracted_terms_count / source_terms_count if source_terms_count > 0 else 0,
                "relationships": extracted_relationships_count / source_relationships_count if source_relationships_count > 0 else 0,
            },
            "compression_achieved": {
                "terms": 1 - (extracted_terms_count / source_terms_count) if source_terms_count > 0 else 0,
                "relationships": 1 - (extracted_relationships_count / source_relationships_count) if source_relationships_count > 0 else 0,
            }
        }

    def get_extraction_statistics(self) -> Dict[str, Any]:
        """Get overall extraction statistics.

        Returns:
            Dict[str, Any]: Overall extraction statistics
        """
        return self.extraction_statistics.copy()

    def reset_statistics(self) -> None:
        """Reset extraction statistics."""
        self.extraction_statistics = {
            "total_extractions": 0,
            "successful_extractions": 0,
            "failed_extractions": 0,
            "terms_extracted": 0,
            "relationships_extracted": 0,
        }
        self.logger.info("Reset extraction statistics")