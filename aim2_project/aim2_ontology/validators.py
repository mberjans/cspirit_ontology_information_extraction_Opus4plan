"""
Validation Pipeline Module

This module provides comprehensive validation functionality for ontologies.
It includes various validators for checking ontology consistency, structure,
and data integrity.

Classes:
    ValidationPipeline: Pipeline for validating ontologies
    OntologyValidator: Base validator class
    StructuralValidator: Validates ontology structure
    ConsistencyValidator: Validates ontology consistency
    DataIntegrityValidator: Validates data integrity
"""

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List


@dataclass
class ValidationResult:
    """Result of a validation operation."""

    is_valid: bool
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    validator_name: str = ""
    details: Dict[str, Any] = field(default_factory=dict)


class OntologyValidator(ABC):
    """Abstract base class for ontology validators."""

    def __init__(self, name: str):
        """Initialize validator.

        Args:
            name: Name of the validator
        """
        self.name = name
        self.logger = logging.getLogger(__name__)

    @abstractmethod
    def validate(self, ontology: Any) -> ValidationResult:
        """Validate an ontology.

        Args:
            ontology: The ontology to validate

        Returns:
            ValidationResult: Result of validation
        """


class StructuralValidator(OntologyValidator):
    """Validator for checking ontology structural integrity."""

    def __init__(self):
        """Initialize structural validator."""
        super().__init__("StructuralValidator")

    def validate(self, ontology: Any) -> ValidationResult:
        """Validate ontology structure.

        Args:
            ontology: The ontology to validate

        Returns:
            ValidationResult: Result of structural validation
        """
        errors = []
        warnings = []
        details = {}

        try:
            # Check if ontology has required attributes
            if not hasattr(ontology, "id"):
                errors.append("Ontology missing required 'id' attribute")
            elif not ontology.id:
                errors.append("Ontology 'id' cannot be empty")

            if not hasattr(ontology, "name"):
                warnings.append("Ontology missing 'name' attribute")
            elif not ontology.name:
                warnings.append("Ontology 'name' is empty")

            if not hasattr(ontology, "terms"):
                errors.append("Ontology missing 'terms' attribute")
            else:
                terms_count = len(ontology.terms) if ontology.terms else 0
                details["terms_count"] = terms_count
                if terms_count == 0:
                    warnings.append("Ontology has no terms")

            if not hasattr(ontology, "relationships"):
                errors.append("Ontology missing 'relationships' attribute")
            else:
                rel_count = len(ontology.relationships) if ontology.relationships else 0
                details["relationships_count"] = rel_count
                if rel_count == 0:
                    warnings.append("Ontology has no relationships")

            # Check version information
            if hasattr(ontology, "version"):
                if not ontology.version:
                    warnings.append("Ontology version is empty")
                else:
                    details["version"] = ontology.version
            else:
                warnings.append("Ontology missing version information")

            # Check namespaces
            if hasattr(ontology, "namespaces"):
                if not ontology.namespaces:
                    warnings.append("Ontology has no namespaces defined")
                else:
                    details["namespaces_count"] = len(ontology.namespaces)

        except Exception as e:
            errors.append(f"Error during structural validation: {str(e)}")

        is_valid = len(errors) == 0

        return ValidationResult(
            is_valid=is_valid,
            errors=errors,
            warnings=warnings,
            validator_name=self.name,
            details=details,
        )


class ConsistencyValidator(OntologyValidator):
    """Validator for checking ontology consistency."""

    def __init__(self):
        """Initialize consistency validator."""
        super().__init__("ConsistencyValidator")

    def validate(self, ontology: Any) -> ValidationResult:
        """Validate ontology consistency.

        Args:
            ontology: The ontology to validate

        Returns:
            ValidationResult: Result of consistency validation
        """
        errors = []
        warnings = []
        details = {}

        try:
            # Check for duplicate term IDs
            if hasattr(ontology, "terms") and ontology.terms:
                term_ids = list(ontology.terms.keys())
                unique_term_ids = set(term_ids)
                if len(term_ids) != len(unique_term_ids):
                    duplicates = self._find_duplicates(term_ids)
                    errors.append(f"Duplicate term IDs found: {duplicates}")
                    details["duplicate_term_ids"] = duplicates

            # Check for duplicate relationship IDs
            if hasattr(ontology, "relationships") and ontology.relationships:
                rel_ids = list(ontology.relationships.keys())
                unique_rel_ids = set(rel_ids)
                if len(rel_ids) != len(unique_rel_ids):
                    duplicates = self._find_duplicates(rel_ids)
                    errors.append(f"Duplicate relationship IDs found: {duplicates}")
                    details["duplicate_relationship_ids"] = duplicates

            # Check relationship references
            if hasattr(ontology, "terms") and hasattr(ontology, "relationships"):
                if ontology.terms and ontology.relationships:
                    orphaned_refs = self._check_relationship_references(ontology)
                    if orphaned_refs:
                        warnings.extend(
                            [
                                f"Relationship references non-existent term: {ref}"
                                for ref in orphaned_refs
                            ]
                        )
                        details["orphaned_references"] = orphaned_refs

            # Check for circular dependencies
            if hasattr(ontology, "relationships") and ontology.relationships:
                circular_deps = self._check_circular_dependencies(ontology)
                if circular_deps:
                    warnings.append(
                        f"Potential circular dependencies detected: {len(circular_deps)} cycles"
                    )
                    details["circular_dependencies"] = circular_deps

            # Check consistency flags
            if hasattr(ontology, "is_consistent"):
                if not ontology.is_consistent:
                    warnings.append("Ontology marked as inconsistent")
                details["marked_consistent"] = ontology.is_consistent

        except Exception as e:
            errors.append(f"Error during consistency validation: {str(e)}")

        is_valid = len(errors) == 0

        return ValidationResult(
            is_valid=is_valid,
            errors=errors,
            warnings=warnings,
            validator_name=self.name,
            details=details,
        )

    def _find_duplicates(self, items: List[str]) -> List[str]:
        """Find duplicate items in a list.

        Args:
            items: List of items to check

        Returns:
            List[str]: List of duplicate items
        """
        seen = set()
        duplicates = set()
        for item in items:
            if item in seen:
                duplicates.add(item)
            else:
                seen.add(item)
        return list(duplicates)

    def _check_relationship_references(self, ontology: Any) -> List[str]:
        """Check if relationships reference existing terms.

        Args:
            ontology: The ontology to check

        Returns:
            List[str]: List of orphaned references
        """
        orphaned = []
        term_ids = set(ontology.terms.keys())

        for rel in ontology.relationships.values():
            if hasattr(rel, "subject") and rel.subject not in term_ids:
                orphaned.append(f"{rel.id}.subject -> {rel.subject}")
            if hasattr(rel, "object") and rel.object not in term_ids:
                orphaned.append(f"{rel.id}.object -> {rel.object}")

        return orphaned

    def _check_circular_dependencies(self, ontology: Any) -> List[List[str]]:
        """Check for circular dependencies in relationships.

        Args:
            ontology: The ontology to check

        Returns:
            List[List[str]]: List of circular dependency chains
        """
        # Simple cycle detection using DFS
        graph = {}

        # Build adjacency list
        for rel in ontology.relationships.values():
            if hasattr(rel, "subject") and hasattr(rel, "object"):
                if rel.subject not in graph:
                    graph[rel.subject] = []
                graph[rel.subject].append(rel.object)

        cycles = []
        visited = set()
        rec_stack = set()

        def dfs(node: str, path: List[str]) -> None:
            if node in rec_stack:
                # Found a cycle
                cycle_start = path.index(node)
                cycles.append(path[cycle_start:] + [node])
                return

            if node in visited:
                return

            visited.add(node)
            rec_stack.add(node)

            if node in graph:
                for neighbor in graph[node]:
                    dfs(neighbor, path + [node])

            rec_stack.remove(node)

        for node in graph:
            if node not in visited:
                dfs(node, [])

        return cycles


class DataIntegrityValidator(OntologyValidator):
    """Validator for checking data integrity."""

    def __init__(self):
        """Initialize data integrity validator."""
        super().__init__("DataIntegrityValidator")

    def validate(self, ontology: Any) -> ValidationResult:
        """Validate data integrity.

        Args:
            ontology: The ontology to validate

        Returns:
            ValidationResult: Result of data integrity validation
        """
        errors = []
        warnings = []
        details = {}

        try:
            # Check term data integrity
            if hasattr(ontology, "terms") and ontology.terms:
                term_issues = self._validate_terms(ontology.terms)
                errors.extend(term_issues["errors"])
                warnings.extend(term_issues["warnings"])
                details.update(term_issues["details"])

            # Check relationship data integrity
            if hasattr(ontology, "relationships") and ontology.relationships:
                rel_issues = self._validate_relationships(ontology.relationships)
                errors.extend(rel_issues["errors"])
                warnings.extend(rel_issues["warnings"])
                details.update(rel_issues["details"])

            # Check validation errors attribute
            if hasattr(ontology, "validation_errors"):
                if ontology.validation_errors:
                    warnings.append(
                        f"Ontology has {len(ontology.validation_errors)} validation errors recorded"
                    )
                    details["recorded_validation_errors"] = ontology.validation_errors

        except Exception as e:
            errors.append(f"Error during data integrity validation: {str(e)}")

        is_valid = len(errors) == 0

        return ValidationResult(
            is_valid=is_valid,
            errors=errors,
            warnings=warnings,
            validator_name=self.name,
            details=details,
        )

    def _validate_terms(self, terms: Dict[str, Any]) -> Dict[str, Any]:
        """Validate term data.

        Args:
            terms: Dictionary of terms to validate

        Returns:
            Dict[str, Any]: Validation results
        """
        errors = []
        warnings = []
        details = {}

        empty_names = 0
        empty_definitions = 0
        obsolete_terms = 0

        for term_id, term in terms.items():
            # Check term ID consistency
            if hasattr(term, "id") and term.id != term_id:
                errors.append(f"Term ID mismatch: key='{term_id}', term.id='{term.id}'")

            # Check for empty names
            if not hasattr(term, "name") or not term.name:
                empty_names += 1
                if empty_names <= 5:  # Limit warning spam
                    warnings.append(f"Term '{term_id}' has empty name")

            # Check for empty definitions
            if not hasattr(term, "definition") or not term.definition:
                empty_definitions += 1

            # Count obsolete terms
            if hasattr(term, "is_obsolete") and term.is_obsolete:
                obsolete_terms += 1

        details["empty_names_count"] = empty_names
        details["empty_definitions_count"] = empty_definitions
        details["obsolete_terms_count"] = obsolete_terms

        if empty_definitions > 0:
            warnings.append(f"{empty_definitions} terms have empty definitions")

        return {"errors": errors, "warnings": warnings, "details": details}

    def _validate_relationships(self, relationships: Dict[str, Any]) -> Dict[str, Any]:
        """Validate relationship data.

        Args:
            relationships: Dictionary of relationships to validate

        Returns:
            Dict[str, Any]: Validation results
        """
        errors = []
        warnings = []
        details = {}

        missing_subjects = 0
        missing_objects = 0
        missing_predicates = 0
        low_confidence = 0

        for rel_id, rel in relationships.items():
            # Check relationship ID consistency
            if hasattr(rel, "id") and rel.id != rel_id:
                errors.append(
                    f"Relationship ID mismatch: key='{rel_id}', rel.id='{rel.id}'"
                )

            # Check for missing components
            if not hasattr(rel, "subject") or not rel.subject:
                missing_subjects += 1

            if not hasattr(rel, "object") or not rel.object:
                missing_objects += 1

            if not hasattr(rel, "predicate") or not rel.predicate:
                missing_predicates += 1

            # Check confidence scores
            if hasattr(rel, "confidence"):
                if rel.confidence is not None and rel.confidence < 0.5:
                    low_confidence += 1

        details["missing_subjects_count"] = missing_subjects
        details["missing_objects_count"] = missing_objects
        details["missing_predicates_count"] = missing_predicates
        details["low_confidence_count"] = low_confidence

        if missing_subjects > 0:
            errors.append(f"{missing_subjects} relationships have missing subjects")
        if missing_objects > 0:
            errors.append(f"{missing_objects} relationships have missing objects")
        if missing_predicates > 0:
            errors.append(f"{missing_predicates} relationships have missing predicates")
        if low_confidence > 0:
            warnings.append(
                f"{low_confidence} relationships have low confidence scores (<0.5)"
            )

        return {"errors": errors, "warnings": warnings, "details": details}


class ValidationPipeline:
    """Pipeline for validating ontologies with multiple validators."""

    def __init__(self):
        """Initialize validation pipeline."""
        self.validators: List[OntologyValidator] = []
        self.logger = logging.getLogger(__name__)

        # Add default validators
        self.add_validator(StructuralValidator())
        self.add_validator(ConsistencyValidator())
        self.add_validator(DataIntegrityValidator())

    def validate_ontology(self, ontology: Any) -> Dict[str, Any]:
        """Validate an ontology using all registered validators.

        Args:
            ontology: The ontology to validate

        Returns:
            Dict[str, Any]: Comprehensive validation results
        """
        if not ontology:
            return {
                "is_valid": False,
                "errors": ["Ontology is None or empty"],
                "warnings": [],
                "validator_results": {},
                "summary": {
                    "total_validators": len(self.validators),
                    "passed_validators": 0,
                    "failed_validators": 0,
                    "total_errors": 1,
                    "total_warnings": 0,
                },
            }

        all_errors = []
        all_warnings = []
        validator_results = {}
        passed_validators = 0
        failed_validators = 0

        for validator in self.validators:
            try:
                result = validator.validate(ontology)
                validator_results[validator.name] = {
                    "is_valid": result.is_valid,
                    "errors": result.errors,
                    "warnings": result.warnings,
                    "details": result.details,
                }

                all_errors.extend(result.errors)
                all_warnings.extend(result.warnings)

                if result.is_valid:
                    passed_validators += 1
                else:
                    failed_validators += 1

            except Exception as e:
                error_msg = f"Validator '{validator.name}' failed with error: {str(e)}"
                all_errors.append(error_msg)
                failed_validators += 1
                validator_results[validator.name] = {
                    "is_valid": False,
                    "errors": [error_msg],
                    "warnings": [],
                    "details": {},
                }
                self.logger.exception(f"Validator {validator.name} failed")

        is_valid = len(all_errors) == 0

        return {
            "is_valid": is_valid,
            "errors": all_errors,
            "warnings": all_warnings,
            "validator_results": validator_results,
            "summary": {
                "total_validators": len(self.validators),
                "passed_validators": passed_validators,
                "failed_validators": failed_validators,
                "total_errors": len(all_errors),
                "total_warnings": len(all_warnings),
            },
        }

    def add_validator(self, validator: OntologyValidator) -> None:
        """Add a validator to the pipeline.

        Args:
            validator: The validator to add
        """
        if not isinstance(validator, OntologyValidator):
            raise ValueError("Validator must be an instance of OntologyValidator")

        self.validators.append(validator)
        self.logger.debug(f"Added validator: {validator.name}")

    def remove_validator(self, validator_name: str) -> bool:
        """Remove a validator from the pipeline.

        Args:
            validator_name: Name of the validator to remove

        Returns:
            bool: True if validator was removed, False if not found
        """
        for i, validator in enumerate(self.validators):
            if validator.name == validator_name:
                del self.validators[i]
                self.logger.debug(f"Removed validator: {validator_name}")
                return True
        return False

    def get_validation_errors(self) -> List[str]:
        """Get validation errors from the last validation run.

        Note: This method is kept for backward compatibility.
        Use validate_ontology() for comprehensive results.

        Returns:
            List[str]: List of validation errors
        """
        # This is a placeholder for backward compatibility
        # In practice, errors should be retrieved from validate_ontology() results
        return []

    def list_validators(self) -> List[str]:
        """Get list of registered validator names.

        Returns:
            List[str]: List of validator names
        """
        return [validator.name for validator in self.validators]
