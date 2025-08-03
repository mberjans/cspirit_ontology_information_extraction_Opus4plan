"""
AIM2 Ontology Models Module

This module provides data models for representing and manipulating ontological terms
and relationships within the AIM2 project. The models are designed to support
comprehensive ontology management including term validation, serialization,
and relationship handling.

The primary model is the Term dataclass which represents individual ontological
terms with their associated metadata, relationships, and validation capabilities.

Classes:
    Term: Core dataclass for representing ontological terms

The Term class provides:
    - Complete term metadata management (id, name, definition, synonyms, etc.)
    - Comprehensive validation for term format and content
    - Serialization support for JSON and dictionary formats
    - Standard Python object methods (equality, hashing, string representation)
    - Ontology-specific functionality (namespace handling, relationship management)

Dependencies:
    - dataclasses: For dataclass functionality
    - typing: For type hints
    - json: For JSON serialization
    - re: For pattern matching validation

Usage:
    # Create a basic term
    term = Term(id="CHEBI:12345", name="glucose")

    # Create a comprehensive term
    term = Term(
        id="CHEBI:12345",
        name="glucose",
        definition="A monosaccharide that is aldehydo-D-glucose",
        synonyms=["dextrose", "D-glucose", "grape sugar"],
        namespace="chemical",
        xrefs=["CAS:50-99-7", "KEGG:C00031"],
        relationships={"is_a": ["CHEBI:33917"]}
    )

    # Validate term
    if term.is_valid():
        print("Term is valid")

    # Serialize term
    term_dict = term.to_dict()
    term_json = term.to_json()

    # Deserialize term
    restored_term = Term.from_dict(term_dict)
    restored_from_json = Term.from_json(term_json)

Authors:
    AIM2 Development Team

Version:
    1.0.0

Created:
    2025-08-03
"""

from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional
import json
import re


@dataclass
class Term:
    """
    Dataclass representing an ontological term with comprehensive metadata and validation.

    This dataclass serves as the core model for representing ontological terms within
    the AIM2 project. It provides complete term management functionality including
    validation, serialization, and relationship handling.

    The Term class is designed to be flexible and extensible while maintaining strict
    validation for term integrity. It supports various ontology formats and standards
    commonly used in bioinformatics and knowledge representation.

    Attributes:
        id (str): Unique identifier for the term, typically in format "PREFIX:NUMBER"
        name (str): Human-readable name for the term
        definition (Optional[str]): Detailed definition or description of the term
        synonyms (List[str]): Alternative names or synonyms for the term
        namespace (Optional[str]): Ontology namespace or category
        is_obsolete (bool): Flag indicating if the term is obsolete
        replaced_by (Optional[str]): ID of the term that replaces this obsolete term
        alt_ids (List[str]): Alternative identifiers for the term
        xrefs (List[str]): Cross-references to external databases
        parents (List[str]): Direct parent term IDs in the ontology hierarchy
        children (List[str]): Direct child term IDs in the ontology hierarchy
        relationships (Dict[str, List[str]]): Relationship types and their targets
        metadata (Dict[str, Any]): Additional metadata for the term

    Validation Features:
        - Term ID format validation (PREFIX:NUMBER pattern)
        - Name validation (non-empty string)
        - Synonym list validation
        - Relationship structure validation
        - Namespace validation for common ontology types
        - Comprehensive term validation combining all checks

    Serialization Features:
        - Dictionary serialization (to_dict/from_dict)
        - JSON serialization (to_json/from_json)
        - Round-trip serialization preservation
        - Error handling for invalid serialization data

    Standard Methods:
        - String representation (__str__, __repr__)
        - Equality comparison (__eq__)
        - Hashing support (__hash__) for use in sets and as dict keys
        - Collection support (usable in sets, as dictionary keys)

    Examples:
        Basic term creation:
            >>> term = Term(id="GO:0008150", name="biological_process")
            >>> print(term)
            biological_process (GO:0008150)

        Full term with metadata:
            >>> term = Term(
            ...     id="CHEBI:15422",
            ...     name="ATP",
            ...     definition="Adenosine 5'-triphosphate",
            ...     synonyms=["adenosine triphosphate", "adenosine 5'-triphosphate"],
            ...     namespace="chemical",
            ...     xrefs=["CAS:56-65-5", "KEGG:C00002"],
            ...     relationships={"is_a": ["CHEBI:33019"]}
            ... )

        Validation:
            >>> if term.is_valid():
            ...     print("Term passes all validation checks")

        Serialization:
            >>> term_data = term.to_dict()
            >>> json_string = term.to_json()
            >>> restored = Term.from_dict(term_data)
            >>> assert term == restored
    """

    # Required attributes
    id: str
    name: str

    # Optional attributes with defaults
    definition: Optional[str] = None
    synonyms: List[str] = field(default_factory=list)
    namespace: Optional[str] = None
    is_obsolete: bool = False
    replaced_by: Optional[str] = None
    alt_ids: List[str] = field(default_factory=list)
    xrefs: List[str] = field(default_factory=list)
    parents: List[str] = field(default_factory=list)
    children: List[str] = field(default_factory=list)
    relationships: Dict[str, List[str]] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        """
        Post-initialization validation and setup.

        Validates the term ID and name immediately after creation to ensure
        basic integrity requirements are met.

        Raises:
            ValueError: If term ID format is invalid or name is empty
        """
        # Validate required fields
        if not self.validate_id_format(self.id):
            raise ValueError("Invalid term ID format")

        if not self.validate_name(self.name):
            raise ValueError("Term name cannot be empty")

    def validate_id_format(self, term_id: str) -> bool:
        """
        Validate the format of a term ID.

        Checks if the term ID follows the standard ontology ID format of
        "PREFIX:NUMBER" where PREFIX is alphabetic and NUMBER is numeric.

        Args:
            term_id (str): The term ID to validate

        Returns:
            bool: True if the ID format is valid, False otherwise

        Examples:
            >>> term = Term(id="GO:0008150", name="test")
            >>> term.validate_id_format("GO:0008150")
            True
            >>> term.validate_id_format("invalid_id")
            False
            >>> term.validate_id_format("GO:")
            False
        """
        if not term_id or not isinstance(term_id, str):
            return False

        # Pattern: PREFIX:NUMBER where PREFIX contains letters and NUMBER contains digits
        pattern = r"^[A-Za-z]+:\d+$"
        return bool(re.match(pattern, term_id))

    def validate_name(self, name: str) -> bool:
        """
        Validate a term name.

        Checks if the name is a non-empty string with meaningful content.

        Args:
            name (str): The name to validate

        Returns:
            bool: True if the name is valid, False otherwise

        Examples:
            >>> term = Term(id="GO:0008150", name="biological_process")
            >>> term.validate_name("biological_process")
            True
            >>> term.validate_name("")
            False
            >>> term.validate_name("   ")
            False
        """
        return bool(name and isinstance(name, str) and name.strip())

    def validate_synonyms(self, synonyms: List[str]) -> bool:
        """
        Validate a list of synonyms.

        Ensures the synonyms list is properly formatted and contains valid strings.

        Args:
            synonyms (List[str]): List of synonyms to validate

        Returns:
            bool: True if the synonyms list is valid, False otherwise

        Examples:
            >>> term = Term(id="GO:0008150", name="test")
            >>> term.validate_synonyms(["synonym1", "synonym2"])
            True
            >>> term.validate_synonyms([])
            True
            >>> term.validate_synonyms(["valid", ""])  # Contains empty string
            True  # Still valid as empty synonyms are filtered
        """
        if not isinstance(synonyms, list):
            return False

        # All items should be strings (empty strings are allowed)
        return all(isinstance(synonym, str) for synonym in synonyms)

    def validate_relationships(self, relationships: Dict[str, List[str]]) -> bool:
        """
        Validate the relationships dictionary structure.

        Ensures relationships follow the expected format where keys are relationship
        types and values are lists of target term IDs.

        Args:
            relationships (Dict[str, List[str]]): Relationships to validate

        Returns:
            bool: True if relationships structure is valid, False otherwise

        Examples:
            >>> term = Term(id="GO:0008150", name="test")
            >>> term.validate_relationships({"is_a": ["GO:0003674"]})
            True
            >>> term.validate_relationships({})
            True
            >>> term.validate_relationships({"is_a": "GO:0003674"})  # Wrong type
            False
        """
        if not isinstance(relationships, dict):
            return False

        for rel_type, targets in relationships.items():
            if not isinstance(rel_type, str) or not isinstance(targets, list):
                return False
            if not all(isinstance(target, str) for target in targets):
                return False

        return True

    def validate_namespace(self, namespace: str) -> bool:
        """
        Validate an ontology namespace.

        Checks if the namespace is one of the commonly recognized ontology
        namespaces or follows valid namespace conventions.

        Args:
            namespace (str): The namespace to validate

        Returns:
            bool: True if namespace is valid, False otherwise

        Examples:
            >>> term = Term(id="GO:0008150", name="test", namespace="biological_process")
            >>> term.validate_namespace("biological_process")
            True
            >>> term.validate_namespace("chemical")
            True
            >>> term.validate_namespace("invalid@namespace")
            False
        """
        if not namespace or not isinstance(namespace, str):
            return False

        # Common valid namespaces
        valid_namespaces = {
            "biological_process",
            "molecular_function",
            "cellular_component",
            "chemical",
            "anatomy",
            "phenotype",
            "disease",
            "pathway",
            "protein",
            "gene",
            "organism",
            "tissue",
            "development",
        }

        # Allow known namespaces or any string that looks like a valid identifier
        if namespace in valid_namespaces:
            return True

        # Allow other reasonable namespace formats (letters, numbers, underscores, hyphens)
        pattern = r"^[a-zA-Z][a-zA-Z0-9_-]*$"
        return bool(re.match(pattern, namespace))

    def is_valid(self) -> bool:
        """
        Perform comprehensive validation of the term.

        Checks all aspects of the term for validity including ID format,
        name, synonyms, relationships, and other attributes.

        Returns:
            bool: True if the term passes all validation checks, False otherwise

        Examples:
            >>> term = Term(
            ...     id="GO:0008150",
            ...     name="biological_process",
            ...     synonyms=["process"]
            ... )
            >>> term.is_valid()
            True

            >>> invalid_term = Term(id="invalid", name="")
            Traceback (most recent call last):
            ...
            ValueError: Invalid term ID format
        """
        try:
            # Basic required field validation
            if not self.validate_id_format(self.id):
                return False

            if not self.validate_name(self.name):
                return False

            # Optional field validation
            if not self.validate_synonyms(self.synonyms):
                return False

            if not self.validate_relationships(self.relationships):
                return False

            # Namespace validation (if provided)
            if self.namespace is not None and not self.validate_namespace(
                self.namespace
            ):
                return False

            # Validate list fields are actually lists
            list_fields = [
                self.synonyms,
                self.alt_ids,
                self.xrefs,
                self.parents,
                self.children,
            ]
            if not all(isinstance(field, list) for field in list_fields):
                return False

            # Validate string fields in lists
            for field in list_fields:
                if not all(isinstance(item, str) for item in field):
                    return False

            # Validate metadata is a dictionary
            if not isinstance(self.metadata, dict):
                return False

            # Validate boolean fields
            if not isinstance(self.is_obsolete, bool):
                return False

            # Validate replaced_by if provided
            if self.replaced_by is not None:
                if (
                    not isinstance(self.replaced_by, str)
                    or not self.replaced_by.strip()
                ):
                    return False

            return True

        except Exception:
            return False

    def to_dict(self) -> Dict[str, Any]:
        """
        Serialize the term to a dictionary.

        Creates a comprehensive dictionary representation of the term that
        preserves all attributes and can be used for storage or transmission.

        Returns:
            Dict[str, Any]: Dictionary representation of the term

        Examples:
            >>> term = Term(id="GO:0008150", name="biological_process")
            >>> term_dict = term.to_dict()
            >>> term_dict["id"]
            'GO:0008150'
            >>> term_dict["name"]
            'biological_process'
        """
        return {
            "id": self.id,
            "name": self.name,
            "definition": self.definition,
            "synonyms": self.synonyms.copy(),
            "namespace": self.namespace,
            "is_obsolete": self.is_obsolete,
            "replaced_by": self.replaced_by,
            "alt_ids": self.alt_ids.copy(),
            "xrefs": self.xrefs.copy(),
            "parents": self.parents.copy(),
            "children": self.children.copy(),
            "relationships": {k: v.copy() for k, v in self.relationships.items()},
            "metadata": self.metadata.copy(),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Term":
        """
        Deserialize a term from a dictionary.

        Creates a Term instance from a dictionary representation, typically
        created by the to_dict() method.

        Args:
            data (Dict[str, Any]): Dictionary containing term data

        Returns:
            Term: Term instance created from the dictionary data

        Raises:
            KeyError: If required fields are missing from the dictionary
            ValueError: If the data contains invalid values

        Examples:
            >>> data = {
            ...     "id": "GO:0008150",
            ...     "name": "biological_process",
            ...     "synonyms": ["process"],
            ...     "relationships": {"is_a": ["GO:0003674"]}
            ... }
            >>> term = Term.from_dict(data)
            >>> term.id
            'GO:0008150'
        """
        # Required fields
        term_id = data["id"]
        name = data["name"]

        # Optional fields with defaults
        definition = data.get("definition")
        synonyms = data.get("synonyms", [])
        namespace = data.get("namespace")
        is_obsolete = data.get("is_obsolete", False)
        replaced_by = data.get("replaced_by")
        alt_ids = data.get("alt_ids", [])
        xrefs = data.get("xrefs", [])
        parents = data.get("parents", [])
        children = data.get("children", [])
        relationships = data.get("relationships", {})
        metadata = data.get("metadata", {})

        return cls(
            id=term_id,
            name=name,
            definition=definition,
            synonyms=synonyms,
            namespace=namespace,
            is_obsolete=is_obsolete,
            replaced_by=replaced_by,
            alt_ids=alt_ids,
            xrefs=xrefs,
            parents=parents,
            children=children,
            relationships=relationships,
            metadata=metadata,
        )

    def to_json(self) -> str:
        """
        Serialize the term to a JSON string.

        Creates a JSON representation of the term that can be stored in files
        or transmitted over networks.

        Returns:
            str: JSON string representation of the term

        Raises:
            TypeError: If the term contains non-serializable data

        Examples:
            >>> term = Term(id="GO:0008150", name="biological_process")
            >>> json_str = term.to_json()
            >>> isinstance(json_str, str)
            True
            >>> "GO:0008150" in json_str
            True
        """
        return json.dumps(self.to_dict(), ensure_ascii=False, indent=2)

    @classmethod
    def from_json(cls, json_str: str) -> "Term":
        """
        Deserialize a term from a JSON string.

        Creates a Term instance from a JSON string representation, typically
        created by the to_json() method.

        Args:
            json_str (str): JSON string containing term data

        Returns:
            Term: Term instance created from the JSON data

        Raises:
            json.JSONDecodeError: If the JSON string is invalid
            KeyError: If required fields are missing
            ValueError: If the data contains invalid values

        Examples:
            >>> json_str = '{"id": "GO:0008150", "name": "biological_process"}'
            >>> term = Term.from_json(json_str)
            >>> term.id
            'GO:0008150'
        """
        data = json.loads(json_str)
        return cls.from_dict(data)

    def __str__(self) -> str:
        """
        Return a user-friendly string representation of the term.

        Creates a readable string that includes the term name, ID, and
        optionally the definition or obsolete status.

        Returns:
            str: User-friendly string representation

        Examples:
            >>> term = Term(id="GO:0008150", name="biological_process")
            >>> str(term)
            'biological_process (GO:0008150)'

            >>> term_with_def = Term(
            ...     id="GO:0008150",
            ...     name="biological_process",
            ...     definition="Any process specifically pertinent to the functioning of integrated living units"
            ... )
            >>> str(term_with_def)
            'biological_process (GO:0008150): Any process specifically pertinent to the functioning of integrated living units'
        """
        if self.is_obsolete:
            base = f"[OBSOLETE] {self.name} ({self.id})"
        else:
            base = f"{self.name} ({self.id})"

        if self.definition:
            return f"{base}: {self.definition}"

        return base

    def __repr__(self) -> str:
        """
        Return a detailed string representation for debugging.

        Creates a comprehensive string that shows the term's class and
        key attributes, suitable for debugging and development.

        Returns:
            str: Detailed string representation

        Examples:
            >>> term = Term(id="GO:0008150", name="biological_process")
            >>> repr(term)
            "Term(id='GO:0008150', name='biological_process')"

            >>> term_with_extras = Term(
            ...     id="GO:0008150",
            ...     name="biological_process",
            ...     synonyms=["process"],
            ...     is_obsolete=False
            ... )
            >>> "synonyms=['process']" in repr(term_with_extras)
            True
        """
        # Build a list of key attributes to show
        attrs = [f"id='{self.id}'", f"name='{self.name}'"]

        # Add optional attributes that have non-default values
        if self.definition:
            attrs.append(
                f"definition='{self.definition[:50]}{'...' if len(self.definition) > 50 else ''}'"
            )

        if self.synonyms:
            attrs.append(f"synonyms={self.synonyms}")

        if self.namespace:
            attrs.append(f"namespace='{self.namespace}'")

        if self.is_obsolete:
            attrs.append(f"is_obsolete={self.is_obsolete}")

        if self.relationships:
            attrs.append(f"relationships={self.relationships}")

        return f"Term({', '.join(attrs)})"

    def __eq__(self, other) -> bool:
        """
        Compare two terms for equality.

        Two terms are considered equal if they have the same ID and all
        other attributes match exactly.

        Args:
            other: Object to compare with

        Returns:
            bool: True if terms are equal, False otherwise

        Examples:
            >>> term1 = Term(id="GO:0008150", name="biological_process")
            >>> term2 = Term(id="GO:0008150", name="biological_process")
            >>> term1 == term2
            True

            >>> term3 = Term(id="GO:0008151", name="biological_process")
            >>> term1 == term3
            False
        """
        if not isinstance(other, Term):
            return False

        return (
            self.id == other.id
            and self.name == other.name
            and self.definition == other.definition
            and self.synonyms == other.synonyms
            and self.namespace == other.namespace
            and self.is_obsolete == other.is_obsolete
            and self.replaced_by == other.replaced_by
            and self.alt_ids == other.alt_ids
            and self.xrefs == other.xrefs
            and self.parents == other.parents
            and self.children == other.children
            and self.relationships == other.relationships
            and self.metadata == other.metadata
        )

    def __hash__(self) -> int:
        """
        Generate a hash code for the term.

        Uses the term ID as the primary hash component since it should be
        unique. This allows terms to be used in sets and as dictionary keys.

        Returns:
            int: Hash code for the term

        Examples:
            >>> term1 = Term(id="GO:0008150", name="biological_process")
            >>> term2 = Term(id="GO:0008150", name="biological_process")
            >>> hash(term1) == hash(term2)
            True

            >>> term_set = {term1, term2}
            >>> len(term_set)  # Should be 1 since they're equal
            1
        """
        # Use ID as primary hash since it should be unique
        # Include name as secondary component for additional distinction
        return hash((self.id, self.name))
