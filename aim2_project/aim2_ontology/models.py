"""
AIM2 Ontology Models Module

This module provides data models for representing and manipulating ontological terms
and relationships within the AIM2 project. The models are designed to support
comprehensive ontology management including term validation, serialization,
and relationship handling.

The module provides three primary models: the Term dataclass for representing individual
ontological terms, the Relationship dataclass for representing relationships between
terms, and the Ontology dataclass for representing complete ontologies with comprehensive
metadata and validation capabilities.

Classes:
    Term: Core dataclass for representing ontological terms
    Relationship: Core dataclass for representing ontological relationships
    Ontology: Core dataclass for representing complete ontologies

The Term class provides:
    - Complete term metadata management (id, name, definition, synonyms, etc.)
    - Comprehensive validation for term format and content
    - Serialization support for JSON and dictionary formats
    - Standard Python object methods (equality, hashing, string representation)
    - Ontology-specific functionality (namespace handling, relationship management)

The Relationship class provides:
    - Complete relationship metadata management (id, subject, predicate, object, etc.)
    - Comprehensive validation for relationship integrity and format
    - Support for confidence scoring, evidence tracking, and provenance
    - Serialization support for JSON and dictionary formats
    - Standard Python object methods (equality, hashing, string representation)
    - Ontology-specific functionality (predicate validation, circular relationship prevention)

The Ontology class provides:
    - Complete ontology container management (terms, relationships, metadata)
    - Comprehensive validation for ontology integrity and consistency
    - Container operations for term and relationship management
    - Statistical analysis and reporting capabilities
    - Serialization support for JSON and dictionary formats
    - Standard Python object methods (equality, hashing, string representation)
    - Ontology-specific functionality (circular dependency detection, orphan term detection)

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

    # Create a basic relationship
    rel = Relationship(
        id="REL:001",
        subject="CHEBI:12345",
        predicate="is_a",
        object="CHEBI:33917"
    )

    # Create a comprehensive relationship
    rel = Relationship(
        id="REL:001",
        subject="CHEBI:15422",
        predicate="regulates",
        object="GO:0008152",
        confidence=0.95,
        source="ChEBI",
        evidence="experimental_evidence",
        context="cellular metabolism",
        sentence="ATP regulates metabolic processes"
    )

    # Validate relationship
    if rel.is_valid():
        print("Relationship is valid")

    # Serialize relationship
    rel_dict = rel.to_dict()
    rel_json = rel.to_json()

    # Deserialize relationship
    restored_rel = Relationship.from_dict(rel_dict)
    restored_from_json = Relationship.from_json(rel_json)

    # Create a basic ontology
    ontology = Ontology(id="ONT:001", name="Chemical Ontology")

    # Create a comprehensive ontology
    ontology = Ontology(
        id="ONT:001",
        name="Chemical Ontology",
        version="2023.1",
        description="A comprehensive chemical ontology",
        namespaces=["chemical", "biological_process"],
        metadata={"source": "ChEBI", "license": "CC BY 4.0"}
    )

    # Add terms and relationships
    term = Term(id="CHEBI:12345", name="glucose")
    ontology.add_term(term)
    rel = Relationship(id="REL:001", subject="CHEBI:12345", predicate="is_a", object="CHEBI:33917")
    ontology.add_relationship(rel)

    # Validate ontology
    if ontology.is_valid():
        print("Ontology is valid")

    # Get statistics
    stats = ontology.get_statistics()
    print(f"Ontology has {stats['term_count']} terms and {stats['relationship_count']} relationships")

    # Serialize ontology
    ontology_dict = ontology.to_dict()
    ontology_json = ontology.to_json()

    # Deserialize ontology
    restored_ontology = Ontology.from_dict(ontology_dict)
    restored_from_json = Ontology.from_json(ontology_json)

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
from datetime import datetime


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


@dataclass
class Relationship:
    """
    Dataclass representing an ontological relationship with comprehensive metadata and validation.

    This dataclass serves as the core model for representing relationships between ontological
    terms within the AIM2 project. It provides complete relationship management functionality
    including validation, serialization, and metadata handling.

    The Relationship class is designed to be flexible and extensible while maintaining strict
    validation for relationship integrity. It supports various relationship types commonly
    used in bioinformatics and knowledge representation, with specific support for chemical
    and plant ontology relationships.

    Attributes:
        id (str): Unique identifier for the relationship, typically in format "PREFIX:NUMBER"
        subject (str): Subject term ID in the relationship triple
        predicate (str): Relationship type/predicate from supported predicate list
        object (str): Object term ID in the relationship triple
        confidence (float): Confidence score for the relationship (0.0-1.0, default 1.0)
        source (Optional[str]): Source of the relationship information
        evidence (Optional[str]): Type of evidence supporting the relationship
        extraction_method (Optional[str]): Method used to extract the relationship
        created_date (Optional[str]): Creation timestamp in ISO format
        provenance (Optional[str]): Data provenance information
        is_validated (bool): Whether the relationship has been validated (default False)
        context (Optional[str]): Contextual information for the relationship
        sentence (Optional[str]): Source sentence containing the relationship
        negated (bool): Whether the relationship is negated (default False)
        modality (str): Certainty level of the relationship (default "certain")
        directional (bool): Whether the relationship is directional (default True)

    Supported Predicates:
        is_a, part_of, has_role, participates_in, located_in, derives_from, regulates,
        catalyzes, accumulates_in, affects, involved_in, upregulates, downregulates,
        made_via, occurs_in

    Validation Features:
        - Relationship ID format validation (PREFIX:NUMBER pattern)
        - Predicate validation against supported predicate list
        - Subject/object format validation
        - Confidence range validation (0.0-1.0)
        - Circular relationship prevention
        - Comprehensive relationship validation combining all checks

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
        Basic relationship creation:
            >>> rel = Relationship(
            ...     id="REL:001",
            ...     subject="CHEBI:12345",
            ...     predicate="is_a",
            ...     object="CHEBI:33917"
            ... )
            >>> print(rel)
            CHEBI:12345 is_a CHEBI:33917

        Full relationship with metadata:
            >>> rel = Relationship(
            ...     id="REL:001",
            ...     subject="CHEBI:15422",
            ...     predicate="regulates",
            ...     object="GO:0008152",
            ...     confidence=0.95,
            ...     source="ChEBI",
            ...     evidence="experimental_evidence",
            ...     context="cellular metabolism",
            ...     sentence="ATP regulates metabolic processes"
            ... )

        Validation:
            >>> if rel.is_valid():
            ...     print("Relationship passes all validation checks")

        Serialization:
            >>> rel_data = rel.to_dict()
            >>> json_string = rel.to_json()
            >>> restored = Relationship.from_dict(rel_data)
            >>> assert rel == restored
    """

    # Required attributes
    id: str
    subject: str
    predicate: str
    object: str

    # Optional attributes with defaults
    confidence: float = 1.0
    source: Optional[str] = None
    evidence: Optional[str] = None
    extraction_method: Optional[str] = None
    created_date: Optional[str] = None
    provenance: Optional[str] = None
    is_validated: bool = False
    context: Optional[str] = None
    sentence: Optional[str] = None
    negated: bool = False
    modality: str = "certain"
    directional: bool = True

    def __post_init__(self):
        """
        Post-initialization validation and setup.

        Validates the relationship ID, predicate, subject/object format, confidence range,
        and checks for circular relationships immediately after creation to ensure
        basic integrity requirements are met.

        Raises:
            ValueError: If relationship ID format is invalid, predicate is unsupported,
                       subject/object format is invalid, confidence is out of range,
                       or circular relationship is detected
        """
        # Validate required fields
        if not self.validate_id_format(self.id):
            raise ValueError("Invalid relationship ID format")

        if not self.validate_predicate(self.predicate):
            raise ValueError("Invalid predicate type")

        if not self.validate_subject_object_format(self.subject, self.object):
            raise ValueError("Subject and object cannot be empty")

        if not self.validate_confidence(self.confidence):
            raise ValueError("Confidence must be between 0.0 and 1.0")

        if not self.validate_circular_relationship(self.subject, self.object):
            raise ValueError("Circular relationship detected")

    def validate_id_format(self, relationship_id: str) -> bool:
        """
        Validate the format of a relationship ID.

        Checks if the relationship ID follows the standard ontology ID format of
        "PREFIX:NUMBER" where PREFIX is alphabetic and NUMBER is numeric.

        Args:
            relationship_id (str): The relationship ID to validate

        Returns:
            bool: True if the ID format is valid, False otherwise

        Examples:
            >>> rel = Relationship(id="REL:001", subject="A:1", predicate="is_a", object="B:1")
            >>> rel.validate_id_format("REL:001")
            True
            >>> rel.validate_id_format("invalid_id")
            False
            >>> rel.validate_id_format("REL:")
            False
        """
        if not relationship_id or not isinstance(relationship_id, str):
            return False

        # Pattern: PREFIX:NUMBER where PREFIX contains letters and NUMBER contains digits
        pattern = r"^[A-Za-z]+:\d+$"
        return bool(re.match(pattern, relationship_id))

    def validate_predicate(self, predicate: str) -> bool:
        """
        Validate a relationship predicate.

        Checks if the predicate is in the list of supported relationship types.

        Args:
            predicate (str): The predicate to validate

        Returns:
            bool: True if the predicate is valid, False otherwise

        Examples:
            >>> rel = Relationship(id="REL:001", subject="A:1", predicate="is_a", object="B:1")
            >>> rel.validate_predicate("is_a")
            True
            >>> rel.validate_predicate("invalid_predicate")
            False
        """
        if not predicate or not isinstance(predicate, str):
            return False

        supported_predicates = {
            "is_a",
            "part_of",
            "has_part",
            "has_role",
            "participates_in",
            "located_in",
            "derives_from",
            "derives_to",
            "regulates",
            "regulated_by",
            "catalyzes",
            "catalyzed_by",
            "accumulates_in",
            "accumulates",
            "affects",
            "involved_in",
            "upregulates",
            "upregulated_by",
            "downregulates",
            "downregulated_by",
            "made_via",
            "occurs_in",
            "contains",
        }

        return predicate in supported_predicates

    def validate_subject_object_format(self, subject: str, object: str) -> bool:
        """
        Validate the format of subject and object IDs.

        Checks if both subject and object are non-empty strings with valid content.

        Args:
            subject (str): The subject ID to validate
            object (str): The object ID to validate

        Returns:
            bool: True if both subject and object are valid, False otherwise

        Examples:
            >>> rel = Relationship(id="REL:001", subject="A:1", predicate="is_a", object="B:1")
            >>> rel.validate_subject_object_format("CHEBI:12345", "CHEBI:33917")
            True
            >>> rel.validate_subject_object_format("", "CHEBI:33917")
            False
            >>> rel.validate_subject_object_format("CHEBI:12345", "")
            False
        """
        if not subject or not isinstance(subject, str) or not subject.strip():
            return False
        if not object or not isinstance(object, str) or not object.strip():
            return False
        return True

    def validate_confidence(self, confidence: float) -> bool:
        """
        Validate a confidence score.

        Checks if the confidence score is within the valid range of 0.0 to 1.0.

        Args:
            confidence (float): The confidence score to validate

        Returns:
            bool: True if the confidence is valid, False otherwise

        Examples:
            >>> rel = Relationship(id="REL:001", subject="A:1", predicate="is_a", object="B:1")
            >>> rel.validate_confidence(0.95)
            True
            >>> rel.validate_confidence(1.5)
            False
            >>> rel.validate_confidence(-0.1)
            False
        """
        if not isinstance(confidence, (int, float)):
            return False
        return 0.0 <= confidence <= 1.0

    def validate_circular_relationship(self, subject: str, object: str) -> bool:
        """
        Validate that the relationship is not circular.

        Checks if the subject and object are different to prevent circular relationships.

        Args:
            subject (str): The subject ID
            object (str): The object ID

        Returns:
            bool: True if the relationship is not circular, False otherwise

        Examples:
            >>> rel = Relationship(id="REL:001", subject="A:1", predicate="is_a", object="B:1")
            >>> rel.validate_circular_relationship("CHEBI:12345", "CHEBI:33917")
            True
            >>> rel.validate_circular_relationship("CHEBI:12345", "CHEBI:12345")
            False
        """
        if not subject or not object:
            return False
        return subject != object

    def validate_predicate_semantics(
        self, predicate: str, subject: str, object: str
    ) -> bool:
        """
        Validate that the predicate makes semantic sense with the given subject and object.

        Performs semantic validation of relationship triples by checking if the predicate
        is appropriate for the types of entities involved (chemical, biological process,
        cellular component, etc.) based on their namespace prefixes.

        Args:
            predicate (str): The relationship predicate to validate
            subject (str): The subject entity ID
            object (str): The object entity ID

        Returns:
            bool: True if the predicate is semantically valid for the subject-object pair, False otherwise

        Examples:
            >>> rel = Relationship(id="REL:001", subject="A:1", predicate="is_a", object="B:1")
            >>> rel.validate_predicate_semantics("is_a", "CHEBI:12345", "CHEBI:33917")
            True
            >>> rel.validate_predicate_semantics("located_in", "CHEBI:12345", "GO:0005623")
            True
            >>> rel.validate_predicate_semantics("catalyzes", "CHEBI:12345", "GO:0008152")
            False  # Chemical entities don't typically catalyze processes directly
        """
        if not predicate or not subject or not object:
            return False

        # Extract namespace prefixes to determine entity types
        subject_prefix = subject.split(":")[0] if ":" in subject else ""
        object_prefix = object.split(":")[0] if ":" in object else ""

        # Define semantic rules for different predicate types
        semantic_rules = {
            "is_a": {
                # is_a relationships should be between entities of similar types
                "valid_combinations": [
                    ("CHEBI", "CHEBI"),  # chemical is_a chemical
                    ("GO", "GO"),  # process/function is_a process/function
                    ("PO", "PO"),  # plant structure is_a plant structure
                    ("NCIT", "NCIT"),  # concept is_a concept
                ]
            },
            "part_of": {
                # part_of relationships for structural hierarchies
                "valid_combinations": [
                    ("GO", "GO"),  # cellular component part_of cellular component
                    ("PO", "PO"),  # plant structure part_of plant structure
                    ("CHEBI", "CHEBI"),  # molecular part part_of molecule
                ]
            },
            "has_part": {
                # has_part relationships for structural hierarchies (inverse of part_of)
                "valid_combinations": [
                    ("GO", "GO"),  # cellular component has_part cellular component
                    ("PO", "PO"),  # plant structure has_part plant structure
                    ("CHEBI", "CHEBI"),  # molecule has_part molecular part
                ]
            },
            "located_in": {
                # located_in relationships for spatial containment
                "valid_combinations": [
                    ("CHEBI", "GO"),  # chemical located_in cellular component
                    ("CHEBI", "PO"),  # chemical located_in plant structure
                    ("GO", "GO"),  # process located_in cellular component
                ]
            },
            "regulates": {
                # regulates relationships for functional regulation
                "valid_combinations": [
                    ("CHEBI", "GO"),  # chemical regulates biological process
                    ("GO", "GO"),  # process regulates process
                ]
            },
            "regulated_by": {
                # regulated_by relationships (inverse of regulates)
                "valid_combinations": [
                    ("GO", "CHEBI"),  # process regulated_by chemical
                    ("GO", "GO"),  # process regulated_by process
                ]
            },
            "catalyzes": {
                # catalyzes relationships - typically enzymes catalyze reactions
                "valid_combinations": [
                    ("GO", "GO"),  # molecular function catalyzes biological process
                ]
            },
            "accumulates_in": {
                # accumulates_in for chemical accumulation
                "valid_combinations": [
                    ("CHEBI", "GO"),  # chemical accumulates_in cellular component
                    ("CHEBI", "PO"),  # chemical accumulates_in plant structure
                ]
            },
            "participates_in": {
                # participates_in for process participation
                "valid_combinations": [
                    ("CHEBI", "GO"),  # chemical participates_in biological process
                    ("GO", "GO"),  # function participates_in process
                ]
            },
            "derives_from": {
                # derives_from for derivation relationships
                "valid_combinations": [
                    ("CHEBI", "CHEBI"),  # chemical derives_from chemical
                    ("PO", "PO"),  # structure derives_from structure
                ]
            },
            "upregulates": {
                # upregulates for positive regulation
                "valid_combinations": [
                    ("CHEBI", "GO"),  # chemical upregulates process
                    ("GO", "GO"),  # process upregulates process
                ]
            },
            "downregulates": {
                # downregulates for negative regulation
                "valid_combinations": [
                    ("CHEBI", "GO"),  # chemical downregulates process
                    ("GO", "GO"),  # process downregulates process
                ]
            },
            "occurs_in": {
                # occurs_in for temporal/spatial occurrence
                "valid_combinations": [
                    ("GO", "GO"),  # process occurs_in cellular component
                    ("GO", "PO"),  # process occurs_in plant structure
                ]
            },
            "made_via": {
                # made_via for production relationships
                "valid_combinations": [
                    ("CHEBI", "GO"),  # chemical made_via biological process
                ]
            },
            "affects": {
                # affects for general influence relationships
                "valid_combinations": [
                    ("CHEBI", "GO"),  # chemical affects process/component
                    ("GO", "GO"),  # process affects process/component
                ]
            },
            "has_role": {
                # has_role for functional roles
                "valid_combinations": [
                    ("CHEBI", "GO"),  # chemical has_role molecular function
                ]
            },
            "involved_in": {
                # involved_in for process involvement
                "valid_combinations": [
                    ("CHEBI", "GO"),  # chemical involved_in biological process
                    ("GO", "GO"),  # function involved_in process
                ]
            },
            "catalyzed_by": {
                # catalyzed_by relationships (inverse of catalyzes)
                "valid_combinations": [
                    ("GO", "GO"),  # biological process catalyzed_by molecular function
                ]
            },
            "upregulated_by": {
                # upregulated_by relationships (inverse of upregulates)
                "valid_combinations": [
                    ("GO", "CHEBI"),  # process upregulated_by chemical
                    ("GO", "GO"),  # process upregulated_by process
                ]
            },
            "downregulated_by": {
                # downregulated_by relationships (inverse of downregulates)
                "valid_combinations": [
                    ("GO", "CHEBI"),  # process downregulated_by chemical
                    ("GO", "GO"),  # process downregulated_by process
                ]
            },
            "derives_to": {
                # derives_to relationships (inverse of derives_from)
                "valid_combinations": [
                    ("CHEBI", "CHEBI"),  # chemical derives_to chemical
                    ("PO", "PO"),  # structure derives_to structure
                ]
            },
            "contains": {
                # contains relationships (inverse of located_in)
                "valid_combinations": [
                    ("GO", "CHEBI"),  # cellular component contains chemical
                    ("PO", "CHEBI"),  # plant structure contains chemical
                    ("GO", "GO"),  # cellular component contains process
                ]
            },
            "accumulates": {
                # accumulates relationships (inverse of accumulates_in)
                "valid_combinations": [
                    ("GO", "CHEBI"),  # cellular component accumulates chemical
                    ("PO", "CHEBI"),  # plant structure accumulates chemical
                ]
            },
        }

        # Check if predicate has defined semantic rules
        if predicate not in semantic_rules:
            # For undefined predicates, allow if both entities have valid prefixes
            return bool(subject_prefix and object_prefix)

        # Check if the subject-object combination is semantically valid
        valid_combinations = semantic_rules[predicate]["valid_combinations"]
        return (subject_prefix, object_prefix) in valid_combinations

    def validate_domain_constraints(
        self, subject: str, predicate: str, object: str
    ) -> bool:
        """
        Validate domain-specific constraints for chemical and plant ontology relationships.

        Enforces domain-specific rules and constraints that apply to relationships within
        chemical, biological, and plant ontology domains. This includes checking for
        appropriate relationship types, entity combinations, and domain-specific semantics.

        Args:
            subject (str): The subject entity ID
            predicate (str): The relationship predicate
            object (str): The object entity ID

        Returns:
            bool: True if domain constraints are satisfied, False otherwise

        Examples:
            >>> rel = Relationship(id="REL:001", subject="A:1", predicate="is_a", object="B:1")
            >>> rel.validate_domain_constraints("CHEBI:12345", "is_a", "CHEBI:33917")
            True
            >>> rel.validate_domain_constraints("CHEBI:15422", "regulates", "GO:0008152")
            True
            >>> rel.validate_domain_constraints("PO:0000030", "catalyzes", "GO:0008152")
            False  # Plant structures don't catalyze processes
        """
        if not subject or not predicate or not object:
            return False

        # Extract namespace prefixes
        subject_prefix = subject.split(":")[0] if ":" in subject else ""
        object_prefix = object.split(":")[0] if ":" in object else ""

        # Chemical domain constraints
        if subject_prefix == "CHEBI":
            chemical_constraints = {
                "is_a": ["CHEBI"],  # Chemicals can only be subtypes of other chemicals
                "derives_from": ["CHEBI"],  # Chemical derivation from other chemicals
                "regulates": ["GO"],  # Chemicals can regulate biological processes
                "upregulates": ["GO"],  # Chemicals can upregulate processes
                "downregulates": ["GO"],  # Chemicals can downregulate processes
                "participates_in": ["GO"],  # Chemicals participate in processes
                "located_in": [
                    "GO",
                    "PO",
                ],  # Chemicals located in components/structures
                "accumulates_in": ["GO", "PO"],  # Chemicals accumulate in locations
                "affects": ["GO"],  # Chemicals affect processes/components
                "has_role": ["GO"],  # Chemicals have molecular functions
                "involved_in": ["GO"],  # Chemicals involved in processes
                "made_via": ["GO"],  # Chemicals made via processes
            }

            if predicate in chemical_constraints:
                return object_prefix in chemical_constraints[predicate]

        # Plant ontology domain constraints
        elif subject_prefix == "PO":
            plant_constraints = {
                "is_a": ["PO"],  # Plant structures subtype of other structures
                "part_of": ["PO"],  # Plant structures part_of other structures
                "has_part": ["PO"],  # Plant structures have parts
                "derives_from": ["PO"],  # Structure derivation
                "derives_to": ["PO"],  # Structure derivation (inverse)
                "develops_from": ["PO"],  # Developmental relationships
                "contains": ["CHEBI"],  # Plant structures contain chemicals
                "accumulates": ["CHEBI"],  # Plant structures accumulate chemicals
            }

            if predicate in plant_constraints:
                return object_prefix in plant_constraints[predicate]

        # Gene Ontology domain constraints
        elif subject_prefix == "GO":
            go_constraints = {
                "is_a": ["GO"],  # GO terms subtype of other GO terms
                "part_of": ["GO"],  # Cellular components part of other components
                "has_part": ["GO"],  # Components have parts
                "regulates": ["GO"],  # Processes regulate other processes
                "regulated_by": [
                    "GO",
                    "CHEBI",
                ],  # Processes regulated by other processes/chemicals
                "positively_regulates": ["GO"],  # Positive regulation
                "negatively_regulates": ["GO"],  # Negative regulation
                "upregulated_by": [
                    "GO",
                    "CHEBI",
                ],  # Processes upregulated by other processes/chemicals
                "downregulated_by": [
                    "GO",
                    "CHEBI",
                ],  # Processes downregulated by other processes/chemicals
                "catalyzed_by": ["GO"],  # Processes catalyzed by molecular functions
                "occurs_in": ["GO", "PO"],  # Processes occur in components/structures
                "contains": ["CHEBI", "GO"],  # Components contain chemicals/processes
            }

            if predicate in go_constraints:
                return object_prefix in go_constraints[predicate]

            # Special case: molecular functions can catalyze biological processes
            if predicate == "catalyzes":
                return object_prefix == "GO"

        # Cross-domain relationship constraints
        cross_domain_constraints = {
            "located_in": {
                "CHEBI": ["GO", "PO"],  # Chemicals located in components/structures
            },
            "contains": {
                "GO": ["CHEBI"],  # Components contain chemicals
                "PO": ["CHEBI"],  # Plant structures contain chemicals
            },
            "accumulates_in": {
                "CHEBI": ["GO", "PO"],  # Chemicals accumulate in locations
            },
            "accumulates": {
                "GO": ["CHEBI"],  # Components accumulate chemicals
                "PO": ["CHEBI"],  # Plant structures accumulate chemicals
            },
            "occurs_in": {
                "GO": ["GO", "PO"],  # Processes occur in components/structures
            },
        }

        if predicate in cross_domain_constraints:
            if subject_prefix in cross_domain_constraints[predicate]:
                return (
                    object_prefix in cross_domain_constraints[predicate][subject_prefix]
                )

        # If no specific constraints defined, allow the relationship
        # This provides flexibility for new relationship types
        return True

    def validate_temporal_constraints(self) -> bool:
        """
        Validate temporal aspects like creation date format.

        Checks if temporal information associated with the relationship is valid,
        including creation date format (ISO 8601), and logical temporal consistency.

        Returns:
            bool: True if temporal constraints are satisfied, False otherwise

        Examples:
            >>> rel = Relationship(
            ...     id="REL:001", subject="A:1", predicate="is_a", object="B:1",
            ...     created_date="2023-01-01T00:00:00Z"
            ... )
            >>> rel.validate_temporal_constraints()
            True

            >>> rel.created_date = "invalid-date"
            >>> rel.validate_temporal_constraints()
            False
        """
        # If no creation date is provided, temporal constraints are satisfied
        if not self.created_date:
            return True

        # Validate ISO 8601 date format
        try:
            # Try to parse the date string
            if isinstance(self.created_date, str):
                # Handle various ISO 8601 formats
                date_formats = [
                    "%Y-%m-%dT%H:%M:%SZ",  # 2023-01-01T00:00:00Z
                    "%Y-%m-%dT%H:%M:%S.%fZ",  # 2023-01-01T00:00:00.123456Z
                    "%Y-%m-%dT%H:%M:%S%z",  # 2023-01-01T00:00:00+00:00
                    "%Y-%m-%dT%H:%M:%S.%f%z",  # 2023-01-01T00:00:00.123456+00:00
                    "%Y-%m-%d",  # 2023-01-01
                    "%Y-%m-%dT%H:%M:%S",  # 2023-01-01T00:00:00
                ]

                parsed_date = None
                for date_format in date_formats:
                    try:
                        parsed_date = datetime.strptime(self.created_date, date_format)
                        break
                    except ValueError:
                        continue

                if parsed_date is None:
                    return False

                # Check if date is not in the future (allowing some tolerance)
                current_time = datetime.utcnow()
                if parsed_date > current_time:
                    return False

                # Check if date is not unreasonably old (before 1900)
                if parsed_date.year < 1900:
                    return False

                return True
            else:
                # If created_date is not a string, it's invalid
                return False

        except Exception:
            return False

    def validate_full(self) -> bool:
        """
        Comprehensive validation that calls all validation methods.

        Performs complete validation of the relationship by calling all individual
        validation methods. This is the most thorough validation check available.

        Returns:
            bool: True if the relationship passes all validation checks, False otherwise

        Examples:
            >>> rel = Relationship(
            ...     id="REL:001",
            ...     subject="CHEBI:12345",
            ...     predicate="is_a",
            ...     object="CHEBI:33917",
            ...     confidence=0.95,
            ...     created_date="2023-01-01T00:00:00Z"
            ... )
            >>> rel.validate_full()
            True

            >>> invalid_rel = Relationship(id="invalid", subject="", predicate="bad", object="")
            Traceback (most recent call last):
            ...
            ValueError: Invalid relationship ID format
        """
        try:
            # Call all individual validation methods
            validations = [
                self.validate_id_format(self.id),
                self.validate_predicate(self.predicate),
                self.validate_subject_object_format(self.subject, self.object),
                self.validate_confidence(self.confidence),
                self.validate_circular_relationship(self.subject, self.object),
                self.validate_predicate_semantics(
                    self.predicate, self.subject, self.object
                ),
                self.validate_domain_constraints(
                    self.subject, self.predicate, self.object
                ),
                self.validate_temporal_constraints(),
            ]

            # All validations must pass
            if not all(validations):
                return False

            # Additional comprehensive validation from existing is_valid method
            return self.is_valid()

        except Exception:
            return False

    def get_inverse(self) -> Optional["Relationship"]:
        """
        Return the inverse relationship if one exists.

        Creates the inverse relationship for predicates that have well-defined inverses.
        Not all predicates have meaningful inverses, so this method may return None.

        Returns:
            Optional[Relationship]: The inverse relationship if one exists, None otherwise

        Examples:
            >>> rel = Relationship(id="REL:001", subject="A:1", predicate="part_of", object="B:1")
            >>> inverse = rel.get_inverse()
            >>> inverse.predicate
            'has_part'
            >>> inverse.subject
            'B:1'
            >>> inverse.object
            'A:1'

            >>> is_a_rel = Relationship(id="REL:002", subject="A:1", predicate="is_a", object="B:1")
            >>> is_a_rel.get_inverse()
            None  # is_a doesn't have a simple inverse
        """
        # Define predicate inverses
        predicate_inverses = {
            "part_of": "has_part",
            "has_part": "part_of",
            "regulates": "regulated_by",
            "regulated_by": "regulates",
            "upregulates": "upregulated_by",
            "upregulated_by": "upregulates",
            "downregulates": "downregulated_by",
            "downregulated_by": "downregulates",
            "catalyzes": "catalyzed_by",
            "catalyzed_by": "catalyzes",
            "derives_from": "derives_to",
            "derives_to": "derives_from",
            "located_in": "contains",
            "contains": "located_in",
            "accumulates_in": "accumulates",
            "accumulates": "accumulates_in",
        }

        # Check if current predicate has an inverse
        if self.predicate not in predicate_inverses:
            return None

        try:
            # Create inverse relationship with swapped subject/object and inverse predicate
            inverse_predicate = predicate_inverses[self.predicate]

            # Generate new ID for inverse relationship
            # Extract the numeric part and create a valid inverse ID
            if ":" in self.id:
                prefix, number = self.id.split(":", 1)
                inverse_id = f"INV{prefix}:{number}"
            else:
                # Fallback if ID format is unexpected
                inverse_id = f"INV:999999"

            # Create inverse relationship
            return Relationship(
                id=inverse_id,
                subject=self.object,  # Swap subject and object
                predicate=inverse_predicate,
                object=self.subject,
                confidence=self.confidence,
                source=self.source,
                evidence=self.evidence,
                extraction_method=self.extraction_method,
                created_date=self.created_date,
                provenance=self.provenance,
                is_validated=self.is_validated,
                context=self.context,
                sentence=self.sentence,
                negated=self.negated,
                modality=self.modality,
                directional=True,  # Inverse relationships are directional
            )

        except Exception:
            # If inverse creation fails, return None
            return None

    def get_related_entities(self) -> List[str]:
        """
        Return a list of related entities (subject and object).

        Provides a convenient way to get all entities involved in the relationship,
        which is useful for network analysis and entity extraction.

        Returns:
            List[str]: List containing the subject and object entity IDs

        Examples:
            >>> rel = Relationship(id="REL:001", subject="CHEBI:12345", predicate="is_a", object="CHEBI:33917")
            >>> rel.get_related_entities()
            ['CHEBI:12345', 'CHEBI:33917']

            >>> complex_rel = Relationship(
            ...     id="REL:002", subject="GO:0008150", predicate="occurs_in", object="GO:0005623"
            ... )
            >>> entities = complex_rel.get_related_entities()
            >>> len(entities)
            2
            >>> "GO:0008150" in entities
            True
            >>> "GO:0005623" in entities
            True
        """
        entities = []

        # Add subject if it exists and is not empty
        if self.subject and self.subject.strip():
            entities.append(self.subject)

        # Add object if it exists and is not empty
        if self.object and self.object.strip():
            entities.append(self.object)

        # Remove duplicates while preserving order (in case subject == object, though that should be invalid)
        seen = set()
        unique_entities = []
        for entity in entities:
            if entity not in seen:
                seen.add(entity)
                unique_entities.append(entity)

        return unique_entities

    def is_valid(self) -> bool:
        """
        Perform comprehensive validation of the relationship.

        Checks all aspects of the relationship for validity including ID format,
        predicate, subject/object format, confidence range, and circular relationship
        prevention.

        Returns:
            bool: True if the relationship passes all validation checks, False otherwise

        Examples:
            >>> rel = Relationship(
            ...     id="REL:001",
            ...     subject="CHEBI:12345",
            ...     predicate="is_a",
            ...     object="CHEBI:33917",
            ...     confidence=0.95
            ... )
            >>> rel.is_valid()
            True

            >>> invalid_rel = Relationship(id="invalid", subject="", predicate="bad", object="")
            Traceback (most recent call last):
            ...
            ValueError: Invalid relationship ID format
        """
        try:
            # Basic required field validation
            if not self.validate_id_format(self.id):
                return False

            if not self.validate_predicate(self.predicate):
                return False

            if not self.validate_subject_object_format(self.subject, self.object):
                return False

            if not self.validate_confidence(self.confidence):
                return False

            if not self.validate_circular_relationship(self.subject, self.object):
                return False

            # Validate optional string fields
            string_fields = [
                self.source,
                self.evidence,
                self.extraction_method,
                self.created_date,
                self.provenance,
                self.context,
                self.sentence,
            ]
            for field in string_fields:
                if field is not None and not isinstance(field, str):
                    return False

            # Validate boolean fields
            if not isinstance(self.is_validated, bool):
                return False

            if not isinstance(self.negated, bool):
                return False

            if not isinstance(self.directional, bool):
                return False

            # Validate modality
            if not isinstance(self.modality, str) or not self.modality.strip():
                return False

            return True

        except Exception:
            return False

    def to_dict(self) -> Dict[str, Any]:
        """
        Serialize the relationship to a dictionary.

        Creates a comprehensive dictionary representation of the relationship that
        preserves all attributes and can be used for storage or transmission.

        Returns:
            Dict[str, Any]: Dictionary representation of the relationship

        Examples:
            >>> rel = Relationship(id="REL:001", subject="A:1", predicate="is_a", object="B:1")
            >>> rel_dict = rel.to_dict()
            >>> rel_dict["id"]
            'REL:001'
            >>> rel_dict["subject"]
            'A:1'
        """
        return {
            "id": self.id,
            "subject": self.subject,
            "predicate": self.predicate,
            "object": self.object,
            "confidence": self.confidence,
            "source": self.source,
            "evidence": self.evidence,
            "extraction_method": self.extraction_method,
            "created_date": self.created_date,
            "provenance": self.provenance,
            "is_validated": self.is_validated,
            "context": self.context,
            "sentence": self.sentence,
            "negated": self.negated,
            "modality": self.modality,
            "directional": self.directional,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Relationship":
        """
        Deserialize a relationship from a dictionary.

        Creates a Relationship instance from a dictionary representation, typically
        created by the to_dict() method.

        Args:
            data (Dict[str, Any]): Dictionary containing relationship data

        Returns:
            Relationship: Relationship instance created from the dictionary data

        Raises:
            KeyError: If required fields are missing from the dictionary
            ValueError: If the data contains invalid values

        Examples:
            >>> data = {
            ...     "id": "REL:001",
            ...     "subject": "CHEBI:12345",
            ...     "predicate": "is_a",
            ...     "object": "CHEBI:33917",
            ...     "confidence": 0.95
            ... }
            >>> rel = Relationship.from_dict(data)
            >>> rel.id
            'REL:001'
        """
        # Required fields
        relationship_id = data["id"]
        subject = data["subject"]
        predicate = data["predicate"]
        object = data["object"]

        # Optional fields with defaults
        confidence = data.get("confidence", 1.0)
        source = data.get("source")
        evidence = data.get("evidence")
        extraction_method = data.get("extraction_method")
        created_date = data.get("created_date")
        provenance = data.get("provenance")
        is_validated = data.get("is_validated", False)
        context = data.get("context")
        sentence = data.get("sentence")
        negated = data.get("negated", False)
        modality = data.get("modality", "certain")
        directional = data.get("directional", True)

        return cls(
            id=relationship_id,
            subject=subject,
            predicate=predicate,
            object=object,
            confidence=confidence,
            source=source,
            evidence=evidence,
            extraction_method=extraction_method,
            created_date=created_date,
            provenance=provenance,
            is_validated=is_validated,
            context=context,
            sentence=sentence,
            negated=negated,
            modality=modality,
            directional=directional,
        )

    def to_json(self) -> str:
        """
        Serialize the relationship to a JSON string.

        Creates a JSON representation of the relationship that can be stored in files
        or transmitted over networks.

        Returns:
            str: JSON string representation of the relationship

        Raises:
            TypeError: If the relationship contains non-serializable data

        Examples:
            >>> rel = Relationship(id="REL:001", subject="A:1", predicate="is_a", object="B:1")
            >>> json_str = rel.to_json()
            >>> isinstance(json_str, str)
            True
            >>> "REL:001" in json_str
            True
        """
        return json.dumps(self.to_dict(), ensure_ascii=False, indent=2)

    @classmethod
    def from_json(cls, json_str: str) -> "Relationship":
        """
        Deserialize a relationship from a JSON string.

        Creates a Relationship instance from a JSON string representation, typically
        created by the to_json() method.

        Args:
            json_str (str): JSON string containing relationship data

        Returns:
            Relationship: Relationship instance created from the JSON data

        Raises:
            json.JSONDecodeError: If the JSON string is invalid
            KeyError: If required fields are missing
            ValueError: If the data contains invalid values

        Examples:
            >>> json_str = '{"id": "REL:001", "subject": "A:1", "predicate": "is_a", "object": "B:1"}'
            >>> rel = Relationship.from_json(json_str)
            >>> rel.id
            'REL:001'
        """
        data = json.loads(json_str)
        return cls.from_dict(data)

    def __str__(self) -> str:
        """
        Return a user-friendly string representation of the relationship.

        Creates a readable string that shows the relationship triple in subject-predicate-object
        format, with optional confidence score, negation indicator, and context information.

        Returns:
            str: User-friendly string representation

        Examples:
            >>> rel = Relationship(id="REL:001", subject="A:1", predicate="is_a", object="B:1")
            >>> str(rel)
            'A:1 is_a B:1'

            >>> rel_with_confidence = Relationship(
            ...     id="REL:001",
            ...     subject="CHEBI:12345",
            ...     predicate="is_a",
            ...     object="CHEBI:33917",
            ...     confidence=0.95
            ... )
            >>> str(rel_with_confidence)
            'CHEBI:12345 is_a CHEBI:33917 (confidence: 0.95)'

            >>> negated_rel = Relationship(
            ...     id="REL:001",
            ...     subject="A:1",
            ...     predicate="is_a",
            ...     object="B:1",
            ...     negated=True
            ... )
            >>> str(negated_rel)
            'A:1 NOT is_a B:1'
        """
        # Build the basic triple
        predicate_str = f"NOT {self.predicate}" if self.negated else self.predicate
        base = f"{self.subject} {predicate_str} {self.object}"

        # Add confidence if not default (1.0)
        if self.confidence != 1.0:
            base += f" (confidence: {self.confidence})"

        # Add context if present
        if self.context:
            base += f" [context: {self.context}]"

        return base

    def __repr__(self) -> str:
        """
        Return a detailed string representation for debugging.

        Creates a comprehensive string that shows the relationship's class and
        key attributes, suitable for debugging and development.

        Returns:
            str: Detailed string representation

        Examples:
            >>> rel = Relationship(id="REL:001", subject="A:1", predicate="is_a", object="B:1")
            >>> repr(rel)
            "Relationship(id='REL:001', subject='A:1', predicate='is_a', object='B:1')"

            >>> rel_with_extras = Relationship(
            ...     id="REL:001",
            ...     subject="A:1",
            ...     predicate="is_a",
            ...     object="B:1",
            ...     confidence=0.95,
            ...     negated=True
            ... )
            >>> "confidence=0.95" in repr(rel_with_extras)
            True
        """
        # Build a list of key attributes to show
        attrs = [
            f"id='{self.id}'",
            f"subject='{self.subject}'",
            f"predicate='{self.predicate}'",
            f"object='{self.object}'",
        ]

        # Add optional attributes that have non-default values
        if self.confidence != 1.0:
            attrs.append(f"confidence={self.confidence}")

        if self.source:
            attrs.append(f"source='{self.source}'")

        if self.evidence:
            attrs.append(f"evidence='{self.evidence}'")

        if self.is_validated:
            attrs.append(f"is_validated={self.is_validated}")

        if self.context:
            # Truncate long context for readability
            context_display = (
                self.context[:30] + "..." if len(self.context) > 30 else self.context
            )
            attrs.append(f"context='{context_display}'")

        if self.negated:
            attrs.append(f"negated={self.negated}")

        if self.modality != "certain":
            attrs.append(f"modality='{self.modality}'")

        if not self.directional:
            attrs.append(f"directional={self.directional}")

        return f"Relationship({', '.join(attrs)})"

    def __eq__(self, other) -> bool:
        """
        Compare two relationships for equality.

        Two relationships are considered equal if they have the same ID and all
        other attributes match exactly.

        Args:
            other: Object to compare with

        Returns:
            bool: True if relationships are equal, False otherwise

        Examples:
            >>> rel1 = Relationship(id="REL:001", subject="A:1", predicate="is_a", object="B:1")
            >>> rel2 = Relationship(id="REL:001", subject="A:1", predicate="is_a", object="B:1")
            >>> rel1 == rel2
            True

            >>> rel3 = Relationship(id="REL:002", subject="A:1", predicate="is_a", object="B:1")
            >>> rel1 == rel3
            False
        """
        if not isinstance(other, Relationship):
            return False

        return (
            self.id == other.id
            and self.subject == other.subject
            and self.predicate == other.predicate
            and self.object == other.object
            and self.confidence == other.confidence
            and self.source == other.source
            and self.evidence == other.evidence
            and self.extraction_method == other.extraction_method
            and self.created_date == other.created_date
            and self.provenance == other.provenance
            and self.is_validated == other.is_validated
            and self.context == other.context
            and self.sentence == other.sentence
            and self.negated == other.negated
            and self.modality == other.modality
            and self.directional == other.directional
        )

    def __hash__(self) -> int:
        """
        Generate a hash code for the relationship.

        Uses the relationship ID as the primary hash component since it should be
        unique. This allows relationships to be used in sets and as dictionary keys.

        Returns:
            int: Hash code for the relationship

        Examples:
            >>> rel1 = Relationship(id="REL:001", subject="A:1", predicate="is_a", object="B:1")
            >>> rel2 = Relationship(id="REL:001", subject="A:1", predicate="is_a", object="B:1")
            >>> hash(rel1) == hash(rel2)
            True

            >>> rel_set = {rel1, rel2}
            >>> len(rel_set)  # Should be 1 since they're equal
            1
        """
        # Use ID as primary hash since it should be unique
        # Include subject, predicate, object as secondary components for additional distinction
        return hash((self.id, self.subject, self.predicate, self.object))


@dataclass
class Ontology:
    """
    Dataclass representing a complete ontology with comprehensive container and management functionality.

    This dataclass serves as the primary container for ontological knowledge within the AIM2 project.
    It provides complete ontology management functionality including term and relationship management,
    validation, serialization, and statistical analysis.

    The Ontology class is designed to be flexible and extensible while maintaining strict validation
    for ontology integrity. It supports various ontology formats and standards commonly used in
    bioinformatics and knowledge representation, with comprehensive container operations for
    managing large-scale ontological data.

    Attributes:
        id (str): Unique identifier for the ontology, typically in format "PREFIX:NUMBER"
        name (str): Human-readable name for the ontology
        version (str): Version string for the ontology (default "1.0")
        description (Optional[str]): Detailed description of the ontology
        terms (Dict[str, Term]): Dictionary mapping term IDs to Term objects
        relationships (Dict[str, Relationship]): Dictionary mapping relationship IDs to Relationship objects
        namespaces (List[str]): List of supported namespaces in the ontology
        metadata (Dict[str, Any]): Additional metadata for the ontology
        base_iris (List[str]): List of base IRIs for the ontology
        imports (List[str]): List of imported ontologies
        synonymtypedef (Dict[str, Any]): Synonym type definitions
        term_count (int): Current count of terms in the ontology
        relationship_count (int): Current count of relationships in the ontology
        is_consistent (bool): Flag indicating if the ontology is internally consistent
        validation_errors (List[str]): List of validation errors found in the ontology

    Container Operations:
        - Term management: add_term, remove_term, get_term, find_terms
        - Relationship management: add_relationship, remove_relationship, get_relationships
        - Statistics generation: get_statistics, update_counts
        - Validation: validate_structure, validate_consistency, is_valid, validate_full
        - Container utilities: comprehensive term and relationship container operations

    Validation Features:
        - Ontology ID format validation (PREFIX:NUMBER pattern)
        - Structural integrity validation
        - Consistency validation (circular dependency detection)
        - Term and relationship validation
        - Namespace validation and management
        - Comprehensive ontology validation combining all checks

    Serialization Features:
        - Dictionary serialization (to_dict/from_dict)
        - JSON serialization (to_json/from_json)
        - Round-trip serialization preservation
        - Error handling for invalid serialization data

    Statistical Analysis:
        - Term and relationship counting
        - Namespace analysis
        - Depth and hierarchy analysis
        - Orphan term detection
        - consistency reporting

    Standard Methods:
        - String representation (__str__, __repr__)
        - Equality comparison (__eq__)
        - Hashing support (__hash__) for use in sets and as dict keys
        - Collection support (usable in sets, as dictionary keys)

    Examples:
        Basic ontology creation:
            >>> ontology = Ontology(id="ONT:001", name="Chemical Ontology")
            >>> print(ontology)
            Chemical Ontology v1.0 (ONT:001)

        Full ontology with metadata:
            >>> ontology = Ontology(
            ...     id="ONT:001",
            ...     name="Chemical Ontology",
            ...     version="2023.1",
            ...     description="A comprehensive chemical ontology",
            ...     namespaces=["chemical", "biological_process"],
            ...     metadata={"source": "ChEBI", "license": "CC BY 4.0"}
            ... )

        Term management:
            >>> term = Term(id="CHEBI:12345", name="glucose")
            >>> ontology.add_term(term)
            True
            >>> retrieved_term = ontology.get_term("CHEBI:12345")
            >>> retrieved_term.name
            'glucose'

        Relationship management:
            >>> rel = Relationship(id="REL:001", subject="CHEBI:12345", predicate="is_a", object="CHEBI:33917")
            >>> ontology.add_relationship(rel)
            True
            >>> relationships = ontology.get_relationships("CHEBI:12345")
            >>> len(relationships) > 0
            True

        Validation:
            >>> if ontology.is_valid():
            ...     print("Ontology passes all validation checks")

        Statistics:
            >>> stats = ontology.get_statistics()
            >>> stats["term_count"]
            1
            >>> stats["relationship_count"]
            1

        Serialization:
            >>> ontology_data = ontology.to_dict()
            >>> json_string = ontology.to_json()
            >>> restored = Ontology.from_dict(ontology_data)
            >>> assert ontology == restored
    """

    # Required attributes
    id: str
    name: str

    # Optional attributes with defaults
    version: str = "1.0"
    description: Optional[str] = None
    terms: Dict[str, Term] = field(default_factory=dict)
    relationships: Dict[str, Relationship] = field(default_factory=dict)
    namespaces: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    base_iris: List[str] = field(default_factory=list)
    imports: List[str] = field(default_factory=list)
    synonymtypedef: Dict[str, Any] = field(default_factory=dict)
    term_count: int = 0
    relationship_count: int = 0
    is_consistent: bool = True
    validation_errors: List[str] = field(default_factory=list)

    # Private indexes for efficient term lookups
    _name_index: Dict[str, str] = field(default_factory=dict, init=False, repr=False)
    _synonym_index: Dict[str, List[str]] = field(
        default_factory=dict, init=False, repr=False
    )
    _namespace_index: Dict[str, List[str]] = field(
        default_factory=dict, init=False, repr=False
    )
    _alt_id_index: Dict[str, str] = field(default_factory=dict, init=False, repr=False)

    def __post_init__(self):
        """
        Post-initialization validation and setup.

        Validates the ontology ID and name immediately after creation to ensure
        basic integrity requirements are met. Also updates counts to match
        the current contents.

        Raises:
            ValueError: If ontology ID format is invalid or name is empty
        """
        # Validate required fields
        if not self.validate_id_format(self.id):
            raise ValueError("Invalid ontology ID format")

        if not self.name or not isinstance(self.name, str) or not self.name.strip():
            raise ValueError("Ontology name cannot be empty")

        # Update counts to match current contents
        self.update_counts()

        # Initialize term indexes
        self._build_indexes()

    def validate_id_format(self, ontology_id: str) -> bool:
        """
        Validate the format of an ontology ID.

        Checks if the ontology ID follows the standard ontology ID format of
        "PREFIX:NUMBER" where PREFIX is alphabetic and NUMBER is alphanumeric.

        Args:
            ontology_id (str): The ontology ID to validate

        Returns:
            bool: True if the ID format is valid, False otherwise

        Examples:
            >>> ontology = Ontology(id="ONT:001", name="Test Ontology")
            >>> ontology.validate_id_format("ONT:001")
            True
            >>> ontology.validate_id_format("ONTO:12345")
            True
            >>> ontology.validate_id_format("invalid_id")
            False
            >>> ontology.validate_id_format("ONT:")
            False
        """
        if not ontology_id or not isinstance(ontology_id, str):
            return False

        # Pattern: PREFIX:IDENTIFIER where PREFIX contains letters and IDENTIFIER is alphanumeric
        pattern = r"^[A-Za-z]+:[A-Za-z0-9]+$"
        return bool(re.match(pattern, ontology_id))

    def validate_structure(self) -> bool:
        """
        Validate the structural integrity of the ontology.

        Checks if the ontology structure is valid including proper term and
        relationship dictionaries, valid namespaces, and structural consistency.

        Returns:
            bool: True if the structure is valid, False otherwise

        Examples:
            >>> ontology = Ontology(
            ...     id="ONT:001",
            ...     name="Test Ontology",
            ...     terms={"CHEBI:12345": Term(id="CHEBI:12345", name="glucose")},
            ...     namespaces=["chemical"]
            ... )
            >>> ontology.validate_structure()
            True
        """
        try:
            # Clear previous validation errors
            self.validation_errors = []

            # Validate terms dictionary
            if not isinstance(self.terms, dict):
                self.validation_errors.append("Terms must be a dictionary")
                return False

            # Validate each term in the dictionary
            for term_id, term in self.terms.items():
                if not isinstance(term_id, str):
                    self.validation_errors.append(f"Term ID must be string: {term_id}")
                    return False

                if not isinstance(term, Term):
                    self.validation_errors.append(
                        f"Term value must be Term instance: {term_id}"
                    )
                    return False

                if term.id != term_id:
                    self.validation_errors.append(
                        f"Term ID mismatch: key={term_id}, term.id={term.id}"
                    )
                    return False

            # Validate relationships dictionary
            if not isinstance(self.relationships, dict):
                self.validation_errors.append("Relationships must be a dictionary")
                return False

            # Validate each relationship in the dictionary
            for rel_id, relationship in self.relationships.items():
                if not isinstance(rel_id, str):
                    self.validation_errors.append(
                        f"Relationship ID must be string: {rel_id}"
                    )
                    return False

                if not isinstance(relationship, Relationship):
                    self.validation_errors.append(
                        f"Relationship value must be Relationship instance: {rel_id}"
                    )
                    return False

                if relationship.id != rel_id:
                    self.validation_errors.append(
                        f"Relationship ID mismatch: key={rel_id}, rel.id={relationship.id}"
                    )
                    return False

            # Validate namespaces
            if not isinstance(self.namespaces, list):
                self.validation_errors.append("Namespaces must be a list")
                return False

            if not all(isinstance(ns, str) for ns in self.namespaces):
                self.validation_errors.append("All namespaces must be strings")
                return False

            # Validate metadata
            if not isinstance(self.metadata, dict):
                self.validation_errors.append("Metadata must be a dictionary")
                return False

            # Validate base_iris
            if not isinstance(self.base_iris, list):
                self.validation_errors.append("Base IRIs must be a list")
                return False

            if not all(isinstance(iri, str) for iri in self.base_iris):
                self.validation_errors.append("All base IRIs must be strings")
                return False

            # Validate imports
            if not isinstance(self.imports, list):
                self.validation_errors.append("Imports must be a list")
                return False

            if not all(isinstance(imp, str) for imp in self.imports):
                self.validation_errors.append("All imports must be strings")
                return False

            # Validate synonymtypedef
            if not isinstance(self.synonymtypedef, dict):
                self.validation_errors.append("Synonymtypedef must be a dictionary")
                return False

            # Validate counts
            if not isinstance(self.term_count, int) or self.term_count < 0:
                self.validation_errors.append(
                    "Term count must be a non-negative integer"
                )
                return False

            if (
                not isinstance(self.relationship_count, int)
                or self.relationship_count < 0
            ):
                self.validation_errors.append(
                    "Relationship count must be a non-negative integer"
                )
                return False

            # Validate consistency flag
            if not isinstance(self.is_consistent, bool):
                self.validation_errors.append("Consistency flag must be boolean")
                return False

            # Validate validation_errors list
            if not isinstance(self.validation_errors, list):
                # This is a bit recursive, but we need to validate the validation_errors structure
                self.validation_errors = ["Validation errors must be a list"]
                return False

            return True

        except Exception as e:
            self.validation_errors.append(f"Structure validation error: {str(e)}")
            return False

    def validate_consistency(self) -> bool:
        """
        Validate the consistency of the ontology.

        Checks for circular dependencies, orphaned relationships, and other
        consistency issues within the ontology structure.

        Returns:
            bool: True if the ontology is consistent, False otherwise

        Examples:
            >>> ontology = Ontology(id="ONT:001", name="Test Ontology")
            >>> term1 = Term(id="CHEBI:001", name="term1")
            >>> term2 = Term(id="CHEBI:002", name="term2")
            >>> ontology.add_term(term1)
            True
            >>> ontology.add_term(term2)
            True
            >>> rel = Relationship(id="REL:001", subject="CHEBI:001", predicate="is_a", object="CHEBI:002")
            >>> ontology.add_relationship(rel)
            True
            >>> ontology.validate_consistency()
            True
        """
        try:
            # Track consistency issues
            consistency_errors = []

            # Check for circular dependencies in is_a relationships
            circular_deps = self._detect_circular_dependencies()
            if circular_deps:
                consistency_errors.extend(circular_deps)

            # Check for orphaned relationships (relationships referencing non-existent terms)
            orphaned_rels = self._detect_orphaned_relationships()
            if orphaned_rels:
                consistency_errors.extend(orphaned_rels)

            # Check for duplicate relationships
            duplicate_rels = self._detect_duplicate_relationships()
            if duplicate_rels:
                consistency_errors.extend(duplicate_rels)

            # Update consistency status
            if consistency_errors:
                self.is_consistent = False
                self.validation_errors.extend(consistency_errors)
                return False
            else:
                self.is_consistent = True
                return True

        except Exception as e:
            self.is_consistent = False
            self.validation_errors.append(f"Consistency validation error: {str(e)}")
            return False

    def _detect_circular_dependencies(self) -> List[str]:
        """
        Detect circular dependencies in is_a relationships.

        Returns:
            List[str]: List of circular dependency error messages
        """
        errors = []
        visited = set()
        recursion_stack = set()

        def dfs(term_id: str, path: List[str]) -> bool:
            """Depth-first search to detect cycles."""
            if term_id in recursion_stack:
                cycle_path = path[path.index(term_id) :] + [term_id]
                cycle_str = " -> ".join(cycle_path)
                errors.append(f"Circular dependency detected: {cycle_str}")
                return True

            if term_id in visited:
                return False

            visited.add(term_id)
            recursion_stack.add(term_id)

            # Find all is_a relationships where this term is the subject
            for relationship in self.relationships.values():
                if (
                    relationship.subject == term_id
                    and relationship.predicate == "is_a"
                    and relationship.object in self.terms
                ):
                    if dfs(relationship.object, path + [term_id]):
                        return True

            recursion_stack.remove(term_id)
            return False

        # Check each term for cycles
        for term_id in self.terms:
            if term_id not in visited:
                dfs(term_id, [])

        return errors

    def _detect_orphaned_relationships(self) -> List[str]:
        """
        Detect relationships that reference non-existent terms.

        Returns:
            List[str]: List of orphaned relationship error messages
        """
        errors = []

        for rel_id, relationship in self.relationships.items():
            # Check if subject exists
            if relationship.subject not in self.terms:
                errors.append(
                    f"Orphaned relationship {rel_id}: subject '{relationship.subject}' not found in terms"
                )

            # Check if object exists
            if relationship.object not in self.terms:
                errors.append(
                    f"Orphaned relationship {rel_id}: object '{relationship.object}' not found in terms"
                )

        return errors

    def _detect_duplicate_relationships(self) -> List[str]:
        """
        Detect duplicate relationships (same subject, predicate, object).

        Returns:
            List[str]: List of duplicate relationship error messages
        """
        errors = []
        seen_triples = {}

        for rel_id, relationship in self.relationships.items():
            triple = (relationship.subject, relationship.predicate, relationship.object)
            if triple in seen_triples:
                errors.append(
                    f"Duplicate relationship detected: {rel_id} and {seen_triples[triple]} have same triple {triple}"
                )
            else:
                seen_triples[triple] = rel_id

        return errors

    def is_valid(self) -> bool:
        """
        Perform comprehensive validation of the ontology.

        Checks all aspects of the ontology for validity including ID format,
        name, structure, consistency, and all contained terms and relationships.

        Returns:
            bool: True if the ontology passes all validation checks, False otherwise

        Examples:
            >>> ontology = Ontology(
            ...     id="ONT:001",
            ...     name="Test Ontology",
            ...     version="1.0"
            ... )
            >>> ontology.is_valid()
            True
        """
        try:
            # Clear previous validation errors
            self.validation_errors = []

            # Basic required field validation
            if not self.validate_id_format(self.id):
                return False

            if not self.name or not isinstance(self.name, str) or not self.name.strip():
                return False

            # Structure validation
            if not self.validate_structure():
                return False

            # Consistency validation
            if not self.validate_consistency():
                return False

            # Validate all terms
            for term in self.terms.values():
                if not term.is_valid():
                    self.validation_errors.append(f"Invalid term: {term.id}")
                    return False

            # Validate all relationships
            for relationship in self.relationships.values():
                if not relationship.is_valid():
                    self.validation_errors.append(
                        f"Invalid relationship: {relationship.id}"
                    )
                    return False

            # Validate version format
            if not isinstance(self.version, str) or not self.version.strip():
                self.validation_errors.append("Version must be a non-empty string")
                return False

            # Validate description if provided
            if self.description is not None and not isinstance(self.description, str):
                self.validation_errors.append("Description must be a string or None")
                return False

            return True

        except Exception as e:
            self.validation_errors.append(f"Validation error: {str(e)}")
            return False

    def validate_full(self) -> bool:
        """
        Comprehensive validation that calls all validation methods.

        Performs complete validation of the ontology by calling all individual
        validation methods. This is the most thorough validation check available.

        Returns:
            bool: True if the ontology passes all validation checks, False otherwise

        Examples:
            >>> ontology = Ontology(
            ...     id="ONT:001",
            ...     name="Test Ontology",
            ...     version="1.0"
            ... )
            >>> ontology.validate_full()
            True
        """
        try:
            # Call all validation methods
            validations = [
                self.validate_id_format(self.id),
                self.validate_structure(),
                self.validate_consistency(),
            ]

            # All basic validations must pass
            if not all(validations):
                return False

            # Use comprehensive is_valid validation
            return self.is_valid()

        except Exception as e:
            self.validation_errors.append(f"Full validation error: {str(e)}")
            return False

    def add_term(self, term: Term) -> bool:
        """
        Add a term to the ontology.

        Args:
            term (Term): The term to add

        Returns:
            bool: True if the term was added successfully, False otherwise

        Examples:
            >>> ontology = Ontology(id="ONT:001", name="Test Ontology")
            >>> term = Term(id="CHEBI:12345", name="glucose")
            >>> ontology.add_term(term)
            True
            >>> len(ontology.terms)
            1
        """
        try:
            if not isinstance(term, Term):
                return False

            if not term.is_valid():
                return False

            # Add the term
            self.terms[term.id] = term
            self._add_term_to_indexes(term)
            self.update_counts()
            return True

        except Exception:
            return False

    def remove_term(self, term_id: str) -> bool:
        """
        Remove a term from the ontology.

        Args:
            term_id (str): The ID of the term to remove

        Returns:
            bool: True if the term was removed successfully, False otherwise

        Examples:
            >>> ontology = Ontology(id="ONT:001", name="Test Ontology")
            >>> term = Term(id="CHEBI:12345", name="glucose")
            >>> ontology.add_term(term)
            True
            >>> ontology.remove_term("CHEBI:12345")
            True
            >>> len(ontology.terms)
            0
        """
        try:
            if term_id not in self.terms:
                return False

            # Get the term before removing it (needed for index cleanup)
            term = self.terms[term_id]

            # Remove from indexes first
            self._remove_term_from_indexes(term)

            # Remove the term
            del self.terms[term_id]
            self.update_counts()
            return True

        except Exception:
            return False

    def get_term(self, term_id: str) -> Optional[Term]:
        """
        Get a term from the ontology by ID.

        Args:
            term_id (str): The ID of the term to retrieve

        Returns:
            Optional[Term]: The term if found, None otherwise

        Examples:
            >>> ontology = Ontology(id="ONT:001", name="Test Ontology")
            >>> term = Term(id="CHEBI:12345", name="glucose")
            >>> ontology.add_term(term)
            True
            >>> retrieved = ontology.get_term("CHEBI:12345")
            >>> retrieved.name
            'glucose'
        """
        return self.terms.get(term_id)

    def find_terms(self, search_criteria) -> List[Term]:
        """
        Find terms based on search criteria.

        Args:
            search_criteria: Search criteria (basic implementation)

        Returns:
            List[Term]: List of matching terms

        Examples:
            >>> ontology = Ontology(id="ONT:001", name="Test Ontology")
            >>> term = Term(id="CHEBI:12345", name="glucose")
            >>> ontology.add_term(term)
            True
            >>> found = ontology.find_terms("glucose")
            >>> len(found)
            1
        """
        # Basic implementation - can be extended
        results = []
        if isinstance(search_criteria, str):
            for term in self.terms.values():
                if (
                    search_criteria.lower() in term.name.lower()
                    or (
                        term.definition
                        and search_criteria.lower() in term.definition.lower()
                    )
                    or any(
                        search_criteria.lower() in synonym.lower()
                        for synonym in term.synonyms
                    )
                ):
                    results.append(term)
        return results

    def find_terms_by_name(
        self, name: str, case_sensitive: bool = False
    ) -> Optional[Term]:
        """
        Find a term by its exact name using the name index for fast lookup.

        Args:
            name (str): The name to search for
            case_sensitive (bool): Whether to perform case-sensitive matching (default False)

        Returns:
            Optional[Term]: The term with the exact name, None if not found

        Examples:
            >>> ontology = Ontology(id="ONT:001", name="Test Ontology")
            >>> term = Term(id="CHEBI:12345", name="glucose")
            >>> ontology.add_term(term)
            True
            >>> found = ontology.find_terms_by_name("glucose")
            >>> found.id if found else None
            'CHEBI:12345'
            >>> found = ontology.find_terms_by_name("GLUCOSE")
            >>> found.id if found else None
            'CHEBI:12345'
        """
        if not name:
            return None

        search_key = name.strip() if case_sensitive else name.lower().strip()
        if not search_key:
            return None

        term_id = self._name_index.get(search_key)
        if term_id:
            return self.terms.get(term_id)
        return None

    def find_terms_by_synonym(
        self, synonym: str, case_sensitive: bool = False
    ) -> List[Term]:
        """
        Find terms by synonym using the synonym index for fast lookup.

        Args:
            synonym (str): The synonym to search for
            case_sensitive (bool): Whether to perform case-sensitive matching (default False)

        Returns:
            List[Term]: List of terms that have the given synonym

        Examples:
            >>> ontology = Ontology(id="ONT:001", name="Test Ontology")
            >>> term = Term(id="CHEBI:12345", name="glucose", synonyms=["dextrose", "blood sugar"])
            >>> ontology.add_term(term)
            True
            >>> found = ontology.find_terms_by_synonym("dextrose")
            >>> len(found)
            1
            >>> found[0].name
            'glucose'
        """
        if not synonym:
            return []

        search_key = synonym.strip() if case_sensitive else synonym.lower().strip()
        if not search_key:
            return []

        term_ids = self._synonym_index.get(search_key, [])
        return [self.terms[term_id] for term_id in term_ids if term_id in self.terms]

    def find_terms_by_namespace(self, namespace: str) -> List[Term]:
        """
        Find all terms in a specific namespace using the namespace index for fast lookup.

        Args:
            namespace (str): The namespace to search for

        Returns:
            List[Term]: List of terms in the given namespace

        Examples:
            >>> ontology = Ontology(id="ONT:001", name="Test Ontology")
            >>> term1 = Term(id="CHEBI:12345", name="glucose", namespace="chemical")
            >>> term2 = Term(id="GO:0008150", name="biological_process", namespace="biological_process")
            >>> ontology.add_term(term1)
            True
            >>> ontology.add_term(term2)
            True
            >>> chem_terms = ontology.find_terms_by_namespace("chemical")
            >>> len(chem_terms)
            1
            >>> chem_terms[0].name
            'glucose'
        """
        if not namespace:
            return []

        namespace_key = namespace.strip()
        if not namespace_key:
            return []

        term_ids = self._namespace_index.get(namespace_key, [])
        return [self.terms[term_id] for term_id in term_ids if term_id in self.terms]

    def find_term_by_alt_id(self, alt_id: str) -> Optional[Term]:
        """
        Find a term by its alternative ID using the alternative ID index for fast lookup.

        Args:
            alt_id (str): The alternative ID to search for

        Returns:
            Optional[Term]: The term with the given alternative ID, None if not found

        Examples:
            >>> ontology = Ontology(id="ONT:001", name="Test Ontology")
            >>> term = Term(id="CHEBI:12345", name="glucose", alt_ids=["CHEBI:4167"])
            >>> ontology.add_term(term)
            True
            >>> found = ontology.find_term_by_alt_id("CHEBI:4167")
            >>> found.id if found else None
            'CHEBI:12345'
        """
        if not alt_id:
            return None

        alt_id_key = alt_id.strip()
        if not alt_id_key:
            return None

        term_id = self._alt_id_index.get(alt_id_key)
        if term_id:
            return self.terms.get(term_id)
        return None

    def get_indexed_namespaces(self) -> List[str]:
        """
        Get all namespaces that have terms in the ontology.

        Returns:
            List[str]: List of namespaces with terms

        Examples:
            >>> ontology = Ontology(id="ONT:001", name="Test Ontology")
            >>> term1 = Term(id="CHEBI:12345", name="glucose", namespace="chemical")
            >>> term2 = Term(id="GO:0008150", name="biological_process", namespace="biological_process")
            >>> ontology.add_term(term1)
            True
            >>> ontology.add_term(term2)
            True
            >>> namespaces = ontology.get_indexed_namespaces()
            >>> "chemical" in namespaces
            True
            >>> "biological_process" in namespaces
            True
        """
        return list(self._namespace_index.keys())

    def rebuild_indexes(self) -> None:
        """
        Rebuild all term indexes from scratch.

        This method completely rebuilds all internal indexes used for efficient term lookups.
        It should be called if the indexes become inconsistent or after bulk modifications
        to terms that bypass the normal add_term/remove_term methods.

        Examples:
            >>> ontology = Ontology(id="ONT:001", name="Test Ontology")
            >>> term = Term(id="CHEBI:12345", name="glucose")
            >>> ontology.terms["CHEBI:12345"] = term  # Direct manipulation bypasses indexing
            >>> ontology.rebuild_indexes()  # Rebuild indexes to be consistent
            >>> found = ontology.find_terms_by_name("glucose")
            >>> found.id if found else None
            'CHEBI:12345'
        """
        self._build_indexes()

    def add_relationship(self, relationship: Relationship) -> bool:
        """
        Add a relationship to the ontology.

        Args:
            relationship (Relationship): The relationship to add

        Returns:
            bool: True if the relationship was added successfully, False otherwise

        Examples:
            >>> ontology = Ontology(id="ONT:001", name="Test Ontology")
            >>> rel = Relationship(id="REL:001", subject="CHEBI:001", predicate="is_a", object="CHEBI:002")
            >>> ontology.add_relationship(rel)
            True
        """
        try:
            if not isinstance(relationship, Relationship):
                return False

            if not relationship.is_valid():
                return False

            # Add the relationship
            self.relationships[relationship.id] = relationship
            self.update_counts()
            return True

        except Exception:
            return False

    def remove_relationship(self, relationship_id: str) -> bool:
        """
        Remove a relationship from the ontology.

        Args:
            relationship_id (str): The ID of the relationship to remove

        Returns:
            bool: True if the relationship was removed successfully, False otherwise

        Examples:
            >>> ontology = Ontology(id="ONT:001", name="Test Ontology")
            >>> rel = Relationship(id="REL:001", subject="CHEBI:001", predicate="is_a", object="CHEBI:002")
            >>> ontology.add_relationship(rel)
            True
            >>> ontology.remove_relationship("REL:001")
            True
        """
        try:
            if relationship_id not in self.relationships:
                return False

            # Remove the relationship
            del self.relationships[relationship_id]
            self.update_counts()
            return True

        except Exception:
            return False

    def get_relationships(self, term_id: Optional[str] = None) -> List[Relationship]:
        """
        Get relationships, optionally filtered by term ID.

        Args:
            term_id (Optional[str]): If provided, return only relationships involving this term

        Returns:
            List[Relationship]: List of relationships

        Examples:
            >>> ontology = Ontology(id="ONT:001", name="Test Ontology")
            >>> rel = Relationship(id="REL:001", subject="CHEBI:001", predicate="is_a", object="CHEBI:002")
            >>> ontology.add_relationship(rel)
            True
            >>> rels = ontology.get_relationships("CHEBI:001")
            >>> len(rels)
            1
        """
        if term_id is None:
            return list(self.relationships.values())

        # Filter relationships involving the specified term
        results = []
        for relationship in self.relationships.values():
            if relationship.subject == term_id or relationship.object == term_id:
                results.append(relationship)
        return results

    def update_counts(self):
        """
        Update term and relationship counts to match current contents.

        Examples:
            >>> ontology = Ontology(id="ONT:001", name="Test Ontology")
            >>> ontology.term_count
            0
            >>> term = Term(id="CHEBI:12345", name="glucose")
            >>> ontology.terms["CHEBI:12345"] = term
            >>> ontology.update_counts()
            >>> ontology.term_count
            1
        """
        self.term_count = len(self.terms)
        self.relationship_count = len(self.relationships)

    def _build_indexes(self) -> None:
        """
        Build all term indexes from current terms.

        This method reconstructs all internal indexes used for efficient term lookups.
        It should be called after bulk modifications to the terms dictionary or when
        indexes need to be rebuilt from scratch.

        The following indexes are built:
        - _name_index: Maps term names to term IDs for fast name-based lookups
        - _synonym_index: Maps synonyms to lists of term IDs that use them
        - _namespace_index: Maps namespaces to lists of term IDs in that namespace
        - _alt_id_index: Maps alternative IDs to the primary term IDs

        Examples:
            >>> ontology = Ontology(id="ONT:001", name="Test Ontology")
            >>> term = Term(id="CHEBI:12345", name="glucose", synonyms=["dextrose"])
            >>> ontology.terms["CHEBI:12345"] = term
            >>> ontology._build_indexes()
            >>> "glucose" in ontology._name_index
            True
            >>> "dextrose" in ontology._synonym_index
            True
        """
        # Clear existing indexes
        self._name_index.clear()
        self._synonym_index.clear()
        self._namespace_index.clear()
        self._alt_id_index.clear()

        # Build indexes from current terms
        for term_id, term in self.terms.items():
            # Index by name (case-insensitive for better matching)
            if term.name:
                name_key = term.name.lower().strip()
                if name_key:
                    self._name_index[name_key] = term_id

            # Index by synonyms (case-insensitive)
            for synonym in term.synonyms:
                if synonym:
                    synonym_key = synonym.lower().strip()
                    if synonym_key:
                        if synonym_key not in self._synonym_index:
                            self._synonym_index[synonym_key] = []
                        self._synonym_index[synonym_key].append(term_id)

            # Index by namespace
            if term.namespace:
                namespace_key = term.namespace.strip()
                if namespace_key:
                    if namespace_key not in self._namespace_index:
                        self._namespace_index[namespace_key] = []
                    self._namespace_index[namespace_key].append(term_id)

            # Index by alternative IDs
            for alt_id in term.alt_ids:
                if alt_id and alt_id != term_id:
                    alt_id_key = alt_id.strip()
                    if alt_id_key:
                        self._alt_id_index[alt_id_key] = term_id

    def _add_term_to_indexes(self, term: Term) -> None:
        """
        Add a single term to all indexes.

        This method updates all internal indexes when a new term is added to the ontology.
        It's more efficient than rebuilding all indexes when only one term is added.

        Args:
            term (Term): The term to add to indexes

        Examples:
            >>> ontology = Ontology(id="ONT:001", name="Test Ontology")
            >>> term = Term(id="CHEBI:12345", name="glucose", synonyms=["dextrose"])
            >>> ontology._add_term_to_indexes(term)
            >>> "glucose" in ontology._name_index
            True
        """
        # Index by name (case-insensitive for better matching)
        if term.name:
            name_key = term.name.lower().strip()
            if name_key:
                self._name_index[name_key] = term.id

        # Index by synonyms (case-insensitive)
        for synonym in term.synonyms:
            if synonym:
                synonym_key = synonym.lower().strip()
                if synonym_key:
                    if synonym_key not in self._synonym_index:
                        self._synonym_index[synonym_key] = []
                    self._synonym_index[synonym_key].append(term.id)

        # Index by namespace
        if term.namespace:
            namespace_key = term.namespace.strip()
            if namespace_key:
                if namespace_key not in self._namespace_index:
                    self._namespace_index[namespace_key] = []
                self._namespace_index[namespace_key].append(term.id)

        # Index by alternative IDs
        for alt_id in term.alt_ids:
            if alt_id and alt_id != term.id:
                alt_id_key = alt_id.strip()
                if alt_id_key:
                    self._alt_id_index[alt_id_key] = term.id

    def _remove_term_from_indexes(self, term: Term) -> None:
        """
        Remove a single term from all indexes.

        This method updates all internal indexes when a term is removed from the ontology.
        It cleans up all references to the term from the various indexes.

        Args:
            term (Term): The term to remove from indexes

        Examples:
            >>> ontology = Ontology(id="ONT:001", name="Test Ontology")
            >>> term = Term(id="CHEBI:12345", name="glucose", synonyms=["dextrose"])
            >>> ontology._add_term_to_indexes(term)
            >>> ontology._remove_term_from_indexes(term)
            >>> "glucose" in ontology._name_index
            False
        """
        # Remove from name index (case-insensitive)
        if term.name:
            name_key = term.name.lower().strip()
            if name_key and name_key in self._name_index:
                if self._name_index[name_key] == term.id:
                    del self._name_index[name_key]

        # Remove from synonym index (case-insensitive)
        for synonym in term.synonyms:
            if synonym:
                synonym_key = synonym.lower().strip()
                if synonym_key and synonym_key in self._synonym_index:
                    if term.id in self._synonym_index[synonym_key]:
                        self._synonym_index[synonym_key].remove(term.id)
                    # Remove the synonym key if no terms use it anymore
                    if not self._synonym_index[synonym_key]:
                        del self._synonym_index[synonym_key]

        # Remove from namespace index
        if term.namespace:
            namespace_key = term.namespace.strip()
            if namespace_key and namespace_key in self._namespace_index:
                if term.id in self._namespace_index[namespace_key]:
                    self._namespace_index[namespace_key].remove(term.id)
                # Remove the namespace key if no terms use it anymore
                if not self._namespace_index[namespace_key]:
                    del self._namespace_index[namespace_key]

        # Remove from alternative ID index
        for alt_id in term.alt_ids:
            if alt_id and alt_id != term.id:
                alt_id_key = alt_id.strip()
                if alt_id_key and alt_id_key in self._alt_id_index:
                    if self._alt_id_index[alt_id_key] == term.id:
                        del self._alt_id_index[alt_id_key]

    def get_statistics(self) -> Dict[str, Any]:
        """
        Get comprehensive statistics about the ontology.

        Returns:
            Dict[str, Any]: Dictionary containing various statistics

        Examples:
            >>> ontology = Ontology(id="ONT:001", name="Test Ontology")
            >>> stats = ontology.get_statistics()
            >>> stats["term_count"]
            0
            >>> stats["relationship_count"]
            0
        """
        # Update counts first
        self.update_counts()

        # Calculate additional statistics
        namespace_count = len(self.namespaces)

        # Count orphan terms (terms with no relationships)
        terms_with_relationships = set()
        for rel in self.relationships.values():
            terms_with_relationships.add(rel.subject)
            terms_with_relationships.add(rel.object)

        orphan_terms = len(self.terms) - len(
            terms_with_relationships.intersection(self.terms.keys())
        )

        # Count relationship types
        predicate_counts = {}
        for rel in self.relationships.values():
            predicate_counts[rel.predicate] = predicate_counts.get(rel.predicate, 0) + 1

        # Calculate average relationships per term
        avg_relationships_per_term = (
            self.relationship_count * 2 / self.term_count if self.term_count > 0 else 0
        )

        # Estimate max depth (simple implementation)
        max_depth = self._calculate_max_depth()

        return {
            "id": self.id,
            "name": self.name,
            "version": self.version,
            "term_count": self.term_count,
            "relationship_count": self.relationship_count,
            "namespace_count": namespace_count,
            "orphan_terms": orphan_terms,
            "predicate_counts": predicate_counts,
            "avg_relationships_per_term": round(avg_relationships_per_term, 2),
            "max_depth": max_depth,
            "is_consistent": self.is_consistent,
            "validation_error_count": len(self.validation_errors),
            "has_imports": len(self.imports) > 0,
            "has_base_iris": len(self.base_iris) > 0,
            "metadata_keys": list(self.metadata.keys()),
        }

    def _calculate_max_depth(self) -> int:
        """
        Calculate the maximum depth of the ontology hierarchy.

        Returns:
            int: Maximum depth found in the hierarchy
        """
        if not self.terms or not self.relationships:
            return 0

        # Find root terms (terms that are not subjects in any is_a relationship)
        subjects_in_is_a = set()
        for rel in self.relationships.values():
            if rel.predicate == "is_a":
                subjects_in_is_a.add(rel.subject)

        root_terms = [
            term_id for term_id in self.terms if term_id not in subjects_in_is_a
        ]

        if not root_terms:
            # If there are no clear roots, all terms might be in cycles
            # Return 1 for any terms that exist
            return 1 if self.terms else 0

        max_depth = 0

        def calculate_depth(term_id: str, visited: set) -> int:
            """Calculate depth from a given term downward."""
            if term_id in visited:
                return 0  # Avoid cycles

            visited.add(term_id)
            max_child_depth = 0

            # Find all children (subjects in is_a relationships where this term is object)
            for rel in self.relationships.values():
                if rel.predicate == "is_a" and rel.object == term_id:
                    child_depth = calculate_depth(rel.subject, visited.copy())
                    max_child_depth = max(max_child_depth, child_depth)

            return max_child_depth + 1

        # Calculate depth from each root
        for root in root_terms:
            depth = calculate_depth(root, set())
            max_depth = max(max_depth, depth)

        return max_depth

    def to_dict(self) -> Dict[str, Any]:
        """
        Serialize the ontology to a dictionary.

        Creates a comprehensive dictionary representation of the ontology that
        preserves all attributes and can be used for storage or transmission.

        Returns:
            Dict[str, Any]: Dictionary representation of the ontology

        Examples:
            >>> ontology = Ontology(id="ONT:001", name="Test Ontology")
            >>> ontology_dict = ontology.to_dict()
            >>> ontology_dict["id"]
            'ONT:001'
            >>> ontology_dict["name"]
            'Test Ontology'
        """
        return {
            "id": self.id,
            "name": self.name,
            "version": self.version,
            "description": self.description,
            "terms": {term_id: term.to_dict() for term_id, term in self.terms.items()},
            "relationships": {
                rel_id: rel.to_dict() for rel_id, rel in self.relationships.items()
            },
            "namespaces": self.namespaces.copy(),
            "metadata": self.metadata.copy(),
            "base_iris": self.base_iris.copy(),
            "imports": self.imports.copy(),
            "synonymtypedef": self.synonymtypedef.copy(),
            "term_count": self.term_count,
            "relationship_count": self.relationship_count,
            "is_consistent": self.is_consistent,
            "validation_errors": self.validation_errors.copy(),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Ontology":
        """
        Deserialize an ontology from a dictionary.

        Creates an Ontology instance from a dictionary representation, typically
        created by the to_dict() method.

        Args:
            data (Dict[str, Any]): Dictionary containing ontology data

        Returns:
            Ontology: Ontology instance created from the dictionary data

        Raises:
            KeyError: If required fields are missing from the dictionary
            ValueError: If the data contains invalid values

        Examples:
            >>> data = {
            ...     "id": "ONT:001",
            ...     "name": "Test Ontology",
            ...     "version": "1.0"
            ... }
            >>> ontology = Ontology.from_dict(data)
            >>> ontology.id
            'ONT:001'
        """
        # Required fields
        ontology_id = data["id"]
        name = data["name"]

        # Optional fields with defaults
        version = data.get("version", "1.0")
        description = data.get("description")

        # Deserialize terms
        terms_data = data.get("terms", {})
        terms = {
            term_id: Term.from_dict(term_dict)
            for term_id, term_dict in terms_data.items()
        }

        # Deserialize relationships
        relationships_data = data.get("relationships", {})
        relationships = {
            rel_id: Relationship.from_dict(rel_dict)
            for rel_id, rel_dict in relationships_data.items()
        }

        namespaces = data.get("namespaces", [])
        metadata = data.get("metadata", {})
        base_iris = data.get("base_iris", [])
        imports = data.get("imports", [])
        synonymtypedef = data.get("synonymtypedef", {})
        term_count = data.get("term_count", 0)
        relationship_count = data.get("relationship_count", 0)
        is_consistent = data.get("is_consistent", True)
        validation_errors = data.get("validation_errors", [])

        return cls(
            id=ontology_id,
            name=name,
            version=version,
            description=description,
            terms=terms,
            relationships=relationships,
            namespaces=namespaces,
            metadata=metadata,
            base_iris=base_iris,
            imports=imports,
            synonymtypedef=synonymtypedef,
            term_count=term_count,
            relationship_count=relationship_count,
            is_consistent=is_consistent,
            validation_errors=validation_errors,
        )

    def to_json(self) -> str:
        """
        Serialize the ontology to a JSON string.

        Creates a JSON representation of the ontology that can be stored in files
        or transmitted over networks.

        Returns:
            str: JSON string representation of the ontology

        Raises:
            TypeError: If the ontology contains non-serializable data

        Examples:
            >>> ontology = Ontology(id="ONT:001", name="Test Ontology")
            >>> json_str = ontology.to_json()
            >>> isinstance(json_str, str)
            True
            >>> "ONT:001" in json_str
            True
        """
        return json.dumps(self.to_dict(), ensure_ascii=False, indent=2)

    @classmethod
    def from_json(cls, json_str: str) -> "Ontology":
        """
        Deserialize an ontology from a JSON string.

        Creates an Ontology instance from a JSON string representation, typically
        created by the to_json() method.

        Args:
            json_str (str): JSON string containing ontology data

        Returns:
            Ontology: Ontology instance created from the JSON data

        Raises:
            json.JSONDecodeError: If the JSON string is invalid
            KeyError: If required fields are missing
            ValueError: If the data contains invalid values

        Examples:
            >>> json_str = '{"id": "ONT:001", "name": "Test Ontology"}'
            >>> ontology = Ontology.from_json(json_str)
            >>> ontology.id
            'ONT:001'
        """
        data = json.loads(json_str)
        return cls.from_dict(data)

    def __str__(self) -> str:
        """
        Return a user-friendly string representation of the ontology.

        Creates a readable string that includes the ontology name, version, ID,
        and optionally statistics or inconsistency status.

        Returns:
            str: User-friendly string representation

        Examples:
            >>> ontology = Ontology(id="ONT:001", name="Test Ontology")
            >>> str(ontology)
            'Test Ontology v1.0 (ONT:001)'

            >>> ontology_with_desc = Ontology(
            ...     id="ONT:001",
            ...     name="Chemical Ontology",
            ...     description="A chemical ontology"
            ... )
            >>> str(ontology_with_desc)
            'Chemical Ontology v1.0 (ONT:001): A chemical ontology'
        """
        # Base representation
        base = f"{self.name} v{self.version} ({self.id})"

        # Add inconsistency flag if needed
        if not self.is_consistent:
            base = f"[INCONSISTENT] {base}"

        # Add description if present
        if self.description:
            base += f": {self.description}"

        # Add statistics if there are terms/relationships
        elif self.term_count > 0 or self.relationship_count > 0:
            base += (
                f" - {self.term_count} terms, {self.relationship_count} relationships"
            )

        return base

    def __repr__(self) -> str:
        """
        Return a detailed string representation for debugging.

        Creates a comprehensive string that shows the ontology's class and
        key attributes, suitable for debugging and development.

        Returns:
            str: Detailed string representation

        Examples:
            >>> ontology = Ontology(id="ONT:001", name="Test Ontology")
            >>> repr(ontology)
            "Ontology(id='ONT:001', name='Test Ontology', version='1.0')"
        """
        # Build a list of key attributes to show
        attrs = [f"id='{self.id}'", f"name='{self.name}'", f"version='{self.version}'"]

        # Add optional attributes that have non-default values
        if self.description:
            desc_display = (
                self.description[:50] + "..."
                if len(self.description) > 50
                else self.description
            )
            attrs.append(f"description='{desc_display}'")

        if self.term_count > 0:
            attrs.append(f"term_count={self.term_count}")

        if self.relationship_count > 0:
            attrs.append(f"relationship_count={self.relationship_count}")

        if self.namespaces:
            attrs.append(f"namespaces={self.namespaces}")

        if not self.is_consistent:
            attrs.append(f"is_consistent={self.is_consistent}")

        if self.validation_errors:
            attrs.append(f"validation_errors={len(self.validation_errors)} errors")

        return f"Ontology({', '.join(attrs)})"

    def __eq__(self, other) -> bool:
        """
        Compare two ontologies for equality.

        Two ontologies are considered equal if they have the same ID and all
        other attributes match exactly.

        Args:
            other: Object to compare with

        Returns:
            bool: True if ontologies are equal, False otherwise

        Examples:
            >>> ont1 = Ontology(id="ONT:001", name="Test Ontology")
            >>> ont2 = Ontology(id="ONT:001", name="Test Ontology")
            >>> ont1 == ont2
            True

            >>> ont3 = Ontology(id="ONT:002", name="Test Ontology")
            >>> ont1 == ont3
            False
        """
        if not isinstance(other, Ontology):
            return False

        return (
            self.id == other.id
            and self.name == other.name
            and self.version == other.version
            and self.description == other.description
            and self.terms == other.terms
            and self.relationships == other.relationships
            and self.namespaces == other.namespaces
            and self.metadata == other.metadata
            and self.base_iris == other.base_iris
            and self.imports == other.imports
            and self.synonymtypedef == other.synonymtypedef
            and self.term_count == other.term_count
            and self.relationship_count == other.relationship_count
            and self.is_consistent == other.is_consistent
            and self.validation_errors == other.validation_errors
        )

    def __hash__(self) -> int:
        """
        Generate a hash code for the ontology.

        Uses the ontology ID as the primary hash component since it should be
        unique. This allows ontologies to be used in sets and as dictionary keys.

        Returns:
            int: Hash code for the ontology

        Examples:
            >>> ont1 = Ontology(id="ONT:001", name="Test Ontology")
            >>> ont2 = Ontology(id="ONT:001", name="Test Ontology")
            >>> hash(ont1) == hash(ont2)
            True

            >>> ont_set = {ont1, ont2}
            >>> len(ont_set)  # Should be 1 since they're equal
            1
        """
        # Use ID as primary hash since it should be unique
        # Include name and version as secondary components for additional distinction
        return hash((self.id, self.name, self.version))
