"""
RDF Triple Extraction Design for AIM2 Ontology Project

This module designs the RDF triple extraction interface and data structures
for integration with the existing OWL parser architecture.

Design follows existing patterns from:
- aim2_project/aim2_ontology/models.py (Term, Relationship, Ontology classes)
- aim2_project/aim2_ontology/parsers/__init__.py (AbstractParser, OWLParser)
- aim2_project/exceptions.py (AIM2Exception hierarchy)
"""

import json
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
from urllib.parse import urlparse


@dataclass
class RDFTriple:
    """
    Represents an RDF triple (subject-predicate-object) with comprehensive validation and metadata.

    This class follows the same design patterns as the existing Term and Relationship classes,
    providing validation, serialization, and standard Python object methods.

    Attributes:
        subject (str): The subject URI or identifier
        predicate (str): The predicate URI (relationship type)
        object (str): The object URI, literal value, or identifier
        object_type (str): Type of object ('uri', 'literal', 'blank_node')
        language (Optional[str]): Language tag for literal objects (e.g., 'en', 'es')
        datatype (Optional[str]): Datatype URI for typed literals (e.g., xsd:string)
        source_graph (Optional[str]): Named graph or context where triple was found
        confidence (float): Confidence score for extracted triple (0.0-1.0)
        extraction_method (str): Method used for extraction ('rdflib', 'owlready2', 'manual')
        namespace_prefixes (Dict[str, str]): Namespace prefix mappings for compact representation
        metadata (Dict[str, Any]): Additional metadata about the triple

    Examples:
        Basic RDF triple:
            >>> triple = RDFTriple(
            ...     subject="http://example.org/Person1",
            ...     predicate="http://www.w3.org/1999/02/22-rdf-syntax-ns#type",
            ...     object="http://example.org/Person"
            ... )

        Literal object triple:
            >>> triple = RDFTriple(
            ...     subject="http://example.org/Person1",
            ...     predicate="http://example.org/hasName",
            ...     object="John Doe",
            ...     object_type="literal",
            ...     language="en"
            ... )

        Typed literal triple:
            >>> triple = RDFTriple(
            ...     subject="http://example.org/Person1",
            ...     predicate="http://example.org/hasAge",
            ...     object="25",
            ...     object_type="literal",
            ...     datatype="http://www.w3.org/2001/XMLSchema#integer"
            ... )
    """

    subject: str
    predicate: str
    object: str
    object_type: str = "uri"  # 'uri', 'literal', 'blank_node'
    language: Optional[str] = None
    datatype: Optional[str] = None
    source_graph: Optional[str] = None
    confidence: float = 1.0
    extraction_method: str = "unknown"
    namespace_prefixes: Dict[str, str] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        """Validate triple after initialization."""
        self._validation_errors: List[str] = []
        self.validate()

    def validate(self) -> bool:
        """
        Validate the RDF triple for correctness and consistency.

        Performs comprehensive validation including:
        - URI format validation for URIs
        - Object type consistency checks
        - Language and datatype validation
        - Confidence score validation
        - Namespace prefix validation

        Returns:
            bool: True if valid, False otherwise

        Side Effects:
            Populates self._validation_errors with any validation issues
        """
        self._validation_errors.clear()

        # Validate required fields
        if not self.subject or not self.subject.strip():
            self._validation_errors.append("Subject cannot be empty")
        if not self.predicate or not self.predicate.strip():
            self._validation_errors.append("Predicate cannot be empty")
        if not self.object or not self.object.strip():
            self._validation_errors.append("Object cannot be empty")

        # Validate object_type
        valid_object_types = {"uri", "literal", "blank_node"}
        if self.object_type not in valid_object_types:
            self._validation_errors.append(
                f"Invalid object_type '{self.object_type}'. Must be one of: {valid_object_types}"
            )

        # Validate URIs (subject, predicate, and URI objects)
        if not self._is_valid_uri_or_identifier(self.subject):
            self._validation_errors.append(f"Invalid subject URI: {self.subject}")

        if not self._is_valid_uri_or_identifier(self.predicate):
            self._validation_errors.append(f"Invalid predicate URI: {self.predicate}")

        if self.object_type == "uri" and not self._is_valid_uri_or_identifier(
            self.object
        ):
            self._validation_errors.append(f"Invalid object URI: {self.object}")

        # Validate blank nodes
        if self.object_type == "blank_node" and not self._is_valid_blank_node(
            self.object
        ):
            self._validation_errors.append(f"Invalid blank node: {self.object}")

        # Validate language tag
        if self.language and not self._is_valid_language_tag(self.language):
            self._validation_errors.append(f"Invalid language tag: {self.language}")

        # Validate datatype
        if self.datatype and not self._is_valid_uri_or_identifier(self.datatype):
            self._validation_errors.append(f"Invalid datatype URI: {self.datatype}")

        # Validate confidence score
        if not (0.0 <= self.confidence <= 1.0):
            self._validation_errors.append(
                f"Confidence must be between 0.0 and 1.0, got: {self.confidence}"
            )

        # Validate language/datatype constraints for literals
        if self.object_type == "literal":
            if self.language and self.datatype:
                self._validation_errors.append(
                    "Literal cannot have both language tag and datatype"
                )
        elif self.object_type != "literal":
            if self.language:
                self._validation_errors.append(
                    f"Language tag only valid for literals, not {self.object_type}"
                )
            if self.datatype:
                self._validation_errors.append(
                    f"Datatype only valid for literals, not {self.object_type}"
                )

        return len(self._validation_errors) == 0

    def _is_valid_uri_or_identifier(self, value: str) -> bool:
        """Check if value is a valid URI or namespace:identifier format."""
        if not value:
            return False

        # Check for full URI
        parsed = urlparse(value)
        if parsed.scheme and parsed.netloc:
            return True

        # Check for namespace:identifier format (e.g., "rdfs:label")
        if ":" in value and not value.startswith("http"):
            parts = value.split(":", 1)
            if len(parts) == 2 and parts[0] and parts[1]:
                return True

        # Check for relative URI or fragment
        if value.startswith("#") or value.startswith("/"):
            return True

        return False

    def _is_valid_blank_node(self, value: str) -> bool:
        """Check if value is a valid blank node identifier."""
        # Blank nodes typically start with '_:'
        return value.startswith("_:") and len(value) > 2

    def _is_valid_language_tag(self, lang: str) -> bool:
        """Check if language tag follows RFC 5646 format (simplified)."""
        # Simplified language tag validation (e.g., 'en', 'en-US', 'zh-CN')
        pattern = r"^[a-z]{2,3}(-[A-Z]{2})?(-[a-zA-Z0-9]+)*$"
        return bool(re.match(pattern, lang))

    def is_valid(self) -> bool:
        """
        Check if the triple is valid.

        Returns:
            bool: True if valid, False otherwise
        """
        return self.validate()

    def get_validation_errors(self) -> List[str]:
        """
        Get list of validation errors.

        Returns:
            List[str]: List of validation error messages
        """
        return self._validation_errors.copy()

    def to_compact_form(self) -> str:
        """
        Convert triple to compact string representation using namespace prefixes.

        Returns:
            str: Compact triple representation (e.g., "ex:Person1 rdf:type ex:Person")
        """

        def compact_uri(uri: str) -> str:
            """Convert URI to compact form if possible."""
            for prefix, namespace in self.namespace_prefixes.items():
                if uri.startswith(namespace):
                    return f"{prefix}:{uri[len(namespace):]}"
            return f"<{uri}>" if self._is_valid_uri_or_identifier(uri) else uri

        subject_compact = compact_uri(self.subject)
        predicate_compact = compact_uri(self.predicate)

        if self.object_type == "uri":
            object_compact = compact_uri(self.object)
        elif self.object_type == "literal":
            if self.language:
                object_compact = f'"{self.object}"@{self.language}'
            elif self.datatype:
                datatype_compact = compact_uri(self.datatype)
                object_compact = f'"{self.object}"^^{datatype_compact}'
            else:
                object_compact = f'"{self.object}"'
        else:  # blank_node
            object_compact = self.object

        return f"{subject_compact} {predicate_compact} {object_compact}"

    def to_ntriples(self) -> str:
        """
        Convert triple to N-Triples format.

        Returns:
            str: N-Triples representation
        """

        def format_uri(uri: str) -> str:
            """Format URI for N-Triples."""
            if self._is_valid_uri_or_identifier(uri) and not uri.startswith("<"):
                return f"<{uri}>"
            return uri

        subject_nt = (
            format_uri(self.subject)
            if not self.subject.startswith("_:")
            else self.subject
        )
        predicate_nt = format_uri(self.predicate)

        if self.object_type == "uri":
            object_nt = format_uri(self.object)
        elif self.object_type == "literal":
            escaped_object = self.object.replace("\\", "\\\\").replace('"', '\\"')
            if self.language:
                object_nt = f'"{escaped_object}"@{self.language}'
            elif self.datatype:
                object_nt = f'"{escaped_object}"^^<{self.datatype}>'
            else:
                object_nt = f'"{escaped_object}"'
        else:  # blank_node
            object_nt = self.object

        return f"{subject_nt} {predicate_nt} {object_nt} ."

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert triple to dictionary representation for serialization.

        Returns:
            Dict[str, Any]: Dictionary representation
        """
        return {
            "subject": self.subject,
            "predicate": self.predicate,
            "object": self.object,
            "object_type": self.object_type,
            "language": self.language,
            "datatype": self.datatype,
            "source_graph": self.source_graph,
            "confidence": self.confidence,
            "extraction_method": self.extraction_method,
            "namespace_prefixes": self.namespace_prefixes,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "RDFTriple":
        """
        Create RDFTriple from dictionary representation.

        Args:
            data (Dict[str, Any]): Dictionary representation

        Returns:
            RDFTriple: Reconstructed triple
        """
        return cls(
            subject=data["subject"],
            predicate=data["predicate"],
            object=data["object"],
            object_type=data.get("object_type", "uri"),
            language=data.get("language"),
            datatype=data.get("datatype"),
            source_graph=data.get("source_graph"),
            confidence=data.get("confidence", 1.0),
            extraction_method=data.get("extraction_method", "unknown"),
            namespace_prefixes=data.get("namespace_prefixes", {}),
            metadata=data.get("metadata", {}),
        )

    def to_json(self, indent: Optional[int] = None) -> str:
        """
        Convert triple to JSON representation.

        Args:
            indent (Optional[int]): JSON indentation level

        Returns:
            str: JSON representation
        """
        return json.dumps(self.to_dict(), indent=indent, ensure_ascii=False)

    @classmethod
    def from_json(cls, json_str: str) -> "RDFTriple":
        """
        Create RDFTriple from JSON representation.

        Args:
            json_str (str): JSON representation

        Returns:
            RDFTriple: Reconstructed triple
        """
        data = json.loads(json_str)
        return cls.from_dict(data)

    def __eq__(self, other) -> bool:
        """Check equality based on subject, predicate, object, and object metadata."""
        if not isinstance(other, RDFTriple):
            return False
        return (
            self.subject == other.subject
            and self.predicate == other.predicate
            and self.object == other.object
            and self.object_type == other.object_type
            and self.language == other.language
            and self.datatype == other.datatype
        )

    def __hash__(self) -> int:
        """Generate hash based on core triple components."""
        return hash(
            (
                self.subject,
                self.predicate,
                self.object,
                self.object_type,
                self.language,
                self.datatype,
            )
        )

    def __str__(self) -> str:
        """String representation using compact form."""
        return self.to_compact_form()

    def __repr__(self) -> str:
        """Detailed string representation for debugging."""
        return (
            f"RDFTriple(subject='{self.subject}', predicate='{self.predicate}', "
            f"object='{self.object}', object_type='{self.object_type}', "
            f"confidence={self.confidence})"
        )


class TripleExtractor(ABC):
    """
    Abstract base class for RDF triple extraction strategies.

    This allows for different extraction approaches (rdflib-based, owlready2-based, etc.)
    while maintaining a consistent interface.
    """

    @abstractmethod
    def extract_triples(self, parsed_result: Dict[str, Any]) -> List[RDFTriple]:
        """
        Extract RDF triples from parsed ontology data.

        Args:
            parsed_result (Dict[str, Any]): Parsed ontology data

        Returns:
            List[RDFTriple]: Extracted triples
        """

    @abstractmethod
    def get_namespace_prefixes(self, parsed_result: Dict[str, Any]) -> Dict[str, str]:
        """
        Extract namespace prefixes from parsed data.

        Args:
            parsed_result (Dict[str, Any]): Parsed ontology data

        Returns:
            Dict[str, str]: Namespace prefix mappings
        """


class RDFlibTripleExtractor(TripleExtractor):
    """
    RDF triple extractor using rdflib.

    Extracts triples directly from rdflib.Graph objects with comprehensive
    type detection and metadata extraction.
    """

    def __init__(self, options: Optional[Dict[str, Any]] = None):
        """
        Initialize the rdflib-based triple extractor.

        Args:
            options (Optional[Dict[str, Any]]): Extraction options
        """
        self.options = options or {}
        self._rdflib = None
        self._rdflib_available = False

        # Try to import rdflib
        try:
            import rdflib

            self._rdflib = rdflib
            self._rdflib_available = True
        except ImportError:
            pass

    def extract_triples(self, parsed_result: Dict[str, Any]) -> List[RDFTriple]:
        """
        Extract RDF triples from rdflib graph.

        Args:
            parsed_result (Dict[str, Any]): Result from OWL parser containing rdf_graph

        Returns:
            List[RDFTriple]: Extracted triples

        Raises:
            OntologyException: If rdflib is not available or extraction fails
        """
        if not self._rdflib_available:
            from ..exceptions import OntologyException

            raise OntologyException(
                "rdflib not available for triple extraction",
                error_code="AIM2_ONTO_PARS_E_002",
            )

        rdf_graph = parsed_result.get("rdf_graph")
        if not rdf_graph:
            return []

        triples = []
        namespace_prefixes = self.get_namespace_prefixes(parsed_result)

        try:
            for subject, predicate, obj in rdf_graph:
                # Determine object type and extract metadata
                object_type = "uri"
                language = None
                datatype = None

                if isinstance(obj, self._rdflib.Literal):
                    object_type = "literal"
                    language = str(obj.language) if obj.language else None
                    datatype = str(obj.datatype) if obj.datatype else None
                elif isinstance(obj, self._rdflib.BNode):
                    object_type = "blank_node"

                triple = RDFTriple(
                    subject=str(subject),
                    predicate=str(predicate),
                    object=str(obj),
                    object_type=object_type,
                    language=language,
                    datatype=datatype,
                    source_graph=parsed_result.get("source_graph"),
                    confidence=1.0,  # rdflib extractions are considered reliable
                    extraction_method="rdflib",
                    namespace_prefixes=namespace_prefixes,
                    metadata={
                        "parsed_at": parsed_result.get("parsed_at"),
                        "source_format": parsed_result.get("format"),
                    },
                )

                # Only add valid triples
                if triple.is_valid():
                    triples.append(triple)

        except Exception as e:
            from ..exceptions import OntologyException

            raise OntologyException(
                f"Failed to extract triples from rdflib graph: {str(e)}",
                error_code="AIM2_ONTO_PARS_E_002",
                context={
                    "graph_size": len(rdf_graph) if rdf_graph else 0,
                    "extraction_method": "rdflib",
                },
            )

        return triples

    def get_namespace_prefixes(self, parsed_result: Dict[str, Any]) -> Dict[str, str]:
        """
        Extract namespace prefixes from rdflib graph.

        Args:
            parsed_result (Dict[str, Any]): Parsed ontology data

        Returns:
            Dict[str, str]: Namespace prefix mappings
        """
        if not self._rdflib_available:
            return {}

        rdf_graph = parsed_result.get("rdf_graph")
        if not rdf_graph:
            return {}

        # Extract namespace bindings from graph
        prefixes = {}
        try:
            for prefix, namespace in rdf_graph.namespaces():
                if prefix and namespace:
                    prefixes[str(prefix)] = str(namespace)
        except Exception:
            # If namespace extraction fails, return empty dict
            pass

        # Add common RDF/OWL prefixes if not present
        common_prefixes = {
            "rdf": "http://www.w3.org/1999/02/22-rdf-syntax-ns#",
            "rdfs": "http://www.w3.org/2000/01/rdf-schema#",
            "owl": "http://www.w3.org/2002/07/owl#",
            "xsd": "http://www.w3.org/2001/XMLSchema#",
        }

        for prefix, namespace in common_prefixes.items():
            if prefix not in prefixes:
                prefixes[prefix] = namespace

        return prefixes


class OwlReady2TripleExtractor(TripleExtractor):
    """
    RDF triple extractor using owlready2.

    Extracts triples from owlready2 ontology objects with OWL-specific
    knowledge and type inference.
    """

    def __init__(self, options: Optional[Dict[str, Any]] = None):
        """
        Initialize the owlready2-based triple extractor.

        Args:
            options (Optional[Dict[str, Any]]): Extraction options
        """
        self.options = options or {}
        self._owlready2 = None
        self._owlready2_available = False

        # Try to import owlready2
        try:
            import owlready2

            self._owlready2 = owlready2
            self._owlready2_available = True
        except ImportError:
            pass

    def extract_triples(self, parsed_result: Dict[str, Any]) -> List[RDFTriple]:
        """
        Extract RDF triples from owlready2 ontology.

        Args:
            parsed_result (Dict[str, Any]): Result from OWL parser containing owl_ontology

        Returns:
            List[RDFTriple]: Extracted triples

        Raises:
            OntologyException: If owlready2 is not available or extraction fails
        """
        if not self._owlready2_available:
            from ..exceptions import OntologyException

            raise OntologyException(
                "owlready2 not available for triple extraction",
                error_code="AIM2_ONTO_PARS_E_002",
            )

        owl_ontology = parsed_result.get("owl_ontology")
        if not owl_ontology:
            return []

        triples = []
        namespace_prefixes = self.get_namespace_prefixes(parsed_result)

        try:
            # Extract triples from owlready2 ontology
            # This is a simplified approach - full implementation would need
            # to handle all OWL constructs properly

            # Extract class hierarchy triples
            for cls in owl_ontology.classes():
                cls_uri = str(cls.iri) if hasattr(cls, "iri") else str(cls)

                # rdf:type owl:Class
                triples.append(
                    RDFTriple(
                        subject=cls_uri,
                        predicate="http://www.w3.org/1999/02/22-rdf-syntax-ns#type",
                        object="http://www.w3.org/2002/07/owl#Class",
                        object_type="uri",
                        extraction_method="owlready2",
                        namespace_prefixes=namespace_prefixes,
                        confidence=1.0,
                    )
                )

                # rdfs:subClassOf relationships
                for parent in cls.is_a:
                    parent_uri = (
                        str(parent.iri) if hasattr(parent, "iri") else str(parent)
                    )
                    triples.append(
                        RDFTriple(
                            subject=cls_uri,
                            predicate="http://www.w3.org/2000/01/rdf-schema#subClassOf",
                            object=parent_uri,
                            object_type="uri",
                            extraction_method="owlready2",
                            namespace_prefixes=namespace_prefixes,
                            confidence=1.0,
                        )
                    )

                # Labels and comments
                if hasattr(cls, "label") and cls.label:
                    for label in cls.label:
                        triples.append(
                            RDFTriple(
                                subject=cls_uri,
                                predicate="http://www.w3.org/2000/01/rdf-schema#label",
                                object=str(label),
                                object_type="literal",
                                extraction_method="owlready2",
                                namespace_prefixes=namespace_prefixes,
                                confidence=1.0,
                            )
                        )

            # Extract property triples
            for prop in owl_ontology.properties():
                prop_uri = str(prop.iri) if hasattr(prop, "iri") else str(prop)

                # Determine property type
                if isinstance(prop, self._owlready2.ObjectProperty):
                    prop_type = "http://www.w3.org/2002/07/owl#ObjectProperty"
                elif isinstance(prop, self._owlready2.DataProperty):
                    prop_type = "http://www.w3.org/2002/07/owl#DatatypeProperty"
                else:
                    prop_type = "http://www.w3.org/1999/02/22-rdf-syntax-ns#Property"

                triples.append(
                    RDFTriple(
                        subject=prop_uri,
                        predicate="http://www.w3.org/1999/02/22-rdf-syntax-ns#type",
                        object=prop_type,
                        object_type="uri",
                        extraction_method="owlready2",
                        namespace_prefixes=namespace_prefixes,
                        confidence=1.0,
                    )
                )

        except Exception as e:
            from ..exceptions import OntologyException

            raise OntologyException(
                f"Failed to extract triples from owlready2 ontology: {str(e)}",
                error_code="AIM2_ONTO_PARS_E_002",
                context={
                    "ontology_name": str(owl_ontology.name)
                    if hasattr(owl_ontology, "name")
                    else "unknown",
                    "extraction_method": "owlready2",
                },
            )

        return triples

    def get_namespace_prefixes(self, parsed_result: Dict[str, Any]) -> Dict[str, str]:
        """
        Extract namespace prefixes from owlready2 ontology.

        Args:
            parsed_result (Dict[str, Any]): Parsed ontology data

        Returns:
            Dict[str, str]: Namespace prefix mappings
        """
        if not self._owlready2_available:
            return {}

        # For owlready2, we'll use common prefixes and the ontology base IRI
        prefixes = {
            "rdf": "http://www.w3.org/1999/02/22-rdf-syntax-ns#",
            "rdfs": "http://www.w3.org/2000/01/rdf-schema#",
            "owl": "http://www.w3.org/2002/07/owl#",
            "xsd": "http://www.w3.org/2001/XMLSchema#",
        }

        owl_ontology = parsed_result.get("owl_ontology")
        if owl_ontology and hasattr(owl_ontology, "base_iri") and owl_ontology.base_iri:
            # Use the ontology base IRI as the default prefix
            prefixes[""] = str(owl_ontology.base_iri)

        return prefixes


# Integration with existing OWL parser - these methods would be added to the OWLParser class
def extract_triples_method_design():
    """
    Design for the extract_triples() method to be added to OWLParser class.

    This method would integrate with the existing parser architecture and
    follow the same patterns as extract_terms() and extract_relationships().
    """

    def extract_triples(self, parsed_result: Any, **kwargs) -> List[RDFTriple]:
        """
        Extract RDF triples from parsed OWL data.

        This method integrates with the existing OWL parser architecture and provides
        comprehensive RDF triple extraction using both rdflib and owlready2 strategies.

        Args:
            parsed_result (Any): Result from parse method (should be dict with rdf_graph/owl_ontology)
            **kwargs: Additional extraction options:
                - extractor_preference (str): Preferred extractor ('rdflib', 'owlready2', 'both')
                - include_inferred_triples (bool): Whether to include inferred triples
                - filter_predicates (Set[str]): Set of predicate URIs to include (None for all)
                - exclude_predicates (Set[str]): Set of predicate URIs to exclude
                - max_triples (int): Maximum number of triples to extract (None for all)
                - confidence_threshold (float): Minimum confidence score to include

        Returns:
            List[RDFTriple]: List of extracted RDF triples

        Raises:
            OntologyException: If extraction fails or invalid parameters provided

        Examples:
            >>> parser = OWLParser()
            >>> parsed_data = parser.parse(owl_content)
            >>> triples = parser.extract_triples(parsed_data)
            >>> print(f"Extracted {len(triples)} triples")

            >>> # Extract only specific predicate types
            >>> rdfs_triples = parser.extract_triples(
            ...     parsed_data,
            ...     filter_predicates={"http://www.w3.org/2000/01/rdf-schema#subClassOf"}
            ... )
        """
        if not isinstance(parsed_result, dict):
            self.logger.warning("Invalid parsed_result format for triple extraction")
            return []

        # Extract options
        extractor_preference = kwargs.get("extractor_preference", "both")
        kwargs.get("include_inferred_triples", False)
        filter_predicates = kwargs.get("filter_predicates")
        exclude_predicates = kwargs.get("exclude_predicates", set())
        max_triples = kwargs.get("max_triples")
        confidence_threshold = kwargs.get("confidence_threshold", 0.0)

        all_triples = []
        extraction_errors = []

        # Try rdflib extraction
        if extractor_preference in ("rdflib", "both") and self._rdflib_available:
            try:
                extractor = RDFlibTripleExtractor(self.options)
                rdflib_triples = extractor.extract_triples(parsed_result)
                all_triples.extend(rdflib_triples)
                self.logger.debug(
                    f"Extracted {len(rdflib_triples)} triples using rdflib"
                )
            except Exception as e:
                error_msg = f"rdflib triple extraction failed: {str(e)}"
                extraction_errors.append(error_msg)
                self.logger.warning(error_msg)

        # Try owlready2 extraction
        if extractor_preference in ("owlready2", "both") and self._owlready2_available:
            try:
                extractor = OwlReady2TripleExtractor(self.options)
                owlready2_triples = extractor.extract_triples(parsed_result)

                # Merge with existing triples, avoiding duplicates
                existing_triple_set = set(all_triples)
                for triple in owlready2_triples:
                    if triple not in existing_triple_set:
                        all_triples.append(triple)
                        existing_triple_set.add(triple)

                self.logger.debug(
                    f"Extracted {len(owlready2_triples)} triples using owlready2"
                )
            except Exception as e:
                error_msg = f"owlready2 triple extraction failed: {str(e)}"
                extraction_errors.append(error_msg)
                self.logger.warning(error_msg)

        # Apply filters
        filtered_triples = []
        for triple in all_triples:
            # Filter by predicate
            if filter_predicates and triple.predicate not in filter_predicates:
                continue
            if triple.predicate in exclude_predicates:
                continue

            # Filter by confidence
            if triple.confidence < confidence_threshold:
                continue

            filtered_triples.append(triple)

        # Apply limit
        if max_triples and len(filtered_triples) > max_triples:
            # Sort by confidence (descending) and take top N
            filtered_triples.sort(key=lambda t: t.confidence, reverse=True)
            filtered_triples = filtered_triples[:max_triples]

        # Log results
        self.logger.info(
            f"Triple extraction complete: {len(filtered_triples)} triples extracted "
            f"(filtered from {len(all_triples)} total)"
        )

        if extraction_errors and not filtered_triples:
            # If all extraction methods failed and no triples were extracted
            from ..exceptions import OntologyException

            raise OntologyException(
                "All triple extraction methods failed",
                error_code="AIM2_ONTO_PARS_E_002",
                context={
                    "extraction_errors": extraction_errors,
                    "available_extractors": {
                        "rdflib": self._rdflib_available,
                        "owlready2": self._owlready2_available,
                    },
                },
            )

        return filtered_triples


# Parse result integration design
def parse_result_integration_design():
    """
    Design for integrating triples into the existing parse result structure.

    The existing OWL parser returns a dictionary with the following structure:
    {
        "rdf_graph": rdflib.Graph,
        "owl_ontology": owlready2.Ontology,
        "format": str,
        "parsed_at": float,
        "content_size": int,
        "options_used": dict,
        "validation": dict  # if validate_on_parse is True
    }

    With triple extraction, the result would be enhanced to:
    {
        "rdf_graph": rdflib.Graph,
        "owl_ontology": owlready2.Ontology,
        "format": str,
        "parsed_at": float,
        "content_size": int,
        "options_used": dict,
        "validation": dict,  # if validate_on_parse is True
        "triples": List[RDFTriple],  # NEW: extracted triples
        "triple_extraction": {  # NEW: extraction metadata
            "total_triples": int,
            "extraction_methods": List[str],
            "extraction_time": float,
            "extraction_errors": List[str],
            "namespace_prefixes": Dict[str, str],
            "extraction_options": dict
        }
    }
    """


# Error handling design
def error_handling_design():
    """
    Design for error handling in triple extraction using existing AIM2Exception hierarchy.

    Uses existing error codes:
    - AIM2_ONTO_PARS_E_002: "RDF triple extraction failure"

    Error scenarios and handling:
    1. Library unavailable (rdflib/owlready2) - Warning logged, graceful degradation
    2. Invalid parsed_result format - Warning logged, return empty list
    3. Triple extraction failure - OntologyException with context
    4. Validation failure during extraction - Invalid triples filtered out
    5. Memory issues with large ontologies - Streaming/batch processing
    """


# Unit test structure design
def unit_test_structure_design():
    """
    Design for comprehensive unit test structure.

    Test modules:
    1. test_rdf_triple.py - RDFTriple class tests
       - test_creation_and_validation()
       - test_serialization()
       - test_equality_and_hashing()
       - test_compact_form_conversion()
       - test_ntriples_conversion()

    2. test_triple_extractors.py - Extractor tests
       - test_rdflib_extractor()
       - test_owlready2_extractor()
       - test_extraction_failure_handling()

    3. test_owl_parser_triple_integration.py - Integration tests
       - test_extract_triples_method()
       - test_parse_result_integration()
       - test_error_handling()
       - test_filtering_and_options()

    4. test_triple_extraction_performance.py - Performance tests
       - test_large_ontology_extraction()
       - test_memory_usage()
       - test_extraction_speed()
    """


if __name__ == "__main__":
    # Example usage demonstrating the design

    # Create a basic RDF triple
    triple1 = RDFTriple(
        subject="http://example.org/Person1",
        predicate="http://www.w3.org/1999/02/22-rdf-syntax-ns#type",
        object="http://example.org/Person",
        namespace_prefixes={
            "ex": "http://example.org/",
            "rdf": "http://www.w3.org/1999/02/22-rdf-syntax-ns#",
        },
    )

    print("=== RDF Triple Design Example ===")
    print(f"Triple: {triple1}")
    print(f"Compact form: {triple1.to_compact_form()}")
    print(f"N-Triples: {triple1.to_ntriples()}")
    print(f"Valid: {triple1.is_valid()}")
    print(f"JSON: {triple1.to_json(indent=2)}")

    # Create a literal triple
    triple2 = RDFTriple(
        subject="http://example.org/Person1",
        predicate="http://example.org/hasName",
        object="John Doe",
        object_type="literal",
        language="en",
        namespace_prefixes={"ex": "http://example.org/"},
    )

    print(f"\nLiteral triple: {triple2}")
    print(f"N-Triples: {triple2.to_ntriples()}")

    # Demonstrate validation
    invalid_triple = RDFTriple(
        subject="",  # Invalid empty subject
        predicate="invalid_predicate",  # Invalid predicate
        object="test",
        confidence=1.5,  # Invalid confidence > 1.0
    )

    print(f"\nInvalid triple valid: {invalid_triple.is_valid()}")
    print(f"Validation errors: {invalid_triple.get_validation_errors()}")
