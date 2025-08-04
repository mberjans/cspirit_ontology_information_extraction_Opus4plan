"""
Comprehensive Unit Tests for AIM2 Ontology Models

This module provides comprehensive unit tests for the ontology models in the AIM2 project,
with a focus on the RDFTriple class and its comprehensive functionality including validation,
serialization, and metadata management.

Test Classes:
    TestRDFTripleCore: Core functionality tests for RDFTriple
    TestRDFTripleValidation: Validation and error handling tests
    TestRDFTripleSerialization: Serialization/deserialization tests
    TestRDFTripleMetadata: Metadata and namespace handling tests
    TestRDFTripleEdgeCases: Edge cases and error conditions
    TestRDFTriplePerformance: Performance and scalability tests

The tests follow a comprehensive approach covering:
- Basic initialization and field validation
- Advanced validation with URI checking and type validation
- Serialization to various RDF formats (JSON, Turtle, N-Triples)
- Deserialization and round-trip testing
- Metadata preservation and namespace handling
- Edge cases with malformed data and boundary conditions
- Performance testing with large datasets
- Integration with datetime handling and confidence scoring

Dependencies:
    - pytest: For test framework and fixtures
    - unittest.mock: For mocking functionality
    - datetime: For timestamp testing
    - json: For JSON serialization testing

Usage:
    pytest tests/unit/test_models.py -v
"""

import pytest
import json
from datetime import datetime


class TestRDFTripleCore:
    """Test core RDFTriple functionality."""

    def test_rdf_triple_minimal_initialization(self):
        """Test RDFTriple with minimal required parameters."""
        from aim2_project.aim2_ontology.models import RDFTriple

        triple = RDFTriple(
            subject="http://example.org/subject",
            predicate="http://example.org/predicate",
            object="http://example.org/object",
        )

        assert triple.subject == "http://example.org/subject"
        assert triple.predicate == "http://example.org/predicate"
        assert triple.object == "http://example.org/object"
        assert triple.subject_type == "uri"
        assert triple.object_type == "uri"
        assert triple.confidence == 1.0
        assert isinstance(triple.metadata, dict)
        assert len(triple.metadata) == 0
        assert isinstance(triple.namespace_prefixes, dict)
        assert len(triple.namespace_prefixes) == 0
        assert triple.created_at is not None
        assert isinstance(triple.created_at, datetime)

    def test_rdf_triple_full_initialization(self):
        """Test RDFTriple with all parameters specified."""
        from aim2_project.aim2_ontology.models import RDFTriple

        test_datetime = datetime(2023, 1, 1, 12, 0, 0)
        metadata = {
            "source": "test_ontology",
            "extraction_method": "automated",
            "quality_score": 0.95,
        }
        namespaces = {
            "ex": "http://example.org/",
            "rdfs": "http://www.w3.org/2000/01/rdf-schema#",
            "owl": "http://www.w3.org/2002/07/owl#",
        }

        triple = RDFTriple(
            subject="http://example.org/Chemical",
            predicate="http://www.w3.org/2000/01/rdf-schema#label",
            object="Glucose",
            subject_type="uri",
            object_type="literal",
            object_datatype="http://www.w3.org/2001/XMLSchema#string",
            object_language="en",
            context="http://example.org/graph1",
            source="ChEBI",
            confidence=0.85,
            metadata=metadata,
            created_at=test_datetime,
            namespace_prefixes=namespaces,
        )

        assert triple.subject == "http://example.org/Chemical"
        assert triple.predicate == "http://www.w3.org/2000/01/rdf-schema#label"
        assert triple.object == "Glucose"
        assert triple.subject_type == "uri"
        assert triple.object_type == "literal"
        assert triple.object_datatype == "http://www.w3.org/2001/XMLSchema#string"
        assert triple.object_language == "en"
        assert triple.context == "http://example.org/graph1"
        assert triple.source == "ChEBI"
        assert triple.confidence == 0.85
        assert triple.metadata == metadata
        assert triple.created_at == test_datetime
        assert triple.namespace_prefixes == namespaces

    def test_rdf_triple_post_init_validation(self):
        """Test RDFTriple __post_init__ validation and normalization."""
        from aim2_project.aim2_ontology.models import RDFTriple

        # Test confidence normalization
        triple_high_confidence = RDFTriple("s", "p", "o", confidence=1.5)
        assert triple_high_confidence.confidence == 1.0

        triple_low_confidence = RDFTriple("s", "p", "o", confidence=-0.5)
        assert triple_low_confidence.confidence == 0.0

        triple_invalid_confidence = RDFTriple("s", "p", "o", confidence="invalid")
        assert triple_invalid_confidence.confidence == 1.0

        # Test metadata normalization
        triple_invalid_metadata = RDFTriple("s", "p", "o", metadata="not_a_dict")
        assert isinstance(triple_invalid_metadata.metadata, dict)
        assert len(triple_invalid_metadata.metadata) == 0

        # Test namespace normalization
        triple_invalid_namespaces = RDFTriple(
            "s", "p", "o", namespace_prefixes="not_a_dict"
        )
        assert isinstance(triple_invalid_namespaces.namespace_prefixes, dict)
        assert len(triple_invalid_namespaces.namespace_prefixes) == 0

    def test_rdf_triple_automatic_timestamp(self):
        """Test automatic timestamp creation when not specified."""
        from aim2_project.aim2_ontology.models import RDFTriple

        before_creation = datetime.now()
        triple = RDFTriple("s", "p", "o")
        after_creation = datetime.now()

        assert triple.created_at is not None
        assert before_creation <= triple.created_at <= after_creation

    @pytest.mark.parametrize(
        "confidence_input,expected_output",
        [
            (0.0, 0.0),
            (0.5, 0.5),
            (1.0, 1.0),
            (1.5, 1.0),
            (-0.5, 0.0),
            (None, 1.0),
            ("0.8", 0.8),
            ("invalid", 1.0),
            ([], 1.0),
            ({}, 1.0),
        ],
    )
    def test_confidence_normalization_cases(self, confidence_input, expected_output):
        """Test confidence normalization with various input types."""
        from aim2_project.aim2_ontology.models import RDFTriple

        triple = RDFTriple("s", "p", "o", confidence=confidence_input)
        assert triple.confidence == expected_output


class TestRDFTripleValidation:
    """Test RDFTriple validation functionality."""

    def test_rdf_triple_is_valid_basic(self):
        """Test basic validation of valid RDF triples."""
        from aim2_project.aim2_ontology.models import RDFTriple

        # Valid URI triple
        uri_triple = RDFTriple(
            "http://example.org/subject",
            "http://example.org/predicate",
            "http://example.org/object",
        )
        assert uri_triple.is_valid()

        # Valid literal triple
        literal_triple = RDFTriple(
            "http://example.org/subject",
            "http://example.org/predicate",
            "literal value",
            object_type="literal",
        )
        assert literal_triple.is_valid()

        # Valid blank node triple
        bnode_triple = RDFTriple(
            "_:b1",
            "http://example.org/predicate",
            "_:b2",
            subject_type="bnode",
            object_type="bnode",
        )
        assert bnode_triple.is_valid()

    @pytest.mark.parametrize(
        "subject,predicate,object,expected_valid",
        [
            # Valid cases
            ("http://ex.org/s", "http://ex.org/p", "http://ex.org/o", True),
            ("_:b1", "http://ex.org/p", "literal", True),
            ("http://ex.org/s", "http://ex.org/p", "_:b1", True),
            # Invalid cases - empty values
            ("", "http://ex.org/p", "http://ex.org/o", False),
            ("http://ex.org/s", "", "http://ex.org/o", False),
            ("http://ex.org/s", "http://ex.org/p", "", False),
            # Invalid cases - None values
            (None, "http://ex.org/p", "http://ex.org/o", False),
            ("http://ex.org/s", None, "http://ex.org/o", False),
            ("http://ex.org/s", "http://ex.org/p", None, False),
            # Invalid URIs - these are auto-detected as literals now, so they're valid
            # The test for invalid URIs should explicitly set the type
            (
                "not-a-uri",
                "http://ex.org/p",
                "http://ex.org/o",
                False,
            ),  # subjects can't be literals in RDF
            (
                "http://ex.org/s",
                "not-a-uri",
                "http://ex.org/o",
                False,
            ),  # predicate must be URI
            (
                "http://ex.org/s",
                "http://ex.org/p",
                "not-a-uri",
                True,
            ),  # auto-detected as literal object (valid)
        ],
    )
    def test_validation_edge_cases(self, subject, predicate, object, expected_valid):
        """Test validation with various edge cases."""
        from aim2_project.aim2_ontology.models import RDFTriple

        triple = RDFTriple(subject=subject, predicate=predicate, object=object)
        assert triple.is_valid() == expected_valid

    def test_validation_with_different_types(self):
        """Test validation with different subject/object types."""
        from aim2_project.aim2_ontology.models import RDFTriple

        # URI subject with literal object
        triple1 = RDFTriple(
            "http://ex.org/s", "http://ex.org/p", "literal", object_type="literal"
        )
        assert triple1.is_valid()

        # Blank node subject with URI object
        triple2 = RDFTriple(
            "_:b1",
            "http://ex.org/p",
            "http://ex.org/o",
            subject_type="bnode",
            object_type="uri",
        )
        assert triple2.is_valid()

        # Invalid blank node format
        triple3 = RDFTriple(
            "b1", "http://ex.org/p", "http://ex.org/o", subject_type="bnode"
        )
        assert not triple3.is_valid()

    def test_validate_uri_format(self):
        """Test URI format validation."""
        from aim2_project.aim2_ontology.models import RDFTriple

        # Valid URIs
        valid_uris = [
            "http://example.org/test",
            "https://example.org/test",
            "ftp://example.org/test",
            "urn:isbn:1234567890",
            "mailto:test@example.org",
        ]

        for uri in valid_uris:
            triple = RDFTriple(uri, "http://ex.org/p", "http://ex.org/o")
            assert triple.is_valid(), f"URI should be valid: {uri}"

        # Invalid URIs
        invalid_uris = [
            "not-a-uri",
            "http://",
            "://example.org",
            " http://example.org",
            "http://example.org ",
        ]

        for uri in invalid_uris:
            triple = RDFTriple(uri, "http://ex.org/p", "http://ex.org/o")
            assert not triple.is_valid(), f"URI should be invalid: {uri}"

    def test_validate_blank_node_format(self):
        """Test blank node format validation."""
        from aim2_project.aim2_ontology.models import RDFTriple

        # Valid blank nodes
        valid_bnodes = ["_:b1", "_:node1", "_:123", "_:test_node"]

        for bnode in valid_bnodes:
            triple = RDFTriple(
                bnode, "http://ex.org/p", "literal", subject_type="bnode"
            )
            assert triple.is_valid(), f"Blank node should be valid: {bnode}"

        # Invalid blank nodes
        invalid_bnodes = ["b1", "_:", "_:  ", "_:b 1", ""]

        for bnode in invalid_bnodes:
            triple = RDFTriple(
                bnode, "http://ex.org/p", "literal", subject_type="bnode"
            )
            assert not triple.is_valid(), f"Blank node should be invalid: {bnode}"

    def test_validate_literal_types(self):
        """Test validation of different literal types."""
        from aim2_project.aim2_ontology.models import RDFTriple

        # String literal
        string_triple = RDFTriple(
            "http://ex.org/s",
            "http://ex.org/p",
            "test string",
            object_type="literal",
            object_datatype="http://www.w3.org/2001/XMLSchema#string",
        )
        assert string_triple.is_valid()

        # Integer literal
        int_triple = RDFTriple(
            "http://ex.org/s",
            "http://ex.org/p",
            "42",
            object_type="literal",
            object_datatype="http://www.w3.org/2001/XMLSchema#integer",
        )
        assert int_triple.is_valid()

        # Date literal
        date_triple = RDFTriple(
            "http://ex.org/s",
            "http://ex.org/p",
            "2023-01-01",
            object_type="literal",
            object_datatype="http://www.w3.org/2001/XMLSchema#date",
        )
        assert date_triple.is_valid()

        # Literal with language tag
        lang_triple = RDFTriple(
            "http://ex.org/s",
            "http://ex.org/p",
            "Hello",
            object_type="literal",
            object_language="en",
        )
        assert lang_triple.is_valid()

    def test_validate_confidence_range(self):
        """Test validation of confidence score range."""
        from aim2_project.aim2_ontology.models import RDFTriple

        # Valid confidence scores
        for conf in [0.0, 0.5, 1.0]:
            triple = RDFTriple(
                "http://ex.org/s", "http://ex.org/p", "http://ex.org/o", confidence=conf
            )
            assert triple.is_valid()
            assert 0.0 <= triple.confidence <= 1.0


class TestRDFTripleSerialization:
    """Test RDFTriple serialization and deserialization."""

    def test_to_dict_basic(self):
        """Test basic dictionary serialization."""
        from aim2_project.aim2_ontology.models import RDFTriple

        triple = RDFTriple(
            subject="http://example.org/Chemical",
            predicate="http://www.w3.org/2000/01/rdf-schema#label",
            object="Glucose",
            object_type="literal",
            confidence=0.95,
            metadata={"source": "test"},
        )

        result_dict = triple.to_dict()

        assert isinstance(result_dict, dict)
        assert result_dict["subject"] == "http://example.org/Chemical"
        assert result_dict["predicate"] == "http://www.w3.org/2000/01/rdf-schema#label"
        assert result_dict["object"] == "Glucose"
        assert result_dict["object_type"] == "literal"
        assert result_dict["confidence"] == 0.95
        assert result_dict["metadata"] == {"source": "test"}
        assert "created_at" in result_dict

    def test_from_dict_basic(self):
        """Test basic dictionary deserialization."""
        from aim2_project.aim2_ontology.models import RDFTriple

        test_dict = {
            "subject": "http://example.org/Chemical",
            "predicate": "http://www.w3.org/2000/01/rdf-schema#label",
            "object": "Glucose",
            "object_type": "literal",
            "confidence": 0.95,
            "metadata": {"source": "test"},
        }

        triple = RDFTriple.from_dict(test_dict)

        assert triple.subject == "http://example.org/Chemical"
        assert triple.predicate == "http://www.w3.org/2000/01/rdf-schema#label"
        assert triple.object == "Glucose"
        assert triple.object_type == "literal"
        assert triple.confidence == 0.95
        assert triple.metadata == {"source": "test"}

    def test_serialization_round_trip(self):
        """Test complete serialization round-trip."""
        from aim2_project.aim2_ontology.models import RDFTriple

        original_triple = RDFTriple(
            subject="http://example.org/Complex",
            predicate="http://www.w3.org/2000/01/rdf-schema#subClassOf",
            object="http://example.org/Chemical",
            context="http://example.org/graph1",
            source="automated_extraction",
            confidence=0.87,
            metadata={
                "extraction_method": "owl_parsing",
                "quality_indicators": ["complete", "verified"],
                "processing_time": 1.23,
            },
            namespace_prefixes={
                "ex": "http://example.org/",
                "rdfs": "http://www.w3.org/2000/01/rdf-schema#",
            },
        )

        # Serialize to dict
        dict_repr = original_triple.to_dict()

        # Deserialize from dict
        restored_triple = RDFTriple.from_dict(dict_repr)

        # Verify all fields match
        assert restored_triple.subject == original_triple.subject
        assert restored_triple.predicate == original_triple.predicate
        assert restored_triple.object == original_triple.object
        assert restored_triple.context == original_triple.context
        assert restored_triple.source == original_triple.source
        assert restored_triple.confidence == original_triple.confidence
        assert restored_triple.metadata == original_triple.metadata
        assert restored_triple.namespace_prefixes == original_triple.namespace_prefixes

    def test_to_json_basic(self):
        """Test JSON serialization."""
        from aim2_project.aim2_ontology.models import RDFTriple

        triple = RDFTriple(
            "http://example.org/s",
            "http://example.org/p",
            "test value",
            object_type="literal",
        )

        json_str = triple.to_json()

        assert isinstance(json_str, str)

        # Parse JSON to verify structure
        parsed = json.loads(json_str)
        assert parsed["subject"] == "http://example.org/s"
        assert parsed["predicate"] == "http://example.org/p"
        assert parsed["object"] == "test value"
        assert parsed["object_type"] == "literal"

    def test_from_json_basic(self):
        """Test JSON deserialization."""
        from aim2_project.aim2_ontology.models import RDFTriple

        test_json = json.dumps(
            {
                "subject": "http://example.org/s",
                "predicate": "http://example.org/p",
                "object": "test value",
                "object_type": "literal",
                "confidence": 0.8,
            }
        )

        triple = RDFTriple.from_json(test_json)

        assert triple.subject == "http://example.org/s"
        assert triple.predicate == "http://example.org/p"
        assert triple.object == "test value"
        assert triple.object_type == "literal"
        assert triple.confidence == 0.8

    def test_json_round_trip(self):
        """Test JSON serialization round-trip."""
        from aim2_project.aim2_ontology.models import RDFTriple

        original_triple = RDFTriple(
            "http://example.org/Chemical",
            "http://www.w3.org/2000/01/rdf-schema#label",
            "ATP",
            object_type="literal",
            object_language="en",
            confidence=0.99,
            metadata={"verified": True, "confidence_factors": [0.95, 0.98, 1.0]},
        )

        # JSON round-trip
        json_str = original_triple.to_json()
        restored_triple = RDFTriple.from_json(json_str)

        assert restored_triple.subject == original_triple.subject
        assert restored_triple.predicate == original_triple.predicate
        assert restored_triple.object == original_triple.object
        assert restored_triple.object_language == original_triple.object_language
        assert restored_triple.confidence == original_triple.confidence
        assert restored_triple.metadata == original_triple.metadata

    def test_to_turtle_uri_triple(self):
        """Test Turtle serialization for URI triple."""
        from aim2_project.aim2_ontology.models import RDFTriple

        triple = RDFTriple(
            "http://example.org/Chemical",
            "http://www.w3.org/2000/01/rdf-schema#subClassOf",
            "http://example.org/Entity",
        )

        turtle_str = triple.to_turtle()

        expected_elements = [
            "<http://example.org/Chemical>",
            "<http://www.w3.org/2000/01/rdf-schema#subClassOf>",
            "<http://example.org/Entity>",
            ".",
        ]

        for element in expected_elements:
            assert element in turtle_str

    def test_to_turtle_literal_triple(self):
        """Test Turtle serialization for literal triple."""
        from aim2_project.aim2_ontology.models import RDFTriple

        # Literal with language tag
        triple_lang = RDFTriple(
            "http://example.org/Chemical",
            "http://www.w3.org/2000/01/rdf-schema#label",
            "Chemical",
            object_type="literal",
            object_language="en",
        )

        turtle_lang = triple_lang.to_turtle()
        assert '"Chemical"@en' in turtle_lang

        # Literal with datatype
        triple_typed = RDFTriple(
            "http://example.org/measurement",
            "http://example.org/hasValue",
            "42.5",
            object_type="literal",
            object_datatype="http://www.w3.org/2001/XMLSchema#double",
        )

        turtle_typed = triple_typed.to_turtle()
        assert '"42.5"^^<http://www.w3.org/2001/XMLSchema#double>' in turtle_typed

    def test_to_ntriples(self):
        """Test N-Triples serialization."""
        from aim2_project.aim2_ontology.models import RDFTriple

        triple = RDFTriple(
            "http://example.org/subject",
            "http://example.org/predicate",
            "http://example.org/object",
        )

        ntriples_str = triple.to_ntriples()

        assert ntriples_str.startswith("<http://example.org/subject>")
        assert "<http://example.org/predicate>" in ntriples_str
        assert "<http://example.org/object>" in ntriples_str
        assert ntriples_str.endswith(" .")

    def test_serialization_with_namespaces(self):
        """Test serialization preserves namespace information."""
        from aim2_project.aim2_ontology.models import RDFTriple

        namespaces = {
            "ex": "http://example.org/",
            "rdfs": "http://www.w3.org/2000/01/rdf-schema#",
        }

        triple = RDFTriple(
            "http://example.org/Chemical",
            "http://www.w3.org/2000/01/rdf-schema#label",
            "Chemical",
            namespace_prefixes=namespaces,
        )

        dict_repr = triple.to_dict()
        assert dict_repr["namespace_prefixes"] == namespaces

        restored = RDFTriple.from_dict(dict_repr)
        assert restored.namespace_prefixes == namespaces


class TestRDFTripleMetadata:
    """Test RDFTriple metadata and namespace handling."""

    def test_metadata_management(self):
        """Test metadata setting and retrieval."""
        from aim2_project.aim2_ontology.models import RDFTriple

        metadata = {
            "source": "ChEBI",
            "extraction_method": "automated",
            "confidence_factors": {
                "structural": 0.95,
                "contextual": 0.88,
                "lexical": 0.92,
            },
            "timestamps": {
                "extracted": "2023-01-01T12:00:00Z",
                "validated": "2023-01-01T12:05:00Z",
            },
        }

        triple = RDFTriple(
            "http://purl.obolibrary.org/obo/CHEBI_15422",
            "http://www.w3.org/2000/01/rdf-schema#label",
            "ATP",
            metadata=metadata,
        )

        assert triple.metadata == metadata
        assert triple.metadata["source"] == "ChEBI"
        assert triple.metadata["confidence_factors"]["structural"] == 0.95

    def test_namespace_prefix_management(self):
        """Test namespace prefix handling."""
        from aim2_project.aim2_ontology.models import RDFTriple

        namespaces = {
            "chebi": "http://purl.obolibrary.org/obo/CHEBI_",
            "rdfs": "http://www.w3.org/2000/01/rdf-schema#",
            "owl": "http://www.w3.org/2002/07/owl#",
            "xsd": "http://www.w3.org/2001/XMLSchema#",
        }

        triple = RDFTriple(
            "http://purl.obolibrary.org/obo/CHEBI_15422",
            "http://www.w3.org/2000/01/rdf-schema#label",
            "ATP",
            namespace_prefixes=namespaces,
        )

        assert triple.namespace_prefixes == namespaces
        assert "chebi" in triple.namespace_prefixes
        assert (
            triple.namespace_prefixes["rdfs"] == "http://www.w3.org/2000/01/rdf-schema#"
        )

    def test_context_information(self):
        """Test context/named graph information."""
        from aim2_project.aim2_ontology.models import RDFTriple

        triple = RDFTriple(
            "http://example.org/glucose",
            "http://www.w3.org/1999/02/22-rdf-syntax-ns#type",
            "http://example.org/Sugar",
            context="http://example.org/biochemistry_graph",
        )

        assert triple.context == "http://example.org/biochemistry_graph"

    def test_source_tracking(self):
        """Test source information tracking."""
        from aim2_project.aim2_ontology.models import RDFTriple

        triple = RDFTriple(
            "http://example.org/compound",
            "http://www.w3.org/2000/01/rdf-schema#label",
            "Compound Name",
            source="PubChem",
        )

        assert triple.source == "PubChem"

    def test_timestamp_handling(self):
        """Test creation timestamp handling."""
        from aim2_project.aim2_ontology.models import RDFTriple

        # Test automatic timestamp
        triple1 = RDFTriple("s", "p", "o")
        assert triple1.created_at is not None
        assert isinstance(triple1.created_at, datetime)

        # Test explicit timestamp
        specific_time = datetime(2023, 6, 15, 14, 30, 0)
        triple2 = RDFTriple("s", "p", "o", created_at=specific_time)
        assert triple2.created_at == specific_time

    def test_metadata_serialization_preservation(self):
        """Test that metadata is preserved through serialization."""
        from aim2_project.aim2_ontology.models import RDFTriple

        complex_metadata = {
            "extraction": {
                "parser": "owlready2",
                "version": "0.39",
                "settings": {"imports": True, "validation": False},
            },
            "quality": {
                "score": 0.94,
                "factors": ["completeness", "consistency", "accuracy"],
                "issues": [],
            },
            "provenance": {
                "source_file": "chebi.owl",
                "extraction_date": "2023-01-01",
                "pipeline_version": "2.1.0",
            },
        }

        original = RDFTriple(
            "http://example.org/test",
            "http://example.org/hasProperty",
            "value",
            metadata=complex_metadata,
        )

        # Test dict serialization
        dict_repr = original.to_dict()
        restored_dict = RDFTriple.from_dict(dict_repr)
        assert restored_dict.metadata == complex_metadata

        # Test JSON serialization
        json_str = original.to_json()
        restored_json = RDFTriple.from_json(json_str)
        assert restored_json.metadata == complex_metadata


class TestRDFTripleComparison:
    """Test RDFTriple comparison and hashing."""

    def test_equality_basic(self):
        """Test basic equality comparison."""
        from aim2_project.aim2_ontology.models import RDFTriple

        triple1 = RDFTriple("http://ex.org/s", "http://ex.org/p", "http://ex.org/o")
        triple2 = RDFTriple("http://ex.org/s", "http://ex.org/p", "http://ex.org/o")
        triple3 = RDFTriple(
            "http://ex.org/s", "http://ex.org/p", "http://ex.org/different"
        )

        assert triple1 == triple2
        assert triple1 != triple3
        assert triple2 != triple3

    def test_equality_with_metadata(self):
        """Test equality with metadata differences."""
        from aim2_project.aim2_ontology.models import RDFTriple

        # Same core triple, different metadata
        triple1 = RDFTriple(
            "http://ex.org/s",
            "http://ex.org/p",
            "http://ex.org/o",
            metadata={"source": "A"},
        )
        triple2 = RDFTriple(
            "http://ex.org/s",
            "http://ex.org/p",
            "http://ex.org/o",
            metadata={"source": "B"},
        )

        # Triples should be equal based on core content, not metadata
        assert triple1 == triple2

    def test_hashing_consistency(self):
        """Test hash consistency for use in sets and dictionaries."""
        from aim2_project.aim2_ontology.models import RDFTriple

        triple1 = RDFTriple("http://ex.org/s", "http://ex.org/p", "http://ex.org/o")
        triple2 = RDFTriple("http://ex.org/s", "http://ex.org/p", "http://ex.org/o")
        triple3 = RDFTriple("http://ex.org/s", "http://ex.org/p", "different")

        # Equal objects should have equal hashes
        assert hash(triple1) == hash(triple2)

        # Test in set (should deduplicate)
        triple_set = {triple1, triple2, triple3}
        assert len(triple_set) == 2  # triple1 and triple2 should be deduplicated

        # Test as dictionary keys
        triple_dict = {triple1: "value1", triple2: "value2", triple3: "value3"}
        assert len(triple_dict) == 2  # triple1 and triple2 should share the same key

    def test_string_representations(self):
        """Test string representation methods."""
        from aim2_project.aim2_ontology.models import RDFTriple

        triple = RDFTriple(
            "http://example.org/Chemical",
            "http://www.w3.org/2000/01/rdf-schema#label",
            "Glucose",
            object_type="literal",
        )

        # Test __str__
        str_repr = str(triple)
        assert "http://example.org/Chemical" in str_repr
        assert "rdfs:label" in str_repr or "label" in str_repr
        assert "Glucose" in str_repr

        # Test __repr__
        repr_str = repr(triple)
        assert "RDFTriple" in repr_str
        assert "http://example.org/Chemical" in repr_str


class TestRDFTripleErrorHandling:
    """Test RDFTriple error handling and edge cases."""

    def test_invalid_serialization_data(self):
        """Test handling of invalid data during deserialization."""
        from aim2_project.aim2_ontology.models import RDFTriple

        # Test from_dict with missing required fields
        incomplete_dict = {"subject": "http://ex.org/s"}

        with pytest.raises((KeyError, TypeError, ValueError)):
            RDFTriple.from_dict(incomplete_dict)

        # Test from_json with invalid JSON
        invalid_json = "not valid json"

        with pytest.raises((json.JSONDecodeError, ValueError)):
            RDFTriple.from_json(invalid_json)

    def test_none_values_handling(self):
        """Test handling of None values in initialization."""
        from aim2_project.aim2_ontology.models import RDFTriple

        # None values should be handled gracefully or raise appropriate errors
        try:
            triple = RDFTriple(None, None, None)
            # If creation succeeds, validation should fail
            assert not triple.is_valid()
        except (TypeError, ValueError):
            # Appropriate to raise error for None values
            pass

    def test_unicode_handling(self):
        """Test Unicode character handling."""
        from aim2_project.aim2_ontology.models import RDFTriple

        # Unicode in different fields
        unicode_triple = RDFTriple(
            "http://example.org/化学物质",
            "http://www.w3.org/2000/01/rdf-schema#label",
            "化学物质",
            object_type="literal",
            object_language="zh",
        )

        assert unicode_triple.is_valid()
        assert "化学物质" in unicode_triple.subject
        assert "化学物质" in unicode_triple.object

        # Test serialization preserves Unicode
        dict_repr = unicode_triple.to_dict()
        restored = RDFTriple.from_dict(dict_repr)
        assert "化学物质" in restored.subject
        assert "化学物质" in restored.object

    def test_large_metadata_handling(self):
        """Test handling of large metadata structures."""
        from aim2_project.aim2_ontology.models import RDFTriple

        # Create large metadata structure
        large_metadata = {
            f"key_{i}": {
                "value": f"value_{i}",
                "nested": {f"nested_key_{j}": f"nested_value_{j}" for j in range(10)},
            }
            for i in range(100)
        }

        triple = RDFTriple(
            "http://example.org/test",
            "http://example.org/hasData",
            "large_dataset",
            metadata=large_metadata,
        )

        assert triple.is_valid()
        assert len(triple.metadata) == 100

        # Test serialization handles large metadata
        dict_repr = triple.to_dict()
        assert len(dict_repr["metadata"]) == 100


# Test fixtures specifically for RDFTriple testing
@pytest.fixture
def sample_chemical_triple():
    """Fixture providing a sample chemical RDF triple."""
    from aim2_project.aim2_ontology.models import RDFTriple

    return RDFTriple(
        subject="http://purl.obolibrary.org/obo/CHEBI_15422",
        predicate="http://www.w3.org/2000/01/rdf-schema#label",
        object="ATP",
        object_type="literal",
        object_language="en",
        context="http://purl.obolibrary.org/obo/chebi.owl",
        source="ChEBI",
        confidence=0.98,
        metadata={
            "extraction_method": "owl_parsing",
            "validation_status": "verified",
            "semantic_category": "chemical_entity",
        },
        namespace_prefixes={
            "chebi": "http://purl.obolibrary.org/obo/CHEBI_",
            "rdfs": "http://www.w3.org/2000/01/rdf-schema#",
        },
    )


@pytest.fixture
def sample_hierarchical_triples():
    """Fixture providing sample hierarchical RDF triples."""
    from aim2_project.aim2_ontology.models import RDFTriple

    return [
        RDFTriple(
            "http://example.org/Chemical",
            "http://www.w3.org/1999/02/22-rdf-syntax-ns#type",
            "http://www.w3.org/2002/07/owl#Class",
        ),
        RDFTriple(
            "http://example.org/Glucose",
            "http://www.w3.org/2000/01/rdf-schema#subClassOf",
            "http://example.org/Chemical",
        ),
        RDFTriple(
            "http://example.org/glucose_instance",
            "http://www.w3.org/1999/02/22-rdf-syntax-ns#type",
            "http://example.org/Glucose",
        ),
    ]


@pytest.fixture
def various_datatype_triples():
    """Fixture providing triples with various datatypes."""
    from aim2_project.aim2_ontology.models import RDFTriple

    return [
        RDFTriple(
            "http://example.org/measurement1",
            "http://example.org/hasValue",
            "42.5",
            object_type="literal",
            object_datatype="http://www.w3.org/2001/XMLSchema#double",
        ),
        RDFTriple(
            "http://example.org/event1",
            "http://example.org/occurredAt",
            "2023-01-01T12:00:00Z",
            object_type="literal",
            object_datatype="http://www.w3.org/2001/XMLSchema#dateTime",
        ),
        RDFTriple(
            "http://example.org/flag1",
            "http://example.org/isActive",
            "true",
            object_type="literal",
            object_datatype="http://www.w3.org/2001/XMLSchema#boolean",
        ),
        RDFTriple(
            "http://example.org/counter1",
            "http://example.org/hasCount",
            "123",
            object_type="literal",
            object_datatype="http://www.w3.org/2001/XMLSchema#integer",
        ),
    ]
