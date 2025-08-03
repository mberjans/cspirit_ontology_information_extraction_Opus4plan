"""
Comprehensive Unit Tests for JSON-LD Parser (TDD Approach)

This module provides comprehensive unit tests for the JSON-LD parser functionality in the AIM2 ontology
information extraction system. The tests follow test-driven development (TDD) approach,
defining the expected behavior of the JSON-LD parser classes before implementation.

Test Classes:
    TestJSONLDParserCreation: Tests for JSONLDParser instantiation and configuration
    TestJSONLDParserFormatSupport: Tests for JSON-LD format detection and validation
    TestJSONLDParserParsing: Tests for core JSON-LD parsing functionality
    TestJSONLDParserContextHandling: Tests for @context processing
    TestJSONLDParserConversion: Tests for converting parsed JSON-LD to internal models
    TestJSONLDParserErrorHandling: Tests for error handling and validation
    TestJSONLDParserValidation: Tests for JSON-LD validation functionality
    TestJSONLDParserOptions: Tests for parser configuration and options
    TestJSONLDParserIntegration: Integration tests with other components
    TestJSONLDParserPerformance: Performance and scalability tests

The JSONLDParser is expected to be a concrete implementation providing:
- JSON-LD format support: Compact, expanded, flattened forms
- @context processing and namespace resolution
- @id, @type, @graph structure handling
- RDF vocabulary expansion and compaction
- Integration with JSON-LD processing libraries (pyld, rdflib-jsonld)
- Conversion to internal Term/Relationship/Ontology models
- Comprehensive validation and error reporting
- Performance optimization for large JSON-LD documents
- Configurable parsing options and context management

Key JSON-LD concepts tested:
- @context processing and inheritance
- @id and @type handling for node identification
- @graph structures for multiple graphs
- Nested objects and arrays
- RDF vocabulary expansion and compaction
- Frame-based queries and filtering
- Namespace resolution and aliasing
- Blank node handling

Dependencies:
    - pytest: For test framework
    - unittest.mock: For mocking functionality
    - json: For JSON processing (mocked)
    - typing: For type hints
    - pyld: For JSON-LD processing (mocked)
    - rdflib: For RDF operations (mocked)

Usage:
    pytest tests/unit/test_jsonld_parser.py -v
"""

import pytest
from unittest.mock import Mock, patch
from pathlib import Path
import tempfile
import io
import json


# Mock the parser classes since they don't exist yet (TDD approach)
@pytest.fixture
def mock_jsonld_parser():
    """Mock JSONLDParser class."""
    with patch("aim2_project.aim2_ontology.parsers.JSONLDParser") as mock_parser:
        mock_instance = Mock()

        # Core parsing methods
        mock_instance.parse = Mock()
        mock_instance.parse_file = Mock()
        mock_instance.parse_string = Mock()
        mock_instance.parse_stream = Mock()

        # Format detection and validation
        mock_instance.detect_format = Mock()
        mock_instance.detect_encoding = Mock()
        mock_instance.validate_format = Mock()
        mock_instance.get_supported_formats = Mock(
            return_value=["jsonld", "json", "json-ld"]
        )

        # JSON-LD specific methods
        mock_instance.expand = Mock()
        mock_instance.compact = Mock()
        mock_instance.flatten = Mock()
        mock_instance.frame = Mock()
        mock_instance.normalize = Mock()

        # Context handling
        mock_instance.resolve_context = Mock()
        mock_instance.set_context = Mock()
        mock_instance.get_context = Mock()
        mock_instance.merge_contexts = Mock()
        mock_instance.validate_context = Mock()

        # RDF operations
        mock_instance.to_rdf = Mock()
        mock_instance.from_rdf = Mock()
        mock_instance.get_namespaces = Mock()
        mock_instance.expand_namespaces = Mock()

        # Graph operations
        mock_instance.extract_graphs = Mock()
        mock_instance.merge_graphs = Mock()
        mock_instance.filter_graph = Mock()

        # Node operations
        mock_instance.get_nodes = Mock()
        mock_instance.get_node_by_id = Mock()
        mock_instance.get_nodes_by_type = Mock()
        mock_instance.resolve_references = Mock()

        # Conversion methods
        mock_instance.to_ontology = Mock()
        mock_instance.extract_terms = Mock()
        mock_instance.extract_relationships = Mock()
        mock_instance.extract_metadata = Mock()

        # Configuration
        mock_instance.set_options = Mock()
        mock_instance.get_options = Mock(return_value={})
        mock_instance.reset_options = Mock()

        # Validation
        mock_instance.validate = Mock()
        mock_instance.validate_jsonld = Mock()
        mock_instance.get_validation_errors = Mock(return_value=[])

        mock_parser.return_value = mock_instance
        yield mock_parser


@pytest.fixture
def mock_pyld():
    """Mock pyld library."""
    with patch("pyld.jsonld") as mock_jsonld:
        mock_jsonld.expand = Mock()
        mock_jsonld.compact = Mock()
        mock_jsonld.flatten = Mock()
        mock_jsonld.frame = Mock()
        mock_jsonld.normalize = Mock()
        mock_jsonld.to_rdf = Mock()
        mock_jsonld.from_rdf = Mock()
        yield mock_jsonld


@pytest.fixture
def mock_rdflib():
    """Mock rdflib library."""
    with patch("rdflib.Graph") as mock_graph:
        mock_graph_instance = Mock()
        mock_graph_instance.parse = Mock()
        mock_graph_instance.serialize = Mock()
        mock_graph_instance.query = Mock()
        mock_graph_instance.triples = Mock(return_value=[])
        mock_graph_instance.subjects = Mock(return_value=[])
        mock_graph_instance.predicates = Mock(return_value=[])
        mock_graph_instance.objects = Mock(return_value=[])
        mock_graph.return_value = mock_graph_instance
        yield mock_graph


@pytest.fixture
def sample_jsonld_content():
    """Sample JSON-LD content for testing."""
    return {
        "@context": {
            "name": "http://schema.org/name",
            "description": "http://schema.org/description",
            "Chemical": "http://example.org/Chemical",
            "hasFormula": "http://example.org/hasFormula",
            "hasMolecularWeight": "http://example.org/hasMolecularWeight",
        },
        "@graph": [
            {
                "@id": "http://example.org/chemical/glucose",
                "@type": "Chemical",
                "name": "Glucose",
                "description": "Simple sugar with formula C6H12O6",
                "hasFormula": "C6H12O6",
                "hasMolecularWeight": 180.16,
            },
            {
                "@id": "http://example.org/chemical/fructose",
                "@type": "Chemical",
                "name": "Fructose",
                "description": "Simple sugar isomer of glucose",
                "hasFormula": "C6H12O6",
                "hasMolecularWeight": 180.16,
            },
        ],
    }


@pytest.fixture
def sample_jsonld_compact():
    """Sample compact JSON-LD for testing."""
    return {
        "@context": "http://schema.org/",
        "@id": "http://example.org/person/john",
        "@type": "Person",
        "name": "John Doe",
        "email": "john@example.org",
        "knows": {
            "@id": "http://example.org/person/jane",
            "@type": "Person",
            "name": "Jane Smith",
        },
    }


@pytest.fixture
def sample_jsonld_expanded():
    """Sample expanded JSON-LD for testing."""
    return [
        {
            "@id": "http://example.org/person/john",
            "@type": ["http://schema.org/Person"],
            "http://schema.org/name": [{"@value": "John Doe"}],
            "http://schema.org/email": [{"@value": "john@example.org"}],
            "http://schema.org/knows": [
                {
                    "@id": "http://example.org/person/jane",
                    "@type": ["http://schema.org/Person"],
                    "http://schema.org/name": [{"@value": "Jane Smith"}],
                }
            ],
        }
    ]


@pytest.fixture
def sample_jsonld_flattened():
    """Sample flattened JSON-LD for testing."""
    return {
        "@context": "http://schema.org/",
        "@graph": [
            {
                "@id": "http://example.org/person/john",
                "@type": "Person",
                "name": "John Doe",
                "email": "john@example.org",
                "knows": {"@id": "http://example.org/person/jane"},
            },
            {
                "@id": "http://example.org/person/jane",
                "@type": "Person",
                "name": "Jane Smith",
            },
        ],
    }


@pytest.fixture
def sample_jsonld_with_frame():
    """Sample JSON-LD frame for testing."""
    return {
        "@context": "http://schema.org/",
        "@type": "Person",
        "name": {},
        "knows": {"@type": "Person", "name": {}},
    }


@pytest.fixture
def sample_context_document():
    """Sample JSON-LD context document."""
    return {
        "@context": {
            "name": "http://schema.org/name",
            "description": "http://schema.org/description",
            "Chemical": "http://example.org/ontology/Chemical",
            "Compound": "http://example.org/ontology/Compound",
            "hasFormula": {
                "@id": "http://example.org/ontology/hasFormula",
                "@type": "http://www.w3.org/2001/XMLSchema#string",
            },
            "hasMolecularWeight": {
                "@id": "http://example.org/ontology/hasMolecularWeight",
                "@type": "http://www.w3.org/2001/XMLSchema#decimal",
            },
            "sameAs": {"@id": "http://www.w3.org/2002/07/owl#sameAs", "@type": "@id"},
        }
    }


@pytest.fixture
def sample_ontology_jsonld():
    """Sample ontology in JSON-LD format."""
    return {
        "@context": {
            "owl": "http://www.w3.org/2002/07/owl#",
            "rdfs": "http://www.w3.org/2000/01/rdf-schema#",
            "xsd": "http://www.w3.org/2001/XMLSchema#",
            "onto": "http://example.org/ontology/",
            "label": "rdfs:label",
            "comment": "rdfs:comment",
            "subClassOf": "rdfs:subClassOf",
            "domain": "rdfs:domain",
            "range": "rdfs:range",
        },
        "@graph": [
            {
                "@id": "onto:Chemical",
                "@type": "owl:Class",
                "label": "Chemical",
                "comment": "A chemical substance or compound",
            },
            {
                "@id": "onto:Molecule",
                "@type": "owl:Class",
                "label": "Molecule",
                "comment": "A group of atoms bonded together",
                "subClassOf": "onto:Chemical",
            },
            {
                "@id": "onto:hasFormula",
                "@type": "owl:DatatypeProperty",
                "label": "has formula",
                "domain": "onto:Chemical",
                "range": "xsd:string",
            },
        ],
    }


class TestJSONLDParserCreation:
    """Test JSONLDParser instantiation and configuration."""

    def test_jsonld_parser_creation_default(self, mock_jsonld_parser):
        """Test creating JSONLDParser with default settings."""
        parser = mock_jsonld_parser()

        # Verify parser was created
        assert parser is not None
        mock_jsonld_parser.assert_called_once()

    def test_jsonld_parser_creation_with_options(self, mock_jsonld_parser):
        """Test creating JSONLDParser with custom options."""
        options = {
            "context_url": "http://example.org/context.jsonld",
            "expand_context": True,
            "validate_syntax": True,
            "resolve_references": True,
            "process_sequences": True,
        }

        parser = mock_jsonld_parser(options=options)
        mock_jsonld_parser.assert_called_once_with(options=options)

    def test_jsonld_parser_creation_with_invalid_options(self, mock_jsonld_parser):
        """Test creating JSONLDParser with invalid options raises error."""
        mock_jsonld_parser.side_effect = ValueError(
            "Invalid option: unknown_context_processor"
        )

        invalid_options = {"unknown_context_processor": "invalid"}

        with pytest.raises(ValueError, match="Invalid option"):
            mock_jsonld_parser(options=invalid_options)

    def test_jsonld_parser_inherits_from_abstract_parser(self, mock_jsonld_parser):
        """Test that JSONLDParser implements AbstractParser interface."""
        parser = mock_jsonld_parser()

        # Verify all required methods are available
        required_methods = [
            "parse",
            "validate",
            "get_supported_formats",
            "set_options",
            "get_options",
        ]
        for method in required_methods:
            assert hasattr(parser, method)

    def test_jsonld_parser_specific_methods(self, mock_jsonld_parser):
        """Test JSONLDParser has JSON-LD specific methods."""
        parser = mock_jsonld_parser()

        # Verify JSON-LD specific methods are available
        jsonld_methods = [
            "expand",
            "compact",
            "flatten",
            "frame",
            "normalize",
            "resolve_context",
            "to_rdf",
            "from_rdf",
        ]
        for method in jsonld_methods:
            assert hasattr(parser, method)


class TestJSONLDParserFormatSupport:
    """Test JSON-LD format detection and validation."""

    def test_get_supported_formats(self, mock_jsonld_parser):
        """Test getting list of supported JSON-LD formats."""
        parser = mock_jsonld_parser()

        formats = parser.get_supported_formats()

        expected_formats = ["jsonld", "json", "json-ld"]
        assert all(fmt in formats for fmt in expected_formats)

    def test_detect_format_jsonld(self, mock_jsonld_parser, sample_jsonld_content):
        """Test detecting JSON-LD format."""
        parser = mock_jsonld_parser()
        parser.detect_format.return_value = "jsonld"

        content_str = json.dumps(sample_jsonld_content)
        detected_format = parser.detect_format(content_str)

        assert detected_format == "jsonld"
        parser.detect_format.assert_called_once_with(content_str)

    def test_detect_format_from_context(self, mock_jsonld_parser):
        """Test detecting JSON-LD format from @context presence."""
        parser = mock_jsonld_parser()
        parser.detect_format.return_value = "jsonld"

        content_with_context = '{"@context": "http://schema.org/", "name": "Test"}'
        detected_format = parser.detect_format(content_with_context)

        assert detected_format == "jsonld"

    def test_detect_format_from_id_type(self, mock_jsonld_parser):
        """Test detecting JSON-LD format from @id/@type presence."""
        parser = mock_jsonld_parser()
        parser.detect_format.return_value = "jsonld"

        content_with_id_type = (
            '{"@id": "http://example.org/1", "@type": "Person", "name": "John"}'
        )
        detected_format = parser.detect_format(content_with_id_type)

        assert detected_format == "jsonld"

    def test_validate_format_valid_jsonld(
        self, mock_jsonld_parser, sample_jsonld_content
    ):
        """Test validating valid JSON-LD content."""
        parser = mock_jsonld_parser()
        parser.validate_format.return_value = True

        content_str = json.dumps(sample_jsonld_content)
        is_valid = parser.validate_format(content_str, "jsonld")

        assert is_valid is True
        parser.validate_format.assert_called_once_with(content_str, "jsonld")

    def test_validate_format_invalid_json(self, mock_jsonld_parser):
        """Test validating invalid JSON content."""
        parser = mock_jsonld_parser()
        parser.validate_format.return_value = False

        invalid_json = (
            '{"@context": "http://schema.org/", "name": "Test"'  # Missing closing brace
        )
        is_valid = parser.validate_format(invalid_json, "jsonld")

        assert is_valid is False

    def test_validate_format_valid_json_not_jsonld(self, mock_jsonld_parser):
        """Test validating valid JSON that is not JSON-LD."""
        parser = mock_jsonld_parser()
        parser.validate_format.return_value = False

        regular_json = '{"name": "Test", "value": 123}'
        is_valid = parser.validate_format(regular_json, "jsonld")

        assert is_valid is False


class TestJSONLDParserParsing:
    """Test core JSON-LD parsing functionality."""

    def test_parse_string_jsonld_content(
        self, mock_jsonld_parser, sample_jsonld_content
    ):
        """Test parsing JSON-LD content from string."""
        parser = mock_jsonld_parser()
        mock_result = Mock()
        mock_result.data = sample_jsonld_content
        mock_result.format = "jsonld"
        parser.parse_string.return_value = mock_result

        content_str = json.dumps(sample_jsonld_content)
        result = parser.parse_string(content_str)

        assert result == mock_result
        assert result.format == "jsonld"
        parser.parse_string.assert_called_once_with(content_str)

    def test_parse_file_jsonld_path(self, mock_jsonld_parser):
        """Test parsing JSON-LD file from file path."""
        parser = mock_jsonld_parser()
        mock_result = Mock()
        mock_result.graphs = 2
        mock_result.nodes = 10
        parser.parse_file.return_value = mock_result

        file_path = "/path/to/ontology.jsonld"
        result = parser.parse_file(file_path)

        assert result == mock_result
        assert result.graphs == 2
        assert result.nodes == 10
        parser.parse_file.assert_called_once_with(file_path)

    def test_parse_stream_jsonld_data(self, mock_jsonld_parser, sample_jsonld_compact):
        """Test parsing JSON-LD from stream/file-like object."""
        parser = mock_jsonld_parser()
        mock_result = Mock()
        mock_result.type = "compact"
        parser.parse_stream.return_value = mock_result

        jsonld_stream = io.StringIO(json.dumps(sample_jsonld_compact))
        result = parser.parse_stream(jsonld_stream)

        assert result == mock_result
        assert result.type == "compact"
        parser.parse_stream.assert_called_once_with(jsonld_stream)

    def test_parse_with_context_resolution(
        self, mock_jsonld_parser, sample_jsonld_content
    ):
        """Test parsing with context resolution."""
        parser = mock_jsonld_parser()
        mock_result = Mock()
        mock_result.context_resolved = True
        parser.parse_string.return_value = mock_result

        # Configure parser for context resolution
        parser.set_options({"resolve_context": True, "expand_context": True})
        content_str = json.dumps(sample_jsonld_content)
        result = parser.parse_string(content_str)

        assert result == mock_result
        assert result.context_resolved is True
        parser.set_options.assert_called_with(
            {"resolve_context": True, "expand_context": True}
        )

    def test_parse_with_reference_resolution(
        self, mock_jsonld_parser, sample_jsonld_flattened
    ):
        """Test parsing with reference resolution."""
        parser = mock_jsonld_parser()
        mock_result = Mock()
        mock_result.references_resolved = True
        parser.parse_string.return_value = mock_result

        parser.set_options({"resolve_references": True})
        content_str = json.dumps(sample_jsonld_flattened)
        result = parser.parse_string(content_str)

        assert result == mock_result
        assert result.references_resolved is True
        parser.set_options.assert_called_with({"resolve_references": True})

    def test_parse_with_base_uri(self, mock_jsonld_parser, sample_jsonld_content):
        """Test parsing with base URI setting."""
        parser = mock_jsonld_parser()
        mock_result = Mock()
        mock_result.base_uri = "http://example.org/"
        parser.parse_string.return_value = mock_result

        parser.set_options({"base": "http://example.org/"})
        content_str = json.dumps(sample_jsonld_content)
        result = parser.parse_string(content_str)

        assert result == mock_result
        assert result.base_uri == "http://example.org/"
        parser.set_options.assert_called_with({"base": "http://example.org/"})

    def test_parse_with_processing_mode(
        self, mock_jsonld_parser, sample_jsonld_content
    ):
        """Test parsing with specific processing mode."""
        parser = mock_jsonld_parser()
        mock_result = Mock()
        mock_result.processing_mode = "json-ld-1.1"
        parser.parse_string.return_value = mock_result

        parser.set_options({"processingMode": "json-ld-1.1"})
        content_str = json.dumps(sample_jsonld_content)
        result = parser.parse_string(content_str)

        assert result == mock_result
        assert result.processing_mode == "json-ld-1.1"
        parser.set_options.assert_called_with({"processingMode": "json-ld-1.1"})


class TestJSONLDParserContextHandling:
    """Test @context processing and management."""

    def test_resolve_context_from_url(self, mock_jsonld_parser):
        """Test resolving context from URL."""
        parser = mock_jsonld_parser()
        mock_context = {
            "name": "http://schema.org/name",
            "description": "http://schema.org/description",
        }
        parser.resolve_context.return_value = mock_context

        context_url = "http://schema.org/"
        resolved_context = parser.resolve_context(context_url)

        assert resolved_context == mock_context
        parser.resolve_context.assert_called_once_with(context_url)

    def test_resolve_context_from_document(
        self, mock_jsonld_parser, sample_context_document
    ):
        """Test resolving context from document."""
        parser = mock_jsonld_parser()
        parser.resolve_context.return_value = sample_context_document["@context"]

        resolved_context = parser.resolve_context(sample_context_document)

        assert "Chemical" in resolved_context
        assert "hasFormula" in resolved_context
        parser.resolve_context.assert_called_once_with(sample_context_document)

    def test_set_custom_context(self, mock_jsonld_parser):
        """Test setting custom context."""
        parser = mock_jsonld_parser()
        custom_context = {
            "name": "http://xmlns.com/foaf/0.1/name",
            "email": "http://xmlns.com/foaf/0.1/mbox",
            "Person": "http://xmlns.com/foaf/0.1/Person",
        }

        parser.set_context(custom_context)
        parser.set_context.assert_called_once_with(custom_context)

    def test_get_current_context(self, mock_jsonld_parser):
        """Test getting current context."""
        parser = mock_jsonld_parser()
        expected_context = {
            "@vocab": "http://schema.org/",
            "name": "http://schema.org/name",
            "description": "http://schema.org/description",
        }
        parser.get_context.return_value = expected_context

        current_context = parser.get_context()

        assert current_context == expected_context
        assert "@vocab" in current_context

    def test_merge_contexts(self, mock_jsonld_parser):
        """Test merging multiple contexts."""
        parser = mock_jsonld_parser()

        context1 = {"name": "http://schema.org/name"}
        context2 = {"email": "http://schema.org/email"}
        merged_context = {
            "name": "http://schema.org/name",
            "email": "http://schema.org/email",
        }
        parser.merge_contexts.return_value = merged_context

        result = parser.merge_contexts([context1, context2])

        assert result == merged_context
        assert "name" in result
        assert "email" in result
        parser.merge_contexts.assert_called_once_with([context1, context2])

    def test_validate_context_valid(self, mock_jsonld_parser, sample_context_document):
        """Test validating valid context."""
        parser = mock_jsonld_parser()
        parser.validate_context.return_value = True

        is_valid = parser.validate_context(sample_context_document["@context"])

        assert is_valid is True
        parser.validate_context.assert_called_once_with(
            sample_context_document["@context"]
        )

    def test_validate_context_invalid(self, mock_jsonld_parser):
        """Test validating invalid context."""
        parser = mock_jsonld_parser()
        parser.validate_context.return_value = False

        invalid_context = {"@context": {"@invalid": "not allowed"}}
        is_valid = parser.validate_context(invalid_context)

        assert is_valid is False

    def test_context_inheritance(self, mock_jsonld_parser):
        """Test context inheritance in nested structures."""
        parser = mock_jsonld_parser()

        nested_doc = {
            "@context": {"name": "http://schema.org/name"},
            "name": "John",
            "address": {
                "@context": {"street": "http://schema.org/streetAddress"},
                "street": "123 Main St",
            },
        }

        mock_result = Mock()
        mock_result.contexts = 2
        mock_result.inherited = True
        parser.parse_string.return_value = mock_result

        result = parser.parse_string(json.dumps(nested_doc))

        assert result.contexts == 2
        assert result.inherited is True


class TestJSONLDParserConversion:
    """Test converting parsed JSON-LD to internal models."""

    def test_expand_jsonld_document(self, mock_jsonld_parser, sample_jsonld_compact):
        """Test expanding compact JSON-LD document."""
        parser = mock_jsonld_parser()
        mock_expanded = [
            {
                "@id": "http://example.org/person/john",
                "@type": ["http://schema.org/Person"],
                "http://schema.org/name": [{"@value": "John Doe"}],
            }
        ]
        parser.expand.return_value = mock_expanded

        expanded = parser.expand(sample_jsonld_compact)

        assert expanded == mock_expanded
        assert len(expanded) == 1
        parser.expand.assert_called_once_with(sample_jsonld_compact)

    def test_compact_jsonld_document(self, mock_jsonld_parser, sample_jsonld_expanded):
        """Test compacting expanded JSON-LD document."""
        parser = mock_jsonld_parser()
        mock_compacted = {
            "@context": "http://schema.org/",
            "@id": "http://example.org/person/john",
            "@type": "Person",
            "name": "John Doe",
        }
        parser.compact.return_value = mock_compacted

        context = {"@context": "http://schema.org/"}
        compacted = parser.compact(sample_jsonld_expanded, context)

        assert compacted == mock_compacted
        assert compacted["@type"] == "Person"
        parser.compact.assert_called_once_with(sample_jsonld_expanded, context)

    def test_flatten_jsonld_document(self, mock_jsonld_parser, sample_jsonld_compact):
        """Test flattening JSON-LD document."""
        parser = mock_jsonld_parser()
        mock_flattened = {
            "@context": "http://schema.org/",
            "@graph": [
                {
                    "@id": "http://example.org/person/john",
                    "@type": "Person",
                    "name": "John Doe",
                    "knows": {"@id": "http://example.org/person/jane"},
                },
                {
                    "@id": "http://example.org/person/jane",
                    "@type": "Person",
                    "name": "Jane Smith",
                },
            ],
        }
        parser.flatten.return_value = mock_flattened

        flattened = parser.flatten(sample_jsonld_compact)

        assert flattened == mock_flattened
        assert "@graph" in flattened
        assert len(flattened["@graph"]) == 2
        parser.flatten.assert_called_once_with(sample_jsonld_compact)

    def test_frame_jsonld_document(
        self, mock_jsonld_parser, sample_jsonld_flattened, sample_jsonld_with_frame
    ):
        """Test framing JSON-LD document."""
        parser = mock_jsonld_parser()
        mock_framed = {
            "@context": "http://schema.org/",
            "@graph": [
                {
                    "@id": "http://example.org/person/john",
                    "@type": "Person",
                    "name": "John Doe",
                    "knows": {
                        "@id": "http://example.org/person/jane",
                        "@type": "Person",
                        "name": "Jane Smith",
                    },
                }
            ],
        }
        parser.frame.return_value = mock_framed

        framed = parser.frame(sample_jsonld_flattened, sample_jsonld_with_frame)

        assert framed == mock_framed
        parser.frame.assert_called_once_with(
            sample_jsonld_flattened, sample_jsonld_with_frame
        )

    def test_normalize_jsonld_document(self, mock_jsonld_parser, sample_jsonld_content):
        """Test normalizing JSON-LD document."""
        parser = mock_jsonld_parser()
        mock_normalized = '_:b0 <http://example.org/hasFormula> "C6H12O6" .\n_:b0 <http://schema.org/name> "Glucose" .'
        parser.normalize.return_value = mock_normalized

        normalized = parser.normalize(sample_jsonld_content)

        assert normalized == mock_normalized
        assert "C6H12O6" in normalized
        parser.normalize.assert_called_once_with(sample_jsonld_content)

    def test_to_ontology_conversion(self, mock_jsonld_parser, sample_ontology_jsonld):
        """Test converting parsed JSON-LD to Ontology model."""
        parser = mock_jsonld_parser()

        # Mock the conversion result
        mock_ontology = Mock()
        mock_ontology.id = "jsonld_ontology_001"
        mock_ontology.name = "JSON-LD Parsed Ontology"
        mock_ontology.terms = {}
        mock_ontology.relationships = {}
        parser.to_ontology.return_value = mock_ontology

        # Parse and convert
        parsed_result = parser.parse_string(json.dumps(sample_ontology_jsonld))
        ontology = parser.to_ontology(parsed_result)

        assert ontology == mock_ontology
        assert ontology.id == "jsonld_ontology_001"
        assert ontology.name == "JSON-LD Parsed Ontology"

    def test_extract_terms_from_parsed_jsonld(self, mock_jsonld_parser):
        """Test extracting Term objects from parsed JSON-LD."""
        parser = mock_jsonld_parser()

        # Mock extracted terms
        mock_term1 = Mock()
        mock_term1.id = "http://example.org/chemical/glucose"
        mock_term1.name = "Glucose"
        mock_term1.definition = "Simple sugar with formula C6H12O6"
        mock_term1.type = "Chemical"

        mock_term2 = Mock()
        mock_term2.id = "http://example.org/chemical/fructose"
        mock_term2.name = "Fructose"
        mock_term2.definition = "Simple sugar isomer of glucose"
        mock_term2.type = "Chemical"

        mock_terms = [mock_term1, mock_term2]
        parser.extract_terms.return_value = mock_terms

        parsed_result = Mock()
        terms = parser.extract_terms(parsed_result)

        assert len(terms) == 2
        assert terms[0].id == "http://example.org/chemical/glucose"
        assert terms[1].name == "Fructose"

    def test_extract_relationships_from_parsed_jsonld(self, mock_jsonld_parser):
        """Test extracting Relationship objects from parsed JSON-LD."""
        parser = mock_jsonld_parser()

        # Mock extracted relationships
        mock_relationships = [
            Mock(
                id="REL:001",
                subject="http://example.org/ontology/Molecule",
                predicate="http://www.w3.org/2000/01/rdf-schema#subClassOf",
                object="http://example.org/ontology/Chemical",
                confidence=1.0,
                source="jsonld_inference",
            )
        ]
        parser.extract_relationships.return_value = mock_relationships

        parsed_result = Mock()
        relationships = parser.extract_relationships(parsed_result)

        assert len(relationships) == 1
        assert relationships[0].subject == "http://example.org/ontology/Molecule"
        assert (
            relationships[0].predicate
            == "http://www.w3.org/2000/01/rdf-schema#subClassOf"
        )
        assert relationships[0].source == "jsonld_inference"

    def test_extract_metadata_from_parsed_jsonld(self, mock_jsonld_parser):
        """Test extracting metadata from parsed JSON-LD."""
        parser = mock_jsonld_parser()

        # Mock extracted metadata
        mock_metadata = {
            "source_format": "jsonld",
            "node_count": 50,
            "graph_count": 2,
            "context_urls": ["http://schema.org/", "http://example.org/context.jsonld"],
            "namespaces": {
                "schema": "http://schema.org/",
                "owl": "http://www.w3.org/2002/07/owl#",
            },
            "processing_mode": "json-ld-1.1",
            "expanded_form": True,
            "parser_version": "1.0.0",
        }
        parser.extract_metadata.return_value = mock_metadata

        parsed_result = Mock()
        metadata = parser.extract_metadata(parsed_result)

        assert metadata["source_format"] == "jsonld"
        assert metadata["node_count"] == 50
        assert len(metadata["context_urls"]) == 2
        assert "schema" in metadata["namespaces"]

    def test_to_rdf_conversion(self, mock_jsonld_parser, sample_jsonld_content):
        """Test converting JSON-LD to RDF."""
        parser = mock_jsonld_parser()
        mock_rdf_graph = Mock()
        mock_rdf_graph.format = "turtle"
        mock_rdf_graph.triples_count = 15
        parser.to_rdf.return_value = mock_rdf_graph

        rdf_graph = parser.to_rdf(sample_jsonld_content)

        assert rdf_graph == mock_rdf_graph
        assert rdf_graph.format == "turtle"
        assert rdf_graph.triples_count == 15
        parser.to_rdf.assert_called_once_with(sample_jsonld_content)

    def test_from_rdf_conversion(self, mock_jsonld_parser):
        """Test converting RDF to JSON-LD."""
        parser = mock_jsonld_parser()
        mock_jsonld_doc = {
            "@context": "http://schema.org/",
            "@graph": [
                {"@id": "http://example.org/1", "@type": "Person", "name": "John"}
            ],
        }
        parser.from_rdf.return_value = mock_jsonld_doc

        mock_rdf_input = Mock()
        jsonld_doc = parser.from_rdf(mock_rdf_input)

        assert jsonld_doc == mock_jsonld_doc
        assert "@graph" in jsonld_doc
        parser.from_rdf.assert_called_once_with(mock_rdf_input)


class TestJSONLDParserErrorHandling:
    """Test error handling and validation."""

    def test_parse_invalid_json_content(self, mock_jsonld_parser):
        """Test parsing invalid JSON content raises appropriate error."""
        parser = mock_jsonld_parser()
        parser.parse_string.side_effect = json.JSONDecodeError(
            "Expecting ',' delimiter", '{"@context": "http://schema.org/"', 35
        )

        invalid_json = (
            '{"@context": "http://schema.org/", "name": "Test"'  # Missing closing brace
        )

        with pytest.raises(json.JSONDecodeError):
            parser.parse_string(invalid_json)

    def test_parse_nonexistent_file(self, mock_jsonld_parser):
        """Test parsing nonexistent file raises FileNotFoundError."""
        parser = mock_jsonld_parser()
        parser.parse_file.side_effect = FileNotFoundError(
            "File not found: /nonexistent/file.jsonld"
        )

        with pytest.raises(FileNotFoundError, match="File not found"):
            parser.parse_file("/nonexistent/file.jsonld")

    def test_parse_empty_file(self, mock_jsonld_parser):
        """Test parsing empty file raises appropriate error."""
        parser = mock_jsonld_parser()
        parser.parse_string.side_effect = ValueError("Empty JSON-LD document")

        with pytest.raises(ValueError, match="Empty JSON-LD document"):
            parser.parse_string("")

    def test_context_resolution_error(self, mock_jsonld_parser):
        """Test handling of context resolution errors."""
        parser = mock_jsonld_parser()
        parser.resolve_context.side_effect = Exception(
            "Failed to resolve context: Connection timeout"
        )

        with pytest.raises(Exception, match="Failed to resolve context"):
            parser.resolve_context("http://unreachable-context.example.org/")

    def test_invalid_jsonld_structure_error(self, mock_jsonld_parser):
        """Test handling of invalid JSON-LD structure."""
        parser = mock_jsonld_parser()
        parser.parse_string.side_effect = ValueError(
            "Invalid JSON-LD: @context must be an object, array, or string"
        )

        invalid_jsonld = '{"@context": 123, "name": "Test"}'

        with pytest.raises(ValueError, match="Invalid JSON-LD"):
            parser.parse_string(invalid_jsonld)

    def test_circular_context_reference_error(self, mock_jsonld_parser):
        """Test handling of circular context references."""
        parser = mock_jsonld_parser()
        parser.resolve_context.side_effect = ValueError(
            "Circular context reference detected"
        )

        circular_context = {
            "@context": ["http://example.org/context1", "http://example.org/context2"]
        }

        with pytest.raises(ValueError, match="Circular context reference"):
            parser.resolve_context(circular_context)

    def test_validation_errors_collection(
        self, mock_jsonld_parser, sample_jsonld_content
    ):
        """Test collection and reporting of validation errors."""
        parser = mock_jsonld_parser()

        # Mock validation errors
        validation_errors = [
            "Warning: Context URL http://missing.example.org/ not reachable",
            "Error: Invalid IRI in @id field: 'not-a-valid-iri'",
            "Warning: Unknown property 'unknownProp' without context mapping",
        ]
        parser.get_validation_errors.return_value = validation_errors

        # Parse with validation
        parser.set_options({"validate_on_parse": True})
        parser.parse_string(json.dumps(sample_jsonld_content))

        errors = parser.get_validation_errors()
        assert len(errors) == 3
        assert any("Context URL" in error for error in errors)
        assert any("Invalid IRI" in error for error in errors)

    def test_memory_limit_exceeded(self, mock_jsonld_parser):
        """Test handling of memory limit exceeded during parsing."""
        parser = mock_jsonld_parser()
        parser.parse_file.side_effect = MemoryError(
            "Memory limit exceeded while processing large JSON-LD document"
        )

        with pytest.raises(MemoryError, match="Memory limit exceeded"):
            parser.parse_file("/path/to/huge_jsonld_file.jsonld")

    def test_malformed_jsonld_recovery(self, mock_jsonld_parser):
        """Test error recovery with malformed JSON-LD."""
        parser = mock_jsonld_parser()
        mock_result = Mock()
        mock_result.errors = [
            "Node 'http://example.org/invalid': Invalid @type value",
            "Context resolution failed for 'http://missing.example.org/'",
        ]
        mock_result.valid_nodes = 18
        mock_result.total_nodes = 20
        parser.parse_string.return_value = mock_result

        parser.set_options({"error_recovery": True, "skip_invalid_nodes": True})

        malformed_jsonld = {
            "@context": ["http://schema.org/", "http://missing.example.org/"],
            "@graph": [
                {"@id": "http://example.org/1", "@type": "Person", "name": "John"},
                {"@id": "http://example.org/invalid", "@type": 123, "name": "Invalid"},
            ],
        }

        result = parser.parse_string(json.dumps(malformed_jsonld))

        assert result == mock_result
        assert result.valid_nodes == 18
        assert len(result.errors) == 2

    def test_rdf_conversion_error(self, mock_jsonld_parser):
        """Test handling of RDF conversion errors."""
        parser = mock_jsonld_parser()
        parser.to_rdf.side_effect = ValueError(
            "Cannot convert to RDF: Invalid property IRI"
        )

        invalid_jsonld = {
            "@context": {"invalid:property": "not-a-valid-iri"},
            "@id": "http://example.org/test",
            "invalid:property": "value",
        }

        with pytest.raises(ValueError, match="Cannot convert to RDF"):
            parser.to_rdf(invalid_jsonld)


class TestJSONLDParserValidation:
    """Test JSON-LD validation functionality."""

    def test_validate_jsonld_structure(self, mock_jsonld_parser, sample_jsonld_content):
        """Test validating JSON-LD structure and format."""
        parser = mock_jsonld_parser()
        parser.validate.return_value = True

        content_str = json.dumps(sample_jsonld_content)
        is_valid = parser.validate(content_str)

        assert is_valid is True
        parser.validate.assert_called_once_with(content_str)

    def test_validate_jsonld_context(self, mock_jsonld_parser, sample_context_document):
        """Test validating JSON-LD context."""
        parser = mock_jsonld_parser()
        parser.validate_jsonld.return_value = {
            "valid_structure": True,
            "valid_context": True,
            "context_resolvable": True,
            "valid_syntax": True,
            "errors": [],
            "warnings": [],
        }

        content_str = json.dumps(sample_context_document)
        validation_result = parser.validate_jsonld(content_str)

        assert validation_result["valid_context"] is True
        assert validation_result["context_resolvable"] is True
        assert len(validation_result["errors"]) == 0

    def test_validate_jsonld_iris(self, mock_jsonld_parser, sample_jsonld_content):
        """Test validating JSON-LD IRIs."""
        parser = mock_jsonld_parser()
        parser.validate_jsonld.return_value = {
            "valid_structure": True,
            "valid_iris": True,
            "iri_consistency": True,
            "valid_node_ids": True,
            "errors": [],
            "warnings": [
                "Node http://example.org/chemical/glucose: Missing rdfs:label"
            ],
        }

        content_str = json.dumps(sample_jsonld_content)
        validation_result = parser.validate_jsonld(content_str)

        assert validation_result["valid_iris"] is True
        assert validation_result["valid_node_ids"] is True
        assert len(validation_result["warnings"]) == 1

    def test_validate_required_properties(self, mock_jsonld_parser):
        """Test validation of required properties."""
        parser = mock_jsonld_parser()
        parser.validate_jsonld.return_value = {
            "valid_structure": False,
            "required_properties": ["@id", "@type"],
            "missing_properties": ["@type"],
            "nodes_missing_properties": 3,
        }

        jsonld_missing_type = {
            "@context": "http://schema.org/",
            "@graph": [{"@id": "http://example.org/1", "name": "Missing Type"}],
        }

        content_str = json.dumps(jsonld_missing_type)
        validation_result = parser.validate_jsonld(content_str)

        assert validation_result["valid_structure"] is False
        assert "@type" in validation_result["missing_properties"]
        assert validation_result["nodes_missing_properties"] == 3

    def test_validate_property_ranges(self, mock_jsonld_parser, sample_ontology_jsonld):
        """Test validation of property ranges and domains."""
        parser = mock_jsonld_parser()
        parser.validate_jsonld.return_value = {
            "valid_structure": True,
            "valid_property_ranges": True,
            "valid_property_domains": True,
            "range_violations": [],
            "domain_violations": [],
        }

        content_str = json.dumps(sample_ontology_jsonld)
        validation_result = parser.validate_jsonld(content_str)

        assert validation_result["valid_property_ranges"] is True
        assert validation_result["valid_property_domains"] is True
        assert len(validation_result["range_violations"]) == 0

    def test_validate_rdf_compliance(self, mock_jsonld_parser, sample_jsonld_content):
        """Test validation of RDF compliance."""
        parser = mock_jsonld_parser()
        parser.validate_jsonld.return_value = {
            "valid_structure": True,
            "rdf_compliant": True,
            "valid_triples": True,
            "blank_node_handling": True,
            "errors": [],
        }

        content_str = json.dumps(sample_jsonld_content)
        validation_result = parser.validate_jsonld(content_str)

        assert validation_result["rdf_compliant"] is True
        assert validation_result["valid_triples"] is True
        assert len(validation_result["errors"]) == 0


class TestJSONLDParserOptions:
    """Test parser configuration and options."""

    def test_set_parsing_options(self, mock_jsonld_parser):
        """Test setting various parsing options."""
        parser = mock_jsonld_parser()

        options = {
            "base": "http://example.org/",
            "expandContext": True,
            "compactArrays": True,
            "processingMode": "json-ld-1.1",
            "documentLoader": None,
            "safe": True,
        }

        parser.set_options(options)
        parser.set_options.assert_called_once_with(options)

    def test_get_current_options(self, mock_jsonld_parser):
        """Test getting current parser options."""
        parser = mock_jsonld_parser()

        expected_options = {
            "base": "http://example.org/",
            "expandContext": True,
            "compactArrays": True,
            "processingMode": "json-ld-1.1",
            "validate_on_parse": False,
            "resolve_references": True,
        }
        parser.get_options.return_value = expected_options

        current_options = parser.get_options()

        assert current_options == expected_options
        assert "processingMode" in current_options

    def test_reset_options_to_defaults(self, mock_jsonld_parser):
        """Test resetting options to default values."""
        parser = mock_jsonld_parser()

        # Set some custom options first
        parser.set_options({"base": "http://custom.org/", "expandContext": False})

        # Reset to defaults
        parser.reset_options()
        parser.reset_options.assert_called_once()

    def test_invalid_option_handling(self, mock_jsonld_parser):
        """Test handling of invalid configuration options."""
        parser = mock_jsonld_parser()
        parser.set_options.side_effect = ValueError(
            "Unknown option: invalid_context_processor"
        )

        invalid_options = {"invalid_context_processor": "unknown"}

        with pytest.raises(ValueError, match="Unknown option"):
            parser.set_options(invalid_options)

    def test_option_validation(self, mock_jsonld_parser):
        """Test validation of option values."""
        parser = mock_jsonld_parser()
        parser.set_options.side_effect = ValueError(
            "Invalid value for option 'processingMode': must be 'json-ld-1.0' or 'json-ld-1.1'"
        )

        invalid_options = {"processingMode": "invalid-mode"}

        with pytest.raises(ValueError, match="Invalid value for option"):
            parser.set_options(invalid_options)

    def test_context_loading_options(self, mock_jsonld_parser):
        """Test context loading configuration options."""
        parser = mock_jsonld_parser()

        context_options = {
            "documentLoader": Mock(),
            "timeout": 30,
            "max_redirects": 5,
            "cache_contexts": True,
            "secure": True,
        }

        parser.set_options(context_options)
        parser.set_options.assert_called_once_with(context_options)

    def test_processing_options(self, mock_jsonld_parser):
        """Test JSON-LD processing options."""
        parser = mock_jsonld_parser()

        processing_options = {
            "expandContext": True,
            "compactArrays": False,
            "graph": True,
            "skipExpansion": False,
            "ordered": True,
            "produceGeneralizedRdf": False,
        }

        parser.set_options(processing_options)
        parser.set_options.assert_called_once_with(processing_options)


class TestJSONLDParserIntegration:
    """Integration tests with other components."""

    def test_integration_with_ontology_manager(self, mock_jsonld_parser):
        """Test integration with OntologyManager."""
        parser = mock_jsonld_parser()

        # Mock integration with ontology manager
        with patch(
            "aim2_project.aim2_ontology.ontology_manager.OntologyManager"
        ) as MockManager:
            manager = MockManager()
            manager.add_ontology = Mock(return_value=True)

            # Parse and add to manager
            mock_ontology = Mock()
            parser.to_ontology.return_value = mock_ontology

            parsed_result = parser.parse_string('{"@context": "http://schema.org/"}')
            ontology = parser.to_ontology(parsed_result)
            manager.add_ontology(ontology)

            manager.add_ontology.assert_called_once_with(ontology)

    def test_integration_with_rdf_stores(self, mock_jsonld_parser, mock_rdflib):
        """Test integration with RDF triple stores."""
        parser = mock_jsonld_parser()

        # Mock RDF store integration
        rdf_graph = mock_rdflib()
        mock_rdf_data = Mock()
        parser.to_rdf.return_value = mock_rdf_data

        jsonld_content = {
            "@context": "http://schema.org/",
            "@id": "http://example.org/1",
        }
        parsed_result = parser.parse_string(json.dumps(jsonld_content))
        rdf_data = parser.to_rdf(parsed_result)

        # Add to RDF store
        rdf_graph.parse(data=rdf_data, format="json-ld")

        rdf_graph.parse.assert_called_once_with(data=rdf_data, format="json-ld")

    def test_integration_with_validation_pipeline(self, mock_jsonld_parser):
        """Test integration with validation pipeline."""
        parser = mock_jsonld_parser()

        # Mock validation pipeline
        with patch(
            "aim2_project.aim2_ontology.validators.ValidationPipeline"
        ) as MockPipeline:
            validator = MockPipeline()
            validator.validate_jsonld = Mock(return_value={"valid": True, "errors": []})

            jsonld_content = {
                "@context": "http://schema.org/",
                "@id": "http://example.org/1",
            }
            parsed_result = parser.parse_string(json.dumps(jsonld_content))

            # Run validation
            validation_result = validator.validate_jsonld(parsed_result)

            assert validation_result["valid"] is True
            validator.validate_jsonld.assert_called_once_with(parsed_result)

    def test_integration_with_sparql_queries(self, mock_jsonld_parser, mock_rdflib):
        """Test integration with SPARQL queries."""
        parser = mock_jsonld_parser()
        rdf_graph = mock_rdflib()

        # Mock SPARQL query results
        mock_query_result = [
            Mock(
                subject="http://example.org/1",
                predicate="http://schema.org/name",
                object="John",
            )
        ]
        rdf_graph.query.return_value = mock_query_result

        jsonld_content = {
            "@context": "http://schema.org/",
            "@id": "http://example.org/1",
            "name": "John",
        }
        parsed_result = parser.parse_string(json.dumps(jsonld_content))
        rdf_data = parser.to_rdf(parsed_result)

        rdf_graph.parse(data=rdf_data, format="json-ld")

        sparql_query = """
        SELECT ?subject ?name WHERE {
            ?subject <http://schema.org/name> ?name .
        }
        """
        results = rdf_graph.query(sparql_query)

        assert len(results) == 1
        rdf_graph.query.assert_called_once_with(sparql_query)

    def test_end_to_end_parsing_workflow(
        self, mock_jsonld_parser, sample_ontology_jsonld
    ):
        """Test complete end-to-end parsing workflow."""
        parser = mock_jsonld_parser()

        # Mock the complete workflow
        mock_parsed = Mock()
        mock_ontology = Mock()
        mock_ontology.terms = {
            "http://example.org/ontology/Chemical": Mock(),
            "http://example.org/ontology/Molecule": Mock(),
        }
        mock_ontology.relationships = {
            "REL:001": Mock(),
        }

        parser.parse_string.return_value = mock_parsed
        parser.to_ontology.return_value = mock_ontology

        # Execute workflow
        content_str = json.dumps(sample_ontology_jsonld)
        parsed_result = parser.parse_string(content_str)
        ontology = parser.to_ontology(parsed_result)

        # Verify results
        assert len(ontology.terms) == 2
        assert len(ontology.relationships) == 1

        parser.parse_string.assert_called_once_with(content_str)
        parser.to_ontology.assert_called_once_with(mock_parsed)


class TestJSONLDParserPerformance:
    """Performance and scalability tests."""

    def test_parse_large_jsonld_performance(self, mock_jsonld_parser):
        """Test parsing performance with large JSON-LD documents."""
        parser = mock_jsonld_parser()

        # Configure for performance testing
        parser.set_options(
            {"streaming": True, "lazy_loading": True, "memory_limit": "1GB"}
        )

        mock_result = Mock()
        mock_result.node_count = 100000
        mock_result.graph_count = 5
        mock_result.memory_usage = "512MB"
        mock_result.processing_time = 45.2
        parser.parse_file.return_value = mock_result

        result = parser.parse_file("/path/to/large_ontology.jsonld")

        assert result.node_count == 100000
        assert result.graph_count == 5
        assert result.processing_time < 60  # Less than 1 minute

    def test_memory_usage_optimization(self, mock_jsonld_parser):
        """Test memory usage optimization features."""
        parser = mock_jsonld_parser()

        # Configure memory optimization
        parser.set_options({"streaming": True, "chunk_size": 1000, "gc_frequency": 100})

        mock_result = Mock()
        mock_result.memory_efficient = True
        mock_result.peak_memory = "256MB"
        parser.parse_file.return_value = mock_result

        result = parser.parse_file("/path/to/memory_intensive_ontology.jsonld")

        assert result == mock_result
        assert result.memory_efficient is True
        assert result.peak_memory == "256MB"

    def test_streaming_processing(self, mock_jsonld_parser):
        """Test streaming processing for very large JSON-LD documents."""
        parser = mock_jsonld_parser()

        # Configure for streaming processing
        parser.set_options(
            {"streaming": True, "chunk_size": 5000, "lazy_contexts": True}
        )

        mock_chunks = [Mock(), Mock(), Mock(), Mock()]
        parser.parse_file.return_value = mock_chunks

        chunks = parser.parse_file("/path/to/huge_knowledge_graph.jsonld")

        assert len(chunks) == 4
        parser.set_options.assert_called_with(
            {"streaming": True, "chunk_size": 5000, "lazy_contexts": True}
        )

    def test_parallel_context_resolution(self, mock_jsonld_parser):
        """Test parallel context resolution for multiple context URLs."""
        parser = mock_jsonld_parser()

        # Configure for parallel processing
        parser.set_options({"parallel_contexts": True, "max_workers": 4, "timeout": 30})

        context_urls = [
            "http://schema.org/",
            "http://example.org/context1.jsonld",
            "http://example.org/context2.jsonld",
            "http://example.org/context3.jsonld",
        ]

        mock_contexts = [Mock(), Mock(), Mock(), Mock()]
        parser.resolve_context.side_effect = mock_contexts

        contexts = []
        for url in context_urls:
            context = parser.resolve_context(url)
            contexts.append(context)

        assert len(contexts) == 4
        assert parser.resolve_context.call_count == 4

    def test_caching_performance(self, mock_jsonld_parser):
        """Test caching performance for repeated operations."""
        parser = mock_jsonld_parser()

        # Configure caching
        parser.set_options(
            {
                "cache_contexts": True,
                "cache_expanded": True,
                "cache_size": 1000,
                "cache_ttl": 3600,
            }
        )

        # Mock cached operations
        context_url = "http://schema.org/"
        mock_context = {"name": "http://schema.org/name"}

        parser.resolve_context.return_value = mock_context

        # First call - should cache
        context1 = parser.resolve_context(context_url)
        # Second call - should use cache
        context2 = parser.resolve_context(context_url)

        assert context1 == context2
        assert context1 == mock_context


# Additional test fixtures for complex scenarios
@pytest.fixture
def complex_jsonld_content():
    """Fixture providing complex JSON-LD data for testing."""
    return {
        "@context": [
            "http://schema.org/",
            {
                "chem": "http://example.org/chemistry/",
                "onto": "http://example.org/ontology/",
                "hasChemicalFormula": "chem:hasChemicalFormula",
                "hasMolecularWeight": {
                    "@id": "chem:hasMolecularWeight",
                    "@type": "http://www.w3.org/2001/XMLSchema#decimal",
                },
                "isIsomerOf": {"@id": "chem:isIsomerOf", "@type": "@id"},
            },
        ],
        "@graph": [
            {
                "@id": "chem:glucose",
                "@type": ["onto:Monosaccharide", "onto:Chemical"],
                "name": "D-Glucose",
                "description": "Primary energy source for cellular metabolism",
                "hasChemicalFormula": "C6H12O6",
                "hasMolecularWeight": 180.156,
                "sameAs": [
                    "http://www.wikidata.org/entity/Q37525",
                    "http://purl.obolibrary.org/obo/CHEBI_17234",
                ],
                "isIsomerOf": ["chem:fructose", "chem:galactose"],
            },
            {
                "@id": "chem:fructose",
                "@type": ["onto:Monosaccharide", "onto:Chemical"],
                "name": "D-Fructose",
                "description": "Fruit sugar, sweeter than glucose",
                "hasChemicalFormula": "C6H12O6",
                "hasMolecularWeight": 180.156,
                "isIsomerOf": ["chem:glucose", "chem:galactose"],
            },
        ],
    }


@pytest.fixture
def malformed_jsonld_content():
    """Fixture providing malformed JSON-LD content for error testing."""
    return {
        "@context": "http://schema.org/",
        "@graph": [
            {
                "@id": 123,  # Invalid: @id must be string
                "@type": "Person",
                "name": "Invalid ID",
            },
            {
                "@id": "http://example.org/2",
                "@type": ["Person", 456],  # Invalid: @type array contains non-string
                "name": "Invalid Type",
            },
            {
                "@id": "http://example.org/3",
                "@context": {"@invalid": "not-allowed"},  # Invalid: @invalid keyword
                "name": "Invalid Context",
            },
        ],
    }


@pytest.fixture
def jsonld_configuration_options():
    """Fixture providing comprehensive JSON-LD parser configuration options."""
    return {
        "basic_options": {
            "base": None,
            "expandContext": True,
            "compactArrays": True,
            "graph": False,
            "skipExpansion": False,
            "processingMode": "json-ld-1.1",
        },
        "performance_options": {
            "streaming": False,
            "lazy_loading": False,
            "chunk_size": None,
            "memory_limit": None,
            "parallel_contexts": False,
            "max_workers": 1,
        },
        "validation_options": {
            "validate_on_parse": True,
            "strict_mode": False,
            "validate_iris": True,
            "validate_contexts": True,
            "check_duplicates": True,
        },
        "context_options": {
            "cache_contexts": True,
            "cache_size": 100,
            "cache_ttl": 3600,
            "timeout": 30,
            "max_redirects": 5,
            "secure": True,
        },
        "conversion_options": {
            "rdf_direction": None,
            "use_native_types": True,
            "use_rdf_type": False,
            "produce_generalized_rdf": False,
            "ordered": False,
        },
        "error_handling_options": {
            "error_recovery": False,
            "skip_invalid_nodes": False,
            "max_errors": None,
            "continue_on_error": False,
            "collect_warnings": True,
        },
    }


@pytest.fixture
def temp_jsonld_files():
    """Create temporary JSON-LD files for testing."""
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)

        # Create various JSON-LD files
        files = {}

        # Simple JSON-LD file
        simple_jsonld = temp_path / "simple.jsonld"
        simple_content = {
            "@context": "http://schema.org/",
            "@id": "http://example.org/person/1",
            "@type": "Person",
            "name": "John Doe",
        }
        simple_jsonld.write_text(json.dumps(simple_content, indent=2))
        files["simple"] = str(simple_jsonld)

        # Ontology JSON-LD file
        ontology_jsonld = temp_path / "ontology.jsonld"
        ontology_content = {
            "@context": {
                "owl": "http://www.w3.org/2002/07/owl#",
                "rdfs": "http://www.w3.org/2000/01/rdf-schema#",
            },
            "@graph": [
                {
                    "@id": "http://example.org/Chemical",
                    "@type": "owl:Class",
                    "rdfs:label": "Chemical",
                }
            ],
        }
        ontology_jsonld.write_text(json.dumps(ontology_content, indent=2))
        files["ontology"] = str(ontology_jsonld)

        # Large JSON-LD file (simulated)
        large_jsonld = temp_path / "large.jsonld"
        large_content = {"@context": "http://schema.org/", "@graph": []}
        for i in range(100):
            large_content["@graph"].append(
                {
                    "@id": f"http://example.org/person/{i}",
                    "@type": "Person",
                    "name": f"Person {i}",
                    "email": f"person{i}@example.org",
                }
            )
        large_jsonld.write_text(json.dumps(large_content, indent=2))
        files["large"] = str(large_jsonld)

        yield files
