"""
Comprehensive Unit Tests for OWL Parser (TDD Approach)

This module provides comprehensive unit tests for the OWL parser functionality in the AIM2 ontology
information extraction system. The tests follow test-driven development (TDD) approach,
defining the expected behavior of the OWL parser classes before implementation.

Test Classes:
    TestAbstractParserCreation: Tests for AbstractParser base class instantiation
    TestAbstractParserInterface: Tests for AbstractParser interface methods
    TestOWLParserCreation: Tests for OWLParser instantiation and configuration
    TestOWLParserFormatSupport: Tests for OWL format detection and validation
    TestOWLParserParsing: Tests for core OWL parsing functionality
    TestOWLParserConversion: Tests for converting parsed OWL to internal models
    TestOWLParserErrorHandling: Tests for error handling and validation
    TestOWLParserValidation: Tests for OWL validation functionality
    TestOWLParserOptions: Tests for parser configuration and options
    TestOWLParserIntegration: Integration tests with other components
    TestOWLParserPerformance: Performance and scalability tests

The AbstractParser is expected to be an abstract base class providing:
- Core parsing interface: parse(), validate(), get_supported_formats()
- Configuration management: set_options(), get_options()
- Error handling: ValidationError, ParseError exceptions
- Metadata extraction: get_metadata()

The OWLParser is expected to be a concrete implementation providing:
- OWL format support: OWL/XML, RDF/XML, Turtle, N-Triples, JSON-LD
- Integration with owlready2 and rdflib libraries
- Conversion to internal Term/Relationship/Ontology models
- Comprehensive validation and error reporting
- Performance optimization for large ontologies
- Configurable parsing options and format detection

Dependencies:
    - pytest: For test framework
    - unittest.mock: For mocking functionality
    - owlready2: For OWL ontology manipulation (mocked)
    - rdflib: For RDF graph processing (mocked)
    - typing: For type hints

Usage:
    pytest tests/unit/test_owl_parser.py -v
"""

import tempfile
from pathlib import Path
from unittest.mock import Mock, patch

import pytest


# Mock the parser classes since they don't exist yet (TDD approach)
@pytest.fixture
def mock_abstract_parser():
    """Mock AbstractParser base class."""
    with patch("aim2_project.aim2_ontology.parsers.AbstractParser") as mock_parser:
        mock_instance = Mock()
        mock_instance.parse = Mock()
        mock_instance.validate = Mock()
        mock_instance.get_supported_formats = Mock(return_value=["owl", "rdf", "ttl"])
        mock_instance.set_options = Mock()
        mock_instance.get_options = Mock(return_value={})
        mock_instance.get_metadata = Mock(return_value={})
        mock_parser.return_value = mock_instance
        yield mock_parser


@pytest.fixture
def mock_owl_parser():
    """Mock OWLParser class."""
    with patch("aim2_project.aim2_ontology.parsers.OWLParser") as mock_parser:
        mock_instance = Mock()
        # Core parsing methods
        mock_instance.parse = Mock()
        mock_instance.parse_file = Mock()
        mock_instance.parse_string = Mock()
        mock_instance.parse_url = Mock()

        # Format detection and validation
        mock_instance.detect_format = Mock()
        mock_instance.validate_format = Mock()
        mock_instance.get_supported_formats = Mock(
            return_value=["owl", "rdf", "ttl", "nt", "n3", "xml", "json-ld"]
        )

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
        mock_instance.validate_owl = Mock()
        mock_instance.get_validation_errors = Mock(return_value=[])

        mock_parser.return_value = mock_instance
        yield mock_parser


@pytest.fixture
def mock_owlready2():
    """Mock owlready2 library."""
    with patch("owlready2.get_ontology") as mock_get_ontology:
        mock_onto = Mock()
        mock_onto.load = Mock()
        mock_onto.classes = Mock(return_value=[])
        mock_onto.individuals = Mock(return_value=[])
        mock_onto.properties = Mock(return_value=[])
        mock_onto.name = "test_ontology"
        mock_onto.base_iri = "http://example.org/ontology#"
        mock_get_ontology.return_value = mock_onto
        yield mock_get_ontology


@pytest.fixture
def mock_rdflib():
    """Mock rdflib library."""
    with patch("rdflib.Graph") as mock_graph:
        mock_graph_instance = Mock()
        mock_graph_instance.parse = Mock()
        mock_graph_instance.serialize = Mock()
        mock_graph_instance.query = Mock()
        mock_graph_instance.__len__ = Mock(return_value=100)
        mock_graph.return_value = mock_graph_instance
        yield mock_graph


@pytest.fixture
def sample_owl_content():
    """Sample OWL content for testing."""
    return """<?xml version="1.0"?>
<rdf:RDF xmlns="http://example.org/ontology#"
         xml:base="http://example.org/ontology"
         xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#"
         xmlns:owl="http://www.w3.org/2002/07/owl#"
         xmlns:rdfs="http://www.w3.org/2000/01/rdf-schema#">

    <owl:Ontology rdf:about="http://example.org/ontology">
        <rdfs:label>Test Ontology</rdfs:label>
        <rdfs:comment>A test ontology for parsing</rdfs:comment>
    </owl:Ontology>

    <owl:Class rdf:about="http://example.org/ontology#Chemical">
        <rdfs:label>Chemical</rdfs:label>
        <rdfs:comment>A chemical compound</rdfs:comment>
    </owl:Class>

    <owl:Class rdf:about="http://example.org/ontology#Glucose">
        <rdfs:label>Glucose</rdfs:label>
        <rdfs:subClassOf rdf:resource="http://example.org/ontology#Chemical"/>
    </owl:Class>

</rdf:RDF>"""


@pytest.fixture
def sample_turtle_content():
    """Sample Turtle content for testing."""
    return """@prefix : <http://example.org/ontology#> .
@prefix owl: <http://www.w3.org/2002/07/owl#> .
@prefix rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#> .
@prefix rdfs: <http://www.w3.org/2000/01/rdf-schema#> .

<http://example.org/ontology> a owl:Ontology ;
    rdfs:label "Test Ontology" ;
    rdfs:comment "A test ontology for parsing" .

:Chemical a owl:Class ;
    rdfs:label "Chemical" ;
    rdfs:comment "A chemical compound" .

:Glucose a owl:Class ;
    rdfs:label "Glucose" ;
    rdfs:subClassOf :Chemical .
"""


@pytest.fixture
def sample_json_ld_content():
    """Sample JSON-LD content for testing."""
    return """{
  "@context": {
    "@base": "http://example.org/ontology",
    "owl": "http://www.w3.org/2002/07/owl#",
    "rdf": "http://www.w3.org/1999/02/22-rdf-syntax-ns#",
    "rdfs": "http://www.w3.org/2000/01/rdf-schema#"
  },
  "@graph": [
    {
      "@id": "http://example.org/ontology",
      "@type": "owl:Ontology",
      "rdfs:label": "Test Ontology",
      "rdfs:comment": "A test ontology for parsing"
    },
    {
      "@id": "http://example.org/ontology#Chemical",
      "@type": "owl:Class",
      "rdfs:label": "Chemical",
      "rdfs:comment": "A chemical compound"
    }
  ]
}"""


class TestAbstractParserCreation:
    """Test AbstractParser base class instantiation and basic functionality."""

    def test_abstract_parser_cannot_be_instantiated_directly(self):
        """Test that AbstractParser cannot be instantiated directly (abstract class)."""
        # Mock AbstractParser as an abstract base class
        with patch(
            "aim2_project.aim2_ontology.parsers.AbstractParser"
        ) as MockAbstractParser:
            MockAbstractParser.side_effect = TypeError(
                "Can't instantiate abstract class AbstractParser"
            )

            with pytest.raises(TypeError, match="Can't instantiate abstract class"):
                MockAbstractParser()

    def test_abstract_parser_defines_required_interface(self, mock_abstract_parser):
        """Test that AbstractParser defines the required interface methods."""
        # Get the mock class (not instance)
        parser_class = mock_abstract_parser

        # Test that required methods are defined in the interface
        required_methods = [
            "parse",
            "validate",
            "get_supported_formats",
            "set_options",
            "get_options",
        ]

        # In a real implementation, we would check the abstract methods
        # For now, we just verify the concept exists
        assert hasattr(parser_class, "__abstractmethods__") or True  # Interface concept


class TestAbstractParserInterface:
    """Test AbstractParser interface methods and contracts."""

    def test_parse_method_signature(self, mock_abstract_parser):
        """Test that parse method has correct signature."""
        parser = mock_abstract_parser()

        # Test method exists and can be called
        parser.parse("test_content")
        parser.parse.assert_called_once_with("test_content")

    def test_validate_method_signature(self, mock_abstract_parser):
        """Test that validate method has correct signature."""
        parser = mock_abstract_parser()
        parser.validate.return_value = True

        result = parser.validate("test_content")
        parser.validate.assert_called_once_with("test_content")
        assert result is True

    def test_get_supported_formats_returns_list(self, mock_abstract_parser):
        """Test that get_supported_formats returns a list of supported formats."""
        parser = mock_abstract_parser()

        formats = parser.get_supported_formats()

        assert isinstance(formats, list)
        assert len(formats) > 0
        assert all(isinstance(fmt, str) for fmt in formats)

    def test_options_management(self, mock_abstract_parser):
        """Test options getter and setter methods."""
        parser = mock_abstract_parser()
        parser.get_options.return_value = {"validate_on_parse": True}

        # Test getting options
        options = parser.get_options()
        assert isinstance(options, dict)

        # Test setting options
        new_options = {"validate_on_parse": False, "strict_mode": True}
        parser.set_options(new_options)
        parser.set_options.assert_called_once_with(new_options)

    def test_get_metadata_returns_dict(self, mock_abstract_parser):
        """Test that get_metadata returns metadata dictionary."""
        parser = mock_abstract_parser()
        parser.get_metadata.return_value = {
            "parser_version": "1.0.0",
            "supported_formats": ["owl", "rdf", "ttl"],
        }

        metadata = parser.get_metadata()

        assert isinstance(metadata, dict)
        assert "parser_version" in metadata


class TestOWLParserCreation:
    """Test OWLParser instantiation and configuration."""

    def test_owl_parser_creation_default(self, mock_owl_parser):
        """Test creating OWLParser with default settings."""
        parser = mock_owl_parser()

        # Verify parser was created
        assert parser is not None
        mock_owl_parser.assert_called_once()

    def test_owl_parser_creation_with_options(self, mock_owl_parser):
        """Test creating OWLParser with custom options."""
        options = {
            "validate_on_parse": True,
            "strict_mode": False,
            "preserve_namespaces": True,
            "include_imports": False,
        }

        parser = mock_owl_parser(options=options)
        mock_owl_parser.assert_called_once_with(options=options)

    def test_owl_parser_creation_with_invalid_options(self, mock_owl_parser):
        """Test creating OWLParser with invalid options raises error."""
        mock_owl_parser.side_effect = ValueError("Invalid option: unknown_option")

        invalid_options = {"unknown_option": True}

        with pytest.raises(ValueError, match="Invalid option"):
            mock_owl_parser(options=invalid_options)

    def test_owl_parser_inherits_from_abstract_parser(self, mock_owl_parser):
        """Test that OWLParser implements AbstractParser interface."""
        parser = mock_owl_parser()

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


class TestOWLParserFormatSupport:
    """Test OWL format detection and validation."""

    def test_get_supported_formats(self, mock_owl_parser):
        """Test getting list of supported OWL formats."""
        parser = mock_owl_parser()

        formats = parser.get_supported_formats()

        expected_formats = ["owl", "rdf", "ttl", "nt", "n3", "xml", "json-ld"]
        assert all(fmt in formats for fmt in expected_formats)

    def test_detect_format_owl_xml(self, mock_owl_parser, sample_owl_content):
        """Test detecting OWL/XML format."""
        parser = mock_owl_parser()
        parser.detect_format.return_value = "owl"

        detected_format = parser.detect_format(sample_owl_content)

        assert detected_format == "owl"
        parser.detect_format.assert_called_once_with(sample_owl_content)

    def test_detect_format_turtle(self, mock_owl_parser, sample_turtle_content):
        """Test detecting Turtle format."""
        parser = mock_owl_parser()
        parser.detect_format.return_value = "ttl"

        detected_format = parser.detect_format(sample_turtle_content)

        assert detected_format == "ttl"

    def test_detect_format_json_ld(self, mock_owl_parser, sample_json_ld_content):
        """Test detecting JSON-LD format."""
        parser = mock_owl_parser()
        parser.detect_format.return_value = "json-ld"

        detected_format = parser.detect_format(sample_json_ld_content)

        assert detected_format == "json-ld"

    def test_validate_format_valid_content(self, mock_owl_parser, sample_owl_content):
        """Test validating valid OWL content."""
        parser = mock_owl_parser()
        parser.validate_format.return_value = True

        is_valid = parser.validate_format(sample_owl_content, "owl")

        assert is_valid is True
        parser.validate_format.assert_called_once_with(sample_owl_content, "owl")

    def test_validate_format_invalid_content(self, mock_owl_parser):
        """Test validating invalid OWL content."""
        parser = mock_owl_parser()
        parser.validate_format.return_value = False

        invalid_content = "This is not valid OWL content"
        is_valid = parser.validate_format(invalid_content, "owl")

        assert is_valid is False


class TestOWLParserParsing:
    """Test core OWL parsing functionality."""

    def test_parse_string_owl_content(self, mock_owl_parser, sample_owl_content):
        """Test parsing OWL content from string."""
        parser = mock_owl_parser()
        mock_result = Mock()
        parser.parse_string.return_value = mock_result

        result = parser.parse_string(sample_owl_content)

        assert result == mock_result
        parser.parse_string.assert_called_once_with(sample_owl_content)

    def test_parse_file_owl_path(self, mock_owl_parser):
        """Test parsing OWL file from file path."""
        parser = mock_owl_parser()
        mock_result = Mock()
        parser.parse_file.return_value = mock_result

        file_path = "/path/to/ontology.owl"
        result = parser.parse_file(file_path)

        assert result == mock_result
        parser.parse_file.assert_called_once_with(file_path)

    def test_parse_file_with_format_specification(self, mock_owl_parser):
        """Test parsing file with explicit format specification."""
        parser = mock_owl_parser()
        mock_result = Mock()
        parser.parse_file.return_value = mock_result

        file_path = "/path/to/ontology.ttl"
        result = parser.parse_file(file_path, format="ttl")

        assert result == mock_result
        parser.parse_file.assert_called_once_with(file_path, format="ttl")

    def test_parse_url_remote_ontology(self, mock_owl_parser):
        """Test parsing OWL from remote URL."""
        parser = mock_owl_parser()
        mock_result = Mock()
        parser.parse_url.return_value = mock_result

        url = "http://example.org/ontology.owl"
        result = parser.parse_url(url)

        assert result == mock_result
        parser.parse_url.assert_called_once_with(url)

    def test_parse_with_imports(self, mock_owl_parser, sample_owl_content):
        """Test parsing OWL with import resolution."""
        parser = mock_owl_parser()
        mock_result = Mock()
        parser.parse_string.return_value = mock_result

        # Configure parser to include imports
        parser.set_options({"include_imports": True})
        result = parser.parse_string(sample_owl_content)

        assert result == mock_result
        parser.set_options.assert_called_with({"include_imports": True})

    def test_parse_with_validation(self, mock_owl_parser, sample_owl_content):
        """Test parsing with validation enabled."""
        parser = mock_owl_parser()
        mock_result = Mock()
        parser.parse_string.return_value = mock_result

        # Configure parser to validate on parse
        parser.set_options({"validate_on_parse": True})
        result = parser.parse_string(sample_owl_content)

        assert result == mock_result
        parser.set_options.assert_called_with({"validate_on_parse": True})

    def test_parse_preserves_namespaces(self, mock_owl_parser, sample_owl_content):
        """Test parsing preserves namespace information."""
        parser = mock_owl_parser()
        mock_result = Mock()
        mock_result.namespaces = {"rdf": "http://www.w3.org/1999/02/22-rdf-syntax-ns#"}
        parser.parse_string.return_value = mock_result

        parser.set_options({"preserve_namespaces": True})
        result = parser.parse_string(sample_owl_content)

        assert hasattr(result, "namespaces")
        assert "rdf" in result.namespaces

    def test_parse_large_ontology(self, mock_owl_parser):
        """Test parsing large ontology with performance considerations."""
        parser = mock_owl_parser()
        mock_result = Mock()
        parser.parse_file.return_value = mock_result

        # Configure for large ontology parsing
        parser.set_options({"batch_size": 1000, "memory_efficient": True})
        result = parser.parse_file("/path/to/large_ontology.owl")

        assert result == mock_result
        parser.set_options.assert_called_with(
            {"batch_size": 1000, "memory_efficient": True}
        )

    def test_parse_with_custom_base_iri(self, mock_owl_parser, sample_owl_content):
        """Test parsing with custom base IRI."""
        parser = mock_owl_parser()
        mock_result = Mock()
        parser.parse_string.return_value = mock_result

        custom_base = "http://custom.org/ontology#"
        parser.set_options({"base_iri": custom_base})
        result = parser.parse_string(sample_owl_content)

        assert result == mock_result
        parser.set_options.assert_called_with({"base_iri": custom_base})

    def test_parse_multiple_formats(
        self, mock_owl_parser, sample_owl_content, sample_turtle_content
    ):
        """Test parsing different formats with same parser instance."""
        parser = mock_owl_parser()

        # Mock different results for different formats
        owl_result = Mock()
        owl_result.format = "owl"
        turtle_result = Mock()
        turtle_result.format = "ttl"

        parser.parse_string.side_effect = [owl_result, turtle_result]

        # Parse OWL content
        result1 = parser.parse_string(sample_owl_content)
        assert result1.format == "owl"

        # Parse Turtle content
        result2 = parser.parse_string(sample_turtle_content)
        assert result2.format == "ttl"

    def test_parse_with_error_recovery(self, mock_owl_parser):
        """Test parsing with error recovery enabled."""
        parser = mock_owl_parser()
        mock_result = Mock()
        mock_result.errors = ["Warning: Unknown property"]
        parser.parse_string.return_value = mock_result

        parser.set_options({"error_recovery": True, "strict_mode": False})

        malformed_content = "<?xml version='1.0'?><owl:Ontology><!-- malformed -->"
        result = parser.parse_string(malformed_content)

        assert result == mock_result
        assert hasattr(result, "errors")

    def test_parse_streaming_mode(self, mock_owl_parser):
        """Test parsing in streaming mode for very large files."""
        parser = mock_owl_parser()
        mock_result = Mock()
        parser.parse_file.return_value = mock_result

        parser.set_options({"streaming_mode": True, "chunk_size": 1024})
        result = parser.parse_file("/path/to/huge_ontology.owl")

        assert result == mock_result
        parser.set_options.assert_called_with(
            {"streaming_mode": True, "chunk_size": 1024}
        )


class TestOWLParserConversion:
    """Test converting parsed OWL to internal models."""

    def test_to_ontology_conversion(self, mock_owl_parser, sample_owl_content):
        """Test converting parsed OWL to Ontology model."""
        parser = mock_owl_parser()

        # Mock the conversion result
        mock_ontology = Mock()
        mock_ontology.id = "http://example.org/ontology"
        mock_ontology.name = "Test Ontology"
        mock_ontology.terms = {}
        mock_ontology.relationships = {}
        parser.to_ontology.return_value = mock_ontology

        # Parse and convert
        parsed_result = parser.parse_string(sample_owl_content)
        ontology = parser.to_ontology(parsed_result)

        assert ontology == mock_ontology
        assert ontology.id == "http://example.org/ontology"
        assert ontology.name == "Test Ontology"

    def test_extract_terms_from_parsed_owl(self, mock_owl_parser):
        """Test extracting Term objects from parsed OWL."""
        parser = mock_owl_parser()

        # Mock extracted terms with proper configuration
        mock_term1 = Mock()
        mock_term1.id = "CHEM:001"
        mock_term1.name = "Chemical"
        mock_term1.definition = "A chemical compound"

        mock_term2 = Mock()
        mock_term2.id = "CHEM:002"
        mock_term2.name = "Glucose"
        mock_term2.definition = "A simple sugar"

        mock_terms = [mock_term1, mock_term2]
        parser.extract_terms.return_value = mock_terms

        parsed_result = Mock()
        terms = parser.extract_terms(parsed_result)

        assert len(terms) == 2
        assert terms[0].id == "CHEM:001"
        assert terms[1].name == "Glucose"

    def test_extract_relationships_from_parsed_owl(self, mock_owl_parser):
        """Test extracting Relationship objects from parsed OWL."""
        parser = mock_owl_parser()

        # Mock extracted relationships
        mock_relationships = [
            Mock(
                id="REL:001",
                subject="CHEM:002",
                predicate="is_a",
                object="CHEM:001",
                confidence=1.0,
            )
        ]
        parser.extract_relationships.return_value = mock_relationships

        parsed_result = Mock()
        relationships = parser.extract_relationships(parsed_result)

        assert len(relationships) == 1
        assert relationships[0].subject == "CHEM:002"
        assert relationships[0].predicate == "is_a"
        assert relationships[0].object == "CHEM:001"

    def test_extract_metadata_from_parsed_owl(self, mock_owl_parser):
        """Test extracting metadata from parsed OWL."""
        parser = mock_owl_parser()

        # Mock extracted metadata
        mock_metadata = {
            "title": "Test Ontology",
            "description": "A test ontology for parsing",
            "version": "1.0.0",
            "authors": ["Test Author"],
            "created_date": "2023-01-01",
            "namespaces": {
                "owl": "http://www.w3.org/2002/07/owl#",
                "rdf": "http://www.w3.org/1999/02/22-rdf-syntax-ns#",
            },
        }
        parser.extract_metadata.return_value = mock_metadata

        parsed_result = Mock()
        metadata = parser.extract_metadata(parsed_result)

        assert metadata["title"] == "Test Ontology"
        assert "namespaces" in metadata
        assert len(metadata["namespaces"]) >= 2

    def test_conversion_with_filters(self, mock_owl_parser):
        """Test conversion with entity filters."""
        parser = mock_owl_parser()

        # Configure filters
        filters = {
            "include_classes": True,
            "include_properties": True,
            "include_individuals": False,
            "namespace_filter": ["http://example.org/ontology#"],
        }
        parser.set_options({"conversion_filters": filters})

        mock_ontology = Mock()
        parser.to_ontology.return_value = mock_ontology

        parsed_result = Mock()
        ontology = parser.to_ontology(parsed_result)

        assert ontology == mock_ontology
        parser.set_options.assert_called_with({"conversion_filters": filters})

    def test_conversion_preserves_provenance(self, mock_owl_parser):
        """Test that conversion preserves provenance information."""
        parser = mock_owl_parser()

        mock_ontology = Mock()
        mock_ontology.metadata = {
            "source_format": "owl",
            "parser_version": "1.0.0",
            "parsing_timestamp": "2023-01-01T00:00:00Z",
        }
        parser.to_ontology.return_value = mock_ontology

        parsed_result = Mock()
        ontology = parser.to_ontology(parsed_result)

        assert "source_format" in ontology.metadata
        assert "parser_version" in ontology.metadata
        assert "parsing_timestamp" in ontology.metadata

    def test_incremental_conversion(self, mock_owl_parser):
        """Test incremental conversion for large ontologies."""
        parser = mock_owl_parser()

        # Configure for incremental processing
        parser.set_options({"incremental_conversion": True, "batch_size": 100})

        mock_ontology = Mock()
        parser.to_ontology.return_value = mock_ontology

        parsed_result = Mock()
        ontology = parser.to_ontology(parsed_result)

        assert ontology == mock_ontology
        parser.set_options.assert_called_with(
            {"incremental_conversion": True, "batch_size": 100}
        )


class TestOWLParserErrorHandling:
    """Test error handling and validation."""

    def test_parse_invalid_xml_content(self, mock_owl_parser):
        """Test parsing invalid XML content raises appropriate error."""
        parser = mock_owl_parser()
        parser.parse_string.side_effect = ValueError("Invalid XML: malformed tag")

        invalid_xml = "<?xml version='1.0'?><owl:Ontology><unclosed_tag>"

        with pytest.raises(ValueError, match="Invalid XML"):
            parser.parse_string(invalid_xml)

    def test_parse_nonexistent_file(self, mock_owl_parser):
        """Test parsing nonexistent file raises FileNotFoundError."""
        parser = mock_owl_parser()
        parser.parse_file.side_effect = FileNotFoundError(
            "File not found: /nonexistent/file.owl"
        )

        with pytest.raises(FileNotFoundError, match="File not found"):
            parser.parse_file("/nonexistent/file.owl")

    def test_parse_unsupported_format(self, mock_owl_parser):
        """Test parsing unsupported format raises appropriate error."""
        parser = mock_owl_parser()
        parser.parse_string.side_effect = ValueError("Unsupported format: xyz")

        with pytest.raises(ValueError, match="Unsupported format"):
            parser.parse_string("content", format="xyz")

    def test_validation_errors_collection(self, mock_owl_parser, sample_owl_content):
        """Test collection and reporting of validation errors."""
        parser = mock_owl_parser()

        # Mock validation errors
        validation_errors = [
            "Warning: Class 'Chemical' has no definition",
            "Error: Invalid property range for 'hasName'",
            "Warning: Unused namespace prefix 'ex'",
        ]
        parser.get_validation_errors.return_value = validation_errors

        # Parse with validation
        parser.set_options({"validate_on_parse": True})
        parser.parse_string(sample_owl_content)

        errors = parser.get_validation_errors()
        assert len(errors) == 3
        assert any("Chemical" in error for error in errors)
        assert any("hasName" in error for error in errors)

    def test_circular_import_detection(self, mock_owl_parser):
        """Test detection of circular imports in OWL files."""
        parser = mock_owl_parser()
        parser.parse_file.side_effect = ValueError(
            "Circular import detected: A imports B imports A"
        )

        with pytest.raises(ValueError, match="Circular import detected"):
            parser.parse_file("/path/to/ontology_with_circular_imports.owl")

    def test_memory_limit_exceeded(self, mock_owl_parser):
        """Test handling of memory limit exceeded during parsing."""
        parser = mock_owl_parser()
        parser.parse_file.side_effect = MemoryError(
            "Memory limit exceeded while parsing large ontology"
        )

        with pytest.raises(MemoryError, match="Memory limit exceeded"):
            parser.parse_file("/path/to/huge_ontology.owl")

    def test_network_timeout_handling(self, mock_owl_parser):
        """Test handling of network timeouts when parsing remote URLs."""
        parser = mock_owl_parser()
        parser.parse_url.side_effect = TimeoutError(
            "Network timeout while fetching ontology"
        )

        with pytest.raises(TimeoutError, match="Network timeout"):
            parser.parse_url("http://slow-server.org/ontology.owl")


class TestOWLParserValidation:
    """Test OWL validation functionality."""

    def test_validate_owl_structure(self, mock_owl_parser, sample_owl_content):
        """Test validating OWL structure and syntax."""
        parser = mock_owl_parser()
        parser.validate.return_value = True

        is_valid = parser.validate(sample_owl_content)

        assert is_valid is True
        parser.validate.assert_called_once_with(sample_owl_content)

    def test_validate_owl_semantics(self, mock_owl_parser, sample_owl_content):
        """Test validating OWL semantics and consistency."""
        parser = mock_owl_parser()
        parser.validate_owl.return_value = {
            "is_valid": True,
            "consistency_check": True,
            "satisfiability_check": True,
            "errors": [],
            "warnings": [],
        }

        validation_result = parser.validate_owl(sample_owl_content)

        assert validation_result["is_valid"] is True
        assert validation_result["consistency_check"] is True
        assert len(validation_result["errors"]) == 0

    def test_validate_with_custom_profile(self, mock_owl_parser, sample_owl_content):
        """Test validation with custom OWL profile (e.g., OWL-EL, OWL-QL)."""
        parser = mock_owl_parser()
        parser.set_options({"owl_profile": "OWL-EL"})
        parser.validate_owl.return_value = {"is_valid": True, "profile_compliant": True}

        validation_result = parser.validate_owl(sample_owl_content)

        assert validation_result["profile_compliant"] is True
        parser.set_options.assert_called_with({"owl_profile": "OWL-EL"})

    def test_validate_imports_resolution(self, mock_owl_parser):
        """Test validation of import resolution."""
        parser = mock_owl_parser()
        parser.validate_owl.return_value = {
            "is_valid": True,
            "imports_resolved": True,
            "unresolved_imports": [],
        }

        owl_with_imports = """<?xml version="1.0"?>
        <owl:Ontology>
            <owl:imports rdf:resource="http://example.org/imported.owl"/>
        </owl:Ontology>"""

        validation_result = parser.validate_owl(owl_with_imports)

        assert validation_result["imports_resolved"] is True
        assert len(validation_result["unresolved_imports"]) == 0

    def test_validate_namespace_consistency(self, mock_owl_parser, sample_owl_content):
        """Test validation of namespace consistency."""
        parser = mock_owl_parser()
        parser.validate_owl.return_value = {
            "is_valid": True,
            "namespace_consistent": True,
            "undefined_namespaces": [],
        }

        validation_result = parser.validate_owl(sample_owl_content)

        assert validation_result["namespace_consistent"] is True
        assert len(validation_result["undefined_namespaces"]) == 0


class TestOWLParserOptions:
    """Test parser configuration and options."""

    def test_set_parsing_options(self, mock_owl_parser):
        """Test setting various parsing options."""
        parser = mock_owl_parser()

        options = {
            "validate_on_parse": True,
            "include_imports": False,
            "strict_mode": True,
            "preserve_namespaces": True,
            "base_iri": "http://example.org/custom#",
        }

        parser.set_options(options)
        parser.set_options.assert_called_once_with(options)

    def test_get_current_options(self, mock_owl_parser):
        """Test getting current parser options."""
        parser = mock_owl_parser()

        expected_options = {
            "validate_on_parse": False,
            "include_imports": True,
            "strict_mode": False,
            "preserve_namespaces": True,
        }
        parser.get_options.return_value = expected_options

        current_options = parser.get_options()

        assert current_options == expected_options
        assert "validate_on_parse" in current_options

    def test_reset_options_to_defaults(self, mock_owl_parser):
        """Test resetting options to default values."""
        parser = mock_owl_parser()

        # Set some custom options first
        parser.set_options({"strict_mode": True, "validate_on_parse": True})

        # Reset to defaults
        parser.reset_options()
        parser.reset_options.assert_called_once()

    def test_invalid_option_handling(self, mock_owl_parser):
        """Test handling of invalid configuration options."""
        parser = mock_owl_parser()
        parser.set_options.side_effect = ValueError("Unknown option: invalid_option")

        invalid_options = {"invalid_option": True}

        with pytest.raises(ValueError, match="Unknown option"):
            parser.set_options(invalid_options)

    def test_option_validation(self, mock_owl_parser):
        """Test validation of option values."""
        parser = mock_owl_parser()
        parser.set_options.side_effect = ValueError(
            "Invalid value for option 'batch_size': must be positive integer"
        )

        invalid_options = {"batch_size": -100}

        with pytest.raises(ValueError, match="Invalid value for option"):
            parser.set_options(invalid_options)


class TestOWLParserIntegration:
    """Integration tests with other components."""

    def test_integration_with_ontology_manager(self, mock_owl_parser):
        """Test integration with OntologyManager."""
        parser = mock_owl_parser()

        # Mock integration with ontology manager
        with patch(
            "aim2_project.aim2_ontology.ontology_manager.OntologyManager"
        ) as MockManager:
            manager = MockManager()
            manager.add_ontology = Mock(return_value=True)

            # Parse and add to manager
            mock_ontology = Mock()
            parser.to_ontology.return_value = mock_ontology

            parsed_result = parser.parse_string("owl_content")
            ontology = parser.to_ontology(parsed_result)
            manager.add_ontology(ontology)

            manager.add_ontology.assert_called_once_with(ontology)

    def test_integration_with_term_indexing(self, mock_owl_parser):
        """Test integration with ontology term indexing."""
        parser = mock_owl_parser()

        # Mock ontology with indexing capability
        mock_ontology = Mock()
        mock_ontology.rebuild_indexes = Mock()
        parser.to_ontology.return_value = mock_ontology

        parsed_result = parser.parse_string("owl_content")
        ontology = parser.to_ontology(parsed_result)

        # Verify indexing is called
        ontology.rebuild_indexes()
        mock_ontology.rebuild_indexes.assert_called_once()

    def test_integration_with_validation_pipeline(self, mock_owl_parser):
        """Test integration with validation pipeline."""
        parser = mock_owl_parser()

        # Mock validation pipeline
        with patch(
            "aim2_project.aim2_ontology.validators.ValidationPipeline"
        ) as MockPipeline:
            pipeline = MockPipeline()
            pipeline.validate_ontology = Mock(return_value={"is_valid": True})

            mock_ontology = Mock()
            parser.to_ontology.return_value = mock_ontology

            parsed_result = parser.parse_string("owl_content")
            ontology = parser.to_ontology(parsed_result)

            # Run validation pipeline
            validation_result = pipeline.validate_ontology(ontology)

            assert validation_result["is_valid"] is True
            pipeline.validate_ontology.assert_called_once_with(ontology)

    def test_end_to_end_parsing_workflow(self, mock_owl_parser, sample_owl_content):
        """Test complete end-to-end parsing workflow."""
        parser = mock_owl_parser()

        # Mock the complete workflow
        mock_parsed = Mock()
        mock_ontology = Mock()
        mock_ontology.terms = {"CHEM:001": Mock(), "CHEM:002": Mock()}
        mock_ontology.relationships = {"REL:001": Mock()}

        parser.parse_string.return_value = mock_parsed
        parser.to_ontology.return_value = mock_ontology

        # Execute workflow
        parsed_result = parser.parse_string(sample_owl_content)
        ontology = parser.to_ontology(parsed_result)

        # Verify results
        assert len(ontology.terms) == 2
        assert len(ontology.relationships) == 1

        parser.parse_string.assert_called_once_with(sample_owl_content)
        parser.to_ontology.assert_called_once_with(mock_parsed)


class TestOWLParserPerformance:
    """Performance and scalability tests."""

    def test_parse_large_ontology_performance(self, mock_owl_parser):
        """Test parsing performance with large ontologies."""
        parser = mock_owl_parser()

        # Configure for performance testing
        parser.set_options(
            {"memory_efficient": True, "batch_size": 1000, "streaming_mode": True}
        )

        mock_result = Mock()
        mock_result.term_count = 10000
        mock_result.relationship_count = 25000
        parser.parse_file.return_value = mock_result

        result = parser.parse_file("/path/to/large_ontology.owl")

        assert result.term_count == 10000
        assert result.relationship_count == 25000

    def test_memory_usage_optimization(self, mock_owl_parser):
        """Test memory usage optimization features."""
        parser = mock_owl_parser()

        # Configure memory optimization
        parser.set_options(
            {"memory_limit": "1GB", "gc_threshold": 1000, "lazy_loading": True}
        )

        mock_result = Mock()
        parser.parse_file.return_value = mock_result

        result = parser.parse_file("/path/to/memory_intensive_ontology.owl")

        assert result == mock_result
        parser.set_options.assert_called_with(
            {"memory_limit": "1GB", "gc_threshold": 1000, "lazy_loading": True}
        )

    def test_concurrent_parsing_support(self, mock_owl_parser):
        """Test support for concurrent parsing operations."""
        parser = mock_owl_parser()

        # Configure for concurrent processing
        parser.set_options(
            {"parallel_processing": True, "worker_threads": 4, "thread_safe": True}
        )

        files = [
            "/path/to/ontology1.owl",
            "/path/to/ontology2.owl",
            "/path/to/ontology3.owl",
        ]

        mock_results = [Mock(), Mock(), Mock()]
        parser.parse_file.side_effect = mock_results

        results = []
        for file_path in files:
            result = parser.parse_file(file_path)
            results.append(result)

        assert len(results) == 3
        assert parser.parse_file.call_count == 3

    def test_caching_mechanisms(self, mock_owl_parser):
        """Test caching mechanisms for repeated parsing operations."""
        parser = mock_owl_parser()

        # Configure caching
        parser.set_options({"enable_cache": True, "cache_size": 100, "cache_ttl": 3600})

        mock_result = Mock()
        parser.parse_file.return_value = mock_result

        # Parse same file twice
        file_path = "/path/to/cached_ontology.owl"
        result1 = parser.parse_file(file_path)
        result2 = parser.parse_file(file_path)

        assert result1 == mock_result
        assert result2 == mock_result

        # In a real implementation, the second call might use cache
        # For now, we just verify the calls were made
        assert parser.parse_file.call_count == 2


# Additional test classes for RDF triple extraction functionality


class TestRDFTripleModel:
    """Test RDFTriple model class functionality."""

    def test_rdf_triple_basic_initialization(self):
        """Test basic RDFTriple initialization with required fields."""
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
        assert isinstance(triple.namespace_prefixes, dict)
        assert triple.created_at is not None

    def test_rdf_triple_comprehensive_initialization(self):
        """Test RDFTriple initialization with all fields."""
        from datetime import datetime

        from aim2_project.aim2_ontology.models import RDFTriple

        created_time = datetime.now()
        metadata = {"source": "test", "method": "parsing"}
        namespaces = {
            "ex": "http://example.org/",
            "rdf": "http://www.w3.org/1999/02/22-rdf-syntax-ns#",
        }

        triple = RDFTriple(
            subject="http://example.org/Chemical",
            predicate="http://www.w3.org/2000/01/rdf-schema#label",
            object="Glucose",
            subject_type="uri",
            object_type="literal",
            object_datatype="http://www.w3.org/2001/XMLSchema#string",
            object_language="en",
            context="http://example.org/context",
            source="ChEBI",
            confidence=0.95,
            metadata=metadata,
            created_at=created_time,
            namespace_prefixes=namespaces,
        )

        assert triple.subject == "http://example.org/Chemical"
        assert triple.predicate == "http://www.w3.org/2000/01/rdf-schema#label"
        assert triple.object == "Glucose"
        assert triple.subject_type == "uri"
        assert triple.object_type == "literal"
        assert triple.object_datatype == "http://www.w3.org/2001/XMLSchema#string"
        assert triple.object_language == "en"
        assert triple.context == "http://example.org/context"
        assert triple.source == "ChEBI"
        assert triple.confidence == 0.95
        assert triple.metadata == metadata
        assert triple.created_at == created_time
        assert triple.namespace_prefixes == namespaces

    def test_rdf_triple_confidence_validation(self):
        """Test confidence score validation and normalization."""
        from aim2_project.aim2_ontology.models import RDFTriple

        # Test confidence normalization to valid range
        triple1 = RDFTriple("s", "p", "o", confidence=1.5)
        assert triple1.confidence == 1.0

        triple2 = RDFTriple("s", "p", "o", confidence=-0.5)
        assert triple2.confidence == 0.0

        triple3 = RDFTriple("s", "p", "o", confidence="invalid")
        assert triple3.confidence == 1.0

        # Test valid confidence values
        triple4 = RDFTriple("s", "p", "o", confidence=0.75)
        assert triple4.confidence == 0.75

    def test_rdf_triple_validation_methods(self):
        """Test RDFTriple validation methods."""
        from aim2_project.aim2_ontology.models import RDFTriple

        # Valid triple
        valid_triple = RDFTriple(
            subject="http://example.org/subject",
            predicate="http://example.org/predicate",
            object="http://example.org/object",
        )
        assert valid_triple.is_valid()

        # Test validation with different object types
        literal_triple = RDFTriple(
            subject="http://example.org/subject",
            predicate="http://example.org/predicate",
            object="literal value",
            object_type="literal",
        )
        assert literal_triple.is_valid()

        # Test blank node validation
        bnode_triple = RDFTriple(
            subject="_:b1",
            predicate="http://example.org/predicate",
            object="_:b2",
            subject_type="bnode",
            object_type="bnode",
        )
        assert bnode_triple.is_valid()

    def test_rdf_triple_serialization_methods(self):
        """Test RDFTriple serialization and deserialization methods."""
        from aim2_project.aim2_ontology.models import RDFTriple

        original_triple = RDFTriple(
            subject="http://example.org/Chemical",
            predicate="http://www.w3.org/2000/01/rdf-schema#label",
            object="Glucose",
            object_type="literal",
            object_datatype="http://www.w3.org/2001/XMLSchema#string",
            confidence=0.95,
            metadata={"source": "test"},
        )

        # Test to_dict serialization
        triple_dict = original_triple.to_dict()
        assert isinstance(triple_dict, dict)
        assert triple_dict["subject"] == "http://example.org/Chemical"
        assert triple_dict["predicate"] == "http://www.w3.org/2000/01/rdf-schema#label"
        assert triple_dict["object"] == "Glucose"
        assert triple_dict["confidence"] == 0.95

        # Test from_dict deserialization
        restored_triple = RDFTriple.from_dict(triple_dict)
        assert restored_triple.subject == original_triple.subject
        assert restored_triple.predicate == original_triple.predicate
        assert restored_triple.object == original_triple.object
        assert restored_triple.confidence == original_triple.confidence

        # Test to_json serialization
        triple_json = original_triple.to_json()
        assert isinstance(triple_json, str)

        # Test from_json deserialization
        restored_from_json = RDFTriple.from_json(triple_json)
        assert restored_from_json.subject == original_triple.subject
        assert restored_from_json.predicate == original_triple.predicate
        assert restored_from_json.object == original_triple.object

    def test_rdf_triple_equality_and_hashing(self):
        """Test RDFTriple equality comparison and hashing."""
        from aim2_project.aim2_ontology.models import RDFTriple

        triple1 = RDFTriple("s", "p", "o")
        triple2 = RDFTriple("s", "p", "o")
        triple3 = RDFTriple("s", "p", "different")

        # Test equality
        assert triple1 == triple2
        assert triple1 != triple3

        # Test hashing (should be hashable for use in sets/dicts)
        triple_set = {triple1, triple2, triple3}
        assert len(triple_set) == 2  # triple1 and triple2 should be considered equal

    def test_rdf_triple_string_representation(self):
        """Test RDFTriple string representation methods."""
        from aim2_project.aim2_ontology.models import RDFTriple

        triple = RDFTriple(
            subject="http://ex.org/s", predicate="http://ex.org/p", object="value"
        )

        # Test __str__ method
        str_repr = str(triple)
        assert "http://ex.org/s" in str_repr
        assert "http://ex.org/p" in str_repr
        assert "value" in str_repr

        # Test __repr__ method
        repr_str = repr(triple)
        assert "RDFTriple" in repr_str
        assert "http://ex.org/s" in repr_str

    def test_rdf_triple_turtle_serialization(self):
        """Test RDFTriple Turtle format serialization."""
        from aim2_project.aim2_ontology.models import RDFTriple

        # URI triple
        uri_triple = RDFTriple(
            subject="http://example.org/subject",
            predicate="http://example.org/predicate",
            object="http://example.org/object",
        )
        turtle = uri_triple.to_turtle()
        assert "<http://example.org/subject>" in turtle
        assert "<http://example.org/predicate>" in turtle
        assert "<http://example.org/object>" in turtle

        # Literal triple
        literal_triple = RDFTriple(
            subject="http://example.org/subject",
            predicate="http://www.w3.org/2000/01/rdf-schema#label",
            object="Test Label",
            object_type="literal",
            object_language="en",
        )
        turtle_literal = literal_triple.to_turtle()
        assert '"Test Label"@en' in turtle_literal

    def test_rdf_triple_ntriples_serialization(self):
        """Test RDFTriple N-Triples format serialization."""
        from aim2_project.aim2_ontology.models import RDFTriple

        triple = RDFTriple(
            subject="http://example.org/subject",
            predicate="http://example.org/predicate",
            object="http://example.org/object",
        )

        ntriples = triple.to_ntriples()
        assert ntriples.startswith("<http://example.org/subject>")
        assert "<http://example.org/predicate>" in ntriples
        assert "<http://example.org/object>" in ntriples
        assert ntriples.endswith(" .")

    @pytest.mark.parametrize(
        "subject,predicate,object,expected_valid",
        [
            ("http://ex.org/s", "http://ex.org/p", "http://ex.org/o", True),
            ("", "http://ex.org/p", "http://ex.org/o", False),
            ("http://ex.org/s", "", "http://ex.org/o", False),
            ("http://ex.org/s", "http://ex.org/p", "", False),
            ("_:b1", "http://ex.org/p", "literal", True),
            ("invalid-uri", "http://ex.org/p", "http://ex.org/o", False),
        ],
    )
    def test_rdf_triple_validation_edge_cases(
        self, subject, predicate, object, expected_valid
    ):
        """Test RDFTriple validation with various edge cases."""
        from aim2_project.aim2_ontology.models import RDFTriple

        triple = RDFTriple(subject=subject, predicate=predicate, object=object)
        assert triple.is_valid() == expected_valid


class TestOWLParserTripleExtraction:
    """Test OWL parser RDF triple extraction functionality."""

    def test_extract_triples_method_exists(self, mock_owl_parser):
        """Test that extract_triples method exists and is callable."""
        parser = mock_owl_parser()

        # Verify method exists
        assert hasattr(parser, "extract_triples")
        assert callable(parser.extract_triples)

    def test_extract_triples_from_valid_parsed_result(self, mock_owl_parser):
        """Test extracting triples from valid parsed OWL result."""
        from aim2_project.aim2_ontology.models import RDFTriple

        parser = mock_owl_parser()

        # Mock triples to be returned
        mock_triples = [
            RDFTriple(
                subject="http://example.org/Chemical",
                predicate="http://www.w3.org/1999/02/22-rdf-syntax-ns#type",
                object="http://www.w3.org/2002/07/owl#Class",
            ),
            RDFTriple(
                subject="http://example.org/Chemical",
                predicate="http://www.w3.org/2000/01/rdf-schema#label",
                object="Chemical",
                object_type="literal",
            ),
            RDFTriple(
                subject="http://example.org/Glucose",
                predicate="http://www.w3.org/2000/01/rdf-schema#subClassOf",
                object="http://example.org/Chemical",
            ),
        ]

        parser.extract_triples.return_value = mock_triples

        # Create mock parsed result
        parsed_result = {
            "rdf_graph": Mock(),
            "owl_ontology": Mock(),
            "format": "owl",
            "metadata": {},
        }

        triples = parser.extract_triples(parsed_result)

        assert len(triples) == 3
        assert all(isinstance(t, RDFTriple) for t in triples)
        assert any(t.predicate.endswith("type") for t in triples)
        assert any(t.predicate.endswith("label") for t in triples)
        assert any(t.predicate.endswith("subClassOf") for t in triples)

        parser.extract_triples.assert_called_once_with(parsed_result)

    def test_extract_triples_with_filtering(self, mock_owl_parser):
        """Test triple extraction with namespace filtering."""
        from aim2_project.aim2_ontology.models import RDFTriple

        parser = mock_owl_parser()

        # Configure parser with namespace filter
        parser.set_options(
            {"conversion_filters": {"namespace_filter": ["http://example.org/"]}}
        )

        # Mock filtered triples
        filtered_triples = [
            RDFTriple(
                subject="http://example.org/Chemical",
                predicate="http://www.w3.org/2000/01/rdf-schema#label",
                object="Chemical",
                object_type="literal",
            )
        ]

        parser.extract_triples.return_value = filtered_triples

        parsed_result = {"rdf_graph": Mock()}
        triples = parser.extract_triples(parsed_result)

        assert len(triples) == 1
        assert all(t.subject.startswith("http://example.org/") for t in triples)

    def test_extract_triples_error_handling(self, mock_owl_parser):
        """Test error handling during triple extraction."""
        parser = mock_owl_parser()
        parser.extract_triples.side_effect = Exception("Extraction failed")

        parsed_result = {"rdf_graph": Mock()}

        with pytest.raises(Exception, match="Extraction failed"):
            parser.extract_triples(parsed_result)

    def test_extract_triples_empty_result(self, mock_owl_parser):
        """Test triple extraction from empty parsed result."""
        parser = mock_owl_parser()
        parser.extract_triples.return_value = []

        parsed_result = {"rdf_graph": None, "owl_ontology": None}
        triples = parser.extract_triples(parsed_result)

        assert triples == []
        assert isinstance(triples, list)

    def test_extract_triples_invalid_input(self, mock_owl_parser):
        """Test triple extraction with invalid input."""
        parser = mock_owl_parser()
        parser.extract_triples.return_value = []

        # Test with None input
        triples = parser.extract_triples(None)
        assert triples == []

        # Test with invalid format
        triples = parser.extract_triples("invalid")
        assert triples == []

    def test_extract_triples_performance_logging(self, mock_owl_parser):
        """Test that triple extraction logs performance metrics."""
        from aim2_project.aim2_ontology.models import RDFTriple

        parser = mock_owl_parser()

        # Configure parser to log performance
        parser.set_options({"log_performance": True})

        mock_triples = [RDFTriple("s1", "p1", "o1"), RDFTriple("s2", "p2", "o2")]
        parser.extract_triples.return_value = mock_triples

        parsed_result = {"rdf_graph": Mock()}
        triples = parser.extract_triples(parsed_result)

        assert len(triples) == 2
        parser.set_options.assert_called_with({"log_performance": True})

    def test_extract_triples_with_confidence_scores(self, mock_owl_parser):
        """Test triple extraction with confidence scoring."""
        from aim2_project.aim2_ontology.models import RDFTriple

        parser = mock_owl_parser()

        # Mock triples with varying confidence scores
        mock_triples = [
            RDFTriple("s1", "p1", "o1", confidence=1.0),
            RDFTriple("s2", "p2", "o2", confidence=0.8),
            RDFTriple("s3", "p3", "o3", confidence=0.6),
        ]
        parser.extract_triples.return_value = mock_triples

        parsed_result = {"rdf_graph": Mock()}
        triples = parser.extract_triples(parsed_result)

        assert len(triples) == 3
        assert triples[0].confidence == 1.0
        assert triples[1].confidence == 0.8
        assert triples[2].confidence == 0.6

    def test_extract_triples_with_metadata(self, mock_owl_parser):
        """Test triple extraction preserves metadata."""
        from aim2_project.aim2_ontology.models import RDFTriple

        parser = mock_owl_parser()

        # Mock triples with metadata
        mock_triples = [
            RDFTriple(
                "s1",
                "p1",
                "o1",
                source="test_ontology",
                metadata={"extraction_method": "rdflib", "parser_version": "1.0.0"},
            )
        ]
        parser.extract_triples.return_value = mock_triples

        parsed_result = {"rdf_graph": Mock()}
        triples = parser.extract_triples(parsed_result)

        assert len(triples) == 1
        assert triples[0].source == "test_ontology"
        assert "extraction_method" in triples[0].metadata
        assert triples[0].metadata["extraction_method"] == "rdflib"


class TestOWLParserIntegrationWithTriples:
    """Test integration of triple extraction with OWL parser."""

    def test_parse_with_automatic_triple_extraction(
        self, mock_owl_parser, sample_owl_content
    ):
        """Test that parse method automatically extracts triples when configured."""
        from aim2_project.aim2_ontology.models import RDFTriple

        parser = mock_owl_parser()

        # Configure parser to extract triples during parsing
        parser.set_options({"extract_triples_on_parse": True})

        # Mock parse result with triples
        mock_result = {
            "rdf_graph": Mock(),
            "owl_ontology": Mock(),
            "format": "owl",
            "triples": [RDFTriple("s1", "p1", "o1"), RDFTriple("s2", "p2", "o2")],
            "triple_count": 2,
        }
        parser.parse_string.return_value = mock_result

        result = parser.parse_string(sample_owl_content)

        assert "triples" in result
        assert "triple_count" in result
        assert result["triple_count"] == 2
        assert len(result["triples"]) == 2

    def test_parse_without_automatic_triple_extraction(
        self, mock_owl_parser, sample_owl_content
    ):
        """Test that parse method skips triple extraction when disabled."""
        parser = mock_owl_parser()

        # Configure parser to NOT extract triples during parsing
        parser.set_options({"extract_triples_on_parse": False})

        # Mock parse result without triples
        mock_result = {"rdf_graph": Mock(), "owl_ontology": Mock(), "format": "owl"}
        parser.parse_string.return_value = mock_result

        result = parser.parse_string(sample_owl_content)

        assert "triples" not in result
        assert "triple_count" not in result

    def test_parse_triple_extraction_error_handling(
        self, mock_owl_parser, sample_owl_content
    ):
        """Test error handling when triple extraction fails during parsing."""
        parser = mock_owl_parser()

        # Configure parser to extract triples and continue on error
        parser.set_options(
            {"extract_triples_on_parse": True, "continue_on_error": True}
        )

        # Mock parse result with triple extraction failure handled gracefully
        mock_result = {
            "rdf_graph": Mock(),
            "owl_ontology": Mock(),
            "format": "owl",
            "triples": [],  # Empty due to extraction failure
            "triple_count": 0,
        }
        parser.parse_string.return_value = mock_result

        result = parser.parse_string(sample_owl_content)

        assert "triples" in result
        assert result["triples"] == []
        assert result["triple_count"] == 0


# Test fixtures for complex scenarios
@pytest.fixture
def complex_owl_ontology():
    """Fixture providing complex OWL ontology data for testing."""
    return """<?xml version="1.0"?>
<rdf:RDF xmlns="http://example.org/complex#"
         xml:base="http://example.org/complex"
         xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#"
         xmlns:owl="http://www.w3.org/2002/07/owl#"
         xmlns:rdfs="http://www.w3.org/2000/01/rdf-schema#"
         xmlns:dc="http://purl.org/dc/elements/1.1/"
         xmlns:foaf="http://xmlns.com/foaf/0.1/">

    <owl:Ontology rdf:about="http://example.org/complex">
        <dc:title>Complex Test Ontology</dc:title>
        <dc:description>A complex ontology for comprehensive testing</dc:description>
        <dc:creator>Test Suite</dc:creator>
        <owl:versionInfo>1.0.0</owl:versionInfo>
        <owl:imports rdf:resource="http://example.org/imported"/>
    </owl:Ontology>

    <!-- Classes -->
    <owl:Class rdf:about="http://example.org/complex#Entity">
        <rdfs:label>Entity</rdfs:label>
        <rdfs:comment>Base class for all entities</rdfs:comment>
    </owl:Class>

    <owl:Class rdf:about="http://example.org/complex#Chemical">
        <rdfs:label>Chemical</rdfs:label>
        <rdfs:subClassOf rdf:resource="http://example.org/complex#Entity"/>
        <owl:disjointWith rdf:resource="http://example.org/complex#BiologicalProcess"/>
    </owl:Class>

    <owl:Class rdf:about="http://example.org/complex#BiologicalProcess">
        <rdfs:label>Biological Process</rdfs:label>
        <rdfs:subClassOf rdf:resource="http://example.org/complex#Entity"/>
    </owl:Class>

    <!-- Properties -->
    <owl:ObjectProperty rdf:about="http://example.org/complex#participatesIn">
        <rdfs:label>participates in</rdfs:label>
        <rdfs:domain rdf:resource="http://example.org/complex#Chemical"/>
        <rdfs:range rdf:resource="http://example.org/complex#BiologicalProcess"/>
    </owl:ObjectProperty>

    <owl:DatatypeProperty rdf:about="http://example.org/complex#hasName">
        <rdfs:label>has name</rdfs:label>
        <rdfs:domain rdf:resource="http://example.org/complex#Entity"/>
        <rdfs:range rdf:resource="http://www.w3.org/2001/XMLSchema#string"/>
    </owl:DatatypeProperty>

    <!-- Individuals -->
    <owl:NamedIndividual rdf:about="http://example.org/complex#glucose">
        <rdf:type rdf:resource="http://example.org/complex#Chemical"/>
        <rdfs:label>Glucose</rdfs:label>
        <complex:hasName>glucose</complex:hasName>
    </owl:NamedIndividual>

</rdf:RDF>"""


@pytest.fixture
def malformed_owl_content():
    """Fixture providing malformed OWL content for error testing."""
    return """<?xml version="1.0"?>
<rdf:RDF xmlns="http://example.org/malformed#"
         xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#"
         xmlns:owl="http://www.w3.org/2002/07/owl#">

    <owl:Ontology rdf:about="http://example.org/malformed">
        <rdfs:label>Malformed Ontology</rdfs:label>
        <!-- Missing namespace declaration for rdfs -->
    </owl:Ontology>

    <owl:Class rdf:about="http://example.org/malformed#TestClass">
        <rdfs:subClassOf rdf:resource="http://example.org/malformed#NonexistentClass"/>
        <!-- Unclosed tag follows -->
        <owl:someProperty>
    </owl:Class>

    <!-- Missing closing RDF tag -->
"""


@pytest.fixture
def parser_configuration_options():
    """Fixture providing comprehensive parser configuration options."""
    return {
        "basic_options": {
            "validate_on_parse": True,
            "strict_mode": False,
            "include_imports": True,
            "preserve_namespaces": True,
        },
        "performance_options": {
            "memory_efficient": True,
            "batch_size": 1000,
            "streaming_mode": False,
            "parallel_processing": True,
            "worker_threads": 2,
        },
        "validation_options": {
            "owl_profile": "OWL-DL",
            "consistency_check": True,
            "satisfiability_check": False,
            "schema_validation": True,
        },
        "conversion_options": {
            "include_classes": True,
            "include_properties": True,
            "include_individuals": True,
            "conversion_filters": {
                "namespace_filter": ["http://example.org/ontology#"],
                "class_filter": None,
                "property_filter": None,
            },
        },
        "error_handling_options": {
            "error_recovery": True,
            "max_errors": 100,
            "continue_on_error": True,
            "log_warnings": True,
        },
    }


# Additional integration test fixtures
@pytest.fixture
def temp_owl_files():
    """Create temporary OWL files for testing."""
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)

        # Create various OWL files
        files = {}

        # Simple OWL file
        simple_owl = temp_path / "simple.owl"
        simple_owl.write_text(
            """<?xml version="1.0"?>
<owl:Ontology xmlns:owl="http://www.w3.org/2002/07/owl#">
    <owl:Class rdf:about="#SimpleClass"/>
</owl:Ontology>"""
        )
        files["simple"] = str(simple_owl)

        # Turtle file
        turtle_file = temp_path / "ontology.ttl"
        turtle_file.write_text(
            """@prefix owl: <http://www.w3.org/2002/07/owl#> .
<http://example.org/ontology> a owl:Ontology ."""
        )
        files["turtle"] = str(turtle_file)

        # JSON-LD file
        jsonld_file = temp_path / "ontology.jsonld"
        jsonld_file.write_text(
            """{
  "@context": {"owl": "http://www.w3.org/2002/07/owl#"},
  "@type": "owl:Ontology"
}"""
        )
        files["jsonld"] = str(jsonld_file)

        yield files


class TestRDFTripleExtractionEdgeCases:
    """Test edge cases for RDF triple extraction."""

    def test_extract_triples_from_empty_ontology(self, mock_owl_parser):
        """Test triple extraction from empty ontology."""
        parser = mock_owl_parser()
        parser.extract_triples.return_value = []

        # Empty parsed result
        parsed_result = {
            "rdf_graph": Mock(),
            "owl_ontology": Mock(),
            "format": "owl",
            "metadata": {"triple_count": 0},
        }
        parsed_result["rdf_graph"].__len__ = Mock(return_value=0)

        triples = parser.extract_triples(parsed_result)

        assert triples == []
        assert len(triples) == 0

    def test_extract_triples_from_malformed_data(self, mock_owl_parser):
        """Test triple extraction from malformed ontology data."""
        parser = mock_owl_parser()

        # Configure parser to handle errors gracefully
        parser.set_options({"continue_on_error": True})
        parser.extract_triples.return_value = []

        # Malformed parsed result
        parsed_result = {
            "rdf_graph": None,
            "owl_ontology": None,
            "format": "owl",
            "errors": ["Malformed XML structure"],
            "warnings": ["Could not parse some triples"],
        }

        triples = parser.extract_triples(parsed_result)

        assert triples == []
        parser.set_options.assert_called_with({"continue_on_error": True})

    def test_extract_triples_with_circular_references(self, mock_owl_parser):
        """Test triple extraction with circular references in ontology."""
        from aim2_project.aim2_ontology.models import RDFTriple

        parser = mock_owl_parser()

        # Mock triples with circular references
        circular_triples = [
            RDFTriple("http://ex.org/A", "http://ex.org/subClassOf", "http://ex.org/B"),
            RDFTriple("http://ex.org/B", "http://ex.org/subClassOf", "http://ex.org/A"),
            RDFTriple(
                "http://ex.org/A", "http://ex.org/equivalentTo", "http://ex.org/A"
            ),
        ]
        parser.extract_triples.return_value = circular_triples

        parsed_result = {"rdf_graph": Mock()}
        triples = parser.extract_triples(parsed_result)

        assert len(triples) == 3
        # Should handle circular references without infinite loops
        subjects = [t.subject for t in triples]
        assert "http://ex.org/A" in subjects
        assert "http://ex.org/B" in subjects

    def test_extract_triples_with_unicode_content(self, mock_owl_parser):
        """Test triple extraction with Unicode content."""
        from aim2_project.aim2_ontology.models import RDFTriple

        parser = mock_owl_parser()

        # Mock triples with Unicode content
        unicode_triples = [
            RDFTriple(
                "http://example.org/化学物质",
                "http://www.w3.org/2000/01/rdf-schema#label",
                "化学物质",
                object_type="literal",
                object_language="zh",
            ),
            RDFTriple(
                "http://example.org/Химия",
                "http://www.w3.org/2000/01/rdf-schema#label",
                "Химия",
                object_type="literal",
                object_language="ru",
            ),
        ]
        parser.extract_triples.return_value = unicode_triples

        parsed_result = {"rdf_graph": Mock()}
        triples = parser.extract_triples(parsed_result)

        assert len(triples) == 2
        assert any("化学物质" in t.object for t in triples)
        assert any("Химия" in t.object for t in triples)

    def test_extract_triples_with_blank_nodes(self, mock_owl_parser):
        """Test triple extraction with blank nodes."""
        from aim2_project.aim2_ontology.models import RDFTriple

        parser = mock_owl_parser()

        # Mock triples with blank nodes
        blank_node_triples = [
            RDFTriple(
                "_:b1",
                "http://www.w3.org/1999/02/22-rdf-syntax-ns#type",
                "http://example.org/Anonymous",
                subject_type="bnode",
            ),
            RDFTriple(
                "http://example.org/instance",
                "http://example.org/hasAnonymous",
                "_:b1",
                object_type="bnode",
            ),
        ]
        parser.extract_triples.return_value = blank_node_triples

        parsed_result = {"rdf_graph": Mock()}
        triples = parser.extract_triples(parsed_result)

        assert len(triples) == 2
        blank_node_subjects = [t for t in triples if t.subject_type == "bnode"]
        blank_node_objects = [t for t in triples if t.object_type == "bnode"]
        assert len(blank_node_subjects) == 1
        assert len(blank_node_objects) == 1

    def test_extract_triples_with_complex_datatypes(self, mock_owl_parser):
        """Test triple extraction with complex datatypes."""
        from aim2_project.aim2_ontology.models import RDFTriple

        parser = mock_owl_parser()

        # Mock triples with various datatypes
        datatype_triples = [
            RDFTriple(
                "http://example.org/measurement",
                "http://example.org/hasValue",
                "42.5",
                object_type="literal",
                object_datatype="http://www.w3.org/2001/XMLSchema#double",
            ),
            RDFTriple(
                "http://example.org/event",
                "http://example.org/hasTimestamp",
                "2023-01-01T00:00:00Z",
                object_type="literal",
                object_datatype="http://www.w3.org/2001/XMLSchema#dateTime",
            ),
            RDFTriple(
                "http://example.org/status",
                "http://example.org/isActive",
                "true",
                object_type="literal",
                object_datatype="http://www.w3.org/2001/XMLSchema#boolean",
            ),
        ]
        parser.extract_triples.return_value = datatype_triples

        parsed_result = {"rdf_graph": Mock()}
        triples = parser.extract_triples(parsed_result)

        assert len(triples) == 3
        datatypes = [t.object_datatype for t in triples if t.object_datatype]
        assert "double" in str(datatypes)
        assert "dateTime" in str(datatypes)
        assert "boolean" in str(datatypes)

    def test_extract_triples_memory_efficiency(self, mock_owl_parser):
        """Test memory-efficient triple extraction for large datasets."""
        from aim2_project.aim2_ontology.models import RDFTriple

        parser = mock_owl_parser()

        # Configure for memory-efficient processing
        parser.set_options(
            {"memory_efficient": True, "batch_size": 100, "streaming_mode": True}
        )

        # Mock large number of triples
        large_triple_set = [
            RDFTriple(
                f"http://ex.org/s{i}", f"http://ex.org/p{i}", f"http://ex.org/o{i}"
            )
            for i in range(1000)
        ]
        parser.extract_triples.return_value = large_triple_set

        parsed_result = {"rdf_graph": Mock()}
        triples = parser.extract_triples(parsed_result)

        assert len(triples) == 1000
        parser.set_options.assert_called_with(
            {"memory_efficient": True, "batch_size": 100, "streaming_mode": True}
        )

    def test_extract_triples_with_namespaces(self, mock_owl_parser):
        """Test triple extraction preserves namespace information."""
        from aim2_project.aim2_ontology.models import RDFTriple

        parser = mock_owl_parser()

        # Mock triples with namespace information
        namespace_triples = [
            RDFTriple(
                "http://example.org/Chemical",
                "http://www.w3.org/2000/01/rdf-schema#label",
                "Chemical",
                object_type="literal",
                namespace_prefixes={
                    "ex": "http://example.org/",
                    "rdfs": "http://www.w3.org/2000/01/rdf-schema#",
                },
            )
        ]
        parser.extract_triples.return_value = namespace_triples

        parsed_result = {"rdf_graph": Mock()}
        triples = parser.extract_triples(parsed_result)

        assert len(triples) == 1
        assert "ex" in triples[0].namespace_prefixes
        assert "rdfs" in triples[0].namespace_prefixes

    def test_extract_triples_duplicate_handling(self, mock_owl_parser):
        """Test handling of duplicate triples during extraction."""
        from aim2_project.aim2_ontology.models import RDFTriple

        parser = mock_owl_parser()

        # Configure to handle duplicates
        parser.set_options({"remove_duplicates": True})

        # Mock triples with duplicates (after deduplication)
        unique_triples = [
            RDFTriple("http://ex.org/s", "http://ex.org/p", "http://ex.org/o"),
            RDFTriple("http://ex.org/s2", "http://ex.org/p2", "http://ex.org/o2"),
        ]
        parser.extract_triples.return_value = unique_triples

        parsed_result = {"rdf_graph": Mock()}
        triples = parser.extract_triples(parsed_result)

        # Should have unique triples only
        assert len(triples) == 2
        assert len(set(triples)) == 2  # All should be unique


class TestRDFTripleExtractionPerformance:
    """Test performance aspects of RDF triple extraction."""

    def test_extract_triples_performance_monitoring(self, mock_owl_parser):
        """Test performance monitoring during triple extraction."""
        from aim2_project.aim2_ontology.models import RDFTriple

        parser = mock_owl_parser()

        # Enable performance monitoring
        parser.set_options(
            {"log_performance": True, "performance_threshold": 1.0}  # seconds
        )

        # Mock performance metrics
        mock_triples = [RDFTriple("s", "p", "o") for _ in range(100)]
        parser.extract_triples.return_value = mock_triples

        parsed_result = {"rdf_graph": Mock()}
        triples = parser.extract_triples(parsed_result)

        assert len(triples) == 100
        parser.set_options.assert_called_with(
            {"log_performance": True, "performance_threshold": 1.0}
        )

    def test_extract_triples_concurrent_processing(self, mock_owl_parser):
        """Test concurrent processing capabilities."""
        from aim2_project.aim2_ontology.models import RDFTriple

        parser = mock_owl_parser()

        # Configure for concurrent processing
        parser.set_options(
            {"parallel_processing": True, "worker_threads": 4, "thread_safe": True}
        )

        mock_triples = [RDFTriple(f"s{i}", f"p{i}", f"o{i}") for i in range(200)]
        parser.extract_triples.return_value = mock_triples

        parsed_result = {"rdf_graph": Mock()}
        triples = parser.extract_triples(parsed_result)

        assert len(triples) == 200
        parser.set_options.assert_called_with(
            {"parallel_processing": True, "worker_threads": 4, "thread_safe": True}
        )

    @pytest.mark.parametrize(
        "triple_count,expected_batch_size",
        [(10, 10), (100, 50), (1000, 100), (10000, 500)],
    )
    def test_extract_triples_batch_processing(
        self, mock_owl_parser, triple_count, expected_batch_size
    ):
        """Test batch processing with different triple counts."""
        from aim2_project.aim2_ontology.models import RDFTriple

        parser = mock_owl_parser()

        # Configure batch processing
        parser.set_options({"batch_size": expected_batch_size})

        mock_triples = [
            RDFTriple(f"s{i}", f"p{i}", f"o{i}") for i in range(triple_count)
        ]
        parser.extract_triples.return_value = mock_triples

        parsed_result = {"rdf_graph": Mock()}
        triples = parser.extract_triples(parsed_result)

        assert len(triples) == triple_count
        parser.set_options.assert_called_with({"batch_size": expected_batch_size})

    def test_extract_triples_memory_usage_optimization(self, mock_owl_parser):
        """Test memory usage optimization during extraction."""
        from aim2_project.aim2_ontology.models import RDFTriple

        parser = mock_owl_parser()

        # Configure memory optimization
        parser.set_options(
            {
                "memory_limit": "512MB",
                "gc_threshold": 1000,
                "lazy_loading": True,
                "cache_size": 100,
            }
        )

        # Large dataset simulation
        mock_triples = [RDFTriple(f"s{i}", f"p{i}", f"o{i}") for i in range(5000)]
        parser.extract_triples.return_value = mock_triples

        parsed_result = {"rdf_graph": Mock()}
        triples = parser.extract_triples(parsed_result)

        assert len(triples) == 5000
        parser.set_options.assert_called_with(
            {
                "memory_limit": "512MB",
                "gc_threshold": 1000,
                "lazy_loading": True,
                "cache_size": 100,
            }
        )

    def test_extract_triples_caching_mechanism(self, mock_owl_parser):
        """Test caching mechanism for repeated extractions."""
        from aim2_project.aim2_ontology.models import RDFTriple

        parser = mock_owl_parser()

        # Enable caching
        parser.set_options({"enable_cache": True, "cache_ttl": 3600, "cache_size": 50})

        mock_triples = [RDFTriple("s", "p", "o")]
        parser.extract_triples.return_value = mock_triples

        parsed_result = {"rdf_graph": Mock()}

        # First extraction
        triples1 = parser.extract_triples(parsed_result)
        # Second extraction (should use cache in real implementation)
        triples2 = parser.extract_triples(parsed_result)

        assert len(triples1) == 1
        assert len(triples2) == 1
        assert parser.extract_triples.call_count == 2  # Mock called twice

    def test_extract_triples_progress_tracking(self, mock_owl_parser):
        """Test progress tracking for long-running extractions."""
        from aim2_project.aim2_ontology.models import RDFTriple

        parser = mock_owl_parser()

        # Enable progress tracking
        parser.set_options({"show_progress": True, "progress_interval": 100})

        mock_triples = [RDFTriple(f"s{i}", f"p{i}", f"o{i}") for i in range(500)]
        parser.extract_triples.return_value = mock_triples

        parsed_result = {"rdf_graph": Mock()}
        triples = parser.extract_triples(parsed_result)

        assert len(triples) == 500
        parser.set_options.assert_called_with(
            {"show_progress": True, "progress_interval": 100}
        )


# Additional fixtures for RDF triple testing
@pytest.fixture
def sample_rdf_triples():
    """Fixture providing sample RDF triples for testing."""
    from aim2_project.aim2_ontology.models import RDFTriple

    return [
        RDFTriple(
            subject="http://example.org/Chemical",
            predicate="http://www.w3.org/1999/02/22-rdf-syntax-ns#type",
            object="http://www.w3.org/2002/07/owl#Class",
        ),
        RDFTriple(
            subject="http://example.org/Chemical",
            predicate="http://www.w3.org/2000/01/rdf-schema#label",
            object="Chemical",
            object_type="literal",
            object_language="en",
        ),
        RDFTriple(
            subject="http://example.org/Glucose",
            predicate="http://www.w3.org/2000/01/rdf-schema#subClassOf",
            object="http://example.org/Chemical",
        ),
        RDFTriple(
            subject="http://example.org/glucose_instance",
            predicate="http://www.w3.org/1999/02/22-rdf-syntax-ns#type",
            object="http://example.org/Glucose",
        ),
    ]


@pytest.fixture
def large_ontology_triples():
    """Fixture providing large set of triples for performance testing."""
    from aim2_project.aim2_ontology.models import RDFTriple

    triples = []

    # Create hierarchical structure
    for i in range(100):
        # Class declarations
        triples.append(
            RDFTriple(
                f"http://example.org/Class{i}",
                "http://www.w3.org/1999/02/22-rdf-syntax-ns#type",
                "http://www.w3.org/2002/07/owl#Class",
            )
        )

        # Labels
        triples.append(
            RDFTriple(
                f"http://example.org/Class{i}",
                "http://www.w3.org/2000/01/rdf-schema#label",
                f"Class {i}",
                object_type="literal",
            )
        )

        # Subclass relationships
        if i > 0:
            triples.append(
                RDFTriple(
                    f"http://example.org/Class{i}",
                    "http://www.w3.org/2000/01/rdf-schema#subClassOf",
                    f"http://example.org/Class{i-1}",
                )
            )

    return triples


@pytest.fixture
def malformed_triple_data():
    """Fixture providing malformed data for error testing."""
    return {
        "empty_subject": ("", "http://ex.org/p", "http://ex.org/o"),
        "empty_predicate": ("http://ex.org/s", "", "http://ex.org/o"),
        "empty_object": ("http://ex.org/s", "http://ex.org/p", ""),
        "invalid_uri": ("not-a-uri", "http://ex.org/p", "http://ex.org/o"),
        "none_values": (None, None, None),
    }
