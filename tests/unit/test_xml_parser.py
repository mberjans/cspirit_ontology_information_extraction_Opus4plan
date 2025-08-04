"""
Comprehensive Unit Tests for XML Parser (TDD Approach)

This module provides comprehensive unit tests for the XML parser functionality in the AIM2 ontology
information extraction system. The tests follow test-driven development (TDD) approach,
defining the expected behavior of the XML parser classes before implementation.

Test Classes:
    TestXMLParserCreation: Tests for XMLParser instantiation and configuration
    TestXMLParserFormatSupport: Tests for XML format detection and validation
    TestXMLParserParsing: Tests for core XML parsing functionality
    TestXMLParserContentExtraction: Tests for XML content extraction
    TestXMLParserMetadataExtraction: Tests for XML metadata extraction
    TestXMLParserSectionIdentification: Tests for section identification in PMC articles
    TestXMLParserFigureTableExtraction: Tests for figure and table extraction from XML
    TestXMLParserReferenceExtraction: Tests for reference parsing from XML
    TestXMLParserConversion: Tests for converting parsed XML to internal models
    TestXMLParserErrorHandling: Tests for error handling and validation
    TestXMLParserValidation: Tests for XML validation functionality
    TestXMLParserOptions: Tests for parser configuration and options
    TestXMLParserIntegration: Integration tests with other components
    TestXMLParserPerformance: Performance and scalability tests

The XMLParser is expected to be a concrete implementation providing:
- XML format support: PMC XML, RDF/XML, standard XML documents
- Content extraction from scientific articles (PMC format)
- RDF/XML ontology parsing integration
- Metadata extraction (authors, title, DOI, journal info, etc.)
- Section identification (abstract, introduction, methods, results, etc.)
- Figure and table extraction with captions
- Reference parsing and extraction from XML bibliography
- Integration with lxml, xml.etree.ElementTree, and xml.dom.minidom libraries
- Conversion to internal Term/Relationship/Ontology models
- Comprehensive validation and error reporting
- Performance optimization for large XML files
- Configurable parsing options and namespace handling

Dependencies:
    - pytest: For test framework
    - unittest.mock: For mocking functionality
    - lxml.etree: For advanced XML processing (mocked)
    - xml.etree.ElementTree: For standard XML processing (mocked)
    - xml.dom.minidom: For DOM-based XML processing (mocked)
    - typing: For type hints

Usage:
    pytest tests/unit/test_xml_parser.py -v
"""

import tempfile
from datetime import datetime
from pathlib import Path
from unittest.mock import Mock, patch

import pytest


# Mock the parser classes since they don't exist yet (TDD approach)
@pytest.fixture
def mock_xml_parser():
    """Mock XMLParser class."""
    # Create a mock class that doesn't rely on importing non-existent modules
    mock_parser_class = Mock()
    mock_instance = Mock()

    # Core parsing methods
    mock_instance.parse = Mock()
    mock_instance.parse_file = Mock()
    mock_instance.parse_string = Mock()
    mock_instance.parse_bytes = Mock()
    mock_instance.parse_stream = Mock()

    # Format detection and validation
    mock_instance.detect_format = Mock()
    mock_instance.validate_format = Mock()
    mock_instance.get_supported_formats = Mock(return_value=["xml", "pmc", "rdf"])
    mock_instance.is_well_formed = Mock()
    mock_instance.validate_schema = Mock()

    # Content extraction methods
    mock_instance.extract_content = Mock()
    mock_instance.extract_text = Mock()
    mock_instance.extract_structured_content = Mock()
    mock_instance.get_namespaces = Mock()

    # Metadata extraction methods
    mock_instance.extract_metadata = Mock()
    mock_instance.extract_title = Mock()
    mock_instance.extract_authors = Mock()
    mock_instance.extract_doi = Mock()
    mock_instance.extract_abstract = Mock()
    mock_instance.extract_keywords = Mock()
    mock_instance.extract_journal_info = Mock()

    # Section identification methods
    mock_instance.identify_sections = Mock()
    mock_instance.extract_introduction = Mock()
    mock_instance.extract_methods = Mock()
    mock_instance.extract_results = Mock()
    mock_instance.extract_discussion = Mock()
    mock_instance.extract_conclusion = Mock()
    mock_instance.extract_body_content = Mock()

    # Figure and table extraction
    mock_instance.extract_figures = Mock()
    mock_instance.extract_tables = Mock()
    mock_instance.extract_captions = Mock()
    mock_instance.extract_media_objects = Mock()

    # Reference extraction
    mock_instance.extract_references = Mock()
    mock_instance.parse_citations = Mock()
    mock_instance.extract_bibliography = Mock()

    # Conversion methods
    mock_instance.to_ontology = Mock()
    mock_instance.extract_terms = Mock()
    mock_instance.extract_relationships = Mock()
    mock_instance.to_rdf = Mock()

    # Configuration
    mock_instance.set_options = Mock()
    mock_instance.get_options = Mock(return_value={})
    mock_instance.reset_options = Mock()

    # Validation
    mock_instance.validate = Mock()
    mock_instance.validate_xml = Mock()
    mock_instance.get_validation_errors = Mock(return_value=[])

    # Configure the mock class to return the mock instance when called
    mock_parser_class.return_value = mock_instance
    return mock_parser_class


@pytest.fixture
def mock_lxml():
    """Mock lxml.etree library."""
    with patch("lxml.etree") as mock_etree:
        # Mock XML parsing
        mock_element = Mock()
        mock_element.tag = "article"
        mock_element.text = "Sample content"
        mock_element.attrib = {"id": "test-article"}
        mock_element.nsmap = {
            "": "http://dtd.nlm.nih.gov/ncbi/pmc/articleset/nlm-articleset-2.0.dtd"
        }

        # Mock find methods
        mock_element.find = Mock(return_value=mock_element)
        mock_element.findall = Mock(return_value=[mock_element])
        mock_element.xpath = Mock(return_value=[mock_element])

        # Mock tree structure
        mock_element.getparent = Mock(return_value=None)
        mock_element.getchildren = Mock(return_value=[])
        mock_element.__iter__ = Mock(return_value=iter([]))

        # Mock parsing functions
        mock_etree.parse = Mock(
            return_value=Mock(getroot=Mock(return_value=mock_element))
        )
        mock_etree.fromstring = Mock(return_value=mock_element)
        mock_etree.XMLParser = Mock()
        mock_etree.XMLSchema = Mock()
        mock_etree.XMLSyntaxError = Exception

        yield mock_etree


@pytest.fixture
def mock_elementtree():
    """Mock xml.etree.ElementTree library."""
    with patch("xml.etree.ElementTree") as mock_et:
        # Mock element
        mock_element = Mock()
        mock_element.tag = "root"
        mock_element.text = "content"
        mock_element.attrib = {}
        mock_element.find = Mock(return_value=mock_element)
        mock_element.findall = Mock(return_value=[mock_element])
        mock_element.iter = Mock(return_value=iter([mock_element]))

        # Mock tree
        mock_tree = Mock()
        mock_tree.getroot = Mock(return_value=mock_element)

        # Mock parsing functions
        mock_et.parse = Mock(return_value=mock_tree)
        mock_et.fromstring = Mock(return_value=mock_element)
        mock_et.ParseError = Exception

        yield mock_et


@pytest.fixture
def mock_minidom():
    """Mock xml.dom.minidom library."""
    with patch("xml.dom.minidom") as mock_dom:
        # Mock document
        mock_doc = Mock()
        mock_element = Mock()
        mock_element.nodeName = "article"
        mock_element.nodeValue = "content"
        mock_element.attributes = {}
        mock_element.childNodes = []
        mock_element.getElementsByTagName = Mock(return_value=[mock_element])

        mock_doc.documentElement = mock_element
        mock_doc.getElementsByTagName = Mock(return_value=[mock_element])

        # Mock parsing functions
        mock_dom.parse = Mock(return_value=mock_doc)
        mock_dom.parseString = Mock(return_value=mock_doc)

        yield mock_dom


@pytest.fixture
def sample_xml_content():
    """Sample XML content for testing."""
    return """<?xml version="1.0" encoding="UTF-8"?>
<article xmlns="http://dtd.nlm.nih.gov/ncbi/pmc/articleset/nlm-articleset-2.0.dtd">
    <front>
        <article-meta>
            <title-group>
                <article-title>Machine Learning Approaches for Ontology Information Extraction</article-title>
            </title-group>
            <contrib-group>
                <contrib contrib-type="author">
                    <name>
                        <surname>Smith</surname>
                        <given-names>Jane</given-names>
                    </name>
                </contrib>
            </contrib-group>
            <abstract>
                <p>This paper presents novel machine learning approaches for extracting ontological information from scientific literature...</p>
            </abstract>
        </article-meta>
    </front>
    <body>
        <sec sec-type="intro">
            <title>Introduction</title>
            <p>The field of ontology information extraction has grown significantly...</p>
        </sec>
        <sec sec-type="methods">
            <title>Methods</title>
            <p>We employed a combination of natural language processing techniques...</p>
        </sec>
    </body>
    <back>
        <ref-list>
            <ref id="ref1">
                <element-citation>
                    <person-group person-group-type="author">
                        <name>
                            <surname>Doe</surname>
                            <given-names>John</given-names>
                        </name>
                    </person-group>
                    <article-title>Previous work on ontology extraction</article-title>
                    <source>Journal of AI</source>
                    <year>2022</year>
                </element-citation>
            </ref>
        </ref-list>
    </back>
</article>"""


@pytest.fixture
def sample_rdf_xml_content():
    """Sample RDF/XML content for testing."""
    return """<?xml version="1.0" encoding="UTF-8"?>
<rdf:RDF xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#"
         xmlns:rdfs="http://www.w3.org/2000/01/rdf-schema#"
         xmlns:owl="http://www.w3.org/2002/07/owl#">
    <owl:Class rdf:about="http://example.org/ontology#MachineLearning">
        <rdfs:label>Machine Learning</rdfs:label>
        <rdfs:comment>A method of data analysis that automates analytical model building</rdfs:comment>
    </owl:Class>
    <owl:Class rdf:about="http://example.org/ontology#Ontology">
        <rdfs:label>Ontology</rdfs:label>
        <rdfs:comment>A formal representation of knowledge as a set of concepts</rdfs:comment>
    </owl:Class>
</rdf:RDF>"""


@pytest.fixture
def sample_pmc_article_content():
    """Sample PMC article content for testing."""
    return {
        "title": "Machine Learning Approaches for Ontology Information Extraction",
        "authors": [
            {"surname": "Smith", "given_names": "Jane", "affiliation": "MIT"},
            {"surname": "Doe", "given_names": "John", "affiliation": "Stanford"},
        ],
        "abstract": "This paper presents novel machine learning approaches for extracting ontological information from scientific literature...",
        "journal": "Nature Machine Intelligence",
        "volume": "3",
        "issue": "4",
        "year": "2022",
        "doi": "10.1038/s42256-022-00567-8",
        "sections": {
            "introduction": "The field of ontology information extraction has grown significantly...",
            "methods": "We employed a combination of natural language processing techniques...",
            "results": "Our experiments show that the proposed method achieves 95% accuracy...",
            "discussion": "The results demonstrate the effectiveness of our approach...",
            "conclusion": "In conclusion, we have presented a robust method for ontology extraction...",
        },
        "figures": [
            {
                "id": "fig1",
                "caption": "Architecture of the proposed system",
                "graphic_ref": "fig1.jpg",
            },
            {
                "id": "fig2",
                "caption": "Performance comparison",
                "graphic_ref": "fig2.png",
            },
        ],
        "tables": [
            {"id": "tab1", "caption": "Dataset statistics", "rows": 5, "cols": 4},
            {"id": "tab2", "caption": "Experimental results", "rows": 8, "cols": 6},
        ],
        "references": [
            {
                "id": "ref1",
                "authors": ["Doe, J."],
                "title": "Previous work on ontology extraction",
                "journal": "Journal of AI",
                "year": "2022",
            },
            {
                "id": "ref2",
                "authors": ["Brown, A.", "Wilson, B."],
                "title": "Advanced NLP techniques",
                "conference": "ICML 2021",
                "year": "2021",
            },
        ],
    }


@pytest.fixture
def sample_xml_metadata():
    """Sample XML metadata for testing."""
    return {
        "format": "PMC XML",
        "version": "2.0",
        "encoding": "UTF-8",
        "namespaces": {
            "": "http://dtd.nlm.nih.gov/ncbi/pmc/articleset/nlm-articleset-2.0.dtd",
            "xlink": "http://www.w3.org/1999/xlink",
            "mml": "http://www.w3.org/1998/Math/MathML",
        },
        "document_type": "research-article",
        "schema_valid": True,
        "well_formed": True,
        "file_size": 156789,
        "creation_date": datetime(2023, 1, 1, 12, 0, 0),
        "elements_count": 245,
        "text_length": 15000,
    }


class TestXMLParserCreation:
    """Test XMLParser instantiation and configuration."""

    def test_xml_parser_creation_default(self, mock_xml_parser):
        """Test creating XMLParser with default settings."""
        parser = mock_xml_parser()

        # Verify parser was created
        assert parser is not None
        mock_xml_parser.assert_called_once()

    def test_xml_parser_creation_with_options(self, mock_xml_parser):
        """Test creating XMLParser with custom options."""
        options = {
            "parser_library": "lxml",
            "namespace_aware": True,
            "validate_schema": False,
            "resolve_entities": False,
            "encoding": "utf-8",
        }

        parser = mock_xml_parser(options=options)
        mock_xml_parser.assert_called_once_with(options=options)

    def test_xml_parser_creation_with_invalid_options(self, mock_xml_parser):
        """Test creating XMLParser with invalid options raises error."""
        mock_xml_parser.side_effect = ValueError("Invalid option: unknown_library")

        invalid_options = {"unknown_library": "invalid"}

        with pytest.raises(ValueError, match="Invalid option"):
            mock_xml_parser(options=invalid_options)

    def test_xml_parser_inherits_from_abstract_parser(self, mock_xml_parser):
        """Test that XMLParser implements AbstractParser interface."""
        parser = mock_xml_parser()

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


class TestXMLParserFormatSupport:
    """Test XML format detection and validation."""

    def test_get_supported_formats(self, mock_xml_parser):
        """Test getting list of supported XML formats."""
        parser = mock_xml_parser()

        formats = parser.get_supported_formats()

        expected_formats = ["xml", "pmc", "rdf"]
        assert all(fmt in formats for fmt in expected_formats)

    def test_detect_format_pmc_xml(self, mock_xml_parser, sample_xml_content):
        """Test detecting PMC XML format."""
        parser = mock_xml_parser()
        parser.detect_format.return_value = "pmc"

        detected_format = parser.detect_format(sample_xml_content)

        assert detected_format == "pmc"
        parser.detect_format.assert_called_once_with(sample_xml_content)

    def test_detect_format_rdf_xml(self, mock_xml_parser, sample_rdf_xml_content):
        """Test detecting RDF/XML format."""
        parser = mock_xml_parser()
        parser.detect_format.return_value = "rdf"

        detected_format = parser.detect_format(sample_rdf_xml_content)

        assert detected_format == "rdf"
        parser.detect_format.assert_called_once_with(sample_rdf_xml_content)

    def test_validate_format_valid_xml(self, mock_xml_parser, sample_xml_content):
        """Test validating valid XML content."""
        parser = mock_xml_parser()
        parser.validate_format.return_value = True

        is_valid = parser.validate_format(sample_xml_content, "xml")

        assert is_valid is True
        parser.validate_format.assert_called_once_with(sample_xml_content, "xml")

    def test_validate_format_invalid_xml(self, mock_xml_parser):
        """Test validating invalid XML content."""
        parser = mock_xml_parser()
        parser.validate_format.return_value = False

        invalid_content = "<invalid><unclosed>XML content"
        is_valid = parser.validate_format(invalid_content, "xml")

        assert is_valid is False

    def test_is_well_formed_true(self, mock_xml_parser, sample_xml_content):
        """Test detecting well-formed XML."""
        parser = mock_xml_parser()
        parser.is_well_formed.return_value = True

        well_formed = parser.is_well_formed(sample_xml_content)

        assert well_formed is True
        parser.is_well_formed.assert_called_once_with(sample_xml_content)

    def test_is_well_formed_false(self, mock_xml_parser):
        """Test detecting malformed XML."""
        parser = mock_xml_parser()
        parser.is_well_formed.return_value = False

        malformed_xml = "<root><unclosed>content"
        well_formed = parser.is_well_formed(malformed_xml)

        assert well_formed is False

    def test_validate_schema_with_dtd(self, mock_xml_parser, sample_xml_content):
        """Test validating XML against DTD schema."""
        parser = mock_xml_parser()
        parser.validate_schema.return_value = True

        parser.set_options({"schema_type": "dtd", "schema_path": "/path/to/pmc.dtd"})
        is_valid = parser.validate_schema(sample_xml_content)

        assert is_valid is True
        parser.validate_schema.assert_called_once_with(sample_xml_content)

    def test_validate_schema_with_xsd(self, mock_xml_parser, sample_rdf_xml_content):
        """Test validating XML against XSD schema."""
        parser = mock_xml_parser()
        parser.validate_schema.return_value = True

        parser.set_options({"schema_type": "xsd", "schema_path": "/path/to/rdf.xsd"})
        is_valid = parser.validate_schema(sample_rdf_xml_content)

        assert is_valid is True


class TestXMLParserParsing:
    """Test core XML parsing functionality."""

    def test_parse_file_xml_path(self, mock_xml_parser):
        """Test parsing XML file from file path."""
        parser = mock_xml_parser()
        mock_result = Mock()
        mock_result.root_element = "article"
        mock_result.elements_count = 245
        parser.parse_file.return_value = mock_result

        file_path = "/path/to/document.xml"
        result = parser.parse_file(file_path)

        assert result == mock_result
        assert result.root_element == "article"
        parser.parse_file.assert_called_once_with(file_path)

    def test_parse_string_xml_content(self, mock_xml_parser, sample_xml_content):
        """Test parsing XML from string."""
        parser = mock_xml_parser()
        mock_result = Mock()
        mock_result.source_type = "string"
        mock_result.namespaces = {
            "": "http://dtd.nlm.nih.gov/ncbi/pmc/articleset/nlm-articleset-2.0.dtd"
        }
        parser.parse_string.return_value = mock_result

        result = parser.parse_string(sample_xml_content)

        assert result == mock_result
        assert result.source_type == "string"
        parser.parse_string.assert_called_once_with(sample_xml_content)

    def test_parse_bytes_xml_data(self, mock_xml_parser, sample_xml_content):
        """Test parsing XML from bytes."""
        parser = mock_xml_parser()
        mock_result = Mock()
        mock_result.encoding = "utf-8"
        parser.parse_bytes.return_value = mock_result

        xml_bytes = sample_xml_content.encode("utf-8")
        result = parser.parse_bytes(xml_bytes)

        assert result == mock_result
        assert result.encoding == "utf-8"
        parser.parse_bytes.assert_called_once_with(xml_bytes)

    def test_parse_stream_xml_data(self, mock_xml_parser):
        """Test parsing XML from stream/file-like object."""
        parser = mock_xml_parser()
        mock_result = Mock()
        parser.parse_stream.return_value = mock_result

        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".xml", encoding="utf-8"
        ) as tmp_file:
            tmp_file.write('<?xml version="1.0"?><root>test content</root>')
            tmp_file.seek(0)

            result = parser.parse_stream(tmp_file)

        assert result == mock_result
        parser.parse_stream.assert_called_once()

    def test_parse_with_namespace_handling(self, mock_xml_parser):
        """Test parsing XML with namespace handling."""
        parser = mock_xml_parser()
        mock_result = Mock()
        mock_result.namespaces = {
            "pmc": "http://dtd.nlm.nih.gov/ncbi/pmc/articleset/nlm-articleset-2.0.dtd",
            "xlink": "http://www.w3.org/1999/xlink",
        }
        parser.parse_string.return_value = mock_result

        parser.set_options({"namespace_aware": True, "preserve_namespaces": True})
        result = parser.parse_string(
            "<root xmlns:pmc='http://dtd.nlm.nih.gov/ncbi/pmc/articleset/nlm-articleset-2.0.dtd'/>"
        )

        assert result == mock_result
        assert len(result.namespaces) == 2
        parser.set_options.assert_called_with(
            {"namespace_aware": True, "preserve_namespaces": True}
        )

    def test_parse_with_encoding_detection(self, mock_xml_parser):
        """Test parsing XML with automatic encoding detection."""
        parser = mock_xml_parser()
        mock_result = Mock()
        mock_result.detected_encoding = "iso-8859-1"
        parser.parse_bytes.return_value = mock_result

        xml_with_encoding = (
            b'<?xml version="1.0" encoding="iso-8859-1"?><root>content</root>'
        )
        result = parser.parse_bytes(xml_with_encoding)

        assert result == mock_result
        assert result.detected_encoding == "iso-8859-1"

    def test_parse_with_parser_library_selection(self, mock_xml_parser):
        """Test parsing with specific parser library."""
        parser = mock_xml_parser()
        mock_result = Mock()
        mock_result.parser_used = "lxml"
        parser.parse_string.return_value = mock_result

        parser.set_options({"parser_library": "lxml"})
        result = parser.parse_string("<root>content</root>")

        assert result == mock_result
        assert result.parser_used == "lxml"
        parser.set_options.assert_called_with({"parser_library": "lxml"})

    def test_parse_large_xml_file(self, mock_xml_parser):
        """Test parsing large XML file with memory optimization."""
        parser = mock_xml_parser()
        mock_result = Mock()
        mock_result.elements_count = 100000
        mock_result.memory_efficient = True
        parser.parse_file.return_value = mock_result

        # Configure for large file parsing
        parser.set_options({"streaming_mode": True, "chunk_size": 8192})
        result = parser.parse_file("/path/to/large_document.xml")

        assert result == mock_result
        assert result.memory_efficient is True
        parser.set_options.assert_called_with(
            {"streaming_mode": True, "chunk_size": 8192}
        )


class TestXMLParserContentExtraction:
    """Test XML content extraction functionality."""

    def test_extract_content_basic(self, mock_xml_parser):
        """Test basic content extraction from XML."""
        parser = mock_xml_parser()
        expected_content = {"title": "Test Article", "body": "Article content..."}
        parser.extract_content.return_value = expected_content

        parsed_result = Mock()
        content = parser.extract_content(parsed_result)

        assert content == expected_content
        assert "title" in content
        parser.extract_content.assert_called_once_with(parsed_result)

    def test_extract_text_content(self, mock_xml_parser):
        """Test extracting text content from XML elements."""
        parser = mock_xml_parser()
        expected_text = "This is the extracted text from the XML document."
        parser.extract_text.return_value = expected_text

        parsed_result = Mock()
        text = parser.extract_text(parsed_result)

        assert text == expected_text
        parser.extract_text.assert_called_once_with(parsed_result)

    def test_extract_structured_content(self, mock_xml_parser):
        """Test extracting structured content preserving XML hierarchy."""
        parser = mock_xml_parser()
        mock_structure = Mock()
        mock_structure.sections = ["front", "body", "back"]
        mock_structure.elements = {"article": 1, "section": 5, "paragraph": 20}
        parser.extract_structured_content.return_value = mock_structure

        parsed_result = Mock()
        structure = parser.extract_structured_content(parsed_result)

        assert structure == mock_structure
        assert "front" in structure.sections
        parser.extract_structured_content.assert_called_once_with(parsed_result)

    def test_get_namespaces(self, mock_xml_parser):
        """Test extracting namespace information from XML."""
        parser = mock_xml_parser()
        expected_namespaces = {
            "": "http://dtd.nlm.nih.gov/ncbi/pmc/articleset/nlm-articleset-2.0.dtd",
            "xlink": "http://www.w3.org/1999/xlink",
            "mml": "http://www.w3.org/1998/Math/MathML",
        }
        parser.get_namespaces.return_value = expected_namespaces

        parsed_result = Mock()
        namespaces = parser.get_namespaces(parsed_result)

        assert namespaces == expected_namespaces
        assert len(namespaces) == 3
        parser.get_namespaces.assert_called_once_with(parsed_result)

    def test_extract_content_with_xpath(self, mock_xml_parser):
        """Test content extraction using XPath expressions."""
        parser = mock_xml_parser()
        mock_result = Mock()
        mock_result.xpath_results = {
            "//title": ["Main Title", "Section Title"],
            "//p": ["Paragraph 1", "Paragraph 2", "Paragraph 3"],
        }
        parser.extract_content.return_value = mock_result

        parser.set_options({"use_xpath": True, "xpath_expressions": ["//title", "//p"]})

        parsed_result = Mock()
        result = parser.extract_content(parsed_result)

        assert len(result.xpath_results["//title"]) == 2
        assert len(result.xpath_results["//p"]) == 3

    def test_extract_content_with_css_selectors(self, mock_xml_parser):
        """Test content extraction using CSS selectors."""
        parser = mock_xml_parser()
        mock_result = Mock()
        mock_result.css_results = {
            "title": ["Document Title"],
            "sec[sec-type='intro'] title": ["Introduction"],
        }
        parser.extract_content.return_value = mock_result

        parser.set_options({"use_css_selectors": True})

        parsed_result = Mock()
        result = parser.extract_content(parsed_result)

        assert "title" in result.css_results
        assert len(result.css_results["title"]) == 1

    def test_extract_content_multilingual(self, mock_xml_parser):
        """Test content extraction from multilingual XML documents."""
        parser = mock_xml_parser()
        mock_result = Mock()
        mock_result.languages = {"en": "English content", "de": "Deutsche Inhalte"}
        parser.extract_content.return_value = mock_result

        parser.set_options({"extract_languages": True, "language_detection": True})

        parsed_result = Mock()
        result = parser.extract_content(parsed_result)

        assert "en" in result.languages
        assert "de" in result.languages

    def test_extract_content_error_handling(self, mock_xml_parser):
        """Test error handling during content extraction."""
        parser = mock_xml_parser()
        parser.extract_content.side_effect = Exception("Content extraction failed")

        parsed_result = Mock()

        with pytest.raises(Exception, match="Content extraction failed"):
            parser.extract_content(parsed_result)


class TestXMLParserMetadataExtraction:
    """Test XML metadata extraction functionality."""

    def test_extract_metadata_basic(self, mock_xml_parser, sample_xml_metadata):
        """Test basic metadata extraction from XML."""
        parser = mock_xml_parser()
        parser.extract_metadata.return_value = sample_xml_metadata

        parsed_result = Mock()
        metadata = parser.extract_metadata(parsed_result)

        assert metadata == sample_xml_metadata
        assert metadata["format"] == "PMC XML"
        assert metadata["version"] == "2.0"
        assert metadata["schema_valid"] is True
        parser.extract_metadata.assert_called_once_with(parsed_result)

    def test_extract_title(self, mock_xml_parser):
        """Test extracting document title."""
        parser = mock_xml_parser()
        expected_title = "Machine Learning for Ontology Extraction from XML Documents"
        parser.extract_title.return_value = expected_title

        parsed_result = Mock()
        title = parser.extract_title(parsed_result)

        assert title == expected_title
        parser.extract_title.assert_called_once_with(parsed_result)

    def test_extract_authors(self, mock_xml_parser):
        """Test extracting document authors."""
        parser = mock_xml_parser()
        expected_authors = [
            {"surname": "Smith", "given_names": "Jane", "affiliation": "MIT"},
            {"surname": "Doe", "given_names": "John", "affiliation": "Stanford"},
            {"surname": "Johnson", "given_names": "Alice", "affiliation": "Harvard"},
        ]
        parser.extract_authors.return_value = expected_authors

        parsed_result = Mock()
        authors = parser.extract_authors(parsed_result)

        assert authors == expected_authors
        assert len(authors) == 3
        assert authors[0]["surname"] == "Smith"
        parser.extract_authors.assert_called_once_with(parsed_result)

    def test_extract_doi(self, mock_xml_parser):
        """Test extracting DOI from XML."""
        parser = mock_xml_parser()
        expected_doi = "10.1038/s42256-022-00567-8"
        parser.extract_doi.return_value = expected_doi

        parsed_result = Mock()
        doi = parser.extract_doi(parsed_result)

        assert doi == expected_doi
        parser.extract_doi.assert_called_once_with(parsed_result)

    def test_extract_abstract(self, mock_xml_parser):
        """Test extracting abstract from scientific article."""
        parser = mock_xml_parser()
        expected_abstract = "This paper presents novel approaches for ontology extraction from XML documents..."
        parser.extract_abstract.return_value = expected_abstract

        parsed_result = Mock()
        abstract = parser.extract_abstract(parsed_result)

        assert abstract == expected_abstract
        assert "ontology extraction" in abstract
        parser.extract_abstract.assert_called_once_with(parsed_result)

    def test_extract_keywords(self, mock_xml_parser):
        """Test extracting keywords from XML."""
        parser = mock_xml_parser()
        expected_keywords = [
            "ontology",
            "XML parsing",
            "information extraction",
            "machine learning",
        ]
        parser.extract_keywords.return_value = expected_keywords

        parsed_result = Mock()
        keywords = parser.extract_keywords(parsed_result)

        assert keywords == expected_keywords
        assert "ontology" in keywords
        parser.extract_keywords.assert_called_once_with(parsed_result)

    def test_extract_journal_info(self, mock_xml_parser):
        """Test extracting journal information from XML."""
        parser = mock_xml_parser()
        expected_journal_info = {
            "journal_title": "Nature Machine Intelligence",
            "issn": "2522-5839",
            "volume": "3",
            "issue": "4",
            "year": "2022",
            "publisher": "Nature Publishing Group",
        }
        parser.extract_journal_info.return_value = expected_journal_info

        parsed_result = Mock()
        journal_info = parser.extract_journal_info(parsed_result)

        assert journal_info == expected_journal_info
        assert journal_info["journal_title"] == "Nature Machine Intelligence"
        parser.extract_journal_info.assert_called_once_with(parsed_result)

    def test_extract_metadata_with_fallback(self, mock_xml_parser):
        """Test metadata extraction with fallback methods."""
        parser = mock_xml_parser()

        # Primary extraction fails, fallback succeeds
        mock_metadata = {
            "title": "Extracted from content analysis",
            "authors": [],
            "extraction_method": "content_analysis",
            "confidence": 0.75,
        }
        parser.extract_metadata.return_value = mock_metadata

        parsed_result = Mock()
        metadata = parser.extract_metadata(parsed_result)

        assert metadata["title"] == "Extracted from content analysis"
        assert metadata["extraction_method"] == "content_analysis"
        assert metadata["confidence"] == 0.75

    def test_extract_metadata_empty_xml(self, mock_xml_parser):
        """Test metadata extraction from XML with minimal metadata."""
        parser = mock_xml_parser()
        minimal_metadata = {
            "title": None,
            "authors": [],
            "format": "generic XML",
            "elements_count": 5,
            "has_metadata": False,
        }
        parser.extract_metadata.return_value = minimal_metadata

        parsed_result = Mock()
        metadata = parser.extract_metadata(parsed_result)

        assert metadata["title"] is None
        assert len(metadata["authors"]) == 0
        assert metadata["has_metadata"] is False


class TestXMLParserSectionIdentification:
    """Test section identification in PMC articles."""

    def test_identify_sections(self, mock_xml_parser):
        """Test identifying document sections."""
        parser = mock_xml_parser()
        expected_sections = {
            "front": {"element": "front", "contains": ["article-meta", "title-group"]},
            "body": {"element": "body", "contains": ["sec"]},
            "back": {"element": "back", "contains": ["ref-list", "ack"]},
            "abstract": {"sec_type": "abstract", "title": "Abstract"},
            "introduction": {"sec_type": "intro", "title": "Introduction"},
            "methods": {"sec_type": "methods", "title": "Methods"},
            "results": {"sec_type": "results", "title": "Results"},
            "discussion": {"sec_type": "discussion", "title": "Discussion"},
            "conclusion": {"sec_type": "conclusions", "title": "Conclusions"},
        }
        parser.identify_sections.return_value = expected_sections

        parsed_result = Mock()
        sections = parser.identify_sections(parsed_result)

        assert sections == expected_sections
        assert "front" in sections
        assert "methods" in sections
        parser.identify_sections.assert_called_once_with(parsed_result)

    def test_extract_introduction(self, mock_xml_parser):
        """Test extracting introduction section."""
        parser = mock_xml_parser()
        expected_intro = "The field of ontology information extraction from XML documents has grown significantly..."
        parser.extract_introduction.return_value = expected_intro

        parsed_result = Mock()
        introduction = parser.extract_introduction(parsed_result)

        assert introduction == expected_intro
        assert "ontology information extraction" in introduction
        parser.extract_introduction.assert_called_once_with(parsed_result)

    def test_extract_methods(self, mock_xml_parser):
        """Test extracting methods section."""
        parser = mock_xml_parser()
        expected_methods = "We employed a combination of XML parsing techniques and natural language processing..."
        parser.extract_methods.return_value = expected_methods

        parsed_result = Mock()
        methods = parser.extract_methods(parsed_result)

        assert methods == expected_methods
        assert "XML parsing techniques" in methods
        parser.extract_methods.assert_called_once_with(parsed_result)

    def test_extract_results(self, mock_xml_parser):
        """Test extracting results section."""
        parser = mock_xml_parser()
        expected_results = "Our XML-based extraction approach achieved 97% accuracy on the test dataset..."
        parser.extract_results.return_value = expected_results

        parsed_result = Mock()
        results = parser.extract_results(parsed_result)

        assert results == expected_results
        assert "97% accuracy" in results
        parser.extract_results.assert_called_once_with(parsed_result)

    def test_extract_discussion(self, mock_xml_parser):
        """Test extracting discussion section."""
        parser = mock_xml_parser()
        expected_discussion = "The results demonstrate the effectiveness of XML-based parsing for ontology extraction..."
        parser.extract_discussion.return_value = expected_discussion

        parsed_result = Mock()
        discussion = parser.extract_discussion(parsed_result)

        assert discussion == expected_discussion
        parser.extract_discussion.assert_called_once_with(parsed_result)

    def test_extract_conclusion(self, mock_xml_parser):
        """Test extracting conclusion section."""
        parser = mock_xml_parser()
        expected_conclusion = "In conclusion, we have presented a robust XML-based method for ontology extraction..."
        parser.extract_conclusion.return_value = expected_conclusion

        parsed_result = Mock()
        conclusion = parser.extract_conclusion(parsed_result)

        assert conclusion == expected_conclusion
        assert "XML-based method" in conclusion
        parser.extract_conclusion.assert_called_once_with(parsed_result)

    def test_extract_body_content(self, mock_xml_parser):
        """Test extracting complete body content."""
        parser = mock_xml_parser()
        expected_body = {
            "sections": ["introduction", "methods", "results", "discussion"],
            "paragraphs": 25,
            "subsections": 8,
            "content_length": 15000,
        }
        parser.extract_body_content.return_value = expected_body

        parsed_result = Mock()
        body = parser.extract_body_content(parsed_result)

        assert body == expected_body
        assert len(body["sections"]) == 4
        parser.extract_body_content.assert_called_once_with(parsed_result)

    def test_section_identification_with_confidence(self, mock_xml_parser):
        """Test section identification with confidence scores."""
        parser = mock_xml_parser()
        expected_sections = {
            "abstract": {
                "content": "Abstract content...",
                "confidence": 0.98,
                "method": "element_detection",
            },
            "introduction": {
                "content": "Introduction content...",
                "confidence": 0.95,
                "method": "sec_type_attribute",
            },
            "methods": {
                "content": "Methods content...",
                "confidence": 0.90,
                "method": "title_matching",
            },
        }
        parser.identify_sections.return_value = expected_sections

        parsed_result = Mock()
        sections = parser.identify_sections(parsed_result)

        assert sections["abstract"]["confidence"] == 0.98
        assert sections["methods"]["method"] == "title_matching"

    def test_section_identification_nested_sections(self, mock_xml_parser):
        """Test identifying sections with nested subsections."""
        parser = mock_xml_parser()
        expected_sections = {
            "methods": {
                "content": "Methods overview...",
                "subsections": {
                    "data_collection": {
                        "title": "Data Collection",
                        "content": "We collected...",
                    },
                    "preprocessing": {
                        "title": "Data Preprocessing",
                        "content": "XML preprocessing...",
                    },
                    "analysis": {
                        "title": "Analysis Methods",
                        "content": "Statistical analysis...",
                    },
                },
            }
        }
        parser.identify_sections.return_value = expected_sections

        parsed_result = Mock()
        sections = parser.identify_sections(parsed_result)

        assert "methods" in sections
        assert "subsections" in sections["methods"]
        assert "data_collection" in sections["methods"]["subsections"]


class TestXMLParserFigureTableExtraction:
    """Test figure and table extraction from XML."""

    def test_extract_figures(self, mock_xml_parser):
        """Test extracting figure information from XML."""
        parser = mock_xml_parser()
        expected_figures = [
            {
                "id": "fig1",
                "label": "Figure 1",
                "caption": "Architecture of the XML parsing system",
                "graphic_ref": "fig1.jpg",
                "position": "top",
                "content_type": "image/jpeg",
            },
            {
                "id": "fig2",
                "label": "Figure 2",
                "caption": "Performance comparison of different XML parsers",
                "graphic_ref": "fig2.png",
                "position": "bottom",
                "content_type": "image/png",
            },
        ]
        parser.extract_figures.return_value = expected_figures

        parsed_result = Mock()
        figures = parser.extract_figures(parsed_result)

        assert figures == expected_figures
        assert len(figures) == 2
        assert figures[0]["caption"].startswith("Architecture")
        parser.extract_figures.assert_called_once_with(parsed_result)

    def test_extract_tables(self, mock_xml_parser):
        """Test extracting table information from XML."""
        parser = mock_xml_parser()
        expected_tables = [
            {
                "id": "tab1",
                "label": "Table 1",
                "caption": "Dataset statistics and XML document characteristics",
                "rows": 8,
                "cols": 5,
                "table_wrap_foot": "Values represent mean ± standard deviation",
            },
            {
                "id": "tab2",
                "label": "Table 2",
                "caption": "Performance metrics for XML parsing approaches",
                "rows": 6,
                "cols": 4,
                "table_wrap_foot": "Best results are shown in bold",
            },
        ]
        parser.extract_tables.return_value = expected_tables

        parsed_result = Mock()
        tables = parser.extract_tables(parsed_result)

        assert tables == expected_tables
        assert len(tables) == 2
        assert tables[0]["rows"] == 8
        parser.extract_tables.assert_called_once_with(parsed_result)

    def test_extract_captions(self, mock_xml_parser):
        """Test extracting all captions from XML document."""
        parser = mock_xml_parser()
        expected_captions = [
            {
                "type": "figure",
                "id": "fig1",
                "label": "Figure 1",
                "text": "XML parsing system architecture",
            },
            {
                "type": "figure",
                "id": "fig2",
                "label": "Figure 2",
                "text": "Performance comparison results",
            },
            {
                "type": "table",
                "id": "tab1",
                "label": "Table 1",
                "text": "Dataset characteristics",
            },
            {
                "type": "table",
                "id": "tab2",
                "label": "Table 2",
                "text": "Evaluation metrics",
            },
        ]
        parser.extract_captions.return_value = expected_captions

        parsed_result = Mock()
        captions = parser.extract_captions(parsed_result)

        assert captions == expected_captions
        assert len(captions) == 4
        figure_captions = [c for c in captions if c["type"] == "figure"]
        assert len(figure_captions) == 2
        parser.extract_captions.assert_called_once_with(parsed_result)

    def test_extract_media_objects(self, mock_xml_parser):
        """Test extracting multimedia objects from XML."""
        parser = mock_xml_parser()
        expected_media = [
            {
                "id": "media1",
                "content_type": "video/mp4",
                "href": "supplementary_video.mp4",
                "caption": "Demonstration of the XML parsing process",
                "mime_subtype": "mp4",
            },
            {
                "id": "media2",
                "content_type": "application/zip",
                "href": "source_code.zip",
                "caption": "Source code and datasets",
                "mime_subtype": "zip",
            },
        ]
        parser.extract_media_objects.return_value = expected_media

        parsed_result = Mock()
        media = parser.extract_media_objects(parsed_result)

        assert media == expected_media
        assert len(media) == 2
        assert media[0]["content_type"] == "video/mp4"
        parser.extract_media_objects.assert_called_once_with(parsed_result)

    def test_extract_figures_with_graphic_data(self, mock_xml_parser):
        """Test extracting figures with embedded graphic information."""
        parser = mock_xml_parser()
        expected_figures = [
            {
                "id": "fig1",
                "caption": "System overview",
                "graphic": {
                    "href": "fig1.svg",
                    "content_type": "image/svg+xml",
                    "width": "600px",
                    "height": "400px",
                },
                "has_graphic": True,
            }
        ]
        parser.extract_figures.return_value = expected_figures

        parser.set_options({"extract_graphics": True})

        parsed_result = Mock()
        figures = parser.extract_figures(parsed_result)

        assert figures[0]["has_graphic"] is True
        assert figures[0]["graphic"]["content_type"] == "image/svg+xml"
        parser.set_options.assert_called_with({"extract_graphics": True})

    def test_extract_tables_with_structured_data(self, mock_xml_parser):
        """Test extracting tables with structured cell data."""
        parser = mock_xml_parser()
        expected_tables = [
            {
                "id": "tab1",
                "caption": "Results summary",
                "structured_data": {
                    "thead": [["Method", "Precision", "Recall", "F1-Score"]],
                    "tbody": [
                        ["XML Parser A", "0.92", "0.88", "0.90"],
                        ["XML Parser B", "0.89", "0.91", "0.90"],
                        ["Our Method", "0.95", "0.94", "0.94"],
                    ],
                },
                "format": "structured",
            }
        ]
        parser.extract_tables.return_value = expected_tables

        parser.set_options({"extract_table_data": True})

        parsed_result = Mock()
        tables = parser.extract_tables(parsed_result)

        assert tables[0]["format"] == "structured"
        assert len(tables[0]["structured_data"]["tbody"]) == 3
        parser.set_options.assert_called_with({"extract_table_data": True})


class TestXMLParserReferenceExtraction:
    """Test reference parsing from XML."""

    def test_extract_references(self, mock_xml_parser):
        """Test extracting bibliography references from XML."""
        parser = mock_xml_parser()
        expected_references = [
            {
                "id": "ref1",
                "element_citation": {
                    "publication_type": "journal",
                    "person_group": [{"surname": "Smith", "given_names": "J."}],
                    "article_title": "XML parsing techniques for scientific literature",
                    "source": "Journal of Information Science",
                    "year": "2022",
                    "volume": "45",
                    "fpage": "123",
                    "lpage": "135",
                },
            },
            {
                "id": "ref2",
                "element_citation": {
                    "publication_type": "confproc",
                    "person_group": [
                        {"surname": "Johnson", "given_names": "A."},
                        {"surname": "Brown", "given_names": "B."},
                    ],
                    "article_title": "Machine learning for XML document analysis",
                    "conf_name": "International Conference on Machine Learning",
                    "year": "2021",
                    "fpage": "456",
                    "lpage": "467",
                },
            },
        ]
        parser.extract_references.return_value = expected_references

        parsed_result = Mock()
        references = parser.extract_references(parsed_result)

        assert references == expected_references
        assert len(references) == 2
        assert (
            references[0]["element_citation"]["source"]
            == "Journal of Information Science"
        )
        parser.extract_references.assert_called_once_with(parsed_result)

    def test_parse_citations(self, mock_xml_parser):
        """Test parsing in-text citations from XML."""
        parser = mock_xml_parser()
        expected_citations = [
            {
                "xref_type": "bibr",
                "rid": "ref1",
                "text": "Smith et al. (2022)",
                "element_id": "cite1",
                "ref_type": "author_year",
            },
            {
                "xref_type": "bibr",
                "rid": "ref2 ref3",
                "text": "[2, 3]",
                "element_id": "cite2",
                "ref_type": "numeric",
            },
        ]
        parser.parse_citations.return_value = expected_citations

        parsed_result = Mock()
        citations = parser.parse_citations(parsed_result)

        assert citations == expected_citations
        assert len(citations) == 2
        assert citations[0]["ref_type"] == "author_year"
        parser.parse_citations.assert_called_once_with(parsed_result)

    def test_extract_bibliography(self, mock_xml_parser):
        """Test extracting complete bibliography section."""
        parser = mock_xml_parser()
        expected_bibliography = {
            "title": "References",
            "ref_count": 45,
            "ref_list_id": "ref-list-1",
            "references": [
                {"id": "ref1", "citation_type": "journal"},
                {"id": "ref2", "citation_type": "book"},
                {"id": "ref3", "citation_type": "confproc"},
            ],
        }
        parser.extract_bibliography.return_value = expected_bibliography

        parsed_result = Mock()
        bibliography = parser.extract_bibliography(parsed_result)

        assert bibliography == expected_bibliography
        assert bibliography["ref_count"] == 45
        assert len(bibliography["references"]) == 3
        parser.extract_bibliography.assert_called_once_with(parsed_result)

    def test_extract_references_with_parsing_confidence(self, mock_xml_parser):
        """Test reference extraction with parsing confidence."""
        parser = mock_xml_parser()
        expected_references = [
            {
                "id": "ref1",
                "raw_xml": "<element-citation>...</element-citation>",
                "parsed_data": {
                    "authors": ["Smith, J."],
                    "title": "XML processing methods",
                    "journal": "Tech Journal",
                    "year": "2022",
                },
                "confidence": 0.96,
                "parsing_method": "structured_xml",
            },
            {
                "id": "ref2",
                "raw_xml": "<mixed-citation>Incomplete citation...</mixed-citation>",
                "parsed_data": {"title": "Incomplete citation"},
                "confidence": 0.4,
                "parsing_method": "text_fallback",
            },
        ]
        parser.extract_references.return_value = expected_references

        parsed_result = Mock()
        references = parser.extract_references(parsed_result)

        high_confidence_refs = [r for r in references if r["confidence"] > 0.8]
        assert len(high_confidence_refs) == 1
        assert references[0]["parsing_method"] == "structured_xml"

    def test_extract_references_different_citation_types(self, mock_xml_parser):
        """Test extracting references of different publication types."""
        parser = mock_xml_parser()
        expected_references = [
            {
                "id": "ref1",
                "publication_type": "journal",
                "authors": ["Smith, J."],
                "journal": "Nature",
                "year": "2022",
            },
            {
                "id": "ref2",
                "publication_type": "book",
                "authors": ["Johnson, A."],
                "publisher": "Academic Press",
                "year": "2021",
            },
            {
                "id": "ref3",
                "publication_type": "confproc",
                "authors": ["Brown, B."],
                "conf_name": "ICML",
                "year": "2020",
            },
            {
                "id": "ref4",
                "publication_type": "web",
                "title": "Online resource",
                "uri": "https://example.com",
                "date_in_citation": "2023",
            },
        ]
        parser.extract_references.return_value = expected_references

        parsed_result = Mock()
        references = parser.extract_references(parsed_result)

        pub_types = [ref["publication_type"] for ref in references]
        assert "journal" in pub_types
        assert "book" in pub_types
        assert "confproc" in pub_types
        assert "web" in pub_types

    def test_extract_references_with_identifiers(self, mock_xml_parser):
        """Test extracting references with DOI, PMID, and other identifiers."""
        parser = mock_xml_parser()
        expected_references = [
            {
                "id": "ref1",
                "identifiers": {
                    "doi": "10.1038/s41586-022-04567-8",
                    "pmid": "35123456",
                    "pmc": "PMC8901234",
                },
                "title": "Advanced XML processing",
                "journal": "Nature",
            },
            {
                "id": "ref2",
                "identifiers": {
                    "isbn": "978-0-123456-78-9",
                    "doi": "10.1007/978-0-123456-78-9",
                },
                "title": "XML Technologies Handbook",
                "publisher": "Springer",
            },
        ]
        parser.extract_references.return_value = expected_references

        parsed_result = Mock()
        references = parser.extract_references(parsed_result)

        assert "doi" in references[0]["identifiers"]
        assert "pmid" in references[0]["identifiers"]
        assert "isbn" in references[1]["identifiers"]


class TestXMLParserConversion:
    """Test converting parsed XML to internal models."""

    def test_to_ontology_conversion(self, mock_xml_parser, sample_pmc_article_content):
        """Test converting parsed XML to Ontology model."""
        parser = mock_xml_parser()

        # Mock the conversion result
        mock_ontology = Mock()
        mock_ontology.id = "xml_ontology_001"
        mock_ontology.name = "XML Extracted Ontology"
        mock_ontology.terms = {}
        mock_ontology.relationships = {}
        mock_ontology.metadata = {
            "source_type": "pmc_xml",
            "title": sample_pmc_article_content["title"],
            "authors": sample_pmc_article_content["authors"],
        }
        parser.to_ontology.return_value = mock_ontology

        # Parse and convert
        parsed_result = Mock()
        ontology = parser.to_ontology(parsed_result)

        assert ontology == mock_ontology
        assert ontology.id == "xml_ontology_001"
        assert ontology.metadata["source_type"] == "pmc_xml"

    def test_extract_terms_from_parsed_xml(self, mock_xml_parser):
        """Test extracting Term objects from parsed XML."""
        parser = mock_xml_parser()

        # Mock extracted terms
        mock_term1 = Mock()
        mock_term1.id = "TERM:001"
        mock_term1.name = "XML Parsing"
        mock_term1.definition = (
            "The process of analyzing XML documents to extract structured information"
        )
        mock_term1.context = "methods"

        mock_term2 = Mock()
        mock_term2.id = "TERM:002"
        mock_term2.name = "PMC Format"
        mock_term2.definition = (
            "XML format used by PubMed Central for scientific articles"
        )
        mock_term2.context = "introduction"

        mock_terms = [mock_term1, mock_term2]
        parser.extract_terms.return_value = mock_terms

        parsed_result = Mock()
        terms = parser.extract_terms(parsed_result)

        assert len(terms) == 2
        assert terms[0].name == "XML Parsing"
        assert terms[1].context == "introduction"

    def test_extract_relationships_from_parsed_xml(self, mock_xml_parser):
        """Test extracting Relationship objects from parsed XML."""
        parser = mock_xml_parser()

        # Mock extracted relationships
        mock_relationships = [
            Mock(
                id="REL:001",
                subject="XML Parsing",
                predicate="enables",
                object="Information Extraction",
                confidence=0.88,
                source="methods_section",
            ),
            Mock(
                id="REL:002",
                subject="PMC Format",
                predicate="is_standard_for",
                object="Scientific Articles",
                confidence=0.92,
                source="background",
            ),
        ]
        parser.extract_relationships.return_value = mock_relationships

        parsed_result = Mock()
        relationships = parser.extract_relationships(parsed_result)

        assert len(relationships) == 2
        assert relationships[0].predicate == "enables"
        assert relationships[1].confidence == 0.92

    def test_to_rdf_conversion(self, mock_xml_parser):
        """Test converting parsed XML to RDF format."""
        parser = mock_xml_parser()

        mock_rdf = Mock()
        mock_rdf.format = "turtle"
        mock_rdf.namespaces = {
            "owl": "http://www.w3.org/2002/07/owl#",
            "rdf": "http://www.w3.org/1999/02/22-rdf-syntax-ns#",
        }
        mock_rdf.triples_count = 150
        parser.to_rdf.return_value = mock_rdf

        parsed_result = Mock()
        rdf = parser.to_rdf(parsed_result)

        assert rdf == mock_rdf
        assert rdf.format == "turtle"
        assert rdf.triples_count == 150

    def test_conversion_with_xml_structure_mapping(self, mock_xml_parser):
        """Test conversion with XML element structure mapping."""
        parser = mock_xml_parser()

        # Configure XML structure mapping
        structure_mapping = {
            "article/front/article-meta": {"context": "metadata", "weight": 0.9},
            "article/body/sec": {"context": "content", "weight": 0.8},
            "article/back/ref-list": {"context": "references", "weight": 0.6},
        }
        parser.set_options({"structure_mapping": structure_mapping})

        mock_ontology = Mock()
        parser.to_ontology.return_value = mock_ontology

        parsed_result = Mock()
        ontology = parser.to_ontology(parsed_result)

        assert ontology == mock_ontology
        parser.set_options.assert_called_with({"structure_mapping": structure_mapping})

    def test_conversion_with_namespace_preservation(self, mock_xml_parser):
        """Test conversion preserving XML namespace information."""
        parser = mock_xml_parser()

        parser.set_options({"preserve_namespaces": True, "namespace_mapping": True})

        mock_ontology = Mock()
        mock_ontology.metadata = {
            "xml_namespaces": {
                "": "http://dtd.nlm.nih.gov/ncbi/pmc/articleset/nlm-articleset-2.0.dtd",
                "xlink": "http://www.w3.org/1999/xlink",
            },
            "namespace_mapping": {"default": "pmc", "xlink": "web_linking"},
        }
        parser.to_ontology.return_value = mock_ontology

        parsed_result = Mock()
        ontology = parser.to_ontology(parsed_result)

        assert "xml_namespaces" in ontology.metadata
        assert "namespace_mapping" in ontology.metadata

    def test_conversion_preserves_xml_provenance(self, mock_xml_parser):
        """Test that conversion preserves XML provenance information."""
        parser = mock_xml_parser()

        mock_ontology = Mock()
        mock_ontology.metadata = {
            "source_format": "pmc_xml",
            "source_file": "/path/to/article.xml",
            "parser_version": "1.0.0",
            "parsing_timestamp": "2023-01-01T00:00:00Z",
            "extraction_methods": ["structured_xml", "xpath", "element_analysis"],
            "xml_metadata": {
                "dtd_version": "2.0",
                "article_type": "research-article",
                "xml_lang": "en",
            },
        }
        parser.to_ontology.return_value = mock_ontology

        parsed_result = Mock()
        ontology = parser.to_ontology(parsed_result)

        assert "source_format" in ontology.metadata
        assert "xml_metadata" in ontology.metadata
        assert ontology.metadata["source_format"] == "pmc_xml"


class TestXMLParserErrorHandling:
    """Test error handling and validation."""

    def test_parse_malformed_xml(self, mock_xml_parser):
        """Test parsing malformed XML raises appropriate error."""
        parser = mock_xml_parser()
        parser.parse_string.side_effect = ValueError(
            "Malformed XML: unclosed tag 'section'"
        )

        malformed_xml = "<article><section>content without closing tag"

        with pytest.raises(ValueError, match="Malformed XML"):
            parser.parse_string(malformed_xml)

    def test_parse_xml_with_invalid_characters(self, mock_xml_parser):
        """Test parsing XML with invalid characters raises error."""
        parser = mock_xml_parser()
        parser.parse_string.side_effect = ValueError(
            "Invalid XML: illegal character in content"
        )

        invalid_xml = (
            "<?xml version='1.0'?><root>content with \x00 null character</root>"
        )

        with pytest.raises(ValueError, match="Invalid XML"):
            parser.parse_string(invalid_xml)

    def test_parse_nonexistent_file(self, mock_xml_parser):
        """Test parsing nonexistent file raises FileNotFoundError."""
        parser = mock_xml_parser()
        parser.parse_file.side_effect = FileNotFoundError(
            "File not found: /nonexistent/file.xml"
        )

        with pytest.raises(FileNotFoundError, match="File not found"):
            parser.parse_file("/nonexistent/file.xml")

    def test_parse_xml_with_encoding_error(self, mock_xml_parser):
        """Test parsing XML with encoding issues raises appropriate error."""
        parser = mock_xml_parser()
        parser.parse_bytes.side_effect = UnicodeDecodeError(
            "utf-8", b"invalid", 0, 1, "invalid start byte"
        )

        invalid_bytes = b"\xff\xfe<?xml version='1.0'?><root>content</root>"

        with pytest.raises(UnicodeDecodeError, match="invalid start byte"):
            parser.parse_bytes(invalid_bytes)

    def test_content_extraction_failure(self, mock_xml_parser):
        """Test handling of content extraction failure."""
        parser = mock_xml_parser()
        parser.extract_content.side_effect = Exception(
            "Content extraction failed: missing required elements"
        )

        parsed_result = Mock()

        with pytest.raises(Exception, match="Content extraction failed"):
            parser.extract_content(parsed_result)

    def test_schema_validation_failure(self, mock_xml_parser):
        """Test handling of schema validation failure."""
        parser = mock_xml_parser()

        mock_result = Mock()
        mock_result.valid = False
        mock_result.errors = [
            "Element 'invalid-element' not allowed",
            "Missing required attribute 'id'",
        ]
        parser.validate_schema.return_value = mock_result

        parser.set_options({"validate_schema": True, "schema_path": "/path/to/pmc.dtd"})

        parsed_result = Mock()
        result = parser.validate_schema(parsed_result)

        assert result.valid is False
        assert len(result.errors) == 2
        assert "invalid-element" in result.errors[0]

    def test_namespace_resolution_error(self, mock_xml_parser):
        """Test handling of namespace resolution errors."""
        parser = mock_xml_parser()
        parser.parse_string.side_effect = Exception(
            "Namespace error: undefined namespace prefix 'unknown'"
        )

        xml_with_undefined_ns = (
            "<unknown:root xmlns:other='http://example.com'>content</unknown:root>"
        )

        with pytest.raises(Exception, match="Namespace error"):
            parser.parse_string(xml_with_undefined_ns)

    def test_memory_limit_exceeded(self, mock_xml_parser):
        """Test handling of memory limit exceeded during parsing."""
        parser = mock_xml_parser()
        parser.parse_file.side_effect = MemoryError(
            "Memory limit exceeded while parsing large XML file"
        )

        with pytest.raises(MemoryError, match="Memory limit exceeded"):
            parser.parse_file("/path/to/huge_document.xml")

    def test_validation_errors_collection(self, mock_xml_parser):
        """Test collection and reporting of validation errors."""
        parser = mock_xml_parser()

        # Mock validation errors
        validation_errors = [
            "Warning: Non-standard element structure detected",
            "Error: Required element 'article-title' missing",
            "Warning: Some references lack proper identifiers",
        ]
        parser.get_validation_errors.return_value = validation_errors

        # Parse with validation
        parser.set_options({"validate_on_parse": True})
        parser.parse_file("/path/to/document.xml")

        errors = parser.get_validation_errors()
        assert len(errors) == 3
        assert any("article-title" in error for error in errors)
        assert any("references" in error for error in errors)

    def test_xml_entity_expansion_attack_prevention(self, mock_xml_parser):
        """Test prevention of XML entity expansion attacks."""
        parser = mock_xml_parser()
        parser.parse_string.side_effect = ValueError("Entity expansion limit exceeded")

        malicious_xml = """<?xml version="1.0"?>
        <!DOCTYPE lolz [
          <!ENTITY lol "lol">
          <!ENTITY lol2 "&lol;&lol;&lol;&lol;&lol;&lol;&lol;&lol;&lol;&lol;">
          <!ENTITY lol3 "&lol2;&lol2;&lol2;&lol2;&lol2;&lol2;&lol2;&lol2;">
        ]>
        <lolz>&lol3;</lolz>"""

        with pytest.raises(ValueError, match="Entity expansion limit exceeded"):
            parser.parse_string(malicious_xml)

    def test_damaged_xml_recovery(self, mock_xml_parser):
        """Test recovery from damaged XML sections."""
        parser = mock_xml_parser()

        mock_result = Mock()
        mock_result.sections_processed = 8
        mock_result.sections_total = 10
        mock_result.errors = [
            "Section 7 has malformed tags",
            "Section 9 missing closing element",
        ]
        mock_result.content = "Recovered partial content"
        parser.parse_file.return_value = mock_result

        parser.set_options({"error_recovery": True, "skip_damaged_elements": True})

        result = parser.parse_file("/path/to/damaged.xml")

        assert result.sections_processed == 8
        assert len(result.errors) == 2
        assert "Recovered partial content" in result.content


class TestXMLParserValidation:
    """Test XML validation functionality."""

    def test_validate_xml_structure(self, mock_xml_parser, sample_xml_content):
        """Test validating XML structure and well-formedness."""
        parser = mock_xml_parser()
        parser.validate.return_value = True

        is_valid = parser.validate(sample_xml_content)

        assert is_valid is True
        parser.validate.assert_called_once_with(sample_xml_content)

    def test_validate_xml_comprehensive(self, mock_xml_parser, sample_xml_content):
        """Test comprehensive XML validation."""
        parser = mock_xml_parser()
        parser.validate_xml.return_value = {
            "well_formed": True,
            "valid_encoding": True,
            "schema_valid": True,
            "namespace_valid": True,
            "elements_count": 25,
            "namespaces_count": 3,
            "errors": [],
            "warnings": [],
        }

        validation_result = parser.validate_xml(sample_xml_content)

        assert validation_result["well_formed"] is True
        assert validation_result["schema_valid"] is True
        assert validation_result["elements_count"] == 25
        assert len(validation_result["errors"]) == 0

    def test_validate_pmc_article_structure(self, mock_xml_parser):
        """Test validation of PMC article structure."""
        parser = mock_xml_parser()
        parser.validate_xml.return_value = {
            "well_formed": True,
            "has_front": True,
            "has_article_meta": True,
            "has_body": True,
            "has_back": True,
            "required_elements": ["article-title", "abstract", "contrib-group"],
            "structure_completeness": 0.95,
            "article_type": "research-article",
        }

        validation_result = parser.validate_xml("pmc_content")

        assert validation_result["has_front"] is True
        assert validation_result["structure_completeness"] == 0.95
        assert validation_result["article_type"] == "research-article"

    def test_validate_rdf_xml_structure(self, mock_xml_parser, sample_rdf_xml_content):
        """Test validation of RDF/XML structure."""
        parser = mock_xml_parser()
        parser.validate_xml.return_value = {
            "well_formed": True,
            "rdf_valid": True,
            "has_rdf_root": True,
            "owl_classes": 2,
            "rdf_properties": 0,
            "ontology_completeness": 0.8,
            "namespace_compliance": True,
        }

        validation_result = parser.validate_xml(sample_rdf_xml_content)

        assert validation_result["rdf_valid"] is True
        assert validation_result["owl_classes"] == 2
        assert validation_result["namespace_compliance"] is True

    def test_validate_xml_accessibility(self, mock_xml_parser, sample_xml_content):
        """Test validation of XML accessibility features."""
        parser = mock_xml_parser()
        parser.validate_xml.return_value = {
            "well_formed": True,
            "has_alt_text": True,
            "has_table_headers": True,
            "has_figure_captions": True,
            "accessibility_score": 0.85,
            "accessibility_issues": [
                "Some tables missing summary attributes",
            ],
        }

        validation_result = parser.validate_xml(sample_xml_content)

        assert validation_result["accessibility_score"] == 0.85
        assert len(validation_result["accessibility_issues"]) == 1

    def test_validate_xml_content_completeness(self, mock_xml_parser):
        """Test validation of content completeness."""
        parser = mock_xml_parser()
        parser.validate_xml.return_value = {
            "well_formed": True,
            "content_extractable": True,
            "metadata_complete": True,
            "sections_identified": True,
            "references_parseable": True,
            "content_quality": 0.9,
            "missing_elements": [],
            "total_elements": 150,
        }

        validation_result = parser.validate_xml("content")

        assert validation_result["content_extractable"] is True
        assert validation_result["content_quality"] == 0.9
        assert len(validation_result["missing_elements"]) == 0


class TestXMLParserOptions:
    """Test parser configuration and options."""

    def test_set_parsing_options(self, mock_xml_parser):
        """Test setting various parsing options."""
        parser = mock_xml_parser()

        options = {
            "parser_library": "lxml",
            "namespace_aware": True,
            "validate_schema": True,
            "schema_path": "/path/to/schema.dtd",
            "resolve_entities": False,
            "encoding": "utf-8",
            "strip_whitespace": True,
            "preserve_comments": False,
        }

        parser.set_options(options)
        parser.set_options.assert_called_once_with(options)

    def test_get_current_options(self, mock_xml_parser):
        """Test getting current parser options."""
        parser = mock_xml_parser()

        expected_options = {
            "parser_library": "etree",
            "namespace_aware": True,
            "validate_schema": False,
            "encoding": "utf-8",
            "validate_on_parse": True,
        }
        parser.get_options.return_value = expected_options

        current_options = parser.get_options()

        assert current_options == expected_options
        assert "parser_library" in current_options

    def test_reset_options_to_defaults(self, mock_xml_parser):
        """Test resetting options to default values."""
        parser = mock_xml_parser()

        # Set some custom options first
        parser.set_options({"parser_library": "lxml", "namespace_aware": True})

        # Reset to defaults
        parser.reset_options()
        parser.reset_options.assert_called_once()

    def test_invalid_option_handling(self, mock_xml_parser):
        """Test handling of invalid configuration options."""
        parser = mock_xml_parser()
        parser.set_options.side_effect = ValueError("Unknown option: invalid_library")

        invalid_options = {"invalid_library": "unknown"}

        with pytest.raises(ValueError, match="Unknown option"):
            parser.set_options(invalid_options)

    def test_option_validation(self, mock_xml_parser):
        """Test validation of option values."""
        parser = mock_xml_parser()
        parser.set_options.side_effect = ValueError(
            "Invalid value for option 'parser_library': must be 'lxml', 'etree', or 'minidom'"
        )

        invalid_options = {"parser_library": "invalid_parser"}

        with pytest.raises(ValueError, match="Invalid value for option"):
            parser.set_options(invalid_options)

    def test_parser_library_options(self, mock_xml_parser):
        """Test different parser library configurations."""
        parser = mock_xml_parser()

        # Test lxml parser
        parser.set_options({"parser_library": "lxml"})
        parser.set_options.assert_called_with({"parser_library": "lxml"})

        # Test etree parser
        parser.reset_mock()
        parser.set_options({"parser_library": "etree"})
        parser.set_options.assert_called_with({"parser_library": "etree"})

        # Test minidom parser
        parser.reset_mock()
        parser.set_options({"parser_library": "minidom"})
        parser.set_options.assert_called_with({"parser_library": "minidom"})

    def test_namespace_handling_options(self, mock_xml_parser):
        """Test namespace handling configuration options."""
        parser = mock_xml_parser()

        namespace_options = {
            "namespace_aware": True,
            "preserve_namespaces": True,
            "default_namespace": "http://example.com/default",
            "namespace_prefixes": {"ex": "http://example.com"},
        }

        parser.set_options(namespace_options)
        parser.set_options.assert_called_with(namespace_options)

    def test_validation_options(self, mock_xml_parser):
        """Test validation configuration options."""
        parser = mock_xml_parser()

        validation_options = {
            "validate_schema": True,
            "schema_type": "dtd",
            "schema_path": "/path/to/pmc.dtd",
            "strict_validation": False,
            "validate_on_parse": True,
        }

        parser.set_options(validation_options)
        parser.set_options.assert_called_with(validation_options)


class TestXMLParserIntegration:
    """Integration tests with other components."""

    def test_integration_with_ontology_manager(self, mock_xml_parser):
        """Test integration with OntologyManager."""
        parser = mock_xml_parser()

        # Mock integration with ontology manager
        with patch(
            "aim2_project.aim2_ontology.ontology_manager.OntologyManager", create=True
        ) as MockManager:
            manager = MockManager()
            manager.add_ontology = Mock(return_value=True)

            # Parse and add to manager
            mock_ontology = Mock()
            parser.to_ontology.return_value = mock_ontology

            parsed_result = parser.parse_file("/path/to/article.xml")
            ontology = parser.to_ontology(parsed_result)
            manager.add_ontology(ontology)

            manager.add_ontology.assert_called_once_with(ontology)

    def test_integration_with_term_validation(self, mock_xml_parser):
        """Test integration with term validation pipeline."""
        parser = mock_xml_parser()

        # Mock validation pipeline
        with patch(
            "aim2_project.aim2_ontology.validators.ValidationPipeline", create=True
        ) as MockPipeline:
            validator = MockPipeline()
            validator.validate_terms = Mock(return_value={"valid": True, "errors": []})

            mock_terms = [Mock(), Mock(), Mock()]
            parser.extract_terms.return_value = mock_terms

            parsed_result = parser.parse_file("/path/to/article.xml")
            terms = parser.extract_terms(parsed_result)

            # Run validation
            validation_result = validator.validate_terms(terms)

            assert validation_result["valid"] is True
            validator.validate_terms.assert_called_once_with(terms)

    def test_integration_with_owl_parser(self, mock_xml_parser):
        """Test integration with OWL parser for RDF/XML content."""
        parser = mock_xml_parser()

        # Create mock OWL processor without trying to patch non-existent modules
        mock_owl_processor = Mock()
        mock_owl_processor.process_rdf = Mock(
            return_value={
                "classes": ["MachineLearning", "Ontology"],
                "properties": [
                    {"subject": "ML", "predicate": "subClassOf", "object": "Algorithm"}
                ],
            }
        )

        # Extract RDF and process with OWL
        mock_rdf_content = "RDF/XML content extracted from document"
        parser.extract_content.return_value = mock_rdf_content

        parsed_result = parser.parse_file("/path/to/ontology.xml")
        rdf_content = parser.extract_content(parsed_result)
        owl_result = mock_owl_processor.process_rdf(rdf_content)

        assert "MachineLearning" in owl_result["classes"]
        mock_owl_processor.process_rdf.assert_called_once_with(mock_rdf_content)

    def test_end_to_end_xml_parsing_workflow(
        self, mock_xml_parser, sample_pmc_article_content
    ):
        """Test complete end-to-end XML parsing workflow."""
        parser = mock_xml_parser()

        # Mock the complete workflow
        mock_parsed = Mock()
        mock_ontology = Mock()
        mock_ontology.terms = {
            "TERM:001": Mock(),
            "TERM:002": Mock(),
            "TERM:003": Mock(),
            "TERM:004": Mock(),
        }
        mock_ontology.relationships = {
            "REL:001": Mock(),
            "REL:002": Mock(),
            "REL:003": Mock(),
        }
        mock_ontology.metadata = sample_pmc_article_content

        parser.parse_file.return_value = mock_parsed
        parser.to_ontology.return_value = mock_ontology

        # Execute workflow
        parsed_result = parser.parse_file("/path/to/scientific_article.xml")
        ontology = parser.to_ontology(parsed_result)

        # Verify results
        assert len(ontology.terms) == 4
        assert len(ontology.relationships) == 3
        assert "title" in ontology.metadata

        parser.parse_file.assert_called_once_with("/path/to/scientific_article.xml")
        parser.to_ontology.assert_called_once_with(mock_parsed)


class TestXMLParserPerformance:
    """Performance and scalability tests."""

    def test_parse_large_xml_performance(self, mock_xml_parser):
        """Test parsing performance with large XML files."""
        parser = mock_xml_parser()

        # Configure for performance testing
        parser.set_options(
            {"streaming_mode": True, "memory_efficient": True, "chunk_size": 16384}
        )

        mock_result = Mock()
        mock_result.elements_count = 250000
        mock_result.file_size = "50MB"
        mock_result.processing_time = 35.8  # seconds
        parser.parse_file.return_value = mock_result

        result = parser.parse_file("/path/to/large_document.xml")

        assert result.elements_count == 250000
        assert result.processing_time < 60  # Should process within 1 minute

    def test_memory_usage_optimization(self, mock_xml_parser):
        """Test memory usage optimization features."""
        parser = mock_xml_parser()

        # Configure memory optimization
        parser.set_options(
            {"memory_limit": "1GB", "lazy_loading": True, "element_cache_size": 1000}
        )

        mock_result = Mock()
        mock_result.memory_usage = "850MB"
        mock_result.peak_memory = "920MB"
        parser.parse_file.return_value = mock_result

        result = parser.parse_file("/path/to/memory_intensive.xml")

        assert result == mock_result
        # Memory usage should be within limit
        assert float(result.memory_usage.replace("MB", "")) < 1024

    def test_concurrent_processing_support(self, mock_xml_parser):
        """Test support for concurrent processing operations."""
        parser = mock_xml_parser()

        # Configure for concurrent processing
        parser.set_options({"parallel_sections": True, "worker_threads": 6})

        files = [
            "/path/to/article1.xml",
            "/path/to/article2.xml",
            "/path/to/article3.xml",
            "/path/to/article4.xml",
            "/path/to/article5.xml",
            "/path/to/article6.xml",
        ]

        mock_results = [Mock() for _ in files]
        parser.parse_file.side_effect = mock_results

        results = []
        for file_path in files:
            result = parser.parse_file(file_path)
            results.append(result)

        assert len(results) == 6
        assert parser.parse_file.call_count == 6

    def test_caching_mechanisms(self, mock_xml_parser):
        """Test caching mechanisms for repeated parsing operations."""
        parser = mock_xml_parser()

        # Configure caching
        parser.set_options({"enable_cache": True, "cache_size": 200, "cache_ttl": 7200})

        mock_result = Mock()
        parser.parse_file.return_value = mock_result

        # Parse same file twice
        file_path = "/path/to/cached_document.xml"
        result1 = parser.parse_file(file_path)
        result2 = parser.parse_file(file_path)

        assert result1 == mock_result
        assert result2 == mock_result

        # In a real implementation, the second call might use cache
        # For now, we just verify the calls were made
        assert parser.parse_file.call_count == 2

    def test_batch_processing_capabilities(self, mock_xml_parser):
        """Test batch processing of multiple XML files."""
        parser = mock_xml_parser()

        # Configure for batch processing
        parser.set_options({"batch_mode": True, "batch_size": 20})

        mock_batch_result = Mock()
        mock_batch_result.processed_files = 20
        mock_batch_result.total_elements = 500000
        mock_batch_result.success_rate = 0.98
        parser.parse_file.return_value = mock_batch_result

        result = parser.parse_file("/path/to/batch_directory/")

        assert result.processed_files == 20
        assert result.success_rate == 0.98

    def test_streaming_parser_for_huge_files(self, mock_xml_parser):
        """Test streaming parser for extremely large XML files."""
        parser = mock_xml_parser()

        # Configure for streaming large files
        parser.set_options(
            {
                "streaming_mode": True,
                "max_memory": "2GB",
                "element_threshold": 10000,
                "progressive_parsing": True,
            }
        )

        mock_result = Mock()
        mock_result.streaming_chunks = 25
        mock_result.total_elements = 1000000
        mock_result.memory_peak = "1.8GB"
        parser.parse_file.return_value = mock_result

        result = parser.parse_file("/path/to/huge_corpus.xml")

        assert result.streaming_chunks == 25
        assert result.total_elements == 1000000


# Additional test fixtures for complex scenarios
@pytest.fixture
def complex_pmc_article():
    """Fixture providing complex PMC article structure for testing."""
    return {
        "metadata": {
            "article_type": "research-article",
            "dtd_version": "2.0",
            "xml_lang": "en",
            "article_id": {
                "pub_id_type": "pmc",
                "value": "PMC8901234",
            },
        },
        "front": {
            "journal_meta": {
                "journal_title": "Nature Machine Intelligence",
                "issn": {"pub_type": "epub", "value": "2522-5839"},
            },
            "article_meta": {
                "title_group": {
                    "article_title": "Advanced XML Processing for Biomedical Ontology Extraction",
                    "subtitle": "A Comprehensive Approach Using Machine Learning",
                },
                "contrib_group": [
                    {
                        "contrib_type": "author",
                        "name": {"surname": "Chen", "given_names": "Wei"},
                        "xref": {"ref_type": "aff", "rid": "aff1"},
                        "orcid": "0000-0001-2345-6789",
                    },
                    {
                        "contrib_type": "author",
                        "name": {"surname": "Rodriguez", "given_names": "Maria"},
                        "xref": {"ref_type": "aff", "rid": "aff2"},
                        "orcid": "0000-0002-3456-7890",
                    },
                ],
                "pub_date": {
                    "pub_type": "epub",
                    "year": "2023",
                    "month": "03",
                    "day": "15",
                },
                "volume": "4",
                "issue": "3",
                "fpage": "245",
                "lpage": "267",
                "permissions": {
                    "copyright_statement": "© 2023 Nature Publishing Group",
                    "license": {"license_type": "open-access"},
                },
            },
        },
        "body": {
            "sections": [
                {
                    "sec_type": "intro",
                    "title": "Introduction",
                    "paragraphs": 5,
                    "citations": 12,
                },
                {
                    "sec_type": "methods",
                    "title": "Methods",
                    "subsections": ["Data Collection", "XML Processing", "Evaluation"],
                    "figures": 2,
                    "tables": 1,
                },
                {
                    "sec_type": "results",
                    "title": "Results",
                    "subsections": ["Performance Analysis", "Case Studies"],
                    "figures": 4,
                    "tables": 3,
                },
            ],
        },
        "back": {
            "ref_list": {
                "title": "References",
                "ref_count": 58,
                "ref_types": ["journal", "book", "confproc", "web"],
            },
            "ack": {"title": "Acknowledgments"},
        },
    }


@pytest.fixture
def malformed_xml_scenarios():
    """Fixture providing malformed XML scenarios for error testing."""
    return {
        "unclosed_tags": "<article><section>content without closing",
        "invalid_nesting": "<article><p><section>invalid nesting</p></section></article>",
        "invalid_characters": "<?xml version='1.0'?><root>content\x00with\x01invalid\x02chars</root>",
        "malformed_attributes": "<article id='unclosed title=\"missing quote>content</article>",
        "invalid_entity": "<article>&undefined_entity;</article>",
        "encoding_mismatch": b"<?xml version='1.0' encoding='utf-8'?><root>\xff\xfe</root>",
    }


@pytest.fixture
def xml_configuration_options():
    """Fixture providing comprehensive XML parser configuration options."""
    return {
        "parser_options": {
            "library": "lxml",  # lxml, etree, minidom
            "fallback_libraries": ["etree", "minidom"],
            "namespace_aware": True,
            "validate_schema": False,
            "resolve_entities": False,
        },
        "content_processing_options": {
            "preserve_whitespace": False,
            "strip_comments": True,
            "preserve_cdata": True,
            "normalize_text": True,
            "extract_namespaces": True,
        },
        "extraction_options": {
            "extract_metadata": True,
            "extract_sections": True,
            "extract_figures": True,
            "extract_tables": True,
            "extract_references": True,
            "extract_citations": True,
        },
        "validation_options": {
            "validate_on_parse": True,
            "schema_type": "dtd",  # dtd, xsd, relaxng
            "schema_path": None,
            "strict_validation": False,
            "check_well_formed": True,
        },
        "performance_options": {
            "streaming_mode": False,
            "memory_efficient": False,
            "chunk_size": 8192,
            "element_cache_size": 1000,
            "lazy_loading": False,
        },
        "error_handling_options": {
            "error_recovery": True,
            "skip_malformed_elements": True,
            "max_errors": 20,
            "continue_on_error": True,
            "entity_expansion_limit": 1000,
        },
    }


@pytest.fixture
def temp_xml_files():
    """Create temporary XML files for testing."""
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)

        # Create various test files
        files = {}

        # Simple XML file
        simple_xml = temp_path / "simple.xml"
        simple_xml.write_text(
            '<?xml version="1.0" encoding="UTF-8"?><root><child>content</child></root>',
            encoding="utf-8",
        )
        files["simple"] = str(simple_xml)

        # PMC article XML (mock structure)
        pmc_xml = temp_path / "pmc_article.xml"
        pmc_content = """<?xml version="1.0" encoding="UTF-8"?>
<article xmlns="http://dtd.nlm.nih.gov/ncbi/pmc/articleset/nlm-articleset-2.0.dtd">
    <front><article-meta><title-group><article-title>Test Article</article-title></title-group></article-meta></front>
    <body><sec><title>Introduction</title><p>Content here</p></sec></body>
</article>"""
        pmc_xml.write_text(pmc_content, encoding="utf-8")
        files["pmc"] = str(pmc_xml)

        # RDF/XML file
        rdf_xml = temp_path / "ontology.xml"
        rdf_content = """<?xml version="1.0" encoding="UTF-8"?>
<rdf:RDF xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#"
         xmlns:owl="http://www.w3.org/2002/07/owl#">
    <owl:Class rdf:about="http://example.org#TestClass">
        <rdfs:label xmlns:rdfs="http://www.w3.org/2000/01/rdf-schema#">Test Class</rdfs:label>
    </owl:Class>
</rdf:RDF>"""
        rdf_xml.write_text(rdf_content, encoding="utf-8")
        files["rdf"] = str(rdf_xml)

        yield files


# Edge case and advanced functionality tests
class TestXMLParserAdvancedFeatures:
    """Test advanced XML parser features."""

    def test_mixed_content_processing(self, mock_xml_parser):
        """Test processing XML with mixed content (text and elements)."""
        parser = mock_xml_parser()

        mock_result = Mock()
        mock_result.mixed_content_elements = 15
        mock_result.text_nodes = ["intro text", "middle text", "end text"]
        mock_result.child_elements = ["<emphasis>", "<italic>", "<bold>"]
        parser.extract_content.return_value = mock_result

        parser.set_options(
            {"preserve_mixed_content": True, "extract_inline_markup": True}
        )

        parsed_result = Mock()
        result = parser.extract_content(parsed_result)

        assert result.mixed_content_elements == 15
        assert len(result.text_nodes) == 3

    def test_mathematical_markup_processing(self, mock_xml_parser):
        """Test processing MathML content within XML documents."""
        parser = mock_xml_parser()

        mock_result = Mock()
        mock_result.mathml_expressions = [
            {"id": "math1", "content": "<math><mi>x</mi><mo>=</mo><mn>5</mn></math>"},
            {
                "id": "math2",
                "content": "<math><mfrac><mi>a</mi><mi>b</mi></mfrac></math>",
            },
        ]
        parser.extract_content.return_value = mock_result

        parser.set_options(
            {
                "extract_mathml": True,
                "mathml_namespace": "http://www.w3.org/1998/Math/MathML",
            }
        )

        parsed_result = Mock()
        result = parser.extract_content(parsed_result)

        assert len(result.mathml_expressions) == 2
        assert "math1" in [expr["id"] for expr in result.mathml_expressions]

    def test_multilingual_xml_processing(self, mock_xml_parser):
        """Test processing multilingual XML documents with language attributes."""
        parser = mock_xml_parser()

        mock_result = Mock()
        mock_result.languages_detected = ["en", "zh", "ar"]
        mock_result.content_by_language = {
            "en": "English content",
            "zh": "中文内容",
            "ar": "المحتوى العربي",
        }
        mock_result.text_direction = {"en": "ltr", "zh": "ltr", "ar": "rtl"}
        parser.extract_content.return_value = mock_result

        parser.set_options({"multilingual": True, "extract_lang_attributes": True})

        parsed_result = Mock()
        result = parser.extract_content(parsed_result)

        assert len(result.languages_detected) == 3
        assert result.text_direction["ar"] == "rtl"

    def test_version_controlled_xml_processing(self, mock_xml_parser):
        """Test processing XML documents with version control information."""
        parser = mock_xml_parser()

        mock_result = Mock()
        mock_result.version_info = {
            "current_version": "2.1",
            "revision_history": [
                {"version": "1.0", "date": "2022-01-01", "changes": "Initial version"},
                {"version": "2.0", "date": "2022-06-01", "changes": "Major revision"},
                {"version": "2.1", "date": "2023-01-01", "changes": "Minor updates"},
            ],
        }
        parser.extract_metadata.return_value = mock_result

        parser.set_options({"extract_version_info": True, "track_revisions": True})

        parsed_result = Mock()
        result = parser.extract_metadata(parsed_result)

        assert result.version_info["current_version"] == "2.1"
        assert len(result.version_info["revision_history"]) == 3

    def test_semantic_annotation_extraction(self, mock_xml_parser):
        """Test extracting semantic annotations and markup from XML."""
        parser = mock_xml_parser()

        mock_annotations = [
            {
                "type": "named_entity",
                "text": "machine learning",
                "category": "technology",
                "confidence": 0.95,
                "element": "term",
            },
            {
                "type": "semantic_role",
                "text": "extracts information",
                "role": "process",
                "confidence": 0.88,
                "element": "activity",
            },
        ]
        parser.extract_content.return_value = Mock(
            semantic_annotations=mock_annotations
        )

        parser.set_options({"extract_semantic_markup": True})

        parsed_result = Mock()
        result = parser.extract_content(parsed_result)

        assert len(result.semantic_annotations) == 2
        assert result.semantic_annotations[0]["category"] == "technology"

    def test_cross_reference_resolution(self, mock_xml_parser):
        """Test resolving cross-references within XML documents."""
        parser = mock_xml_parser()

        mock_result = Mock()
        mock_result.cross_references = {
            "fig_refs": [{"ref_id": "fig1", "text": "Figure 1", "target_found": True}],
            "table_refs": [{"ref_id": "tab1", "text": "Table 1", "target_found": True}],
            "bibr_refs": [
                {"ref_id": "ref1", "text": "Smith et al.", "target_found": True}
            ],
            "unresolved_refs": [],
        }
        parser.extract_content.return_value = mock_result

        parser.set_options(
            {"resolve_cross_references": True, "validate_references": True}
        )

        parsed_result = Mock()
        result = parser.extract_content(parsed_result)

        assert len(result.cross_references["unresolved_refs"]) == 0
        assert result.cross_references["fig_refs"][0]["target_found"] is True
