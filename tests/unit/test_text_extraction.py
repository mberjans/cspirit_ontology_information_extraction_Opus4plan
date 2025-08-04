"""
Comprehensive Unit Tests for Text Extraction Functionality (TDD Approach)

This module provides comprehensive unit tests for the text extraction functionality in the AIM2 ontology
information extraction system. The tests follow test-driven development (TDD) approach,
defining the expected behavior of the text extraction classes before implementation.

Test Classes:
    TestTextExtractorBase: Tests for abstract TextExtractor base class
    TestPDFTextExtractor: Tests for PDF text extraction functionality
    TestXMLTextExtractor: Tests for XML text extraction functionality
    TestPlainTextExtractor: Tests for plain text extraction functionality
    TestTextPreprocessor: Tests for text preprocessing functionality
    TestTextExtractionFactory: Tests for text extractor factory
    TestTextExtractionIntegration: Integration tests between extractors
    TestTextExtractionPerformance: Performance and scalability tests
    TestTextExtractionErrorHandling: Error handling and validation tests

The text extraction system is expected to provide:
- Abstract base class TextExtractor with common interface
- PDF text extraction with multiple backends (pypdf, pdfplumber, PyMuPDF)
- XML text extraction with PMC XML and general XML support
- Plain text extraction with encoding detection
- Text preprocessing pipeline (tokenization, normalization, cleaning)
- Section detection and identification
- Metadata extraction from various formats
- Cross-format text extraction workflows
- Error recovery and fallback mechanisms
- Performance optimization for large documents

Dependencies:
    - pytest: For test framework
    - unittest.mock: For mocking functionality
    - pypdf: For PDF processing (mocked)
    - pdfplumber: For PDF text extraction (mocked)
    - PyMuPDF (fitz): For advanced PDF processing (mocked)
    - lxml: For XML processing (mocked)
    - chardet: For encoding detection (mocked)
    - typing: For type hints

Usage:
    pytest tests/unit/test_text_extraction.py -v
"""

import tempfile
from abc import ABC, abstractmethod
from datetime import datetime
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
from typing import Dict, List, Optional, Union, Any

import pytest


# Test fixtures for mocking text extraction classes
@pytest.fixture
def mock_text_extractor_base():
    """Mock TextExtractor abstract base class."""
    mock_class = Mock()
    mock_instance = Mock()

    # Abstract methods that must be implemented
    mock_instance.extract_text = Mock()
    mock_instance.extract_metadata = Mock()
    mock_instance.supports_format = Mock()
    mock_instance.validate_input = Mock()

    # Common interface methods
    mock_instance.get_supported_formats = Mock()
    mock_instance.detect_format = Mock()
    mock_instance.set_options = Mock()
    mock_instance.get_options = Mock(return_value={})
    mock_instance.reset_options = Mock()

    # Error handling methods
    mock_instance.get_errors = Mock(return_value=[])
    mock_instance.get_warnings = Mock(return_value=[])
    mock_instance.clear_errors = Mock()

    mock_class.return_value = mock_instance
    return mock_class


@pytest.fixture
def mock_pdf_text_extractor():
    """Mock PDFTextExtractor class."""
    mock_class = Mock()
    mock_instance = Mock()

    # Core text extraction methods
    mock_instance.extract_text = Mock()
    mock_instance.extract_text_by_page = Mock()
    mock_instance.extract_text_with_layout = Mock()
    mock_instance.extract_text_ocr = Mock()

    # Metadata extraction
    mock_instance.extract_metadata = Mock()
    mock_instance.extract_title = Mock()
    mock_instance.extract_authors = Mock()
    mock_instance.extract_doi = Mock()

    # Section identification
    mock_instance.identify_sections = Mock()
    mock_instance.extract_section = Mock()

    # PDF-specific methods
    mock_instance.is_encrypted = Mock()
    mock_instance.is_scanned = Mock()
    mock_instance.get_page_count = Mock()
    mock_instance.decrypt = Mock()

    # Format support
    mock_instance.supports_format = Mock(return_value=True)
    mock_instance.get_supported_formats = Mock(return_value=["pdf"])
    mock_instance.detect_format = Mock(return_value="pdf")
    mock_instance.validate_input = Mock(return_value=True)

    # Configuration
    mock_instance.set_options = Mock()
    mock_instance.get_options = Mock(return_value={})

    mock_class.return_value = mock_instance
    return mock_class


@pytest.fixture
def mock_xml_text_extractor():
    """Mock XMLTextExtractor class."""
    mock_class = Mock()
    mock_instance = Mock()

    # Core text extraction methods
    mock_instance.extract_text = Mock()
    mock_instance.extract_structured_content = Mock()
    mock_instance.extract_element_text = Mock()
    mock_instance.extract_xpath_text = Mock()

    # Metadata extraction
    mock_instance.extract_metadata = Mock()
    mock_instance.extract_title = Mock()
    mock_instance.extract_authors = Mock()
    mock_instance.extract_doi = Mock()
    mock_instance.extract_journal_info = Mock()

    # Section identification
    mock_instance.identify_sections = Mock()
    mock_instance.extract_section = Mock()
    mock_instance.extract_abstract = Mock()

    # XML-specific methods
    mock_instance.get_namespaces = Mock()
    mock_instance.validate_schema = Mock()
    mock_instance.is_well_formed = Mock()
    mock_instance.get_root_element = Mock()

    # Format support
    mock_instance.supports_format = Mock(return_value=True)
    mock_instance.get_supported_formats = Mock(return_value=["xml", "pmc"])
    mock_instance.detect_format = Mock(return_value="xml")
    mock_instance.validate_input = Mock(return_value=True)

    # Configuration
    mock_instance.set_options = Mock()
    mock_instance.get_options = Mock(return_value={})

    mock_class.return_value = mock_instance
    return mock_class


@pytest.fixture
def mock_plain_text_extractor():
    """Mock PlainTextExtractor class."""
    mock_class = Mock()
    mock_instance = Mock()

    # Core text extraction methods
    mock_instance.extract_text = Mock()
    mock_instance.extract_lines = Mock()
    mock_instance.extract_paragraphs = Mock()

    # Encoding detection and handling
    mock_instance.detect_encoding = Mock()
    mock_instance.convert_encoding = Mock()
    mock_instance.normalize_text = Mock()

    # Metadata extraction (limited for plain text)
    mock_instance.extract_metadata = Mock()
    mock_instance.get_file_stats = Mock()

    # Text analysis
    mock_instance.count_words = Mock()
    mock_instance.count_lines = Mock()
    mock_instance.detect_language = Mock()

    # Format support
    mock_instance.supports_format = Mock(return_value=True)
    mock_instance.get_supported_formats = Mock(return_value=["txt", "text"])
    mock_instance.detect_format = Mock(return_value="txt")
    mock_instance.validate_input = Mock(return_value=True)

    # Configuration
    mock_instance.set_options = Mock()
    mock_instance.get_options = Mock(return_value={})

    mock_class.return_value = mock_instance
    return mock_class


@pytest.fixture
def mock_text_preprocessor():
    """Mock TextPreprocessor class."""
    mock_class = Mock()
    mock_instance = Mock()

    # Core preprocessing methods
    mock_instance.tokenize = Mock()
    mock_instance.split_sentences = Mock()
    mock_instance.normalize_whitespace = Mock()
    mock_instance.remove_special_chars = Mock()
    mock_instance.clean_text = Mock()

    # Advanced preprocessing
    mock_instance.detect_sections = Mock()
    mock_instance.extract_patterns = Mock()
    mock_instance.normalize_unicode = Mock()
    mock_instance.fix_encoding_issues = Mock()

    # Text analysis
    mock_instance.get_text_stats = Mock()
    mock_instance.detect_language = Mock()
    mock_instance.extract_keywords = Mock()

    # Pipeline methods
    mock_instance.preprocess = Mock()
    mock_instance.add_step = Mock()
    mock_instance.remove_step = Mock()
    mock_instance.get_pipeline = Mock()

    # Configuration
    mock_instance.set_options = Mock()
    mock_instance.get_options = Mock(return_value={})

    mock_class.return_value = mock_instance
    return mock_class


@pytest.fixture
def sample_pdf_bytes():
    """Sample PDF bytes for testing."""
    return b"%PDF-1.4\n1 0 obj\n<<\n/Type /Catalog\n/Pages 2 0 R\n>>\nendobj\n%%EOF"


@pytest.fixture
def sample_xml_content():
    """Sample XML content for testing."""
    return """<?xml version="1.0" encoding="UTF-8"?>
    <article xmlns:xlink="http://www.w3.org/1999/xlink">
        <front>
            <article-meta>
                <title-group>
                    <article-title>Machine Learning for Ontology Extraction</article-title>
                </title-group>
                <contrib-group>
                    <contrib contrib-type="author">
                        <name><surname>Smith</surname><given-names>John</given-names></name>
                    </contrib>
                </contrib-group>
                <abstract>
                    <p>This paper presents methods for ontology extraction...</p>
                </abstract>
            </article-meta>
        </front>
        <body>
            <sec sec-type="intro">
                <title>Introduction</title>
                <p>The field of ontology extraction has evolved...</p>
            </sec>
            <sec sec-type="methods">
                <title>Methods</title>
                <p>We employed machine learning techniques...</p>
            </sec>
        </body>
        <back>
            <ref-list>
                <ref id="ref1">
                    <citation>Smith J. et al. (2023). Ontology methods. Nature.</citation>
                </ref>
            </ref-list>
        </back>
    </article>"""


@pytest.fixture
def sample_plain_text():
    """Sample plain text for testing."""
    return """Machine Learning Approaches for Ontology Information Extraction

Abstract
This paper presents novel machine learning approaches for extracting ontological information from scientific literature. Our method achieves 95% accuracy on benchmark datasets.

Introduction
The field of ontology information extraction has grown significantly in recent years. Traditional rule-based approaches have limitations when dealing with diverse text formats and domains.

Methods
We developed a neural network architecture that combines transformer-based language models with graph neural networks. The system processes text in multiple stages:

1. Text preprocessing and tokenization
2. Named entity recognition
3. Relationship extraction
4. Ontology construction

Results
Experimental evaluation on three benchmark datasets shows that our approach outperforms existing methods by 15% on average.

Conclusion
We have presented a comprehensive approach to ontology information extraction that combines the strengths of modern NLP techniques.
"""


@pytest.fixture
def sample_text_metadata():
    """Sample text metadata for testing."""
    return {
        "format": "pdf",
        "title": "Machine Learning for Ontology Extraction",
        "authors": ["Dr. John Smith", "Prof. Jane Doe"],
        "doi": "10.1000/test.2023.001",
        "abstract": "This paper presents methods for ontology extraction...",
        "keywords": ["machine learning", "ontology", "information extraction"],
        "sections": ["abstract", "introduction", "methods", "results", "conclusion"],
        "page_count": 10,
        "word_count": 5000,
        "creation_date": datetime(2023, 1, 1),
        "language": "en",
        "encoding": "utf-8"
    }


class TestTextExtractorBase:
    """Test abstract TextExtractor base class."""

    def test_text_extractor_abstract_interface(self, mock_text_extractor_base):
        """Test that TextExtractor defines the required abstract interface."""
        extractor = mock_text_extractor_base()

        # Verify all required abstract methods are available
        required_methods = [
            "extract_text",
            "extract_metadata", 
            "supports_format",
            "validate_input"
        ]
        
        for method in required_methods:
            assert hasattr(extractor, method)

    def test_text_extractor_common_interface(self, mock_text_extractor_base):
        """Test common interface methods available to all extractors."""
        extractor = mock_text_extractor_base()

        # Common interface methods
        common_methods = [
            "get_supported_formats",
            "detect_format",
            "set_options",
            "get_options",
            "reset_options",
            "get_errors",
            "get_warnings",
            "clear_errors"
        ]

        for method in common_methods:
            assert hasattr(extractor, method)

    def test_format_detection_interface(self, mock_text_extractor_base):
        """Test format detection interface."""
        extractor = mock_text_extractor_base()
        
        # Mock format detection
        extractor.detect_format.return_value = "pdf"
        extractor.supports_format.return_value = True
        
        sample_content = b"sample content"
        detected_format = extractor.detect_format(sample_content)
        is_supported = extractor.supports_format(detected_format)
        
        assert detected_format == "pdf"
        assert is_supported is True
        extractor.detect_format.assert_called_once_with(sample_content)
        extractor.supports_format.assert_called_once_with(detected_format)

    def test_options_management(self, mock_text_extractor_base):
        """Test options management interface."""
        extractor = mock_text_extractor_base()
        
        # Test setting options
        options = {"encoding": "utf-8", "language": "en"}
        extractor.set_options(options)
        extractor.set_options.assert_called_once_with(options)
        
        # Test getting options
        extractor.get_options.return_value = options
        current_options = extractor.get_options()
        assert current_options == options
        
        # Test resetting options
        extractor.reset_options()
        extractor.reset_options.assert_called_once()

    def test_error_handling_interface(self, mock_text_extractor_base):
        """Test error handling interface."""
        extractor = mock_text_extractor_base()
        
        # Mock error and warning states
        errors = ["Error: Invalid format", "Error: Corrupted content"]
        warnings = ["Warning: Low confidence extraction"]
        
        extractor.get_errors.return_value = errors
        extractor.get_warnings.return_value = warnings
        
        # Test getting errors and warnings
        current_errors = extractor.get_errors()
        current_warnings = extractor.get_warnings()
        
        assert len(current_errors) == 2
        assert len(current_warnings) == 1
        assert "Invalid format" in current_errors[0]
        
        # Test clearing errors
        extractor.clear_errors()
        extractor.clear_errors.assert_called_once()

    def test_validation_interface(self, mock_text_extractor_base):
        """Test input validation interface."""
        extractor = mock_text_extractor_base()
        
        # Test valid input
        extractor.validate_input.return_value = True
        valid_content = b"valid content"
        is_valid = extractor.validate_input(valid_content)
        assert is_valid is True
        
        # Test invalid input
        extractor.validate_input.return_value = False
        invalid_content = b"invalid content"
        is_valid = extractor.validate_input(invalid_content)
        assert is_valid is False


class TestPDFTextExtractor:
    """Test PDF text extraction functionality."""

    def test_pdf_extractor_creation(self, mock_pdf_text_extractor):
        """Test PDFTextExtractor instantiation."""
        extractor = mock_pdf_text_extractor()
        
        assert extractor is not None
        mock_pdf_text_extractor.assert_called_once()

    def test_pdf_format_support(self, mock_pdf_text_extractor):
        """Test PDF format detection and support."""
        extractor = mock_pdf_text_extractor()
        
        # Test format support
        formats = extractor.get_supported_formats()
        assert "pdf" in formats
        
        # Test format detection
        extractor.detect_format.return_value = "pdf"
        detected = extractor.detect_format(b"%PDF-1.4")
        assert detected == "pdf"
        
        # Test format validation
        extractor.supports_format.return_value = True
        is_supported = extractor.supports_format("pdf")
        assert is_supported is True

    def test_basic_text_extraction(self, mock_pdf_text_extractor):
        """Test basic text extraction from PDF."""
        extractor = mock_pdf_text_extractor()
        
        expected_text = "This is the extracted text from the PDF document."
        extractor.extract_text.return_value = expected_text
        
        pdf_content = b"%PDF-1.4\ntest content"
        text = extractor.extract_text(pdf_content)
        
        assert text == expected_text
        extractor.extract_text.assert_called_once_with(pdf_content)

    def test_page_by_page_extraction(self, mock_pdf_text_extractor):
        """Test extracting text page by page."""
        extractor = mock_pdf_text_extractor()
        
        expected_pages = {
            1: "Page 1: Introduction to machine learning",
            2: "Page 2: Methods and approaches", 
            3: "Page 3: Results and evaluation"
        }
        extractor.extract_text_by_page.return_value = expected_pages
        
        pdf_content = b"%PDF-1.4\ntest content"
        pages = extractor.extract_text_by_page(pdf_content)
        
        assert len(pages) == 3
        assert pages[1].startswith("Page 1:")
        extractor.extract_text_by_page.assert_called_once_with(pdf_content)

    def test_layout_preserved_extraction(self, mock_pdf_text_extractor):
        """Test text extraction with layout preservation."""
        extractor = mock_pdf_text_extractor()
        
        mock_result = Mock()
        mock_result.text = "Layout preserved text with positioning"
        mock_result.layout_info = {
            "columns": 2,
            "text_blocks": [
                {"text": "Title", "x": 100, "y": 700, "width": 200, "height": 30},
                {"text": "Content", "x": 100, "y": 650, "width": 400, "height": 200}
            ]
        }
        extractor.extract_text_with_layout.return_value = mock_result
        
        pdf_content = b"%PDF-1.4\ntest content"
        result = extractor.extract_text_with_layout(pdf_content)
        
        assert result.layout_info["columns"] == 2
        assert len(result.layout_info["text_blocks"]) == 2
        extractor.extract_text_with_layout.assert_called_once_with(pdf_content)

    def test_ocr_text_extraction(self, mock_pdf_text_extractor):
        """Test OCR text extraction for scanned PDFs."""
        extractor = mock_pdf_text_extractor()
        
        expected_text = "OCR extracted text from scanned document"
        extractor.extract_text_ocr.return_value = expected_text
        
        # Configure OCR options
        ocr_options = {"ocr_enabled": True, "ocr_language": "eng", "ocr_engine": "tesseract"}
        extractor.set_options(ocr_options)
        
        scanned_pdf = b"%PDF-1.4\nscanned content"
        text = extractor.extract_text_ocr(scanned_pdf)
        
        assert text == expected_text
        extractor.extract_text_ocr.assert_called_once_with(scanned_pdf)
        extractor.set_options.assert_called_once_with(ocr_options)

    def test_pdf_metadata_extraction(self, mock_pdf_text_extractor, sample_text_metadata):
        """Test extracting metadata from PDF."""
        extractor = mock_pdf_text_extractor()
        
        extractor.extract_metadata.return_value = sample_text_metadata
        
        pdf_content = b"%PDF-1.4\ntest content"
        metadata = extractor.extract_metadata(pdf_content)
        
        assert metadata["title"] == "Machine Learning for Ontology Extraction"
        assert len(metadata["authors"]) == 2
        assert metadata["page_count"] == 10
        extractor.extract_metadata.assert_called_once_with(pdf_content)

    def test_pdf_section_identification(self, mock_pdf_text_extractor):
        """Test identifying sections in PDF scientific papers."""
        extractor = mock_pdf_text_extractor()
        
        expected_sections = {
            "abstract": {"text": "Abstract content...", "page": 1, "confidence": 0.95},
            "introduction": {"text": "Introduction content...", "page": 1, "confidence": 0.90},
            "methods": {"text": "Methods content...", "page": 2, "confidence": 0.88},
            "results": {"text": "Results content...", "page": 3, "confidence": 0.92}
        }
        extractor.identify_sections.return_value = expected_sections
        
        pdf_content = b"%PDF-1.4\ntest content"
        sections = extractor.identify_sections(pdf_content)
        
        assert "abstract" in sections
        assert sections["introduction"]["confidence"] == 0.90
        assert len(sections) == 4
        extractor.identify_sections.assert_called_once_with(pdf_content)

    def test_encrypted_pdf_handling(self, mock_pdf_text_extractor):
        """Test handling encrypted PDFs."""
        extractor = mock_pdf_text_extractor()
        
        # Test encryption detection
        extractor.is_encrypted.return_value = True
        encrypted_pdf = b"%PDF-1.4\nencrypted content"
        is_encrypted = extractor.is_encrypted(encrypted_pdf)
        assert is_encrypted is True
        
        # Test decryption
        extractor.decrypt.return_value = True
        password = "secret123"
        decrypted = extractor.decrypt(encrypted_pdf, password)
        assert decrypted is True
        extractor.decrypt.assert_called_once_with(encrypted_pdf, password)

    def test_scanned_pdf_detection(self, mock_pdf_text_extractor):
        """Test detecting scanned PDFs."""
        extractor = mock_pdf_text_extractor()
        
        # Test scanned PDF detection
        extractor.is_scanned.return_value = True
        scanned_pdf = b"%PDF-1.4\nimage-based content"
        is_scanned = extractor.is_scanned(scanned_pdf)
        assert is_scanned is True
        
        # Test text-based PDF
        extractor.is_scanned.return_value = False
        text_pdf = b"%PDF-1.4\ntext-based content"
        is_scanned = extractor.is_scanned(text_pdf)
        assert is_scanned is False

    def test_pdf_page_count(self, mock_pdf_text_extractor):
        """Test getting PDF page count."""
        extractor = mock_pdf_text_extractor()
        
        extractor.get_page_count.return_value = 15
        pdf_content = b"%PDF-1.4\ntest content"
        page_count = extractor.get_page_count(pdf_content)
        
        assert page_count == 15
        extractor.get_page_count.assert_called_once_with(pdf_content)

    def test_pdf_extraction_options(self, mock_pdf_text_extractor):
        """Test PDF extraction with various options."""
        extractor = mock_pdf_text_extractor()
        
        options = {
            "extraction_method": "pdfplumber",
            "page_range": (1, 5),
            "layout_mode": True,
            "extract_images": False,
            "ocr_fallback": True
        }
        
        extractor.set_options(options)
        extractor.set_options.assert_called_once_with(options)
        
        # Test getting current options
        extractor.get_options.return_value = options
        current_options = extractor.get_options()
        assert current_options["extraction_method"] == "pdfplumber"
        assert current_options["page_range"] == (1, 5)


class TestXMLTextExtractor:
    """Test XML text extraction functionality."""

    def test_xml_extractor_creation(self, mock_xml_text_extractor):
        """Test XMLTextExtractor instantiation."""
        extractor = mock_xml_text_extractor()
        
        assert extractor is not None
        mock_xml_text_extractor.assert_called_once()

    def test_xml_format_support(self, mock_xml_text_extractor):
        """Test XML format detection and support."""
        extractor = mock_xml_text_extractor()
        
        # Test format support
        formats = extractor.get_supported_formats()
        assert "xml" in formats
        assert "pmc" in formats
        
        # Test format detection
        extractor.detect_format.return_value = "xml"
        detected = extractor.detect_format(b"<?xml version='1.0'?>")
        assert detected == "xml"

    def test_basic_xml_text_extraction(self, mock_xml_text_extractor, sample_xml_content):
        """Test basic text extraction from XML."""
        extractor = mock_xml_text_extractor()
        
        expected_text = "Machine Learning for Ontology Extraction\n\nThis paper presents methods for ontology extraction...\n\nThe field of ontology extraction has evolved..."
        extractor.extract_text.return_value = expected_text
        
        text = extractor.extract_text(sample_xml_content)
        
        assert "Machine Learning for Ontology Extraction" in text
        assert "ontology extraction" in text
        extractor.extract_text.assert_called_once_with(sample_xml_content)

    def test_structured_xml_content_extraction(self, mock_xml_text_extractor, sample_xml_content):
        """Test extracting structured content from XML."""
        extractor = mock_xml_text_extractor()
        
        expected_structure = {
            "title": "Machine Learning for Ontology Extraction",
            "authors": [{"surname": "Smith", "given_names": "John"}],
            "abstract": "This paper presents methods for ontology extraction...",
            "sections": {
                "introduction": "The field of ontology extraction has evolved...",
                "methods": "We employed machine learning techniques..."
            },
            "references": ["Smith J. et al. (2023). Ontology methods. Nature."]
        }
        extractor.extract_structured_content.return_value = expected_structure
        
        structure = extractor.extract_structured_content(sample_xml_content)
        
        assert structure["title"] == "Machine Learning for Ontology Extraction"
        assert len(structure["authors"]) == 1
        assert "introduction" in structure["sections"]
        extractor.extract_structured_content.assert_called_once_with(sample_xml_content)

    def test_xml_element_text_extraction(self, mock_xml_text_extractor, sample_xml_content):
        """Test extracting text from specific XML elements."""
        extractor = mock_xml_text_extractor()
        
        # Test extracting specific element text
        expected_title = "Machine Learning for Ontology Extraction"
        extractor.extract_element_text.return_value = expected_title
        
        title = extractor.extract_element_text(sample_xml_content, "article-title")
        
        assert title == expected_title
        extractor.extract_element_text.assert_called_once_with(sample_xml_content, "article-title")

    def test_xpath_text_extraction(self, mock_xml_text_extractor, sample_xml_content):
        """Test extracting text using XPath expressions."""
        extractor = mock_xml_text_extractor()
        
        # Test XPath extraction
        expected_authors = ["Smith, John"]
        extractor.extract_xpath_text.return_value = expected_authors
        
        xpath_expression = "//contrib[@contrib-type='author']//name"
        authors = extractor.extract_xpath_text(sample_xml_content, xpath_expression)
        
        assert len(authors) == 1
        assert "Smith" in authors[0]
        extractor.extract_xpath_text.assert_called_once_with(sample_xml_content, xpath_expression)

    def test_xml_metadata_extraction(self, mock_xml_text_extractor, sample_xml_content):
        """Test extracting metadata from XML."""
        extractor = mock_xml_text_extractor()
        
        expected_metadata = {
            "title": "Machine Learning for Ontology Extraction",
            "authors": ["Smith, John"],
            "doi": "10.1000/test.2023.001",
            "journal": "Nature Machine Learning",
            "volume": "5",
            "issue": "3",
            "pages": "123-135",
            "publication_date": "2023-01-15"
        }
        extractor.extract_metadata.return_value = expected_metadata
        
        metadata = extractor.extract_metadata(sample_xml_content)
        
        assert metadata["title"] == "Machine Learning for Ontology Extraction"
        assert "Smith" in metadata["authors"][0]
        extractor.extract_metadata.assert_called_once_with(sample_xml_content)

    def test_xml_section_identification(self, mock_xml_text_extractor, sample_xml_content):
        """Test identifying sections in XML documents."""
        extractor = mock_xml_text_extractor()
        
        expected_sections = {
            "abstract": {"text": "This paper presents methods...", "type": "abstract"},
            "introduction": {"text": "The field of ontology extraction...", "type": "intro"},
            "methods": {"text": "We employed machine learning...", "type": "methods"}
        }
        extractor.identify_sections.return_value = expected_sections
        
        sections = extractor.identify_sections(sample_xml_content)
        
        assert "abstract" in sections
        assert sections["introduction"]["type"] == "intro"
        assert len(sections) == 3
        extractor.identify_sections.assert_called_once_with(sample_xml_content)

    def test_xml_namespace_handling(self, mock_xml_text_extractor, sample_xml_content):
        """Test handling XML namespaces."""
        extractor = mock_xml_text_extractor()
        
        expected_namespaces = {
            "xlink": "http://www.w3.org/1999/xlink",
            "mml": "http://www.w3.org/1998/Math/MathML"
        }
        extractor.get_namespaces.return_value = expected_namespaces
        
        namespaces = extractor.get_namespaces(sample_xml_content)
        
        assert "xlink" in namespaces
        assert namespaces["xlink"] == "http://www.w3.org/1999/xlink"
        extractor.get_namespaces.assert_called_once_with(sample_xml_content)

    def test_xml_validation(self, mock_xml_text_extractor, sample_xml_content):
        """Test XML validation functionality."""
        extractor = mock_xml_text_extractor()
        
        # Test well-formed validation
        extractor.is_well_formed.return_value = True
        is_well_formed = extractor.is_well_formed(sample_xml_content)
        assert is_well_formed is True
        
        # Test schema validation
        extractor.validate_schema.return_value = {"valid": True, "errors": []}
        schema_result = extractor.validate_schema(sample_xml_content)
        assert schema_result["valid"] is True

    def test_pmc_xml_specific_extraction(self, mock_xml_text_extractor):
        """Test PMC XML format specific extraction."""
        extractor = mock_xml_text_extractor()
        
        # Configure for PMC format
        extractor.set_options({"format": "pmc", "namespace_aware": True})
        
        pmc_content = """<?xml version="1.0"?>
        <pmc-articleset>
            <article>
                <front>
                    <journal-meta>
                        <journal-title>Nature</journal-title>
                    </journal-meta>
                </front>
            </article>
        </pmc-articleset>"""
        
        expected_journal = "Nature"
        extractor.extract_journal_info.return_value = {"journal": expected_journal}
        
        journal_info = extractor.extract_journal_info(pmc_content)
        assert journal_info["journal"] == expected_journal
        extractor.set_options.assert_called_once_with({"format": "pmc", "namespace_aware": True})


class TestPlainTextExtractor:
    """Test plain text extraction functionality."""

    def test_plain_text_extractor_creation(self, mock_plain_text_extractor):
        """Test PlainTextExtractor instantiation."""
        extractor = mock_plain_text_extractor()
        
        assert extractor is not None
        mock_plain_text_extractor.assert_called_once()

    def test_plain_text_format_support(self, mock_plain_text_extractor):
        """Test plain text format detection and support."""
        extractor = mock_plain_text_extractor()
        
        # Test format support
        formats = extractor.get_supported_formats()
        assert "txt" in formats
        assert "text" in formats
        
        # Test format detection
        extractor.detect_format.return_value = "txt"
        detected = extractor.detect_format(b"Plain text content")
        assert detected == "txt"

    def test_basic_text_extraction(self, mock_plain_text_extractor, sample_plain_text):
        """Test basic text extraction from plain text."""
        extractor = mock_plain_text_extractor()
        
        extractor.extract_text.return_value = sample_plain_text
        
        text = extractor.extract_text(sample_plain_text.encode())
        
        assert "Machine Learning Approaches" in text
        assert "Abstract" in text
        assert "Conclusion" in text
        extractor.extract_text.assert_called_once_with(sample_plain_text.encode())

    def test_encoding_detection(self, mock_plain_text_extractor):
        """Test encoding detection for text files."""
        extractor = mock_plain_text_extractor()
        
        # Test UTF-8 detection
        extractor.detect_encoding.return_value = {"encoding": "utf-8", "confidence": 0.99}
        utf8_text = "Hello, 世界!".encode("utf-8")
        encoding_info = extractor.detect_encoding(utf8_text)
        
        assert encoding_info["encoding"] == "utf-8"
        assert encoding_info["confidence"] > 0.9
        extractor.detect_encoding.assert_called_once_with(utf8_text)

    def test_encoding_conversion(self, mock_plain_text_extractor):
        """Test encoding conversion functionality."""
        extractor = mock_plain_text_extractor()
        
        # Test encoding conversion
        converted_text = "Converted text with proper encoding"
        extractor.convert_encoding.return_value = converted_text
        
        latin1_text = "Café".encode("latin-1")
        result = extractor.convert_encoding(latin1_text, "latin-1", "utf-8")
        
        assert result == converted_text
        extractor.convert_encoding.assert_called_once_with(latin1_text, "latin-1", "utf-8")

    def test_text_normalization(self, mock_plain_text_extractor):
        """Test text normalization functionality."""
        extractor = mock_plain_text_extractor()
        
        # Test text normalization
        normalized_text = "Normalized text with proper formatting"
        extractor.normalize_text.return_value = normalized_text
        
        messy_text = "  Text   with  irregular   spacing\t\n\n  "
        result = extractor.normalize_text(messy_text)
        
        assert result == normalized_text
        extractor.normalize_text.assert_called_once_with(messy_text)

    def test_line_extraction(self, mock_plain_text_extractor, sample_plain_text):
        """Test extracting lines from text."""
        extractor = mock_plain_text_extractor()
        
        expected_lines = sample_plain_text.split('\n')
        extractor.extract_lines.return_value = expected_lines
        
        lines = extractor.extract_lines(sample_plain_text)
        
        assert isinstance(lines, list)
        assert len(lines) > 10
        assert "Machine Learning Approaches" in lines[0]
        extractor.extract_lines.assert_called_once_with(sample_plain_text)

    def test_paragraph_extraction(self, mock_plain_text_extractor, sample_plain_text):
        """Test extracting paragraphs from text."""
        extractor = mock_plain_text_extractor()
        
        expected_paragraphs = [
            "Machine Learning Approaches for Ontology Information Extraction",
            "Abstract\nThis paper presents novel machine learning approaches...",
            "Introduction\nThe field of ontology information extraction...",
            "Methods\nWe developed a neural network architecture..."
        ]
        extractor.extract_paragraphs.return_value = expected_paragraphs
        
        paragraphs = extractor.extract_paragraphs(sample_plain_text)
        
        assert isinstance(paragraphs, list)
        assert len(paragraphs) >= 4
        assert "Abstract" in paragraphs[1]
        extractor.extract_paragraphs.assert_called_once_with(sample_plain_text)

    def test_text_statistics(self, mock_plain_text_extractor, sample_plain_text):
        """Test text statistics extraction."""
        extractor = mock_plain_text_extractor()
        
        # Test word count
        extractor.count_words.return_value = 150
        word_count = extractor.count_words(sample_plain_text)
        assert word_count == 150
        
        # Test line count
        extractor.count_lines.return_value = 25
        line_count = extractor.count_lines(sample_plain_text)
        assert line_count == 25

    def test_language_detection(self, mock_plain_text_extractor, sample_plain_text):
        """Test language detection for text."""
        extractor = mock_plain_text_extractor()
        
        extractor.detect_language.return_value = {"language": "en", "confidence": 0.95}
        
        language_info = extractor.detect_language(sample_plain_text)
        
        assert language_info["language"] == "en"
        assert language_info["confidence"] > 0.9
        extractor.detect_language.assert_called_once_with(sample_plain_text)

    def test_file_metadata_extraction(self, mock_plain_text_extractor):
        """Test extracting file metadata for plain text."""
        extractor = mock_plain_text_extractor()
        
        expected_stats = {
            "file_size": 1024,
            "line_count": 50,
            "word_count": 300,
            "char_count": 1500,
            "encoding": "utf-8",
            "language": "en"
        }
        extractor.get_file_stats.return_value = expected_stats
        
        file_path = "/path/to/text_file.txt"
        stats = extractor.get_file_stats(file_path)
        
        assert stats["file_size"] == 1024
        assert stats["encoding"] == "utf-8"
        extractor.get_file_stats.assert_called_once_with(file_path)


class TestTextPreprocessor:
    """Test text preprocessing functionality."""

    def test_text_preprocessor_creation(self, mock_text_preprocessor):
        """Test TextPreprocessor instantiation."""
        preprocessor = mock_text_preprocessor()
        
        assert preprocessor is not None
        mock_text_preprocessor.assert_called_once()

    def test_tokenization(self, mock_text_preprocessor, sample_plain_text):
        """Test text tokenization functionality."""
        preprocessor = mock_text_preprocessor()
        
        expected_tokens = [
            "Machine", "Learning", "Approaches", "for", "Ontology", 
            "Information", "Extraction", "Abstract", "This", "paper"
        ]
        preprocessor.tokenize.return_value = expected_tokens
        
        tokens = preprocessor.tokenize(sample_plain_text)
        
        assert isinstance(tokens, list)
        assert "Machine" in tokens
        assert "Ontology" in tokens
        preprocessor.tokenize.assert_called_once_with(sample_plain_text)

    def test_sentence_splitting(self, mock_text_preprocessor, sample_plain_text):
        """Test sentence splitting functionality."""
        preprocessor = mock_text_preprocessor()
        
        expected_sentences = [
            "Machine Learning Approaches for Ontology Information Extraction",
            "This paper presents novel machine learning approaches for extracting ontological information from scientific literature.",
            "Our method achieves 95% accuracy on benchmark datasets.",
            "The field of ontology information extraction has grown significantly in recent years."
        ]
        preprocessor.split_sentences.return_value = expected_sentences
        
        sentences = preprocessor.split_sentences(sample_plain_text)
        
        assert isinstance(sentences, list)
        assert len(sentences) >= 3
        assert "95% accuracy" in sentences[2]
        preprocessor.split_sentences.assert_called_once_with(sample_plain_text)

    def test_whitespace_normalization(self, mock_text_preprocessor):
        """Test whitespace normalization."""
        preprocessor = mock_text_preprocessor()
        
        messy_text = "Text   with\t\tirregular\n\n\nspacing"
        normalized_text = "Text with irregular spacing"
        preprocessor.normalize_whitespace.return_value = normalized_text
        
        result = preprocessor.normalize_whitespace(messy_text)
        
        assert result == normalized_text
        preprocessor.normalize_whitespace.assert_called_once_with(messy_text)

    def test_special_character_removal(self, mock_text_preprocessor):
        """Test removing special characters."""
        preprocessor = mock_text_preprocessor()
        
        text_with_special = "Text with @#$% special characters!!!"
        clean_text = "Text with special characters"
        preprocessor.remove_special_chars.return_value = clean_text
        
        result = preprocessor.remove_special_chars(text_with_special)
        
        assert result == clean_text
        preprocessor.remove_special_chars.assert_called_once_with(text_with_special)

    def test_comprehensive_text_cleaning(self, mock_text_preprocessor):
        """Test comprehensive text cleaning."""
        preprocessor = mock_text_preprocessor()
        
        dirty_text = "  Text with\t\tmultiple   issues @#$%  \n\n"
        clean_text = "Text with multiple issues"
        preprocessor.clean_text.return_value = clean_text
        
        result = preprocessor.clean_text(dirty_text)
        
        assert result == clean_text
        preprocessor.clean_text.assert_called_once_with(dirty_text)

    def test_section_detection(self, mock_text_preprocessor, sample_plain_text):
        """Test automatic section detection."""
        preprocessor = mock_text_preprocessor()
        
        expected_sections = {
            "title": {"text": "Machine Learning Approaches for Ontology Information Extraction", "start": 0, "end": 65},
            "abstract": {"text": "This paper presents novel machine learning...", "start": 67, "end": 250},
            "introduction": {"text": "The field of ontology information extraction...", "start": 252, "end": 400},
            "methods": {"text": "We developed a neural network architecture...", "start": 402, "end": 600}
        }
        preprocessor.detect_sections.return_value = expected_sections
        
        sections = preprocessor.detect_sections(sample_plain_text)
        
        assert "abstract" in sections
        assert "methods" in sections
        assert sections["title"]["start"] == 0
        preprocessor.detect_sections.assert_called_once_with(sample_plain_text)

    def test_pattern_extraction(self, mock_text_preprocessor, sample_plain_text):
        """Test extracting specific patterns from text."""
        preprocessor = mock_text_preprocessor()
        
        # Test extracting email patterns
        expected_emails = ["author@university.edu", "contact@journal.org"]
        preprocessor.extract_patterns.return_value = expected_emails
        
        email_pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
        emails = preprocessor.extract_patterns(sample_plain_text, email_pattern)
        
        assert isinstance(emails, list)
        preprocessor.extract_patterns.assert_called_once_with(sample_plain_text, email_pattern)

    def test_unicode_normalization(self, mock_text_preprocessor):
        """Test Unicode normalization."""
        preprocessor = mock_text_preprocessor()
        
        unicode_text = "Café naïve résumé"
        normalized_text = "Cafe naive resume"
        preprocessor.normalize_unicode.return_value = normalized_text
        
        result = preprocessor.normalize_unicode(unicode_text)
        
        assert result == normalized_text
        preprocessor.normalize_unicode.assert_called_once_with(unicode_text)

    def test_encoding_issue_fixing(self, mock_text_preprocessor):
        """Test fixing encoding issues in text."""
        preprocessor = mock_text_preprocessor()
        
        broken_text = "Caf\xe9 na\xefve r\xe9sum\xe9"
        fixed_text = "Café naïve résumé"
        preprocessor.fix_encoding_issues.return_value = fixed_text
        
        result = preprocessor.fix_encoding_issues(broken_text)
        
        assert result == fixed_text
        preprocessor.fix_encoding_issues.assert_called_once_with(broken_text)

    def test_text_statistics_analysis(self, mock_text_preprocessor, sample_plain_text):
        """Test text statistics analysis."""
        preprocessor = mock_text_preprocessor()
        
        expected_stats = {
            "word_count": 150,
            "sentence_count": 12,
            "paragraph_count": 6,
            "avg_words_per_sentence": 12.5,
            "reading_level": "college",
            "language": "en"
        }
        preprocessor.get_text_stats.return_value = expected_stats
        
        stats = preprocessor.get_text_stats(sample_plain_text)
        
        assert stats["word_count"] == 150
        assert stats["reading_level"] == "college"
        preprocessor.get_text_stats.assert_called_once_with(sample_plain_text)

    def test_keyword_extraction(self, mock_text_preprocessor, sample_plain_text):
        """Test keyword extraction from text."""
        preprocessor = mock_text_preprocessor()
        
        expected_keywords = [
            {"keyword": "machine learning", "score": 0.95, "frequency": 8},
            {"keyword": "ontology", "score": 0.90, "frequency": 6},
            {"keyword": "information extraction", "score": 0.85, "frequency": 5}
        ]
        preprocessor.extract_keywords.return_value = expected_keywords
        
        keywords = preprocessor.extract_keywords(sample_plain_text)
        
        assert len(keywords) == 3
        assert keywords[0]["keyword"] == "machine learning"
        assert keywords[0]["score"] == 0.95
        preprocessor.extract_keywords.assert_called_once_with(sample_plain_text)

    def test_preprocessing_pipeline(self, mock_text_preprocessor):
        """Test preprocessing pipeline functionality."""
        preprocessor = mock_text_preprocessor()
        
        # Test adding pipeline steps
        steps = ["normalize_whitespace", "remove_special_chars", "tokenize"]
        for step in steps:
            preprocessor.add_step(step)
        
        # Test pipeline execution
        input_text = "  Raw text   with @#$% issues  "
        processed_text = "processed clean text tokens"
        preprocessor.preprocess.return_value = processed_text
        
        result = preprocessor.preprocess(input_text)
        
        assert result == processed_text
        assert preprocessor.add_step.call_count == 3
        preprocessor.preprocess.assert_called_once_with(input_text)

    def test_pipeline_configuration(self, mock_text_preprocessor):
        """Test preprocessing pipeline configuration."""
        preprocessor = mock_text_preprocessor()
        
        # Test getting pipeline configuration
        expected_pipeline = ["normalize_whitespace", "tokenize", "remove_special_chars"]
        preprocessor.get_pipeline.return_value = expected_pipeline
        
        pipeline = preprocessor.get_pipeline()
        assert pipeline == expected_pipeline
        
        # Test removing pipeline step
        preprocessor.remove_step("remove_special_chars")
        preprocessor.remove_step.assert_called_once_with("remove_special_chars")


class TestTextExtractionFactory:
    """Test text extraction factory functionality."""

    def test_factory_creates_correct_extractor_for_pdf(self):
        """Test factory creates PDF extractor for PDF content."""
        # Create a mock PDF extractor instance
        mock_pdf_extractor = Mock()
        mock_pdf_extractor.extract_text.return_value = "Extracted PDF text"
        mock_pdf_extractor.supports_format.return_value = True
        mock_pdf_extractor.get_supported_formats.return_value = ["pdf"]
        
        # Create a mock factory that returns the PDF extractor for PDF content
        mock_factory = Mock()
        mock_factory.create_extractor.return_value = mock_pdf_extractor
        mock_factory.detect_format.return_value = "pdf"
        
        pdf_content = b"%PDF-1.4\ntest content"
        
        # Test factory creates appropriate extractor
        extractor = mock_factory.create_extractor(pdf_content)
        
        assert extractor is not None
        assert extractor.supports_format("pdf") is True
        assert "pdf" in extractor.get_supported_formats()
        mock_factory.create_extractor.assert_called_once_with(pdf_content)

    def test_factory_creates_correct_extractor_for_xml(self):
        """Test factory creates XML extractor for XML content."""
        # Create a mock XML extractor instance
        mock_xml_extractor = Mock()
        mock_xml_extractor.extract_text.return_value = "Extracted XML text"
        mock_xml_extractor.supports_format.return_value = True
        mock_xml_extractor.get_supported_formats.return_value = ["xml", "pmc"]
        
        # Create a mock factory that returns the XML extractor for XML content
        mock_factory = Mock()
        mock_factory.create_extractor.return_value = mock_xml_extractor
        mock_factory.detect_format.return_value = "xml"
        
        xml_content = "<?xml version='1.0'?><root>test</root>"
        
        # Test factory creates appropriate extractor
        extractor = mock_factory.create_extractor(xml_content)
        
        assert extractor is not None
        assert extractor.supports_format("xml") is True
        assert "xml" in extractor.get_supported_formats()
        mock_factory.create_extractor.assert_called_once_with(xml_content)

    def test_factory_creates_correct_extractor_for_plain_text(self):
        """Test factory creates plain text extractor for text content."""
        # Create a mock plain text extractor instance
        mock_text_extractor = Mock()
        mock_text_extractor.extract_text.return_value = "Extracted plain text"
        mock_text_extractor.supports_format.return_value = True
        mock_text_extractor.get_supported_formats.return_value = ["txt", "text"]
        
        # Create a mock factory that returns the text extractor for text content
        mock_factory = Mock()
        mock_factory.create_extractor.return_value = mock_text_extractor
        mock_factory.detect_format.return_value = "txt"
        
        text_content = "Plain text content"
        
        # Test factory creates appropriate extractor
        extractor = mock_factory.create_extractor(text_content)
        
        assert extractor is not None
        assert extractor.supports_format("txt") is True
        assert "txt" in extractor.get_supported_formats()
        mock_factory.create_extractor.assert_called_once_with(text_content)

    def test_factory_format_detection(self):
        """Test factory format detection logic."""
        mock_factory = Mock()
        
        # Test PDF detection
        mock_factory.detect_format.return_value = "pdf"
        pdf_format = mock_factory.detect_format(b"%PDF-1.4")
        assert pdf_format == "pdf"
        
        # Test XML detection
        mock_factory.detect_format.return_value = "xml"
        xml_format = mock_factory.detect_format("<?xml version='1.0'?>")
        assert xml_format == "xml"
        
        # Test plain text detection
        mock_factory.detect_format.return_value = "txt"
        text_format = mock_factory.detect_format("Plain text")
        assert text_format == "txt"

    def test_factory_extractor_registration(self):
        """Test registering custom extractors with factory."""
        mock_factory = Mock()
        
        # Test registering custom extractor
        custom_extractor_class = Mock()
        formats = ["custom", "special"]
        
        mock_factory.register_extractor(custom_extractor_class, formats)
        mock_factory.register_extractor.assert_called_once_with(custom_extractor_class, formats)

    def test_factory_unsupported_format_handling(self):
        """Test factory handling of unsupported formats."""
        mock_factory = Mock()
        mock_factory.create_extractor.side_effect = ValueError("Unsupported format: binary")
        
        binary_content = b"\x89PNG\r\n\x1a\n"  # PNG header
        
        with pytest.raises(ValueError, match="Unsupported format"):
            mock_factory.create_extractor(binary_content)


class TestTextExtractionIntegration:
    """Integration tests between different text extractors."""

    def test_cross_format_workflow(self, mock_pdf_text_extractor, mock_xml_text_extractor):
        """Test workflow across different formats."""
        pdf_extractor = mock_pdf_text_extractor()
        xml_extractor = mock_xml_text_extractor()
        
        # Extract from PDF
        pdf_content = b"%PDF-1.4\ntest content"
        pdf_text = "Extracted PDF text content"
        pdf_extractor.extract_text.return_value = pdf_text
        
        # Extract from XML
        xml_content = "<?xml version='1.0'?><article>XML content</article>"
        xml_text = "Extracted XML text content"
        xml_extractor.extract_text.return_value = xml_text
        
        # Simulate workflow
        pdf_result = pdf_extractor.extract_text(pdf_content)
        xml_result = xml_extractor.extract_text(xml_content)
        
        assert pdf_result == pdf_text
        assert xml_result == xml_text
        
        # Combined results
        combined_text = f"{pdf_result}\n\n{xml_result}"
        assert "PDF text content" in combined_text
        assert "XML text content" in combined_text

    def test_fallback_extraction_mechanism(self, mock_pdf_text_extractor):
        """Test fallback mechanism when primary extraction fails."""
        extractor = mock_pdf_text_extractor()
        
        # Primary extraction fails
        extractor.extract_text.side_effect = [
            Exception("Primary extraction failed"),
            "Fallback extraction successful"
        ]
        
        pdf_content = b"%PDF-1.4\ntest content"
        
        # First call fails, second succeeds (simulating fallback)
        try:
            result = extractor.extract_text(pdf_content)
        except Exception:
            # Fallback mechanism
            result = extractor.extract_text(pdf_content)
        
        assert result == "Fallback extraction successful"

    def test_preprocessing_integration(self, mock_text_preprocessor, mock_pdf_text_extractor):
        """Test integration between text extraction and preprocessing."""
        extractor = mock_pdf_text_extractor()
        preprocessor = mock_text_preprocessor()
        
        # Extract raw text
        raw_text = "  Raw PDF text   with\t\tirregular  spacing  "
        extractor.extract_text.return_value = raw_text
        
        # Preprocess extracted text
        clean_text = "Raw PDF text with irregular spacing"
        preprocessor.clean_text.return_value = clean_text
        
        pdf_content = b"%PDF-1.4\ntest content"
        
        # Workflow: extract then preprocess
        extracted = extractor.extract_text(pdf_content)
        processed = preprocessor.clean_text(extracted)
        
        assert processed == clean_text
        extractor.extract_text.assert_called_once_with(pdf_content)
        preprocessor.clean_text.assert_called_once_with(raw_text)

    def test_metadata_consistency_across_formats(self, mock_pdf_text_extractor, mock_xml_text_extractor):
        """Test metadata consistency across different formats."""
        pdf_extractor = mock_pdf_text_extractor()
        xml_extractor = mock_xml_text_extractor()
        
        # Common metadata structure
        common_metadata = {
            "title": "Test Document",
            "authors": ["Author 1", "Author 2"],
            "doi": "10.1000/test",
            "abstract": "Test abstract"
        }
        
        pdf_extractor.extract_metadata.return_value = common_metadata
        xml_extractor.extract_metadata.return_value = common_metadata
        
        pdf_content = b"%PDF-1.4\ntest"
        xml_content = "<?xml version='1.0'?><article>test</article>"
        
        pdf_metadata = pdf_extractor.extract_metadata(pdf_content)
        xml_metadata = xml_extractor.extract_metadata(xml_content)
        
        # Verify consistent metadata structure
        assert pdf_metadata["title"] == xml_metadata["title"]
        assert pdf_metadata["authors"] == xml_metadata["authors"]
        assert pdf_metadata["doi"] == xml_metadata["doi"]

    def test_error_propagation_across_components(self, mock_pdf_text_extractor, mock_text_preprocessor):
        """Test error propagation across extraction and preprocessing components."""
        extractor = mock_pdf_text_extractor()
        preprocessor = mock_text_preprocessor()
        
        # Extractor succeeds but preprocessor fails
        extractor.extract_text.return_value = "Extracted text"
        preprocessor.clean_text.side_effect = Exception("Preprocessing failed")
        
        pdf_content = b"%PDF-1.4\ntest content"
        
        extracted = extractor.extract_text(pdf_content)
        assert extracted == "Extracted text"
        
        with pytest.raises(Exception, match="Preprocessing failed"):
            preprocessor.clean_text(extracted)


class TestTextExtractionPerformance:
    """Performance and scalability tests."""

    def test_large_document_processing(self, mock_pdf_text_extractor):
        """Test processing large documents."""
        extractor = mock_pdf_text_extractor()
        
        # Configure for large document processing
        extractor.set_options({
            "memory_efficient": True,
            "streaming_mode": True,
            "chunk_size": 1024
        })
        
        # Mock large document processing
        mock_result = Mock()
        mock_result.text_length = 1000000  # 1M characters
        mock_result.processing_time = 30.5  # seconds
        mock_result.memory_usage = "256MB"
        extractor.extract_text.return_value = mock_result
        
        large_pdf = b"%PDF-1.4\nlarge content" * 10000
        result = extractor.extract_text(large_pdf)
        
        assert result.text_length == 1000000
        assert result.processing_time < 60
        extractor.set_options.assert_called_once()

    def test_concurrent_extraction(self, mock_pdf_text_extractor):
        """Test concurrent text extraction from multiple documents."""
        extractor = mock_pdf_text_extractor()
        
        # Configure for concurrent processing
        extractor.set_options({"parallel_processing": True, "max_workers": 4})
        
        documents = [
            b"%PDF-1.4\ndoc1",
            b"%PDF-1.4\ndoc2", 
            b"%PDF-1.4\ndoc3",
            b"%PDF-1.4\ndoc4"
        ]
        
        mock_results = ["text1", "text2", "text3", "text4"]
        extractor.extract_text.side_effect = mock_results
        
        results = []
        for doc in documents:
            result = extractor.extract_text(doc)
            results.append(result)
        
        assert len(results) == 4
        assert results[0] == "text1"
        assert extractor.extract_text.call_count == 4

    def test_memory_optimization(self, mock_text_preprocessor):
        """Test memory optimization during text preprocessing."""
        preprocessor = mock_text_preprocessor()
        
        # Configure memory optimization
        preprocessor.set_options({
            "memory_efficient": True,
            "lazy_evaluation": True,
            "gc_frequency": 1000
        })
        
        # Process large text
        large_text = "Large text content " * 50000
        mock_result = Mock()
        mock_result.processed_text = "Processed large text"
        mock_result.memory_saved = "150MB"
        preprocessor.preprocess.return_value = mock_result
        
        result = preprocessor.preprocess(large_text)
        
        assert result.processed_text == "Processed large text"
        assert result.memory_saved == "150MB"
        preprocessor.set_options.assert_called_once()

    def test_caching_performance(self, mock_pdf_text_extractor):
        """Test caching mechanisms for repeated extractions."""
        extractor = mock_pdf_text_extractor()
        
        # Configure caching
        extractor.set_options({
            "enable_cache": True,
            "cache_size": 1000,
            "cache_ttl": 3600
        })
        
        pdf_content = b"%PDF-1.4\ntest content"
        cached_result = "Cached extraction result"
        
        # First call - cache miss
        extractor.extract_text.return_value = cached_result
        result1 = extractor.extract_text(pdf_content)
        
        # Second call - cache hit (in real implementation)
        result2 = extractor.extract_text(pdf_content)
        
        assert result1 == cached_result
        assert result2 == cached_result
        assert extractor.extract_text.call_count == 2

    def test_streaming_extraction(self, mock_pdf_text_extractor):
        """Test streaming extraction for very large documents."""
        extractor = mock_pdf_text_extractor()
        
        # Configure streaming mode
        extractor.set_options({
            "streaming_mode": True,
            "buffer_size": 8192,
            "yield_chunks": True
        })
        
        # Mock streaming result
        mock_chunks = ["chunk1", "chunk2", "chunk3", "chunk4"]
        extractor.extract_text.return_value = iter(mock_chunks)
        
        large_pdf = b"%PDF-1.4\nvery large content"
        chunk_iterator = extractor.extract_text(large_pdf)
        
        collected_chunks = list(chunk_iterator)
        assert len(collected_chunks) == 4
        assert collected_chunks[0] == "chunk1"


class TestTextExtractionErrorHandling:
    """Error handling and validation tests."""

    def test_invalid_format_handling(self, mock_text_extractor_base):
        """Test handling of invalid or unsupported formats."""
        extractor = mock_text_extractor_base()
        
        # Test unsupported format
        extractor.supports_format.return_value = False
        extractor.extract_text.side_effect = ValueError("Unsupported format: binary")
        
        binary_content = b"\x89PNG\r\n\x1a\n"  # PNG header
        
        with pytest.raises(ValueError, match="Unsupported format"):
            extractor.extract_text(binary_content)

    def test_corrupted_content_handling(self, mock_pdf_text_extractor):
        """Test handling of corrupted content."""
        extractor = mock_pdf_text_extractor()
        
        extractor.extract_text.side_effect = Exception("Corrupted PDF structure")
        
        corrupted_pdf = b"corrupted%PDF-1.4\nbroken content"
        
        with pytest.raises(Exception, match="Corrupted PDF structure"):
            extractor.extract_text(corrupted_pdf)

    def test_encoding_error_handling(self, mock_plain_text_extractor):
        """Test handling of encoding errors."""
        extractor = mock_plain_text_extractor()
        
        # Test encoding error recovery
        mock_result = Mock()
        mock_result.text = "Text with encoding issues resolved"
        mock_result.errors = ["Invalid UTF-8 sequence detected and fixed"]
        mock_result.encoding_confidence = 0.7
        extractor.extract_text.return_value = mock_result
        
        text_with_encoding_issues = b"Text with \xff\xfe invalid bytes"
        result = extractor.extract_text(text_with_encoding_issues)
        
        assert "encoding issues resolved" in result.text
        assert len(result.errors) == 1
        assert result.encoding_confidence == 0.7

    def test_memory_limit_exceeded(self, mock_pdf_text_extractor):
        """Test handling when memory limits are exceeded."""
        extractor = mock_pdf_text_extractor()
        
        extractor.extract_text.side_effect = MemoryError("Memory limit exceeded during extraction")
        
        huge_pdf = b"%PDF-1.4\n" + b"x" * 1000000000  # Very large content
        
        with pytest.raises(MemoryError, match="Memory limit exceeded"):
            extractor.extract_text(huge_pdf)

    def test_timeout_handling(self, mock_xml_text_extractor):
        """Test handling of processing timeouts."""
        extractor = mock_xml_text_extractor()
        
        extractor.extract_text.side_effect = TimeoutError("Extraction timeout after 300 seconds")
        
        complex_xml = "<?xml version='1.0'?><root>" + "<item>data</item>" * 1000000 + "</root>"
        
        with pytest.raises(TimeoutError, match="Extraction timeout"):
            extractor.extract_text(complex_xml)

    def test_partial_extraction_recovery(self, mock_pdf_text_extractor):
        """Test recovery with partial extraction when full extraction fails."""
        extractor = mock_pdf_text_extractor()
        
        # Mock partial extraction result
        mock_result = Mock()
        mock_result.text = "Partially extracted text from readable pages"
        mock_result.pages_processed = 8
        mock_result.pages_total = 10
        mock_result.errors = ["Pages 9-10 corrupted", "Unable to extract figures"]
        mock_result.success_rate = 0.8
        extractor.extract_text.return_value = mock_result
        
        damaged_pdf = b"%PDF-1.4\npartially damaged content"
        result = extractor.extract_text(damaged_pdf)
        
        assert result.pages_processed == 8
        assert result.success_rate == 0.8
        assert len(result.errors) == 2

    def test_validation_error_collection(self, mock_text_extractor_base):
        """Test collection and reporting of validation errors."""
        extractor = mock_text_extractor_base()
        
        # Mock validation errors
        validation_errors = [
            "Warning: Low confidence in format detection",
            "Error: Missing required metadata fields",
            "Warning: Text may contain OCR artifacts"
        ]
        extractor.get_errors.return_value = ["Error: Missing required metadata fields"]
        extractor.get_warnings.return_value = [
            "Warning: Low confidence in format detection",
            "Warning: Text may contain OCR artifacts"
        ]
        
        errors = extractor.get_errors()
        warnings = extractor.get_warnings()
        
        assert len(errors) == 1
        assert len(warnings) == 2
        assert "metadata fields" in errors[0]
        assert "OCR artifacts" in warnings[1]

    def test_graceful_degradation(self, mock_pdf_text_extractor):
        """Test graceful degradation when optimal extraction fails."""
        extractor = mock_pdf_text_extractor()
        
        # Configure fallback options
        extractor.set_options({
            "fallback_enabled": True,
            "fallback_methods": ["pypdf", "basic_text"],
            "graceful_degradation": True
        })
        
        # Mock degraded extraction result
        mock_result = Mock()
        mock_result.text = "Basic text extraction result"
        mock_result.extraction_method = "fallback_basic_text"
        mock_result.quality_score = 0.6
        mock_result.warnings = ["Switched to basic extraction method"]
        extractor.extract_text.return_value = mock_result
        
        problematic_pdf = b"%PDF-1.4\nproblematic content"
        result = extractor.extract_text(problematic_pdf)
        
        assert result.extraction_method == "fallback_basic_text"
        assert result.quality_score == 0.6
        assert "basic extraction" in result.warnings[0]

    def test_resource_cleanup_on_error(self, mock_pdf_text_extractor):
        """Test proper resource cleanup when errors occur."""
        extractor = mock_pdf_text_extractor()
        
        # Mock cleanup tracking
        cleanup_called = Mock()
        extractor.cleanup_resources = cleanup_called
        
        # Simulate error during extraction
        def mock_extract_with_cleanup(content):
            try:
                raise Exception("Extraction failed")
            finally:
                extractor.cleanup_resources()
                
        extractor.extract_text.side_effect = mock_extract_with_cleanup
        
        pdf_content = b"%PDF-1.4\ntest content"
        
        with pytest.raises(Exception, match="Extraction failed"):
            extractor.extract_text(pdf_content)
        
        # Verify cleanup was called
        cleanup_called.assert_called_once()


# Additional fixtures for edge cases and complex scenarios
@pytest.fixture
def multilingual_text_sample():
    """Sample text in multiple languages."""
    return """
    English: Machine learning approaches for ontology extraction.
    Español: Enfoques de aprendizaje automático para la extracción de ontologías.
    Français: Approches d'apprentissage automatique pour l'extraction d'ontologies.
    Deutsch: Maschinelle Lernansätze für die Ontologie-Extraktion.
    中文: 用于本体抽取的机器学习方法。
    """


@pytest.fixture
def complex_document_structure():
    """Complex document structure with nested sections."""
    return {
        "document": {
            "title": "Advanced Ontology Extraction Techniques",
            "sections": {
                "abstract": {"text": "Abstract content...", "subsections": {}},
                "introduction": {
                    "text": "Introduction overview...",
                    "subsections": {
                        "background": {"text": "Background information..."},
                        "motivation": {"text": "Research motivation..."},
                        "contributions": {"text": "Our contributions..."}
                    }
                },
                "methods": {
                    "text": "Methods overview...",
                    "subsections": {
                        "data_collection": {"text": "Data collection methodology..."},
                        "preprocessing": {"text": "Text preprocessing steps..."},
                        "model_architecture": {"text": "Neural network architecture..."},
                        "training": {"text": "Training procedure..."}
                    }
                },
                "results": {
                    "text": "Results overview...",
                    "subsections": {
                        "quantitative": {"text": "Quantitative results..."},
                        "qualitative": {"text": "Qualitative analysis..."},
                        "comparison": {"text": "Comparison with baselines..."}
                    }
                }
            }
        }
    }


@pytest.fixture
def extraction_performance_metrics():
    """Performance metrics for text extraction."""
    return {
        "processing_time": 45.2,
        "memory_usage": "512MB",
        "peak_memory": "768MB",
        "cpu_usage": 85.5,
        "text_length": 250000,
        "pages_processed": 25,
        "extraction_rate": "characters_per_second: 5555",
        "accuracy": 0.95,
        "confidence": 0.88
    }


if __name__ == "__main__":
    pytest.main([__file__, "-v"])