"""
Comprehensive Unit Tests for PDF Parser (TDD Approach)

This module provides comprehensive unit tests for the PDF parser functionality in the AIM2 ontology
information extraction system. The tests follow test-driven development (TDD) approach,
defining the expected behavior of the PDF parser classes before implementation.

Test Classes:
    TestPDFParserCreation: Tests for PDFParser instantiation and configuration
    TestPDFParserFormatSupport: Tests for PDF format detection and validation
    TestPDFParserParsing: Tests for core PDF parsing functionality
    TestPDFParserTextExtraction: Tests for PDF text extraction
    TestPDFParserMetadataExtraction: Tests for PDF metadata extraction
    TestPDFParserSectionIdentification: Tests for section identification in scientific papers
    TestPDFParserFigureTableExtraction: Tests for figure and table caption extraction
    TestPDFParserReferenceExtraction: Tests for reference parsing
    TestPDFParserConversion: Tests for converting parsed PDF to internal models
    TestPDFParserErrorHandling: Tests for error handling and validation
    TestPDFParserValidation: Tests for PDF validation functionality
    TestPDFParserOptions: Tests for parser configuration and options
    TestPDFParserIntegration: Integration tests with other components
    TestPDFParserPerformance: Performance and scalability tests

The PDFParser is expected to be a concrete implementation providing:
- PDF format support: Standard PDF, encrypted PDF, scanned PDF
- Text extraction from scientific papers
- Metadata extraction (title, authors, DOI, etc.)
- Section identification (abstract, introduction, methods, results, etc.)
- Figure and table caption extraction
- Reference parsing and extraction
- Integration with pypdf, pdfplumber, and PyMuPDF libraries
- Conversion to internal Term/Relationship/Ontology models
- Comprehensive validation and error reporting
- Performance optimization for large PDF files
- Configurable parsing options and encoding detection

Dependencies:
    - pytest: For test framework
    - unittest.mock: For mocking functionality
    - pypdf: For PDF processing (mocked)
    - pdfplumber: For PDF text extraction (mocked)
    - PyMuPDF (fitz): For advanced PDF processing (mocked)
    - typing: For type hints

Usage:
    pytest tests/unit/test_pdf_parser.py -v
"""

import tempfile
from datetime import datetime
from pathlib import Path
from unittest.mock import Mock, patch

import pytest


# Mock the parser classes since they don't exist yet (TDD approach)
@pytest.fixture
def mock_pdf_parser():
    """Mock PDFParser class."""
    # Create a mock class that doesn't rely on importing non-existent modules
    mock_parser_class = Mock()
    mock_instance = Mock()

    # Core parsing methods
    mock_instance.parse = Mock()
    mock_instance.parse_file = Mock()
    mock_instance.parse_bytes = Mock()
    mock_instance.parse_stream = Mock()

    # Format detection and validation
    mock_instance.detect_format = Mock()
    mock_instance.validate_format = Mock()
    mock_instance.get_supported_formats = Mock(return_value=["pdf"])
    mock_instance.is_encrypted = Mock()
    mock_instance.is_scanned = Mock()

    # Text extraction methods
    mock_instance.extract_text = Mock()
    mock_instance.extract_text_by_page = Mock()
    mock_instance.extract_text_with_layout = Mock()
    mock_instance.extract_text_ocr = Mock()

    # Metadata extraction methods
    mock_instance.extract_metadata = Mock()
    mock_instance.extract_title = Mock()
    mock_instance.extract_authors = Mock()
    mock_instance.extract_doi = Mock()
    mock_instance.extract_abstract = Mock()
    mock_instance.extract_keywords = Mock()

    # Section identification methods
    mock_instance.identify_sections = Mock()
    mock_instance.extract_introduction = Mock()
    mock_instance.extract_methods = Mock()
    mock_instance.extract_results = Mock()
    mock_instance.extract_discussion = Mock()
    mock_instance.extract_conclusion = Mock()

    # Figure and table extraction
    mock_instance.extract_figures = Mock()
    mock_instance.extract_tables = Mock()
    mock_instance.extract_captions = Mock()

    # Reference extraction
    mock_instance.extract_references = Mock()
    mock_instance.parse_citations = Mock()

    # Conversion methods
    mock_instance.to_ontology = Mock()
    mock_instance.extract_terms = Mock()
    mock_instance.extract_relationships = Mock()

    # Configuration
    mock_instance.set_options = Mock()
    mock_instance.get_options = Mock(return_value={})
    mock_instance.reset_options = Mock()

    # Validation
    mock_instance.validate = Mock()
    mock_instance.validate_pdf = Mock()
    mock_instance.get_validation_errors = Mock(return_value=[])

    # Configure the mock class to return the mock instance when called
    mock_parser_class.return_value = mock_instance
    return mock_parser_class


@pytest.fixture
def mock_pypdf():
    """Mock pypdf library."""
    with patch("pypdf.PdfReader") as mock_reader:
        mock_reader_instance = Mock()
        mock_reader_instance.pages = [Mock(), Mock(), Mock()]
        mock_reader_instance.metadata = {
            "/Title": "Test Document",
            "/Author": "Test Author",
            "/Subject": "Test Subject",
            "/Creator": "Test Creator",
            "/CreationDate": "D:20230101000000+00'00'",
        }
        mock_reader_instance.is_encrypted = False
        mock_reader_instance.decrypt = Mock(return_value=True)

        # Mock page extraction
        for i, page in enumerate(mock_reader_instance.pages):
            page.extract_text = Mock(return_value=f"Page {i+1} content")
            page.mediabox = Mock()
            page.mediabox.width = 612.0
            page.mediabox.height = 792.0

        mock_reader.return_value = mock_reader_instance
        yield mock_reader


@pytest.fixture
def mock_pdfplumber():
    """Mock pdfplumber library."""
    with patch("pdfplumber.open") as mock_open:
        mock_pdf = Mock()
        mock_pdf.__enter__ = Mock(return_value=mock_pdf)
        mock_pdf.__exit__ = Mock(return_value=None)

        # Mock pages
        mock_page1 = Mock()
        mock_page1.extract_text = Mock(return_value="Page 1 text content")
        mock_page1.extract_tables = Mock(return_value=[])
        mock_page1.width = 612.0
        mock_page1.height = 792.0

        mock_page2 = Mock()
        mock_page2.extract_text = Mock(return_value="Page 2 text content")
        mock_page2.extract_tables = Mock(return_value=[])

        mock_pdf.pages = [mock_page1, mock_page2]
        mock_pdf.metadata = {"Title": "Test Document", "Author": "Test Author"}

        mock_open.return_value = mock_pdf
        yield mock_open


@pytest.fixture
def mock_fitz():
    """Mock PyMuPDF (fitz) library."""
    with patch("fitz.open") as mock_open:
        mock_doc = Mock()
        mock_doc.page_count = 3
        mock_doc.metadata = {
            "title": "Test Document",
            "author": "Test Author",
            "subject": "Test Subject",
            "creator": "Test Creator",
        }
        mock_doc.is_encrypted = False
        mock_doc.authenticate = Mock(return_value=True)

        # Mock pages
        mock_pages = []
        for i in range(3):
            mock_page = Mock()
            mock_page.get_text = Mock(return_value=f"Page {i+1} content from fitz")
            mock_page.get_textpage = Mock()
            mock_page.get_images = Mock(return_value=[])
            mock_page.get_drawings = Mock(return_value=[])
            mock_page.rect = Mock()
            mock_page.rect.width = 612.0
            mock_page.rect.height = 792.0
            mock_pages.append(mock_page)

        mock_doc.__getitem__ = lambda self, index: mock_pages[index]
        mock_doc.__iter__ = lambda self: iter(mock_pages)
        mock_doc.__len__ = lambda self: len(mock_pages)

        mock_open.return_value = mock_doc
        yield mock_open


@pytest.fixture
def sample_pdf_bytes():
    """Sample PDF bytes for testing."""
    # Simple PDF header for testing
    return b"%PDF-1.4\n1 0 obj\n<<\n/Type /Catalog\n/Pages 2 0 R\n>>\nendobj\n%%EOF"


@pytest.fixture
def sample_scientific_paper_content():
    """Sample scientific paper content for testing."""
    return {
        "title": "Machine Learning Approaches for Ontology Information Extraction",
        "authors": ["Dr. Jane Smith", "Prof. John Doe", "Dr. Alice Johnson"],
        "abstract": "This paper presents novel machine learning approaches for extracting ontological information from scientific literature...",
        "introduction": "The field of ontology information extraction has grown significantly...",
        "methods": "We employed a combination of natural language processing techniques...",
        "results": "Our experiments show that the proposed method achieves 95% accuracy...",
        "discussion": "The results demonstrate the effectiveness of our approach...",
        "conclusion": "In conclusion, we have presented a robust method for ontology extraction...",
        "references": [
            "Smith, J. et al. (2022). Ontology extraction methods. Nature, 123(4), 567-578.",
            "Doe, J. (2021). Machine learning in bioinformatics. Science, 456(7), 890-901.",
        ],
        "figures": [
            {"caption": "Figure 1: Architecture of the proposed system", "page": 3},
            {"caption": "Figure 2: Performance comparison", "page": 5},
        ],
        "tables": [
            {"caption": "Table 1: Dataset statistics", "page": 4},
            {"caption": "Table 2: Experimental results", "page": 6},
        ],
    }


@pytest.fixture
def sample_pdf_metadata():
    """Sample PDF metadata for testing."""
    return {
        "title": "Test Document Title",
        "author": "Test Author",
        "subject": "Test Subject",
        "creator": "Test Creator Application",
        "producer": "Test PDF Producer",
        "creation_date": datetime(2023, 1, 1, 12, 0, 0),
        "modification_date": datetime(2023, 1, 15, 14, 30, 0),
        "pages": 10,
        "file_size": 1024000,  # 1MB
        "pdf_version": "1.4",
        "encrypted": False,
        "has_forms": False,
        "has_javascript": False,
    }


class TestPDFParserCreation:
    """Test PDFParser instantiation and configuration."""

    def test_pdf_parser_creation_default(self, mock_pdf_parser):
        """Test creating PDFParser with default settings."""
        parser = mock_pdf_parser()

        # Verify parser was created
        assert parser is not None
        mock_pdf_parser.assert_called_once()

    def test_pdf_parser_creation_with_options(self, mock_pdf_parser):
        """Test creating PDFParser with custom options."""
        options = {
            "extraction_method": "pdfplumber",
            "ocr_enabled": True,
            "password": None,
            "page_range": None,
            "layout_mode": False,
        }

        parser = mock_pdf_parser(options=options)
        mock_pdf_parser.assert_called_once_with(options=options)

    def test_pdf_parser_creation_with_invalid_options(self, mock_pdf_parser):
        """Test creating PDFParser with invalid options raises error."""
        mock_pdf_parser.side_effect = ValueError("Invalid option: unknown_method")

        invalid_options = {"unknown_method": "invalid"}

        with pytest.raises(ValueError, match="Invalid option"):
            mock_pdf_parser(options=invalid_options)

    def test_pdf_parser_inherits_from_abstract_parser(self, mock_pdf_parser):
        """Test that PDFParser implements AbstractParser interface."""
        parser = mock_pdf_parser()

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


class TestPDFParserFormatSupport:
    """Test PDF format detection and validation."""

    def test_get_supported_formats(self, mock_pdf_parser):
        """Test getting list of supported PDF formats."""
        parser = mock_pdf_parser()

        formats = parser.get_supported_formats()

        expected_formats = ["pdf"]
        assert all(fmt in formats for fmt in expected_formats)

    def test_detect_format_pdf(self, mock_pdf_parser, sample_pdf_bytes):
        """Test detecting PDF format."""
        parser = mock_pdf_parser()
        parser.detect_format.return_value = "pdf"

        detected_format = parser.detect_format(sample_pdf_bytes)

        assert detected_format == "pdf"
        parser.detect_format.assert_called_once_with(sample_pdf_bytes)

    def test_validate_format_valid_pdf(self, mock_pdf_parser, sample_pdf_bytes):
        """Test validating valid PDF content."""
        parser = mock_pdf_parser()
        parser.validate_format.return_value = True

        is_valid = parser.validate_format(sample_pdf_bytes, "pdf")

        assert is_valid is True
        parser.validate_format.assert_called_once_with(sample_pdf_bytes, "pdf")

    def test_validate_format_invalid_pdf(self, mock_pdf_parser):
        """Test validating invalid PDF content."""
        parser = mock_pdf_parser()
        parser.validate_format.return_value = False

        invalid_content = b"This is not PDF content"
        is_valid = parser.validate_format(invalid_content, "pdf")

        assert is_valid is False

    def test_is_encrypted_false(self, mock_pdf_parser, sample_pdf_bytes):
        """Test detecting non-encrypted PDF."""
        parser = mock_pdf_parser()
        parser.is_encrypted.return_value = False

        encrypted = parser.is_encrypted(sample_pdf_bytes)

        assert encrypted is False
        parser.is_encrypted.assert_called_once_with(sample_pdf_bytes)

    def test_is_encrypted_true(self, mock_pdf_parser):
        """Test detecting encrypted PDF."""
        parser = mock_pdf_parser()
        parser.is_encrypted.return_value = True

        encrypted_pdf = b"%PDF-1.4\n(encrypted content)"
        encrypted = parser.is_encrypted(encrypted_pdf)

        assert encrypted is True

    def test_is_scanned_false(self, mock_pdf_parser, sample_pdf_bytes):
        """Test detecting text-based PDF."""
        parser = mock_pdf_parser()
        parser.is_scanned.return_value = False

        scanned = parser.is_scanned(sample_pdf_bytes)

        assert scanned is False
        parser.is_scanned.assert_called_once_with(sample_pdf_bytes)

    def test_is_scanned_true(self, mock_pdf_parser):
        """Test detecting scanned PDF."""
        parser = mock_pdf_parser()
        parser.is_scanned.return_value = True

        scanned_pdf = b"%PDF-1.4\n(image-based content)"
        scanned = parser.is_scanned(scanned_pdf)

        assert scanned is True


class TestPDFParserParsing:
    """Test core PDF parsing functionality."""

    def test_parse_file_pdf_path(self, mock_pdf_parser):
        """Test parsing PDF file from file path."""
        parser = mock_pdf_parser()
        mock_result = Mock()
        mock_result.pages = 10
        mock_result.text_length = 5000
        parser.parse_file.return_value = mock_result

        file_path = "/path/to/document.pdf"
        result = parser.parse_file(file_path)

        assert result == mock_result
        assert result.pages == 10
        parser.parse_file.assert_called_once_with(file_path)

    def test_parse_bytes_pdf_content(self, mock_pdf_parser, sample_pdf_bytes):
        """Test parsing PDF from bytes."""
        parser = mock_pdf_parser()
        mock_result = Mock()
        mock_result.source_type = "bytes"
        parser.parse_bytes.return_value = mock_result

        result = parser.parse_bytes(sample_pdf_bytes)

        assert result == mock_result
        assert result.source_type == "bytes"
        parser.parse_bytes.assert_called_once_with(sample_pdf_bytes)

    def test_parse_stream_pdf_data(self, mock_pdf_parser):
        """Test parsing PDF from stream/file-like object."""
        parser = mock_pdf_parser()
        mock_result = Mock()
        parser.parse_stream.return_value = mock_result

        with tempfile.NamedTemporaryFile(suffix=".pdf") as tmp_file:
            tmp_file.write(b"%PDF-1.4\ntest content")
            tmp_file.seek(0)

            result = parser.parse_stream(tmp_file)

        assert result == mock_result
        parser.parse_stream.assert_called_once()

    def test_parse_with_password(self, mock_pdf_parser):
        """Test parsing encrypted PDF with password."""
        parser = mock_pdf_parser()
        mock_result = Mock()
        mock_result.encrypted = True
        mock_result.decrypted = True
        parser.parse_file.return_value = mock_result

        parser.set_options({"password": "secret123"})
        result = parser.parse_file("/path/to/encrypted.pdf")

        assert result == mock_result
        assert result.decrypted is True
        parser.set_options.assert_called_with({"password": "secret123"})

    def test_parse_with_page_range(self, mock_pdf_parser):
        """Test parsing specific page range."""
        parser = mock_pdf_parser()
        mock_result = Mock()
        mock_result.pages_processed = [1, 2, 3]
        parser.parse_file.return_value = mock_result

        parser.set_options({"page_range": (1, 3)})
        result = parser.parse_file("/path/to/document.pdf")

        assert result == mock_result
        assert len(result.pages_processed) == 3
        parser.set_options.assert_called_with({"page_range": (1, 3)})

    def test_parse_with_extraction_method(self, mock_pdf_parser):
        """Test parsing with specific extraction method."""
        parser = mock_pdf_parser()
        mock_result = Mock()
        mock_result.extraction_method = "pdfplumber"
        parser.parse_file.return_value = mock_result

        parser.set_options({"extraction_method": "pdfplumber"})
        result = parser.parse_file("/path/to/document.pdf")

        assert result == mock_result
        assert result.extraction_method == "pdfplumber"
        parser.set_options.assert_called_with({"extraction_method": "pdfplumber"})

    def test_parse_large_pdf_file(self, mock_pdf_parser):
        """Test parsing large PDF file with memory optimization."""
        parser = mock_pdf_parser()
        mock_result = Mock()
        mock_result.pages = 1000
        mock_result.memory_efficient = True
        parser.parse_file.return_value = mock_result

        # Configure for large file parsing
        parser.set_options({"memory_efficient": True, "lazy_loading": True})
        result = parser.parse_file("/path/to/large_document.pdf")

        assert result == mock_result
        assert result.memory_efficient is True
        parser.set_options.assert_called_with(
            {"memory_efficient": True, "lazy_loading": True}
        )


class TestPDFParserTextExtraction:
    """Test PDF text extraction functionality."""

    def test_extract_text_basic(self, mock_pdf_parser):
        """Test basic text extraction from PDF."""
        parser = mock_pdf_parser()
        expected_text = "This is the extracted text from the PDF document."
        parser.extract_text.return_value = expected_text

        parsed_result = Mock()
        text = parser.extract_text(parsed_result)

        assert text == expected_text
        parser.extract_text.assert_called_once_with(parsed_result)

    def test_extract_text_by_page(self, mock_pdf_parser):
        """Test extracting text page by page."""
        parser = mock_pdf_parser()
        expected_pages = {
            1: "Page 1 content",
            2: "Page 2 content",
            3: "Page 3 content",
        }
        parser.extract_text_by_page.return_value = expected_pages

        parsed_result = Mock()
        pages_text = parser.extract_text_by_page(parsed_result)

        assert pages_text == expected_pages
        assert len(pages_text) == 3
        parser.extract_text_by_page.assert_called_once_with(parsed_result)

    def test_extract_text_with_layout(self, mock_pdf_parser):
        """Test extracting text with layout preservation."""
        parser = mock_pdf_parser()
        mock_result = Mock()
        mock_result.text = "Layout preserved text"
        mock_result.layout_info = {"columns": 2, "tables": 1}
        parser.extract_text_with_layout.return_value = mock_result

        parsed_result = Mock()
        result = parser.extract_text_with_layout(parsed_result)

        assert result == mock_result
        assert "columns" in result.layout_info
        parser.extract_text_with_layout.assert_called_once_with(parsed_result)

    def test_extract_text_ocr(self, mock_pdf_parser):
        """Test OCR text extraction from scanned PDF."""
        parser = mock_pdf_parser()
        expected_text = "OCR extracted text from scanned document"
        parser.extract_text_ocr.return_value = expected_text

        # Configure OCR options
        parser.set_options({"ocr_enabled": True, "ocr_language": "eng"})

        parsed_result = Mock()
        text = parser.extract_text_ocr(parsed_result)

        assert text == expected_text
        parser.extract_text_ocr.assert_called_once_with(parsed_result)
        parser.set_options.assert_called_with(
            {"ocr_enabled": True, "ocr_language": "eng"}
        )

    def test_extract_text_multilingual(self, mock_pdf_parser):
        """Test text extraction from multilingual PDF."""
        parser = mock_pdf_parser()
        expected_text = "English text. Texto en español. Texte français."
        parser.extract_text.return_value = expected_text

        parser.set_options({"languages": ["en", "es", "fr"]})

        parsed_result = Mock()
        text = parser.extract_text(parsed_result)

        assert "English" in text
        assert "español" in text
        assert "français" in text
        parser.set_options.assert_called_with({"languages": ["en", "es", "fr"]})

    def test_extract_text_with_coordinates(self, mock_pdf_parser):
        """Test extracting text with position coordinates."""
        parser = mock_pdf_parser()
        mock_result = Mock()
        mock_result.text_blocks = [
            {"text": "Title", "x": 100, "y": 700, "width": 200, "height": 20},
            {"text": "Content", "x": 100, "y": 650, "width": 400, "height": 400},
        ]
        parser.extract_text_with_layout.return_value = mock_result

        parsed_result = Mock()
        result = parser.extract_text_with_layout(parsed_result)

        assert len(result.text_blocks) == 2
        assert result.text_blocks[0]["text"] == "Title"
        assert "x" in result.text_blocks[0]

    def test_extract_text_error_handling(self, mock_pdf_parser):
        """Test error handling during text extraction."""
        parser = mock_pdf_parser()
        parser.extract_text.side_effect = Exception("Text extraction failed")

        parsed_result = Mock()

        with pytest.raises(Exception, match="Text extraction failed"):
            parser.extract_text(parsed_result)


class TestPDFParserMetadataExtraction:
    """Test PDF metadata extraction functionality."""

    def test_extract_metadata_basic(self, mock_pdf_parser, sample_pdf_metadata):
        """Test basic metadata extraction from PDF."""
        parser = mock_pdf_parser()
        parser.extract_metadata.return_value = sample_pdf_metadata

        parsed_result = Mock()
        metadata = parser.extract_metadata(parsed_result)

        assert metadata == sample_pdf_metadata
        assert metadata["title"] == "Test Document Title"
        assert metadata["author"] == "Test Author"
        assert metadata["pages"] == 10
        parser.extract_metadata.assert_called_once_with(parsed_result)

    def test_extract_title(self, mock_pdf_parser):
        """Test extracting document title."""
        parser = mock_pdf_parser()
        expected_title = "Machine Learning for Ontology Extraction"
        parser.extract_title.return_value = expected_title

        parsed_result = Mock()
        title = parser.extract_title(parsed_result)

        assert title == expected_title
        parser.extract_title.assert_called_once_with(parsed_result)

    def test_extract_authors(self, mock_pdf_parser):
        """Test extracting document authors."""
        parser = mock_pdf_parser()
        expected_authors = ["Dr. Jane Smith", "Prof. John Doe", "Dr. Alice Johnson"]
        parser.extract_authors.return_value = expected_authors

        parsed_result = Mock()
        authors = parser.extract_authors(parsed_result)

        assert authors == expected_authors
        assert len(authors) == 3
        parser.extract_authors.assert_called_once_with(parsed_result)

    def test_extract_doi(self, mock_pdf_parser):
        """Test extracting DOI from PDF."""
        parser = mock_pdf_parser()
        expected_doi = "10.1000/182"
        parser.extract_doi.return_value = expected_doi

        parsed_result = Mock()
        doi = parser.extract_doi(parsed_result)

        assert doi == expected_doi
        parser.extract_doi.assert_called_once_with(parsed_result)

    def test_extract_abstract(self, mock_pdf_parser):
        """Test extracting abstract from scientific paper."""
        parser = mock_pdf_parser()
        expected_abstract = (
            "This paper presents novel approaches for ontology extraction..."
        )
        parser.extract_abstract.return_value = expected_abstract

        parsed_result = Mock()
        abstract = parser.extract_abstract(parsed_result)

        assert abstract == expected_abstract
        assert "ontology extraction" in abstract
        parser.extract_abstract.assert_called_once_with(parsed_result)

    def test_extract_keywords(self, mock_pdf_parser):
        """Test extracting keywords from PDF."""
        parser = mock_pdf_parser()
        expected_keywords = [
            "ontology",
            "machine learning",
            "information extraction",
            "NLP",
        ]
        parser.extract_keywords.return_value = expected_keywords

        parsed_result = Mock()
        keywords = parser.extract_keywords(parsed_result)

        assert keywords == expected_keywords
        assert "ontology" in keywords
        parser.extract_keywords.assert_called_once_with(parsed_result)

    def test_extract_metadata_with_fallback(self, mock_pdf_parser):
        """Test metadata extraction with fallback methods."""
        parser = mock_pdf_parser()

        # Primary extraction fails, fallback succeeds
        mock_metadata = {
            "title": "Extracted from content",
            "author": "Unknown",
            "extraction_method": "content_analysis",
            "confidence": 0.7,
        }
        parser.extract_metadata.return_value = mock_metadata

        parsed_result = Mock()
        metadata = parser.extract_metadata(parsed_result)

        assert metadata["title"] == "Extracted from content"
        assert metadata["extraction_method"] == "content_analysis"
        assert metadata["confidence"] == 0.7

    def test_extract_metadata_empty_pdf(self, mock_pdf_parser):
        """Test metadata extraction from PDF with no metadata."""
        parser = mock_pdf_parser()
        empty_metadata = {
            "title": None,
            "author": None,
            "pages": 1,
            "file_size": 1024,
            "has_metadata": False,
        }
        parser.extract_metadata.return_value = empty_metadata

        parsed_result = Mock()
        metadata = parser.extract_metadata(parsed_result)

        assert metadata["title"] is None
        assert metadata["author"] is None
        assert metadata["has_metadata"] is False


class TestPDFParserSectionIdentification:
    """Test section identification in scientific papers."""

    def test_identify_sections(self, mock_pdf_parser):
        """Test identifying document sections."""
        parser = mock_pdf_parser()
        expected_sections = {
            "abstract": {"start_page": 1, "start_pos": 100, "end_pos": 500},
            "introduction": {"start_page": 1, "start_pos": 600, "end_pos": 1200},
            "methods": {"start_page": 2, "start_pos": 0, "end_pos": 800},
            "results": {"start_page": 3, "start_pos": 0, "end_pos": 1000},
            "discussion": {"start_page": 4, "start_pos": 0, "end_pos": 600},
            "conclusion": {"start_page": 4, "start_pos": 700, "end_pos": 900},
            "references": {"start_page": 5, "start_pos": 0, "end_pos": -1},
        }
        parser.identify_sections.return_value = expected_sections

        parsed_result = Mock()
        sections = parser.identify_sections(parsed_result)

        assert sections == expected_sections
        assert "abstract" in sections
        assert "methods" in sections
        parser.identify_sections.assert_called_once_with(parsed_result)

    def test_extract_introduction(self, mock_pdf_parser):
        """Test extracting introduction section."""
        parser = mock_pdf_parser()
        expected_intro = (
            "The field of ontology information extraction has grown significantly..."
        )
        parser.extract_introduction.return_value = expected_intro

        parsed_result = Mock()
        introduction = parser.extract_introduction(parsed_result)

        assert introduction == expected_intro
        assert "ontology information extraction" in introduction
        parser.extract_introduction.assert_called_once_with(parsed_result)

    def test_extract_methods(self, mock_pdf_parser):
        """Test extracting methods section."""
        parser = mock_pdf_parser()
        expected_methods = (
            "We employed a combination of natural language processing techniques..."
        )
        parser.extract_methods.return_value = expected_methods

        parsed_result = Mock()
        methods = parser.extract_methods(parsed_result)

        assert methods == expected_methods
        assert "natural language processing" in methods
        parser.extract_methods.assert_called_once_with(parsed_result)

    def test_extract_results(self, mock_pdf_parser):
        """Test extracting results section."""
        parser = mock_pdf_parser()
        expected_results = (
            "Our experiments show that the proposed method achieves 95% accuracy..."
        )
        parser.extract_results.return_value = expected_results

        parsed_result = Mock()
        results = parser.extract_results(parsed_result)

        assert results == expected_results
        assert "95% accuracy" in results
        parser.extract_results.assert_called_once_with(parsed_result)

    def test_extract_discussion(self, mock_pdf_parser):
        """Test extracting discussion section."""
        parser = mock_pdf_parser()
        expected_discussion = (
            "The results demonstrate the effectiveness of our approach..."
        )
        parser.extract_discussion.return_value = expected_discussion

        parsed_result = Mock()
        discussion = parser.extract_discussion(parsed_result)

        assert discussion == expected_discussion
        parser.extract_discussion.assert_called_once_with(parsed_result)

    def test_extract_conclusion(self, mock_pdf_parser):
        """Test extracting conclusion section."""
        parser = mock_pdf_parser()
        expected_conclusion = "In conclusion, we have presented a robust method for ontology extraction..."
        parser.extract_conclusion.return_value = expected_conclusion

        parsed_result = Mock()
        conclusion = parser.extract_conclusion(parsed_result)

        assert conclusion == expected_conclusion
        assert "ontology extraction" in conclusion
        parser.extract_conclusion.assert_called_once_with(parsed_result)

    def test_section_identification_with_confidence(self, mock_pdf_parser):
        """Test section identification with confidence scores."""
        parser = mock_pdf_parser()
        expected_sections = {
            "abstract": {
                "content": "Abstract content...",
                "confidence": 0.95,
                "method": "header_detection",
            },
            "introduction": {
                "content": "Introduction content...",
                "confidence": 0.90,
                "method": "pattern_matching",
            },
            "methods": {
                "content": "Methods content...",
                "confidence": 0.85,
                "method": "ml_classification",
            },
        }
        parser.identify_sections.return_value = expected_sections

        parsed_result = Mock()
        sections = parser.identify_sections(parsed_result)

        assert sections["abstract"]["confidence"] == 0.95
        assert sections["methods"]["method"] == "ml_classification"

    def test_section_identification_multilevel_headers(self, mock_pdf_parser):
        """Test identifying sections with multilevel headers."""
        parser = mock_pdf_parser()
        expected_sections = {
            "methods": {
                "content": "Methods overview...",
                "subsections": {
                    "data_collection": {"content": "Data collection methods..."},
                    "analysis": {"content": "Analysis procedures..."},
                    "validation": {"content": "Validation approach..."},
                },
            }
        }
        parser.identify_sections.return_value = expected_sections

        parsed_result = Mock()
        sections = parser.identify_sections(parsed_result)

        assert "methods" in sections
        assert "subsections" in sections["methods"]
        assert "data_collection" in sections["methods"]["subsections"]


class TestPDFParserFigureTableExtraction:
    """Test figure and table caption extraction."""

    def test_extract_figures(self, mock_pdf_parser):
        """Test extracting figure information."""
        parser = mock_pdf_parser()
        expected_figures = [
            {
                "id": "figure_1",
                "caption": "Figure 1: Architecture of the proposed system",
                "page": 3,
                "position": {"x": 100, "y": 200, "width": 400, "height": 300},
                "type": "diagram",
            },
            {
                "id": "figure_2",
                "caption": "Figure 2: Performance comparison with baseline methods",
                "page": 5,
                "position": {"x": 50, "y": 150, "width": 500, "height": 350},
                "type": "chart",
            },
        ]
        parser.extract_figures.return_value = expected_figures

        parsed_result = Mock()
        figures = parser.extract_figures(parsed_result)

        assert figures == expected_figures
        assert len(figures) == 2
        assert figures[0]["caption"].startswith("Figure 1:")
        parser.extract_figures.assert_called_once_with(parsed_result)

    def test_extract_tables(self, mock_pdf_parser):
        """Test extracting table information."""
        parser = mock_pdf_parser()
        expected_tables = [
            {
                "id": "table_1",
                "caption": "Table 1: Dataset statistics and characteristics",
                "page": 4,
                "position": {"x": 100, "y": 400, "width": 400, "height": 200},
                "columns": 4,
                "rows": 10,
                "data": [["Dataset", "Size", "Features", "Labels"]],
            },
            {
                "id": "table_2",
                "caption": "Table 2: Experimental results and performance metrics",
                "page": 6,
                "position": {"x": 80, "y": 300, "width": 450, "height": 250},
                "columns": 5,
                "rows": 8,
                "data": [["Method", "Precision", "Recall", "F1", "Accuracy"]],
            },
        ]
        parser.extract_tables.return_value = expected_tables

        parsed_result = Mock()
        tables = parser.extract_tables(parsed_result)

        assert tables == expected_tables
        assert len(tables) == 2
        assert tables[0]["columns"] == 4
        parser.extract_tables.assert_called_once_with(parsed_result)

    def test_extract_captions(self, mock_pdf_parser):
        """Test extracting all captions from document."""
        parser = mock_pdf_parser()
        expected_captions = [
            {
                "type": "figure",
                "number": 1,
                "text": "Architecture of the proposed system",
            },
            {"type": "figure", "number": 2, "text": "Performance comparison"},
            {"type": "table", "number": 1, "text": "Dataset statistics"},
            {"type": "table", "number": 2, "text": "Experimental results"},
        ]
        parser.extract_captions.return_value = expected_captions

        parsed_result = Mock()
        captions = parser.extract_captions(parsed_result)

        assert captions == expected_captions
        assert len(captions) == 4
        figure_captions = [c for c in captions if c["type"] == "figure"]
        assert len(figure_captions) == 2
        parser.extract_captions.assert_called_once_with(parsed_result)

    def test_extract_figures_with_image_data(self, mock_pdf_parser):
        """Test extracting figures with actual image data."""
        parser = mock_pdf_parser()
        expected_figures = [
            {
                "id": "figure_1",
                "caption": "Figure 1: System architecture",
                "page": 2,
                "image_data": b"mock_image_bytes",
                "image_format": "PNG",
                "image_size": (800, 600),
                "has_image": True,
            }
        ]
        parser.extract_figures.return_value = expected_figures

        parser.set_options({"extract_images": True})

        parsed_result = Mock()
        figures = parser.extract_figures(parsed_result)

        assert figures[0]["has_image"] is True
        assert figures[0]["image_format"] == "PNG"
        parser.set_options.assert_called_with({"extract_images": True})

    def test_extract_tables_with_structured_data(self, mock_pdf_parser):
        """Test extracting tables with structured data."""
        parser = mock_pdf_parser()
        expected_tables = [
            {
                "id": "table_1",
                "caption": "Table 1: Results summary",
                "page": 3,
                "structured_data": {
                    "headers": ["Method", "Accuracy", "Time"],
                    "rows": [
                        ["Method A", "95.2%", "10.5s"],
                        ["Method B", "92.8%", "8.3s"],
                        ["Method C", "97.1%", "12.7s"],
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
        assert len(tables[0]["structured_data"]["rows"]) == 3
        parser.set_options.assert_called_with({"extract_table_data": True})


class TestPDFParserReferenceExtraction:
    """Test reference parsing and extraction."""

    def test_extract_references(self, mock_pdf_parser):
        """Test extracting bibliography references."""
        parser = mock_pdf_parser()
        expected_references = [
            {
                "id": "ref_1",
                "authors": ["Smith, J.", "Doe, J."],
                "title": "Ontology extraction methods in scientific literature",
                "journal": "Nature Machine Intelligence",
                "year": 2022,
                "volume": 3,
                "pages": "567-578",
                "doi": "10.1038/s42256-022-00567-8",
            },
            {
                "id": "ref_2",
                "authors": ["Johnson, A.", "Brown, B."],
                "title": "Machine learning approaches for information extraction",
                "conference": "ICML 2021",
                "year": 2021,
                "pages": "123-135",
            },
        ]
        parser.extract_references.return_value = expected_references

        parsed_result = Mock()
        references = parser.extract_references(parsed_result)

        assert references == expected_references
        assert len(references) == 2
        assert references[0]["journal"] == "Nature Machine Intelligence"
        parser.extract_references.assert_called_once_with(parsed_result)

    def test_parse_citations(self, mock_pdf_parser):
        """Test parsing in-text citations."""
        parser = mock_pdf_parser()
        expected_citations = [
            {
                "text": "Smith et al. (2022)",
                "page": 2,
                "position": 150,
                "reference_id": "ref_1",
                "type": "author_year",
            },
            {
                "text": "[1, 3]",
                "page": 3,
                "position": 300,
                "reference_ids": ["ref_1", "ref_3"],
                "type": "numeric",
            },
        ]
        parser.parse_citations.return_value = expected_citations

        parsed_result = Mock()
        citations = parser.parse_citations(parsed_result)

        assert citations == expected_citations
        assert len(citations) == 2
        assert citations[0]["type"] == "author_year"
        parser.parse_citations.assert_called_once_with(parsed_result)

    def test_extract_references_with_parsing_confidence(self, mock_pdf_parser):
        """Test reference extraction with parsing confidence."""
        parser = mock_pdf_parser()
        expected_references = [
            {
                "id": "ref_1",
                "raw_text": "Smith, J. et al. (2022). Title here. Journal, 123, 456-789.",
                "parsed_data": {
                    "authors": ["Smith, J."],
                    "title": "Title here",
                    "journal": "Journal",
                    "year": 2022,
                },
                "confidence": 0.95,
                "parsing_method": "regex_pattern",
            },
            {
                "id": "ref_2",
                "raw_text": "Incomplete reference...",
                "parsed_data": {"title": "Incomplete reference"},
                "confidence": 0.3,
                "parsing_method": "fallback",
            },
        ]
        parser.extract_references.return_value = expected_references

        parsed_result = Mock()
        references = parser.extract_references(parsed_result)

        high_confidence_refs = [r for r in references if r["confidence"] > 0.8]
        assert len(high_confidence_refs) == 1
        assert references[0]["parsing_method"] == "regex_pattern"

    def test_extract_references_different_formats(self, mock_pdf_parser):
        """Test extracting references in different citation styles."""
        parser = mock_pdf_parser()
        expected_references = [
            {
                "id": "ref_1",
                "style": "APA",
                "authors": ["Smith, J."],
                "year": 2022,
                "title": "Research paper",
                "journal": "Science",
            },
            {
                "id": "ref_2",
                "style": "IEEE",
                "authors": ["A. Johnson"],
                "title": "Technical paper",
                "conference": "ICML",
                "year": 2021,
            },
            {
                "id": "ref_3",
                "style": "MLA",
                "authors": ["Brown, Bob"],
                "title": "Academic work",
                "publisher": "Academic Press",
                "year": 2020,
            },
        ]
        parser.extract_references.return_value = expected_references

        parsed_result = Mock()
        references = parser.extract_references(parsed_result)

        styles = [ref["style"] for ref in references]
        assert "APA" in styles
        assert "IEEE" in styles
        assert "MLA" in styles


class TestPDFParserConversion:
    """Test converting parsed PDF to internal models."""

    def test_to_ontology_conversion(
        self, mock_pdf_parser, sample_scientific_paper_content
    ):
        """Test converting parsed PDF to Ontology model."""
        parser = mock_pdf_parser()

        # Mock the conversion result
        mock_ontology = Mock()
        mock_ontology.id = "pdf_ontology_001"
        mock_ontology.name = "PDF Extracted Ontology"
        mock_ontology.terms = {}
        mock_ontology.relationships = {}
        mock_ontology.metadata = {
            "source_type": "scientific_paper",
            "title": sample_scientific_paper_content["title"],
            "authors": sample_scientific_paper_content["authors"],
        }
        parser.to_ontology.return_value = mock_ontology

        # Parse and convert
        parsed_result = Mock()
        ontology = parser.to_ontology(parsed_result)

        assert ontology == mock_ontology
        assert ontology.id == "pdf_ontology_001"
        assert ontology.metadata["source_type"] == "scientific_paper"

    def test_extract_terms_from_parsed_pdf(self, mock_pdf_parser):
        """Test extracting Term objects from parsed PDF."""
        parser = mock_pdf_parser()

        # Mock extracted terms
        mock_term1 = Mock()
        mock_term1.id = "TERM:001"
        mock_term1.name = "Machine Learning"
        mock_term1.definition = (
            "A method of data analysis that automates analytical model building"
        )
        mock_term1.context = "abstract"

        mock_term2 = Mock()
        mock_term2.id = "TERM:002"
        mock_term2.name = "Ontology"
        mock_term2.definition = (
            "A formal representation of knowledge as a set of concepts"
        )
        mock_term2.context = "introduction"

        mock_terms = [mock_term1, mock_term2]
        parser.extract_terms.return_value = mock_terms

        parsed_result = Mock()
        terms = parser.extract_terms(parsed_result)

        assert len(terms) == 2
        assert terms[0].name == "Machine Learning"
        assert terms[1].context == "introduction"

    def test_extract_relationships_from_parsed_pdf(self, mock_pdf_parser):
        """Test extracting Relationship objects from parsed PDF."""
        parser = mock_pdf_parser()

        # Mock extracted relationships
        mock_relationships = [
            Mock(
                id="REL:001",
                subject="Machine Learning",
                predicate="is_used_for",
                object="Ontology Extraction",
                confidence=0.85,
                source="methods_section",
            ),
            Mock(
                id="REL:002",
                subject="Neural Networks",
                predicate="is_type_of",
                object="Machine Learning",
                confidence=0.90,
                source="related_work",
            ),
        ]
        parser.extract_relationships.return_value = mock_relationships

        parsed_result = Mock()
        relationships = parser.extract_relationships(parsed_result)

        assert len(relationships) == 2
        assert relationships[0].predicate == "is_used_for"
        assert relationships[1].confidence == 0.90

    def test_conversion_with_section_mapping(self, mock_pdf_parser):
        """Test conversion with section-based term mapping."""
        parser = mock_pdf_parser()

        # Configure section mapping
        section_mapping = {
            "abstract": {"weight": 0.9, "context": "high_level"},
            "introduction": {"weight": 0.7, "context": "background"},
            "methods": {"weight": 0.8, "context": "technical"},
            "results": {"weight": 0.6, "context": "experimental"},
        }
        parser.set_options({"section_mapping": section_mapping})

        mock_ontology = Mock()
        parser.to_ontology.return_value = mock_ontology

        parsed_result = Mock()
        ontology = parser.to_ontology(parsed_result)

        assert ontology == mock_ontology
        parser.set_options.assert_called_with({"section_mapping": section_mapping})

    def test_conversion_with_citation_analysis(self, mock_pdf_parser):
        """Test conversion incorporating citation analysis."""
        parser = mock_pdf_parser()

        # Configure citation analysis
        parser.set_options({"include_citations": True, "citation_weight": 0.1})

        mock_ontology = Mock()
        mock_ontology.metadata = {
            "citation_count": 25,
            "referenced_ontologies": ["ChEBI", "Gene Ontology"],
            "citation_network": {"nodes": 50, "edges": 120},
        }
        parser.to_ontology.return_value = mock_ontology

        parsed_result = Mock()
        ontology = parser.to_ontology(parsed_result)

        assert ontology.metadata["citation_count"] == 25
        assert "ChEBI" in ontology.metadata["referenced_ontologies"]

    def test_conversion_preserves_provenance(self, mock_pdf_parser):
        """Test that conversion preserves provenance information."""
        parser = mock_pdf_parser()

        mock_ontology = Mock()
        mock_ontology.metadata = {
            "source_format": "pdf",
            "source_file": "/path/to/paper.pdf",
            "parser_version": "1.0.0",
            "parsing_timestamp": "2023-01-01T00:00:00Z",
            "extraction_methods": ["text", "metadata", "sections"],
            "document_metadata": {
                "title": "Original Paper Title",
                "authors": ["Author 1", "Author 2"],
                "doi": "10.1000/182",
            },
        }
        parser.to_ontology.return_value = mock_ontology

        parsed_result = Mock()
        ontology = parser.to_ontology(parsed_result)

        assert "source_format" in ontology.metadata
        assert "document_metadata" in ontology.metadata
        assert ontology.metadata["source_format"] == "pdf"


class TestPDFParserErrorHandling:
    """Test error handling and validation."""

    def test_parse_corrupted_pdf(self, mock_pdf_parser):
        """Test parsing corrupted PDF raises appropriate error."""
        parser = mock_pdf_parser()
        parser.parse_file.side_effect = ValueError("Corrupted PDF: invalid structure")

        with pytest.raises(ValueError, match="Corrupted PDF"):
            parser.parse_file("/path/to/corrupted.pdf")

    def test_parse_encrypted_pdf_without_password(self, mock_pdf_parser):
        """Test parsing encrypted PDF without password raises error."""
        parser = mock_pdf_parser()
        parser.parse_file.side_effect = ValueError(
            "PDF is encrypted, password required"
        )

        with pytest.raises(ValueError, match="encrypted, password required"):
            parser.parse_file("/path/to/encrypted.pdf")

    def test_parse_nonexistent_file(self, mock_pdf_parser):
        """Test parsing nonexistent file raises FileNotFoundError."""
        parser = mock_pdf_parser()
        parser.parse_file.side_effect = FileNotFoundError(
            "File not found: /nonexistent/file.pdf"
        )

        with pytest.raises(FileNotFoundError, match="File not found"):
            parser.parse_file("/nonexistent/file.pdf")

    def test_parse_invalid_pdf_format(self, mock_pdf_parser):
        """Test parsing invalid PDF format raises appropriate error."""
        parser = mock_pdf_parser()
        parser.parse_bytes.side_effect = ValueError(
            "Invalid PDF format: missing header"
        )

        invalid_content = b"This is not a PDF file"

        with pytest.raises(ValueError, match="Invalid PDF format"):
            parser.parse_bytes(invalid_content)

    def test_text_extraction_failure(self, mock_pdf_parser):
        """Test handling of text extraction failure."""
        parser = mock_pdf_parser()
        parser.extract_text.side_effect = Exception(
            "Text extraction failed: corrupted encoding"
        )

        parsed_result = Mock()

        with pytest.raises(Exception, match="Text extraction failed"):
            parser.extract_text(parsed_result)

    def test_ocr_failure_fallback(self, mock_pdf_parser):
        """Test OCR failure fallback to regular text extraction."""
        parser = mock_pdf_parser()

        # OCR fails, but regular extraction succeeds
        mock_result = Mock()
        mock_result.text = "Fallback text extraction result"
        mock_result.extraction_method = "fallback"
        mock_result.errors = ["OCR failed, using fallback method"]
        parser.extract_text_ocr.return_value = mock_result

        parser.set_options({"ocr_enabled": True, "fallback_on_ocr_failure": True})

        parsed_result = Mock()
        result = parser.extract_text_ocr(parsed_result)

        assert result.extraction_method == "fallback"
        assert "OCR failed" in result.errors[0]

    def test_memory_limit_exceeded(self, mock_pdf_parser):
        """Test handling of memory limit exceeded during parsing."""
        parser = mock_pdf_parser()
        parser.parse_file.side_effect = MemoryError(
            "Memory limit exceeded while parsing large PDF file"
        )

        with pytest.raises(MemoryError, match="Memory limit exceeded"):
            parser.parse_file("/path/to/huge_document.pdf")

    def test_validation_errors_collection(self, mock_pdf_parser):
        """Test collection and reporting of validation errors."""
        parser = mock_pdf_parser()

        # Mock validation errors
        validation_errors = [
            "Warning: Document structure appears non-standard",
            "Error: Unable to extract title from metadata",
            "Warning: Some figures may be missing captions",
        ]
        parser.get_validation_errors.return_value = validation_errors

        # Parse with validation
        parser.set_options({"validate_on_parse": True})
        parser.parse_file("/path/to/document.pdf")

        errors = parser.get_validation_errors()
        assert len(errors) == 3
        assert any("title" in error for error in errors)
        assert any("figures" in error for error in errors)

    def test_unsupported_pdf_version(self, mock_pdf_parser):
        """Test handling of unsupported PDF version."""
        parser = mock_pdf_parser()
        parser.parse_file.side_effect = ValueError("Unsupported PDF version: 2.0")

        with pytest.raises(ValueError, match="Unsupported PDF version"):
            parser.parse_file("/path/to/pdf_v2.pdf")

    def test_damaged_pdf_recovery(self, mock_pdf_parser):
        """Test recovery from damaged PDF sections."""
        parser = mock_pdf_parser()

        mock_result = Mock()
        mock_result.pages_processed = 8
        mock_result.pages_total = 10
        mock_result.errors = ["Page 7 corrupted", "Page 9 unreadable"]
        mock_result.text = "Partial text extraction"
        parser.parse_file.return_value = mock_result

        parser.set_options({"error_recovery": True, "skip_damaged_pages": True})

        result = parser.parse_file("/path/to/damaged.pdf")

        assert result.pages_processed == 8
        assert len(result.errors) == 2
        assert "Partial text extraction" in result.text


class TestPDFParserValidation:
    """Test PDF validation functionality."""

    def test_validate_pdf_structure(self, mock_pdf_parser, sample_pdf_bytes):
        """Test validating PDF structure and format."""
        parser = mock_pdf_parser()
        parser.validate.return_value = True

        is_valid = parser.validate(sample_pdf_bytes)

        assert is_valid is True
        parser.validate.assert_called_once_with(sample_pdf_bytes)

    def test_validate_pdf_comprehensive(self, mock_pdf_parser, sample_pdf_bytes):
        """Test comprehensive PDF validation."""
        parser = mock_pdf_parser()
        parser.validate_pdf.return_value = {
            "valid_structure": True,
            "valid_metadata": True,
            "extractable_text": True,
            "readable_pages": 10,
            "total_pages": 10,
            "encryption_status": "none",
            "errors": [],
            "warnings": [],
        }

        validation_result = parser.validate_pdf(sample_pdf_bytes)

        assert validation_result["valid_structure"] is True
        assert validation_result["extractable_text"] is True
        assert validation_result["readable_pages"] == 10
        assert len(validation_result["errors"]) == 0

    def test_validate_scientific_paper_structure(self, mock_pdf_parser):
        """Test validation of scientific paper structure."""
        parser = mock_pdf_parser()
        parser.validate_pdf.return_value = {
            "valid_structure": True,
            "has_abstract": True,
            "has_introduction": True,
            "has_methods": True,
            "has_results": True,
            "has_references": True,
            "section_completeness": 0.9,
            "paper_type": "research_article",
        }

        validation_result = parser.validate_pdf("paper_content")

        assert validation_result["has_abstract"] is True
        assert validation_result["section_completeness"] == 0.9
        assert validation_result["paper_type"] == "research_article"

    def test_validate_pdf_accessibility(self, mock_pdf_parser, sample_pdf_bytes):
        """Test validation of PDF accessibility features."""
        parser = mock_pdf_parser()
        parser.validate_pdf.return_value = {
            "valid_structure": True,
            "has_bookmarks": False,
            "has_alt_text": False,
            "tagged_pdf": False,
            "accessibility_score": 0.3,
            "accessibility_issues": [
                "No alt text for images",
                "Missing document structure tags",
            ],
        }

        validation_result = parser.validate_pdf(sample_pdf_bytes)

        assert validation_result["accessibility_score"] == 0.3
        assert len(validation_result["accessibility_issues"]) == 2

    def test_validate_pdf_content_extractability(self, mock_pdf_parser):
        """Test validation of content extractability."""
        parser = mock_pdf_parser()
        parser.validate_pdf.return_value = {
            "valid_structure": True,
            "text_extractable": True,
            "images_extractable": True,
            "tables_extractable": True,
            "metadata_available": True,
            "extraction_quality": 0.85,
            "scanned_pages": 0,
            "total_pages": 10,
        }

        validation_result = parser.validate_pdf("content")

        assert validation_result["text_extractable"] is True
        assert validation_result["extraction_quality"] == 0.85
        assert validation_result["scanned_pages"] == 0


class TestPDFParserOptions:
    """Test parser configuration and options."""

    def test_set_parsing_options(self, mock_pdf_parser):
        """Test setting various parsing options."""
        parser = mock_pdf_parser()

        options = {
            "extraction_method": "pdfplumber",
            "ocr_enabled": True,
            "ocr_language": "eng",
            "password": None,
            "page_range": (1, 10),
            "layout_mode": True,
            "extract_images": True,
            "extract_tables": True,
        }

        parser.set_options(options)
        parser.set_options.assert_called_once_with(options)

    def test_get_current_options(self, mock_pdf_parser):
        """Test getting current parser options."""
        parser = mock_pdf_parser()

        expected_options = {
            "extraction_method": "pypdf",
            "ocr_enabled": False,
            "password": None,
            "layout_mode": False,
            "validate_on_parse": True,
        }
        parser.get_options.return_value = expected_options

        current_options = parser.get_options()

        assert current_options == expected_options
        assert "extraction_method" in current_options

    def test_reset_options_to_defaults(self, mock_pdf_parser):
        """Test resetting options to default values."""
        parser = mock_pdf_parser()

        # Set some custom options first
        parser.set_options({"extraction_method": "fitz", "ocr_enabled": True})

        # Reset to defaults
        parser.reset_options()
        parser.reset_options.assert_called_once()

    def test_invalid_option_handling(self, mock_pdf_parser):
        """Test handling of invalid configuration options."""
        parser = mock_pdf_parser()
        parser.set_options.side_effect = ValueError("Unknown option: invalid_method")

        invalid_options = {"invalid_method": "unknown"}

        with pytest.raises(ValueError, match="Unknown option"):
            parser.set_options(invalid_options)

    def test_option_validation(self, mock_pdf_parser):
        """Test validation of option values."""
        parser = mock_pdf_parser()
        parser.set_options.side_effect = ValueError(
            "Invalid value for option 'page_range': must be tuple"
        )

        invalid_options = {"page_range": "invalid"}

        with pytest.raises(ValueError, match="Invalid value for option"):
            parser.set_options(invalid_options)

    def test_extraction_method_options(self, mock_pdf_parser):
        """Test different extraction method configurations."""
        parser = mock_pdf_parser()

        # Test pypdf method
        parser.set_options({"extraction_method": "pypdf"})
        parser.set_options.assert_called_with({"extraction_method": "pypdf"})

        # Test pdfplumber method
        parser.reset_mock()
        parser.set_options({"extraction_method": "pdfplumber"})
        parser.set_options.assert_called_with({"extraction_method": "pdfplumber"})

        # Test fitz method
        parser.reset_mock()
        parser.set_options({"extraction_method": "fitz"})
        parser.set_options.assert_called_with({"extraction_method": "fitz"})


class TestPDFParserIntegration:
    """Integration tests with other components."""

    def test_integration_with_ontology_manager(self, mock_pdf_parser):
        """Test integration with OntologyManager."""
        parser = mock_pdf_parser()

        # Mock integration with ontology manager
        with patch(
            "aim2_project.aim2_ontology.ontology_manager.OntologyManager", create=True
        ) as MockManager:
            manager = MockManager()
            manager.add_ontology = Mock(return_value=True)

            # Parse and add to manager
            mock_ontology = Mock()
            parser.to_ontology.return_value = mock_ontology

            parsed_result = parser.parse_file("/path/to/paper.pdf")
            ontology = parser.to_ontology(parsed_result)
            manager.add_ontology(ontology)

            manager.add_ontology.assert_called_once_with(ontology)

    def test_integration_with_term_validation(self, mock_pdf_parser):
        """Test integration with term validation pipeline."""
        parser = mock_pdf_parser()

        # Mock validation pipeline
        with patch(
            "aim2_project.aim2_ontology.validators.ValidationPipeline", create=True
        ) as MockPipeline:
            validator = MockPipeline()
            validator.validate_terms = Mock(return_value={"valid": True, "errors": []})

            mock_terms = [Mock(), Mock(), Mock()]
            parser.extract_terms.return_value = mock_terms

            parsed_result = parser.parse_file("/path/to/paper.pdf")
            terms = parser.extract_terms(parsed_result)

            # Run validation
            validation_result = validator.validate_terms(terms)

            assert validation_result["valid"] is True
            validator.validate_terms.assert_called_once_with(terms)

    def test_integration_with_nlp_pipeline(self, mock_pdf_parser):
        """Test integration with NLP processing pipeline."""
        parser = mock_pdf_parser()

        # Create mock NLP processor without trying to patch non-existent modules
        mock_nlp_processor = Mock()
        mock_nlp_processor.process_text = Mock(
            return_value={
                "entities": ["machine learning", "ontology"],
                "relationships": [
                    {"subject": "ML", "predicate": "uses", "object": "data"}
                ],
            }
        )

        # Extract text and process with NLP
        mock_text = "Machine learning is used for ontology extraction."
        parser.extract_text.return_value = mock_text

        parsed_result = parser.parse_file("/path/to/paper.pdf")
        text = parser.extract_text(parsed_result)
        nlp_result = mock_nlp_processor.process_text(text)

        assert "machine learning" in nlp_result["entities"]
        mock_nlp_processor.process_text.assert_called_once_with(mock_text)

    def test_end_to_end_parsing_workflow(
        self, mock_pdf_parser, sample_scientific_paper_content
    ):
        """Test complete end-to-end parsing workflow."""
        parser = mock_pdf_parser()

        # Mock the complete workflow
        mock_parsed = Mock()
        mock_ontology = Mock()
        mock_ontology.terms = {
            "TERM:001": Mock(),
            "TERM:002": Mock(),
            "TERM:003": Mock(),
        }
        mock_ontology.relationships = {
            "REL:001": Mock(),
            "REL:002": Mock(),
        }
        mock_ontology.metadata = sample_scientific_paper_content

        parser.parse_file.return_value = mock_parsed
        parser.to_ontology.return_value = mock_ontology

        # Execute workflow
        parsed_result = parser.parse_file("/path/to/scientific_paper.pdf")
        ontology = parser.to_ontology(parsed_result)

        # Verify results
        assert len(ontology.terms) == 3
        assert len(ontology.relationships) == 2
        assert "title" in ontology.metadata

        parser.parse_file.assert_called_once_with("/path/to/scientific_paper.pdf")
        parser.to_ontology.assert_called_once_with(mock_parsed)


class TestPDFParserPerformance:
    """Performance and scalability tests."""

    def test_parse_large_pdf_performance(self, mock_pdf_parser):
        """Test parsing performance with large PDF files."""
        parser = mock_pdf_parser()

        # Configure for performance testing
        parser.set_options(
            {"memory_efficient": True, "lazy_loading": True, "chunk_size": 10}
        )

        mock_result = Mock()
        mock_result.pages = 500
        mock_result.text_length = 1000000  # 1M characters
        mock_result.processing_time = 45.2  # seconds
        parser.parse_file.return_value = mock_result

        result = parser.parse_file("/path/to/large_document.pdf")

        assert result.pages == 500
        assert result.text_length == 1000000
        assert result.processing_time < 60  # Should process within 1 minute

    def test_memory_usage_optimization(self, mock_pdf_parser):
        """Test memory usage optimization features."""
        parser = mock_pdf_parser()

        # Configure memory optimization
        parser.set_options(
            {"memory_limit": "512MB", "streaming_mode": True, "page_cache_size": 5}
        )

        mock_result = Mock()
        mock_result.memory_usage = "450MB"
        mock_result.peak_memory = "480MB"
        parser.parse_file.return_value = mock_result

        result = parser.parse_file("/path/to/memory_intensive.pdf")

        assert result == mock_result
        # Memory usage should be within limit
        assert float(result.memory_usage.replace("MB", "")) < 512

    def test_concurrent_processing_support(self, mock_pdf_parser):
        """Test support for concurrent processing operations."""
        parser = mock_pdf_parser()

        # Configure for concurrent processing
        parser.set_options({"parallel_pages": True, "worker_threads": 4})

        files = [
            "/path/to/paper1.pdf",
            "/path/to/paper2.pdf",
            "/path/to/paper3.pdf",
            "/path/to/paper4.pdf",
        ]

        mock_results = [Mock(), Mock(), Mock(), Mock()]
        parser.parse_file.side_effect = mock_results

        results = []
        for file_path in files:
            result = parser.parse_file(file_path)
            results.append(result)

        assert len(results) == 4
        assert parser.parse_file.call_count == 4

    def test_caching_mechanisms(self, mock_pdf_parser):
        """Test caching mechanisms for repeated parsing operations."""
        parser = mock_pdf_parser()

        # Configure caching
        parser.set_options({"enable_cache": True, "cache_size": 100, "cache_ttl": 3600})

        mock_result = Mock()
        parser.parse_file.return_value = mock_result

        # Parse same file twice
        file_path = "/path/to/cached_document.pdf"
        result1 = parser.parse_file(file_path)
        result2 = parser.parse_file(file_path)

        assert result1 == mock_result
        assert result2 == mock_result

        # In a real implementation, the second call might use cache
        # For now, we just verify the calls were made
        assert parser.parse_file.call_count == 2

    def test_batch_processing_capabilities(self, mock_pdf_parser):
        """Test batch processing of multiple PDF files."""
        parser = mock_pdf_parser()

        # Configure for batch processing
        parser.set_options({"batch_mode": True, "batch_size": 10})

        mock_batch_result = Mock()
        mock_batch_result.processed_files = 10
        mock_batch_result.total_pages = 150
        mock_batch_result.success_rate = 0.95
        parser.parse_file.return_value = mock_batch_result

        result = parser.parse_file("/path/to/batch_directory/")

        assert result.processed_files == 10
        assert result.success_rate == 0.95


# Additional test fixtures for complex scenarios
@pytest.fixture
def complex_scientific_paper():
    """Fixture providing complex scientific paper structure for testing."""
    return {
        "metadata": {
            "title": "Advanced Machine Learning Techniques for Biomedical Ontology Information Extraction",
            "authors": [
                {
                    "name": "Dr. Sarah Johnson",
                    "affiliation": "MIT",
                    "orcid": "0000-0001-2345-6789",
                },
                {
                    "name": "Prof. Michael Chen",
                    "affiliation": "Stanford",
                    "orcid": "0000-0002-3456-7890",
                },
                {
                    "name": "Dr. Emily Rodriguez",
                    "affiliation": "Harvard",
                    "orcid": "0000-0003-4567-8901",
                },
            ],
            "doi": "10.1038/s41586-023-12345-6",
            "journal": "Nature",
            "volume": 615,
            "issue": 7952,
            "pages": "123-135",
            "publication_date": "2023-03-15",
            "keywords": [
                "machine learning",
                "ontology",
                "biomedical",
                "information extraction",
            ],
        },
        "sections": {
            "abstract": {
                "text": "Background: Extracting structured information from biomedical literature...",
                "word_count": 250,
                "key_terms": [
                    "biomedical ontology",
                    "information extraction",
                    "neural networks",
                ],
            },
            "introduction": {
                "text": "The rapid growth of biomedical literature presents unprecedented challenges...",
                "subsections": ["Background", "Related Work", "Contributions"],
                "citations": 25,
            },
            "methods": {
                "text": "We developed a novel deep learning architecture...",
                "subsections": [
                    "Dataset",
                    "Model Architecture",
                    "Training Procedure",
                    "Evaluation Metrics",
                ],
                "figures": 3,
                "tables": 2,
            },
            "results": {
                "text": "Our proposed method achieved state-of-the-art performance...",
                "subsections": [
                    "Benchmark Results",
                    "Ablation Study",
                    "Qualitative Analysis",
                ],
                "figures": 5,
                "tables": 4,
            },
            "discussion": {
                "text": "The experimental results demonstrate several key findings...",
                "subsections": ["Performance Analysis", "Limitations", "Future Work"],
                "citations": 15,
            },
            "conclusion": {
                "text": "In conclusion, we have presented a comprehensive approach...",
                "word_count": 200,
            },
        },
        "figures": [
            {
                "id": "fig1",
                "caption": "Overview of the proposed architecture",
                "page": 3,
            },
            {"id": "fig2", "caption": "Training and validation curves", "page": 5},
            {"id": "fig3", "caption": "Comparison with baseline methods", "page": 7},
        ],
        "tables": [
            {"id": "tab1", "caption": "Dataset statistics", "page": 4},
            {"id": "tab2", "caption": "Hyperparameter settings", "page": 5},
            {"id": "tab3", "caption": "Performance comparison", "page": 8},
        ],
        "references": {
            "count": 45,
            "styles": ["Nature", "IEEE"],
            "types": ["journal", "conference", "book", "preprint"],
        },
    }


@pytest.fixture
def malformed_pdf_scenarios():
    """Fixture providing malformed PDF scenarios for error testing."""
    return {
        "corrupted_header": b"CORRUPTED%PDF-1.4\nmalformed content",
        "truncated_file": b"%PDF-1.4\n1 0 obj\n<<\n/Type /Catalog\n",  # Incomplete
        "encrypted_no_password": b"%PDF-1.4\n/Encrypt 123 0 R\nencrypted content",
        "invalid_xref": b"%PDF-1.4\ncontent\nxref\ninvalid_xref_table\n%%EOF",
        "missing_eof": b"%PDF-1.4\n1 0 obj\n<<\n/Type /Catalog\n>>\nendobj\n",  # No %%EOF
    }


@pytest.fixture
def pdf_configuration_options():
    """Fixture providing comprehensive PDF parser configuration options."""
    return {
        "extraction_options": {
            "method": "pdfplumber",  # pypdf, pdfplumber, fitz
            "fallback_methods": ["pypdf", "fitz"],
            "ocr_enabled": False,
            "ocr_language": "eng",
            "ocr_fallback": True,
        },
        "text_processing_options": {
            "preserve_layout": False,
            "extract_coordinates": False,
            "clean_text": True,
            "normalize_whitespace": True,
            "remove_headers_footers": True,
        },
        "content_extraction_options": {
            "extract_metadata": True,
            "extract_images": False,
            "extract_tables": True,
            "extract_figures": True,
            "extract_captions": True,
            "extract_references": True,
        },
        "section_identification_options": {
            "identify_sections": True,
            "section_patterns": [
                "abstract",
                "introduction",
                "methods",
                "results",
                "discussion",
                "conclusion",
            ],
            "header_detection": True,
            "confidence_threshold": 0.7,
        },
        "performance_options": {
            "memory_efficient": False,
            "lazy_loading": False,
            "page_cache_size": 10,
            "parallel_pages": False,
            "worker_threads": 1,
        },
        "validation_options": {
            "validate_on_parse": True,
            "check_extractability": True,
            "validate_structure": True,
            "accessibility_check": False,
        },
        "error_handling_options": {
            "error_recovery": True,
            "skip_damaged_pages": True,
            "max_errors": 10,
            "continue_on_error": True,
            "fallback_on_failure": True,
        },
    }


@pytest.fixture
def temp_pdf_files():
    """Create temporary PDF files for testing."""
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)

        # Create various test files
        files = {}

        # Simple PDF file (mock content)
        simple_pdf = temp_path / "simple.pdf"
        simple_pdf.write_bytes(
            b"%PDF-1.4\n1 0 obj\n<<\n/Type /Catalog\n>>\nendobj\n%%EOF"
        )
        files["simple"] = str(simple_pdf)

        # Large PDF file (mock content)
        large_pdf = temp_path / "large.pdf"
        content = b"%PDF-1.4\n" + b"Mock large content\n" * 1000 + b"%%EOF"
        large_pdf.write_bytes(content)
        files["large"] = str(large_pdf)

        # Scientific paper PDF (mock content)
        paper_pdf = temp_path / "scientific_paper.pdf"
        paper_content = b"%PDF-1.4\nMock scientific paper content with sections\n%%EOF"
        paper_pdf.write_bytes(paper_content)
        files["paper"] = str(paper_pdf)

        yield files


# Edge case and advanced functionality tests
class TestPDFParserAdvancedFeatures:
    """Test advanced PDF parser features."""

    def test_multilingual_document_processing(self, mock_pdf_parser):
        """Test processing multilingual scientific documents."""
        parser = mock_pdf_parser()

        mock_result = Mock()
        mock_result.languages_detected = ["en", "de", "fr"]
        mock_result.text_by_language = {
            "en": "English abstract and conclusion",
            "de": "Deutsche Zusammenfassung",
            "fr": "Résumé français",
        }
        parser.extract_text.return_value = mock_result

        parser.set_options({"multilingual": True, "detect_languages": True})

        parsed_result = Mock()
        result = parser.extract_text(parsed_result)

        assert len(result.languages_detected) == 3
        assert "en" in result.text_by_language

    def test_document_version_comparison(self, mock_pdf_parser):
        """Test comparing different versions of the same document."""
        parser = mock_pdf_parser()

        mock_result = Mock()
        mock_result.version_changes = {
            "added_sections": ["supplementary_material"],
            "modified_sections": ["results", "discussion"],
            "removed_content": [],
        }
        mock_result.similarity_score = 0.87
        parser.parse_file.return_value = mock_result

        parser.set_options(
            {"version_comparison": True, "baseline_version": "/path/to/v1.pdf"}
        )

        result = parser.parse_file("/path/to/v2.pdf")

        assert result.similarity_score == 0.87
        assert "supplementary_material" in result.version_changes["added_sections"]

    def test_collaborative_annotation_extraction(self, mock_pdf_parser):
        """Test extracting annotations and comments from collaborative documents."""
        parser = mock_pdf_parser()

        mock_annotations = [
            {
                "type": "highlight",
                "text": "important finding",
                "page": 5,
                "author": "reviewer1",
                "timestamp": "2023-01-15T10:30:00Z",
            },
            {
                "type": "comment",
                "text": "Consider adding more details",
                "page": 7,
                "author": "reviewer2",
                "timestamp": "2023-01-16T14:20:00Z",
            },
        ]
        parser.extract_text.return_value = Mock(annotations=mock_annotations)

        parser.set_options({"extract_annotations": True})

        parsed_result = Mock()
        result = parser.extract_text(parsed_result)

        assert len(result.annotations) == 2
        assert result.annotations[0]["type"] == "highlight"

    def test_adaptive_extraction_strategy(self, mock_pdf_parser):
        """Test adaptive extraction strategy based on document characteristics."""
        parser = mock_pdf_parser()

        mock_result = Mock()
        mock_result.extraction_strategy = "scientific_paper"
        mock_result.confidence = 0.92
        mock_result.document_type = "research_article"
        parser.parse_file.return_value = mock_result

        parser.set_options({"adaptive_strategy": True, "auto_detect_type": True})

        result = parser.parse_file("/path/to/document.pdf")

        assert result.extraction_strategy == "scientific_paper"
        assert result.confidence > 0.9
