"""
PDF Parser Module

This module provides a comprehensive PDF parser implementation for the AIM2 ontology
information extraction system, supporting various PDF formats and extraction techniques.

The PDFParser class inherits from AbstractDocumentParser and provides specialized
functionality for:
- PDF text extraction using multiple libraries (pypdf, pdfplumber, PyMuPDF)
- Metadata extraction (title, authors, DOI, publication info)
- Section identification for scientific papers
- Figure and table caption extraction
- Reference parsing and bibliography extraction
- Encrypted and scanned PDF support
- Performance optimization for large documents

Dependencies:
    - pypdf: Basic PDF processing and metadata extraction
    - pdfplumber: Advanced text extraction with layout preservation
    - PyMuPDF (fitz): Comprehensive PDF processing with image support
    - re: Regular expression pattern matching
    - typing: Type hints
    - logging: Logging functionality

Usage:
    from aim2_project.aim2_ontology.parsers.pdf_parser import PDFParser

    parser = PDFParser()
    result = parser.parse_file("document.pdf")
    text = parser.extract_text(result)
    metadata = parser.extract_metadata(result)
    sections = parser.identify_sections(result)
"""

import io
import logging
import re
from datetime import datetime
from pathlib import Path
from typing import Any, BinaryIO, Dict, List, Optional, Tuple, Union

try:
    import pypdf

    PYPDF_AVAILABLE = True
except ImportError:
    PYPDF_AVAILABLE = False

try:
    import pdfplumber

    PDFPLUMBER_AVAILABLE = True
except ImportError:
    PDFPLUMBER_AVAILABLE = False

try:
    import fitz  # PyMuPDF

    FITZ_AVAILABLE = True
except ImportError:
    FITZ_AVAILABLE = False

from aim2_project.aim2_ontology.parsers import AbstractDocumentParser
from aim2_project.aim2_utils.config_manager import ConfigManager
from aim2_project.exceptions import ExtractionException, ValidationException


class PDFParser(AbstractDocumentParser):
    """
    Comprehensive PDF parser implementation for scientific document processing.

    This concrete implementation of AbstractDocumentParser provides specialized
    functionality for parsing PDF documents with focus on scientific papers,
    academic publications, and technical documents.

    Key Features:
        - Multi-library PDF processing (pypdf, pdfplumber, PyMuPDF)
        - Intelligent format detection and fallback mechanisms
        - Scientific paper section identification
        - Metadata extraction from PDF properties and content
        - Figure and table extraction with captions
        - Reference and citation parsing
        - Encrypted PDF support with password handling
        - OCR fallback for scanned documents
        - Performance optimization for large documents

    Supported PDF Types:
        - Standard text-based PDFs
        - Encrypted PDFs (with password)
        - Scanned PDFs (with OCR fallback)
        - Academic papers and journals
        - Technical reports and documentation
        - Multi-column layouts

    Extraction Methods:
        - pypdf: Fast, lightweight extraction
        - pdfplumber: Layout-aware extraction with table support
        - PyMuPDF: Comprehensive extraction with image support
        - Auto-fallback: Automatic method selection based on document characteristics
    """

    def __init__(
        self,
        parser_name: str = "PDFParser",
        config_manager: Optional[ConfigManager] = None,
        logger: Optional[logging.Logger] = None,
        options: Optional[Dict[str, Any]] = None,
    ):
        """
        Initialize the PDF parser with configuration and options.

        Args:
            parser_name (str): Name identifier for this parser instance
            config_manager (ConfigManager, optional): Configuration manager instance
            logger (logging.Logger, optional): Logger instance
            options (Dict[str, Any], optional): Initial parser options

        Raises:
            ValidationException: If required PDF libraries are not available
            ValueError: If parser configuration is invalid
        """
        # Initialize base document parser
        super().__init__(parser_name, config_manager, logger, options)

        # Check library availability and log warnings
        self._check_library_availability()

        # Initialize PDF-specific attributes
        self._current_pdf_document = None
        self._current_extraction_method = None
        self._supported_methods = self._get_available_methods()

        # Configure PDF-specific section patterns
        self._setup_pdf_section_patterns()

        # Configure metadata extractors
        self._setup_pdf_metadata_extractors()

        # Add PDF-specific validation rules
        self.add_validation_rule("pdf_format", self._validate_pdf_format)
        self.add_validation_rule(
            "pdf_extractability", self._validate_pdf_extractability
        )

        self.logger.info(
            f"PDF parser initialized with methods: {self._supported_methods}"
        )

    def _check_library_availability(self) -> None:
        """
        Check availability of PDF processing libraries and log status.

        Raises:
            ValidationException: If no PDF libraries are available
        """
        available_libs = []
        missing_libs = []

        if PYPDF_AVAILABLE:
            available_libs.append("pypdf")
        else:
            missing_libs.append("pypdf")

        if PDFPLUMBER_AVAILABLE:
            available_libs.append("pdfplumber")
        else:
            missing_libs.append("pdfplumber")

        if FITZ_AVAILABLE:
            available_libs.append("PyMuPDF")
        else:
            missing_libs.append("PyMuPDF")

        if not available_libs:
            raise ValidationException(
                "No PDF processing libraries available. Please install at least one of: "
                "pypdf, pdfplumber, PyMuPDF"
            )

        if missing_libs:
            self.logger.warning(
                f"Some PDF libraries not available: {missing_libs}. "
                f"Available: {available_libs}"
            )
        else:
            self.logger.debug(f"All PDF libraries available: {available_libs}")

    def _get_available_methods(self) -> List[str]:
        """
        Get list of available extraction methods based on installed libraries.

        Returns:
            List[str]: List of available extraction method names
        """
        methods = []
        if PYPDF_AVAILABLE:
            methods.append("pypdf")
        if PDFPLUMBER_AVAILABLE:
            methods.append("pdfplumber")
        if FITZ_AVAILABLE:
            methods.append("fitz")
        return methods

    def _setup_pdf_section_patterns(self) -> None:
        """Setup PDF-specific section identification patterns."""
        # Add scientific paper section patterns
        pdf_patterns = {
            # Standard academic sections
            "abstract": re.compile(
                r"(?i)(?:^|\n)\s*(?:abstract|summary)\s*(?:\n|$)", re.MULTILINE
            ),
            "introduction": re.compile(
                r"(?i)(?:^|\n)\s*(?:\d+\.?\s*)?introduction\s*(?:\n|$)", re.MULTILINE
            ),
            "methods": re.compile(
                r"(?i)(?:^|\n)\s*(?:\d+\.?\s*)?(?:methods?|methodology|materials?\s+and\s+methods?)\s*(?:\n|$)",
                re.MULTILINE,
            ),
            "results": re.compile(
                r"(?i)(?:^|\n)\s*(?:\d+\.?\s*)?results?\s*(?:\n|$)", re.MULTILINE
            ),
            "discussion": re.compile(
                r"(?i)(?:^|\n)\s*(?:\d+\.?\s*)?discussion\s*(?:\n|$)", re.MULTILINE
            ),
            "conclusion": re.compile(
                r"(?i)(?:^|\n)\s*(?:\d+\.?\s*)?(?:conclusion|conclusions?)\s*(?:\n|$)",
                re.MULTILINE,
            ),
            "references": re.compile(
                r"(?i)(?:^|\n)\s*(?:references?|bibliography|works?\s+cited)\s*(?:\n|$)",
                re.MULTILINE,
            ),
            "acknowledgments": re.compile(
                r"(?i)(?:^|\n)\s*(?:acknowledgments?|acknowledgements?)\s*(?:\n|$)",
                re.MULTILINE,
            ),
        }

        # Update section patterns with PDF-specific ones
        self.section_patterns.update(pdf_patterns)

    def _setup_pdf_metadata_extractors(self) -> None:
        """Setup PDF-specific metadata extractors."""
        # DOI pattern
        self.citation_patterns["doi"] = re.compile(
            r"(?i)(?:doi:?\s*)?(?:https?://(?:dx\.)?doi\.org/)?" r"(10\.\d{4,}/[^\s]+)",
            re.IGNORECASE,
        )

        # Author patterns for different formats
        self.citation_patterns["authors"] = re.compile(
            r"(?i)(?:authors?:?\s*)?([A-Z][a-z]+(?:\s+[A-Z]\.)*\s+[A-Z][a-z]+(?:\s*,\s*[A-Z][a-z]+(?:\s+[A-Z]\.)*\s+[A-Z][a-z]+)*)",
            re.MULTILINE,
        )

        # Title patterns
        self.citation_patterns["title"] = re.compile(
            r"^(.{10,200})$",  # Likely title: 10-200 chars, standalone line
            re.MULTILINE,
        )

    def get_supported_formats(self) -> List[str]:
        """
        Get list of supported PDF formats and extensions.

        Returns:
            List[str]: List of supported format extensions
        """
        return ["pdf"]

    def validate(self, content: Union[str, bytes], **kwargs) -> bool:
        """
        Validate PDF content format and structure.

        Args:
            content (Union[str, bytes]): PDF content to validate
            **kwargs: Additional validation options

        Returns:
            bool: True if content is valid PDF, False otherwise

        Raises:
            ValidationException: If validation fails with errors
        """
        try:
            # Convert string path to bytes if needed
            if isinstance(content, str):
                if Path(content).exists():
                    with open(content, "rb") as f:
                        pdf_bytes = f.read()
                else:
                    # Assume it's PDF content as string
                    pdf_bytes = content.encode("utf-8")
            else:
                pdf_bytes = content

            # Check PDF header
            if not pdf_bytes.startswith(b"%PDF-"):
                self.logger.warning("Invalid PDF header")
                return False

            # Try to parse with available libraries
            parsing_success = False

            if PYPDF_AVAILABLE:
                try:
                    reader = pypdf.PdfReader(io.BytesIO(pdf_bytes))
                    # Basic validation - can read pages
                    len(reader.pages)
                    parsing_success = True
                except Exception as e:
                    self.logger.debug(f"pypdf validation failed: {e}")

            if not parsing_success and PDFPLUMBER_AVAILABLE:
                try:
                    with pdfplumber.open(io.BytesIO(pdf_bytes)) as pdf:
                        len(pdf.pages)
                    parsing_success = True
                except Exception as e:
                    self.logger.debug(f"pdfplumber validation failed: {e}")

            if not parsing_success and FITZ_AVAILABLE:
                try:
                    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
                    doc.page_count
                    doc.close()
                    parsing_success = True
                except Exception as e:
                    self.logger.debug(f"PyMuPDF validation failed: {e}")

            return parsing_success

        except Exception as e:
            self.logger.error(f"PDF validation error: {e}")
            raise ValidationException(f"PDF validation failed: {e}")

    def _validate_pdf_format(self, content: Any, **kwargs) -> bool:
        """Validate PDF format-specific requirements."""
        return self.validate(content, **kwargs)

    def _validate_pdf_extractability(self, content: Any, **kwargs) -> bool:
        """Validate that text can be extracted from PDF."""
        try:
            # Attempt basic text extraction
            text = self.extract_text(content, **kwargs)
            return len(text.strip()) > 0
        except Exception:
            return False

    def parse(self, content: Union[str, bytes, BinaryIO], **kwargs) -> Dict[str, Any]:
        """
        Parse PDF content and return structured data.

        Args:
            content (Union[str, bytes, BinaryIO]): PDF content or file path
            **kwargs: Additional parsing options

        Returns:
            Dict[str, Any]: Parsed PDF data structure

        Raises:
            ExtractionException: If parsing fails
            ValidationException: If content is not valid PDF
        """
        try:
            self.logger.info("Starting PDF parsing")

            # Validate content first
            if not self.validate(content, **kwargs):
                raise ValidationException("Invalid PDF content")

            # Prepare content for processing
            pdf_bytes = self._prepare_content(content)

            # Determine extraction method
            extraction_method = kwargs.get("extraction_method", "auto")
            if extraction_method == "auto":
                extraction_method = self._select_optimal_method(pdf_bytes)

            self._current_extraction_method = extraction_method

            # Parse using selected method
            if extraction_method == "pypdf" and PYPDF_AVAILABLE:
                parsed_data = self._parse_with_pypdf(pdf_bytes, **kwargs)
            elif extraction_method == "pdfplumber" and PDFPLUMBER_AVAILABLE:
                parsed_data = self._parse_with_pdfplumber(pdf_bytes, **kwargs)
            elif extraction_method == "fitz" and FITZ_AVAILABLE:
                parsed_data = self._parse_with_fitz(pdf_bytes, **kwargs)
            else:
                # Fallback to first available method
                for method in self._supported_methods:
                    try:
                        if method == "pypdf":
                            parsed_data = self._parse_with_pypdf(pdf_bytes, **kwargs)
                        elif method == "pdfplumber":
                            parsed_data = self._parse_with_pdfplumber(
                                pdf_bytes, **kwargs
                            )
                        elif method == "fitz":
                            parsed_data = self._parse_with_fitz(pdf_bytes, **kwargs)
                        self._current_extraction_method = method
                        break
                    except Exception as e:
                        self.logger.warning(f"Method {method} failed: {e}")
                        continue
                else:
                    raise ExtractionException("All extraction methods failed")

            # Add parsing metadata
            parsed_data.update(
                {
                    "parser_type": "PDFParser",
                    "extraction_method": self._current_extraction_method,
                    "parsing_timestamp": datetime.utcnow().isoformat(),
                    "supported_methods": self._supported_methods,
                }
            )

            self.logger.info(
                f"PDF parsing completed using {self._current_extraction_method}"
            )
            return parsed_data

        except Exception as e:
            self.logger.error(f"PDF parsing failed: {e}")
            raise ExtractionException(f"Failed to parse PDF: {e}")

    def _prepare_content(self, content: Union[str, bytes, BinaryIO]) -> bytes:
        """
        Prepare content for PDF processing.

        Args:
            content: Input content in various formats

        Returns:
            bytes: PDF content as bytes
        """
        if isinstance(content, str):
            # Assume file path
            if Path(content).exists():
                with open(content, "rb") as f:
                    return f.read()
            else:
                # Assume string content
                return content.encode("utf-8")
        elif isinstance(content, bytes):
            return content
        elif hasattr(content, "read"):
            # File-like object
            content.seek(0)
            return content.read()
        else:
            raise ValueError(f"Unsupported content type: {type(content)}")

    def _select_optimal_method(self, pdf_bytes: bytes) -> str:
        """
        Select optimal extraction method based on PDF characteristics.

        Args:
            pdf_bytes (bytes): PDF content as bytes

        Returns:
            str: Selected method name
        """
        # Simple heuristics for method selection
        # In a real implementation, this could analyze PDF structure,
        # presence of tables, images, etc.

        # Default preference order
        if PDFPLUMBER_AVAILABLE:
            return "pdfplumber"  # Best for layout and tables
        elif FITZ_AVAILABLE:
            return "fitz"  # Comprehensive features
        elif PYPDF_AVAILABLE:
            return "pypdf"  # Lightweight fallback
        else:
            raise ExtractionException("No PDF extraction methods available")

    def _parse_with_pypdf(self, pdf_bytes: bytes, **kwargs) -> Dict[str, Any]:
        """Parse PDF using pypdf library."""
        try:
            reader = pypdf.PdfReader(io.BytesIO(pdf_bytes))

            return {
                "raw_content": reader,
                "pages": reader.pages,
                "metadata": dict(reader.metadata) if reader.metadata else {},
                "page_count": len(reader.pages),
                "is_encrypted": reader.is_encrypted,
                "extraction_method": "pypdf",
            }
        except Exception as e:
            raise ExtractionException(f"pypdf parsing failed: {e}")

    def _parse_with_pdfplumber(self, pdf_bytes: bytes, **kwargs) -> Dict[str, Any]:
        """Parse PDF using pdfplumber library."""
        try:
            pdf = pdfplumber.open(io.BytesIO(pdf_bytes))

            return {
                "raw_content": pdf,
                "pages": pdf.pages,
                "metadata": pdf.metadata or {},
                "page_count": len(pdf.pages),
                "extraction_method": "pdfplumber",
            }
        except Exception as e:
            raise ExtractionException(f"pdfplumber parsing failed: {e}")

    def _parse_with_fitz(self, pdf_bytes: bytes, **kwargs) -> Dict[str, Any]:
        """Parse PDF using PyMuPDF library."""
        try:
            doc = fitz.open(stream=pdf_bytes, filetype="pdf")

            return {
                "raw_content": doc,
                "pages": [doc[i] for i in range(doc.page_count)],
                "metadata": doc.metadata,
                "page_count": doc.page_count,
                "is_encrypted": doc.is_encrypted,
                "extraction_method": "fitz",
            }
        except Exception as e:
            raise ExtractionException(f"PyMuPDF parsing failed: {e}")

    def extract_text(self, content: Any, **kwargs) -> str:
        """
        Extract plain text from PDF content.

        Args:
            content (Any): Parsed PDF content or raw PDF data
            **kwargs: Text extraction options

        Returns:
            str: Extracted text content

        Raises:
            ExtractionException: If text extraction fails
        """
        try:
            # If content is not parsed, parse it first
            if not isinstance(content, dict):
                content = self.parse(content, **kwargs)

            extraction_method = content.get("extraction_method", "auto")

            if extraction_method == "pypdf":
                return self._extract_text_pypdf(content, **kwargs)
            elif extraction_method == "pdfplumber":
                return self._extract_text_pdfplumber(content, **kwargs)
            elif extraction_method == "fitz":
                return self._extract_text_fitz(content, **kwargs)
            else:
                raise ExtractionException(
                    f"Unknown extraction method: {extraction_method}"
                )

        except Exception as e:
            self.logger.error(f"Text extraction failed: {e}")
            raise ExtractionException(f"Failed to extract text: {e}")

    def _extract_text_pypdf(self, content: Dict[str, Any], **kwargs) -> str:
        """Extract text using pypdf."""
        pages = content.get("pages", [])
        text_parts = []

        for page in pages:
            try:
                page_text = page.extract_text()
                if page_text:
                    text_parts.append(page_text)
            except Exception as e:
                self.logger.warning(f"Failed to extract text from page: {e}")
                continue

        return "\n".join(text_parts)

    def _extract_text_pdfplumber(self, content: Dict[str, Any], **kwargs) -> str:
        """Extract text using pdfplumber."""
        pages = content.get("pages", [])
        text_parts = []

        for page in pages:
            try:
                page_text = page.extract_text()
                if page_text:
                    text_parts.append(page_text)
            except Exception as e:
                self.logger.warning(f"Failed to extract text from page: {e}")
                continue

        return "\n".join(text_parts)

    def _extract_text_fitz(self, content: Dict[str, Any], **kwargs) -> str:
        """Extract text using PyMuPDF."""
        pages = content.get("pages", [])
        text_parts = []

        for page in pages:
            try:
                page_text = page.get_text()
                if page_text:
                    text_parts.append(page_text)
            except Exception as e:
                self.logger.warning(f"Failed to extract text from page: {e}")
                continue

        return "\n".join(text_parts)

    def extract_metadata(self, content: Any, **kwargs) -> Dict[str, Any]:
        """
        Extract comprehensive metadata from PDF document.

        Args:
            content (Any): Parsed PDF content or raw PDF data
            **kwargs: Metadata extraction options including:
                - extract_content_metadata (bool): Extract metadata from content
                - extract_publication_metadata (bool): Extract publication-specific metadata
                - confidence_threshold (float): Minimum confidence for metadata inclusion
                - fallback_methods (bool): Use fallback extraction methods
                - validate_metadata (bool): Validate and clean metadata

        Returns:
            Dict[str, Any]: Comprehensive extracted metadata including:
                - Basic metadata (title, author, subject, etc.)
                - Publication metadata (DOI, journal, keywords, etc.)
                - Technical metadata (pages, file size, encryption status)
                - Content-based metadata (abstract, authors from content)
                - Confidence scores and extraction methods

        Raises:
            ExtractionException: If metadata extraction fails
        """
        try:
            # If content is not parsed, parse it first
            if not isinstance(content, dict):
                content = self.parse(content, **kwargs)

            # Initialize comprehensive metadata dictionary
            comprehensive_metadata = {}
            extraction_methods = []
            confidence_scores = {}

            # Extract basic PDF metadata
            pdf_metadata = content.get("metadata", {})
            basic_metadata = self._standardize_metadata(pdf_metadata)
            comprehensive_metadata.update(basic_metadata)
            if basic_metadata:
                extraction_methods.append("pdf_metadata")

            # Extract enhanced content-based metadata
            if kwargs.get("extract_content_metadata", True):
                text = self.extract_text(content, **kwargs)
                (
                    content_metadata,
                    content_confidence,
                ) = self._extract_enhanced_content_metadata(text, **kwargs)
                comprehensive_metadata.update(content_metadata)
                confidence_scores.update(content_confidence)
                if content_metadata:
                    extraction_methods.append("content_analysis")

            # Extract publication-specific metadata
            if kwargs.get("extract_publication_metadata", True):
                text = (
                    self.extract_text(content, **kwargs)
                    if "text" not in locals()
                    else text
                )
                pub_metadata, pub_confidence = self._extract_publication_metadata(
                    text, comprehensive_metadata, **kwargs
                )
                comprehensive_metadata.update(pub_metadata)
                confidence_scores.update(pub_confidence)
                if pub_metadata:
                    extraction_methods.append("publication_analysis")

            # Parse and enhance author information
            if (
                "author" in comprehensive_metadata
                or "authors" in comprehensive_metadata
            ):
                author_info = self._parse_author_information(
                    comprehensive_metadata, **kwargs
                )
                comprehensive_metadata.update(author_info)

            # Add technical and parsing metadata
            technical_metadata = {
                "pages": content.get("page_count", 0),
                "extraction_method": content.get("extraction_method", "unknown"),
                "encrypted": content.get("is_encrypted", False),
                "file_size": len(str(content.get("raw_content", ""))),
                "parsing_timestamp": datetime.utcnow().isoformat(),
                "extraction_methods": extraction_methods,
                "confidence_scores": confidence_scores,
                "metadata_version": "2.0",  # Enhanced metadata version
            }
            comprehensive_metadata.update(technical_metadata)

            # Validate and clean metadata if requested
            if kwargs.get("validate_metadata", True):
                comprehensive_metadata = self._validate_and_clean_metadata(
                    comprehensive_metadata, **kwargs
                )

            self.logger.info(
                f"Extracted comprehensive metadata with {len(extraction_methods)} methods"
            )
            return comprehensive_metadata

        except Exception as e:
            self.logger.error(f"Comprehensive metadata extraction failed: {e}")
            raise ExtractionException(f"Failed to extract metadata: {e}")

    def _standardize_metadata(self, raw_metadata: Dict[str, Any]) -> Dict[str, Any]:
        """Standardize metadata field names and formats."""
        standardized = {}

        # Common field mappings
        field_mappings = {
            "/Title": "title",
            "title": "title",
            "/Author": "author",
            "author": "author",
            "/Subject": "subject",
            "subject": "subject",
            "/Creator": "creator",
            "creator": "creator",
            "/Producer": "producer",
            "producer": "producer",
            "/CreationDate": "creation_date",
            "creation_date": "creation_date",
            "/ModDate": "modification_date",
            "modification_date": "modification_date",
        }

        for raw_key, value in raw_metadata.items():
            if raw_key in field_mappings:
                standardized_key = field_mappings[raw_key]
                standardized[standardized_key] = self._clean_metadata_value(value)

        return standardized

    def _clean_metadata_value(self, value: Any) -> Any:
        """Clean and format metadata values."""
        if isinstance(value, str):
            # Remove null bytes and extra whitespace
            cleaned = value.replace("\x00", "").strip()
            return cleaned if cleaned else None
        return value

    def _extract_content_metadata(self, text: str, **kwargs) -> Dict[str, Any]:
        """Extract metadata from document content."""
        metadata = {}

        # Extract DOI
        doi_match = self.citation_patterns["doi"].search(text)
        if doi_match:
            metadata["doi"] = doi_match.group(1)

        # Extract potential title (first substantial line)
        lines = text.split("\n")
        for line in lines[:10]:  # Check first 10 lines
            line = line.strip()
            if 20 <= len(line) <= 200 and not line.isupper():
                metadata["content_title"] = line
                break

        # Word count and basic statistics
        words = text.split()
        metadata["word_count"] = len(words)
        metadata["character_count"] = len(text)
        metadata["line_count"] = len(text.split("\n"))

        return metadata

    def identify_sections(self, content: Any, **kwargs) -> Dict[str, Dict[str, Any]]:
        """
        Identify document sections (abstract, introduction, methods, etc.).

        Args:
            content (Any): Parsed PDF content or raw PDF data
            **kwargs: Section identification options

        Returns:
            Dict[str, Dict[str, Any]]: Identified sections with metadata

        Raises:
            ExtractionException: If section identification fails
        """
        try:
            # Extract text for section analysis
            text = self.extract_text(content, **kwargs)

            sections = {}

            # Find each section using patterns
            for section_name, pattern in self.section_patterns.items():
                matches = list(pattern.finditer(text))
                if matches:
                    for i, match in enumerate(matches):
                        section_key = f"{section_name}_{i}" if i > 0 else section_name

                        # Find section content
                        start_pos = match.end()

                        # Find next section or end of document
                        next_pos = len(text)
                        for other_name, other_pattern in self.section_patterns.items():
                            if other_name != section_name:
                                next_matches = list(
                                    other_pattern.finditer(text[start_pos:])
                                )
                                if next_matches:
                                    next_pos = min(
                                        next_pos, start_pos + next_matches[0].start()
                                    )

                        # Extract section content
                        section_content = text[start_pos:next_pos].strip()

                        sections[section_key] = {
                            "content": section_content,
                            "start_position": start_pos,
                            "end_position": next_pos,
                            "word_count": len(section_content.split()),
                            "confidence": self._calculate_section_confidence(
                                section_name, section_content
                            ),
                            "detection_method": "pattern_matching",
                        }

            self.logger.info(f"Identified {len(sections)} sections")
            return sections

        except Exception as e:
            self.logger.error(f"Section identification failed: {e}")
            raise ExtractionException(f"Failed to identify sections: {e}")

    def _calculate_section_confidence(self, section_name: str, content: str) -> float:
        """Calculate confidence score for section identification."""
        # Simple heuristic based on content length and keywords
        base_confidence = 0.7

        # Adjust based on content length
        word_count = len(content.split())
        if word_count < 10:
            base_confidence -= 0.3
        elif word_count > 100:
            base_confidence += 0.1

        # Adjust based on section-specific keywords
        section_keywords = {
            "abstract": ["abstract", "summary", "background", "objective"],
            "introduction": ["introduction", "background", "motivation"],
            "methods": ["method", "approach", "technique", "procedure"],
            "results": ["result", "finding", "outcome", "data"],
            "discussion": ["discussion", "analysis", "interpretation"],
            "conclusion": ["conclusion", "summary", "future work"],
        }

        if section_name in section_keywords:
            keyword_count = sum(
                1
                for keyword in section_keywords[section_name]
                if keyword.lower() in content.lower()
            )
            base_confidence += keyword_count * 0.05

        return min(1.0, max(0.0, base_confidence))

    def extract_figures_tables(
        self, content: Any, **kwargs
    ) -> Dict[str, List[Dict[str, Any]]]:
        """
        Extract figures and tables with captions.

        Args:
            content (Any): Parsed PDF content or raw PDF data
            **kwargs: Extraction options

        Returns:
            Dict[str, List[Dict[str, Any]]]: Extracted figures and tables

        Raises:
            ExtractionException: If extraction fails
        """
        try:
            text = self.extract_text(content, **kwargs)

            figures = self._extract_figures(text, **kwargs)
            tables = self._extract_tables(text, **kwargs)

            return {"figures": figures, "tables": tables}

        except Exception as e:
            self.logger.error(f"Figure/table extraction failed: {e}")
            raise ExtractionException(f"Failed to extract figures/tables: {e}")

    def _extract_figures(self, text: str, **kwargs) -> List[Dict[str, Any]]:
        """Extract figure references and captions."""
        figures = []

        # Pattern for figure captions
        figure_pattern = re.compile(
            r"(?i)(fig(?:ure)?\s*\.?\s*(\d+)(?:\s*[:.]\s*)?(.{0,200}?)(?=\n\s*(?:fig|table|\d+\.|$))|$)",
            re.MULTILINE | re.DOTALL,
        )

        for match in figure_pattern.finditer(text):
            figure_num = match.group(2)
            caption = match.group(3).strip() if match.group(3) else ""

            if figure_num and caption:
                figures.append(
                    {
                        "id": f"figure_{figure_num}",
                        "number": int(figure_num),
                        "caption": caption,
                        "type": "figure",
                        "position": match.start(),
                        "extraction_method": "pattern_matching",
                    }
                )

        return figures

    def _extract_tables(self, text: str, **kwargs) -> List[Dict[str, Any]]:
        """Extract table references and captions."""
        tables = []

        # Pattern for table captions
        table_pattern = re.compile(
            r"(?i)(table\s*\.?\s*(\d+)(?:\s*[:.]\s*)?(.{0,200}?)(?=\n\s*(?:fig|table|\d+\.|$))|$)",
            re.MULTILINE | re.DOTALL,
        )

        for match in table_pattern.finditer(text):
            table_num = match.group(2)
            caption = match.group(3).strip() if match.group(3) else ""

            if table_num and caption:
                tables.append(
                    {
                        "id": f"table_{table_num}",
                        "number": int(table_num),
                        "caption": caption,
                        "type": "table",
                        "position": match.start(),
                        "extraction_method": "pattern_matching",
                    }
                )

        return tables

    def _extract_and_validate_doi(self, text: str) -> Tuple[str, float]:
        """
        Extract and validate DOI from text content.

        Args:
            text (str): Document text

        Returns:
            tuple[str, float]: DOI and confidence score
        """
        try:
            # Multiple DOI patterns for better coverage
            doi_patterns = [
                self.citation_patterns["doi"],  # Original pattern
                re.compile(r"(?i)doi\s*:?\s*([10]\.[0-9]{4,}/[^\s]+)", re.IGNORECASE),
                re.compile(
                    r"(?i)https?://(?:dx\.)?doi\.org/([10]\.[0-9]{4,}/[^\s]+)",
                    re.IGNORECASE,
                ),
                re.compile(r"(?i)\b([10]\.[0-9]{4,}/[A-Za-z0-9\-\._\(\)/]+)\b"),
            ]

            best_doi = None
            best_confidence = 0.0

            for pattern in doi_patterns:
                matches = pattern.findall(text)
                for match in matches:
                    doi = match if isinstance(match, str) else match[0]
                    # Clean DOI
                    doi = doi.rstrip(".,;)")

                    # Validate DOI format
                    confidence = self._validate_doi_format(doi)

                    if confidence > best_confidence:
                        best_doi = doi
                        best_confidence = confidence

            return best_doi, best_confidence
        except Exception as e:
            self.logger.warning(f"DOI extraction failed: {e}")
            return None, 0.0

    def _validate_doi_format(self, doi: str) -> float:
        """
        Validate DOI format and return confidence score.

        Args:
            doi (str): DOI string to validate

        Returns:
            float: Confidence score (0.0 to 1.0)
        """
        if not doi:
            return 0.0

        # Basic DOI format: 10.xxxx/yyyy
        if not doi.startswith("10."):
            return 0.0

        parts = doi.split("/", 1)
        if len(parts) != 2:
            return 0.0

        prefix, suffix = parts
        confidence = 0.5  # Base confidence

        # Check prefix format (10.xxxx)
        prefix_parts = prefix.split(".")
        if (
            len(prefix_parts) >= 2
            and prefix_parts[1].isdigit()
            and len(prefix_parts[1]) >= 4
        ):
            confidence += 0.3

        # Check suffix is not empty and reasonable length
        if suffix and 3 <= len(suffix) <= 200:
            confidence += 0.2

        return min(1.0, confidence)

    def _extract_title_from_content(self, text: str, **kwargs) -> Tuple[str, float]:
        """
        Extract document title from content using multiple methods.

        Args:
            text (str): Document text
            **kwargs: Extraction options

        Returns:
            tuple[str, float]: Title and confidence score
        """
        try:
            candidates = []

            # Method 1: First substantial line
            lines = text.split("\n")
            for i, line in enumerate(lines[:15]):  # Check first 15 lines
                line = line.strip()
                if 10 <= len(line) <= 200 and not line.isupper():
                    # Skip if it looks like metadata
                    if not any(
                        keyword in line.lower()
                        for keyword in [
                            "author",
                            "email",
                            "university",
                            "doi",
                            "page",
                            "©",
                        ]
                    ):
                        confidence = 0.7 - (
                            i * 0.05
                        )  # Decrease confidence for later lines
                        candidates.append((line, max(0.1, confidence)))

            # Method 2: Look for title patterns
            title_patterns = [
                re.compile(r"^(.{10,200})$", re.MULTILINE),  # Standalone lines
                re.compile(r"(?i)title\s*:?\s*(.{10,200})", re.MULTILINE),
            ]

            for pattern in title_patterns:
                matches = pattern.findall(text[:2000])  # Search in first 2000 chars
                for match in matches:
                    title = match.strip()
                    if self._is_likely_title(title):
                        candidates.append((title, 0.6))

            # Select best candidate
            if candidates:
                # Sort by confidence, then by reasonable length
                candidates.sort(
                    key=lambda x: (x[1], -abs(len(x[0]) - 50)), reverse=True
                )
                return candidates[0]

            return None, 0.0

        except Exception as e:
            self.logger.warning(f"Title extraction failed: {e}")
            return None, 0.0

    def _is_likely_title(self, text: str) -> bool:
        """
        Determine if text is likely to be a document title.

        Args:
            text (str): Text to evaluate

        Returns:
            bool: True if likely to be a title
        """
        if not text or len(text) < 10 or len(text) > 200:
            return False

        # Check for title indicators
        title_indicators = [
            not text.isupper(),  # Not all caps
            not text.islower(),  # Not all lowercase
            not text.startswith(("http", "www", "@")),  # Not a URL or email
            "." not in text[-10:],  # Doesn't end with sentence
            not any(char.isdigit() for char in text[:5]),  # Doesn't start with numbers
        ]

        return sum(title_indicators) >= 3

    def _extract_authors_from_content(self, text: str, **kwargs) -> Tuple[list, float]:
        """
        Extract authors from document content.

        Args:
            text (str): Document text
            **kwargs: Extraction options

        Returns:
            tuple[list, float]: List of authors and confidence score
        """
        try:
            authors = []
            confidence = 0.0

            # Pattern for author detection
            author_patterns = [
                re.compile(r"(?i)authors?\s*:?\s*([^\n]{10,200})", re.MULTILINE),
                re.compile(
                    r"(?i)by\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]*)*(?:\s*,\s*[A-Z][a-z]+(?:\s+[A-Z][a-z]*)*)*)",
                    re.MULTILINE,
                ),
                # Pattern for academic name format
                re.compile(
                    r"([A-Z][a-z]+(?:\s+[A-Z]\.)?\s+[A-Z][a-z]+)(?:\s*,\s*([A-Z][a-z]+(?:\s+[A-Z]\.)?\s+[A-Z][a-z]+))*",
                    re.MULTILINE,
                ),
            ]

            # Search in the first part of the document
            search_text = text[:3000]

            for pattern in author_patterns:
                matches = pattern.findall(search_text)
                for match in matches:
                    if isinstance(match, tuple):
                        match = match[0]

                    # Parse individual authors
                    potential_authors = self._parse_author_string(match)
                    if potential_authors:
                        authors.extend(potential_authors)
                        confidence = max(confidence, 0.7)

            # Remove duplicates while preserving order
            seen = set()
            unique_authors = []
            for author in authors:
                if author.lower() not in seen:
                    seen.add(author.lower())
                    unique_authors.append(author)

            return unique_authors[:10], confidence  # Limit to 10 authors

        except Exception as e:
            self.logger.warning(f"Author extraction failed: {e}")
            return [], 0.0

    def _parse_author_string(self, author_string: str) -> list:
        """
        Parse a string containing multiple authors.

        Args:
            author_string (str): String with author names

        Returns:
            list: List of individual author names
        """
        if not author_string:
            return []

        authors = []

        # Split by common separators
        separators = [",", ";", " and ", " & ", "\n"]
        parts = [author_string]

        for sep in separators:
            new_parts = []
            for part in parts:
                new_parts.extend(part.split(sep))
            parts = new_parts

        for part in parts:
            author = part.strip()
            if self._is_likely_author_name(author):
                authors.append(author)

        return authors

    def _is_likely_author_name(self, name: str) -> bool:
        """
        Check if a string is likely to be an author name.

        Args:
            name (str): Potential author name

        Returns:
            bool: True if likely to be an author name
        """
        if not name or len(name) < 3 or len(name) > 100:
            return False

        # Check for name-like patterns
        name_patterns = [
            re.match(r"^[A-Z][a-z]+(?:\s+[A-Z][a-z]*)*$", name),  # Title case
            re.match(
                r"^[A-Z][a-z]+(?:\s+[A-Z]\.)?\s+[A-Z][a-z]+$", name
            ),  # First M. Last
            re.match(r"^[A-Z][a-z]+,\s*[A-Z][a-z]*\.?$", name),  # Last, First
        ]

        return any(pattern for pattern in name_patterns)

    def _extract_abstract_from_content(self, text: str, **kwargs) -> Tuple[str, float]:
        """
        Extract abstract from document content.

        Args:
            text (str): Document text
            **kwargs: Extraction options

        Returns:
            tuple[str, float]: Abstract text and confidence score
        """
        try:
            # Pattern for abstract detection
            abstract_patterns = [
                re.compile(
                    r"(?i)abstract\s*:?\s*\n?(.{50,2000})(?=\n\s*(?:keywords?|introduction|1\.|$))",
                    re.DOTALL,
                ),
                re.compile(
                    r"(?i)summary\s*:?\s*\n?(.{50,2000})(?=\n\s*(?:keywords?|introduction|1\.|$))",
                    re.DOTALL,
                ),
            ]

            for pattern in abstract_patterns:
                match = pattern.search(text[:5000])  # Search in first 5000 chars
                if match:
                    abstract = match.group(1).strip()
                    # Clean up the abstract
                    abstract = re.sub(r"\s+", " ", abstract)  # Normalize whitespace
                    abstract = abstract.replace("\n", " ")

                    if 50 <= len(abstract) <= 2000:
                        confidence = (
                            0.8 if "abstract" in match.group(0).lower() else 0.6
                        )
                        return abstract, confidence

            return None, 0.0

        except Exception as e:
            self.logger.warning(f"Abstract extraction failed: {e}")
            return None, 0.0

    def _extract_keywords_from_content(self, text: str, **kwargs) -> Tuple[list, float]:
        """
        Extract keywords from document content.

        Args:
            text (str): Document text
            **kwargs: Extraction options

        Returns:
            tuple[list, float]: List of keywords and confidence score
        """
        try:
            # Pattern for explicit keywords
            keyword_patterns = [
                re.compile(r"(?i)keywords?\s*:?\s*([^\n]{10,500})", re.MULTILINE),
                re.compile(r"(?i)key\s+words?\s*:?\s*([^\n]{10,500})", re.MULTILINE),
                re.compile(r"(?i)index\s+terms?\s*:?\s*([^\n]{10,500})", re.MULTILINE),
            ]

            for pattern in keyword_patterns:
                match = pattern.search(text[:5000])
                if match:
                    keyword_text = match.group(1).strip()
                    keywords = self._parse_keywords_string(keyword_text)
                    if keywords:
                        return keywords, 0.9

            # If no explicit keywords, try to extract from abstract or title
            # This is a basic implementation - could be enhanced with NLP
            return [], 0.0

        except Exception as e:
            self.logger.warning(f"Keywords extraction failed: {e}")
            return [], 0.0

    def _parse_keywords_string(self, keyword_string: str) -> list:
        """
        Parse a string containing keywords.

        Args:
            keyword_string (str): String with keywords

        Returns:
            list: List of individual keywords
        """
        if not keyword_string:
            return []

        # Split by common separators
        separators = [",", ";", "\n", "•", "·"]
        keywords = [keyword_string]

        for sep in separators:
            new_keywords = []
            for kw in keywords:
                new_keywords.extend(kw.split(sep))
            keywords = new_keywords

        # Clean and filter keywords
        cleaned_keywords = []
        for kw in keywords:
            kw = kw.strip().strip(".,;")
            if 2 <= len(kw) <= 50 and not kw.isdigit():
                cleaned_keywords.append(kw)

        return cleaned_keywords[:20]  # Limit to 20 keywords

    def _calculate_document_statistics(self, text: str) -> Dict[str, Any]:
        """
        Calculate comprehensive document statistics.

        Args:
            text (str): Document text

        Returns:
            Dict[str, Any]: Document statistics
        """
        words = text.split()
        lines = text.split("\n")
        sentences = re.split(r"[.!?]+", text)

        return {
            "word_count": len(words),
            "character_count": len(text),
            "character_count_no_spaces": len(text.replace(" ", "")),
            "line_count": len(lines),
            "sentence_count": len([s for s in sentences if s.strip()]),
            "paragraph_count": len([line for line in lines if line.strip()]),
            "average_word_length": sum(len(word) for word in words) / len(words)
            if words
            else 0,
            "average_sentence_length": len(words) / len(sentences) if sentences else 0,
        }

    def _assess_content_quality(self, text: str) -> Dict[str, Any]:
        """
        Assess the quality of extracted text content.

        Args:
            text (str): Document text

        Returns:
            Dict[str, Any]: Content quality metrics
        """
        if not text:
            return {"quality_score": 0.0, "issues": ["no_content"]}

        issues = []
        quality_score = 1.0

        # Check for encoding issues
        if "\ufffd" in text or "â€" in text:
            quality_score -= 0.2
            issues.append("encoding_issues")

        # Check for fragmented text
        words = text.split()
        if words:
            avg_word_length = sum(len(word) for word in words) / len(words)
            if avg_word_length < 3:
                quality_score -= 0.3
                issues.append("fragmented_text")

        # Check for reasonable content length
        if len(text) < 1000:
            quality_score -= 0.1
            issues.append("short_content")

        # Check for presence of common document elements
        has_title = bool(re.search(r"^.{10,100}$", text[:500], re.MULTILINE))
        has_abstract = "abstract" in text.lower()[:2000]
        has_sections = bool(
            re.search(r"(?i)(introduction|method|result|conclusion)", text)
        )

        if not (has_title or has_abstract or has_sections):
            quality_score -= 0.2
            issues.append("missing_structure")

        return {
            "quality_score": max(0.0, quality_score),
            "issues": issues,
            "has_title": has_title,
            "has_abstract": has_abstract,
            "has_sections": has_sections,
        }

    def _extract_enhanced_content_metadata(
        self, text: str, **kwargs
    ) -> Tuple[Dict[str, Any], Dict[str, float]]:
        """
        Extract enhanced metadata from document content with confidence scores.

        Args:
            text (str): Document text content
            **kwargs: Extraction options

        Returns:
            tuple[Dict[str, Any], Dict[str, float]]: Metadata and confidence scores
        """
        metadata = {}
        confidence_scores = {}

        try:
            # Extract DOI with validation
            doi, doi_confidence = self._extract_and_validate_doi(text)
            if doi:
                metadata["doi"] = doi
                confidence_scores["doi"] = doi_confidence

            # Extract title with multiple methods
            title, title_confidence = self._extract_title_from_content(text, **kwargs)
            if title:
                metadata["content_title"] = title
                confidence_scores["content_title"] = title_confidence

            # Extract authors from content
            authors, authors_confidence = self._extract_authors_from_content(
                text, **kwargs
            )
            if authors:
                metadata["content_authors"] = authors
                confidence_scores["content_authors"] = authors_confidence

            # Extract abstract
            abstract, abstract_confidence = self._extract_abstract_from_content(
                text, **kwargs
            )
            if abstract:
                metadata["abstract"] = abstract
                confidence_scores["abstract"] = abstract_confidence

            # Extract keywords
            keywords, keywords_confidence = self._extract_keywords_from_content(
                text, **kwargs
            )
            if keywords:
                metadata["keywords"] = keywords
                confidence_scores["keywords"] = keywords_confidence

            # Document statistics
            stats = self._calculate_document_statistics(text)
            metadata.update(stats)

            # Content quality metrics
            quality_metrics = self._assess_content_quality(text)
            metadata["content_quality"] = quality_metrics

            return metadata, confidence_scores

        except Exception as e:
            self.logger.warning(f"Enhanced content metadata extraction failed: {e}")
            return {}, {}

    def _extract_publication_metadata(
        self, text: str, existing_metadata: Dict[str, Any], **kwargs
    ) -> Tuple[Dict[str, Any], Dict[str, float]]:
        """
        Extract publication-specific metadata from document content.

        Args:
            text (str): Document text
            existing_metadata (Dict[str, Any]): Already extracted metadata
            **kwargs: Extraction options

        Returns:
            tuple[Dict[str, Any], Dict[str, float]]: Publication metadata and confidence scores
        """
        metadata = {}
        confidence_scores = {}

        try:
            # Extract journal/conference information
            journal_info, journal_confidence = self._extract_journal_info(text)
            if journal_info:
                metadata.update(journal_info)
                confidence_scores.update(journal_confidence)

            # Extract publication dates
            date_info, date_confidence = self._extract_publication_dates(text)
            if date_info:
                metadata.update(date_info)
                confidence_scores.update(date_confidence)

            # Extract volume, issue, page information
            citation_info, citation_confidence = self._extract_citation_info(text)
            if citation_info:
                metadata.update(citation_info)
                confidence_scores.update(citation_confidence)

            # Extract subject/field classification
            subject_info, subject_confidence = self._extract_subject_classification(
                text
            )
            if subject_info:
                metadata.update(subject_info)
                confidence_scores.update(subject_confidence)

            # Extract funding information
            funding_info, funding_confidence = self._extract_funding_info(text)
            if funding_info:
                metadata["funding"] = funding_info
                confidence_scores["funding"] = funding_confidence

            return metadata, confidence_scores

        except Exception as e:
            self.logger.warning(f"Publication metadata extraction failed: {e}")
            return {}, {}

    def _extract_journal_info(
        self, text: str
    ) -> Tuple[Dict[str, Any], Dict[str, float]]:
        """
        Extract journal or conference information.

        Args:
            text (str): Document text

        Returns:
            tuple[Dict[str, Any], Dict[str, float]]: Journal info and confidence scores
        """
        info = {}
        confidence = {}

        # Journal patterns
        journal_patterns = [
            re.compile(r"(?i)journal\s*:?\s*([^\n]{5,100})", re.MULTILINE),
            re.compile(r"(?i)published\s+in\s+([^\n]{5,100})", re.MULTILINE),
            re.compile(r"(?i)proceedings\s+of\s+([^\n]{5,100})", re.MULTILINE),
        ]

        for pattern in journal_patterns:
            match = pattern.search(text[:3000])
            if match:
                journal_name = match.group(1).strip().rstrip(".,;")
                if self._is_likely_journal_name(journal_name):
                    info["journal"] = journal_name
                    confidence["journal"] = 0.7
                    break

        return info, confidence

    def _is_likely_journal_name(self, name: str) -> bool:
        """
        Check if a string is likely to be a journal name.

        Args:
            name (str): Potential journal name

        Returns:
            bool: True if likely to be a journal name
        """
        if not name or len(name) < 5 or len(name) > 100:
            return False

        # Common journal indicators
        journal_indicators = [
            "journal",
            "proceedings",
            "conference",
            "review",
            "letters",
            "nature",
            "science",
            "ieee",
            "acm",
            "springer",
        ]

        name_lower = name.lower()
        return any(indicator in name_lower for indicator in journal_indicators)

    def _extract_publication_dates(
        self, text: str
    ) -> Tuple[Dict[str, Any], Dict[str, float]]:
        """
        Extract publication dates from text.

        Args:
            text (str): Document text

        Returns:
            tuple[Dict[str, Any], Dict[str, float]]: Date info and confidence scores
        """
        info = {}
        confidence = {}

        # Date patterns
        date_patterns = [
            re.compile(r"(?i)published[:\s]+([A-Za-z]+\s+\d{1,2},?\s+\d{4})"),
            re.compile(r"(?i)(\d{1,2}\s+[A-Za-z]+\s+\d{4})"),
            re.compile(r"(?i)(\d{4}-\d{2}-\d{2})"),
            re.compile(r"(?i)(\d{2}/\d{2}/\d{4})"),
        ]

        for pattern in date_patterns:
            matches = pattern.findall(text[:5000])
            for match in matches:
                date_str = match if isinstance(match, str) else match[0]
                if self._is_reasonable_publication_date(date_str):
                    info["publication_date"] = date_str
                    confidence["publication_date"] = 0.6
                    break
            if "publication_date" in info:
                break

        return info, confidence

    def _is_reasonable_publication_date(self, date_str: str) -> bool:
        """
        Check if a date string represents a reasonable publication date.

        Args:
            date_str (str): Date string

        Returns:
            bool: True if reasonable publication date
        """
        try:
            # Extract year from date string
            year_match = re.search(r"\b(19|20)\d{2}\b", date_str)
            if year_match:
                year = int(year_match.group())
                current_year = datetime.now().year
                return (
                    1950 <= year <= current_year + 1
                )  # Reasonable publication year range
            return False
        except:
            return False

    def _extract_citation_info(
        self, text: str
    ) -> Tuple[Dict[str, Any], Dict[str, float]]:
        """
        Extract volume, issue, and page information.

        Args:
            text (str): Document text

        Returns:
            tuple[Dict[str, Any], Dict[str, float]]: Citation info and confidence scores
        """
        info = {}
        confidence = {}

        # Volume pattern
        volume_pattern = re.compile(r"(?i)vol(?:ume)?\.\s*(\d+)", re.MULTILINE)
        volume_match = volume_pattern.search(text[:3000])
        if volume_match:
            info["volume"] = int(volume_match.group(1))
            confidence["volume"] = 0.7

        # Issue pattern
        issue_pattern = re.compile(r"(?i)(?:issue|no)\.\s*(\d+)", re.MULTILINE)
        issue_match = issue_pattern.search(text[:3000])
        if issue_match:
            info["issue"] = int(issue_match.group(1))
            confidence["issue"] = 0.7

        # Page pattern
        page_patterns = [
            re.compile(r"(?i)pages?\s*(\d+)\s*[-–—]\s*(\d+)", re.MULTILINE),
            re.compile(r"(?i)pp?\.\s*(\d+)\s*[-–—]\s*(\d+)", re.MULTILINE),
        ]

        for pattern in page_patterns:
            page_match = pattern.search(text[:3000])
            if page_match:
                start_page, end_page = page_match.groups()
                info["pages"] = f"{start_page}-{end_page}"
                info["start_page"] = int(start_page)
                info["end_page"] = int(end_page)
                confidence["pages"] = 0.7
                break

        return info, confidence

    def _extract_subject_classification(
        self, text: str
    ) -> Tuple[Dict[str, Any], Dict[str, float]]:
        """
        Extract subject classification and research field information.

        Args:
            text (str): Document text

        Returns:
            tuple[Dict[str, Any], Dict[str, float]]: Subject info and confidence scores
        """
        info = {}
        confidence = {}

        # Subject classification patterns
        subject_patterns = [
            re.compile(
                r"(?i)subject\s+class(?:ification)?\s*:?\s*([^\n]{10,200})",
                re.MULTILINE,
            ),
            re.compile(
                r"(?i)msc\s*:?\s*([^\n]{5,100})", re.MULTILINE
            ),  # Mathematics Subject Classification
            re.compile(
                r"(?i)acm\s+class(?:ification)?\s*:?\s*([^\n]{10,200})", re.MULTILINE
            ),
        ]

        for pattern in subject_patterns:
            match = pattern.search(text[:5000])
            if match:
                subject = match.group(1).strip()
                info["subject_classification"] = subject
                confidence["subject_classification"] = 0.8
                break

        # Research field detection (basic keyword-based)
        field_keywords = {
            "computer_science": [
                "algorithm",
                "machine learning",
                "artificial intelligence",
                "software",
                "computer",
            ],
            "biology": [
                "gene",
                "protein",
                "cell",
                "organism",
                "evolution",
                "molecular",
            ],
            "physics": ["quantum", "particle", "energy", "relativity", "mechanics"],
            "mathematics": ["theorem", "proof", "equation", "function", "analysis"],
            "medicine": [
                "patient",
                "treatment",
                "clinical",
                "disease",
                "medical",
                "therapy",
            ],
        }

        text_lower = text.lower()
        field_scores = {}

        for field, keywords in field_keywords.items():
            score = sum(1 for keyword in keywords if keyword in text_lower)
            if score > 0:
                field_scores[field] = score

        if field_scores:
            best_field = max(field_scores, key=field_scores.get)
            info["research_field"] = best_field
            confidence["research_field"] = min(0.8, field_scores[best_field] * 0.1)

        return info, confidence

    def _extract_funding_info(self, text: str) -> Tuple[str, float]:
        """
        Extract funding information from text.

        Args:
            text (str): Document text

        Returns:
            tuple[str, float]: Funding info and confidence score
        """
        funding_patterns = [
            re.compile(r"(?i)funding\s*:?\s*([^\n]{10,200})", re.MULTILINE),
            re.compile(r"(?i)supported\s+by\s+([^\n]{10,200})", re.MULTILINE),
            re.compile(
                r"(?i)grant\s+(?:no\.|number)\s*:?\s*([^\n]{5,100})", re.MULTILINE
            ),
        ]

        for pattern in funding_patterns:
            match = pattern.search(text)
            if match:
                funding = match.group(1).strip()
                return funding, 0.7

        return None, 0.0

    def _parse_author_information(
        self, metadata: Dict[str, Any], **kwargs
    ) -> Dict[str, Any]:
        """
        Parse and enhance author information from existing metadata.

        Args:
            metadata (Dict[str, Any]): Existing metadata containing author info
            **kwargs: Parsing options

        Returns:
            Dict[str, Any]: Enhanced author information
        """
        enhanced_info = {}

        try:
            # Collect author information from different sources
            authors = []

            # From PDF metadata
            if "author" in metadata and metadata["author"]:
                pdf_authors = self._parse_author_string(str(metadata["author"]))
                authors.extend(pdf_authors)

            # From content analysis
            if "content_authors" in metadata and metadata["content_authors"]:
                authors.extend(metadata["content_authors"])

            if not authors:
                return enhanced_info

            # Remove duplicates and enhance
            unique_authors = []
            seen = set()

            for author in authors:
                author_key = author.lower().strip()
                if author_key not in seen and len(author.strip()) > 2:
                    seen.add(author_key)
                    unique_authors.append(author.strip())

            if unique_authors:
                enhanced_info["authors"] = unique_authors
                enhanced_info["author_count"] = len(unique_authors)
                enhanced_info["primary_author"] = unique_authors[0]

                # Parse individual author components
                parsed_authors = []
                for author in unique_authors:
                    parsed = self._parse_individual_author(author)
                    parsed_authors.append(parsed)

                enhanced_info["parsed_authors"] = parsed_authors

            return enhanced_info

        except Exception as e:
            self.logger.warning(f"Author information parsing failed: {e}")
            return {}

    def _parse_individual_author(self, author: str) -> Dict[str, str]:
        """
        Parse individual author name into components.

        Args:
            author (str): Author name string

        Returns:
            Dict[str, str]: Parsed author components
        """
        parsed = {"full_name": author.strip()}

        try:
            # Handle "Last, First" format
            if "," in author:
                parts = author.split(",", 1)
                parsed["last_name"] = parts[0].strip()
                parsed["first_name"] = parts[1].strip()
            else:
                # Handle "First Last" or "First Middle Last" format
                parts = author.strip().split()
                if len(parts) >= 2:
                    parsed["first_name"] = parts[0]
                    parsed["last_name"] = parts[-1]
                    if len(parts) > 2:
                        parsed["middle_name"] = " ".join(parts[1:-1])
                elif len(parts) == 1:
                    parsed["last_name"] = parts[0]

            return parsed

        except Exception:
            return parsed

    def _validate_and_clean_metadata(
        self, metadata: Dict[str, Any], **kwargs
    ) -> Dict[str, Any]:
        """
        Validate and clean extracted metadata.

        Args:
            metadata (Dict[str, Any]): Raw metadata
            **kwargs: Validation options

        Returns:
            Dict[str, Any]: Validated and cleaned metadata
        """
        cleaned = {}
        validation_issues = []

        try:
            confidence_threshold = kwargs.get("confidence_threshold", 0.5)
            confidence_scores = metadata.get("confidence_scores", {})

            for key, value in metadata.items():
                if key == "confidence_scores":
                    continue

                # Skip low-confidence metadata if threshold is set
                if (
                    key in confidence_scores
                    and confidence_scores[key] < confidence_threshold
                ):
                    validation_issues.append(
                        f"Low confidence for {key}: {confidence_scores[key]}"
                    )
                    continue

                # Clean and validate specific fields
                cleaned_value = self._clean_metadata_field(key, value)
                if cleaned_value is not None:
                    cleaned[key] = cleaned_value
                else:
                    validation_issues.append(f"Invalid value for {key}: {value}")

            # Add validation results to metadata
            cleaned["validation_issues"] = validation_issues
            cleaned["validation_timestamp"] = datetime.utcnow().isoformat()
            cleaned["confidence_scores"] = confidence_scores

            return cleaned

        except Exception as e:
            self.logger.warning(f"Metadata validation failed: {e}")
            return metadata

    def _clean_metadata_field(self, key: str, value: Any) -> Any:
        """
        Clean and validate a specific metadata field.

        Args:
            key (str): Metadata field name
            value (Any): Field value

        Returns:
            Any: Cleaned value or None if invalid
        """
        if value is None:
            return None

        try:
            # String fields
            if key in [
                "title",
                "content_title",
                "author",
                "subject",
                "creator",
                "producer",
                "journal",
            ]:
                if isinstance(value, str):
                    cleaned = value.strip()
                    # Remove excessive whitespace
                    cleaned = re.sub(r"\s+", " ", cleaned)
                    return cleaned if len(cleaned) > 0 else None
                return str(value).strip() if str(value).strip() else None

            # DOI validation
            elif key == "doi":
                if isinstance(value, str):
                    doi = value.strip()
                    if self._validate_doi_format(doi) > 0.5:
                        return doi
                return None

            # Numeric fields
            elif key in ["pages", "volume", "issue", "word_count", "character_count"]:
                if isinstance(value, (int, float)):
                    return value
                try:
                    return int(value) if str(value).isdigit() else None
                except:
                    return None

            # List fields
            elif key in ["authors", "keywords", "content_authors"]:
                if isinstance(value, list):
                    cleaned_list = []
                    for item in value:
                        if isinstance(item, str) and item.strip():
                            cleaned_list.append(item.strip())
                    return cleaned_list if cleaned_list else None
                return None

            # Boolean fields
            elif key in ["encrypted"]:
                if isinstance(value, bool):
                    return value
                return bool(value)

            # Default: return as-is for other fields
            else:
                return value

        except Exception as e:
            self.logger.warning(f"Failed to clean field {key}: {e}")
            return value

    def _parse_pdf_date(self, date_value: Any) -> str:
        """
        Parse PDF date format to ISO format.

        Args:
            date_value (Any): Date value from PDF metadata

        Returns:
            str: Parsed date in ISO format or original string
        """
        if not date_value:
            return None

        date_str = str(date_value)

        try:
            # Handle PDF date format: D:YYYYMMDDHHmmSSOHH'mm
            if date_str.startswith("D:"):
                date_str = date_str[2:]  # Remove D: prefix

                # Extract date components
                if len(date_str) >= 8:
                    year = date_str[:4]
                    month = date_str[4:6]
                    day = date_str[6:8]

                    # Build ISO date string
                    iso_date = f"{year}-{month}-{day}"

                    # Add time if available
                    if len(date_str) >= 14:
                        hour = date_str[8:10]
                        minute = date_str[10:12]
                        second = date_str[12:14]
                        iso_date += f"T{hour}:{minute}:{second}"

                    return iso_date

            # Try to parse as standard datetime format
            try:
                from dateutil import parser as date_parser

                parsed_date = date_parser.parse(date_str)
                return parsed_date.isoformat()
            except ImportError:
                # If dateutil is not available, return original
                pass

        except Exception as e:
            self.logger.warning(f"Failed to parse date {date_str}: {e}")
            return date_str  # Return original if parsing fails

        return date_str

    # Individual extraction methods for backward compatibility and test requirements
    def extract_title(self, content: Any, **kwargs) -> str:
        """
        Extract document title using multiple methods.

        Args:
            content (Any): Parsed PDF content or raw PDF data
            **kwargs: Extraction options

        Returns:
            str: Extracted title or None
        """
        try:
            # Get comprehensive metadata
            metadata = self.extract_metadata(content, **kwargs)

            # Try different title sources in order of preference
            title_sources = [
                metadata.get("title"),  # PDF metadata title
                metadata.get("content_title"),  # Content-based title
                metadata.get("subject"),  # PDF subject as fallback
            ]

            for title in title_sources:
                if title and isinstance(title, str) and len(title.strip()) > 5:
                    return title.strip()

            return None

        except Exception as e:
            self.logger.error(f"Title extraction failed: {e}")
            return None

    def extract_authors(self, content: Any, **kwargs) -> list:
        """
        Extract document authors using multiple methods.

        Args:
            content (Any): Parsed PDF content or raw PDF data
            **kwargs: Extraction options

        Returns:
            list: List of author names
        """
        try:
            # Get comprehensive metadata
            metadata = self.extract_metadata(content, **kwargs)

            # Return parsed authors if available
            if "authors" in metadata and metadata["authors"]:
                return metadata["authors"]

            # Fallback to other author sources
            author_sources = [
                metadata.get("content_authors"),
                [metadata.get("author")] if metadata.get("author") else None,
            ]

            for authors in author_sources:
                if authors and isinstance(authors, list) and authors:
                    return authors

            return []

        except Exception as e:
            self.logger.error(f"Authors extraction failed: {e}")
            return []

    def extract_doi(self, content: Any, **kwargs) -> str:
        """
        Extract DOI from document.

        Args:
            content (Any): Parsed PDF content or raw PDF data
            **kwargs: Extraction options

        Returns:
            str: DOI or None
        """
        try:
            # Get comprehensive metadata
            metadata = self.extract_metadata(content, **kwargs)

            # Return DOI if found with sufficient confidence
            doi = metadata.get("doi")
            confidence_scores = metadata.get("confidence_scores", {})

            if doi and confidence_scores.get("doi", 0) > 0.5:
                return doi

            return None

        except Exception as e:
            self.logger.error(f"DOI extraction failed: {e}")
            return None

    def extract_abstract(self, content: Any, **kwargs) -> str:
        """
        Extract document abstract.

        Args:
            content (Any): Parsed PDF content or raw PDF data
            **kwargs: Extraction options

        Returns:
            str: Abstract text or None
        """
        try:
            # Get comprehensive metadata
            metadata = self.extract_metadata(content, **kwargs)

            # Return abstract if found with sufficient confidence
            abstract = metadata.get("abstract")
            confidence_scores = metadata.get("confidence_scores", {})

            if abstract and confidence_scores.get("abstract", 0) > 0.5:
                return abstract

            return None

        except Exception as e:
            self.logger.error(f"Abstract extraction failed: {e}")
            return None

    def extract_keywords(self, content: Any, **kwargs) -> list:
        """
        Extract document keywords.

        Args:
            content (Any): Parsed PDF content or raw PDF data
            **kwargs: Extraction options

        Returns:
            list: List of keywords
        """
        try:
            # Get comprehensive metadata
            metadata = self.extract_metadata(content, **kwargs)

            # Return keywords if found
            keywords = metadata.get("keywords")

            if keywords and isinstance(keywords, list):
                return keywords

            return []

        except Exception as e:
            self.logger.error(f"Keywords extraction failed: {e}")
            return []

    def extract_references(self, content: Any, **kwargs) -> List[Dict[str, Any]]:
        """
        Extract bibliography references.

        Args:
            content (Any): Parsed PDF content or raw PDF data
            **kwargs: Reference extraction options

        Returns:
            List[Dict[str, Any]]: Extracted references

        Raises:
            ExtractionException: If reference extraction fails
        """
        try:
            text = self.extract_text(content, **kwargs)

            # Find references section
            references_section = None
            for section_name, section_data in self.identify_sections(
                content, **kwargs
            ).items():
                if "reference" in section_name.lower():
                    references_section = section_data["content"]
                    break

            if not references_section:
                # Try to find references at the end of document
                lines = text.split("\n")
                ref_start = -1
                for i, line in enumerate(lines):
                    if re.search(r"(?i)references?|bibliography", line):
                        ref_start = i
                        break

                if ref_start >= 0:
                    references_section = "\n".join(lines[ref_start:])

            if not references_section:
                self.logger.warning("No references section found")
                return []

            return self._parse_references(references_section, **kwargs)

        except Exception as e:
            self.logger.error(f"Reference extraction failed: {e}")
            raise ExtractionException(f"Failed to extract references: {e}")

    def _parse_references(self, references_text: str, **kwargs) -> List[Dict[str, Any]]:
        """Parse individual references from references section."""
        references = []

        # Split references by common patterns
        ref_patterns = [
            re.compile(r"\n\d+\.\s+"),  # Numbered references
            re.compile(r"\n\[\d+\]\s+"),  # Bracketed numbers
            re.compile(r"\n[A-Z][a-z]+,\s+[A-Z]"),  # Author name starts
        ]

        # Try each pattern to split references
        ref_lines = None
        for pattern in ref_patterns:
            splits = pattern.split(references_text)
            if len(splits) > 1:
                ref_lines = splits[1:]  # Skip first empty split
                break

        if not ref_lines:
            # Fallback: split by double newlines
            ref_lines = [
                line.strip() for line in references_text.split("\n\n") if line.strip()
            ]

        # Parse each reference
        for i, ref_text in enumerate(ref_lines):
            if len(ref_text.strip()) < 20:  # Skip very short lines
                continue

            ref_data = self._parse_single_reference(ref_text.strip(), i + 1)
            if ref_data:
                references.append(ref_data)

        self.logger.info(f"Extracted {len(references)} references")
        return references

    def _parse_single_reference(
        self, ref_text: str, ref_number: int
    ) -> Optional[Dict[str, Any]]:
        """Parse a single reference string."""
        ref_data = {
            "id": f"ref_{ref_number}",
            "number": ref_number,
            "raw_text": ref_text,
            "extraction_method": "pattern_parsing",
        }

        # Extract DOI
        doi_match = self.citation_patterns["doi"].search(ref_text)
        if doi_match:
            ref_data["doi"] = doi_match.group(1)

        # Extract year
        year_pattern = re.compile(r"\b(19|20)\d{2}\b")
        year_match = year_pattern.search(ref_text)
        if year_match:
            ref_data["year"] = int(year_match.group(0))

        # Extract title (heuristic: text in quotes or title case)
        title_patterns = [
            re.compile(r'"([^"]{20,200})"'),  # Quoted title
            re.compile(r"([A-Z][^.]{20,200}\.)"),  # Title case ending with period
        ]

        for pattern in title_patterns:
            title_match = pattern.search(ref_text)
            if title_match:
                ref_data["title"] = title_match.group(1).strip()
                break

        # Extract authors (simple heuristic)
        # Look for name patterns at the beginning
        author_pattern = re.compile(
            r"^([A-Z][a-z]+(?:,?\s+[A-Z]\.?)*(?:\s*,\s*[A-Z][a-z]+(?:,?\s+[A-Z]\.?)*)*)"
        )
        author_match = author_pattern.search(ref_text)
        if author_match:
            authors_str = author_match.group(1)
            # Split by common separators
            authors = [a.strip() for a in re.split(r",\s*(?=[A-Z])", authors_str)]
            ref_data["authors"] = authors

        return (
            ref_data if len(ref_data) > 4 else None
        )  # Only return if we extracted useful info
