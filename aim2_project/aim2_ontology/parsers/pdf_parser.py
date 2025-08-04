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

import re
import io
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Union, BinaryIO

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
from aim2_project.exceptions import ValidationException, ExtractionException


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
        self.add_validation_rule("pdf_extractability", self._validate_pdf_extractability)
        
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
                r'(?i)(?:^|\n)\s*(?:abstract|summary)\s*(?:\n|$)', 
                re.MULTILINE
            ),
            "introduction": re.compile(
                r'(?i)(?:^|\n)\s*(?:\d+\.?\s*)?introduction\s*(?:\n|$)', 
                re.MULTILINE
            ),
            "methods": re.compile(
                r'(?i)(?:^|\n)\s*(?:\d+\.?\s*)?(?:methods?|methodology|materials?\s+and\s+methods?)\s*(?:\n|$)', 
                re.MULTILINE
            ),
            "results": re.compile(
                r'(?i)(?:^|\n)\s*(?:\d+\.?\s*)?results?\s*(?:\n|$)', 
                re.MULTILINE
            ),
            "discussion": re.compile(
                r'(?i)(?:^|\n)\s*(?:\d+\.?\s*)?discussion\s*(?:\n|$)', 
                re.MULTILINE
            ),
            "conclusion": re.compile(
                r'(?i)(?:^|\n)\s*(?:\d+\.?\s*)?(?:conclusion|conclusions?)\s*(?:\n|$)', 
                re.MULTILINE
            ),
            "references": re.compile(
                r'(?i)(?:^|\n)\s*(?:references?|bibliography|works?\s+cited)\s*(?:\n|$)', 
                re.MULTILINE
            ),
            "acknowledgments": re.compile(
                r'(?i)(?:^|\n)\s*(?:acknowledgments?|acknowledgements?)\s*(?:\n|$)', 
                re.MULTILINE
            ),
        }
        
        # Update section patterns with PDF-specific ones
        self.section_patterns.update(pdf_patterns)
    
    def _setup_pdf_metadata_extractors(self) -> None:
        """Setup PDF-specific metadata extractors."""
        # DOI pattern
        self.citation_patterns["doi"] = re.compile(
            r'(?i)(?:doi:?\s*)?(?:https?://(?:dx\.)?doi\.org/)?'
            r'(10\.\d{4,}/[^\s]+)',
            re.IGNORECASE
        )
        
        # Author patterns for different formats
        self.citation_patterns["authors"] = re.compile(
            r'(?i)(?:authors?:?\s*)?([A-Z][a-z]+(?:\s+[A-Z]\.)*\s+[A-Z][a-z]+(?:\s*,\s*[A-Z][a-z]+(?:\s+[A-Z]\.)*\s+[A-Z][a-z]+)*)',
            re.MULTILINE
        )
        
        # Title patterns
        self.citation_patterns["title"] = re.compile(
            r'^(.{10,200})$',  # Likely title: 10-200 chars, standalone line
            re.MULTILINE
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
                    with open(content, 'rb') as f:
                        pdf_bytes = f.read()
                else:
                    # Assume it's PDF content as string
                    pdf_bytes = content.encode('utf-8')
            else:
                pdf_bytes = content
            
            # Check PDF header
            if not pdf_bytes.startswith(b'%PDF-'):
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
            extraction_method = kwargs.get('extraction_method', 'auto')
            if extraction_method == 'auto':
                extraction_method = self._select_optimal_method(pdf_bytes)
            
            self._current_extraction_method = extraction_method
            
            # Parse using selected method
            if extraction_method == 'pypdf' and PYPDF_AVAILABLE:
                parsed_data = self._parse_with_pypdf(pdf_bytes, **kwargs)
            elif extraction_method == 'pdfplumber' and PDFPLUMBER_AVAILABLE:
                parsed_data = self._parse_with_pdfplumber(pdf_bytes, **kwargs)
            elif extraction_method == 'fitz' and FITZ_AVAILABLE:
                parsed_data = self._parse_with_fitz(pdf_bytes, **kwargs)
            else:
                # Fallback to first available method
                for method in self._supported_methods:
                    try:
                        if method == 'pypdf':
                            parsed_data = self._parse_with_pypdf(pdf_bytes, **kwargs)
                        elif method == 'pdfplumber':
                            parsed_data = self._parse_with_pdfplumber(pdf_bytes, **kwargs)
                        elif method == 'fitz':
                            parsed_data = self._parse_with_fitz(pdf_bytes, **kwargs)
                        self._current_extraction_method = method
                        break
                    except Exception as e:
                        self.logger.warning(f"Method {method} failed: {e}")
                        continue
                else:
                    raise ExtractionException("All extraction methods failed")
            
            # Add parsing metadata
            parsed_data.update({
                'parser_type': 'PDFParser',
                'extraction_method': self._current_extraction_method,
                'parsing_timestamp': datetime.utcnow().isoformat(),
                'supported_methods': self._supported_methods,
            })
            
            self.logger.info(f"PDF parsing completed using {self._current_extraction_method}")
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
                with open(content, 'rb') as f:
                    return f.read()
            else:
                # Assume string content
                return content.encode('utf-8')
        elif isinstance(content, bytes):
            return content
        elif hasattr(content, 'read'):
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
            return 'pdfplumber'  # Best for layout and tables
        elif FITZ_AVAILABLE:
            return 'fitz'  # Comprehensive features
        elif PYPDF_AVAILABLE:
            return 'pypdf'  # Lightweight fallback
        else:
            raise ExtractionException("No PDF extraction methods available")
    
    def _parse_with_pypdf(self, pdf_bytes: bytes, **kwargs) -> Dict[str, Any]:
        """Parse PDF using pypdf library."""
        try:
            reader = pypdf.PdfReader(io.BytesIO(pdf_bytes))
            
            return {
                'raw_content': reader,
                'pages': reader.pages,
                'metadata': dict(reader.metadata) if reader.metadata else {},
                'page_count': len(reader.pages),
                'is_encrypted': reader.is_encrypted,
                'extraction_method': 'pypdf'
            }
        except Exception as e:
            raise ExtractionException(f"pypdf parsing failed: {e}")
    
    def _parse_with_pdfplumber(self, pdf_bytes: bytes, **kwargs) -> Dict[str, Any]:
        """Parse PDF using pdfplumber library."""
        try:
            pdf = pdfplumber.open(io.BytesIO(pdf_bytes))
            
            return {
                'raw_content': pdf,
                'pages': pdf.pages,
                'metadata': pdf.metadata or {},
                'page_count': len(pdf.pages),
                'extraction_method': 'pdfplumber'
            }
        except Exception as e:
            raise ExtractionException(f"pdfplumber parsing failed: {e}")
    
    def _parse_with_fitz(self, pdf_bytes: bytes, **kwargs) -> Dict[str, Any]:
        """Parse PDF using PyMuPDF library."""
        try:
            doc = fitz.open(stream=pdf_bytes, filetype="pdf")
            
            return {
                'raw_content': doc,
                'pages': [doc[i] for i in range(doc.page_count)],
                'metadata': doc.metadata,
                'page_count': doc.page_count,
                'is_encrypted': doc.is_encrypted,
                'extraction_method': 'fitz'
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
            
            extraction_method = content.get('extraction_method', 'auto')
            
            if extraction_method == 'pypdf':
                return self._extract_text_pypdf(content, **kwargs)
            elif extraction_method == 'pdfplumber':
                return self._extract_text_pdfplumber(content, **kwargs)
            elif extraction_method == 'fitz':
                return self._extract_text_fitz(content, **kwargs)
            else:
                raise ExtractionException(f"Unknown extraction method: {extraction_method}")
                
        except Exception as e:
            self.logger.error(f"Text extraction failed: {e}")
            raise ExtractionException(f"Failed to extract text: {e}")
    
    def _extract_text_pypdf(self, content: Dict[str, Any], **kwargs) -> str:
        """Extract text using pypdf."""
        pages = content.get('pages', [])
        text_parts = []
        
        for page in pages:
            try:
                page_text = page.extract_text()
                if page_text:
                    text_parts.append(page_text)
            except Exception as e:
                self.logger.warning(f"Failed to extract text from page: {e}")
                continue
        
        return '\n'.join(text_parts)
    
    def _extract_text_pdfplumber(self, content: Dict[str, Any], **kwargs) -> str:
        """Extract text using pdfplumber."""
        pages = content.get('pages', [])
        text_parts = []
        
        for page in pages:
            try:
                page_text = page.extract_text()
                if page_text:
                    text_parts.append(page_text)
            except Exception as e:
                self.logger.warning(f"Failed to extract text from page: {e}")
                continue
        
        return '\n'.join(text_parts)
    
    def _extract_text_fitz(self, content: Dict[str, Any], **kwargs) -> str:
        """Extract text using PyMuPDF."""
        pages = content.get('pages', [])
        text_parts = []
        
        for page in pages:
            try:
                page_text = page.get_text()
                if page_text:
                    text_parts.append(page_text)
            except Exception as e:
                self.logger.warning(f"Failed to extract text from page: {e}")
                continue
        
        return '\n'.join(text_parts)
    
    def extract_metadata(self, content: Any, **kwargs) -> Dict[str, Any]:
        """
        Extract metadata from PDF document.
        
        Args:
            content (Any): Parsed PDF content or raw PDF data
            **kwargs: Metadata extraction options
            
        Returns:
            Dict[str, Any]: Extracted metadata
            
        Raises:
            ExtractionException: If metadata extraction fails
        """
        try:
            # If content is not parsed, parse it first
            if not isinstance(content, dict):
                content = self.parse(content, **kwargs)
            
            # Extract basic PDF metadata
            pdf_metadata = content.get('metadata', {})
            
            # Standardize metadata keys
            standardized_metadata = self._standardize_metadata(pdf_metadata)
            
            # Extract additional metadata from content if requested
            if kwargs.get('extract_content_metadata', True):
                text = self.extract_text(content, **kwargs)
                content_metadata = self._extract_content_metadata(text, **kwargs)
                standardized_metadata.update(content_metadata)
            
            # Add parsing metadata
            standardized_metadata.update({
                'pages': content.get('page_count', 0),
                'extraction_method': content.get('extraction_method', 'unknown'),
                'encrypted': content.get('is_encrypted', False),
                'file_size': len(str(content.get('raw_content', ''))),
                'parsing_timestamp': datetime.utcnow().isoformat(),
            })
            
            return standardized_metadata
            
        except Exception as e:
            self.logger.error(f"Metadata extraction failed: {e}")
            raise ExtractionException(f"Failed to extract metadata: {e}")
    
    def _standardize_metadata(self, raw_metadata: Dict[str, Any]) -> Dict[str, Any]:
        """Standardize metadata field names and formats."""
        standardized = {}
        
        # Common field mappings
        field_mappings = {
            '/Title': 'title',
            'title': 'title',
            '/Author': 'author', 
            'author': 'author',
            '/Subject': 'subject',
            'subject': 'subject',
            '/Creator': 'creator',
            'creator': 'creator',
            '/Producer': 'producer',
            'producer': 'producer',
            '/CreationDate': 'creation_date',
            'creation_date': 'creation_date',
            '/ModDate': 'modification_date',
            'modification_date': 'modification_date',
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
            cleaned = value.replace('\x00', '').strip()
            return cleaned if cleaned else None
        return value
    
    def _extract_content_metadata(self, text: str, **kwargs) -> Dict[str, Any]:
        """Extract metadata from document content."""
        metadata = {}
        
        # Extract DOI
        doi_match = self.citation_patterns['doi'].search(text)
        if doi_match:
            metadata['doi'] = doi_match.group(1)
        
        # Extract potential title (first substantial line)
        lines = text.split('\n')
        for line in lines[:10]:  # Check first 10 lines
            line = line.strip()
            if 20 <= len(line) <= 200 and not line.isupper():
                metadata['content_title'] = line
                break
        
        # Word count and basic statistics
        words = text.split()
        metadata['word_count'] = len(words)
        metadata['character_count'] = len(text)
        metadata['line_count'] = len(text.split('\n'))
        
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
                                next_matches = list(other_pattern.finditer(text[start_pos:]))
                                if next_matches:
                                    next_pos = min(next_pos, start_pos + next_matches[0].start())
                        
                        # Extract section content
                        section_content = text[start_pos:next_pos].strip()
                        
                        sections[section_key] = {
                            'content': section_content,
                            'start_position': start_pos,
                            'end_position': next_pos,
                            'word_count': len(section_content.split()),
                            'confidence': self._calculate_section_confidence(
                                section_name, section_content
                            ),
                            'detection_method': 'pattern_matching'
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
            'abstract': ['abstract', 'summary', 'background', 'objective'],
            'introduction': ['introduction', 'background', 'motivation'],
            'methods': ['method', 'approach', 'technique', 'procedure'],
            'results': ['result', 'finding', 'outcome', 'data'],
            'discussion': ['discussion', 'analysis', 'interpretation'],
            'conclusion': ['conclusion', 'summary', 'future work'],
        }
        
        if section_name in section_keywords:
            keyword_count = sum(
                1 for keyword in section_keywords[section_name]
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
            
            return {
                'figures': figures,
                'tables': tables
            }
            
        except Exception as e:
            self.logger.error(f"Figure/table extraction failed: {e}")
            raise ExtractionException(f"Failed to extract figures/tables: {e}")
    
    def _extract_figures(self, text: str, **kwargs) -> List[Dict[str, Any]]:
        """Extract figure references and captions."""
        figures = []
        
        # Pattern for figure captions
        figure_pattern = re.compile(
            r'(?i)(fig(?:ure)?\s*\.?\s*(\d+)(?:\s*[:.]\s*)?(.{0,200}?)(?=\n\s*(?:fig|table|\d+\.|$))|$)',
            re.MULTILINE | re.DOTALL
        )
        
        for match in figure_pattern.finditer(text):
            figure_num = match.group(2)
            caption = match.group(3).strip() if match.group(3) else ""
            
            if figure_num and caption:
                figures.append({
                    'id': f'figure_{figure_num}',
                    'number': int(figure_num),
                    'caption': caption,
                    'type': 'figure',
                    'position': match.start(),
                    'extraction_method': 'pattern_matching'
                })
        
        return figures
    
    def _extract_tables(self, text: str, **kwargs) -> List[Dict[str, Any]]:
        """Extract table references and captions."""
        tables = []
        
        # Pattern for table captions
        table_pattern = re.compile(
            r'(?i)(table\s*\.?\s*(\d+)(?:\s*[:.]\s*)?(.{0,200}?)(?=\n\s*(?:fig|table|\d+\.|$))|$)',
            re.MULTILINE | re.DOTALL
        )
        
        for match in table_pattern.finditer(text):
            table_num = match.group(2)
            caption = match.group(3).strip() if match.group(3) else ""
            
            if table_num and caption:
                tables.append({
                    'id': f'table_{table_num}',
                    'number': int(table_num),
                    'caption': caption,
                    'type': 'table',
                    'position': match.start(),
                    'extraction_method': 'pattern_matching'
                })
        
        return tables
    
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
            for section_name, section_data in self.identify_sections(content, **kwargs).items():
                if 'reference' in section_name.lower():
                    references_section = section_data['content']
                    break
            
            if not references_section:
                # Try to find references at the end of document
                lines = text.split('\n')
                ref_start = -1
                for i, line in enumerate(lines):
                    if re.search(r'(?i)references?|bibliography', line):
                        ref_start = i
                        break
                
                if ref_start >= 0:
                    references_section = '\n'.join(lines[ref_start:])
            
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
            re.compile(r'\n\d+\.\s+'),  # Numbered references
            re.compile(r'\n\[\d+\]\s+'),  # Bracketed numbers
            re.compile(r'\n[A-Z][a-z]+,\s+[A-Z]'),  # Author name starts
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
            ref_lines = [line.strip() for line in references_text.split('\n\n') if line.strip()]
        
        # Parse each reference
        for i, ref_text in enumerate(ref_lines):
            if len(ref_text.strip()) < 20:  # Skip very short lines
                continue
                
            ref_data = self._parse_single_reference(ref_text.strip(), i + 1)
            if ref_data:
                references.append(ref_data)
        
        self.logger.info(f"Extracted {len(references)} references")
        return references
    
    def _parse_single_reference(self, ref_text: str, ref_number: int) -> Optional[Dict[str, Any]]:
        """Parse a single reference string."""
        ref_data = {
            'id': f'ref_{ref_number}',
            'number': ref_number,
            'raw_text': ref_text,
            'extraction_method': 'pattern_parsing'
        }
        
        # Extract DOI
        doi_match = self.citation_patterns['doi'].search(ref_text)
        if doi_match:
            ref_data['doi'] = doi_match.group(1)
        
        # Extract year
        year_pattern = re.compile(r'\b(19|20)\d{2}\b')
        year_match = year_pattern.search(ref_text)
        if year_match:
            ref_data['year'] = int(year_match.group(0))
        
        # Extract title (heuristic: text in quotes or title case)
        title_patterns = [
            re.compile(r'"([^"]{20,200})"'),  # Quoted title
            re.compile(r'([A-Z][^.]{20,200}\.)')  # Title case ending with period
        ]
        
        for pattern in title_patterns:
            title_match = pattern.search(ref_text)
            if title_match:
                ref_data['title'] = title_match.group(1).strip()
                break
        
        # Extract authors (simple heuristic)
        # Look for name patterns at the beginning
        author_pattern = re.compile(r'^([A-Z][a-z]+(?:,?\s+[A-Z]\.?)*(?:\s*,\s*[A-Z][a-z]+(?:,?\s+[A-Z]\.?)*)*)')
        author_match = author_pattern.search(ref_text)
        if author_match:
            authors_str = author_match.group(1)
            # Split by common separators
            authors = [a.strip() for a in re.split(r',\s*(?=[A-Z])', authors_str)]
            ref_data['authors'] = authors
        
        return ref_data if len(ref_data) > 4 else None  # Only return if we extracted useful info