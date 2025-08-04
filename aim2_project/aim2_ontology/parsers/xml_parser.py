"""
XML Parser Module for PMC (PubMed Central) Format

This module provides a comprehensive XML parser implementation for the AIM2 ontology
information extraction system, specifically designed for parsing PMC (PubMed Central)
XML format documents.

The XMLParser class inherits from AbstractDocumentParser and provides specialized
functionality for:
- PMC XML format parsing with namespace handling
- Scientific article text extraction from XML elements
- Comprehensive metadata extraction (title, authors, DOI, journal info)
- Section identification for scientific papers (abstract, introduction, methods, etc.)
- Figure and table extraction with captions
- Reference parsing and bibliography extraction
- XML validation and malformed XML handling
- Performance optimization for large XML documents

Dependencies:
    - xml.etree.ElementTree: XML parsing and manipulation
    - lxml: Advanced XML processing with namespace support (optional)
    - re: Regular expression pattern matching
    - typing: Type hints
    - logging: Logging functionality
    - datetime: Date/time handling

Usage:
    from aim2_project.aim2_ontology.parsers.xml_parser import XMLParser

    parser = XMLParser()
    result = parser.parse_file("pmc_article.xml")
    text = parser.extract_text(result)
    metadata = parser.extract_metadata(result)
    sections = parser.identify_sections(result)
"""

import logging
import re
import xml.etree.ElementTree as ET
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
from xml.sax.saxutils import unescape

# Try to import lxml for advanced XML processing
try:
    from lxml import etree as lxml_etree

    LXML_AVAILABLE = True
except ImportError:
    LXML_AVAILABLE = False

from aim2_project.aim2_ontology.parsers import AbstractDocumentParser
from aim2_project.aim2_utils.config_manager import ConfigManager
from aim2_project.exceptions import ExtractionException, ValidationException


class XMLParser(AbstractDocumentParser):
    """
    Comprehensive XML parser implementation for PMC (PubMed Central) format documents.

    This concrete implementation of AbstractDocumentParser provides specialized
    functionality for parsing XML documents with focus on PMC format scientific papers,
    academic publications, and biomedical literature.

    Key Features:
        - PMC XML format parsing with comprehensive namespace support
        - Intelligent XML structure detection and validation
        - Scientific paper section identification and extraction
        - Metadata extraction from XML elements and attributes
        - Figure and table extraction with caption parsing
        - Reference and citation extraction from bibliography
        - XML namespace handling and malformed XML recovery
        - Performance optimization for large XML documents

    Supported XML Types:
        - PMC (PubMed Central) XML format
        - JATS (Journal Article Tag Suite) XML
        - NLM (National Library of Medicine) XML formats
        - Generic scientific article XML with standard elements
        - Malformed XML with recovery mechanisms

    PMC-Specific Elements Supported:
        - article-meta: Article metadata (title, authors, DOI)
        - abstract: Article abstract with structured sections
        - body: Main article content with sections
        - back: Bibliography and appendices
        - fig: Figures with captions and labels
        - table-wrap: Tables with captions and content
        - ref-list: Reference list with structured citations
        - contrib-group: Author information with affiliations
        - journal-meta: Journal publication information
    """

    # PMC XML namespace mappings
    PMC_NAMESPACES = {
        "article": "http://dtd.nlm.nih.gov/xhtml",
        "xlink": "http://www.w3.org/1999/xlink",
        "mml": "http://www.w3.org/1998/Math/MathML",
        "xml": "http://www.w3.org/XML/1998/namespace",
    }

    # PMC-specific element mappings for text extraction
    PMC_TEXT_ELEMENTS = [
        "p",  # Paragraphs
        "sec",  # Sections
        "title",  # Titles
        "abstract",  # Abstract
        "body",  # Body content
        "td",  # Table cells
        "th",  # Table headers
        "caption",  # Captions
        "label",  # Labels
        "list-item",  # List items
        "def",  # Definitions
    ]

    # PMC section type mappings
    PMC_SECTION_TYPES = {
        "abstract": ["abstract", "summary"],
        "introduction": ["intro", "introduction", "background"],
        "methods": ["methods", "materials", "methodology", "experimental"],
        "results": ["results", "findings", "outcomes"],
        "discussion": ["discussion", "analysis", "interpretation"],
        "conclusion": ["conclusion", "conclusions", "summary"],
        "acknowledgments": ["ack", "acknowledgments", "acknowledgements"],
        "references": ["ref-list", "references", "bibliography"],
        "appendix": ["app", "appendix", "supplementary"],
    }

    def __init__(
        self,
        parser_name: str = "XMLParser",
        config_manager: Optional[ConfigManager] = None,
        logger: Optional[logging.Logger] = None,
        options: Optional[Dict[str, Any]] = None,
    ):
        """
        Initialize the XML parser with configuration and options.

        Args:
            parser_name (str): Name identifier for this parser instance
            config_manager (ConfigManager, optional): Configuration manager instance
            logger (logging.Logger, optional): Logger instance
            options (Dict[str, Any], optional): Initial parser options

        Raises:
            ValidationException: If XML processing libraries are not available
            ValueError: If parser configuration is invalid
        """
        # Initialize base document parser
        super().__init__(parser_name, config_manager, logger, options)

        # Check XML library availability
        self._check_xml_library_availability()

        # Initialize XML-specific attributes
        self._current_xml_document = None
        self._current_root_element = None
        self._xml_namespaces = {}
        self._supported_encodings = ["utf-8", "utf-16", "iso-8859-1", "cp1252"]

        # Configure XML-specific section patterns
        self._setup_xml_section_patterns()

        # Configure XML metadata extractors
        self._setup_xml_metadata_extractors()

        # Add XML-specific validation rules
        self.add_validation_rule("xml_format", self._validate_xml_format)
        self.add_validation_rule("pmc_structure", self._validate_pmc_structure)

        self.logger.info(f"XML parser initialized with lxml support: {LXML_AVAILABLE}")

    def _check_xml_library_availability(self) -> None:
        """
        Check availability of XML processing libraries and log status.

        Raises:
            ValidationException: If no XML libraries are available
        """
        if not LXML_AVAILABLE:
            self.logger.warning(
                "lxml library not available. Using standard library xml.etree.ElementTree. "
                "Install lxml for enhanced XML processing capabilities."
            )
        else:
            self.logger.debug("lxml library available for advanced XML processing")

    def _setup_xml_section_patterns(self) -> None:
        """Setup XML-specific section identification patterns."""
        # Add PMC XML section patterns based on element attributes and content
        xml_patterns = {
            # PMC section identification by sec-type attribute
            "abstract": re.compile(
                r'<(?:abstract|sec[^>]*sec-type=["\'](?:abstract|summary)["\'][^>]*)>',
                re.IGNORECASE,
            ),
            "introduction": re.compile(
                r'<sec[^>]*sec-type=["\'](?:intro|introduction|background)["\'][^>]*>',
                re.IGNORECASE,
            ),
            "methods": re.compile(
                r'<sec[^>]*sec-type=["\'](?:methods|materials|methodology)["\'][^>]*>',
                re.IGNORECASE,
            ),
            "results": re.compile(
                r'<sec[^>]*sec-type=["\'](?:results|findings)["\'][^>]*>', re.IGNORECASE
            ),
            "discussion": re.compile(
                r'<sec[^>]*sec-type=["\'](?:discussion|analysis)["\'][^>]*>',
                re.IGNORECASE,
            ),
            "conclusion": re.compile(
                r'<sec[^>]*sec-type=["\'](?:conclusion|conclusions)["\'][^>]*>',
                re.IGNORECASE,
            ),
        }

        # Update section patterns with XML-specific ones
        self.section_patterns.update(xml_patterns)

    def _setup_xml_metadata_extractors(self) -> None:
        """Setup XML-specific metadata extractors."""
        # PMC-specific DOI patterns in XML
        self.citation_patterns["xml_doi"] = re.compile(
            r'<article-id[^>]*pub-id-type=["\']doi["\'][^>]*>([^<]+)</article-id>',
            re.IGNORECASE,
        )

        # PMC author patterns in XML
        self.citation_patterns["xml_authors"] = re.compile(
            r'<contrib[^>]*contrib-type=["\']author["\'][^>]*>.*?</contrib>',
            re.IGNORECASE | re.DOTALL,
        )

        # PMC title patterns in XML
        self.citation_patterns["xml_title"] = re.compile(
            r"<article-title[^>]*>([^<]+)</article-title>", re.IGNORECASE
        )

        # PMC journal patterns in XML
        self.citation_patterns["xml_journal"] = re.compile(
            r"<journal-title[^>]*>([^<]+)</journal-title>", re.IGNORECASE
        )

    def get_supported_formats(self) -> List[str]:
        """
        Get list of supported XML formats and extensions.

        Returns:
            List[str]: List of supported format extensions
        """
        return ["xml", "nxml"]

    def validate(self, content: Union[str, bytes], **kwargs) -> bool:
        """
        Validate XML content format and structure.

        Args:
            content (Union[str, bytes]): XML content to validate
            **kwargs: Additional validation options

        Returns:
            bool: True if content is valid XML, False otherwise

        Raises:
            ValidationException: If validation fails with errors
        """
        try:
            # Convert content to string if needed
            xml_content = self._prepare_xml_content(content)

            # Basic XML well-formedness check
            if not self._is_well_formed_xml(xml_content):
                self.logger.warning("XML content is not well-formed")
                return False

            # Try to parse with standard library
            try:
                ET.fromstring(xml_content)
                parsing_success = True
            except ET.ParseError as e:
                self.logger.debug(f"Standard library XML parsing failed: {e}")
                parsing_success = False

            # Try lxml if standard library fails
            if not parsing_success and LXML_AVAILABLE:
                try:
                    parser = lxml_etree.XMLParser(recover=True)
                    lxml_etree.fromstring(xml_content.encode("utf-8"), parser)
                    parsing_success = True
                except Exception as e:
                    self.logger.debug(f"lxml parsing failed: {e}")

            return parsing_success

        except Exception as e:
            self.logger.error(f"XML validation error: {e}")
            raise ValidationException(f"XML validation failed: {e}")

    def _is_well_formed_xml(self, xml_content: str) -> bool:
        """
        Check if XML content is well-formed using basic heuristics.

        Args:
            xml_content (str): XML content to check

        Returns:
            bool: True if appears to be well-formed XML
        """
        # Check for XML declaration or root element
        xml_content_stripped = xml_content.strip()

        if not xml_content_stripped:
            return False

        # Check for XML-like structure
        has_xml_declaration = xml_content_stripped.startswith("<?xml")
        has_root_element = "<" in xml_content_stripped and ">" in xml_content_stripped

        if not (has_xml_declaration or has_root_element):
            return False

        # Basic tag matching check (simplified)
        open_tags = re.findall(r"<([^/!?][^>]*)>", xml_content)
        close_tags = re.findall(r"</([^>]+)>", xml_content)

        # Extract tag names (ignore attributes)
        open_tag_names = [tag.split()[0] for tag in open_tags if not tag.endswith("/")]
        close_tag_names = close_tags

        # Check if we have reasonable tag structure
        return len(open_tag_names) > 0 and len(close_tag_names) >= 0

    def _validate_xml_format(self, content: Any, **kwargs) -> bool:
        """Validate XML format-specific requirements."""
        return self.validate(content, **kwargs)

    def _validate_pmc_structure(self, content: Any, **kwargs) -> bool:
        """Validate PMC-specific XML structure."""
        try:
            xml_content = self._prepare_xml_content(content)

            # Check for PMC-specific elements
            pmc_indicators = [
                "<article",  # PMC articles start with <article>
                "article-meta",  # Contains article metadata
                "journal-meta",  # Journal information
            ]

            return any(indicator in xml_content for indicator in pmc_indicators)
        except Exception:
            return False

    def parse(self, content: Union[str, bytes], **kwargs) -> Dict[str, Any]:
        """
        Parse XML content and return structured data.

        Args:
            content (Union[str, bytes]): XML content or file path
            **kwargs: Additional parsing options

        Returns:
            Dict[str, Any]: Parsed XML data structure

        Raises:
            ExtractionException: If parsing fails
            ValidationException: If content is not valid XML
        """
        try:
            self.logger.info("Starting XML parsing")

            # Validate content first
            if not self.validate(content, **kwargs):
                raise ValidationException("Invalid XML content")

            # Prepare content for processing
            xml_content = self._prepare_xml_content(content)

            # Determine parsing method (lxml vs standard library)
            use_lxml = kwargs.get("use_lxml", LXML_AVAILABLE)

            if use_lxml and LXML_AVAILABLE:
                parsed_data = self._parse_with_lxml(xml_content, **kwargs)
            else:
                parsed_data = self._parse_with_etree(xml_content, **kwargs)

            # Add parsing metadata
            parsed_data.update(
                {
                    "parser_type": "XMLParser",
                    "parsing_method": "lxml"
                    if use_lxml and LXML_AVAILABLE
                    else "etree",
                    "parsing_timestamp": datetime.utcnow().isoformat(),
                    "xml_encoding": self._detect_xml_encoding(xml_content),
                    "has_namespaces": bool(self._xml_namespaces),
                }
            )

            self.logger.info("XML parsing completed successfully")
            return parsed_data

        except Exception as e:
            self.logger.error(f"XML parsing failed: {e}")
            raise ExtractionException(f"Failed to parse XML: {e}")

    def _prepare_xml_content(self, content: Union[str, bytes]) -> str:
        """
        Prepare XML content for parsing.

        Args:
            content: Input content in various formats

        Returns:
            str: XML content as string

        Raises:
            ValueError: If content is invalid or path is not allowed
            SecurityError: If path traversal attack is detected
        """
        if isinstance(content, str):
            # Check if it's likely a file path (short string with no XML indicators)
            if (
                len(content) < 260
                and not content.strip().startswith("<")
                and Path(content).exists()
            ):
                # Validate and sanitize the file path for security
                validated_path = self._validate_file_path(content)
                with open(validated_path, "r", encoding="utf-8") as f:
                    return f.read()
            else:
                # Assume it's XML content
                return content
        elif isinstance(content, bytes):
            # Try to decode with detected encoding
            encoding = self._detect_encoding(content)
            return content.decode(encoding)
        else:
            raise ValueError(f"Unsupported content type: {type(content)}")

    def _validate_file_path(self, file_path: str) -> Path:
        """
        Validate and sanitize file path to prevent path traversal attacks.

        Args:
            file_path (str): File path to validate

        Returns:
            Path: Validated path object

        Raises:
            ValueError: If path is invalid or not allowed
        """
        try:
            # Convert to Path object and resolve any symbolic links
            path = Path(file_path).resolve()

            # Check for obvious path traversal attempts in the original string
            if ".." in file_path:
                raise ValueError(f"Path traversal detected: {file_path}")

            # Allow absolute paths but check they're in reasonable locations
            # Block access to sensitive system directories
            sensitive_dirs = {
                "/etc",
                "/usr/bin",
                "/usr/sbin",
                "/bin",
                "/sbin",
                "/boot",
                "/root",
            }
            path_str = str(path)

            for sensitive_dir in sensitive_dirs:
                if path_str.startswith(sensitive_dir):
                    raise ValueError(
                        f"Access to sensitive directory blocked: {file_path}"
                    )

            # Check file extension (only allow common document formats)
            allowed_extensions = {".xml", ".nxml", ".txt", ".json"}
            if path.suffix.lower() not in allowed_extensions:
                raise ValueError(f"File extension not allowed: {path.suffix}")

            # Check if file exists and is readable
            if not path.exists():
                raise ValueError(f"File does not exist: {file_path}")

            if not path.is_file():
                raise ValueError(f"Path is not a file: {file_path}")

            # Additional size check to prevent loading extremely large files
            if path.stat().st_size > 100 * 1024 * 1024:  # 100MB limit
                raise ValueError(f"File too large: {path.stat().st_size} bytes")

            return path

        except Exception as e:
            self.logger.error(f"Path validation failed for {file_path}: {e}")
            raise ValueError(f"Invalid file path: {file_path}")

    def _detect_encoding(self, content_bytes: bytes) -> str:
        """
        Detect encoding of XML content.

        Args:
            content_bytes (bytes): XML content as bytes

        Returns:
            str: Detected encoding name
        """
        # Check for BOM
        if content_bytes.startswith(b"\xef\xbb\xbf"):
            return "utf-8-sig"
        elif content_bytes.startswith(b"\xff\xfe"):
            return "utf-16-le"
        elif content_bytes.startswith(b"\xfe\xff"):
            return "utf-16-be"

        # Try to extract encoding from XML declaration
        xml_decl_match = re.match(
            rb'<\?xml[^>]*encoding=["\']([^"\']+)["\']', content_bytes[:200]
        )
        if xml_decl_match:
            declared_encoding = xml_decl_match.group(1).decode("ascii")
            if declared_encoding.lower() in [
                enc.lower() for enc in self._supported_encodings
            ]:
                return declared_encoding

        # Default fallback
        return "utf-8"

    def _detect_xml_encoding(self, xml_content: str) -> str:
        """
        Detect XML encoding from content.

        Args:
            xml_content (str): XML content as string

        Returns:
            str: Detected encoding name
        """
        # Extract encoding from XML declaration
        encoding_match = re.search(
            r'<\?xml[^>]*encoding=["\']([^"\']+)["\']', xml_content[:200], re.IGNORECASE
        )

        if encoding_match:
            return encoding_match.group(1)

        return "utf-8"

    def _parse_with_etree(self, xml_content: str, **kwargs) -> Dict[str, Any]:
        """Parse XML using standard library ElementTree."""
        try:
            # For ElementTree, XXE protection is limited but we can disable some features
            # Note: ElementTree is generally safer than lxml by default for XXE attacks
            try:
                # Try to create a secure parser (Python 3.8+)
                parser = ET.XMLParser()
                # For older Python versions, this may not work, so we'll fallback
                if hasattr(parser, "parser"):
                    parser.parser.DefaultHandler = lambda data: None
                    parser.parser.ExternalEntityRefHandler = (
                        lambda context, base, uri, notationName: False
                    )
            except (AttributeError, TypeError):
                # Fallback to default parser (still relatively safe)
                parser = None

            # Parse XML content
            if parser:
                root = ET.fromstring(xml_content, parser)
            else:
                root = ET.fromstring(xml_content)

            # Extract namespaces
            self._xml_namespaces = self._extract_namespaces_etree(root)

            # Store current document references
            self._current_root_element = root

            return {
                "raw_content": xml_content,
                "root_element": root,
                "namespaces": self._xml_namespaces,
                "parsing_method": "etree",
                "element_count": len(list(root.iter())),
            }
        except ET.ParseError as e:
            raise ExtractionException(f"ElementTree parsing failed: {e}")

    def _parse_with_lxml(self, xml_content: str, **kwargs) -> Dict[str, Any]:
        """Parse XML using lxml library."""
        try:
            # Create secure parser with recovery for malformed XML and XXE protection
            parser = lxml_etree.XMLParser(
                recover=kwargs.get("recover_malformed", True),
                remove_blank_text=kwargs.get("remove_blank_text", False),
                huge_tree=kwargs.get("huge_tree", False),
                # XXE attack prevention
                resolve_entities=False,
                no_network=True,
                dtd_validation=False,
                load_dtd=False,
            )

            # Parse XML content
            root = lxml_etree.fromstring(xml_content.encode("utf-8"), parser)

            # Extract namespaces
            self._xml_namespaces = root.nsmap or {}

            # Store current document references
            self._current_root_element = root

            return {
                "raw_content": xml_content,
                "root_element": root,
                "namespaces": self._xml_namespaces,
                "parsing_method": "lxml",
                "element_count": len(list(root.iter())),
                "parser_log": parser.error_log if hasattr(parser, "error_log") else [],
            }
        except Exception as e:
            raise ExtractionException(f"lxml parsing failed: {e}")

    def _extract_namespaces_etree(self, root: ET.Element) -> Dict[str, str]:
        """
        Extract namespaces from ElementTree root element.

        Args:
            root (ET.Element): Root XML element

        Returns:
            Dict[str, str]: Namespace mappings
        """
        namespaces = {}

        # Extract from root element attributes
        for key, value in root.attrib.items():
            if key.startswith("xmlns"):
                if key == "xmlns":
                    namespaces[""] = value  # Default namespace
                else:
                    prefix = key.split(":", 1)[1]
                    namespaces[prefix] = value

        # Extract from all elements (comprehensive scan)
        for elem in root.iter():
            for key, value in elem.attrib.items():
                if key.startswith("xmlns"):
                    if key == "xmlns":
                        namespaces[""] = value
                    else:
                        prefix = key.split(":", 1)[1]
                        namespaces[prefix] = value

        return namespaces

    def _get_lxml_text(self, element) -> str:
        """Helper method to safely get text content from lxml element."""
        if element is None:
            return ""
        try:
            return lxml_etree.tostring(
                element, method="text", encoding="unicode"
            ).strip()
        except Exception:
            return ""

    def extract_text(self, content: Any, **kwargs) -> str:
        """
        Extract plain text from XML content.

        Args:
            content (Any): Parsed XML content or raw XML data
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

            root_element = content.get("root_element")
            if root_element is None:
                raise ExtractionException("No root element found in parsed content")

            # Extract text using appropriate method
            parsing_method = content.get("parsing_method", "etree")

            if parsing_method == "lxml" and LXML_AVAILABLE:
                return self._extract_text_lxml(root_element, **kwargs)
            else:
                return self._extract_text_etree(root_element, **kwargs)

        except Exception as e:
            self.logger.error(f"Text extraction failed: {e}")
            raise ExtractionException(f"Failed to extract text: {e}")

    def _extract_text_etree(self, root_element: ET.Element, **kwargs) -> str:
        """Extract text using ElementTree."""
        try:
            text_parts = []

            # Get extraction options
            include_attributes = kwargs.get("include_attributes", False)
            preserve_structure = kwargs.get("preserve_structure", True)
            skip_elements = set(kwargs.get("skip_elements", ["script", "style"]))

            # Extract text from all relevant elements
            for element in root_element.iter():
                # Skip unwanted elements
                if element.tag in skip_elements:
                    continue

                # Extract element text
                if element.text:
                    text_parts.append(element.text.strip())

                # Include tail text (text after the element)
                if element.tail:
                    text_parts.append(element.tail.strip())

                # Include attribute values if requested
                if include_attributes:
                    for attr_value in element.attrib.values():
                        if isinstance(attr_value, str) and len(attr_value.strip()) > 0:
                            text_parts.append(attr_value.strip())

            # Join text parts
            if preserve_structure:
                # Try to preserve some structure with double newlines between sections
                text = self._reconstruct_text_structure_etree(root_element, **kwargs)
            else:
                text = " ".join(part for part in text_parts if part)

            return self._clean_extracted_text(text)

        except Exception as e:
            raise ExtractionException(f"ElementTree text extraction failed: {e}")

    def _extract_text_lxml(self, root_element, **kwargs) -> str:
        """Extract text using lxml."""
        try:
            # Get extraction options
            preserve_structure = kwargs.get("preserve_structure", True)

            if preserve_structure:
                # Use lxml's lxml_etree.tostring(element, method='text', encoding='unicode') for structured text
                text = self._reconstruct_text_structure_lxml(root_element, **kwargs)
            else:
                # Simple text extraction using lxml etree
                text = self._get_lxml_text(root_element)

            return self._clean_extracted_text(text)

        except Exception as e:
            raise ExtractionException(f"lxml text extraction failed: {e}")

    def _reconstruct_text_structure_etree(
        self, root_element: ET.Element, **kwargs
    ) -> str:
        """Reconstruct text with structural information using ElementTree."""
        text_parts = []

        # Define elements that should have line breaks
        block_elements = {
            "p",
            "div",
            "sec",
            "abstract",
            "title",
            "caption",
            "h1",
            "h2",
            "h3",
            "h4",
            "h5",
            "h6",
        }

        # Define elements that should be separated by double line breaks
        section_elements = {"sec", "abstract", "body", "back", "front"}

        def extract_element_text(element, depth=0):
            # Add section spacing for major elements
            if element.tag in section_elements and depth > 0:
                text_parts.append("\n\n")

            # Add element text
            if element.text:
                text_parts.append(element.text.strip())

            # Process child elements
            for child in element:
                extract_element_text(child, depth + 1)

                # Add line breaks for block elements
                if child.tag in block_elements:
                    text_parts.append("\n")

            # Add element tail text
            if element.tail:
                text_parts.append(element.tail.strip())

        # Start extraction from root
        extract_element_text(root_element)

        # Join and clean up
        text = "".join(text_parts)
        return re.sub(r"\n{3,}", "\n\n", text)  # Normalize excessive line breaks

    def _reconstruct_text_structure_lxml(self, root_element, **kwargs) -> str:
        """Reconstruct text with structural information using lxml."""
        try:
            text_parts = []

            # Process different section types
            sections = [
                ("abstract", self._extract_abstract_lxml),
                ("body", self._extract_body_lxml),
                ("back", self._extract_back_lxml),
            ]

            for section_name, extractor in sections:
                section_elements = root_element.xpath(f".//{section_name}")
                for element in section_elements:
                    section_text = extractor(element)
                    if section_text:
                        text_parts.append(section_text)
                        text_parts.append("\n\n")

            # If no structured sections found, fall back to simple extraction
            if not text_parts:
                return self._get_lxml_text(root_element)

            return "".join(text_parts).strip()

        except Exception as e:
            self.logger.warning(
                f"Structured text extraction failed, using simple method: {e}"
            )
            return lxml_etree.tostring(root_element, method="text", encoding="unicode")

    def _extract_abstract_lxml(self, abstract_element) -> str:
        """Extract abstract text from lxml element."""
        return self._get_lxml_text(abstract_element)

    def _extract_body_lxml(self, body_element) -> str:
        """Extract body text from lxml element."""
        return self._get_lxml_text(body_element)

    def _extract_back_lxml(self, back_element) -> str:
        """Extract back matter text from lxml element."""
        return self._get_lxml_text(back_element)

    def _clean_extracted_text(self, text: str) -> str:
        """
        Clean and normalize extracted text.

        Args:
            text (str): Raw extracted text

        Returns:
            str: Cleaned text
        """
        if not text:
            return ""

        # Remove excessive whitespace
        text = re.sub(r"\s+", " ", text)

        # Normalize line breaks
        text = re.sub(r"\n\s*\n", "\n\n", text)

        # Remove leading/trailing whitespace
        text = text.strip()

        # Unescape XML entities
        text = unescape(text)

        return text

    def extract_metadata(self, content: Any, **kwargs) -> Dict[str, Any]:
        """
        Extract comprehensive metadata from XML document.

        Args:
            content (Any): Parsed XML content or raw XML data
            **kwargs: Metadata extraction options

        Returns:
            Dict[str, Any]: Comprehensive extracted metadata

        Raises:
            ExtractionException: If metadata extraction fails
        """
        try:
            # If content is not parsed, parse it first
            if not isinstance(content, dict):
                content = self.parse(content, **kwargs)

            root_element = content.get("root_element")
            if root_element is None:
                raise ExtractionException("No root element found in parsed content")

            # Initialize metadata dictionary
            metadata = {}

            # Extract parsing metadata
            metadata.update(
                {
                    "parsing_method": content.get("parsing_method", "unknown"),
                    "element_count": content.get("element_count", 0),
                    "has_namespaces": content.get("has_namespaces", False),
                    "xml_encoding": content.get("xml_encoding", "utf-8"),
                    "extraction_timestamp": datetime.utcnow().isoformat(),
                }
            )

            # Extract PMC-specific metadata
            parsing_method = content.get("parsing_method", "etree")

            if parsing_method == "lxml" and LXML_AVAILABLE:
                pmc_metadata = self._extract_pmc_metadata_lxml(root_element, **kwargs)
            else:
                pmc_metadata = self._extract_pmc_metadata_etree(root_element, **kwargs)

            metadata.update(pmc_metadata)

            # Extract additional content-based metadata
            content_metadata = self._extract_content_metadata_xml(
                root_element, **kwargs
            )
            metadata.update(content_metadata)

            self.logger.info(f"Extracted metadata with {len(metadata)} fields")
            return metadata

        except Exception as e:
            self.logger.error(f"Metadata extraction failed: {e}")
            raise ExtractionException(f"Failed to extract metadata: {e}")

    def _extract_pmc_metadata_etree(
        self, root_element: ET.Element, **kwargs
    ) -> Dict[str, Any]:
        """Extract PMC-specific metadata using ElementTree."""
        metadata = {}

        try:
            # Extract article title
            title_elements = root_element.findall(".//article-title")
            if title_elements:
                metadata["title"] = (
                    title_elements[0].text.strip() if title_elements[0].text else ""
                )

            # Extract DOI
            doi_elements = root_element.findall('.//article-id[@pub-id-type="doi"]')
            if doi_elements:
                metadata["doi"] = (
                    doi_elements[0].text.strip() if doi_elements[0].text else ""
                )

            # Extract authors
            authors = self._extract_authors_etree(root_element)
            if authors:
                metadata["authors"] = authors

            # Extract journal information
            journal_info = self._extract_journal_info_etree(root_element)
            metadata.update(journal_info)

            # Extract publication dates
            pub_dates = self._extract_publication_dates_etree(root_element)
            metadata.update(pub_dates)

            # Extract abstract
            abstract = self._extract_abstract_metadata_etree(root_element)
            if abstract:
                metadata["abstract"] = abstract

            # Extract keywords
            keywords = self._extract_keywords_etree(root_element)
            if keywords:
                metadata["keywords"] = keywords

        except Exception as e:
            self.logger.warning(f"PMC metadata extraction failed: {e}")

        return metadata

    def _extract_pmc_metadata_lxml(self, root_element, **kwargs) -> Dict[str, Any]:
        """Extract PMC-specific metadata using lxml."""
        metadata = {}

        try:
            # Extract article title
            title_elements = root_element.xpath(".//article-title")
            if title_elements:
                metadata["title"] = self._get_lxml_text(title_elements[0])

            # Extract DOI
            doi_elements = root_element.xpath('.//article-id[@pub-id-type="doi"]')
            if doi_elements:
                metadata["doi"] = self._get_lxml_text(doi_elements[0])

            # Extract authors
            authors = self._extract_authors_lxml(root_element)
            if authors:
                metadata["authors"] = authors

            # Extract journal information
            journal_info = self._extract_journal_info_lxml(root_element)
            metadata.update(journal_info)

            # Extract publication dates
            pub_dates = self._extract_publication_dates_lxml(root_element)
            metadata.update(pub_dates)

            # Extract abstract
            abstract = self._extract_abstract_metadata_lxml(root_element)
            if abstract:
                metadata["abstract"] = abstract

            # Extract keywords
            keywords = self._extract_keywords_lxml(root_element)
            if keywords:
                metadata["keywords"] = keywords

        except Exception as e:
            self.logger.warning(f"PMC metadata extraction failed: {e}")

        return metadata

    def _extract_authors_etree(self, root_element: ET.Element) -> List[str]:
        """Extract authors using ElementTree."""
        authors = []

        # Find contributor groups
        contrib_groups = root_element.findall(".//contrib-group")

        for group in contrib_groups:
            # Find author contributors
            author_contribs = group.findall('.//contrib[@contrib-type="author"]')

            for contrib in author_contribs:
                author_name = self._extract_author_name_etree(contrib)
                if author_name:
                    authors.append(author_name)

        return authors

    def _extract_authors_lxml(self, root_element) -> List[str]:
        """Extract authors using lxml."""
        authors = []

        # Find author contributors
        author_contribs = root_element.xpath('.//contrib[@contrib-type="author"]')

        for contrib in author_contribs:
            author_name = self._extract_author_name_lxml(contrib)
            if author_name:
                authors.append(author_name)

        return authors

    def _extract_author_name_etree(self, contrib_element: ET.Element) -> Optional[str]:
        """Extract author name from contributor element using ElementTree."""
        name_parts = []

        # Look for name element
        name_elem = contrib_element.find(".//name")
        if name_elem:
            # Extract given names
            given_names = name_elem.find("given-names")
            if given_names is not None and given_names.text:
                name_parts.append(given_names.text.strip())

            # Extract surname
            surname = name_elem.find("surname")
            if surname is not None and surname.text:
                name_parts.append(surname.text.strip())

        return " ".join(name_parts) if name_parts else None

    def _extract_author_name_lxml(self, contrib_element) -> Optional[str]:
        """Extract author name from contributor element using lxml."""
        name_parts = []

        # Extract given names
        given_names = contrib_element.xpath(".//given-names")
        if given_names:
            name_parts.append(self._get_lxml_text(given_names[0]))

        # Extract surname
        surname = contrib_element.xpath(".//surname")
        if surname:
            name_parts.append(self._get_lxml_text(surname[0]))

        return " ".join(name_parts) if name_parts else None

    def _extract_journal_info_etree(self, root_element: ET.Element) -> Dict[str, Any]:
        """Extract journal information using ElementTree."""
        journal_info = {}

        # Extract journal title
        journal_title = root_element.find(".//journal-title")
        if journal_title is not None and journal_title.text:
            journal_info["journal"] = journal_title.text.strip()

        # Extract ISSN
        issn_elements = root_element.findall(".//issn")
        for issn in issn_elements:
            pub_type = issn.get("pub-type", "unknown")
            if issn.text:
                journal_info[f"issn_{pub_type}"] = issn.text.strip()

        # Extract volume
        volume = root_element.find(".//volume")
        if volume is not None and volume.text:
            journal_info["volume"] = volume.text.strip()

        # Extract issue
        issue = root_element.find(".//issue")
        if issue is not None and issue.text:
            journal_info["issue"] = issue.text.strip()

        return journal_info

    def _extract_journal_info_lxml(self, root_element) -> Dict[str, Any]:
        """Extract journal information using lxml."""
        journal_info = {}

        # Extract journal title
        journal_title = root_element.xpath(".//journal-title")
        if journal_title:
            journal_info["journal"] = self._get_lxml_text(journal_title[0])

        # Extract ISSN
        issn_elements = root_element.xpath(".//issn")
        for issn in issn_elements:
            pub_type = issn.get("pub-type", "unknown")
            journal_info[f"issn_{pub_type}"] = self._get_lxml_text(issn)

        # Extract volume
        volume = root_element.xpath(".//volume")
        if volume:
            journal_info["volume"] = self._get_lxml_text(volume[0])

        # Extract issue
        issue = root_element.xpath(".//issue")
        if issue:
            journal_info["issue"] = self._get_lxml_text(issue[0])

        return journal_info

    def _extract_publication_dates_etree(
        self, root_element: ET.Element
    ) -> Dict[str, Any]:
        """Extract publication dates using ElementTree."""
        date_info = {}

        # Find publication dates
        pub_dates = root_element.findall(".//pub-date")

        for pub_date in pub_dates:
            pub_type = pub_date.get("pub-type", "unknown")

            # Extract date components
            year = pub_date.find("year")
            month = pub_date.find("month")
            day = pub_date.find("day")

            date_parts = []
            if year is not None and year.text:
                date_parts.append(year.text.strip())
            if month is not None and month.text:
                date_parts.append(month.text.strip().zfill(2))
            if day is not None and day.text:
                date_parts.append(day.text.strip().zfill(2))

            if date_parts:
                date_info[f"publication_date_{pub_type}"] = "-".join(date_parts)

        return date_info

    def _extract_publication_dates_lxml(self, root_element) -> Dict[str, Any]:
        """Extract publication dates using lxml."""
        date_info = {}

        # Find publication dates
        pub_dates = root_element.xpath(".//pub-date")

        for pub_date in pub_dates:
            pub_type = pub_date.get("pub-type", "unknown")

            # Extract date components
            year = pub_date.xpath("./year")
            month = pub_date.xpath("./month")
            day = pub_date.xpath("./day")

            date_parts = []
            if year:
                date_parts.append(self._get_lxml_text(year[0]))
            if month:
                date_parts.append(self._get_lxml_text(month[0]).zfill(2))
            if day:
                date_parts.append(self._get_lxml_text(day[0]).zfill(2))

            if date_parts:
                date_info[f"publication_date_{pub_type}"] = "-".join(date_parts)

        return date_info

    def _extract_abstract_metadata_etree(
        self, root_element: ET.Element
    ) -> Optional[str]:
        """Extract abstract text using ElementTree."""
        abstract_elements = root_element.findall(".//abstract")

        if abstract_elements:
            # Get the first abstract
            abstract_elem = abstract_elements[0]
            text_parts = []

            # Extract text from all child elements
            for elem in abstract_elem.iter():
                if elem.text:
                    text_parts.append(elem.text.strip())
                if elem.tail:
                    text_parts.append(elem.tail.strip())

            abstract_text = " ".join(part for part in text_parts if part)
            return self._clean_extracted_text(abstract_text) if abstract_text else None

        return None

    def _extract_abstract_metadata_lxml(self, root_element) -> Optional[str]:
        """Extract abstract text using lxml."""
        abstract_elements = root_element.xpath(".//abstract")

        if abstract_elements:
            abstract_text = self._get_lxml_text(abstract_elements[0])
            return self._clean_extracted_text(abstract_text) if abstract_text else None

        return None

    def _extract_keywords_etree(self, root_element: ET.Element) -> List[str]:
        """Extract keywords using ElementTree."""
        keywords = []

        # Find keyword groups
        kwd_groups = root_element.findall(".//kwd-group")

        for group in kwd_groups:
            # Find individual keywords
            kwd_elements = group.findall(".//kwd")

            for kwd in kwd_elements:
                if kwd.text:
                    keywords.append(kwd.text.strip())

        return keywords

    def _extract_keywords_lxml(self, root_element) -> List[str]:
        """Extract keywords using lxml."""
        keywords = []

        # Find individual keywords
        kwd_elements = root_element.xpath(".//kwd")

        for kwd in kwd_elements:
            keyword_text = self._get_lxml_text(kwd)
            if keyword_text:
                keywords.append(keyword_text)

        return keywords

    def _extract_content_metadata_xml(self, root_element, **kwargs) -> Dict[str, Any]:
        """Extract additional content-based metadata from XML."""
        metadata = {}

        try:
            # Count different element types
            element_counts = {}
            if hasattr(root_element, "iter"):  # ElementTree
                for elem in root_element.iter():
                    tag = elem.tag
                    element_counts[tag] = element_counts.get(tag, 0) + 1
            else:  # lxml
                for elem in root_element.iter():
                    tag = elem.tag
                    element_counts[tag] = element_counts.get(tag, 0) + 1

            metadata["element_counts"] = element_counts

            # Document structure analysis
            has_abstract = "abstract" in element_counts
            has_body = "body" in element_counts
            has_references = any(tag in element_counts for tag in ["ref-list", "ref"])
            has_figures = any(tag in element_counts for tag in ["fig", "figure"])
            has_tables = any(tag in element_counts for tag in ["table-wrap", "table"])

            metadata["document_structure"] = {
                "has_abstract": has_abstract,
                "has_body": has_body,
                "has_references": has_references,
                "has_figures": has_figures,
                "has_tables": has_tables,
            }

        except Exception as e:
            self.logger.warning(f"Content metadata extraction failed: {e}")

        return metadata

    def identify_sections(self, content: Any, **kwargs) -> Dict[str, Dict[str, Any]]:
        """
        Identify document sections (abstract, introduction, methods, etc.) from XML.

        Args:
            content (Any): Parsed XML content or raw XML data
            **kwargs: Section identification options

        Returns:
            Dict[str, Dict[str, Any]]: Identified sections with metadata

        Raises:
            ExtractionException: If section identification fails
        """
        try:
            # If content is not parsed, parse it first
            if not isinstance(content, dict):
                content = self.parse(content, **kwargs)

            root_element = content.get("root_element")
            if root_element is None:
                raise ExtractionException("No root element found in parsed content")

            # Extract sections using appropriate method
            parsing_method = content.get("parsing_method", "etree")

            if parsing_method == "lxml" and LXML_AVAILABLE:
                sections = self._identify_sections_lxml(root_element, **kwargs)
            else:
                sections = self._identify_sections_etree(root_element, **kwargs)

            self.logger.info(f"Identified {len(sections)} sections")
            return sections

        except Exception as e:
            self.logger.error(f"Section identification failed: {e}")
            raise ExtractionException(f"Failed to identify sections: {e}")

    def _identify_sections_etree(
        self, root_element: ET.Element, **kwargs
    ) -> Dict[str, Dict[str, Any]]:
        """Identify sections using ElementTree."""
        sections = {}

        try:
            # Find all section elements
            sec_elements = root_element.findall(".//sec")

            for i, sec_elem in enumerate(sec_elements):
                section_data = self._analyze_section_etree(sec_elem, i)
                if section_data:
                    section_id = section_data.get("id", f"section_{i}")
                    sections[section_id] = section_data

            # Find abstract as a special section
            abstract_elements = root_element.findall(".//abstract")
            for i, abstract_elem in enumerate(abstract_elements):
                abstract_data = self._analyze_abstract_section_etree(abstract_elem, i)
                if abstract_data:
                    abstract_id = f"abstract_{i}" if i > 0 else "abstract"
                    sections[abstract_id] = abstract_data

            # Find references section
            ref_list_elements = root_element.findall(".//ref-list")
            for i, ref_elem in enumerate(ref_list_elements):
                ref_data = self._analyze_references_section_etree(ref_elem, i)
                if ref_data:
                    ref_id = f"references_{i}" if i > 0 else "references"
                    sections[ref_id] = ref_data

        except Exception as e:
            self.logger.warning(f"Section identification with ElementTree failed: {e}")

        return sections

    def _identify_sections_lxml(
        self, root_element, **kwargs
    ) -> Dict[str, Dict[str, Any]]:
        """Identify sections using lxml."""
        sections = {}

        try:
            # Find all section elements
            sec_elements = root_element.xpath(".//sec")

            for i, sec_elem in enumerate(sec_elements):
                section_data = self._analyze_section_lxml(sec_elem, i)
                if section_data:
                    section_id = section_data.get("id", f"section_{i}")
                    sections[section_id] = section_data

            # Find abstract as a special section
            abstract_elements = root_element.xpath(".//abstract")
            for i, abstract_elem in enumerate(abstract_elements):
                abstract_data = self._analyze_abstract_section_lxml(abstract_elem, i)
                if abstract_data:
                    abstract_id = f"abstract_{i}" if i > 0 else "abstract"
                    sections[abstract_id] = abstract_data

            # Find references section
            ref_list_elements = root_element.xpath(".//ref-list")
            for i, ref_elem in enumerate(ref_list_elements):
                ref_data = self._analyze_references_section_lxml(ref_elem, i)
                if ref_data:
                    ref_id = f"references_{i}" if i > 0 else "references"
                    sections[ref_id] = ref_data

        except Exception as e:
            self.logger.warning(f"Section identification with lxml failed: {e}")

        return sections

    def _analyze_section_etree(
        self, sec_elem: ET.Element, index: int
    ) -> Optional[Dict[str, Any]]:
        """Analyze a section element using ElementTree."""
        try:
            # Extract section type from sec-type attribute
            sec_type = sec_elem.get("sec-type", "unknown")

            # Extract section ID from id attribute
            sec_id = sec_elem.get("id", f"section_{index}")

            # Extract section title
            title_elem = sec_elem.find("./title")
            title = (
                title_elem.text.strip()
                if title_elem is not None and title_elem.text
                else ""
            )

            # Extract section content
            content_parts = []
            for elem in sec_elem.iter():
                if elem.tag not in ["title"] and elem.text:
                    content_parts.append(elem.text.strip())
                if elem.tail:
                    content_parts.append(elem.tail.strip())

            content = " ".join(part for part in content_parts if part)
            content = self._clean_extracted_text(content)

            # Determine section category
            section_category = self._categorize_section(sec_type, title)

            return {
                "id": sec_id,
                "type": sec_type,
                "category": section_category,
                "title": title,
                "content": content,
                "word_count": len(content.split()) if content else 0,
                "element_count": len(list(sec_elem.iter())),
                "detection_method": "xml_structure",
                "confidence": self._calculate_section_confidence_xml(
                    sec_type, title, content
                ),
            }

        except Exception as e:
            self.logger.warning(f"Section analysis failed for section {index}: {e}")
            return None

    def _analyze_section_lxml(self, sec_elem, index: int) -> Optional[Dict[str, Any]]:
        """Analyze a section element using lxml."""
        try:
            # Extract section type from sec-type attribute
            sec_type = sec_elem.get("sec-type", "unknown")

            # Extract section ID from id attribute
            sec_id = sec_elem.get("id", f"section_{index}")

            # Extract section title
            title_elems = sec_elem.xpath("./title")
            title = self._get_lxml_text(title_elems[0]) if title_elems else ""

            # Extract section content
            content = self._get_lxml_text(sec_elem)

            # Remove title from content if present
            if title and content.startswith(title):
                content = content[len(title) :].strip()

            content = self._clean_extracted_text(content)

            # Determine section category
            section_category = self._categorize_section(sec_type, title)

            return {
                "id": sec_id,
                "type": sec_type,
                "category": section_category,
                "title": title,
                "content": content,
                "word_count": len(content.split()) if content else 0,
                "element_count": len(list(sec_elem.iter())),
                "detection_method": "xml_structure",
                "confidence": self._calculate_section_confidence_xml(
                    sec_type, title, content
                ),
            }

        except Exception as e:
            self.logger.warning(f"Section analysis failed for section {index}: {e}")
            return None

    def _analyze_abstract_section_etree(
        self, abstract_elem: ET.Element, index: int
    ) -> Optional[Dict[str, Any]]:
        """Analyze an abstract element using ElementTree."""
        try:
            # Extract abstract content
            content_parts = []
            for elem in abstract_elem.iter():
                if elem.text:
                    content_parts.append(elem.text.strip())
                if elem.tail:
                    content_parts.append(elem.tail.strip())

            content = " ".join(part for part in content_parts if part)
            content = self._clean_extracted_text(content)

            return {
                "id": f"abstract_{index}" if index > 0 else "abstract",
                "type": "abstract",
                "category": "abstract",
                "title": "Abstract",
                "content": content,
                "word_count": len(content.split()) if content else 0,
                "element_count": len(list(abstract_elem.iter())),
                "detection_method": "xml_structure",
                "confidence": 1.0,  # High confidence for explicit abstract elements
            }

        except Exception as e:
            self.logger.warning(f"Abstract analysis failed: {e}")
            return None

    def _analyze_abstract_section_lxml(
        self, abstract_elem, index: int
    ) -> Optional[Dict[str, Any]]:
        """Analyze an abstract element using lxml."""
        try:
            # Extract abstract content
            content = self._get_lxml_text(abstract_elem)
            content = self._clean_extracted_text(content)

            return {
                "id": f"abstract_{index}" if index > 0 else "abstract",
                "type": "abstract",
                "category": "abstract",
                "title": "Abstract",
                "content": content,
                "word_count": len(content.split()) if content else 0,
                "element_count": len(list(abstract_elem.iter())),
                "detection_method": "xml_structure",
                "confidence": 1.0,  # High confidence for explicit abstract elements
            }

        except Exception as e:
            self.logger.warning(f"Abstract analysis failed: {e}")
            return None

    def _analyze_references_section_etree(
        self, ref_elem: ET.Element, index: int
    ) -> Optional[Dict[str, Any]]:
        """Analyze a references section element using ElementTree."""
        try:
            # Extract title if present
            title_elem = ref_elem.find("./title")
            title = (
                title_elem.text.strip()
                if title_elem is not None and title_elem.text
                else "References"
            )

            # Count references
            ref_elements = ref_elem.findall(".//ref")
            ref_count = len(ref_elements)

            # Extract content (simplified for references)
            content = f"References section containing {ref_count} references"

            return {
                "id": f"references_{index}" if index > 0 else "references",
                "type": "ref-list",
                "category": "references",
                "title": title,
                "content": content,
                "word_count": ref_count * 10,  # Rough estimate
                "element_count": len(list(ref_elem.iter())),
                "reference_count": ref_count,
                "detection_method": "xml_structure",
                "confidence": 1.0,  # High confidence for explicit reference elements
            }

        except Exception as e:
            self.logger.warning(f"References analysis failed: {e}")
            return None

    def _analyze_references_section_lxml(
        self, ref_elem, index: int
    ) -> Optional[Dict[str, Any]]:
        """Analyze a references section element using lxml."""
        try:
            # Extract title if present
            title_elems = ref_elem.xpath("./title")
            title = self._get_lxml_text(title_elems[0]) if title_elems else "References"

            # Count references
            ref_elements = ref_elem.xpath(".//ref")
            ref_count = len(ref_elements)

            # Extract content (simplified for references)
            content = f"References section containing {ref_count} references"

            return {
                "id": f"references_{index}" if index > 0 else "references",
                "type": "ref-list",
                "category": "references",
                "title": title,
                "content": content,
                "word_count": ref_count * 10,  # Rough estimate
                "element_count": len(list(ref_elem.iter())),
                "reference_count": ref_count,
                "detection_method": "xml_structure",
                "confidence": 1.0,  # High confidence for explicit reference elements
            }

        except Exception as e:
            self.logger.warning(f"References analysis failed: {e}")
            return None

    def _categorize_section(self, sec_type: str, title: str) -> str:
        """
        Categorize a section based on type and title.

        Args:
            sec_type (str): Section type attribute
            title (str): Section title

        Returns:
            str: Section category
        """
        # Check sec-type first
        if sec_type and sec_type != "unknown":
            sec_type_lower = sec_type.lower()
            for category, types in self.PMC_SECTION_TYPES.items():
                if any(type_name in sec_type_lower for type_name in types):
                    return category

        # Check title if sec-type is not informative
        if title:
            title_lower = title.lower()
            for category, types in self.PMC_SECTION_TYPES.items():
                if any(type_name in title_lower for type_name in types):
                    return category

        # Default category
        return "other"

    def _calculate_section_confidence_xml(
        self, sec_type: str, title: str, content: str
    ) -> float:
        """
        Calculate confidence score for XML section identification.

        Args:
            sec_type (str): Section type attribute
            title (str): Section title
            content (str): Section content

        Returns:
            float: Confidence score (0.0 to 1.0)
        """
        confidence = 0.5  # Base confidence

        # Boost confidence for explicit sec-type attributes
        if sec_type and sec_type != "unknown":
            confidence += 0.3

        # Boost confidence for clear titles
        if title and len(title.strip()) > 0:
            confidence += 0.2

        # Reduce confidence for very short content
        if content:
            word_count = len(content.split())
            if word_count < 10:
                confidence -= 0.2
            elif word_count > 50:
                confidence += 0.1

        return min(1.0, max(0.0, confidence))

    def extract_figures_tables(
        self, content: Any, **kwargs
    ) -> Dict[str, List[Dict[str, Any]]]:
        """
        Extract figures and tables with captions from XML.

        Args:
            content (Any): Parsed XML content or raw XML data
            **kwargs: Extraction options

        Returns:
            Dict[str, List[Dict[str, Any]]]: Extracted figures and tables

        Raises:
            ExtractionException: If extraction fails
        """
        try:
            # If content is not parsed, parse it first
            if not isinstance(content, dict):
                content = self.parse(content, **kwargs)

            root_element = content.get("root_element")
            if root_element is None:
                raise ExtractionException("No root element found in parsed content")

            # Extract figures and tables using appropriate method
            parsing_method = content.get("parsing_method", "etree")

            if parsing_method == "lxml" and LXML_AVAILABLE:
                figures = self._extract_figures_lxml(root_element, **kwargs)
                tables = self._extract_tables_lxml(root_element, **kwargs)
            else:
                figures = self._extract_figures_etree(root_element, **kwargs)
                tables = self._extract_tables_etree(root_element, **kwargs)

            result = {"figures": figures, "tables": tables}

            self.logger.info(
                f"Extracted {len(figures)} figures and {len(tables)} tables"
            )
            return result

        except Exception as e:
            self.logger.error(f"Figure/table extraction failed: {e}")
            raise ExtractionException(f"Failed to extract figures/tables: {e}")

    def _extract_figures_etree(
        self, root_element: ET.Element, **kwargs
    ) -> List[Dict[str, Any]]:
        """Extract figures using ElementTree."""
        figures = []

        try:
            # Find figure elements
            fig_elements = root_element.findall(".//fig")

            for i, fig_elem in enumerate(fig_elements):
                figure_data = self._analyze_figure_etree(fig_elem, i)
                if figure_data:
                    figures.append(figure_data)

        except Exception as e:
            self.logger.warning(f"Figure extraction with ElementTree failed: {e}")

        return figures

    def _extract_figures_lxml(self, root_element, **kwargs) -> List[Dict[str, Any]]:
        """Extract figures using lxml."""
        figures = []

        try:
            # Find figure elements
            fig_elements = root_element.xpath(".//fig")

            for i, fig_elem in enumerate(fig_elements):
                figure_data = self._analyze_figure_lxml(fig_elem, i)
                if figure_data:
                    figures.append(figure_data)

        except Exception as e:
            self.logger.warning(f"Figure extraction with lxml failed: {e}")

        return figures

    def _analyze_figure_etree(
        self, fig_elem: ET.Element, index: int
    ) -> Optional[Dict[str, Any]]:
        """Analyze a figure element using ElementTree."""
        try:
            # Extract figure ID
            fig_id = fig_elem.get("id", f"figure_{index + 1}")

            # Extract label
            label_elem = fig_elem.find("./label")
            label = (
                label_elem.text.strip()
                if label_elem is not None and label_elem.text
                else f"Figure {index + 1}"
            )

            # Extract caption
            caption_elem = fig_elem.find(".//caption")
            caption = ""
            if caption_elem is not None:
                caption_parts = []
                for elem in caption_elem.iter():
                    if elem.text:
                        caption_parts.append(elem.text.strip())
                    if elem.tail:
                        caption_parts.append(elem.tail.strip())
                caption = " ".join(part for part in caption_parts if part)
                caption = self._clean_extracted_text(caption)

            # Extract graphic information
            graphic_elements = fig_elem.findall(".//graphic")
            graphics = []
            for graphic in graphic_elements:
                href = graphic.get("{http://www.w3.org/1999/xlink}href", "")
                if not href:
                    href = graphic.get("href", "")
                if href:
                    graphics.append(href)

            return {
                "id": fig_id,
                "label": label,
                "caption": caption,
                "graphics": graphics,
                "type": "figure",
                "extraction_method": "xml_structure",
                "index": index + 1,
            }

        except Exception as e:
            self.logger.warning(f"Figure analysis failed for figure {index}: {e}")
            return None

    def _analyze_figure_lxml(self, fig_elem, index: int) -> Optional[Dict[str, Any]]:
        """Analyze a figure element using lxml."""
        try:
            # Extract figure ID
            fig_id = fig_elem.get("id", f"figure_{index + 1}")

            # Extract label
            label_elems = fig_elem.xpath("./label")
            label = (
                self._get_lxml_text(label_elems[0])
                if label_elems
                else f"Figure {index + 1}"
            )

            # Extract caption
            caption_elems = fig_elem.xpath(".//caption")
            caption = ""
            if caption_elems:
                caption = self._get_lxml_text(caption_elems[0])
                caption = self._clean_extracted_text(caption)

            # Extract graphic information
            graphic_elements = fig_elem.xpath(".//graphic")
            graphics = []
            for graphic in graphic_elements:
                href = graphic.get("{http://www.w3.org/1999/xlink}href", "")
                if not href:
                    href = graphic.get("href", "")
                if href:
                    graphics.append(href)

            return {
                "id": fig_id,
                "label": label,
                "caption": caption,
                "graphics": graphics,
                "type": "figure",
                "extraction_method": "xml_structure",
                "index": index + 1,
            }

        except Exception as e:
            self.logger.warning(f"Figure analysis failed for figure {index}: {e}")
            return None

    def _extract_tables_etree(
        self, root_element: ET.Element, **kwargs
    ) -> List[Dict[str, Any]]:
        """Extract tables using ElementTree."""
        tables = []

        try:
            # Find table-wrap elements (PMC table containers)
            table_elements = root_element.findall(".//table-wrap")

            for i, table_elem in enumerate(table_elements):
                table_data = self._analyze_table_etree(table_elem, i)
                if table_data:
                    tables.append(table_data)

            # Also look for standalone table elements
            standalone_tables = root_element.findall(".//table")
            for i, table_elem in enumerate(standalone_tables):
                # Skip if already processed as part of table-wrap
                # For ElementTree, we need to check if this table is inside a table-wrap
                is_in_table_wrap = False
                # Find all table-wrap elements and check if this table is a child
                table_wraps = root_element.findall(".//table-wrap")
                for wrap in table_wraps:
                    if table_elem in wrap.findall(".//table"):
                        is_in_table_wrap = True
                        break

                if is_in_table_wrap:
                    continue

                table_data = self._analyze_standalone_table_etree(
                    table_elem, len(tables) + i
                )
                if table_data:
                    tables.append(table_data)

        except Exception as e:
            self.logger.warning(f"Table extraction with ElementTree failed: {e}")

        return tables

    def _extract_tables_lxml(self, root_element, **kwargs) -> List[Dict[str, Any]]:
        """Extract tables using lxml."""
        tables = []

        try:
            # Find table-wrap elements (PMC table containers)
            table_elements = root_element.xpath(".//table-wrap")

            for i, table_elem in enumerate(table_elements):
                table_data = self._analyze_table_lxml(table_elem, i)
                if table_data:
                    tables.append(table_data)

            # Also look for standalone table elements
            standalone_tables = root_element.xpath(
                ".//table[not(ancestor::table-wrap)]"
            )
            for i, table_elem in enumerate(standalone_tables):
                table_data = self._analyze_standalone_table_lxml(
                    table_elem, len(tables) + i
                )
                if table_data:
                    tables.append(table_data)

        except Exception as e:
            self.logger.warning(f"Table extraction with lxml failed: {e}")

        return tables

    def _analyze_table_etree(
        self, table_elem: ET.Element, index: int
    ) -> Optional[Dict[str, Any]]:
        """Analyze a table-wrap element using ElementTree."""
        try:
            # Extract table ID
            table_id = table_elem.get("id", f"table_{index + 1}")

            # Extract label
            label_elem = table_elem.find("./label")
            label = (
                label_elem.text.strip()
                if label_elem is not None and label_elem.text
                else f"Table {index + 1}"
            )

            # Extract caption
            caption_elem = table_elem.find(".//caption")
            caption = ""
            if caption_elem is not None:
                caption_parts = []
                for elem in caption_elem.iter():
                    if elem.text:
                        caption_parts.append(elem.text.strip())
                    if elem.tail:
                        caption_parts.append(elem.tail.strip())
                caption = " ".join(part for part in caption_parts if part)
                caption = self._clean_extracted_text(caption)

            # Extract table structure info
            table_inner = table_elem.find(".//table")
            rows = 0
            cols = 0
            if table_inner is not None:
                tr_elements = table_inner.findall(".//tr")
                rows = len(tr_elements)
                if tr_elements:
                    # Count columns from first row
                    first_row = tr_elements[0]
                    td_elements = first_row.findall(".//td") + first_row.findall(
                        ".//th"
                    )
                    cols = len(td_elements)

            return {
                "id": table_id,
                "label": label,
                "caption": caption,
                "rows": rows,
                "columns": cols,
                "type": "table",
                "extraction_method": "xml_structure",
                "index": index + 1,
            }

        except Exception as e:
            self.logger.warning(f"Table analysis failed for table {index}: {e}")
            return None

    def _analyze_table_lxml(self, table_elem, index: int) -> Optional[Dict[str, Any]]:
        """Analyze a table-wrap element using lxml."""
        try:
            # Extract table ID
            table_id = table_elem.get("id", f"table_{index + 1}")

            # Extract label
            label_elems = table_elem.xpath("./label")
            label = (
                self._get_lxml_text(label_elems[0])
                if label_elems
                else f"Table {index + 1}"
            )

            # Extract caption
            caption_elems = table_elem.xpath(".//caption")
            caption = ""
            if caption_elems:
                caption = self._get_lxml_text(caption_elems[0])
                caption = self._clean_extracted_text(caption)

            # Extract table structure info
            table_inner = table_elem.xpath(".//table")
            rows = 0
            cols = 0
            if table_inner:
                tr_elements = table_inner[0].xpath(".//tr")
                rows = len(tr_elements)
                if tr_elements:
                    # Count columns from first row
                    td_elements = tr_elements[0].xpath(".//td | .//th")
                    cols = len(td_elements)

            return {
                "id": table_id,
                "label": label,
                "caption": caption,
                "rows": rows,
                "columns": cols,
                "type": "table",
                "extraction_method": "xml_structure",
                "index": index + 1,
            }

        except Exception as e:
            self.logger.warning(f"Table analysis failed for table {index}: {e}")
            return None

    def _analyze_standalone_table_etree(
        self, table_elem: ET.Element, index: int
    ) -> Optional[Dict[str, Any]]:
        """Analyze a standalone table element using ElementTree."""
        try:
            # Extract table structure info
            tr_elements = table_elem.findall(".//tr")
            rows = len(tr_elements)
            cols = 0
            if tr_elements:
                # Count columns from first row
                first_row = tr_elements[0]
                td_elements = first_row.findall(".//td") + first_row.findall(".//th")
                cols = len(td_elements)

            return {
                "id": f"table_{index + 1}",
                "label": f"Table {index + 1}",
                "caption": "",
                "rows": rows,
                "columns": cols,
                "type": "table",
                "extraction_method": "xml_structure",
                "index": index + 1,
            }

        except Exception as e:
            self.logger.warning(
                f"Standalone table analysis failed for table {index}: {e}"
            )
            return None

    def _analyze_standalone_table_lxml(
        self, table_elem, index: int
    ) -> Optional[Dict[str, Any]]:
        """Analyze a standalone table element using lxml."""
        try:
            # Extract table structure info
            tr_elements = table_elem.xpath(".//tr")
            rows = len(tr_elements)
            cols = 0
            if tr_elements:
                # Count columns from first row
                td_elements = tr_elements[0].xpath(".//td | .//th")
                cols = len(td_elements)

            return {
                "id": f"table_{index + 1}",
                "label": f"Table {index + 1}",
                "caption": "",
                "rows": rows,
                "columns": cols,
                "type": "table",
                "extraction_method": "xml_structure",
                "index": index + 1,
            }

        except Exception as e:
            self.logger.warning(
                f"Standalone table analysis failed for table {index}: {e}"
            )
            return None

    def extract_references(self, content: Any, **kwargs) -> List[Dict[str, Any]]:
        """
        Extract bibliography references from XML.

        Args:
            content (Any): Parsed XML content or raw XML data
            **kwargs: Reference extraction options

        Returns:
            List[Dict[str, Any]]: Extracted references

        Raises:
            ExtractionException: If reference extraction fails
        """
        try:
            # If content is not parsed, parse it first
            if not isinstance(content, dict):
                content = self.parse(content, **kwargs)

            root_element = content.get("root_element")
            if root_element is None:
                raise ExtractionException("No root element found in parsed content")

            # Extract references using appropriate method
            parsing_method = content.get("parsing_method", "etree")

            if parsing_method == "lxml" and LXML_AVAILABLE:
                references = self._extract_references_lxml(root_element, **kwargs)
            else:
                references = self._extract_references_etree(root_element, **kwargs)

            self.logger.info(f"Extracted {len(references)} references")
            return references

        except Exception as e:
            self.logger.error(f"Reference extraction failed: {e}")
            raise ExtractionException(f"Failed to extract references: {e}")

    def _extract_references_etree(
        self, root_element: ET.Element, **kwargs
    ) -> List[Dict[str, Any]]:
        """Extract references using ElementTree."""
        references = []

        try:
            # Find reference list elements
            ref_list_elements = root_element.findall(".//ref-list")

            for ref_list in ref_list_elements:
                # Find individual reference elements
                ref_elements = ref_list.findall(".//ref")

                for i, ref_elem in enumerate(ref_elements):
                    ref_data = self._analyze_reference_etree(
                        ref_elem, len(references) + i + 1
                    )
                    if ref_data:
                        references.append(ref_data)

            # Also look for standalone reference elements
            if not references:
                standalone_refs = root_element.findall(".//ref")
                for i, ref_elem in enumerate(standalone_refs):
                    ref_data = self._analyze_reference_etree(ref_elem, i + 1)
                    if ref_data:
                        references.append(ref_data)

        except Exception as e:
            self.logger.warning(f"Reference extraction with ElementTree failed: {e}")

        return references

    def _extract_references_lxml(self, root_element, **kwargs) -> List[Dict[str, Any]]:
        """Extract references using lxml."""
        references = []

        try:
            # Find reference list elements
            ref_list_elements = root_element.xpath(".//ref-list")

            for ref_list in ref_list_elements:
                # Find individual reference elements
                ref_elements = ref_list.xpath(".//ref")

                for i, ref_elem in enumerate(ref_elements):
                    ref_data = self._analyze_reference_lxml(
                        ref_elem, len(references) + i + 1
                    )
                    if ref_data:
                        references.append(ref_data)

            # Also look for standalone reference elements
            if not references:
                standalone_refs = root_element.xpath(".//ref")
                for i, ref_elem in enumerate(standalone_refs):
                    ref_data = self._analyze_reference_lxml(ref_elem, i + 1)
                    if ref_data:
                        references.append(ref_data)

        except Exception as e:
            self.logger.warning(f"Reference extraction with lxml failed: {e}")

        return references

    def _analyze_reference_etree(
        self, ref_elem: ET.Element, index: int
    ) -> Optional[Dict[str, Any]]:
        """Analyze a reference element using ElementTree."""
        try:
            # Extract reference ID
            ref_id = ref_elem.get("id", f"ref_{index}")

            # Extract label (reference number)
            label_elem = ref_elem.find("./label")
            label = (
                label_elem.text.strip()
                if label_elem is not None and label_elem.text
                else str(index)
            )

            # Initialize reference data
            ref_data = {
                "id": ref_id,
                "label": label,
                "index": index,
                "extraction_method": "xml_structure",
            }

            # Look for element-citation or mixed-citation
            citation_elem = ref_elem.find(".//element-citation")
            if citation_elem is None:
                citation_elem = ref_elem.find(".//mixed-citation")

            if citation_elem is not None:
                # Extract structured citation data
                structured_data = self._extract_structured_citation_etree(citation_elem)
                ref_data.update(structured_data)
            else:
                # Fall back to text extraction
                text_parts = []
                for elem in ref_elem.iter():
                    if elem.tag not in ["label"] and elem.text:
                        text_parts.append(elem.text.strip())
                    if elem.tail:
                        text_parts.append(elem.tail.strip())

                raw_text = " ".join(part for part in text_parts if part)
                ref_data["raw_text"] = self._clean_extracted_text(raw_text)

                # Try to extract basic information from raw text
                parsed_data = self._parse_unstructured_reference(raw_text)
                ref_data.update(parsed_data)

            return ref_data

        except Exception as e:
            self.logger.warning(f"Reference analysis failed for reference {index}: {e}")
            return None

    def _analyze_reference_lxml(self, ref_elem, index: int) -> Optional[Dict[str, Any]]:
        """Analyze a reference element using lxml."""
        try:
            # Extract reference ID
            ref_id = ref_elem.get("id", f"ref_{index}")

            # Extract label (reference number)
            label_elems = ref_elem.xpath("./label")
            label = self._get_lxml_text(label_elems[0]) if label_elems else str(index)

            # Initialize reference data
            ref_data = {
                "id": ref_id,
                "label": label,
                "index": index,
                "extraction_method": "xml_structure",
            }

            # Look for element-citation or mixed-citation
            citation_elems = ref_elem.xpath(".//element-citation | .//mixed-citation")

            if citation_elems:
                # Extract structured citation data
                structured_data = self._extract_structured_citation_lxml(
                    citation_elems[0]
                )
                ref_data.update(structured_data)
            else:
                # Fall back to text extraction
                raw_text = self._get_lxml_text(ref_elem)

                # Remove label from text if present
                if label and raw_text.startswith(label):
                    raw_text = raw_text[len(label) :].strip()
                    if raw_text.startswith("."):
                        raw_text = raw_text[1:].strip()

                ref_data["raw_text"] = self._clean_extracted_text(raw_text)

                # Try to extract basic information from raw text
                parsed_data = self._parse_unstructured_reference(raw_text)
                ref_data.update(parsed_data)

            return ref_data

        except Exception as e:
            self.logger.warning(f"Reference analysis failed for reference {index}: {e}")
            return None

    def _extract_structured_citation_etree(
        self, citation_elem: ET.Element
    ) -> Dict[str, Any]:
        """Extract structured citation data using ElementTree."""
        citation_data = {}

        try:
            # Extract publication type
            pub_type = citation_elem.get("publication-type", "unknown")
            citation_data["publication_type"] = pub_type

            # Extract article title
            title_elem = citation_elem.find(".//article-title")
            if title_elem is not None and title_elem.text:
                citation_data["title"] = title_elem.text.strip()

            # Extract source (journal/book title)
            source_elem = citation_elem.find(".//source")
            if source_elem is not None and source_elem.text:
                citation_data["source"] = source_elem.text.strip()

            # Extract authors
            authors = self._extract_citation_authors_etree(citation_elem)
            if authors:
                citation_data["authors"] = authors

            # Extract year
            year_elem = citation_elem.find(".//year")
            if year_elem is not None and year_elem.text:
                citation_data["year"] = year_elem.text.strip()

            # Extract volume
            volume_elem = citation_elem.find(".//volume")
            if volume_elem is not None and volume_elem.text:
                citation_data["volume"] = volume_elem.text.strip()

            # Extract issue
            issue_elem = citation_elem.find(".//issue")
            if issue_elem is not None and issue_elem.text:
                citation_data["issue"] = issue_elem.text.strip()

            # Extract page range
            fpage_elem = citation_elem.find(".//fpage")
            lpage_elem = citation_elem.find(".//lpage")
            if fpage_elem is not None and fpage_elem.text:
                fpage = fpage_elem.text.strip()
                citation_data["first_page"] = fpage

                if lpage_elem is not None and lpage_elem.text:
                    lpage = lpage_elem.text.strip()
                    citation_data["last_page"] = lpage
                    citation_data["pages"] = f"{fpage}-{lpage}"
                else:
                    citation_data["pages"] = fpage

            # Extract DOI
            doi_elems = citation_elem.findall('.//pub-id[@pub-id-type="doi"]')
            if doi_elems:
                citation_data["doi"] = doi_elems[0].text.strip()

            # Extract PMID
            pmid_elems = citation_elem.findall('.//pub-id[@pub-id-type="pmid"]')
            if pmid_elems:
                citation_data["pmid"] = pmid_elems[0].text.strip()

        except Exception as e:
            self.logger.warning(f"Structured citation extraction failed: {e}")

        return citation_data

    def _extract_structured_citation_lxml(self, citation_elem) -> Dict[str, Any]:
        """Extract structured citation data using lxml."""
        citation_data = {}

        try:
            # Extract publication type
            pub_type = citation_elem.get("publication-type", "unknown")
            citation_data["publication_type"] = pub_type

            # Extract article title
            title_elems = citation_elem.xpath(".//article-title")
            if title_elems:
                citation_data["title"] = self._get_lxml_text(title_elems[0])

            # Extract source (journal/book title)
            source_elems = citation_elem.xpath(".//source")
            if source_elems:
                citation_data["source"] = self._get_lxml_text(source_elems[0])

            # Extract authors
            authors = self._extract_citation_authors_lxml(citation_elem)
            if authors:
                citation_data["authors"] = authors

            # Extract year
            year_elems = citation_elem.xpath(".//year")
            if year_elems:
                citation_data["year"] = self._get_lxml_text(year_elems[0])

            # Extract volume
            volume_elems = citation_elem.xpath(".//volume")
            if volume_elems:
                citation_data["volume"] = self._get_lxml_text(volume_elems[0])

            # Extract issue
            issue_elems = citation_elem.xpath(".//issue")
            if issue_elems:
                citation_data["issue"] = self._get_lxml_text(issue_elems[0])

            # Extract page range
            fpage_elems = citation_elem.xpath(".//fpage")
            lpage_elems = citation_elem.xpath(".//lpage")
            if fpage_elems:
                fpage = self._get_lxml_text(fpage_elems[0])
                citation_data["first_page"] = fpage

                if lpage_elems:
                    lpage = self._get_lxml_text(lpage_elems[0])
                    citation_data["last_page"] = lpage
                    citation_data["pages"] = f"{fpage}-{lpage}"
                else:
                    citation_data["pages"] = fpage

            # Extract DOI
            doi_elems = citation_elem.xpath('.//pub-id[@pub-id-type="doi"]')
            if doi_elems:
                citation_data["doi"] = self._get_lxml_text(doi_elems[0])

            # Extract PMID
            pmid_elems = citation_elem.xpath('.//pub-id[@pub-id-type="pmid"]')
            if pmid_elems:
                citation_data["pmid"] = self._get_lxml_text(pmid_elems[0])

        except Exception as e:
            self.logger.warning(f"Structured citation extraction failed: {e}")

        return citation_data

    def _extract_citation_authors_etree(self, citation_elem: ET.Element) -> List[str]:
        """Extract authors from citation using ElementTree."""
        authors = []

        try:
            # Look for person-group with person-group-type="author"
            author_groups = citation_elem.findall(
                './/person-group[@person-group-type="author"]'
            )

            for group in author_groups:
                # Find individual names
                name_elems = group.findall(".//name")

                for name_elem in name_elems:
                    author_name = self._extract_citation_author_name_etree(name_elem)
                    if author_name:
                        authors.append(author_name)

            # If no structured authors found, look for simple name elements
            if not authors:
                name_elems = citation_elem.findall(".//name")
                for name_elem in name_elems:
                    author_name = self._extract_citation_author_name_etree(name_elem)
                    if author_name:
                        authors.append(author_name)

        except Exception as e:
            self.logger.warning(f"Citation author extraction failed: {e}")

        return authors

    def _extract_citation_authors_lxml(self, citation_elem) -> List[str]:
        """Extract authors from citation using lxml."""
        authors = []

        try:
            # Look for person-group with person-group-type="author"
            author_groups = citation_elem.xpath(
                './/person-group[@person-group-type="author"]'
            )

            for group in author_groups:
                # Find individual names
                name_elems = group.xpath(".//name")

                for name_elem in name_elems:
                    author_name = self._extract_citation_author_name_lxml(name_elem)
                    if author_name:
                        authors.append(author_name)

            # If no structured authors found, look for simple name elements
            if not authors:
                name_elems = citation_elem.xpath(".//name")
                for name_elem in name_elems:
                    author_name = self._extract_citation_author_name_lxml(name_elem)
                    if author_name:
                        authors.append(author_name)

        except Exception as e:
            self.logger.warning(f"Citation author extraction failed: {e}")

        return authors

    def _extract_citation_author_name_etree(
        self, name_elem: ET.Element
    ) -> Optional[str]:
        """Extract author name from citation name element using ElementTree."""
        name_parts = []

        try:
            # Extract given names
            given_names = name_elem.find("given-names")
            if given_names is not None and given_names.text:
                name_parts.append(given_names.text.strip())

            # Extract surname
            surname = name_elem.find("surname")
            if surname is not None and surname.text:
                name_parts.append(surname.text.strip())

            return " ".join(name_parts) if name_parts else None

        except Exception:
            return None

    def _extract_citation_author_name_lxml(self, name_elem) -> Optional[str]:
        """Extract author name from citation name element using lxml."""
        name_parts = []

        try:
            # Extract given names
            given_names = name_elem.xpath("./given-names")
            if given_names:
                name_parts.append(self._get_lxml_text(given_names[0]))

            # Extract surname
            surnames = name_elem.xpath("./surname")
            if surnames:
                name_parts.append(self._get_lxml_text(surnames[0]))

            return " ".join(name_parts) if name_parts else None

        except Exception:
            return None

    def _parse_unstructured_reference(self, raw_text: str) -> Dict[str, Any]:
        """
        Parse unstructured reference text to extract basic information.

        Args:
            raw_text (str): Raw reference text

        Returns:
            Dict[str, Any]: Parsed reference data
        """
        parsed_data = {}

        try:
            # Extract year using pattern
            year_pattern = re.compile(r"\b(19|20)\d{2}\b")
            year_match = year_pattern.search(raw_text)
            if year_match:
                parsed_data["year"] = year_match.group(0)

            # Extract DOI using existing pattern
            doi_match = self.citation_patterns["xml_doi"].search(raw_text)
            if not doi_match:
                # Try general DOI pattern
                doi_pattern = re.compile(
                    r"(?:doi:?\s*)?(?:https?://(?:dx\.)?doi\.org/)?(10\.\d{4,}/[^\s]+)",
                    re.IGNORECASE,
                )
                doi_match = doi_pattern.search(raw_text)

            if doi_match:
                parsed_data["doi"] = (
                    doi_match.group(1)
                    if hasattr(doi_match, "group") and len(doi_match.groups()) > 0
                    else doi_match.group(0)
                )

            # Simple title extraction (text before year or in quotes)
            if "year" in parsed_data:
                year = parsed_data["year"]
                year_pos = raw_text.find(year)
                if year_pos > 20:  # Ensure reasonable title length
                    potential_title = raw_text[:year_pos].strip()
                    # Clean up potential title
                    potential_title = re.sub(
                        r"^[^a-zA-Z]*", "", potential_title
                    )  # Remove leading non-letters
                    potential_title = re.sub(
                        r"[^a-zA-Z\s]*$", "", potential_title
                    )  # Remove trailing non-letters
                    if 10 <= len(potential_title) <= 200:
                        parsed_data["title"] = potential_title

            # Extract quoted titles
            quoted_title = re.search(r'"([^"]{10,200})"', raw_text)
            if quoted_title and "title" not in parsed_data:
                parsed_data["title"] = quoted_title.group(1)

        except Exception as e:
            self.logger.warning(f"Unstructured reference parsing failed: {e}")

        return parsed_data
