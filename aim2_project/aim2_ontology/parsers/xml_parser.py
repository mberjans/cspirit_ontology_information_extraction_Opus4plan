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

# Import enhanced metadata framework
from .metadata_framework import (
    FigureMetadata, TableMetadata, FigureType, TableType, DataType,
    ContextInfo, TechnicalDetails, ContentAnalysis, QualityMetrics,
    ContentExtractor, QualityAssessment, ExtractionSummary
)
from .content_utils import (
    TableContentExtractor, FigureContentExtractor, TextAnalyzer, StatisticalAnalyzer
)


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

    # Enhanced XML namespace mappings for multiple schemas
    COMMON_NAMESPACES = {
        # JATS/PMC Namespaces
        "jats": "http://jats.nlm.nih.gov",
        "article": "http://dtd.nlm.nih.gov/xhtml",
        "xlink": "http://www.w3.org/1999/xlink",
        "mml": "http://www.w3.org/1998/Math/MathML",
        "xml": "http://www.w3.org/XML/1998/namespace",
        
        # NLM DTD Namespaces
        "nlm": "http://dtd.nlm.nih.gov",
        "nlm-book": "http://dtd.nlm.nih.gov/book",
        "nlm-ncbi": "http://www.ncbi.nlm.nih.gov",
        
        # DocBook Namespaces
        "docbook": "http://docbook.org/ns/docbook",
        "db": "http://docbook.org/ns/docbook",
        
        # TEI Namespaces
        "tei": "http://www.tei-c.org/ns/1.0",
        
        # Common XML Namespaces
        "xsi": "http://www.w3.org/2001/XMLSchema-instance",
        "xs": "http://www.w3.org/2001/XMLSchema",
        "xmlns": "http://www.w3.org/2000/xmlns/",
        
        # Dublin Core
        "dc": "http://purl.org/dc/elements/1.1/",
        "dcterms": "http://purl.org/dc/terms/",
        
        # PRISM
        "prism": "http://prismstandard.org/namespaces/basic/2.0/",
        
        # CrossRef
        "crossref": "http://www.crossref.org/schema/4.4.0",
        
        # ORCID
        "orcid": "http://orcid.org/",
    }
    
    # Schema detection patterns
    XML_SCHEMA_INDICATORS = {
        "jats": [
            "http://jats.nlm.nih.gov",
            "JATS-archivearticle1.dtd",
            "JATS-journalpublishing1.dtd",
            "article-meta",
            "journal-meta"
        ],
        "pmc": [
            "http://dtd.nlm.nih.gov",
            "archivearticle.dtd",
            "journalpublishing.dtd",
            "pmc-articleset",
            "article-meta"
        ],
        "nlm": [
            "http://dtd.nlm.nih.gov",
            "nlmmedlinecitationset.dtd",
            "medlinecitation",
            "pubmedarticle"
        ],
        "docbook": [
            "http://docbook.org",
            "docbook.dtd",
            "<book",
            "<article",
            "<chapter"
        ],
        "tei": [
            "http://www.tei-c.org",
            "tei.dtd",
            "<TEI",
            "<teiHeader"
        ]
    }

    # Enhanced element mappings for multiple schemas
    TEXT_ELEMENTS_BY_SCHEMA = {
        "jats": [
            "p", "sec", "title", "abstract", "body", "td", "th", "caption", "label",
            "list-item", "def", "article-title", "subtitle", "trans-title", "alt-title",
            "contrib", "name", "string-name", "given-names", "surname", "prefix", "suffix",
            "aff", "institution", "addr-line", "country", "email", "uri", "phone", "fax"
        ],
        "pmc": [
            "p", "sec", "title", "abstract", "body", "td", "th", "caption", "label",
            "list-item", "def", "article-title", "subtitle", "trans-title", "alt-title",
            "contrib", "name", "string-name", "given-names", "surname", "prefix", "suffix",
            "aff", "institution", "addr-line", "country", "email", "uri", "phone", "fax"
        ],
        "nlm": [
            "ArticleTitle", "AbstractText", "Affiliation", "AuthorList", "Author",
            "LastName", "FirstName", "Initials", "CollectiveName", "Identifier",
            "Journal", "JournalIssue", "Volume", "Issue", "PubDate", "Year", "Month", "Day"
        ],
        "docbook": [
            "para", "section", "title", "abstract", "chapter", "article", "book",
            "emphasis", "literal", "code", "programlisting", "screen", "synopsis",
            "entry", "row", "thead", "tbody", "tfoot", "table", "informaltable"
        ],
        "tei": [
            "p", "div", "head", "title", "text", "body", "front", "back",
            "cell", "row", "table", "list", "item", "label", "note", "ref",
            "persName", "placeName", "orgName", "name", "rs", "term", "gloss"
        ],
        "generic": [
            "p", "div", "span", "section", "article", "title", "h1", "h2", "h3", "h4", "h5", "h6",
            "td", "th", "tr", "table", "thead", "tbody", "tfoot", "caption",
            "li", "ul", "ol", "dl", "dt", "dd", "blockquote", "pre", "code"
        ]
    }

    # Enhanced section type mappings for multiple schemas
    SECTION_TYPES_BY_SCHEMA = {
        "jats": {
            "abstract": ["abstract", "summary", "synopsis"],
            "introduction": ["intro", "introduction", "background", "rationale"],
            "methods": ["methods", "materials", "methodology", "experimental", "materials-methods"],
            "results": ["results", "findings", "outcomes", "observations"],
            "discussion": ["discussion", "analysis", "interpretation", "commentary"],
            "conclusion": ["conclusion", "conclusions", "summary", "closing"],
            "acknowledgments": ["ack", "acknowledgments", "acknowledgements", "thanks"],
            "references": ["ref-list", "references", "bibliography", "citations"],
            "appendix": ["app", "appendix", "supplementary", "supplement", "supporting-info"],
            "ethics": ["ethics", "ethical", "compliance", "consent"],
            "funding": ["funding", "financial", "support", "grant"],
            "availability": ["availability", "data-availability", "code-availability"]
        },
        "pmc": {
            "abstract": ["abstract", "summary", "synopsis"],
            "introduction": ["intro", "introduction", "background", "rationale"],
            "methods": ["methods", "materials", "methodology", "experimental", "materials-methods"],
            "results": ["results", "findings", "outcomes", "observations"],
            "discussion": ["discussion", "analysis", "interpretation", "commentary"],
            "conclusion": ["conclusion", "conclusions", "summary", "closing"],
            "acknowledgments": ["ack", "acknowledgments", "acknowledgements", "thanks"],
            "references": ["ref-list", "references", "bibliography", "citations"],
            "appendix": ["app", "appendix", "supplementary", "supplement", "supporting-info"],
            "ethics": ["ethics", "ethical", "compliance", "consent"],
            "funding": ["funding", "financial", "support", "grant"],
            "availability": ["availability", "data-availability", "code-availability"]
        },
        "generic": {
            "abstract": ["abstract", "summary"],
            "introduction": ["intro", "introduction", "background"],
            "methods": ["methods", "materials", "methodology"],
            "results": ["results", "findings"],
            "discussion": ["discussion", "analysis"],
            "conclusion": ["conclusion", "conclusions"],
            "acknowledgments": ["acknowledgments", "acknowledgements"],
            "references": ["references", "bibliography"],
            "appendix": ["appendix", "supplementary"],
        }
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
        self._detected_schema = None
        self._schema_version = None
        self._supported_encodings = ["utf-8", "utf-16", "iso-8859-1", "cp1252", "ascii", "latin-1"]

        # Configure XML-specific section patterns
        self._setup_xml_section_patterns()

        # Configure XML metadata extractors
        self._setup_xml_metadata_extractors()

        # Add XML-specific validation rules
        self.add_validation_rule("xml_format", self._validate_xml_format)
        self.add_validation_rule("pmc_structure", self._validate_pmc_structure)

        # Initialize enhanced metadata framework utilities
        self.content_extractor = ContentExtractor()
        self.quality_assessor = QualityAssessment()
        self.table_content_extractor = TableContentExtractor()
        self.figure_content_extractor = FigureContentExtractor()
        self.text_analyzer = TextAnalyzer()
        self.statistical_analyzer = StatisticalAnalyzer()

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

    def _detect_xml_schema(self, xml_content: str, root_element=None) -> Dict[str, Any]:
        """
        Detect XML schema and version from content and structure.
        
        Args:
            xml_content (str): XML content as string
            root_element: Parsed root element (optional)
            
        Returns:
            Dict[str, Any]: Schema detection results
        """
        schema_info = {
            "schema": "unknown",
            "version": None,
            "confidence": 0.0,
            "indicators": [],
            "namespaces": {}
        }
        
        try:
            # Extract namespaces from content
            namespace_matches = re.findall(
                r'xmlns(?::([^=]+))?=["\']([^"\']+)["\']',
                xml_content[:2000],  # Check first 2KB for performance
                re.IGNORECASE
            )
            
            detected_namespaces = {}
            for prefix, uri in namespace_matches:
                key = prefix if prefix else "default"
                detected_namespaces[key] = uri
            
            schema_info["namespaces"] = detected_namespaces
            
            # Check for DOCTYPE declarations
            doctype_match = re.search(
                r'<!DOCTYPE\s+([^\s>]+)[^>]*(?:PUBLIC\s+["\']([^"\']+)["\'])?[^>]*(?:["\']([^"\']+)["\'])?',
                xml_content[:1000],
                re.IGNORECASE | re.DOTALL
            )
            
            if doctype_match:
                root_elem_name = doctype_match.group(1)
                public_id = doctype_match.group(2) if doctype_match.group(2) else ""
                system_id = doctype_match.group(3) if doctype_match.group(3) else ""
                
                schema_info["doctype"] = {
                    "root_element": root_elem_name,
                    "public_id": public_id,
                    "system_id": system_id
                }
            
            # Score schemas based on indicators
            schema_scores = {}
            
            for schema_name, indicators in self.XML_SCHEMA_INDICATORS.items():
                score = 0
                matched_indicators = []
                
                for indicator in indicators:
                    if indicator in xml_content:
                        score += 1
                        matched_indicators.append(indicator)
                    
                    # Check in namespaces
                    for ns_uri in detected_namespaces.values():
                        if indicator in ns_uri:
                            score += 2  # Namespace matches are more reliable
                            matched_indicators.append(f"namespace:{indicator}")
                
                if score > 0:
                    schema_scores[schema_name] = {
                        "score": score,
                        "indicators": matched_indicators
                    }
            
            # Determine best match
            if schema_scores:
                best_schema = max(schema_scores.keys(), key=lambda x: schema_scores[x]["score"])
                best_score = schema_scores[best_schema]["score"]
                
                schema_info["schema"] = best_schema
                schema_info["confidence"] = min(best_score / 3.0, 1.0)  # Normalize to 0-1
                schema_info["indicators"] = schema_scores[best_schema]["indicators"]
            
            # Try to detect version information
            version_patterns = [
                r'version=["\']([^"\']+)["\']',
                r'dtd[^>]*([0-9]+\.[0-9]+)',
                r'schema[^>]*([0-9]+\.[0-9]+)',
                r'JATS-[^"\']*([0-9]+\.[0-9]+)',
            ]
            
            for pattern in version_patterns:
                version_match = re.search(pattern, xml_content[:1000], re.IGNORECASE)
                if version_match:
                    schema_info["version"] = version_match.group(1)
                    break
            
            # Store detected schema for later use
            self._detected_schema = schema_info["schema"]
            self._schema_version = schema_info["version"]
            
        except Exception as e:
            self.logger.warning(f"Schema detection failed: {e}")
        
        return schema_info

    def _get_schema_specific_elements(self, schema: str = None) -> List[str]:
        """
        Get text elements specific to detected or specified schema.
        
        Args:
            schema (str, optional): Schema name, uses detected schema if None
            
        Returns:
            List[str]: List of element names for text extraction
        """
        if schema is None:
            schema = self._detected_schema or "generic"
        
        return self.TEXT_ELEMENTS_BY_SCHEMA.get(schema, self.TEXT_ELEMENTS_BY_SCHEMA["generic"])

    def _get_schema_specific_sections(self, schema: str = None) -> Dict[str, List[str]]:
        """
        Get section mappings specific to detected or specified schema.
        
        Args:
            schema (str, optional): Schema name, uses detected schema if None
            
        Returns:
            Dict[str, List[str]]: Section type mappings
        """
        if schema is None:
            schema = self._detected_schema or "generic"
        
        return self.SECTION_TYPES_BY_SCHEMA.get(schema, self.SECTION_TYPES_BY_SCHEMA["generic"])

    def _enhance_namespace_handling(self, root_element, parsing_method: str = "etree") -> Dict[str, Any]:
        """
        Enhanced namespace extraction and handling.
        
        Args:
            root_element: Parsed root element
            parsing_method (str): Parsing method used
            
        Returns:
            Dict[str, Any]: Enhanced namespace information
        """
        namespace_info = {
            "declared": {},
            "used": {},
            "conflicts": [],
            "schema_namespaces": {},
            "prefixes": {}
        }
        
        try:
            if parsing_method == "lxml" and LXML_AVAILABLE:
                # Use lxml's built-in namespace map
                if hasattr(root_element, 'nsmap'):
                    namespace_info["declared"] = dict(root_element.nsmap) or {}
                
                # Find all used namespaces in the document
                for elem in root_element.iter():
                    if elem.tag.startswith("{"):
                        # Extract namespace URI from tag
                        ns_end = elem.tag.find("}")
                        if ns_end > 0:
                            ns_uri = elem.tag[1:ns_end]
                            local_name = elem.tag[ns_end + 1:]
                            
                            if ns_uri not in namespace_info["used"]:
                                namespace_info["used"][ns_uri] = []
                            
                            if local_name not in namespace_info["used"][ns_uri]:
                                namespace_info["used"][ns_uri].append(local_name)
            else:
                # ElementTree namespace handling
                namespace_info["declared"] = self._extract_namespaces_etree(root_element)
                
                # Scan for used namespaces
                for elem in root_element.iter():
                    # Check for namespace prefixes in tag names
                    if ":" in elem.tag:
                        prefix, local_name = elem.tag.split(":", 1)
                        if prefix in namespace_info["declared"]:
                            ns_uri = namespace_info["declared"][prefix]
                            if ns_uri not in namespace_info["used"]:
                                namespace_info["used"][ns_uri] = []
                            if local_name not in namespace_info["used"][ns_uri]:
                                namespace_info["used"][ns_uri].append(local_name)
            
            # Map known namespaces to schemas
            for ns_uri in namespace_info["declared"].values():
                for known_prefix, known_uri in self.COMMON_NAMESPACES.items():
                    if ns_uri == known_uri:
                        namespace_info["schema_namespaces"][known_prefix] = ns_uri
                        break
            
            # Create reverse mapping of URIs to prefixes
            for prefix, uri in namespace_info["declared"].items():
                if uri not in namespace_info["prefixes"]:
                    namespace_info["prefixes"][uri] = []
                namespace_info["prefixes"][uri].append(prefix)
            
            # Detect namespace conflicts (same prefix, different URIs)
            prefix_to_uri = {}
            for prefix, uri in namespace_info["declared"].items():
                if prefix in prefix_to_uri and prefix_to_uri[prefix] != uri:
                    namespace_info["conflicts"].append({
                        "prefix": prefix,
                        "uris": [prefix_to_uri[prefix], uri]
                    })
                else:
                    prefix_to_uri[prefix] = uri
                    
        except Exception as e:
            self.logger.warning(f"Enhanced namespace handling failed: {e}")
        
        return namespace_info

    def _extract_document_structure_metadata(self, root_element, parsing_method: str = "etree") -> Dict[str, Any]:
        """
        Extract comprehensive document structure metadata.
        
        Args:
            root_element: Parsed root element
            parsing_method (str): Parsing method used
            
        Returns:
            Dict[str, Any]: Document structure metadata
        """
        structure_metadata = {
            "root_element": "",
            "depth": 0,
            "element_counts": {},
            "attribute_counts": {},
            "text_density": 0.0,
            "structure_complexity": 0.0,
            "has_mixed_content": False,
            "language_info": {},
            "processing_instructions": [],
            "comments_count": 0
        }
        
        try:
            # Basic structure info
            structure_metadata["root_element"] = root_element.tag
            
            # Count elements and analyze structure
            max_depth = 0
            total_elements = 0
            total_text_length = 0
            total_attributes = 0
            element_counts = {}
            attribute_counts = {}
            languages = set()
            
            def analyze_element(elem, depth=0):
                nonlocal max_depth, total_elements, total_text_length, total_attributes
                
                max_depth = max(max_depth, depth)
                total_elements += 1
                
                # Count element types
                tag_name = elem.tag.split("}")[-1] if "}" in elem.tag else elem.tag
                element_counts[tag_name] = element_counts.get(tag_name, 0) + 1
                
                # Count attributes
                total_attributes += len(elem.attrib)
                for attr_name in elem.attrib.keys():
                    clean_attr = attr_name.split("}")[-1] if "}" in attr_name else attr_name
                    attribute_counts[clean_attr] = attribute_counts.get(clean_attr, 0) + 1
                
                # Extract text content
                if elem.text:
                    total_text_length += len(elem.text.strip())
                if elem.tail:
                    total_text_length += len(elem.tail.strip())
                
                # Check for language attributes
                for lang_attr in ["xml:lang", "lang"]:
                    if lang_attr in elem.attrib:
                        languages.add(elem.attrib[lang_attr])
                
                # Check for mixed content
                if elem.text and elem.text.strip() and len(elem) > 0:
                    structure_metadata["has_mixed_content"] = True
                
                # Recursively analyze children
                for child in elem:
                    analyze_element(child, depth + 1)
            
            analyze_element(root_element)
            
            structure_metadata["depth"] = max_depth
            structure_metadata["element_counts"] = element_counts
            structure_metadata["attribute_counts"] = attribute_counts
            structure_metadata["total_elements"] = total_elements
            structure_metadata["total_attributes"] = total_attributes
            
            # Calculate metrics
            if total_elements > 0:
                structure_metadata["text_density"] = total_text_length / total_elements
                structure_metadata["structure_complexity"] = (max_depth * len(element_counts)) / total_elements
            
            # Language information
            if languages:
                structure_metadata["language_info"] = {
                    "languages": list(languages),
                    "is_multilingual": len(languages) > 1,
                    "primary_language": list(languages)[0] if languages else None
                }
            
            # Extract processing instructions if using lxml
            if parsing_method == "lxml" and LXML_AVAILABLE:
                try:
                    # Get processing instructions from the document
                    if hasattr(root_element, 'getroot'):
                        doc = root_element.getroot()
                    else:
                        doc = root_element
                        
                    # This would require access to the document tree, simplified for now
                    structure_metadata["processing_instructions"] = []
                    
                except Exception:
                    pass
                    
        except Exception as e:
            self.logger.warning(f"Document structure metadata extraction failed: {e}")
        
        return structure_metadata

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
            
            # Detect XML schema and structure
            schema_info = self._detect_xml_schema(xml_content, root)
            enhanced_namespaces = self._enhance_namespace_handling(root, "etree")
            structure_metadata = self._extract_document_structure_metadata(root, "etree")

            return {
                "raw_content": xml_content,
                "root_element": root,
                "namespaces": self._xml_namespaces,
                "enhanced_namespaces": enhanced_namespaces,
                "schema_info": schema_info,
                "structure_metadata": structure_metadata,
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
            
            # Detect XML schema and structure
            schema_info = self._detect_xml_schema(xml_content, root)
            enhanced_namespaces = self._enhance_namespace_handling(root, "lxml")
            structure_metadata = self._extract_document_structure_metadata(root, "lxml")

            return {
                "raw_content": xml_content,
                "root_element": root,
                "namespaces": self._xml_namespaces,
                "enhanced_namespaces": enhanced_namespaces,
                "schema_info": schema_info,
                "structure_metadata": structure_metadata,
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
        Categorize a section based on type and title using schema-specific mappings.

        Args:
            sec_type (str): Section type attribute
            title (str): Section title

        Returns:
            str: Section category
        """
        # Get schema-specific section types
        section_types = self._get_schema_specific_sections()
        
        # Check sec-type first
        if sec_type and sec_type != "unknown":
            sec_type_lower = sec_type.lower()
            for category, types in section_types.items():
                if any(type_name in sec_type_lower for type_name in types):
                    return category

        # Check title if sec-type is not informative
        if title:
            title_lower = title.lower()
            for category, types in section_types.items():
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
        Extract figures and tables with comprehensive metadata and content analysis.

        This enhanced method provides:
        1. Comprehensive metadata extraction using standardized structures
        2. Advanced content analysis and quality assessment
        3. Statistical analysis for numerical table data
        4. XML-specific parsing with namespace support
        5. Quality metrics and validation scoring

        Args:
            content (Any): Parsed XML content or raw XML data
            **kwargs: Enhanced extraction options including:
                - extract_table_data (bool): Extract actual table content (default: True)
                - analyze_content (bool): Perform content analysis (default: True)
                - include_quality_metrics (bool): Include quality assessment (default: True)
                - confidence_threshold (float): Minimum confidence for inclusion (default: 0.0)
                - return_metadata_objects (bool): Return metadata objects instead of dicts (default: False)

        Returns:
            Dict[str, Any]: Comprehensive extraction results with standardized format:
                {
                    "figures": List[Dict[str, Any]],
                    "tables": List[Dict[str, Any]],
                    "extraction_summary": Dict[str, Any]
                }

        Raises:
            ExtractionException: If extraction fails
        """
        start_time = datetime.now()
        
        try:
            # If content is not parsed, parse it first
            if not isinstance(content, dict):
                content = self.parse(content, **kwargs)

            root_element = content.get("root_element")
            if root_element is None:
                raise ExtractionException("No root element found in parsed content")

            # Extract figures with comprehensive metadata
            figure_metadata_list = self._extract_figures_comprehensive_xml(root_element, content, **kwargs)

            # Extract tables with comprehensive metadata
            table_metadata_list = self._extract_tables_comprehensive_xml(root_element, content, **kwargs)

            # Convert to dictionary format unless metadata objects requested
            if kwargs.get("return_metadata_objects", False):
                figures_result = figure_metadata_list
                tables_result = table_metadata_list
            else:
                figures_result = [fig.to_dict() for fig in figure_metadata_list]
                tables_result = [table.to_dict() for table in table_metadata_list]

            # Generate extraction summary
            extraction_time = (datetime.now() - start_time).total_seconds()
            parsing_method = content.get("parsing_method", "etree")
            
            summary = self.quality_assessor.generate_extraction_summary(
                figure_metadata_list, table_metadata_list, extraction_time, f"XML_{parsing_method}"
            )

            result = {
                "figures": figures_result,
                "tables": tables_result,
                "extraction_summary": summary.__dict__ if not kwargs.get("return_metadata_objects", False) else summary
            }

            self.logger.info(
                f"Comprehensive XML extraction completed: {len(figures_result)} figures, "
                f"{len(tables_result)} tables (avg quality: {summary.average_quality_score:.2f})"
            )
            
            return result

        except Exception as e:
            self.logger.error(f"Comprehensive XML figure/table extraction failed: {e}")
            raise ExtractionException(f"Failed to extract figures/tables: {e}")

    def _extract_figures_comprehensive_xml(
        self, root_element, content: Dict[str, Any], **kwargs
    ) -> List[FigureMetadata]:
        """Extract figures with comprehensive metadata from XML."""
        figure_metadata_list = []
        
        try:
            # Use existing extraction methods as base
            parsing_method = content.get("parsing_method", "etree")
            
            if parsing_method == "lxml" and LXML_AVAILABLE:
                basic_figures = self._extract_figures_lxml(root_element, **kwargs)
            else:
                basic_figures = self._extract_figures_etree(root_element, **kwargs)
            
            # Get full document text for context analysis
            document_text = self._extract_text_from_xml(root_element)
            
            for i, basic_figure in enumerate(basic_figures):
                # Create comprehensive metadata object
                figure_meta = FigureMetadata()
                
                # Basic information from existing extraction
                figure_meta.id = basic_figure.get("id", f"figure_{i+1}")
                figure_meta.label = basic_figure.get("label", "")
                figure_meta.caption = basic_figure.get("caption", "")
                figure_meta.position = basic_figure.get("position", i)
                figure_meta.source_parser = "XMLParser"
                figure_meta.extraction_method = f"xml_{parsing_method}"
                
                # Classify figure type
                figure_meta.figure_type = self.content_extractor.classify_figure_type(
                    figure_meta.caption, document_text
                )
                
                # Extract context information
                figure_meta.context = self._extract_xml_figure_context(
                    figure_meta, root_element, document_text, **kwargs
                )
                
                # Extract technical details from XML attributes
                figure_meta.technical = self._extract_xml_figure_technical_details(
                    basic_figure, root_element, **kwargs
                )
                
                # Perform content analysis
                if kwargs.get("analyze_content", True):
                    extracted_text = basic_figure.get("extracted_text", [])
                    figure_meta.analysis = self.figure_content_extractor.analyze_figure_content(
                        figure_meta.caption, extracted_text
                    )
                
                # Quality assessment
                if kwargs.get("include_quality_metrics", True):
                    figure_meta.quality = self.quality_assessor.assess_figure_quality(figure_meta)
                
                # Store raw data for reference
                figure_meta.raw_data = basic_figure
                
                # Filter by confidence threshold
                confidence_threshold = kwargs.get("confidence_threshold", 0.0)
                if figure_meta.quality.extraction_confidence >= confidence_threshold:
                    figure_metadata_list.append(figure_meta)
            
            return figure_metadata_list
            
        except Exception as e:
            self.logger.error(f"Comprehensive XML figure extraction failed: {e}")
            return self._create_fallback_xml_figure_metadata(basic_figures if 'basic_figures' in locals() else [])

    def _extract_tables_comprehensive_xml(
        self, root_element, content: Dict[str, Any], **kwargs
    ) -> List[TableMetadata]:
        """Extract tables with comprehensive metadata from XML."""
        table_metadata_list = []
        
        try:
            # Use existing extraction methods as base
            parsing_method = content.get("parsing_method", "etree")
            
            if parsing_method == "lxml" and LXML_AVAILABLE:
                basic_tables = self._extract_tables_lxml(root_element, **kwargs)
            else:
                basic_tables = self._extract_tables_etree(root_element, **kwargs)
            
            # Get full document text for context analysis
            document_text = self._extract_text_from_xml(root_element)
            
            for i, basic_table in enumerate(basic_tables):
                # Create comprehensive metadata object
                table_meta = TableMetadata()
                
                # Basic information from existing extraction
                table_meta.id = basic_table.get("id", f"table_{i+1}")
                table_meta.label = basic_table.get("label", "")
                table_meta.caption = basic_table.get("caption", "")
                table_meta.position = basic_table.get("position", i)
                table_meta.source_parser = "XMLParser"
                table_meta.extraction_method = f"xml_{parsing_method}"
                
                # Extract table content from XML structure
                raw_table_data = self._extract_xml_table_data(basic_table, root_element, parsing_method)
                
                if raw_table_data and kwargs.get("extract_table_data", True):
                    # Parse table structure
                    table_meta.structure = self.table_content_extractor.parse_table_structure(raw_table_data)
                    
                    # Extract table content
                    table_meta.content = self.table_content_extractor.extract_table_content(
                        raw_table_data, table_meta.structure
                    )
                    
                    # Classify table type
                    table_meta.table_type = self.content_extractor.classify_table_type(
                        table_meta.caption, table_meta.structure.column_headers, document_text
                    )
                    
                    # Perform content analysis
                    if kwargs.get("analyze_content", True):
                        table_meta.analysis = self.table_content_extractor.analyze_table_content(
                            table_meta.content, table_meta.caption
                        )
                else:
                    # Basic structure from available XML information
                    table_meta.structure.rows = basic_table.get("rows", 0)
                    table_meta.structure.columns = basic_table.get("columns", 0)
                    table_meta.table_type = self.content_extractor.classify_table_type(
                        table_meta.caption, [], document_text
                    )
                
                # Extract context information
                table_meta.context = self._extract_xml_table_context(
                    table_meta, root_element, document_text, **kwargs
                )
                
                # Extract technical details from XML attributes
                table_meta.technical = self._extract_xml_table_technical_details(
                    basic_table, root_element, **kwargs
                )
                
                # Quality assessment
                if kwargs.get("include_quality_metrics", True):
                    table_meta.quality = self.quality_assessor.assess_table_quality(table_meta)
                
                # Store raw data for reference
                table_meta.raw_data = basic_table
                
                # Filter by confidence threshold
                confidence_threshold = kwargs.get("confidence_threshold", 0.0)
                if table_meta.quality.extraction_confidence >= confidence_threshold:
                    table_metadata_list.append(table_meta)
            
            return table_metadata_list
            
        except Exception as e:
            self.logger.error(f"Comprehensive XML table extraction failed: {e}")
            return self._create_fallback_xml_table_metadata(basic_tables if 'basic_tables' in locals() else [])

    def _extract_text_from_xml(self, root_element) -> str:
        """Extract all text content from XML for context analysis."""
        try:
            if hasattr(root_element, 'itertext'):  # lxml
                return ' '.join(root_element.itertext())
            else:  # ElementTree
                # Recursive text extraction for ElementTree
                def extract_text_recursive(element):
                    text = element.text or ""
                    for child in element:
                        text += extract_text_recursive(child)
                        text += child.tail or ""
                    return text
                
                return extract_text_recursive(root_element)
        except Exception as e:
            self.logger.warning(f"Failed to extract text from XML: {e}")
            return ""

    def _extract_xml_table_data(self, basic_table: Dict[str, Any], root_element, parsing_method: str) -> List[List[str]]:
        """Extract actual table data from XML table element."""
        try:
            # Get table element from basic_table data
            table_element = basic_table.get("element")
            if not table_element:
                return []
            
            table_data = []
            
            if parsing_method == "lxml" and LXML_AVAILABLE:
                # Use lxml xpath for better table parsing
                rows = table_element.xpath(".//tr")
                for row in rows:
                    row_data = []
                    cells = row.xpath(".//td | .//th")
                    for cell in cells:
                        cell_text = ''.join(cell.itertext()).strip()
                        row_data.append(cell_text)
                    if row_data:
                        table_data.append(row_data)
            else:
                # Use ElementTree for standard parsing
                rows = table_element.findall(".//tr")
                for row in rows:
                    row_data = []
                    cells = row.findall(".//td") + row.findall(".//th")
                    for cell in cells:
                        cell_text = ''.join(cell.itertext()).strip() if hasattr(cell, 'itertext') else (cell.text or "").strip()
                        row_data.append(cell_text)
                    if row_data:
                        table_data.append(row_data)
            
            return table_data
            
        except Exception as e:
            self.logger.warning(f"Failed to extract XML table data: {e}")
            return []

    def _extract_xml_figure_context(
        self, figure_meta: FigureMetadata, root_element, document_text: str, **kwargs
    ) -> ContextInfo:
        """Extract contextual information for XML figure."""
        context = ContextInfo()
        
        try:
            # Find the section containing this figure
            if figure_meta.label or figure_meta.id:
                search_pattern = figure_meta.label or figure_meta.id
                context.section_context = self._find_xml_section_context(search_pattern, root_element)
            
            # Find cross-references in document text
            if figure_meta.label:
                context.cross_references = self._find_cross_references(figure_meta.label, document_text, "figure")
            
            # Extract related text from caption
            if figure_meta.caption:
                context.related_text = figure_meta.caption[:200]
            
            # Calculate document position
            if document_text and figure_meta.caption in document_text:
                caption_pos = document_text.find(figure_meta.caption)
                context.document_position = caption_pos / len(document_text) if document_text else 0.0
            
        except Exception as e:
            self.logger.warning(f"XML figure context extraction failed: {e}")
        
        return context

    def _extract_xml_table_context(
        self, table_meta: TableMetadata, root_element, document_text: str, **kwargs
    ) -> ContextInfo:
        """Extract contextual information for XML table."""
        context = ContextInfo()
        
        try:
            # Find the section containing this table
            if table_meta.label or table_meta.id:
                search_pattern = table_meta.label or table_meta.id
                context.section_context = self._find_xml_section_context(search_pattern, root_element)
            
            # Find cross-references in document text
            if table_meta.label:
                context.cross_references = self._find_cross_references(table_meta.label, document_text, "table")
            
            # Extract related text from caption
            if table_meta.caption:
                context.related_text = table_meta.caption[:200]
            
            # Calculate document position
            if document_text and table_meta.caption in document_text:
                caption_pos = document_text.find(table_meta.caption)
                context.document_position = caption_pos / len(document_text) if document_text else 0.0
            
        except Exception as e:
            self.logger.warning(f"XML table context extraction failed: {e}")
        
        return context

    def _find_xml_section_context(self, search_pattern: str, root_element) -> str:
        """Find section context from XML structure."""
        try:
            # Look for common section elements
            section_tags = ['sec', 'section', 'div']
            
            for tag in section_tags:
                sections = root_element.findall(f".//{tag}")
                for section in sections:
                    section_text = ''.join(section.itertext()) if hasattr(section, 'itertext') else ""
                    if search_pattern in section_text:
                        # Try to find section title
                        title_elem = section.find(".//title") or section.find(".//h1") or section.find(".//h2")
                        if title_elem is not None:
                            title_text = ''.join(title_elem.itertext()) if hasattr(title_elem, 'itertext') else title_elem.text or ""
                            return title_text.strip()
            
            return ""
            
        except Exception as e:
            self.logger.warning(f"XML section context extraction failed: {e}")
            return ""

    def _extract_xml_figure_technical_details(
        self, basic_figure: Dict[str, Any], root_element, **kwargs
    ) -> TechnicalDetails:
        """Extract technical details for XML figure."""
        technical = TechnicalDetails()
        
        try:
            # Extract from XML attributes
            fig_element = basic_figure.get("element")
            if fig_element is not None:
                # Check for graphic elements
                graphic_elems = fig_element.findall(".//graphic")
                if graphic_elems:
                    graphic = graphic_elems[0]
                    technical.file_format = graphic.get("mimetype", "unknown")
                    
                    # Try to extract dimensions if available
                    width = graphic.get("width")
                    height = graphic.get("height")
                    if width and height:
                        try:
                            technical.dimensions = (int(width), int(height))
                        except ValueError:
                            pass
            
            # Set encoding
            technical.encoding = "XML"
            
        except Exception as e:
            self.logger.warning(f"XML figure technical details extraction failed: {e}")
        
        return technical

    def _extract_xml_table_technical_details(
        self, basic_table: Dict[str, Any], root_element, **kwargs
    ) -> TechnicalDetails:
        """Extract technical details for XML table."""
        technical = TechnicalDetails()
        
        try:
            # Extract table dimensions from structure
            table_element = basic_table.get("element")
            if table_element is not None:
                rows = table_element.findall(".//tr")
                if rows:
                    technical.dimensions = (len(rows), basic_table.get("columns", 0))
            
            # Set encoding and format
            technical.encoding = "XML"
            technical.file_format = "table"
            
        except Exception as e:
            self.logger.warning(f"XML table technical details extraction failed: {e}")
        
        return technical

    def _find_cross_references(self, label: str, text: str, ref_type: str) -> List[str]:
        """Find cross-references to a figure or table in text."""
        references = []
        
        try:
            # Pattern to find references like "see Figure 1", "Table 2 shows", etc.
            if ref_type == "figure":
                patterns = [
                    rf'(?i)\b(?:see|refer to|as shown in|according to)\s+{re.escape(label)}\b',
                    rf'(?i)\b{re.escape(label)}\s+(?:shows|demonstrates|illustrates|depicts)\b',
                    rf'(?i)\b\({re.escape(label)}\)\b'
                ]
            else:  # table
                patterns = [
                    rf'(?i)\b(?:see|refer to|as shown in|according to)\s+{re.escape(label)}\b',
                    rf'(?i)\b{re.escape(label)}\s+(?:shows|presents|lists|contains)\b',
                    rf'(?i)\b\({re.escape(label)}\)\b'
                ]
            
            for pattern in patterns:
                matches = re.finditer(pattern, text)
                for match in matches:
                    # Extract surrounding context (50 chars before and after)
                    start = max(0, match.start() - 50)
                    end = min(len(text), match.end() + 50)
                    context = text[start:end].strip()
                    references.append(context)
        
        except Exception as e:
            self.logger.warning(f"Cross-reference extraction failed: {e}")
        
        return references[:5]  # Limit to 5 references

    def _create_fallback_xml_figure_metadata(self, basic_figures: List[Dict[str, Any]]) -> List[FigureMetadata]:
        """Create minimal figure metadata objects as fallback for XML."""
        fallback_list = []
        
        for i, basic_figure in enumerate(basic_figures):
            figure_meta = FigureMetadata()
            figure_meta.id = basic_figure.get("id", f"figure_{i+1}")
            figure_meta.caption = basic_figure.get("caption", "")
            figure_meta.label = basic_figure.get("label", "")
            figure_meta.position = i
            figure_meta.source_parser = "XMLParser"
            figure_meta.extraction_method = "xml_fallback"
            figure_meta.quality.extraction_confidence = 0.3  # Low confidence for fallback
            fallback_list.append(figure_meta)
        
        return fallback_list

    def _create_fallback_xml_table_metadata(self, basic_tables: List[Dict[str, Any]]) -> List[TableMetadata]:
        """Create minimal table metadata objects as fallback for XML."""
        fallback_list = []
        
        for i, basic_table in enumerate(basic_tables):
            table_meta = TableMetadata()
            table_meta.id = basic_table.get("id", f"table_{i+1}")
            table_meta.caption = basic_table.get("caption", "")
            table_meta.label = basic_table.get("label", "")
            table_meta.position = i
            table_meta.source_parser = "XMLParser"
            table_meta.extraction_method = "xml_fallback"
            table_meta.quality.extraction_confidence = 0.3  # Low confidence for fallback
            fallback_list.append(table_meta)
        
        return fallback_list

    def _extract_figures_etree(
        self, root_element: ET.Element, **kwargs
    ) -> List[Dict[str, Any]]:
        """Extract figures using ElementTree with enhanced metadata."""
        figures = []

        try:
            # Find figure elements (including fig-group elements)
            fig_elements = root_element.findall(".//fig")
            fig_group_elements = root_element.findall(".//fig-group")

            # Process individual figures
            for i, fig_elem in enumerate(fig_elements):
                figure_data = self._analyze_figure_etree(fig_elem, i)
                if figure_data:
                    figures.append(figure_data)

            # Process figure groups
            for i, fig_group_elem in enumerate(fig_group_elements):
                figure_group_data = self._analyze_figure_group_etree(fig_group_elem, i)
                if figure_group_data:
                    figures.extend(figure_group_data)

        except Exception as e:
            self.logger.warning(f"Figure extraction with ElementTree failed: {e}")

        return figures

    def _extract_figures_lxml(self, root_element, **kwargs) -> List[Dict[str, Any]]:
        """Extract figures using lxml with enhanced metadata."""
        figures = []

        try:
            # Find figure elements (including fig-group elements)
            fig_elements = root_element.xpath(".//fig")
            fig_group_elements = root_element.xpath(".//fig-group")

            # Process individual figures
            for i, fig_elem in enumerate(fig_elements):
                figure_data = self._analyze_figure_lxml(fig_elem, i)
                if figure_data:
                    figures.append(figure_data)

            # Process figure groups
            for i, fig_group_elem in enumerate(fig_group_elements):
                figure_group_data = self._analyze_figure_group_lxml(fig_group_elem, i)
                if figure_group_data:
                    figures.extend(figure_group_data)

        except Exception as e:
            self.logger.warning(f"Figure extraction with lxml failed: {e}")

        return figures

    def _analyze_figure_etree(
        self, fig_elem: ET.Element, index: int
    ) -> Optional[Dict[str, Any]]:
        """Analyze a figure element using ElementTree with enhanced metadata extraction."""
        try:
            # Extract figure ID and basic attributes
            fig_id = fig_elem.get("id", f"figure_{index + 1}")
            fig_type = fig_elem.get("fig-type", "figure")
            position = fig_elem.get("position", "")
            orientation = fig_elem.get("orientation", "")

            # Extract label with enhanced parsing
            label_elem = fig_elem.find("./label")
            label = (
                label_elem.text.strip()
                if label_elem is not None and label_elem.text
                else f"Figure {index + 1}"
            )

            # Extract enhanced caption with structure
            caption_data = self._extract_enhanced_caption_etree(fig_elem)
            
            # Extract comprehensive graphic information
            graphics_data = self._extract_graphics_metadata_etree(fig_elem)
            
            # Extract alternative representations
            alternatives = self._extract_alternatives_etree(fig_elem)
            
            # Extract licensing and attribution
            licensing_info = self._extract_licensing_info_etree(fig_elem)
            
            # Extract positioning and layout information
            layout_info = self._extract_layout_info_etree(fig_elem)
            
            # Extract cross-references
            cross_refs = self._extract_figure_cross_refs_etree(fig_elem)
            
            # Determine figure content type
            content_type = self._determine_figure_content_type(graphics_data, caption_data.get("text", ""))
            
            return {
                "id": fig_id,
                "label": label,
                "caption": caption_data.get("text", ""),
                "caption_structure": caption_data.get("structure", {}),
                "graphics": graphics_data.get("files", []),
                "graphics_metadata": graphics_data.get("metadata", {}),
                "alternatives": alternatives,
                "licensing": licensing_info,
                "layout": layout_info,
                "cross_references": cross_refs,
                "fig_type": fig_type,
                "position": position,
                "orientation": orientation,
                "content_type": content_type,
                "type": "figure",
                "extraction_method": "xml_structure_enhanced",
                "index": index + 1,
                "has_subfigures": len(fig_elem.findall(".//fig")) > 0,
                "element_count": len(list(fig_elem.iter())),
            }

        except Exception as e:
            self.logger.warning(f"Enhanced figure analysis failed for figure {index}: {e}")
            return None

    def _analyze_figure_lxml(self, fig_elem, index: int) -> Optional[Dict[str, Any]]:
        """Analyze a figure element using lxml with enhanced metadata extraction."""
        try:
            # Extract figure ID and basic attributes
            fig_id = fig_elem.get("id", f"figure_{index + 1}")
            fig_type = fig_elem.get("fig-type", "figure")
            position = fig_elem.get("position", "")
            orientation = fig_elem.get("orientation", "")

            # Extract label with enhanced parsing
            label_elems = fig_elem.xpath("./label")
            label = (
                self._get_lxml_text(label_elems[0])
                if label_elems
                else f"Figure {index + 1}"
            )

            # Extract enhanced caption with structure
            caption_data = self._extract_enhanced_caption_lxml(fig_elem)
            
            # Extract comprehensive graphic information
            graphics_data = self._extract_graphics_metadata_lxml(fig_elem)
            
            # Extract alternative representations
            alternatives = self._extract_alternatives_lxml(fig_elem)
            
            # Extract licensing and attribution
            licensing_info = self._extract_licensing_info_lxml(fig_elem)
            
            # Extract positioning and layout information
            layout_info = self._extract_layout_info_lxml(fig_elem)
            
            # Extract cross-references
            cross_refs = self._extract_figure_cross_refs_lxml(fig_elem)
            
            # Determine figure content type
            content_type = self._determine_figure_content_type(graphics_data, caption_data.get("text", ""))
            
            return {
                "id": fig_id,
                "label": label,
                "caption": caption_data.get("text", ""),
                "caption_structure": caption_data.get("structure", {}),
                "graphics": graphics_data.get("files", []),
                "graphics_metadata": graphics_data.get("metadata", {}),
                "alternatives": alternatives,
                "licensing": licensing_info,
                "layout": layout_info,
                "cross_references": cross_refs,
                "fig_type": fig_type,
                "position": position,
                "orientation": orientation,
                "content_type": content_type,
                "type": "figure",
                "extraction_method": "xml_structure_enhanced",
                "index": index + 1,
                "has_subfigures": len(fig_elem.xpath(".//fig")) > 0,
                "element_count": len(list(fig_elem.iter())),
            }

        except Exception as e:
            self.logger.warning(f"Enhanced figure analysis failed for figure {index}: {e}")
            return None

    def _analyze_figure_group_etree(
        self, fig_group_elem: ET.Element, index: int
    ) -> List[Dict[str, Any]]:
        """Analyze a figure group element using ElementTree."""
        figures = []
        try:
            group_id = fig_group_elem.get("id", f"fig_group_{index + 1}")
            group_label_elem = fig_group_elem.find("./label")
            group_label = (
                group_label_elem.text.strip()
                if group_label_elem is not None and group_label_elem.text
                else f"Figure Group {index + 1}"
            )
            
            # Find all figures within the group
            fig_elements = fig_group_elem.findall("./fig")
            for i, fig_elem in enumerate(fig_elements):
                figure_data = self._analyze_figure_etree(fig_elem, i)
                if figure_data:
                    # Add group information to individual figures
                    figure_data["group_id"] = group_id
                    figure_data["group_label"] = group_label
                    figure_data["is_part_of_group"] = True
                    figure_data["group_position"] = i + 1
                    figures.append(figure_data)
                    
        except Exception as e:
            self.logger.warning(f"Figure group analysis failed for group {index}: {e}")
            
        return figures

    def _analyze_figure_group_lxml(self, fig_group_elem, index: int) -> List[Dict[str, Any]]:
        """Analyze a figure group element using lxml."""
        figures = []
        try:
            group_id = fig_group_elem.get("id", f"fig_group_{index + 1}")
            group_label_elems = fig_group_elem.xpath("./label")
            group_label = (
                self._get_lxml_text(group_label_elems[0])
                if group_label_elems
                else f"Figure Group {index + 1}"
            )
            
            # Find all figures within the group
            fig_elements = fig_group_elem.xpath("./fig")
            for i, fig_elem in enumerate(fig_elements):
                figure_data = self._analyze_figure_lxml(fig_elem, i)
                if figure_data:
                    # Add group information to individual figures
                    figure_data["group_id"] = group_id
                    figure_data["group_label"] = group_label
                    figure_data["is_part_of_group"] = True
                    figure_data["group_position"] = i + 1
                    figures.append(figure_data)
                    
        except Exception as e:
            self.logger.warning(f"Figure group analysis failed for group {index}: {e}")
            
        return figures

    def _extract_enhanced_caption_etree(self, fig_elem: ET.Element) -> Dict[str, Any]:
        """Extract enhanced caption information using ElementTree."""
        caption_data = {"text": "", "structure": {}}
        
        try:
            caption_elem = fig_elem.find(".//caption")
            if caption_elem is not None:
                # Extract full caption text
                caption_parts = []
                for elem in caption_elem.iter():
                    if elem.text:
                        caption_parts.append(elem.text.strip())
                    if elem.tail:
                        caption_parts.append(elem.tail.strip())
                caption_text = " ".join(part for part in caption_parts if part)
                caption_data["text"] = self._clean_extracted_text(caption_text)
                
                # Extract structured caption elements
                title_elem = caption_elem.find("./title")
                if title_elem is not None and title_elem.text:
                    caption_data["structure"]["title"] = title_elem.text.strip()
                
                # Extract paragraphs
                p_elements = caption_elem.findall(".//p")
                if p_elements:
                    paragraphs = []
                    for p_elem in p_elements:
                        if p_elem.text:
                            paragraphs.append(p_elem.text.strip())
                    caption_data["structure"]["paragraphs"] = paragraphs
                
                # Check for multi-language content
                lang_attrs = [elem.get("xml:lang") or elem.get("lang") for elem in caption_elem.iter() if elem.get("xml:lang") or elem.get("lang")]
                if lang_attrs:
                    caption_data["structure"]["languages"] = list(set(lang_attrs))
                    
        except Exception as e:
            self.logger.warning(f"Enhanced caption extraction failed: {e}")
            
        return caption_data

    def _extract_enhanced_caption_lxml(self, fig_elem) -> Dict[str, Any]:
        """Extract enhanced caption information using lxml."""
        caption_data = {"text": "", "structure": {}}
        
        try:
            caption_elems = fig_elem.xpath(".//caption")
            if caption_elems:
                caption_elem = caption_elems[0]
                
                # Extract full caption text
                caption_text = self._get_lxml_text(caption_elem)
                caption_data["text"] = self._clean_extracted_text(caption_text)
                
                # Extract structured caption elements
                title_elems = caption_elem.xpath("./title")
                if title_elems:
                    caption_data["structure"]["title"] = self._get_lxml_text(title_elems[0])
                
                # Extract paragraphs
                p_elements = caption_elem.xpath(".//p")
                if p_elements:
                    paragraphs = [self._get_lxml_text(p_elem) for p_elem in p_elements if self._get_lxml_text(p_elem)]
                    caption_data["structure"]["paragraphs"] = paragraphs
                
                # Check for multi-language content
                lang_attrs = caption_elem.xpath(".//@xml:lang | .//@lang")
                if lang_attrs:
                    caption_data["structure"]["languages"] = list(set(lang_attrs))
                    
        except Exception as e:
            self.logger.warning(f"Enhanced caption extraction failed: {e}")
            
        return caption_data

    def _extract_graphics_metadata_etree(self, fig_elem: ET.Element) -> Dict[str, Any]:
        """Extract comprehensive graphics metadata using ElementTree."""
        graphics_data = {"files": [], "metadata": {}}
        
        try:
            graphic_elements = fig_elem.findall(".//graphic")
            
            for graphic in graphic_elements:
                graphic_info = {}
                
                # Extract file references
                href = graphic.get("{http://www.w3.org/1999/xlink}href", "")
                if not href:
                    href = graphic.get("href", "")
                if href:
                    graphic_info["href"] = href
                    graphic_info["filename"] = href.split("/")[-1] if "/" in href else href
                    
                    # Determine file format from extension
                    if "." in graphic_info["filename"]:
                        extension = graphic_info["filename"].split(".")[-1].lower()
                        graphic_info["format"] = extension
                        graphic_info["is_vector"] = extension in ["svg", "eps", "pdf"]
                        graphic_info["is_raster"] = extension in ["png", "jpg", "jpeg", "gif", "tiff", "bmp"]
                
                # Extract graphic attributes
                for attr in ["width", "height", "units", "resolution", "mime-type", "mimetype", "content-type"]:
                    value = graphic.get(attr)
                    if value:
                        graphic_info[attr.replace("-", "_")] = value
                
                # Extract alternative text
                alt_text = graphic.get("alt", "")
                if not alt_text:
                    # Look for alt-text element
                    alt_elem = graphic.find("../alt-text")
                    if alt_elem is not None and alt_elem.text:
                        alt_text = alt_elem.text.strip()
                if alt_text:
                    graphic_info["alt_text"] = alt_text
                
                if graphic_info:
                    graphics_data["files"].append(graphic_info)
            
            # Extract overall graphics metadata
            if graphics_data["files"]:
                graphics_data["metadata"]["total_graphics"] = len(graphics_data["files"])
                formats = [gf.get("format", "unknown") for gf in graphics_data["files"] if gf.get("format")]
                graphics_data["metadata"]["formats"] = list(set(formats))
                graphics_data["metadata"]["has_vector"] = any(gf.get("is_vector", False) for gf in graphics_data["files"])
                graphics_data["metadata"]["has_raster"] = any(gf.get("is_raster", False) for gf in graphics_data["files"])
                
        except Exception as e:
            self.logger.warning(f"Graphics metadata extraction failed: {e}")
            
        return graphics_data

    def _extract_graphics_metadata_lxml(self, fig_elem) -> Dict[str, Any]:
        """Extract comprehensive graphics metadata using lxml."""
        graphics_data = {"files": [], "metadata": {}}
        
        try:
            graphic_elements = fig_elem.xpath(".//graphic")
            
            for graphic in graphic_elements:
                graphic_info = {}
                
                # Extract file references
                href = graphic.get("{http://www.w3.org/1999/xlink}href", "")
                if not href:
                    href = graphic.get("href", "")
                if href:
                    graphic_info["href"] = href
                    graphic_info["filename"] = href.split("/")[-1] if "/" in href else href
                    
                    # Determine file format from extension
                    if "." in graphic_info["filename"]:
                        extension = graphic_info["filename"].split(".")[-1].lower()
                        graphic_info["format"] = extension
                        graphic_info["is_vector"] = extension in ["svg", "eps", "pdf"]
                        graphic_info["is_raster"] = extension in ["png", "jpg", "jpeg", "gif", "tiff", "bmp"]
                
                # Extract graphic attributes
                for attr in ["width", "height", "units", "resolution", "mime-type", "mimetype", "content-type"]:
                    value = graphic.get(attr)
                    if value:
                        graphic_info[attr.replace("-", "_")] = value
                
                # Extract alternative text
                alt_text = graphic.get("alt", "")
                if not alt_text:
                    # Look for alt-text element
                    alt_elems = fig_elem.xpath(".//alt-text")
                    if alt_elems:
                        alt_text = self._get_lxml_text(alt_elems[0])
                if alt_text:
                    graphic_info["alt_text"] = alt_text
                
                if graphic_info:
                    graphics_data["files"].append(graphic_info)
            
            # Extract overall graphics metadata
            if graphics_data["files"]:
                graphics_data["metadata"]["total_graphics"] = len(graphics_data["files"])
                formats = [gf.get("format", "unknown") for gf in graphics_data["files"] if gf.get("format")]
                graphics_data["metadata"]["formats"] = list(set(formats))
                graphics_data["metadata"]["has_vector"] = any(gf.get("is_vector", False) for gf in graphics_data["files"])
                graphics_data["metadata"]["has_raster"] = any(gf.get("is_raster", False) for gf in graphics_data["files"])
                
        except Exception as e:
            self.logger.warning(f"Graphics metadata extraction failed: {e}")
            
        return graphics_data

    def _extract_alternatives_etree(self, fig_elem: ET.Element) -> List[Dict[str, Any]]:
        """Extract alternative representations using ElementTree."""
        alternatives = []
        
        try:
            alt_elements = fig_elem.findall(".//alternatives")
            for alt_elem in alt_elements:
                # Find all alternative content within
                for child in alt_elem:
                    alt_info = {"type": child.tag}
                    
                    # Extract attributes
                    for attr, value in child.attrib.items():
                        alt_info[attr.replace("-", "_")] = value
                    
                    # Extract text content if available
                    if child.text and child.text.strip():
                        alt_info["content"] = child.text.strip()
                    
                    alternatives.append(alt_info)
                    
        except Exception as e:
            self.logger.warning(f"Alternatives extraction failed: {e}")
            
        return alternatives

    def _extract_alternatives_lxml(self, fig_elem) -> List[Dict[str, Any]]:
        """Extract alternative representations using lxml."""
        alternatives = []
        
        try:
            alt_elements = fig_elem.xpath(".//alternatives")
            for alt_elem in alt_elements:
                # Find all alternative content within
                for child in alt_elem:
                    alt_info = {"type": child.tag}
                    
                    # Extract attributes
                    for attr, value in child.attrib.items():
                        alt_info[attr.replace("-", "_")] = value
                    
                    # Extract text content if available
                    text_content = self._get_lxml_text(child)
                    if text_content:
                        alt_info["content"] = text_content
                    
                    alternatives.append(alt_info)
                    
        except Exception as e:
            self.logger.warning(f"Alternatives extraction failed: {e}")
            
        return alternatives

    def _extract_licensing_info_etree(self, fig_elem: ET.Element) -> Dict[str, Any]:
        """Extract licensing and attribution information using ElementTree."""
        licensing = {}
        
        try:
            # Look for permissions or copyright information
            permissions_elem = fig_elem.find(".//permissions")
            if permissions_elem is not None:
                copyright_elem = permissions_elem.find(".//copyright-statement")
                if copyright_elem is not None and copyright_elem.text:
                    licensing["copyright"] = copyright_elem.text.strip()
                
                license_elem = permissions_elem.find(".//license")
                if license_elem is not None:
                    license_type = license_elem.get("license-type", "")
                    if license_type:
                        licensing["license_type"] = license_type
                    
                    # Extract license text or reference
                    if license_elem.text and license_elem.text.strip():
                        licensing["license_text"] = license_elem.text.strip()
                    
                    # Look for license URI
                    license_uri = license_elem.get("{http://www.w3.org/1999/xlink}href")
                    if license_uri:
                        licensing["license_uri"] = license_uri
            
            # Look for attribution information
            attrib_elem = fig_elem.find(".//attrib")
            if attrib_elem is not None and attrib_elem.text:
                licensing["attribution"] = attrib_elem.text.strip()
                
        except Exception as e:
            self.logger.warning(f"Licensing info extraction failed: {e}")
            
        return licensing

    def _extract_licensing_info_lxml(self, fig_elem) -> Dict[str, Any]:
        """Extract licensing and attribution information using lxml."""
        licensing = {}
        
        try:
            # Look for permissions or copyright information
            permissions_elems = fig_elem.xpath(".//permissions")
            if permissions_elems:
                permissions_elem = permissions_elems[0]
                
                copyright_elems = permissions_elem.xpath(".//copyright-statement")
                if copyright_elems:
                    licensing["copyright"] = self._get_lxml_text(copyright_elems[0])
                
                license_elems = permissions_elem.xpath(".//license")
                if license_elems:
                    license_elem = license_elems[0]
                    license_type = license_elem.get("license-type", "")
                    if license_type:
                        licensing["license_type"] = license_type
                    
                    # Extract license text or reference
                    license_text = self._get_lxml_text(license_elem)
                    if license_text:
                        licensing["license_text"] = license_text
                    
                    # Look for license URI
                    license_uri = license_elem.get("{http://www.w3.org/1999/xlink}href")
                    if license_uri:
                        licensing["license_uri"] = license_uri
            
            # Look for attribution information
            attrib_elems = fig_elem.xpath(".//attrib")
            if attrib_elems:
                licensing["attribution"] = self._get_lxml_text(attrib_elems[0])
                
        except Exception as e:
            self.logger.warning(f"Licensing info extraction failed: {e}")
            
        return licensing

    def _extract_layout_info_etree(self, fig_elem: ET.Element) -> Dict[str, Any]:
        """Extract positioning and layout information using ElementTree."""
        layout = {}
        
        try:
            # Extract position attributes
            position = fig_elem.get("position", "")
            if position:
                layout["position"] = position
            
            orientation = fig_elem.get("orientation", "")
            if orientation:
                layout["orientation"] = orientation
            
            # Extract specific-use attribute (gives context about figure purpose)
            specific_use = fig_elem.get("specific-use", "")
            if specific_use:
                layout["specific_use"] = specific_use
            
            # Check for processing instructions or style information
            xml_style = fig_elem.get("style", "")
            if xml_style:
                layout["style"] = xml_style
                
        except Exception as e:
            self.logger.warning(f"Layout info extraction failed: {e}")
            
        return layout

    def _extract_layout_info_lxml(self, fig_elem) -> Dict[str, Any]:
        """Extract positioning and layout information using lxml."""
        layout = {}
        
        try:
            # Extract position attributes
            position = fig_elem.get("position", "")
            if position:
                layout["position"] = position
            
            orientation = fig_elem.get("orientation", "")
            if orientation:
                layout["orientation"] = orientation
            
            # Extract specific-use attribute (gives context about figure purpose)
            specific_use = fig_elem.get("specific-use", "")
            if specific_use:
                layout["specific_use"] = specific_use
            
            # Check for processing instructions or style information
            xml_style = fig_elem.get("style", "")
            if xml_style:
                layout["style"] = xml_style
                
        except Exception as e:
            self.logger.warning(f"Layout info extraction failed: {e}")
            
        return layout

    def _extract_figure_cross_refs_etree(self, fig_elem: ET.Element) -> List[str]:
        """Extract cross-references related to the figure using ElementTree."""
        cross_refs = []
        
        try:
            fig_id = fig_elem.get("id", "")
            if fig_id:
                # Note: This would typically require searching the entire document
                # for xref elements that reference this figure ID
                # For now, we extract any xref elements within the figure
                xref_elements = fig_elem.findall(".//xref")
                for xref in xref_elements:
                    ref_type = xref.get("ref-type", "")
                    rid = xref.get("rid", "")
                    if ref_type and rid:
                        cross_refs.append(f"{ref_type}:{rid}")
                        
        except Exception as e:
            self.logger.warning(f"Cross-references extraction failed: {e}")
            
        return cross_refs

    def _extract_figure_cross_refs_lxml(self, fig_elem) -> List[str]:
        """Extract cross-references related to the figure using lxml."""
        cross_refs = []
        
        try:
            fig_id = fig_elem.get("id", "")
            if fig_id:
                # Note: This would typically require searching the entire document
                # for xref elements that reference this figure ID
                # For now, we extract any xref elements within the figure
                xref_elements = fig_elem.xpath(".//xref")
                for xref in xref_elements:
                    ref_type = xref.get("ref-type", "")
                    rid = xref.get("rid", "")
                    if ref_type and rid:
                        cross_refs.append(f"{ref_type}:{rid}")
                        
        except Exception as e:
            self.logger.warning(f"Cross-references extraction failed: {e}")
            
        return cross_refs

    def _determine_figure_content_type(self, graphics_data: Dict[str, Any], caption_text: str) -> str:
        """Determine the content type of a figure based on graphics and caption."""
        try:
            # Check graphics metadata for format clues
            formats = graphics_data.get("metadata", {}).get("formats", [])
            
            # Vector formats suggest diagrams, charts, or illustrations
            if any(fmt in ["svg", "eps", "pdf"] for fmt in formats):
                # Analyze caption for specific content types
                caption_lower = caption_text.lower()
                
                if any(keyword in caption_lower for keyword in ["flow", "diagram", "schema", "workflow", "flowchart"]):
                    return "diagram"
                elif any(keyword in caption_lower for keyword in ["chart", "graph", "plot", "bar", "line", "scatter"]):
                    return "chart"
                elif any(keyword in caption_lower for keyword in ["model", "structure", "pathway", "network"]):
                    return "model"
                else:
                    return "illustration"
            
            # Raster formats suggest photos or screenshots
            elif any(fmt in ["png", "jpg", "jpeg", "gif", "tiff"] for fmt in formats):
                caption_lower = caption_text.lower()
                
                if any(keyword in caption_lower for keyword in ["microscopy", "microscope", "cell", "tissue", "specimen"]):
                    return "microscopy"
                elif any(keyword in caption_lower for keyword in ["photograph", "photo", "image"]):
                    return "photograph"
                elif any(keyword in caption_lower for keyword in ["screenshot", "interface", "software"]):
                    return "screenshot"
                else:
                    return "image"
            
            # Default based on caption analysis
            caption_lower = caption_text.lower()
            if any(keyword in caption_lower for keyword in ["table", "data", "results"]):
                return "data_visualization"
            elif any(keyword in caption_lower for keyword in ["protocol", "method", "procedure"]):
                return "methodology"
            else:
                return "unknown"
                
        except Exception:
            return "unknown"

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
        """Analyze a table-wrap element using ElementTree with enhanced structure parsing."""
        try:
            # Extract table ID and attributes
            table_id = table_elem.get("id", f"table_{index + 1}")
            table_type = table_elem.get("table-type", "table")
            position = table_elem.get("position", "")
            orientation = table_elem.get("orientation", "")

            # Extract label with enhanced parsing
            label_elem = table_elem.find("./label")
            label = (
                label_elem.text.strip()
                if label_elem is not None and label_elem.text
                else f"Table {index + 1}"
            )

            # Extract enhanced caption with structure
            caption_data = self._extract_enhanced_table_caption_etree(table_elem)
            
            # Extract comprehensive table structure
            table_structure = self._extract_table_structure_etree(table_elem)
            
            # Extract table content data
            table_content = self._extract_table_content_etree(table_elem)
            
            # Extract table notes and footnotes
            table_notes = self._extract_table_notes_etree(table_elem)
            
            # Extract licensing and attribution for tables
            licensing_info = self._extract_licensing_info_etree(table_elem)
            
            # Extract cross-references
            cross_refs = self._extract_table_cross_refs_etree(table_elem)
            
            # Analyze table data types and statistics
            content_analysis = self._analyze_table_content_types(table_content)
            
            return {
                "id": table_id,
                "label": label,
                "caption": caption_data.get("text", ""),
                "caption_structure": caption_data.get("structure", {}),
                "table_structure": table_structure,
                "table_content": table_content,
                "table_notes": table_notes,
                "licensing": licensing_info,
                "cross_references": cross_refs,
                "content_analysis": content_analysis,
                "table_type": table_type,
                "position": position,
                "orientation": orientation,
                "rows": table_structure.get("total_rows", 0),
                "columns": table_structure.get("total_columns", 0),
                "type": "table",
                "extraction_method": "xml_structure_enhanced",
                "index": index + 1,
                "has_header": table_structure.get("has_header", False),
                "has_footer": table_structure.get("has_footer", False),
                "element_count": len(list(table_elem.iter())),
            }

        except Exception as e:
            self.logger.warning(f"Enhanced table analysis failed for table {index}: {e}")
            return None

    def _analyze_table_lxml(self, table_elem, index: int) -> Optional[Dict[str, Any]]:
        """Analyze a table-wrap element using lxml with enhanced structure parsing."""
        try:
            # Extract table ID and attributes
            table_id = table_elem.get("id", f"table_{index + 1}")
            table_type = table_elem.get("table-type", "table")
            position = table_elem.get("position", "")
            orientation = table_elem.get("orientation", "")

            # Extract label with enhanced parsing
            label_elems = table_elem.xpath("./label")
            label = (
                self._get_lxml_text(label_elems[0])
                if label_elems
                else f"Table {index + 1}"
            )

            # Extract enhanced caption with structure
            caption_data = self._extract_enhanced_table_caption_lxml(table_elem)
            
            # Extract comprehensive table structure
            table_structure = self._extract_table_structure_lxml(table_elem)
            
            # Extract table content data
            table_content = self._extract_table_content_lxml(table_elem)
            
            # Extract table notes and footnotes
            table_notes = self._extract_table_notes_lxml(table_elem)
            
            # Extract licensing and attribution for tables
            licensing_info = self._extract_licensing_info_lxml(table_elem)
            
            # Extract cross-references
            cross_refs = self._extract_table_cross_refs_lxml(table_elem)
            
            # Analyze table data types and statistics
            content_analysis = self._analyze_table_content_types(table_content)
            
            return {
                "id": table_id,
                "label": label,
                "caption": caption_data.get("text", ""),
                "caption_structure": caption_data.get("structure", {}),
                "table_structure": table_structure,
                "table_content": table_content,
                "table_notes": table_notes,
                "licensing": licensing_info,
                "cross_references": cross_refs,
                "content_analysis": content_analysis,
                "table_type": table_type,
                "position": position,
                "orientation": orientation,
                "rows": table_structure.get("total_rows", 0),
                "columns": table_structure.get("total_columns", 0),
                "type": "table",
                "extraction_method": "xml_structure_enhanced",
                "index": index + 1,
                "has_header": table_structure.get("has_header", False),
                "has_footer": table_structure.get("has_footer", False),
                "element_count": len(list(table_elem.iter())),
            }

        except Exception as e:
            self.logger.warning(f"Enhanced table analysis failed for table {index}: {e}")
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

    def _extract_enhanced_table_caption_etree(self, table_elem: ET.Element) -> Dict[str, Any]:
        """Extract enhanced table caption information using ElementTree."""
        caption_data = {"text": "", "structure": {}}
        
        try:
            caption_elem = table_elem.find(".//caption")
            if caption_elem is not None:
                # Extract full caption text
                caption_parts = []
                for elem in caption_elem.iter():
                    if elem.text:
                        caption_parts.append(elem.text.strip())
                    if elem.tail:
                        caption_parts.append(elem.tail.strip())
                caption_text = " ".join(part for part in caption_parts if part)
                caption_data["text"] = self._clean_extracted_text(caption_text)
                
                # Extract structured caption elements
                title_elem = caption_elem.find("./title")
                if title_elem is not None and title_elem.text:
                    caption_data["structure"]["title"] = title_elem.text.strip()
                
                # Extract paragraphs
                p_elements = caption_elem.findall(".//p")
                if p_elements:
                    paragraphs = []
                    for p_elem in p_elements:
                        if p_elem.text:
                            paragraphs.append(p_elem.text.strip())
                    caption_data["structure"]["paragraphs"] = paragraphs
                
                # Check for multi-language content
                lang_attrs = [elem.get("xml:lang") or elem.get("lang") for elem in caption_elem.iter() if elem.get("xml:lang") or elem.get("lang")]
                if lang_attrs:
                    caption_data["structure"]["languages"] = list(set(lang_attrs))
                    
        except Exception as e:
            self.logger.warning(f"Enhanced table caption extraction failed: {e}")
            
        return caption_data

    def _extract_enhanced_table_caption_lxml(self, table_elem) -> Dict[str, Any]:
        """Extract enhanced table caption information using lxml."""
        caption_data = {"text": "", "structure": {}}
        
        try:
            caption_elems = table_elem.xpath(".//caption")
            if caption_elems:
                caption_elem = caption_elems[0]
                
                # Extract full caption text
                caption_text = self._get_lxml_text(caption_elem)
                caption_data["text"] = self._clean_extracted_text(caption_text)
                
                # Extract structured caption elements
                title_elems = caption_elem.xpath("./title")
                if title_elems:
                    caption_data["structure"]["title"] = self._get_lxml_text(title_elems[0])
                
                # Extract paragraphs
                p_elements = caption_elem.xpath(".//p")
                if p_elements:
                    paragraphs = [self._get_lxml_text(p_elem) for p_elem in p_elements if self._get_lxml_text(p_elem)]
                    caption_data["structure"]["paragraphs"] = paragraphs
                
                # Check for multi-language content
                lang_attrs = caption_elem.xpath(".//@xml:lang | .//@lang")
                if lang_attrs:
                    caption_data["structure"]["languages"] = list(set(lang_attrs))
                    
        except Exception as e:
            self.logger.warning(f"Enhanced table caption extraction failed: {e}")
            
        return caption_data

    def _extract_table_structure_etree(self, table_elem: ET.Element) -> Dict[str, Any]:
        """Extract comprehensive table structure using ElementTree."""
        structure = {
            "total_rows": 0,
            "total_columns": 0,
            "has_header": False,
            "has_footer": False,
            "sections": {},
            "cell_structure": []
        }
        
        try:
            table_inner = table_elem.find(".//table")
            if table_inner is not None:
                # Extract table sections
                thead = table_inner.find(".//thead")
                tbody = table_inner.find(".//tbody")
                tfoot = table_inner.find(".//tfoot")
                
                if thead is not None:
                    structure["has_header"] = True
                    header_rows = thead.findall(".//tr")
                    structure["sections"]["header"] = {
                        "rows": len(header_rows),
                        "content": self._extract_table_section_content_etree(thead)
                    }
                
                if tbody is not None:
                    body_rows = tbody.findall(".//tr")
                    structure["sections"]["body"] = {
                        "rows": len(body_rows),
                        "content": self._extract_table_section_content_etree(tbody)
                    }
                
                if tfoot is not None:
                    structure["has_footer"] = True
                    footer_rows = tfoot.findall(".//tr")
                    structure["sections"]["footer"] = {
                        "rows": len(footer_rows),
                        "content": self._extract_table_section_content_etree(tfoot)
                    }
                
                # Count total rows and columns
                all_rows = table_inner.findall(".//tr")
                structure["total_rows"] = len(all_rows)
                
                if all_rows:
                    # Find maximum column count across all rows
                    max_cols = 0
                    for row in all_rows:
                        cells = row.findall(".//td") + row.findall(".//th")
                        col_count = sum(int(cell.get("colspan", 1)) for cell in cells)
                        max_cols = max(max_cols, col_count)
                    structure["total_columns"] = max_cols
                
                # Extract detailed cell structure
                structure["cell_structure"] = self._extract_cell_structure_etree(table_inner)
                
        except Exception as e:
            self.logger.warning(f"Table structure extraction failed: {e}")
            
        return structure

    def _extract_table_structure_lxml(self, table_elem) -> Dict[str, Any]:
        """Extract comprehensive table structure using lxml."""
        structure = {
            "total_rows": 0,
            "total_columns": 0,
            "has_header": False,
            "has_footer": False,
            "sections": {},
            "cell_structure": []
        }
        
        try:
            table_inner = table_elem.xpath(".//table")
            if table_inner:
                table_inner = table_inner[0]
                
                # Extract table sections
                thead = table_inner.xpath(".//thead")
                tbody = table_inner.xpath(".//tbody")
                tfoot = table_inner.xpath(".//tfoot")
                
                if thead:
                    structure["has_header"] = True
                    header_rows = thead[0].xpath(".//tr")
                    structure["sections"]["header"] = {
                        "rows": len(header_rows),
                        "content": self._extract_table_section_content_lxml(thead[0])
                    }
                
                if tbody:
                    body_rows = tbody[0].xpath(".//tr")
                    structure["sections"]["body"] = {
                        "rows": len(body_rows),
                        "content": self._extract_table_section_content_lxml(tbody[0])
                    }
                
                if tfoot:
                    structure["has_footer"] = True
                    footer_rows = tfoot[0].xpath(".//tr")
                    structure["sections"]["footer"] = {
                        "rows": len(footer_rows),
                        "content": self._extract_table_section_content_lxml(tfoot[0])
                    }
                
                # Count total rows and columns
                all_rows = table_inner.xpath(".//tr")
                structure["total_rows"] = len(all_rows)
                
                if all_rows:
                    # Find maximum column count across all rows
                    max_cols = 0
                    for row in all_rows:
                        cells = row.xpath(".//td | .//th")
                        col_count = sum(int(cell.get("colspan", 1)) for cell in cells)
                        max_cols = max(max_cols, col_count)
                    structure["total_columns"] = max_cols
                
                # Extract detailed cell structure
                structure["cell_structure"] = self._extract_cell_structure_lxml(table_inner)
                
        except Exception as e:
            self.logger.warning(f"Table structure extraction failed: {e}")
            
        return structure

    def _extract_table_section_content_etree(self, section_elem: ET.Element) -> List[List[Dict[str, Any]]]:
        """Extract content from a table section (thead, tbody, tfoot) using ElementTree."""
        content = []
        
        try:
            rows = section_elem.findall(".//tr")
            for row in rows:
                row_data = []
                cells = row.findall(".//td") + row.findall(".//th")
                
                for cell in cells:
                    cell_data = {
                        "text": cell.text.strip() if cell.text else "",
                        "tag": cell.tag,
                        "colspan": int(cell.get("colspan", 1)),
                        "rowspan": int(cell.get("rowspan", 1)),
                        "align": cell.get("align", ""),
                        "valign": cell.get("valign", ""),
                        "style": cell.get("style", ""),
                    }
                    
                    # Extract any nested content
                    nested_text = []
                    for elem in cell.iter():
                        if elem.text:
                            nested_text.append(elem.text.strip())
                        if elem.tail:
                            nested_text.append(elem.tail.strip())
                    
                    if nested_text:
                        cell_data["full_text"] = " ".join(part for part in nested_text if part)
                    
                    row_data.append(cell_data)
                
                content.append(row_data)
                
        except Exception as e:
            self.logger.warning(f"Table section content extraction failed: {e}")
            
        return content

    def _extract_table_section_content_lxml(self, section_elem) -> List[List[Dict[str, Any]]]:
        """Extract content from a table section (thead, tbody, tfoot) using lxml."""
        content = []
        
        try:
            rows = section_elem.xpath(".//tr")
            for row in rows:
                row_data = []
                cells = row.xpath(".//td | .//th")
                
                for cell in cells:
                    cell_data = {
                        "text": self._get_lxml_text(cell),
                        "tag": cell.tag,
                        "colspan": int(cell.get("colspan", 1)),
                        "rowspan": int(cell.get("rowspan", 1)),
                        "align": cell.get("align", ""),
                        "valign": cell.get("valign", ""),
                        "style": cell.get("style", ""),
                        "full_text": self._get_lxml_text(cell)
                    }
                    
                    row_data.append(cell_data)
                
                content.append(row_data)
                
        except Exception as e:
            self.logger.warning(f"Table section content extraction failed: {e}")
            
        return content

    def _extract_cell_structure_etree(self, table_elem: ET.Element) -> List[Dict[str, Any]]:
        """Extract detailed cell structure information using ElementTree."""
        cell_structure = []
        
        try:
            rows = table_elem.findall(".//tr")
            for row_idx, row in enumerate(rows):
                cells = row.findall(".//td") + row.findall(".//th")
                
                for col_idx, cell in enumerate(cells):
                    cell_info = {
                        "row": row_idx,
                        "column": col_idx,
                        "tag": cell.tag,
                        "text": cell.text.strip() if cell.text else "",
                        "colspan": int(cell.get("colspan", 1)),
                        "rowspan": int(cell.get("rowspan", 1)),
                        "attributes": dict(cell.attrib),
                        "has_nested_elements": len(list(cell)) > 0
                    }
                    
                    # Check for mathematical content
                    if cell.find(".//math") is not None or cell.find(".//mml:math", {"mml": "http://www.w3.org/1998/Math/MathML"}) is not None:
                        cell_info["has_math"] = True
                    
                    # Check for inline graphics
                    if cell.find(".//graphic") is not None or cell.find(".//inline-graphic") is not None:
                        cell_info["has_graphics"] = True
                    
                    cell_structure.append(cell_info)
                    
        except Exception as e:
            self.logger.warning(f"Cell structure extraction failed: {e}")
            
        return cell_structure

    def _extract_cell_structure_lxml(self, table_elem) -> List[Dict[str, Any]]:
        """Extract detailed cell structure information using lxml."""
        cell_structure = []
        
        try:
            rows = table_elem.xpath(".//tr")
            for row_idx, row in enumerate(rows):
                cells = row.xpath(".//td | .//th")
                
                for col_idx, cell in enumerate(cells):
                    cell_info = {
                        "row": row_idx,
                        "column": col_idx,
                        "tag": cell.tag,
                        "text": self._get_lxml_text(cell),
                        "colspan": int(cell.get("colspan", 1)),
                        "rowspan": int(cell.get("rowspan", 1)),
                        "attributes": dict(cell.attrib),
                        "has_nested_elements": len(list(cell)) > 0
                    }
                    
                    # Check for mathematical content
                    if cell.xpath(".//math | .//mml:math", namespaces={"mml": "http://www.w3.org/1998/Math/MathML"}):
                        cell_info["has_math"] = True
                    
                    # Check for inline graphics
                    if cell.xpath(".//graphic | .//inline-graphic"):
                        cell_info["has_graphics"] = True
                    
                    cell_structure.append(cell_info)
                    
        except Exception as e:
            self.logger.warning(f"Cell structure extraction failed: {e}")
            
        return cell_structure

    def _extract_table_content_etree(self, table_elem: ET.Element) -> Dict[str, Any]:
        """Extract table content in structured format using ElementTree."""
        content = {"data": [], "metadata": {}}
        
        try:
            table_inner = table_elem.find(".//table")
            if table_inner is not None:
                rows = table_inner.findall(".//tr")
                
                for row in rows:
                    row_data = []
                    cells = row.findall(".//td") + row.findall(".//th")
                    
                    for cell in cells:
                        # Extract clean cell text
                        cell_text = ""
                        if cell.text:
                            cell_text = cell.text.strip()
                        
                        # Extract nested text content
                        nested_parts = []
                        for elem in cell.iter():
                            if elem.text:
                                nested_parts.append(elem.text.strip())
                            if elem.tail:
                                nested_parts.append(elem.tail.strip())
                        
                        full_text = " ".join(part for part in nested_parts if part)
                        
                        row_data.append({
                            "value": full_text or cell_text,
                            "raw_value": cell_text,
                            "type": self._infer_data_type(full_text or cell_text),
                            "is_header": cell.tag == "th"
                        })
                    
                    content["data"].append(row_data)
                
                # Extract metadata about the data
                if content["data"]:
                    content["metadata"]["total_cells"] = sum(len(row) for row in content["data"])
                    content["metadata"]["empty_cells"] = sum(
                        1 for row in content["data"] 
                        for cell in row if not cell["value"].strip()
                    )
                    
        except Exception as e:
            self.logger.warning(f"Table content extraction failed: {e}")
            
        return content

    def _extract_table_content_lxml(self, table_elem) -> Dict[str, Any]:
        """Extract table content in structured format using lxml."""
        content = {"data": [], "metadata": {}}
        
        try:
            table_inner = table_elem.xpath(".//table")
            if table_inner:
                table_inner = table_inner[0]
                rows = table_inner.xpath(".//tr")
                
                for row in rows:
                    row_data = []
                    cells = row.xpath(".//td | .//th")
                    
                    for cell in cells:
                        # Extract full cell text
                        full_text = self._get_lxml_text(cell)
                        
                        row_data.append({
                            "value": full_text,
                            "raw_value": cell.text.strip() if cell.text else "",
                            "type": self._infer_data_type(full_text),
                            "is_header": cell.tag == "th"
                        })
                    
                    content["data"].append(row_data)
                
                # Extract metadata about the data
                if content["data"]:
                    content["metadata"]["total_cells"] = sum(len(row) for row in content["data"])
                    content["metadata"]["empty_cells"] = sum(
                        1 for row in content["data"] 
                        for cell in row if not cell["value"].strip()
                    )
                    
        except Exception as e:
            self.logger.warning(f"Table content extraction failed: {e}")
            
        return content

    def _extract_table_notes_etree(self, table_elem: ET.Element) -> List[Dict[str, Any]]:
        """Extract table notes and footnotes using ElementTree."""
        notes = []
        
        try:
            # Look for table-wrap-foot (PMC table footnotes)
            table_foot = table_elem.find(".//table-wrap-foot")
            if table_foot is not None:
                # Extract footnotes
                fn_elements = table_foot.findall(".//fn")
                for fn in fn_elements:
                    note_data = {
                        "id": fn.get("id", ""),
                        "type": "footnote",
                        "symbol": fn.get("symbol", ""),
                        "text": ""
                    }
                    
                    # Extract footnote text
                    fn_parts = []
                    for elem in fn.iter():
                        if elem.text:
                            fn_parts.append(elem.text.strip())
                        if elem.tail:
                            fn_parts.append(elem.tail.strip())
                    
                    note_data["text"] = self._clean_extracted_text(" ".join(part for part in fn_parts if part))
                    
                    if note_data["text"]:
                        notes.append(note_data)
                
                # Extract general notes
                p_elements = table_foot.findall(".//p")
                for p in p_elements:
                    if p.text and p.text.strip():
                        notes.append({
                            "type": "note",
                            "text": self._clean_extracted_text(p.text.strip())
                        })
            
            # Look for general attrib elements (attribution/notes)
            attrib_elements = table_elem.findall(".//attrib")
            for attrib in attrib_elements:
                if attrib.text and attrib.text.strip():
                    notes.append({
                        "type": "attribution",
                        "text": self._clean_extracted_text(attrib.text.strip())
                    })
                    
        except Exception as e:
            self.logger.warning(f"Table notes extraction failed: {e}")
            
        return notes

    def _extract_table_notes_lxml(self, table_elem) -> List[Dict[str, Any]]:
        """Extract table notes and footnotes using lxml."""
        notes = []
        
        try:
            # Look for table-wrap-foot (PMC table footnotes)
            table_foot = table_elem.xpath(".//table-wrap-foot")
            if table_foot:
                table_foot = table_foot[0]
                
                # Extract footnotes
                fn_elements = table_foot.xpath(".//fn")
                for fn in fn_elements:
                    note_data = {
                        "id": fn.get("id", ""),
                        "type": "footnote",
                        "symbol": fn.get("symbol", ""),
                        "text": self._get_lxml_text(fn)
                    }
                    
                    if note_data["text"]:
                        notes.append(note_data)
                
                # Extract general notes
                p_elements = table_foot.xpath(".//p")
                for p in p_elements:
                    p_text = self._get_lxml_text(p)
                    if p_text:
                        notes.append({
                            "type": "note",
                            "text": self._clean_extracted_text(p_text)
                        })
            
            # Look for general attrib elements (attribution/notes)
            attrib_elements = table_elem.xpath(".//attrib")
            for attrib in attrib_elements:
                attrib_text = self._get_lxml_text(attrib)
                if attrib_text:
                    notes.append({
                        "type": "attribution",
                        "text": self._clean_extracted_text(attrib_text)
                    })
                    
        except Exception as e:
            self.logger.warning(f"Table notes extraction failed: {e}")
            
        return notes

    def _extract_table_cross_refs_etree(self, table_elem: ET.Element) -> List[str]:
        """Extract cross-references related to the table using ElementTree."""
        cross_refs = []
        
        try:
            table_id = table_elem.get("id", "")
            if table_id:
                # Extract any xref elements within the table
                xref_elements = table_elem.findall(".//xref")
                for xref in xref_elements:
                    ref_type = xref.get("ref-type", "")
                    rid = xref.get("rid", "")
                    if ref_type and rid:
                        cross_refs.append(f"{ref_type}:{rid}")
                        
        except Exception as e:
            self.logger.warning(f"Table cross-references extraction failed: {e}")
            
        return cross_refs

    def _extract_table_cross_refs_lxml(self, table_elem) -> List[str]:
        """Extract cross-references related to the table using lxml."""
        cross_refs = []
        
        try:
            table_id = table_elem.get("id", "")
            if table_id:
                # Extract any xref elements within the table
                xref_elements = table_elem.xpath(".//xref")
                for xref in xref_elements:
                    ref_type = xref.get("ref-type", "")
                    rid = xref.get("rid", "")
                    if ref_type and rid:
                        cross_refs.append(f"{ref_type}:{rid}")
                        
        except Exception as e:
            self.logger.warning(f"Table cross-references extraction failed: {e}")
            
        return cross_refs

    def _infer_data_type(self, value: str) -> str:
        """Infer the data type of a cell value."""
        if not value or not value.strip():
            return "empty"
        
        value = value.strip()
        
        # Check for numeric values
        try:
            float(value.replace(",", "").replace("%", "").replace("$", ""))
            if "%" in value:
                return "percentage"
            elif "$" in value or "€" in value or "£" in value:
                return "currency"
            elif "." in value or "," in value:
                return "decimal"
            else:
                return "integer"
        except ValueError:
            pass
        
        # Check for dates
        import re
        date_patterns = [
            r"\d{4}-\d{2}-\d{2}",  # YYYY-MM-DD
            r"\d{2}/\d{2}/\d{4}",  # MM/DD/YYYY
            r"\d{2}-\d{2}-\d{4}",  # MM-DD-YYYY
        ]
        
        for pattern in date_patterns:
            if re.match(pattern, value):
                return "date"
        
        # Check for yes/no or boolean-like values
        if value.lower() in ["yes", "no", "true", "false", "y", "n", "1", "0"]:
            return "boolean"
        
        # Check for scientific notation
        if re.match(r"^\d+\.?\d*[eE][+-]?\d+$", value):
            return "scientific"
        
        # Default to text
        return "text"

    def _analyze_table_content_types(self, table_content: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze table content to determine data types and extract statistics."""
        analysis = {
            "data_types": {},
            "statistics": {},
            "patterns": {}
        }
        
        try:
            data = table_content.get("data", [])
            if not data:
                return analysis
            
            # Analyze data types by column
            if data:
                num_cols = len(data[0]) if data[0] else 0
                
                for col_idx in range(num_cols):
                    col_types = []
                    col_values = []
                    
                    for row in data:
                        if col_idx < len(row):
                            cell = row[col_idx]
                            col_types.append(cell.get("type", "text"))
                            col_values.append(cell.get("value", ""))
                    
                    # Determine dominant type for this column
                    type_counts = {}
                    for dtype in col_types:
                        type_counts[dtype] = type_counts.get(dtype, 0) + 1
                    
                    dominant_type = max(type_counts, key=type_counts.get) if type_counts else "text"
                    
                    analysis["data_types"][f"column_{col_idx}"] = {
                        "dominant_type": dominant_type,
                        "type_distribution": type_counts,
                        "total_values": len(col_values),
                        "non_empty_values": len([v for v in col_values if v.strip()])
                    }
                    
                    # Extract basic statistics for numeric columns
                    if dominant_type in ["integer", "decimal", "percentage", "currency"]:
                        numeric_values = []
                        for value in col_values:
                            try:
                                # Clean numeric value
                                clean_val = value.replace(",", "").replace("%", "").replace("$", "").replace("€", "").replace("£", "")
                                if clean_val.strip():
                                    numeric_values.append(float(clean_val))
                            except ValueError:
                                continue
                        
                        if numeric_values:
                            analysis["statistics"][f"column_{col_idx}"] = {
                                "min": min(numeric_values),
                                "max": max(numeric_values),
                                "mean": sum(numeric_values) / len(numeric_values),
                                "count": len(numeric_values)
                            }
            
            # Analyze overall patterns
            total_cells = sum(len(row) for row in data)
            empty_cells = sum(1 for row in data for cell in row if not cell.get("value", "").strip())
            header_cells = sum(1 for row in data for cell in row if cell.get("is_header", False))
            
            analysis["patterns"] = {
                "total_cells": total_cells,
                "empty_cells": empty_cells,
                "header_cells": header_cells,
                "data_density": (total_cells - empty_cells) / total_cells if total_cells > 0 else 0,
                "has_headers": header_cells > 0
            }
            
        except Exception as e:
            self.logger.warning(f"Table content analysis failed: {e}")
            
        return analysis

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
