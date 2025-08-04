"""
Base Citation Handler Module

This module provides the abstract base class for all citation format handlers.
It defines the common interface and shared functionality that all specific
format handlers (APA, MLA, IEEE, etc.) must implement.

Classes:
    BaseCitationHandler: Abstract base class for citation format handlers
    CitationMetadata: Data class for structured citation metadata
    ParseResult: Data class for parsing results with confidence scores

The base handler provides common utilities for text processing, validation,
and metadata extraction while requiring subclasses to implement format-specific
parsing logic.

Dependencies:
    - abc: Abstract base class functionality
    - dataclasses: Structured data classes
    - typing: Type hints and annotations
    - logging: Logging functionality
"""

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class CitationType(Enum):
    """Enumeration of different citation types."""

    JOURNAL_ARTICLE = "journal_article"
    BOOK = "book"
    BOOK_CHAPTER = "book_chapter"
    CONFERENCE_PAPER = "conference_paper"
    THESIS = "thesis"
    WEB_PAGE = "web_page"
    REPORT = "report"
    UNKNOWN = "unknown"


@dataclass
class CitationMetadata:
    """
    Structured metadata extracted from a citation.

    This class holds all the bibliographic information that can be
    extracted from a citation, along with confidence scores for each field.
    """

    # Core bibliographic fields
    authors: List[str] = field(default_factory=list)
    title: Optional[str] = None
    journal: Optional[str] = None
    year: Optional[int] = None
    volume: Optional[str] = None
    issue: Optional[str] = None
    pages: Optional[str] = None
    doi: Optional[str] = None
    url: Optional[str] = None

    # Extended fields
    publisher: Optional[str] = None
    editors: List[str] = field(default_factory=list)
    institution: Optional[str] = None
    address: Optional[str] = None
    isbn: Optional[str] = None
    issn: Optional[str] = None

    # Classification
    citation_type: CitationType = CitationType.UNKNOWN

    # Confidence scores (0.0 to 1.0) for each field
    confidence_scores: Dict[str, float] = field(default_factory=dict)

    # Raw extracted text for each field
    raw_text: Dict[str, str] = field(default_factory=dict)

    def get_confidence(self, field_name: str) -> float:
        """Get confidence score for a specific field."""
        return self.confidence_scores.get(field_name, 0.0)

    def set_confidence(self, field_name: str, confidence: float) -> None:
        """Set confidence score for a specific field."""
        self.confidence_scores[field_name] = max(0.0, min(1.0, confidence))

    def get_overall_confidence(self) -> float:
        """Calculate overall confidence as average of all field confidences."""
        if not self.confidence_scores:
            return 0.0
        return sum(self.confidence_scores.values()) / len(self.confidence_scores)

    def to_dict(self) -> Dict[str, Any]:
        """Convert metadata to dictionary representation."""
        result = {
            "authors": self.authors,
            "title": self.title,
            "journal": self.journal,
            "year": self.year,
            "volume": self.volume,
            "issue": self.issue,
            "pages": self.pages,
            "doi": self.doi,
            "url": self.url,
            "publisher": self.publisher,
            "editors": self.editors,
            "institution": self.institution,
            "address": self.address,
            "isbn": self.isbn,
            "issn": self.issn,
            "citation_type": self.citation_type.value,
            "confidence_scores": self.confidence_scores,
            "overall_confidence": self.get_overall_confidence(),
        }
        return {k: v for k, v in result.items() if v is not None}


@dataclass
class ParseResult:
    """
    Result of parsing a citation.

    Contains the extracted metadata, confidence scores, and additional
    information about the parsing process.
    """

    metadata: CitationMetadata
    format_detected: str
    format_confidence: float
    parsing_errors: List[str] = field(default_factory=list)
    parsing_warnings: List[str] = field(default_factory=list)

    @property
    def success(self) -> bool:
        """Check if parsing was successful (no errors)."""
        return len(self.parsing_errors) == 0

    @property
    def overall_confidence(self) -> float:
        """Get overall confidence combining format detection and metadata extraction."""
        return (self.format_confidence + self.metadata.get_overall_confidence()) / 2.0


class BaseCitationHandler(ABC):
    """
    Abstract base class for citation format handlers.

    This class defines the interface that all specific format handlers
    must implement, and provides common utilities for text processing
    and metadata extraction.
    """

    def __init__(self):
        """Initialize the citation handler."""
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")

    @property
    @abstractmethod
    def format_name(self) -> str:
        """Get the name of this citation format."""

    @property
    @abstractmethod
    def format_description(self) -> str:
        """Get a human-readable description of this citation format."""

    @abstractmethod
    def detect_format(self, citation_text: str) -> float:
        """
        Detect if the citation text matches this format.

        Args:
            citation_text: The citation text to analyze

        Returns:
            Confidence score (0.0 to 1.0) that this text matches the format
        """

    @abstractmethod
    def parse_citation(self, citation_text: str) -> ParseResult:
        """
        Parse a citation text and extract metadata.

        Args:
            citation_text: The citation text to parse

        Returns:
            ParseResult containing extracted metadata and confidence scores
        """

    def preprocess_text(self, text: str) -> str:
        """
        Preprocess citation text before parsing.

        Common preprocessing steps that can be overridden by subclasses.

        Args:
            text: Raw citation text

        Returns:
            Preprocessed text
        """
        if not text:
            return ""

        # Remove extra whitespace
        text = " ".join(text.split())

        # Remove common prefixes
        prefixes_to_remove = [
            "[1]",
            "[2]",
            "[3]",
            "[4]",
            "[5]",
            "[6]",
            "[7]",
            "[8]",
            "[9]",
        ]
        for prefix in prefixes_to_remove:
            if text.startswith(prefix):
                text = text[len(prefix) :].strip()
                break

        return text

    def validate_metadata(self, metadata: CitationMetadata) -> List[str]:
        """
        Validate extracted metadata and return list of validation errors.

        Args:
            metadata: Metadata to validate

        Returns:
            List of validation error messages
        """
        errors = []

        # Check required fields
        if not metadata.authors and not metadata.title:
            errors.append("Citation must have either authors or title")

        # Validate year
        if metadata.year is not None:
            current_year = 2024  # Could use datetime.now().year
            if metadata.year < 1800 or metadata.year > current_year + 5:
                errors.append(f"Invalid publication year: {metadata.year}")

        # Validate DOI format
        if metadata.doi and not self._is_valid_doi(metadata.doi):
            errors.append(f"Invalid DOI format: {metadata.doi}")

        # Validate URL format
        if metadata.url and not self._is_valid_url(metadata.url):
            errors.append(f"Invalid URL format: {metadata.url}")

        return errors

    def _is_valid_doi(self, doi: str) -> bool:
        """Check if DOI has valid format."""
        if not doi:
            return False
        doi = doi.strip()
        return doi.startswith("10.") and "/" in doi and len(doi) > 7

    def _is_valid_url(self, url: str) -> bool:
        """Check if URL has valid format."""
        if not url:
            return False
        url = url.strip().lower()
        return any(url.startswith(prefix) for prefix in ["http://", "https://", "www."])

    def _extract_author_names(self, author_text: str) -> List[str]:
        """
        Extract individual author names from author text.

        This is a basic implementation that can be overridden by subclasses
        for format-specific author parsing.

        Args:
            author_text: Text containing author names

        Returns:
            List of individual author names
        """
        if not author_text:
            return []

        # Split on common separators
        separators = [" and ", " & ", ";", ","]
        authors = [author_text]

        for separator in separators:
            new_authors = []
            for author in authors:
                new_authors.extend(part.strip() for part in author.split(separator))
            authors = new_authors

        # Filter out empty strings and clean up
        authors = [author.strip() for author in authors if author.strip()]

        return authors

    def _clean_text(self, text: str) -> str:
        """Clean and normalize text."""
        if not text:
            return ""

        # Remove extra punctuation and whitespace
        text = text.strip(" .,;:")
        text = " ".join(text.split())

        return text

    def _extract_numeric_value(self, text: str) -> Optional[int]:
        """Extract numeric value from text."""
        if not text:
            return None

        # Extract digits
        digits = "".join(char for char in text if char.isdigit())

        if digits:
            try:
                return int(digits)
            except ValueError:
                return None

        return None
