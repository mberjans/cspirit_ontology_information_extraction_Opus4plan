"""
APA Citation Format Handler

This module provides specialized parsing for American Psychological Association (APA)
citation format. APA style is commonly used in psychology, education, and social sciences.

Key APA Format Characteristics:
- Author-date system: Author, A. A. (Year)
- Authors listed as "Last, F. M." (last name, initials)
- Year in parentheses after author(s)
- Title in sentence case (only first word and proper nouns capitalized)
- Journal names in italics and title case
- Volume numbers in italics, issue numbers in parentheses
- DOI or URL for electronic sources

Example APA Citations:
- Journal Article: Smith, J. A. (2023). Title of the article. Journal Name, 15(3), 123-145. https://doi.org/10.1000/182
- Book: Johnson, M. B. (2022). Title of the book. Publisher Name.
- Chapter: Davis, R. C. (2021). Chapter title. In A. Editor (Ed.), Book title (pp. 45-67). Publisher.

Classes:
    APAHandler: Main handler class for parsing APA format citations

Dependencies:
    - typing: Type hints and annotations
    - logging: Logging functionality
    - .base_handler: Base citation handler classes
    - ..reference_patterns: Pattern matching utilities
"""

import logging
from typing import List, Optional, Tuple

from ..reference_patterns import CitationDetector
from .base_handler import (
    BaseCitationHandler,
    CitationMetadata,
    CitationType,
    ParseResult,
)

logger = logging.getLogger(__name__)


class APAHandler(BaseCitationHandler):
    """
    Handler for parsing APA (American Psychological Association) format citations.

    This handler implements format-specific logic for detecting and parsing
    citations that follow APA style guidelines.
    """

    def __init__(self):
        """Initialize the APA citation handler."""
        super().__init__()
        self.citation_detector = CitationDetector()

    @property
    def format_name(self) -> str:
        """Get the name of this citation format."""
        return "APA"

    @property
    def format_description(self) -> str:
        """Get a human-readable description of this citation format."""
        return "American Psychological Association (APA) Style - Author-date format with year in parentheses"

    def detect_format(self, citation_text: str) -> float:
        """
        Detect if the citation text matches APA format.

        APA format indicators:
        - Year in parentheses after author: (2023)
        - Author format: Last, F. M.
        - Sentence case titles
        - Volume in italics (indicated by formatting or position)

        Args:
            citation_text: The citation text to analyze

        Returns:
            Confidence score (0.0 to 1.0) that this text matches APA format
        """
        if not citation_text:
            return 0.0

        confidence = 0.0
        text = self.preprocess_text(citation_text)

        # Find pattern indicators
        authors = self.citation_detector.find_author_patterns(text)
        years = self.citation_detector.find_year_patterns(text)
        titles = self.citation_detector.find_title_patterns(text)

        # Check for APA-specific patterns

        # 1. Year in parentheses (strong APA indicator)
        has_parenthetical_year = any(
            result.text.startswith("(") and result.text.endswith(")")
            for result in years
        )
        if has_parenthetical_year:
            confidence += 0.4

        # 2. Author format: Last, First (common in APA)
        has_lastname_first = any(
            result.metadata.get("format") == "lastname_first" for result in authors
        )
        if has_lastname_first:
            confidence += 0.3

        # 3. Basic structure: Author (Year). Title.
        if authors and years and titles:
            # Check if year comes after author
            if authors and years:
                author_pos = authors[0].end_pos
                year_pos = years[0].start_pos
                if year_pos > author_pos:
                    confidence += 0.2

        # 4. Sentence case title (harder to detect, but look for patterns)
        if titles:
            title_text = titles[0].text
            # Simple heuristic: fewer capital letters suggests sentence case
            capital_ratio = sum(1 for c in title_text if c.isupper()) / len(title_text)
            if capital_ratio < 0.15:  # Fewer capitals suggests sentence case
                confidence += 0.1

        return min(confidence, 1.0)

    def parse_citation(self, citation_text: str) -> ParseResult:
        """
        Parse an APA format citation and extract metadata.

        Args:
            citation_text: The citation text to parse

        Returns:
            ParseResult containing extracted metadata and confidence scores
        """
        if not citation_text:
            return ParseResult(
                metadata=CitationMetadata(),
                format_detected=self.format_name,
                format_confidence=0.0,
                parsing_errors=["Empty citation text"],
            )

        text = self.preprocess_text(citation_text)
        metadata = CitationMetadata()
        errors = []
        warnings = []

        # Detect overall format confidence
        format_confidence = self.detect_format(text)

        try:
            # Extract all patterns
            authors = self.citation_detector.find_author_patterns(text)
            years = self.citation_detector.find_year_patterns(text)
            titles = self.citation_detector.find_title_patterns(text)
            journals = self.citation_detector.find_journal_patterns(text)
            dois = self.citation_detector.find_doi_patterns(text)
            pages = self.citation_detector.find_page_patterns(text)

            # Extract authors
            if authors:
                author_result = authors[0]  # Take the first/best match
                metadata.authors = self._extract_apa_authors(author_result.text)
                metadata.set_confidence("authors", author_result.confidence)
                metadata.raw_text["authors"] = author_result.text
            else:
                warnings.append("No authors detected")

            # Extract year
            if years:
                year_result = years[0]  # Take the first/best match
                year_text = year_result.text.strip("().,")  # Remove punctuation
                try:
                    metadata.year = int(year_text)
                    metadata.set_confidence("year", year_result.confidence)
                    metadata.raw_text["year"] = year_result.text
                except ValueError:
                    errors.append(f"Invalid year format: {year_text}")
            else:
                warnings.append("No publication year detected")

            # Extract title
            if titles:
                title_result = titles[0]  # Take the first/best match
                metadata.title = self._clean_apa_title(title_result.text)
                metadata.set_confidence("title", title_result.confidence)
                metadata.raw_text["title"] = title_result.text
            else:
                warnings.append("No title detected")

            # Extract journal
            if journals:
                journal_result = journals[0]  # Take the first/best match
                metadata.journal = self._clean_text(journal_result.text)
                metadata.set_confidence("journal", journal_result.confidence)
                metadata.raw_text["journal"] = journal_result.text

            # Extract DOI
            if dois:
                doi_result = dois[0]  # Take the first/best match
                metadata.doi = self._clean_text(doi_result.text)
                metadata.set_confidence("doi", doi_result.confidence)
                metadata.raw_text["doi"] = doi_result.text

            # Extract pages
            if pages:
                page_result = pages[0]  # Take the first/best match
                metadata.pages = self._clean_text(page_result.text)
                metadata.set_confidence("pages", page_result.confidence)
                metadata.raw_text["pages"] = page_result.text

            # Extract volume and issue from context
            volume_issue = self._extract_apa_volume_issue(text)
            if volume_issue:
                volume, issue = volume_issue
                if volume:
                    metadata.volume = volume
                    metadata.set_confidence("volume", 0.7)
                if issue:
                    metadata.issue = issue
                    metadata.set_confidence("issue", 0.7)

            # Determine citation type
            metadata.citation_type = self._determine_apa_citation_type(metadata, text)

            # Validate extracted metadata
            validation_errors = self.validate_metadata(metadata)
            errors.extend(validation_errors)

        except Exception as e:
            self.logger.error(f"Error parsing APA citation: {e}")
            errors.append(f"Parsing error: {str(e)}")

        return ParseResult(
            metadata=metadata,
            format_detected=self.format_name,
            format_confidence=format_confidence,
            parsing_errors=errors,
            parsing_warnings=warnings,
        )

    def _extract_apa_authors(self, author_text: str) -> List[str]:
        """
        Extract authors in APA format.

        APA format: Last, F. M., & Last2, F. M.

        Args:
            author_text: Raw author text

        Returns:
            List of cleaned author names
        """
        if not author_text:
            return []

        # Handle APA-specific separators
        # Replace " & " with " and " for consistent processing
        normalized_text = author_text.replace(" & ", " and ")

        # Split on " and " and clean up
        authors = []
        for author in normalized_text.split(" and "):
            author = author.strip().rstrip(",")
            if author:
                authors.append(author)

        return authors

    def _clean_apa_title(self, title_text: str) -> str:
        """
        Clean title text according to APA conventions.

        APA titles are in sentence case and may end with period.

        Args:
            title_text: Raw title text

        Returns:
            Cleaned title text
        """
        if not title_text:
            return ""

        # Remove quotes if present
        title = title_text.strip("\"'")

        # Remove trailing period
        title = title.rstrip(".")

        # Clean up whitespace
        title = " ".join(title.split())

        return title

    def _extract_apa_volume_issue(
        self, text: str
    ) -> Optional[Tuple[Optional[str], Optional[str]]]:
        """
        Extract volume and issue numbers from APA citation text.

        APA format: Volume(Issue) or Volume, Issue

        Args:
            text: Full citation text

        Returns:
            Tuple of (volume, issue) or None if not found
        """
        words = text.split()

        for i, word in enumerate(words):
            # Look for patterns like "15(3)" or "15(3),"
            if "(" in word and ")" in word:
                # Extract volume and issue
                try:
                    volume_part = word[: word.index("(")]
                    issue_part = word[word.index("(") + 1 : word.index(")")]

                    # Clean up
                    volume = volume_part.strip(",.")
                    issue = issue_part.strip(",.")

                    # Validate they're numeric
                    if volume.isdigit() or issue.isdigit():
                        return (
                            volume if volume.isdigit() else None,
                            issue if issue.isdigit() else None,
                        )
                except (ValueError, IndexError):
                    continue

        return None

    def _determine_apa_citation_type(
        self, metadata: CitationMetadata, text: str
    ) -> CitationType:
        """
        Determine the type of citation based on APA patterns.

        Args:
            metadata: Extracted metadata
            text: Original citation text

        Returns:
            Detected citation type
        """
        text_lower = text.lower()

        # Journal article indicators
        if metadata.journal or metadata.volume or metadata.issue:
            return CitationType.JOURNAL_ARTICLE

        # Book indicators
        if any(
            indicator in text_lower for indicator in ["publisher", "press", "books"]
        ):
            return CitationType.BOOK

        # Chapter indicators
        if any(indicator in text_lower for indicator in ["in ", "pp.", "pages"]):
            return CitationType.BOOK_CHAPTER

        # Conference indicators
        if any(
            indicator in text_lower
            for indicator in ["proceedings", "conference", "symposium"]
        ):
            return CitationType.CONFERENCE_PAPER

        # Thesis indicators
        if any(
            indicator in text_lower
            for indicator in ["dissertation", "thesis", "doctoral"]
        ):
            return CitationType.THESIS

        # Web page indicators
        if metadata.doi or any(
            indicator in text_lower for indicator in ["http", "www", "retrieved"]
        ):
            return CitationType.WEB_PAGE

        # Default to journal article if we have typical journal metadata
        if metadata.authors and metadata.year and metadata.title:
            return CitationType.JOURNAL_ARTICLE

        return CitationType.UNKNOWN
