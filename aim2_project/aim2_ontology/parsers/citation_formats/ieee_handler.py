"""
IEEE Citation Format Handler

This module provides specialized parsing for Institute of Electrical and Electronics
Engineers (IEEE) citation format. IEEE style is commonly used in engineering,
computer science, and technical fields.

Key IEEE Format Characteristics:
- Numerical system: [1], [2], etc.
- Authors listed as First Last (no comma between names)
- Titles in quotes for articles, italics for books/journals
- Journal names often abbreviated
- Year appears after journal/publisher information
- Volume and issue: vol. X, no. Y, or just vol. X
- DOI often included

Example IEEE Citations:
- Journal Article: J. Smith and M. Johnson, "Title of article," Journal Name, vol. 15, no. 3, pp. 123-145, 2023.
- Book: M. B. Johnson, Title of Book. Publisher, 2022.
- Conference: R. Davis, "Paper title," in Proc. Conference Name, City, Country, 2023, pp. 45-52.
- Web: S. Brown, "Article title," Website Name. [Online]. Available: http://example.com. [Accessed: 15-Mar-2023].

Classes:
    IEEEHandler: Main handler class for parsing IEEE format citations

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


class IEEEHandler(BaseCitationHandler):
    """
    Handler for parsing IEEE (Institute of Electrical and Electronics Engineers) format citations.

    This handler implements format-specific logic for detecting and parsing
    citations that follow IEEE style guidelines.
    """

    def __init__(self):
        """Initialize the IEEE citation handler."""
        super().__init__()
        self.citation_detector = CitationDetector()

    @property
    def format_name(self) -> str:
        """Get the name of this citation format."""
        return "IEEE"

    @property
    def format_description(self) -> str:
        """Get a human-readable description of this citation format."""
        return "Institute of Electrical and Electronics Engineers (IEEE) Style - Numerical system with specific formatting"

    def detect_format(self, citation_text: str) -> float:
        """
        Detect if the citation text matches IEEE format.

        IEEE format indicators:
        - Author format: First Last (no comma)
        - Titles in quotes for articles
        - Journal names often abbreviated
        - Year typically appears after journal info
        - "vol." and "no." or "pp." indicators
        - May start with [1], [2], etc.

        Args:
            citation_text: The citation text to analyze

        Returns:
            Confidence score (0.0 to 1.0) that this text matches IEEE format
        """
        if not citation_text:
            return 0.0

        confidence = 0.0
        text = self.preprocess_text(citation_text)
        text_lower = text.lower()

        # Find pattern indicators
        authors = self.citation_detector.find_author_patterns(text)
        years = self.citation_detector.find_year_patterns(text)
        titles = self.citation_detector.find_title_patterns(text)

        # Check for IEEE-specific patterns

        # 1. Author format: First Last (no comma) - IEEE style
        has_first_last = any(
            result.metadata.get("format") == "first_last" for result in authors
        )
        if has_first_last:
            confidence += 0.3

        # 2. Quoted titles (common in IEEE for articles)
        has_quoted_title = any(
            result.metadata.get("format") == "quoted" for result in titles
        )
        if has_quoted_title:
            confidence += 0.3

        # 3. IEEE-specific keywords and patterns
        ieee_keywords = ["vol.", "pp.", "proc.", "in proc", "[online]", "available:"]
        keyword_count = sum(1 for keyword in ieee_keywords if keyword in text_lower)
        confidence += min(
            keyword_count * 0.2, 0.5
        )  # Increased weighting for IEEE keywords

        # 3a. Strong IEEE indicators - patterns more specific to IEEE
        if " and " in text and has_quoted_title and "vol." in text_lower:
            confidence += 0.2  # Strong IEEE pattern: "Author and Author2, "Title," Journal, vol. X"

        # 4. Year typically appears later in citation (not immediately after author)
        if authors and years:
            author_pos = authors[0].end_pos
            year_pos = years[0].start_pos
            # If year appears well after author (suggesting IEEE structure)
            if year_pos > author_pos + 50:  # Rough heuristic
                confidence += 0.2

        # 5. Numerical prefix [1], [2], etc.
        if text.strip().startswith("[") and "]" in text[:10]:
            try:
                bracket_content = text[1 : text.index("]")]
                if bracket_content.isdigit():
                    confidence += 0.2
            except (ValueError, IndexError):
                pass

        # 6. "and" between authors (IEEE style)
        if " and " in text:
            confidence += 0.1

        return min(confidence, 1.0)

    def parse_citation(self, citation_text: str) -> ParseResult:
        """
        Parse an IEEE format citation and extract metadata.

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

            # Extract authors (IEEE prefers First Last format)
            if authors:
                # Prefer first_last format for IEEE
                first_last_authors = [
                    a for a in authors if a.metadata.get("format") == "first_last"
                ]
                author_result = (
                    first_last_authors[0] if first_last_authors else authors[0]
                )

                metadata.authors = self._extract_ieee_authors(author_result.text)
                confidence = author_result.confidence
                if author_result.metadata.get("format") == "first_last":
                    confidence += 0.1  # Bonus for IEEE-style format
                metadata.set_confidence("authors", confidence)
                metadata.raw_text["authors"] = author_result.text
            else:
                warnings.append("No authors detected")

            # Extract title (prefer quoted titles in IEEE)
            if titles:
                # Prefer quoted titles for articles
                quoted_titles = [
                    t for t in titles if t.metadata.get("format") == "quoted"
                ]
                title_result = quoted_titles[0] if quoted_titles else titles[0]

                metadata.title = self._clean_ieee_title(title_result.text)
                confidence = title_result.confidence
                if title_result.metadata.get("format") == "quoted":
                    confidence += 0.1  # Bonus for quoted titles in IEEE
                metadata.set_confidence("title", confidence)
                metadata.raw_text["title"] = title_result.text
            else:
                warnings.append("No title detected")

            # Extract year (typically appears later in IEEE)
            if years:
                year_result = years[0]  # Take the first/best match
                year_text = year_result.text.strip("().,")
                try:
                    metadata.year = int(year_text)
                    metadata.set_confidence("year", year_result.confidence)
                    metadata.raw_text["year"] = year_result.text
                except ValueError:
                    errors.append(f"Invalid year format: {year_text}")
            else:
                warnings.append("No publication year detected")

            # Extract journal
            if journals:
                journal_result = journals[0]  # Take the first/best match
                metadata.journal = self._clean_text(journal_result.text)
                metadata.set_confidence("journal", journal_result.confidence)
                metadata.raw_text["journal"] = journal_result.text

            # Extract DOI (common in IEEE)
            if dois:
                doi_result = dois[0]  # Take the first/best match
                metadata.doi = self._clean_text(doi_result.text)
                metadata.set_confidence("doi", doi_result.confidence)
                metadata.raw_text["doi"] = doi_result.text

            # Extract pages (IEEE format: pp. 123-145)
            if pages:
                page_result = pages[0]  # Take the first/best match
                metadata.pages = self._clean_ieee_pages(page_result.text)
                metadata.set_confidence("pages", page_result.confidence)
                metadata.raw_text["pages"] = page_result.text

            # Extract volume and issue from IEEE-specific patterns
            volume_issue = self._extract_ieee_volume_issue(text)
            if volume_issue:
                volume, issue = volume_issue
                if volume:
                    metadata.volume = volume
                    metadata.set_confidence("volume", 0.8)
                if issue:
                    metadata.issue = issue
                    metadata.set_confidence("issue", 0.8)

            # Extract URL for online sources
            url = self._extract_ieee_url(text)
            if url:
                metadata.url = url
                metadata.set_confidence("url", 0.9)

            # Determine citation type
            metadata.citation_type = self._determine_ieee_citation_type(metadata, text)

            # Validate extracted metadata
            validation_errors = self.validate_metadata(metadata)
            errors.extend(validation_errors)

        except Exception as e:
            self.logger.error(f"Error parsing IEEE citation: {e}")
            errors.append(f"Parsing error: {str(e)}")

        return ParseResult(
            metadata=metadata,
            format_detected=self.format_name,
            format_confidence=format_confidence,
            parsing_errors=errors,
            parsing_warnings=warnings,
        )

    def _extract_ieee_authors(self, author_text: str) -> List[str]:
        """
        Extract authors in IEEE format.

        IEEE format: First Last and First2 Last2

        Args:
            author_text: Raw author text

        Returns:
            List of cleaned author names
        """
        if not author_text:
            return []

        # IEEE typically uses "and" between authors
        authors = []
        for author in author_text.split(" and "):
            author = author.strip().rstrip(",")
            if author:
                authors.append(author)

        return authors

    def _clean_ieee_title(self, title_text: str) -> str:
        """
        Clean title text according to IEEE conventions.

        IEEE titles may be in quotes and use sentence or title case.

        Args:
            title_text: Raw title text

        Returns:
            Cleaned title text
        """
        if not title_text:
            return ""

        # Remove quotes if present
        title = title_text.strip("\"'")

        # Remove trailing comma or period
        title = title.rstrip(",.")

        # Clean up whitespace
        title = " ".join(title.split())

        return title

    def _clean_ieee_pages(self, page_text: str) -> str:
        """
        Clean page text according to IEEE conventions.

        IEEE format: pp. 123-145 or 123-145

        Args:
            page_text: Raw page text

        Returns:
            Cleaned page text
        """
        if not page_text:
            return ""

        # Remove "pp." prefix if present
        cleaned = page_text.replace("pp.", "").strip()

        # Remove trailing punctuation
        cleaned = cleaned.rstrip(".,")

        return cleaned

    def _extract_ieee_volume_issue(
        self, text: str
    ) -> Optional[Tuple[Optional[str], Optional[str]]]:
        """
        Extract volume and issue numbers from IEEE citation text.

        IEEE format: vol. 15, no. 3 or vol. 15

        Args:
            text: Full citation text

        Returns:
            Tuple of (volume, issue) or None if not found
        """
        text.lower()
        volume = None
        issue = None

        # Look for "vol. X" pattern
        words = text.split()
        for i, word in enumerate(words):
            if word.lower() in ["vol.", "volume"]:
                if i + 1 < len(words):
                    vol_candidate = words[i + 1].rstrip(",.")
                    if vol_candidate.isdigit():
                        volume = vol_candidate

            if word.lower() in ["no.", "number"]:
                if i + 1 < len(words):
                    issue_candidate = words[i + 1].rstrip(",.")
                    if issue_candidate.isdigit():
                        issue = issue_candidate

        if volume or issue:
            return (volume, issue)

        return None

    def _extract_ieee_url(self, text: str) -> Optional[str]:
        """
        Extract URL from IEEE citation.

        IEEE online format: Available: http://example.com

        Args:
            text: Full citation text

        Returns:
            Extracted URL or None if not found
        """
        text_lower = text.lower()

        # Look for "Available:" pattern
        if "available:" in text_lower:
            start_idx = text_lower.find("available:") + len("available:")
            remaining_text = text[start_idx:].strip()

            # Extract URL (until space or period)
            words = remaining_text.split()
            if words:
                url_candidate = words[0].rstrip(".,")
                if self._is_valid_url(url_candidate):
                    return url_candidate

        # Look for direct URL patterns
        words = text.split()
        for word in words:
            clean_word = word.rstrip(".,")
            if self._is_valid_url(clean_word):
                return clean_word

        return None

    def _determine_ieee_citation_type(
        self, metadata: CitationMetadata, text: str
    ) -> CitationType:
        """
        Determine the type of citation based on IEEE patterns.

        Args:
            metadata: Extracted metadata
            text: Original citation text

        Returns:
            Detected citation type
        """
        text_lower = text.lower()

        # Conference paper indicators (strong in IEEE)
        if any(
            indicator in text_lower
            for indicator in ["proc.", "proceedings", "conference", "in proc"]
        ):
            return CitationType.CONFERENCE_PAPER

        # Journal article indicators
        if metadata.journal or metadata.volume or "vol." in text_lower:
            return CitationType.JOURNAL_ARTICLE

        # Web page/online indicators
        if metadata.url or any(
            indicator in text_lower
            for indicator in ["[online]", "available:", "accessed:"]
        ):
            return CitationType.WEB_PAGE

        # Book indicators
        if any(
            indicator in text_lower for indicator in ["publisher", "press", "edition"]
        ):
            return CitationType.BOOK

        # Thesis indicators
        if any(
            indicator in text_lower
            for indicator in ["thesis", "dissertation", "phd", "master"]
        ):
            return CitationType.THESIS

        # Report indicators
        if any(
            indicator in text_lower
            for indicator in ["technical report", "tech. rep.", "report"]
        ):
            return CitationType.REPORT

        # Default based on structure
        if metadata.authors and metadata.title and metadata.year:
            # If has quoted title and journal-like structure, likely article
            if metadata.journal or any(
                '"' in str(raw) for raw in metadata.raw_text.values()
            ):
                return CitationType.JOURNAL_ARTICLE
            # Otherwise could be book or conference paper
            return CitationType.UNKNOWN

        return CitationType.UNKNOWN
