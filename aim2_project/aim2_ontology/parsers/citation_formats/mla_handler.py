"""
MLA Citation Format Handler

This module provides specialized parsing for Modern Language Association (MLA)
citation format. MLA style is commonly used in literature, language, and humanities.

Key MLA Format Characteristics:
- Author-page system: Author, First. "Title"
- Authors listed as "Last, First" (full first name preferred)
- No year in parentheses (year appears at end)
- Titles in quotation marks for articles, italics for books/journals
- No volume/issue numbers (typically)
- Page numbers at the end without "pp."
- Container system for complex sources

Example MLA Citations:
- Journal Article: Smith, John. "Title of Article." Journal Name, vol. 15, no. 3, 2023, pp. 123-145.
- Book: Johnson, Mary Beth. Title of Book. Publisher, 2022.
- Chapter: Davis, Robert C. "Chapter Title." Book Title, edited by Anne Editor, Publisher, 2021, pp. 45-67.
- Web Source: Brown, Sarah. "Article Title." Website Name, 15 Mar. 2023, www.example.com/article.

Classes:
    MLAHandler: Main handler class for parsing MLA format citations

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


class MLAHandler(BaseCitationHandler):
    """
    Handler for parsing MLA (Modern Language Association) format citations.

    This handler implements format-specific logic for detecting and parsing
    citations that follow MLA style guidelines.
    """

    def __init__(self):
        """Initialize the MLA citation handler."""
        super().__init__()
        self.citation_detector = CitationDetector()

    @property
    def format_name(self) -> str:
        """Get the name of this citation format."""
        return "MLA"

    @property
    def format_description(self) -> str:
        """Get a human-readable description of this citation format."""
        return "Modern Language Association (MLA) Style - Author-page format with quoted titles"

    def detect_format(self, citation_text: str) -> float:
        """
        Detect if the citation text matches MLA format.

        MLA format indicators:
        - No year in parentheses after author
        - Author format: Last, First (full first name)
        - Titles in quotation marks
        - Year appears near the end
        - Uses "vol." and "no." for volume/issue

        Args:
            citation_text: The citation text to analyze

        Returns:
            Confidence score (0.0 to 1.0) that this text matches MLA format
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

        # Check for MLA-specific patterns

        # 1. Author format: Last, First (strong MLA indicator)
        has_lastname_first = any(
            result.metadata.get("format") == "lastname_first" for result in authors
        )
        if has_lastname_first:
            confidence += 0.3

        # 2. Quoted titles (strong MLA indicator, but only without IEEE patterns)
        has_quoted_title = any(
            result.metadata.get("format") == "quoted" for result in titles
        )
        if has_quoted_title:
            # Reduce confidence if this also looks like IEEE
            ieee_indicators = ["vol.", "no.", " and "]
            ieee_count = sum(
                1 for indicator in ieee_indicators if indicator in text_lower
            )
            if ieee_count >= 2:
                confidence += 0.2  # Reduced confidence when IEEE patterns present
            else:
                confidence += 0.4

        # 3. No parenthetical year after author (MLA typically puts year at end)
        has_parenthetical_year = any(
            result.text.startswith("(") and result.text.endswith(")")
            for result in years
        )
        if not has_parenthetical_year and years:
            confidence += 0.2

        # 4. MLA-specific keywords
        mla_keywords = ["vol.", "no.", "pp.", "edited by", "ed."]
        keyword_count = sum(1 for keyword in mla_keywords if keyword in text_lower)
        confidence += min(keyword_count * 0.1, 0.3)

        # 5. Container-style punctuation (periods between elements)
        period_count = text.count(".")
        if period_count >= 3:  # MLA often has many periods
            confidence += 0.1

        # 6. Penalize for strong IEEE patterns
        strong_ieee_indicators = [" and ", '"', "vol.", "no."]
        strong_ieee_count = sum(
            1 for indicator in strong_ieee_indicators if indicator in text_lower
        )
        if strong_ieee_count >= 3:  # Strong IEEE pattern
            confidence -= 0.3  # Reduce MLA confidence

        return max(0.0, min(confidence, 1.0))

    def parse_citation(self, citation_text: str) -> ParseResult:
        """
        Parse an MLA format citation and extract metadata.

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
                metadata.authors = self._extract_mla_authors(author_result.text)
                metadata.set_confidence("authors", author_result.confidence)
                metadata.raw_text["authors"] = author_result.text
            else:
                warnings.append("No authors detected")

            # Extract title (prefer quoted titles in MLA)
            if titles:
                # Prefer quoted titles
                quoted_titles = [
                    t for t in titles if t.metadata.get("format") == "quoted"
                ]
                title_result = quoted_titles[0] if quoted_titles else titles[0]

                metadata.title = self._clean_mla_title(title_result.text)
                confidence = title_result.confidence
                if title_result.metadata.get("format") == "quoted":
                    confidence += 0.1  # Bonus for quoted titles in MLA
                metadata.set_confidence("title", confidence)
                metadata.raw_text["title"] = title_result.text
            else:
                warnings.append("No title detected")

            # Extract year (typically appears later in MLA)
            if years:
                # In MLA, prefer years that are NOT in parentheses
                non_paren_years = [
                    y
                    for y in years
                    if not (y.text.startswith("(") and y.text.endswith(")"))
                ]
                year_result = non_paren_years[0] if non_paren_years else years[0]

                year_text = year_result.text.strip("().,")
                try:
                    metadata.year = int(year_text)
                    metadata.set_confidence("year", year_result.confidence)
                    metadata.raw_text["year"] = year_result.text
                except ValueError:
                    errors.append(f"Invalid year format: {year_text}")
            else:
                warnings.append("No publication year detected")

            # Extract journal/container
            if journals:
                journal_result = journals[0]  # Take the first/best match
                metadata.journal = self._clean_text(journal_result.text)
                metadata.set_confidence("journal", journal_result.confidence)
                metadata.raw_text["journal"] = journal_result.text

            # Extract DOI (less common in MLA but still possible)
            if dois:
                doi_result = dois[0]  # Take the first/best match
                metadata.doi = self._clean_text(doi_result.text)
                metadata.set_confidence("doi", doi_result.confidence)
                metadata.raw_text["doi"] = doi_result.text

            # Extract pages (MLA format: pp. 123-145)
            if pages:
                page_result = pages[0]  # Take the first/best match
                metadata.pages = self._clean_mla_pages(page_result.text)
                metadata.set_confidence("pages", page_result.confidence)
                metadata.raw_text["pages"] = page_result.text

            # Extract volume and issue from MLA-specific patterns
            volume_issue = self._extract_mla_volume_issue(text)
            if volume_issue:
                volume, issue = volume_issue
                if volume:
                    metadata.volume = volume
                    metadata.set_confidence("volume", 0.8)
                if issue:
                    metadata.issue = issue
                    metadata.set_confidence("issue", 0.8)

            # Extract editor information
            editors = self._extract_mla_editors(text)
            if editors:
                metadata.editors = editors
                metadata.set_confidence("editors", 0.7)

            # Determine citation type
            metadata.citation_type = self._determine_mla_citation_type(metadata, text)

            # Validate extracted metadata
            validation_errors = self.validate_metadata(metadata)
            errors.extend(validation_errors)

        except Exception as e:
            self.logger.error(f"Error parsing MLA citation: {e}")
            errors.append(f"Parsing error: {str(e)}")

        return ParseResult(
            metadata=metadata,
            format_detected=self.format_name,
            format_confidence=format_confidence,
            parsing_errors=errors,
            parsing_warnings=warnings,
        )

    def _extract_mla_authors(self, author_text: str) -> List[str]:
        """
        Extract authors in MLA format.

        MLA format: Last, First, and Last2, First2

        Args:
            author_text: Raw author text

        Returns:
            List of cleaned author names
        """
        if not author_text:
            return []

        # Handle MLA-specific separators
        # MLA typically uses "and" between authors
        authors = []
        for author in author_text.split(" and "):
            author = author.strip().rstrip(",")
            if author:
                authors.append(author)

        return authors

    def _clean_mla_title(self, title_text: str) -> str:
        """
        Clean title text according to MLA conventions.

        MLA titles may be in quotes and use title case.

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

    def _clean_mla_pages(self, page_text: str) -> str:
        """
        Clean page text according to MLA conventions.

        MLA format: pp. 123-145 or 123-145

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

    def _extract_mla_volume_issue(
        self, text: str
    ) -> Optional[Tuple[Optional[str], Optional[str]]]:
        """
        Extract volume and issue numbers from MLA citation text.

        MLA format: vol. 15, no. 3

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

    def _extract_mla_editors(self, text: str) -> List[str]:
        """
        Extract editor information from MLA citation.

        MLA format: edited by John Smith, or Ed. Mary Johnson

        Args:
            text: Full citation text

        Returns:
            List of editor names
        """
        editors = []
        text_lower = text.lower()

        # Look for "edited by" pattern
        if "edited by" in text_lower:
            start_idx = text_lower.find("edited by") + len("edited by")
            remaining_text = text[start_idx:].strip()

            # Extract the next few words as editor name
            words = remaining_text.split()
            if len(words) >= 2:
                # Take first two words as editor name
                editor_name = f"{words[0]} {words[1]}".rstrip(",.")
                editors.append(editor_name)

        # Look for "ed." pattern
        elif "ed." in text_lower:
            words = text.split()
            for i, word in enumerate(words):
                if word.lower() == "ed.":
                    # Look for name before "ed."
                    if i > 1:
                        editor_name = f"{words[i-2]} {words[i-1]}".rstrip(",.")
                        editors.append(editor_name)
                    # Or look for name after "ed."
                    elif i + 2 < len(words):
                        editor_name = f"{words[i+1]} {words[i+2]}".rstrip(",.")
                        editors.append(editor_name)

        return editors

    def _determine_mla_citation_type(
        self, metadata: CitationMetadata, text: str
    ) -> CitationType:
        """
        Determine the type of citation based on MLA patterns.

        Args:
            metadata: Extracted metadata
            text: Original citation text

        Returns:
            Detected citation type
        """
        text_lower = text.lower()

        # Book chapter indicators (strong in MLA)
        if any(indicator in text_lower for indicator in ["edited by", "ed.", "pp."]):
            return CitationType.BOOK_CHAPTER

        # Journal article indicators
        if metadata.journal or "vol." in text_lower or "no." in text_lower:
            return CitationType.JOURNAL_ARTICLE

        # Web page indicators
        if any(indicator in text_lower for indicator in ["web.", "www", "http"]):
            return CitationType.WEB_PAGE

        # Book indicators
        if any(indicator in text_lower for indicator in ["publisher", "press"]):
            return CitationType.BOOK

        # Conference indicators
        if any(indicator in text_lower for indicator in ["proceedings", "conference"]):
            return CitationType.CONFERENCE_PAPER

        # Default based on structure
        if metadata.authors and metadata.title and metadata.year:
            # If has quoted title, likely an article
            if any('"' in str(raw) for raw in metadata.raw_text.values()):
                return CitationType.JOURNAL_ARTICLE
            # Otherwise likely a book
            return CitationType.BOOK

        return CitationType.UNKNOWN
