"""
Reference Pattern Matching Module

This module provides pattern matching utilities for parsing bibliographic references
without using regular expressions. Instead, it uses special pattern matching functions
that analyze text structure, identify key components, and extract citation metadata.

The pattern matching system is designed to:
- Identify different citation formats (APA, MLA, IEEE, etc.)
- Extract structured metadata from references
- Handle variations in formatting and punctuation
- Provide confidence scores for extracted information
- Work without regular expressions for better maintainability

Classes:
    PatternMatchResult: Data class for pattern matching results
    TextAnalyzer: Utility class for text analysis and structure detection
    CitationDetector: Main class for detecting citation patterns and formats

Functions:
    find_author_patterns: Identify author name patterns in text
    find_year_patterns: Identify publication year patterns
    find_title_patterns: Identify title patterns with various formatting
    find_journal_patterns: Identify journal name patterns
    find_doi_patterns: Identify DOI patterns
    find_page_patterns: Identify page range patterns
    detect_citation_format: Auto-detect citation format with confidence score

Dependencies:
    - typing: Type hints and annotations
    - dataclasses: For structured data classes
    - datetime: For date validation
    - logging: For logging functionality
"""

import logging
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Tuple

logger = logging.getLogger(__name__)


class PatternType(Enum):
    """Enumeration of different pattern types in citations."""

    AUTHOR = "author"
    YEAR = "year"
    TITLE = "title"
    JOURNAL = "journal"
    VOLUME = "volume"
    ISSUE = "issue"
    PAGES = "pages"
    DOI = "doi"
    URL = "url"
    PUBLISHER = "publisher"
    EDITOR = "editor"


@dataclass
class PatternMatchResult:
    """
    Result of a pattern matching operation.

    Attributes:
        text: The matched text
        start_pos: Starting position in the original text
        end_pos: Ending position in the original text
        confidence: Confidence score (0.0 to 1.0)
        pattern_type: Type of pattern matched
        metadata: Additional metadata about the match
    """

    text: str
    start_pos: int
    end_pos: int
    confidence: float
    pattern_type: PatternType
    metadata: Dict[str, Any]


class TextAnalyzer:
    """
    Utility class for analyzing text structure without regular expressions.

    This class provides methods for identifying structural patterns in text
    such as capitalization, punctuation, numeric patterns, and formatting.
    """

    @staticmethod
    def is_capitalized_word(word: str) -> bool:
        """Check if a word follows standard capitalization rules."""
        if not word:
            return False

        # Handle abbreviations like "J." or "Jr."
        if word.endswith(".") and len(word) >= 2:
            # Check if it's a single letter abbreviation or a proper abbreviation
            base_word = word[:-1]
            if len(base_word) == 1:
                return base_word.isupper()
            else:
                return base_word[0].isupper() and (
                    base_word[1:].islower() or base_word[1:].isupper()
                )

        # Regular word capitalization check
        return word[0].isupper() and (
            len(word) == 1 or word[1:].islower() or word[1:].isupper()
        )

    @staticmethod
    def is_numeric_range(text: str) -> bool:
        """Check if text represents a numeric range (e.g., '123-456', '45-67')."""
        if not text or "-" not in text:
            return False

        parts = text.split("-")
        if len(parts) != 2:
            return False

        return all(part.strip().isdigit() for part in parts)

    @staticmethod
    def is_year_format(text: str) -> bool:
        """Check if text represents a valid publication year."""
        if not text or not text.isdigit():
            return False

        year = int(text)
        current_year = datetime.now().year
        return 1800 <= year <= current_year + 5

    @staticmethod
    def is_doi_format(text: str) -> bool:
        """Check if text follows DOI format patterns."""
        if not text:
            return False

        text = text.strip()
        # DOI typically starts with "10." and contains "/"
        return text.startswith("10.") and "/" in text and len(text) > 7

    @staticmethod
    def is_url_format(text: str) -> bool:
        """Check if text appears to be a URL."""
        if not text:
            return False

        text = text.strip().lower()
        return any(
            text.startswith(prefix) for prefix in ["http://", "https://", "www."]
        )

    @staticmethod
    def count_capital_letters(text: str) -> int:
        """Count the number of capital letters in text."""
        return sum(1 for char in text if char.isupper())

    @staticmethod
    def find_quoted_text(
        text: str, quote_chars: str = "\"'"
    ) -> List[Tuple[str, int, int]]:
        """Find text within quotes and return with positions."""
        results = []
        i = 0
        while i < len(text):
            if text[i] in quote_chars:
                quote_char = text[i]
                start = i + 1
                end = text.find(quote_char, start)
                if end != -1:
                    results.append((text[start:end], start - 1, end + 1))
                    i = end + 1
                else:
                    i += 1
            else:
                i += 1
        return results

    @staticmethod
    def find_parenthetical(text: str) -> List[Tuple[str, int, int]]:
        """Find text within parentheses and return with positions."""
        results = []
        i = 0
        while i < len(text):
            if text[i] == "(":
                start = i + 1
                level = 1
                j = i + 1
                while j < len(text) and level > 0:
                    if text[j] == "(":
                        level += 1
                    elif text[j] == ")":
                        level -= 1
                    j += 1
                if level == 0:
                    results.append((text[start : j - 1], i, j))
                i = j
            else:
                i += 1
        return results


class CitationDetector:
    """
    Main class for detecting citation patterns and formats.

    This class provides methods for identifying different citation formats
    and extracting structured metadata from reference strings.
    """

    def __init__(self):
        """Initialize the citation detector."""
        self.text_analyzer = TextAnalyzer()

    def find_author_patterns(self, text: str) -> List[PatternMatchResult]:
        """
        Find author name patterns in text.

        Identifies various author name formats including:
        - Last, First
        - First Last
        - Last, F.
        - F. Last
        - Multiple authors with various separators

        Args:
            text: Text to analyze

        Returns:
            List of PatternMatchResult objects for author patterns
        """
        results = []
        words = text.split()

        # Look for author patterns at the beginning of the text
        i = 0
        while i < min(15, len(words)):  # Check first 15 words
            word = words[i]

            # Check for lastname, firstname pattern (APA style: Smith, J. A.)
            if word.endswith(","):
                lastname = word[:-1]
                if self.text_analyzer.is_capitalized_word(lastname) and i + 1 < len(
                    words
                ):
                    # Collect first name and initials
                    author_parts = [word]  # Start with lastname and comma
                    j = i + 1

                    # Collect consecutive capitalized words (first names, initials)
                    while (
                        j < len(words)
                        and self.text_analyzer.is_capitalized_word(words[j])
                        and not words[j].startswith("(")
                    ):  # Stop at year in parentheses
                        author_parts.append(words[j])
                        j += 1

                        # Stop if we hit a period followed by a non-capitalized word
                        if (
                            words[j - 1].endswith(".")
                            and j < len(words)
                            and not self.text_analyzer.is_capitalized_word(words[j])
                        ):
                            break

                    if (
                        len(author_parts) > 1
                    ):  # Must have at least lastname + first name/initial
                        author_text = " ".join(author_parts)
                        start_pos = len(" ".join(words[:i]))
                        if start_pos > 0:
                            start_pos += 1
                        end_pos = start_pos + len(author_text)

                        results.append(
                            PatternMatchResult(
                                text=author_text,
                                start_pos=start_pos,
                                end_pos=end_pos,
                                confidence=0.9,
                                pattern_type=PatternType.AUTHOR,
                                metadata={"format": "lastname_first"},
                            )
                        )

                        i = j  # Skip processed words
                        continue

            # Check for firstname lastname pattern (IEEE style: J. A. Smith)
            elif self.text_analyzer.is_capitalized_word(word) and i + 1 < len(words):
                # Look ahead to see if this could be an author pattern
                author_parts = [word]
                j = i + 1

                # Collect consecutive parts that could be author names
                while (
                    j < len(words)
                    and self.text_analyzer.is_capitalized_word(words[j])
                    and not words[j].startswith("(")
                    and not words[j].startswith('"')  # Stop at year
                ):  # Stop at quoted title
                    author_parts.append(words[j])
                    j += 1

                    # If we see a comma after this word, this might be end of author
                    if words[j - 1].endswith(","):
                        break

                # Check if this looks like an author pattern
                if (
                    len(author_parts) >= 2
                    and j < len(words)  # At least 2 parts
                    and (
                        words[j - 1].endswith(",")
                        or words[j].startswith('"')  # Ends with comma
                        or words[j].startswith("(")  # Followed by quoted title
                    )
                    and i == 0  # Followed by year
                ):  # Only at the very beginning of text for first_last format
                    author_text = " ".join(author_parts)
                    start_pos = len(" ".join(words[:i]))
                    if start_pos > 0:
                        start_pos += 1
                    end_pos = start_pos + len(author_text)

                    results.append(
                        PatternMatchResult(
                            text=author_text,
                            start_pos=start_pos,
                            end_pos=end_pos,
                            confidence=0.8,
                            pattern_type=PatternType.AUTHOR,
                            metadata={"format": "first_last"},
                        )
                    )

                    i = j  # Skip processed words
                    continue

            i += 1

        return results

    def find_year_patterns(self, text: str) -> List[PatternMatchResult]:
        """
        Find publication year patterns in text.

        Args:
            text: Text to analyze

        Returns:
            List of PatternMatchResult objects for year patterns
        """
        results = []
        words = text.split()

        for i, word in enumerate(words):
            # Remove punctuation for year check
            clean_word = word.strip("().,;:")

            if self.text_analyzer.is_year_format(clean_word):
                start_pos = len(" ".join(words[:i]))
                if start_pos > 0:
                    start_pos += 1
                end_pos = start_pos + len(word)

                # Higher confidence if in parentheses
                confidence = 0.9 if word.startswith("(") and word.endswith(")") else 0.7

                results.append(
                    PatternMatchResult(
                        text=word,
                        start_pos=start_pos,
                        end_pos=end_pos,
                        confidence=confidence,
                        pattern_type=PatternType.YEAR,
                        metadata={"year": int(clean_word)},
                    )
                )

        return results

    def find_title_patterns(self, text: str) -> List[PatternMatchResult]:
        """
        Find title patterns in text.

        Identifies titles based on:
        - Quoted text
        - Capitalization patterns
        - Position after author/year

        Args:
            text: Text to analyze

        Returns:
            List of PatternMatchResult objects for title patterns
        """
        results = []

        # Look for quoted titles
        quoted_texts = self.text_analyzer.find_quoted_text(text)
        for quoted_text, start, end in quoted_texts:
            if len(quoted_text) > 5:  # Reasonable title length
                results.append(
                    PatternMatchResult(
                        text=quoted_text,
                        start_pos=start,
                        end_pos=end,
                        confidence=0.9,
                        pattern_type=PatternType.TITLE,
                        metadata={"format": "quoted"},
                    )
                )

        # Look for titles after parenthetical years (APA style)
        parenthetical_years = self.text_analyzer.find_parenthetical(text)
        for year_text, year_start, year_end in parenthetical_years:
            if self.text_analyzer.is_year_format(year_text.strip()):
                # Look for text after the year that could be a title
                remaining_text = text[year_end:].strip()
                if remaining_text.startswith("."):
                    remaining_text = remaining_text[1:].strip()

                # Find the title part (until next period or comma, excluding journal info)
                words = remaining_text.split()
                title_words = []
                for word in words:
                    if word.endswith(".") and len(title_words) > 2:
                        title_words.append(word[:-1])  # Add word without period
                        break
                    title_words.append(word)
                    if len(title_words) > 15:  # Reasonable title length limit
                        break

                if len(title_words) >= 3:  # Minimum title length
                    title_text = " ".join(title_words)
                    results.append(
                        PatternMatchResult(
                            text=title_text,
                            start_pos=year_end,
                            end_pos=year_end + len(title_text),
                            confidence=0.8,
                            pattern_type=PatternType.TITLE,
                            metadata={"format": "after_year"},
                        )
                    )

        # Look for title-like capitalization patterns, but avoid author names
        words = text.split()
        for i in range(len(words) - 2):  # Need at least 3 words for a title
            # Skip if this looks like an author pattern (has comma in first few words)
            if any(
                "," in words[k] for k in range(max(0, i - 3), min(i + 3, len(words)))
            ):
                continue

            if self.text_analyzer.is_capitalized_word(
                words[i]
            ) and self.text_analyzer.is_capitalized_word(words[i + 1]):
                # Extend to find full title
                title_words = [words[i], words[i + 1]]
                j = i + 2
                while (
                    j < len(words)
                    and (
                        self.text_analyzer.is_capitalized_word(words[j])
                        or words[j].lower()
                        in ["and", "or", "the", "a", "an", "of", "in", "on", "for"]
                    )
                    and not words[j - 1].endswith(".")
                ):  # Stop at sentence boundary
                    title_words.append(words[j])
                    j += 1

                if len(title_words) >= 3:  # Reasonable title length
                    title_text = " ".join(title_words)
                    start_pos = len(" ".join(words[:i]))
                    if start_pos > 0:
                        start_pos += 1
                    end_pos = start_pos + len(title_text)

                    results.append(
                        PatternMatchResult(
                            text=title_text,
                            start_pos=start_pos,
                            end_pos=end_pos,
                            confidence=0.6,
                            pattern_type=PatternType.TITLE,
                            metadata={"format": "capitalized"},
                        )
                    )

        return results

    def find_journal_patterns(self, text: str) -> List[PatternMatchResult]:
        """
        Find journal name patterns in text.

        Args:
            text: Text to analyze

        Returns:
            List of PatternMatchResult objects for journal patterns
        """
        results = []
        words = text.split()

        # Look for italicized or emphasized journal names (basic heuristic)
        # In plain text, journals are often all caps or title case
        for i in range(len(words) - 1):
            if (
                self.text_analyzer.is_capitalized_word(words[i])
                and self.text_analyzer.count_capital_letters(words[i]) > 1
            ):
                # Check if this could be a journal name
                journal_words = [words[i]]
                j = i + 1
                while j < len(words) and (
                    self.text_analyzer.is_capitalized_word(words[j])
                    or words[j].lower() in ["of", "and", "for", "in"]
                ):
                    journal_words.append(words[j])
                    j += 1

                if len(journal_words) >= 1:
                    journal_text = " ".join(journal_words)
                    start_pos = len(" ".join(words[:i]))
                    if start_pos > 0:
                        start_pos += 1
                    end_pos = start_pos + len(journal_text)

                    results.append(
                        PatternMatchResult(
                            text=journal_text,
                            start_pos=start_pos,
                            end_pos=end_pos,
                            confidence=0.5,
                            pattern_type=PatternType.JOURNAL,
                            metadata={"format": "title_case"},
                        )
                    )

        return results

    def find_doi_patterns(self, text: str) -> List[PatternMatchResult]:
        """
        Find DOI patterns in text.

        Args:
            text: Text to analyze

        Returns:
            List of PatternMatchResult objects for DOI patterns
        """
        results = []
        words = text.split()

        for i, word in enumerate(words):
            clean_word = word.strip(".,;:")
            if self.text_analyzer.is_doi_format(clean_word):
                start_pos = len(" ".join(words[:i]))
                if start_pos > 0:
                    start_pos += 1
                end_pos = start_pos + len(word)

                results.append(
                    PatternMatchResult(
                        text=word,
                        start_pos=start_pos,
                        end_pos=end_pos,
                        confidence=0.95,
                        pattern_type=PatternType.DOI,
                        metadata={"doi": clean_word},
                    )
                )

        return results

    def find_page_patterns(self, text: str) -> List[PatternMatchResult]:
        """
        Find page range patterns in text.

        Args:
            text: Text to analyze

        Returns:
            List of PatternMatchResult objects for page patterns
        """
        results = []
        words = text.split()

        for i, word in enumerate(words):
            clean_word = word.strip(".,;:")
            if self.text_analyzer.is_numeric_range(clean_word):
                start_pos = len(" ".join(words[:i]))
                if start_pos > 0:
                    start_pos += 1
                end_pos = start_pos + len(word)

                # Check if preceded by 'pp.' or 'pages'
                confidence = 0.8
                if i > 0 and words[i - 1].lower() in ["pp", "pp.", "pages", "p."]:
                    confidence = 0.9

                results.append(
                    PatternMatchResult(
                        text=word,
                        start_pos=start_pos,
                        end_pos=end_pos,
                        confidence=confidence,
                        pattern_type=PatternType.PAGES,
                        metadata={"pages": clean_word},
                    )
                )

        return results

    def detect_citation_format(self, text: str) -> Tuple[str, float]:
        """
        Auto-detect citation format with confidence score.

        Analyzes the structure and patterns in the text to determine
        the most likely citation format.

        Args:
            text: Citation text to analyze

        Returns:
            Tuple of (format_name, confidence_score)
        """
        # Find all pattern types
        authors = self.find_author_patterns(text)
        years = self.find_year_patterns(text)
        titles = self.find_title_patterns(text)

        # Analyze pattern positions and structure
        has_year_in_parens = any(result.text.startswith("(") for result in years)
        has_quoted_title = any(
            result.metadata.get("format") == "quoted" for result in titles
        )
        has_lastname_first = any(
            result.metadata.get("format") == "lastname_first" for result in authors
        )

        # APA format detection
        apa_score = 0.0
        if has_year_in_parens:
            apa_score += 0.4
        if has_lastname_first:
            apa_score += 0.3
        if authors and years:
            apa_score += 0.3

        # MLA format detection
        mla_score = 0.0
        if has_lastname_first:
            mla_score += 0.4
        if has_quoted_title:
            mla_score += 0.3
        if not has_year_in_parens and years:
            mla_score += 0.3

        # IEEE format detection
        ieee_score = 0.0
        if authors and not has_lastname_first:
            ieee_score += 0.3
        if titles and not has_quoted_title:
            ieee_score += 0.3
        if years and not has_year_in_parens:
            ieee_score += 0.4

        # Determine best match
        scores = [("APA", apa_score), ("MLA", mla_score), ("IEEE", ieee_score)]
        best_format, best_score = max(scores, key=lambda x: x[1])

        # If no clear winner, default to "Unknown"
        if best_score < 0.5:
            return "Unknown", best_score

        return best_format, best_score


def find_author_patterns(text: str) -> List[PatternMatchResult]:
    """
    Convenience function to find author patterns in text.

    Args:
        text: Text to analyze

    Returns:
        List of PatternMatchResult objects for author patterns
    """
    detector = CitationDetector()
    return detector.find_author_patterns(text)


def find_year_patterns(text: str) -> List[PatternMatchResult]:
    """
    Convenience function to find year patterns in text.

    Args:
        text: Text to analyze

    Returns:
        List of PatternMatchResult objects for year patterns
    """
    detector = CitationDetector()
    return detector.find_year_patterns(text)


def find_title_patterns(text: str) -> List[PatternMatchResult]:
    """
    Convenience function to find title patterns in text.

    Args:
        text: Text to analyze

    Returns:
        List of PatternMatchResult objects for title patterns
    """
    detector = CitationDetector()
    return detector.find_title_patterns(text)


def find_journal_patterns(text: str) -> List[PatternMatchResult]:
    """
    Convenience function to find journal patterns in text.

    Args:
        text: Text to analyze

    Returns:
        List of PatternMatchResult objects for journal patterns
    """
    detector = CitationDetector()
    return detector.find_journal_patterns(text)


def find_doi_patterns(text: str) -> List[PatternMatchResult]:
    """
    Convenience function to find DOI patterns in text.

    Args:
        text: Text to analyze

    Returns:
        List of PatternMatchResult objects for DOI patterns
    """
    detector = CitationDetector()
    return detector.find_doi_patterns(text)


def find_page_patterns(text: str) -> List[PatternMatchResult]:
    """
    Convenience function to find page range patterns in text.

    Args:
        text: Text to analyze

    Returns:
        List of PatternMatchResult objects for page patterns
    """
    detector = CitationDetector()
    return detector.find_page_patterns(text)


def detect_citation_format(text: str) -> Tuple[str, float]:
    """
    Convenience function to auto-detect citation format.

    Args:
        text: Citation text to analyze

    Returns:
        Tuple of (format_name, confidence_score)
    """
    detector = CitationDetector()
    return detector.detect_citation_format(text)
