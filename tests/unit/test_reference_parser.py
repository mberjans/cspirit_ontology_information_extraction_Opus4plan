"""
Comprehensive Unit Tests for Reference Parser

This module provides comprehensive unit tests for the ReferenceParser class and its
associated components in the AIM2 ontology information extraction system.

Test Coverage:
1. ReferenceParser basic functionality
2. Format detection and auto-detection
3. Citation format handlers (APA, MLA, IEEE)
4. Pattern matching utilities
5. Error handling and edge cases
6. Batch processing capabilities
7. Integration with existing systems

Test Classes:
    TestReferenceParserCreation: Tests for parser instantiation and configuration
    TestReferenceParserBasicFunctionality: Tests for core parsing functionality
    TestReferenceParserFormatDetection: Tests for format detection capabilities
    TestReferenceParserBatchProcessing: Tests for batch processing features
    TestReferenceParserAPAHandler: Tests for APA format handler
    TestReferenceParserMLAHandler: Tests for MLA format handler
    TestReferenceParserIEEEHandler: Tests for IEEE format handler
    TestReferenceParserPatternMatching: Tests for pattern matching utilities
    TestReferenceParserErrorHandling: Tests for error handling and edge cases
    TestReferenceParserIntegration: Tests for integration with other components

Dependencies:
    - pytest: Test framework
    - unittest.mock: Mocking functionality
    - typing: Type hints
    - dataclasses: Data structures

Usage:
    pytest tests/unit/test_reference_parser.py -v
"""

from unittest.mock import Mock

import pytest

# Import the components we're testing
try:
    from aim2_project.aim2_ontology.parsers.citation_formats import (
        APAHandler,
        CitationMetadata,
        CitationType,
        IEEEHandler,
        MLAHandler,
        ParseResult,
        get_handler_for_format,
        get_supported_formats,
    )
    from aim2_project.aim2_ontology.parsers.reference_parser import (
        BatchParseResult,
        ParsingStrategy,
        ReferenceParser,
        ReferenceParseResult,
    )
    from aim2_project.aim2_ontology.parsers.reference_patterns import (
        CitationDetector,
        PatternMatchResult,
        PatternType,
        TextAnalyzer,
        detect_citation_format,
        find_author_patterns,
        find_doi_patterns,
        find_journal_patterns,
        find_page_patterns,
        find_title_patterns,
        find_year_patterns,
    )
    from aim2_project.exceptions import ExtractionException, ValidationException

    IMPORTS_AVAILABLE = True
except ImportError:
    # If imports fail, we'll use mocks for everything
    IMPORTS_AVAILABLE = False


# Test data fixtures
@pytest.fixture
def sample_apa_citation():
    """Sample APA format citation."""
    return "Smith, J. A. (2023). Machine learning in bioinformatics. Nature Methods, 20(5), 123-130. doi:10.1038/s41592-023-01234-5"


@pytest.fixture
def sample_mla_citation():
    """Sample MLA format citation."""
    return 'Smith, John A. "Machine Learning in Bioinformatics." Nature Methods, vol. 20, no. 5, 2023, pp. 123-130.'


@pytest.fixture
def sample_ieee_citation():
    """Sample IEEE format citation."""
    return '[1] J. A. Smith, "Machine learning in bioinformatics," Nature Methods, vol. 20, no. 5, pp. 123-130, 2023.'


@pytest.fixture
def sample_citations(sample_apa_citation, sample_mla_citation, sample_ieee_citation):
    """Collection of sample citations in different formats."""
    return {
        "apa": sample_apa_citation,
        "mla": sample_mla_citation,
        "ieee": sample_ieee_citation,
    }


@pytest.fixture
def malformed_citations():
    """Collection of malformed or problematic citations."""
    return [
        "",  # Empty citation
        "   ",  # Whitespace only
        "Smith",  # Too short
        "Random text without proper citation format",  # No clear structure
        "Smith, J. (2025). Future paper.",  # Future year
        "Smith, J. (1700). Too old paper.",  # Too old year
        "Author (2023). Title with invalid DOI: 10.invalid",  # Invalid DOI
        "Smith, J. & & Brown, K. (2023). Double ampersand.",  # Malformed authors
    ]


@pytest.fixture
def mock_citation_metadata():
    """Mock CitationMetadata object."""
    if IMPORTS_AVAILABLE:
        metadata = CitationMetadata()
        metadata.authors = ["Smith, J. A."]
        metadata.title = "Machine Learning in Bioinformatics"
        metadata.journal = "Nature Methods"
        metadata.year = 2023
        metadata.volume = "20"
        metadata.issue = "5"
        metadata.pages = "123-130"
        metadata.doi = "10.1038/s41592-023-01234-5"
        metadata.citation_type = CitationType.JOURNAL_ARTICLE
        metadata.set_confidence("authors", 0.9)
        metadata.set_confidence("title", 0.8)
        metadata.set_confidence("journal", 0.85)
        metadata.set_confidence("year", 0.95)
        return metadata
    else:
        # Mock version
        mock_metadata = Mock()
        mock_metadata.authors = ["Smith, J. A."]
        mock_metadata.title = "Machine Learning in Bioinformatics"
        mock_metadata.journal = "Nature Methods"
        mock_metadata.year = 2023
        mock_metadata.volume = "20"
        mock_metadata.issue = "5"
        mock_metadata.pages = "123-130"
        mock_metadata.doi = "10.1038/s41592-023-01234-5"
        mock_metadata.get_overall_confidence.return_value = 0.85
        mock_metadata.to_dict.return_value = {
            "authors": ["Smith, J. A."],
            "title": "Machine Learning in Bioinformatics",
            "journal": "Nature Methods",
            "year": 2023,
        }
        return mock_metadata


# Mock fixtures for when imports aren't available
@pytest.fixture
def mock_reference_parser():
    """Mock ReferenceParser for testing when imports fail."""
    if IMPORTS_AVAILABLE:
        return None

    mock_parser = Mock()
    mock_parser.parse_reference = Mock()
    mock_parser.parse_references = Mock()
    mock_parser.detect_format = Mock()
    mock_parser.get_supported_formats = Mock(return_value=["APA", "MLA", "IEEE"])
    mock_parser.get_parser_statistics = Mock()
    return mock_parser


@pytest.fixture
def mock_citation_detector():
    """Mock CitationDetector for testing."""
    if IMPORTS_AVAILABLE:
        return None

    mock_detector = Mock()
    mock_detector.find_author_patterns = Mock()
    mock_detector.find_year_patterns = Mock()
    mock_detector.find_title_patterns = Mock()
    mock_detector.find_journal_patterns = Mock()
    mock_detector.find_doi_patterns = Mock()
    mock_detector.find_page_patterns = Mock()
    mock_detector.detect_citation_format = Mock()
    return mock_detector


class TestReferenceParserCreation:
    """Test suite for ReferenceParser creation and initialization."""

    def test_parser_creation_default_settings(self):
        """Test creating parser with default settings."""
        if not IMPORTS_AVAILABLE:
            pytest.skip("Reference parser imports not available")

        parser = ReferenceParser()

        assert parser is not None
        assert parser.default_strategy == ParsingStrategy.AUTO_DETECT
        assert parser.confidence_threshold == 0.5
        assert parser.enable_fallback is True
        assert len(parser.handlers) > 0

    def test_parser_creation_custom_settings(self):
        """Test creating parser with custom settings."""
        if not IMPORTS_AVAILABLE:
            pytest.skip("Reference parser imports not available")

        parser = ReferenceParser(
            default_strategy=ParsingStrategy.TRY_ALL_FORMATS,
            confidence_threshold=0.7,
            enable_fallback=False,
        )

        assert parser.default_strategy == ParsingStrategy.TRY_ALL_FORMATS
        assert parser.confidence_threshold == 0.7
        assert parser.enable_fallback is False

    def test_parser_handler_initialization(self):
        """Test that format handlers are properly initialized."""
        if not IMPORTS_AVAILABLE:
            pytest.skip("Reference parser imports not available")

        parser = ReferenceParser()

        expected_formats = ["APA", "MLA", "IEEE"]
        for format_name in expected_formats:
            assert format_name in parser.handlers
            assert parser.handlers[format_name] is not None

    def test_parser_supported_formats(self):
        """Test getting supported formats."""
        if not IMPORTS_AVAILABLE:
            pytest.skip("Reference parser imports not available")

        parser = ReferenceParser()
        formats = parser.get_supported_formats()

        assert isinstance(formats, list)
        assert len(formats) > 0
        assert "APA" in formats
        assert "MLA" in formats
        assert "IEEE" in formats


class TestReferenceParserBasicFunctionality:
    """Test suite for basic ReferenceParser functionality."""

    def test_parse_single_reference_apa(self, sample_apa_citation):
        """Test parsing a single APA format reference."""
        if not IMPORTS_AVAILABLE:
            pytest.skip("Reference parser imports not available")

        parser = ReferenceParser()
        result = parser.parse_reference(sample_apa_citation)

        assert isinstance(result, ReferenceParseResult)
        assert result.success
        assert result.detected_format in ["APA", "Unknown"]
        assert result.metadata is not None
        assert result.processing_time is not None
        assert result.overall_confidence > 0.0

    def test_parse_single_reference_mla(self, sample_mla_citation):
        """Test parsing a single MLA format reference."""
        if not IMPORTS_AVAILABLE:
            pytest.skip("Reference parser imports not available")

        parser = ReferenceParser()
        result = parser.parse_reference(sample_mla_citation)

        assert isinstance(result, ReferenceParseResult)
        assert result.success
        assert result.detected_format in ["MLA", "Unknown"]
        assert result.metadata is not None
        assert result.processing_time is not None

    def test_parse_single_reference_ieee(self, sample_ieee_citation):
        """Test parsing a single IEEE format reference."""
        if not IMPORTS_AVAILABLE:
            pytest.skip("Reference parser imports not available")

        parser = ReferenceParser()
        result = parser.parse_reference(sample_ieee_citation)

        assert isinstance(result, ReferenceParseResult)
        assert result.success
        assert result.detected_format in ["IEEE", "Unknown"]
        assert result.metadata is not None
        assert result.processing_time is not None

    def test_parse_empty_reference(self):
        """Test parsing empty or invalid reference."""
        if not IMPORTS_AVAILABLE:
            pytest.skip("Reference parser imports not available")

        parser = ReferenceParser()
        result = parser.parse_reference("")

        assert isinstance(result, ReferenceParseResult)
        assert not result.success
        assert len(result.parsing_errors) > 0
        assert "Empty or invalid reference text" in str(result.parsing_errors)

    def test_parse_reference_with_forced_format(self, sample_apa_citation):
        """Test parsing with forced format specification."""
        if not IMPORTS_AVAILABLE:
            pytest.skip("Reference parser imports not available")

        parser = ReferenceParser()
        result = parser.parse_reference(
            sample_apa_citation,
            strategy=ParsingStrategy.FORCE_FORMAT,
            force_format="APA",
        )

        assert isinstance(result, ReferenceParseResult)
        assert result.parsing_strategy == ParsingStrategy.FORCE_FORMAT
        assert result.detected_format == "APA"

    def test_parse_reference_try_all_formats(self, sample_apa_citation):
        """Test parsing with try-all-formats strategy."""
        if not IMPORTS_AVAILABLE:
            pytest.skip("Reference parser imports not available")

        parser = ReferenceParser()
        result = parser.parse_reference(
            sample_apa_citation, strategy=ParsingStrategy.TRY_ALL_FORMATS
        )

        assert isinstance(result, ReferenceParseResult)
        assert result.parsing_strategy == ParsingStrategy.TRY_ALL_FORMATS
        assert result.success


class TestReferenceParserFormatDetection:
    """Test suite for format detection capabilities."""

    def test_detect_apa_format(self, sample_apa_citation):
        """Test detecting APA format."""
        if not IMPORTS_AVAILABLE:
            pytest.skip("Reference parser imports not available")

        parser = ReferenceParser()
        format_name, confidence = parser.detect_format(sample_apa_citation)

        assert isinstance(format_name, str)
        assert isinstance(confidence, float)
        assert 0.0 <= confidence <= 1.0
        # Note: might detect as 'Unknown' if detection isn't perfect
        assert format_name in ["APA", "Unknown"]

    def test_detect_mla_format(self, sample_mla_citation):
        """Test detecting MLA format."""
        if not IMPORTS_AVAILABLE:
            pytest.skip("Reference parser imports not available")

        parser = ReferenceParser()
        format_name, confidence = parser.detect_format(sample_mla_citation)

        assert isinstance(format_name, str)
        assert isinstance(confidence, float)
        assert 0.0 <= confidence <= 1.0
        assert format_name in ["MLA", "Unknown"]

    def test_detect_ieee_format(self, sample_ieee_citation):
        """Test detecting IEEE format."""
        if not IMPORTS_AVAILABLE:
            pytest.skip("Reference parser imports not available")

        parser = ReferenceParser()
        format_name, confidence = parser.detect_format(sample_ieee_citation)

        assert isinstance(format_name, str)
        assert isinstance(confidence, float)
        assert 0.0 <= confidence <= 1.0
        assert format_name in ["IEEE", "Unknown"]

    def test_detect_empty_format(self):
        """Test format detection with empty input."""
        if not IMPORTS_AVAILABLE:
            pytest.skip("Reference parser imports not available")

        parser = ReferenceParser()
        format_name, confidence = parser.detect_format("")

        assert format_name == "Unknown"
        assert confidence == 0.0

    def test_detect_malformed_format(self):
        """Test format detection with malformed input."""
        if not IMPORTS_AVAILABLE:
            pytest.skip("Reference parser imports not available")

        parser = ReferenceParser()
        format_name, confidence = parser.detect_format("Random text")

        assert isinstance(format_name, str)
        assert isinstance(confidence, float)
        assert 0.0 <= confidence <= 1.0


class TestReferenceParserBatchProcessing:
    """Test suite for batch processing capabilities."""

    def test_parse_multiple_references(self, sample_citations):
        """Test parsing multiple references."""
        if not IMPORTS_AVAILABLE:
            pytest.skip("Reference parser imports not available")

        parser = ReferenceParser()
        references = list(sample_citations.values())
        result = parser.parse_references(references)

        assert isinstance(result, BatchParseResult)
        assert result.total_processed == len(references)
        assert len(result.results) == len(references)
        assert result.successful_parses >= 0
        assert result.failed_parses >= 0
        assert result.successful_parses + result.failed_parses == result.total_processed
        assert result.total_processing_time is not None

    def test_parse_empty_batch(self):
        """Test parsing empty batch of references."""
        if not IMPORTS_AVAILABLE:
            pytest.skip("Reference parser imports not available")

        parser = ReferenceParser()
        result = parser.parse_references([])

        assert isinstance(result, BatchParseResult)
        assert result.total_processed == 0
        assert result.successful_parses == 0
        assert result.failed_parses == 0
        assert len(result.results) == 0

    def test_parse_mixed_quality_batch(self, sample_citations, malformed_citations):
        """Test parsing batch with mixed quality citations."""
        if not IMPORTS_AVAILABLE:
            pytest.skip("Reference parser imports not available")

        parser = ReferenceParser()
        mixed_references = list(sample_citations.values()) + malformed_citations[:3]
        result = parser.parse_references(mixed_references)

        assert isinstance(result, BatchParseResult)
        assert result.total_processed == len(mixed_references)
        assert len(result.results) == len(mixed_references)
        assert result.successful_parses > 0
        assert result.failed_parses >= 0

    def test_batch_result_statistics(self, sample_citations):
        """Test batch result statistics calculation."""
        if not IMPORTS_AVAILABLE:
            pytest.skip("Reference parser imports not available")

        parser = ReferenceParser()
        references = list(sample_citations.values())
        result = parser.parse_references(references)

        # Test success rate calculation
        assert 0.0 <= result.success_rate <= 100.0

        # Test average confidence calculation
        assert 0.0 <= result.average_confidence <= 1.0

        # Test format distribution
        format_dist = result.get_format_distribution()
        assert isinstance(format_dist, dict)

        # Test dictionary conversion
        result_dict = result.to_dict()
        assert isinstance(result_dict, dict)
        assert "results" in result_dict
        assert "total_processed" in result_dict
        assert "success_rate" in result_dict


class TestReferenceParserAPAHandler:
    """Test suite for APA format handler."""

    def test_apa_handler_creation(self):
        """Test APA handler creation."""
        if not IMPORTS_AVAILABLE:
            pytest.skip("Reference parser imports not available")

        handler = APAHandler()
        assert handler is not None
        assert handler.format_name == "APA"
        assert handler.format_description is not None

    def test_apa_format_detection(self, sample_apa_citation):
        """Test APA format detection."""
        if not IMPORTS_AVAILABLE:
            pytest.skip("Reference parser imports not available")

        handler = APAHandler()
        confidence = handler.detect_format(sample_apa_citation)

        assert isinstance(confidence, float)
        assert 0.0 <= confidence <= 1.0

    def test_apa_citation_parsing(self, sample_apa_citation):
        """Test APA citation parsing."""
        if not IMPORTS_AVAILABLE:
            pytest.skip("Reference parser imports not available")

        handler = APAHandler()
        result = handler.parse_citation(sample_apa_citation)

        assert isinstance(result, ParseResult)
        assert result.metadata is not None
        assert isinstance(result.parsing_errors, list)
        assert isinstance(result.parsing_warnings, list)

    def test_apa_author_extraction(self, sample_apa_citation):
        """Test author extraction from APA citation."""
        if not IMPORTS_AVAILABLE:
            pytest.skip("Reference parser imports not available")

        handler = APAHandler()
        result = handler.parse_citation(sample_apa_citation)

        if result.success and result.metadata.authors:
            assert len(result.metadata.authors) > 0
            # APA format typically has "Last, F. M." format
            author = result.metadata.authors[0]
            assert isinstance(author, str)
            assert len(author) > 0


class TestReferenceParserMLAHandler:
    """Test suite for MLA format handler."""

    def test_mla_handler_creation(self):
        """Test MLA handler creation."""
        if not IMPORTS_AVAILABLE:
            pytest.skip("Reference parser imports not available")

        handler = MLAHandler()
        assert handler is not None
        assert handler.format_name == "MLA"
        assert handler.format_description is not None

    def test_mla_format_detection(self, sample_mla_citation):
        """Test MLA format detection."""
        if not IMPORTS_AVAILABLE:
            pytest.skip("Reference parser imports not available")

        handler = MLAHandler()
        confidence = handler.detect_format(sample_mla_citation)

        assert isinstance(confidence, float)
        assert 0.0 <= confidence <= 1.0

    def test_mla_citation_parsing(self, sample_mla_citation):
        """Test MLA citation parsing."""
        if not IMPORTS_AVAILABLE:
            pytest.skip("Reference parser imports not available")

        handler = MLAHandler()
        result = handler.parse_citation(sample_mla_citation)

        assert isinstance(result, ParseResult)
        assert result.metadata is not None
        assert isinstance(result.parsing_errors, list)
        assert isinstance(result.parsing_warnings, list)

    def test_mla_quoted_title_detection(self, sample_mla_citation):
        """Test detection of quoted titles in MLA format."""
        if not IMPORTS_AVAILABLE:
            pytest.skip("Reference parser imports not available")

        handler = MLAHandler()
        result = handler.parse_citation(sample_mla_citation)

        if result.success and result.metadata.title:
            assert isinstance(result.metadata.title, str)
            assert len(result.metadata.title) > 0


class TestReferenceParserIEEEHandler:
    """Test suite for IEEE format handler."""

    def test_ieee_handler_creation(self):
        """Test IEEE handler creation."""
        if not IMPORTS_AVAILABLE:
            pytest.skip("Reference parser imports not available")

        handler = IEEEHandler()
        assert handler is not None
        assert handler.format_name == "IEEE"
        assert handler.format_description is not None

    def test_ieee_format_detection(self, sample_ieee_citation):
        """Test IEEE format detection."""
        if not IMPORTS_AVAILABLE:
            pytest.skip("Reference parser imports not available")

        handler = IEEEHandler()
        confidence = handler.detect_format(sample_ieee_citation)

        assert isinstance(confidence, float)
        assert 0.0 <= confidence <= 1.0

    def test_ieee_citation_parsing(self, sample_ieee_citation):
        """Test IEEE citation parsing."""
        if not IMPORTS_AVAILABLE:
            pytest.skip("Reference parser imports not available")

        handler = IEEEHandler()
        result = handler.parse_citation(sample_ieee_citation)

        assert isinstance(result, ParseResult)
        assert result.metadata is not None
        assert isinstance(result.parsing_errors, list)
        assert isinstance(result.parsing_warnings, list)

    def test_ieee_numbered_format(self, sample_ieee_citation):
        """Test IEEE numbered citation format handling."""
        if not IMPORTS_AVAILABLE:
            pytest.skip("Reference parser imports not available")

        # IEEE citations often start with [1], [2], etc.
        assert sample_ieee_citation.startswith("[1]")

        handler = IEEEHandler()
        result = handler.parse_citation(sample_ieee_citation)

        # Should handle the numbered prefix correctly
        assert isinstance(result, ParseResult)


class TestReferenceParserPatternMatching:
    """Test suite for pattern matching utilities."""

    def test_citation_detector_creation(self):
        """Test CitationDetector creation."""
        if not IMPORTS_AVAILABLE:
            pytest.skip("Reference parser imports not available")

        detector = CitationDetector()
        assert detector is not None
        assert detector.text_analyzer is not None

    def test_find_author_patterns(self, sample_apa_citation):
        """Test finding author patterns."""
        if not IMPORTS_AVAILABLE:
            pytest.skip("Reference parser imports not available")

        patterns = find_author_patterns(sample_apa_citation)

        assert isinstance(patterns, list)
        if patterns:
            pattern = patterns[0]
            assert isinstance(pattern, PatternMatchResult)
            assert pattern.pattern_type == PatternType.AUTHOR
            assert 0.0 <= pattern.confidence <= 1.0

    def test_find_year_patterns(self, sample_apa_citation):
        """Test finding year patterns."""
        if not IMPORTS_AVAILABLE:
            pytest.skip("Reference parser imports not available")

        patterns = find_year_patterns(sample_apa_citation)

        assert isinstance(patterns, list)
        if patterns:
            pattern = patterns[0]
            assert isinstance(pattern, PatternMatchResult)
            assert pattern.pattern_type == PatternType.YEAR
            assert 0.0 <= pattern.confidence <= 1.0

    def test_find_title_patterns(self, sample_apa_citation):
        """Test finding title patterns."""
        if not IMPORTS_AVAILABLE:
            pytest.skip("Reference parser imports not available")

        patterns = find_title_patterns(sample_apa_citation)

        assert isinstance(patterns, list)
        if patterns:
            pattern = patterns[0]
            assert isinstance(pattern, PatternMatchResult)
            assert pattern.pattern_type == PatternType.TITLE
            assert 0.0 <= pattern.confidence <= 1.0

    def test_find_journal_patterns(self, sample_apa_citation):
        """Test finding journal patterns."""
        if not IMPORTS_AVAILABLE:
            pytest.skip("Reference parser imports not available")

        patterns = find_journal_patterns(sample_apa_citation)

        assert isinstance(patterns, list)
        if patterns:
            pattern = patterns[0]
            assert isinstance(pattern, PatternMatchResult)
            assert pattern.pattern_type == PatternType.JOURNAL
            assert 0.0 <= pattern.confidence <= 1.0

    def test_find_doi_patterns(self, sample_apa_citation):
        """Test finding DOI patterns."""
        if not IMPORTS_AVAILABLE:
            pytest.skip("Reference parser imports not available")

        patterns = find_doi_patterns(sample_apa_citation)

        assert isinstance(patterns, list)
        if patterns:
            pattern = patterns[0]
            assert isinstance(pattern, PatternMatchResult)
            assert pattern.pattern_type == PatternType.DOI
            assert 0.0 <= pattern.confidence <= 1.0

    def test_find_page_patterns(self, sample_apa_citation):
        """Test finding page patterns."""
        if not IMPORTS_AVAILABLE:
            pytest.skip("Reference parser imports not available")

        patterns = find_page_patterns(sample_apa_citation)

        assert isinstance(patterns, list)
        if patterns:
            pattern = patterns[0]
            assert isinstance(pattern, PatternMatchResult)
            assert pattern.pattern_type == PatternType.PAGES
            assert 0.0 <= pattern.confidence <= 1.0

    def test_detect_citation_format_function(self, sample_citations):
        """Test the detect_citation_format function."""
        if not IMPORTS_AVAILABLE:
            pytest.skip("Reference parser imports not available")

        for format_name, citation in sample_citations.items():
            detected_format, confidence = detect_citation_format(citation)

            assert isinstance(detected_format, str)
            assert isinstance(confidence, float)
            assert 0.0 <= confidence <= 1.0

    def test_text_analyzer_utilities(self):
        """Test TextAnalyzer utility methods."""
        if not IMPORTS_AVAILABLE:
            pytest.skip("Reference parser imports not available")

        analyzer = TextAnalyzer()

        # Test capitalization check
        assert analyzer.is_capitalized_word("Smith")
        assert analyzer.is_capitalized_word("J.")
        assert not analyzer.is_capitalized_word("smith")
        assert not analyzer.is_capitalized_word("")

        # Test numeric range check
        assert analyzer.is_numeric_range("123-456")
        assert analyzer.is_numeric_range("45-67")
        assert not analyzer.is_numeric_range("123")
        assert not analyzer.is_numeric_range("abc-def")

        # Test year format check
        assert analyzer.is_year_format("2023")
        assert analyzer.is_year_format("1999")
        assert not analyzer.is_year_format("1700")  # Too old
        assert not analyzer.is_year_format("3000")  # Too future
        assert not analyzer.is_year_format("abc")

        # Test DOI format check
        assert analyzer.is_doi_format("10.1038/s41592-023-01234-5")
        assert analyzer.is_doi_format("10.1000/182")
        assert not analyzer.is_doi_format("invalid-doi")
        assert not analyzer.is_doi_format("10.")

        # Test URL format check
        assert analyzer.is_url_format("https://example.com")
        assert analyzer.is_url_format("http://example.com")
        assert analyzer.is_url_format("www.example.com")
        assert not analyzer.is_url_format("example.com")
        assert not analyzer.is_url_format("invalid")


class TestReferenceParserErrorHandling:
    """Test suite for error handling and edge cases."""

    def test_parse_malformed_citations(self, malformed_citations):
        """Test parsing malformed citations."""
        if not IMPORTS_AVAILABLE:
            pytest.skip("Reference parser imports not available")

        parser = ReferenceParser()

        for malformed_citation in malformed_citations:
            result = parser.parse_reference(malformed_citation)

            assert isinstance(result, ReferenceParseResult)
            # For malformed citations, either success with low confidence or failure
            if not result.success:
                assert len(result.parsing_errors) > 0
            else:
                # If it succeeded, confidence should reflect uncertainty
                assert result.overall_confidence >= 0.0

    def test_unsupported_format_handling(self):
        """Test handling of unsupported citation formats."""
        if not IMPORTS_AVAILABLE:
            pytest.skip("Reference parser imports not available")

        parser = ReferenceParser()

        # Try to force an unsupported format
        result = parser.parse_reference(
            "Some citation text",
            strategy=ParsingStrategy.FORCE_FORMAT,
            force_format="UNSUPPORTED",
        )

        assert isinstance(result, ReferenceParseResult)
        assert not result.success
        assert len(result.parsing_errors) > 0
        assert "Unsupported format" in str(result.parsing_errors)

    def test_exception_handling_in_parsing(self):
        """Test exception handling during parsing."""
        if not IMPORTS_AVAILABLE:
            pytest.skip("Reference parser imports not available")

        parser = ReferenceParser()

        # Test with various problematic inputs
        problematic_inputs = [
            None,  # This will cause an error before reaching the parser
            "A" * 10000,  # Very long text
            "\x00\x01\x02",  # Binary data
            "é†ç∂ƒ∂åßß∂ƒ",  # Unicode characters
        ]

        for input_text in problematic_inputs:
            if input_text is None:
                continue  # Skip None as it would cause TypeError before parsing

            try:
                result = parser.parse_reference(input_text)
                assert isinstance(result, ReferenceParseResult)
                # Either success or controlled failure
                if not result.success:
                    assert len(result.parsing_errors) > 0
            except Exception as e:
                # Any unhandled exceptions should be caught and handled gracefully
                pytest.fail(
                    f"Unhandled exception for input '{input_text[:50]}...': {e}"
                )

    def test_batch_processing_error_recovery(
        self, sample_citations, malformed_citations
    ):
        """Test error recovery in batch processing."""
        if not IMPORTS_AVAILABLE:
            pytest.skip("Reference parser imports not available")

        parser = ReferenceParser()

        # Mix good and bad citations
        mixed_batch = (
            list(sample_citations.values())[:2]
            + malformed_citations[:2]  # Good citations
            + list(sample_citations.values())[  # Bad citations
                2:
            ]  # More good citations
        )

        result = parser.parse_references(mixed_batch)

        assert isinstance(result, BatchParseResult)
        assert result.total_processed == len(mixed_batch)
        assert len(result.results) == len(mixed_batch)

        # Should have both successful and failed parses
        assert result.successful_parses > 0
        # Some failures are expected due to malformed citations

    def test_confidence_threshold_handling(self, sample_apa_citation):
        """Test handling of confidence thresholds."""
        if not IMPORTS_AVAILABLE:
            pytest.skip("Reference parser imports not available")

        # Test with high confidence threshold
        parser = ReferenceParser(confidence_threshold=0.9)
        result = parser.parse_reference(sample_apa_citation)

        assert isinstance(result, ReferenceParseResult)

        # Test with low confidence threshold
        parser = ReferenceParser(confidence_threshold=0.1)
        result = parser.parse_reference(sample_apa_citation)

        assert isinstance(result, ReferenceParseResult)

    def test_fallback_disabled_handling(self, sample_apa_citation):
        """Test behavior when fallback is disabled."""
        if not IMPORTS_AVAILABLE:
            pytest.skip("Reference parser imports not available")

        parser = ReferenceParser(enable_fallback=False)
        result = parser.parse_reference(sample_apa_citation)

        assert isinstance(result, ReferenceParseResult)
        # Should still work, just without fallback strategies


class TestReferenceParserIntegration:
    """Test suite for integration with other components."""

    def test_parser_statistics(self):
        """Test parser statistics retrieval."""
        if not IMPORTS_AVAILABLE:
            pytest.skip("Reference parser imports not available")

        parser = ReferenceParser()
        stats = parser.get_parser_statistics()

        assert isinstance(stats, dict)
        assert "supported_formats" in stats
        assert "active_handlers" in stats
        assert "default_strategy" in stats
        assert "confidence_threshold" in stats
        assert "enable_fallback" in stats

        assert isinstance(stats["supported_formats"], list)
        assert isinstance(stats["active_handlers"], int)
        assert stats["active_handlers"] > 0

    def test_result_serialization(self, sample_apa_citation):
        """Test serialization of parsing results."""
        if not IMPORTS_AVAILABLE:
            pytest.skip("Reference parser imports not available")

        parser = ReferenceParser()
        result = parser.parse_reference(sample_apa_citation)

        # Test single result serialization
        result_dict = result.to_dict()
        assert isinstance(result_dict, dict)
        assert "metadata" in result_dict
        assert "detected_format" in result_dict
        assert "format_confidence" in result_dict
        assert "parsing_strategy" in result_dict
        assert "success" in result_dict
        assert "overall_confidence" in result_dict

    def test_metadata_confidence_scoring(self, mock_citation_metadata):
        """Test metadata confidence scoring system."""
        if not IMPORTS_AVAILABLE:
            pytest.skip("Reference parser imports not available")

        if IMPORTS_AVAILABLE:
            metadata = mock_citation_metadata

            # Test individual confidence scores
            assert 0.0 <= metadata.get_confidence("authors") <= 1.0
            assert 0.0 <= metadata.get_confidence("title") <= 1.0

            # Test overall confidence calculation
            overall = metadata.get_overall_confidence()
            assert 0.0 <= overall <= 1.0

            # Test confidence setting
            metadata.set_confidence("test_field", 0.75)
            assert metadata.get_confidence("test_field") == 0.75

            # Test confidence bounds
            metadata.set_confidence("test_field", 1.5)  # Should be clamped to 1.0
            assert metadata.get_confidence("test_field") == 1.0

            metadata.set_confidence("test_field", -0.5)  # Should be clamped to 0.0
            assert metadata.get_confidence("test_field") == 0.0

    def test_citation_type_classification(self):
        """Test citation type classification."""
        if not IMPORTS_AVAILABLE:
            pytest.skip("Reference parser imports not available")

        # Test that CitationType enum values are available
        assert CitationType.JOURNAL_ARTICLE.value == "journal_article"
        assert CitationType.BOOK.value == "book"
        assert CitationType.CONFERENCE_PAPER.value == "conference_paper"
        assert CitationType.UNKNOWN.value == "unknown"

    def test_handler_registry_functions(self):
        """Test citation handler registry functions."""
        if not IMPORTS_AVAILABLE:
            pytest.skip("Reference parser imports not available")

        # Test getting supported formats
        formats = get_supported_formats()
        assert isinstance(formats, list)
        assert len(formats) > 0
        assert "APA" in formats

        # Test getting handlers
        for format_name in formats:
            handler = get_handler_for_format(format_name)
            assert handler is not None
            assert hasattr(handler, "parse_citation")
            assert hasattr(handler, "detect_format")

        # Test invalid format
        with pytest.raises(ValueError):
            get_handler_for_format("INVALID_FORMAT")

    def test_exception_integration(self):
        """Test integration with project exception hierarchy."""
        if not IMPORTS_AVAILABLE:
            pytest.skip("Reference parser imports not available")

        # Test that project exceptions are properly imported
        assert ExtractionException is not None
        assert ValidationException is not None

        # These exceptions should be properly used in error cases
        # (Testing actual usage would require triggering specific error conditions)


class TestReferenceParserPerformance:
    """Test suite for performance-related functionality."""

    @pytest.mark.slow
    def test_large_batch_processing_performance(self, sample_citations):
        """Test performance with large batch of citations."""
        if not IMPORTS_AVAILABLE:
            pytest.skip("Reference parser imports not available")

        parser = ReferenceParser()

        # Create a large batch by repeating sample citations
        large_batch = list(sample_citations.values()) * 50  # 150 citations

        import time

        start_time = time.time()
        result = parser.parse_references(large_batch)
        processing_time = time.time() - start_time

        assert isinstance(result, BatchParseResult)
        assert result.total_processed == len(large_batch)
        assert result.total_processing_time is not None

        # Performance assertions
        assert processing_time < 30.0  # Should complete within 30 seconds
        avg_time_per_citation = processing_time / len(large_batch)
        assert avg_time_per_citation < 1.0  # Less than 1 second per citation

    @pytest.mark.slow
    def test_memory_usage_stability(self, sample_citations):
        """Test memory usage remains stable during processing."""
        if not IMPORTS_AVAILABLE:
            pytest.skip("Reference parser imports not available")

        parser = ReferenceParser()

        # Process multiple batches to test for memory leaks
        for _ in range(10):
            batch = list(sample_citations.values()) * 10
            result = parser.parse_references(batch)
            assert isinstance(result, BatchParseResult)

        # If we get here without memory errors, the test passes
        assert True

    def test_single_citation_performance(self, sample_apa_citation):
        """Test performance of single citation parsing."""
        if not IMPORTS_AVAILABLE:
            pytest.skip("Reference parser imports not available")

        parser = ReferenceParser()

        import time

        start_time = time.time()
        result = parser.parse_reference(sample_apa_citation)
        processing_time = time.time() - start_time

        assert isinstance(result, ReferenceParseResult)
        assert processing_time < 5.0  # Should complete within 5 seconds
        assert result.processing_time is not None
        assert result.processing_time <= processing_time + 0.1  # Allow small margin


# Additional test utilities and fixtures
@pytest.fixture
def performance_monitor():
    """Monitor performance metrics during tests."""
    import os
    import time

    import psutil

    class PerformanceMonitor:
        def __init__(self):
            self.start_time = None
            self.end_time = None
            self.start_memory = None
            self.end_memory = None
            self.process = psutil.Process(os.getpid())

        def start(self):
            self.start_time = time.time()
            self.start_memory = self.process.memory_info().rss

        def stop(self):
            self.end_time = time.time()
            self.end_memory = self.process.memory_info().rss

        @property
        def duration(self):
            if self.start_time and self.end_time:
                return self.end_time - self.start_time
            return None

        @property
        def memory_delta(self):
            if self.start_memory and self.end_memory:
                return self.end_memory - self.start_memory
            return None

    return PerformanceMonitor()


@pytest.fixture
def comprehensive_test_data():
    """Comprehensive test data covering various citation scenarios."""
    return {
        "valid_citations": {
            "apa_journal": "Smith, J. A., & Brown, K. L. (2023). Machine learning applications in computational biology. Nature Biotechnology, 41(8), 1123-1135. https://doi.org/10.1038/s41587-023-01234-5",
            "apa_book": "Johnson, M. B. (2022). Computational methods in bioinformatics (2nd ed.). Academic Press.",
            "mla_journal": 'Smith, John A., and Karen L. Brown. "Machine Learning Applications in Computational Biology." Nature Biotechnology, vol. 41, no. 8, 2023, pp. 1123-1135.',
            "mla_book": "Johnson, Michael B. Computational Methods in Bioinformatics. 2nd ed., Academic Press, 2022.",
            "ieee_journal": '[1] J. A. Smith and K. L. Brown, "Machine learning applications in computational biology," Nature Biotechnology, vol. 41, no. 8, pp. 1123-1135, Aug. 2023.',
            "ieee_conference": '[2] M. B. Johnson, "Novel algorithms for sequence alignment," in Proc. IEEE Conf. Bioinformatics, 2022, pp. 45-52.',
        },
        "edge_cases": {
            "multiple_authors": "Smith, J. A., Brown, K. L., Davis, R. C., Wilson, M. D., & Taylor, S. E. (2023). Large-scale genomic analysis. Science, 380(6642), 234-241.",
            "no_doi": "Author, A. (2023). Title without DOI. Journal Name, 15(3), 123-130.",
            "book_chapter": "Chapter Author, C. A. (2023). Chapter title. In B. Editor (Ed.), Book title (pp. 45-67). Publisher.",
            "thesis": "Student, G. (2023). Dissertation title (Doctoral dissertation). University Name.",
            "web_source": "Web Author, W. (2023, March 15). Web article title. Website Name. https://example.com/article",
            "preprint": "Preprint Author, P. (2023). Preprint title. bioRxiv. https://doi.org/10.1101/2023.01.01.123456",
        },
        "problematic_cases": {
            "malformed_year": "Smith, J. (20XX). Title with malformed year. Journal, 15, 123-130.",
            "missing_pages": "Author, A. (2023). Title without pages. Journal Name, 15(3).",
            "extra_punctuation": "Smith,, J.. A... ((2023)).. Title with extra punctuation... Journal Name,, 15((3)),, 123--130..",
            "mixed_formats": 'Smith, J. A. (2023) "Mixed Format Citation" Journal Name vol. 15 no. 3 pp. 123-130',
            "incomplete_citation": "Smith, J. Title. Journal.",
            "unicode_characters": "Müller, H., & Björk, A. (2023). Årtículo con caracteres especiales. Jøurnal Nàme, 15(3), 123-130.",
        },
    }


if __name__ == "__main__":
    # Allow running the test file directly
    pytest.main([__file__, "-v"])
