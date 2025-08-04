"""
Comprehensive Unit Tests for Format Auto-Detection (TDD Approach)

This module provides comprehensive unit tests for the format auto-detection functionality
in the AIM2 ontology information extraction system. The tests follow test-driven development
(TDD) approach, validating the expected behavior of the auto-detection functions.

Test Classes:
    TestDetectFormatFromExtension: Tests for detect_format_from_extension() function
    TestDetectFormatFromContent: Tests for detect_format_from_content() function
    TestGetParserForFormat: Tests for get_parser_for_format() function
    TestAutoDetectParser: Tests for auto_detect_parser() function
    TestFormatAutoDetectionIntegration: Integration tests for auto-detection workflow

The format auto-detection functions provide:
- Extension-based format detection for all supported formats
- Content-based format detection when extension is ambiguous
- Parser class mapping for detected formats
- Comprehensive auto-detection with fallback strategies
- Error handling and validation for edge cases

Dependencies:
    - pytest: For test framework
    - unittest.mock: For mocking functionality
    - pathlib: For path handling
    - tempfile: For temporary file creation
    - typing: For type hints

Usage:
    pytest tests/unit/test_format_autodetection.py -v
"""

import tempfile
from pathlib import Path
from unittest.mock import Mock, patch

import pytest

# Import the functions to test
from aim2_project.aim2_ontology.parsers import (
    AbstractParser,
    CSVParser,
    JSONLDParser,
    OWLParser,
    auto_detect_parser,
    detect_format_from_content,
    detect_format_from_extension,
    get_parser_for_format,
)


class TestDetectFormatFromExtension:
    """Test cases for detect_format_from_extension() function."""

    @pytest.mark.parametrize(
        "file_path,expected_format",
        [
            # OWL formats
            ("ontology.owl", "owl"),
            ("data.rdf", "rdf"),
            ("knowledge.ttl", "ttl"),
            ("graph.turtle", "ttl"),
            ("triples.nt", "nt"),
            ("data.ntriples", "nt"),
            ("notation.n3", "n3"),
            ("markup.xml", "xml"),
            # JSON-LD formats
            ("linked.jsonld", "jsonld"),
            ("data.json-ld", "jsonld"),
            ("document.json", "json"),
            # CSV formats
            ("table.csv", "csv"),
            ("data.tsv", "tsv"),
            ("plain.txt", "txt"),
            ("tabbed.tab", "tsv"),
            ("dataset.dat", "csv"),
            # Path objects
            (Path("test.owl"), "owl"),
            (Path("data/ontology.ttl"), "ttl"),
        ],
    )
    def test_valid_extensions(self, file_path, expected_format):
        """Test detection of valid file extensions."""
        result = detect_format_from_extension(file_path)
        assert result == expected_format

    @pytest.mark.parametrize(
        "file_path",
        [
            # Case variations
            "DATA.OWL",
            "Ontology.TTL",
            "File.CSV",
            "DOCUMENT.JSON",
        ],
    )
    def test_case_insensitive_extensions(self, file_path):
        """Test that extension detection is case-insensitive."""
        result = detect_format_from_extension(file_path)
        assert result is not None

    @pytest.mark.parametrize(
        "file_path",
        [
            # Unknown extensions
            "unknown.xyz",
            "data.unknown",
            "file.abcd",
            # No extension
            "no_extension",
            "file_without_ext",
            # Multiple dots
            "file.backup.old",
            "data.temp.bak",
        ],
    )
    def test_unknown_extensions(self, file_path):
        """Test handling of unknown or missing extensions."""
        result = detect_format_from_extension(file_path)
        assert result is None

    def test_empty_path(self):
        """Test handling of empty file path."""
        result = detect_format_from_extension("")
        assert result is None

    def test_none_path(self):
        """Test handling of None file path."""
        # The function handles None gracefully and returns None instead of raising
        result = detect_format_from_extension(None)
        assert result is None

    @patch("aim2_project.aim2_ontology.parsers.get_logger")
    def test_logging_behavior(self, mock_get_logger):
        """Test that appropriate logging occurs during detection."""
        mock_logger = Mock()
        mock_get_logger.return_value = mock_logger

        # Test successful detection
        detect_format_from_extension("test.owl")
        mock_logger.debug.assert_called()

        # Test unknown extension
        detect_format_from_extension("test.unknown")
        mock_logger.debug.assert_called()


class TestDetectFormatFromContent:
    """Test cases for detect_format_from_content() function."""

    @pytest.fixture(autouse=True)
    def setup_content(self):
        """Set up sample content for testing."""
        self.sample_owl_content = """<?xml version="1.0"?>
<rdf:RDF xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#"
         xmlns:owl="http://www.w3.org/2002/07/owl#">
    <owl:Ontology rdf:about="http://example.org/ontology"/>
</rdf:RDF>"""

        self.sample_rdf_content = """<?xml version="1.0"?>
<rdf:RDF xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#">
    <rdf:Description rdf:about="http://example.org/resource"/>
</rdf:RDF>"""

        self.sample_turtle_content = """@prefix owl: <http://www.w3.org/2002/07/owl#> .
@prefix rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#> .
<http://example.org/ontology> a owl:Ontology ."""

        self.sample_jsonld_content = """{
    "@context": {
        "@vocab": "http://example.org/",
        "owl": "http://www.w3.org/2002/07/owl#"
    },
    "@type": "owl:Ontology"
}"""

        self.sample_csv_content = """name,age,city
John Doe,30,New York
Jane Smith,25,Los Angeles
Bob Johnson,35,Chicago"""

    @patch("aim2_project.aim2_ontology.parsers.OWLParser")
    def test_owl_content_detection(self, mock_owl_parser_class):
        """Test detection of OWL content."""
        mock_parser = Mock()
        mock_parser.detect_format.return_value = "owl"
        mock_owl_parser_class.return_value = mock_parser

        result = detect_format_from_content(self.sample_owl_content)
        assert result == "owl"
        mock_parser.detect_format.assert_called_once()

    @patch("aim2_project.aim2_ontology.parsers.JSONLDParser")
    @patch("aim2_project.aim2_ontology.parsers.OWLParser")
    def test_jsonld_content_detection(
        self, mock_owl_parser_class, mock_jsonld_parser_class
    ):
        """Test detection of JSON-LD content."""
        # Mock OWL parser to return None/unknown
        mock_owl_parser = Mock()
        mock_owl_parser.detect_format.return_value = "unknown"
        mock_owl_parser_class.return_value = mock_owl_parser

        # Mock JSON-LD parser to return jsonld
        mock_jsonld_parser = Mock()
        mock_jsonld_parser.detect_format.return_value = "jsonld"
        mock_jsonld_parser_class.return_value = mock_jsonld_parser

        result = detect_format_from_content(self.sample_jsonld_content)
        assert result == "jsonld"

    @patch("aim2_project.aim2_ontology.parsers.CSVParser")
    @patch("aim2_project.aim2_ontology.parsers.JSONLDParser")
    @patch("aim2_project.aim2_ontology.parsers.OWLParser")
    def test_csv_content_detection(
        self, mock_owl_parser_class, mock_jsonld_parser_class, mock_csv_parser_class
    ):
        """Test detection of CSV content."""
        # Mock OWL and JSON-LD parsers to return None/unknown
        mock_owl_parser = Mock()
        mock_owl_parser.detect_format.return_value = "unknown"
        mock_owl_parser_class.return_value = mock_owl_parser

        mock_jsonld_parser = Mock()
        mock_jsonld_parser.detect_format.return_value = "unknown"
        mock_jsonld_parser_class.return_value = mock_jsonld_parser

        # Mock CSV parser to return csv
        mock_csv_parser = Mock()
        mock_csv_parser.detect_format.return_value = "csv"
        mock_csv_parser_class.return_value = mock_csv_parser

        result = detect_format_from_content(self.sample_csv_content)
        assert result == "csv"

    def test_empty_content(self):
        """Test handling of empty content."""
        result = detect_format_from_content("")
        assert result is None

        result = detect_format_from_content("   ")
        assert result is None

    def test_none_content(self):
        """Test handling of None content."""
        result = detect_format_from_content(None)
        assert result is None

    @patch("aim2_project.aim2_ontology.parsers.CSVParser")
    @patch("aim2_project.aim2_ontology.parsers.JSONLDParser")
    @patch("aim2_project.aim2_ontology.parsers.OWLParser")
    def test_parser_exception_handling(
        self, mock_owl_parser_class, mock_jsonld_parser_class, mock_csv_parser_class
    ):
        """Test handling of parser exceptions."""
        # Make all parsers raise exceptions
        mock_owl_parser_class.side_effect = Exception("Mock exception")
        mock_jsonld_parser_class.side_effect = Exception("Mock exception")
        mock_csv_parser_class.side_effect = Exception("Mock exception")

        result = detect_format_from_content("some content")
        # Should not raise exception, should return None
        assert result is None

    @patch("aim2_project.aim2_ontology.parsers.CSVParser")
    @patch("aim2_project.aim2_ontology.parsers.JSONLDParser")
    @patch("aim2_project.aim2_ontology.parsers.OWLParser")
    def test_no_format_detected(
        self, mock_owl_parser_class, mock_jsonld_parser_class, mock_csv_parser_class
    ):
        """Test when no format can be detected."""
        # Mock all parsers to return None/unknown
        for parser_class in [
            mock_owl_parser_class,
            mock_jsonld_parser_class,
            mock_csv_parser_class,
        ]:
            mock_parser = Mock()
            mock_parser.detect_format.return_value = "unknown"
            parser_class.return_value = mock_parser

        result = detect_format_from_content("unrecognizable content")
        assert result is None

    def test_large_content_sampling(self):
        """Test that large content is properly sampled."""
        # Create content larger than 10KB
        large_content = "test content\n" * 1000  # > 10KB

        with patch(
            "aim2_project.aim2_ontology.parsers.OWLParser"
        ) as mock_owl_parser_class:
            mock_parser = Mock()
            mock_parser.detect_format.return_value = "owl"
            mock_owl_parser_class.return_value = mock_parser

            result = detect_format_from_content(large_content)

            # Should still return result
            assert result == "owl"

            # Check that parser was called with sampled content (first 10KB)
            called_content = mock_parser.detect_format.call_args[0][0]
            assert len(called_content) <= 10000


class TestGetParserForFormat:
    """Test cases for get_parser_for_format() function."""

    @pytest.mark.parametrize(
        "format_name,expected_parser",
        [
            # OWL formats -> OWLParser
            ("owl", OWLParser),
            ("rdf", OWLParser),
            ("ttl", OWLParser),
            ("turtle", OWLParser),
            ("nt", OWLParser),
            ("ntriples", OWLParser),
            ("n3", OWLParser),
            ("xml", OWLParser),
            # JSON-LD formats -> JSONLDParser
            ("jsonld", JSONLDParser),
            ("json-ld", JSONLDParser),
            ("json", JSONLDParser),
            # CSV formats -> CSVParser
            ("csv", CSVParser),
            ("tsv", CSVParser),
            ("txt", CSVParser),
        ],
    )
    def test_valid_format_mapping(self, format_name, expected_parser):
        """Test mapping of valid formats to parser classes."""
        result = get_parser_for_format(format_name)
        assert result == expected_parser

    def test_case_insensitive_format(self):
        """Test that format matching is case-insensitive."""
        result = get_parser_for_format("OWL")
        assert result == OWLParser

        result = get_parser_for_format("CSV")
        assert result == CSVParser

        result = get_parser_for_format("JsonLD")
        assert result == JSONLDParser

    @pytest.mark.parametrize(
        "format_name", ["unknown", "unsupported", "xyz", "fake_format"]
    )
    def test_unknown_format(self, format_name):
        """Test handling of unknown formats."""
        result = get_parser_for_format(format_name)
        assert result is None

    def test_empty_format(self):
        """Test handling of empty format name."""
        result = get_parser_for_format("")
        assert result is None

        result = get_parser_for_format("   ")
        assert result is None

    def test_none_format(self):
        """Test handling of None format name."""
        result = get_parser_for_format(None)
        assert result is None

    @patch("aim2_project.aim2_ontology.parsers.get_logger")
    def test_logging_behavior(self, mock_get_logger):
        """Test that appropriate logging occurs during parser lookup."""
        mock_logger = Mock()
        mock_get_logger.return_value = mock_logger

        # Test successful lookup
        get_parser_for_format("owl")
        mock_logger.debug.assert_called()

        # Test unknown format
        get_parser_for_format("unknown")
        mock_logger.debug.assert_called()


class TestAutoDetectParser:
    """Test cases for auto_detect_parser() function."""

    def test_extension_based_detection(self):
        """Test auto-detection using file extension."""
        with patch(
            "aim2_project.aim2_ontology.parsers.detect_format_from_extension"
        ) as mock_detect_ext:
            with patch(
                "aim2_project.aim2_ontology.parsers.get_parser_for_format"
            ) as mock_get_parser:
                mock_detect_ext.return_value = "owl"
                mock_parser_class = Mock()
                mock_parser_instance = Mock(spec=AbstractParser)
                mock_parser_class.return_value = mock_parser_instance
                mock_get_parser.return_value = mock_parser_class

                result = auto_detect_parser(file_path="test.owl")

                assert result == mock_parser_instance
                mock_detect_ext.assert_called_once_with("test.owl")
                mock_get_parser.assert_called_once_with("owl")

    def test_content_based_detection(self):
        """Test auto-detection using content analysis."""
        sample_content = '{"@context": "http://example.org"}'

        with patch(
            "aim2_project.aim2_ontology.parsers.detect_format_from_content"
        ) as mock_detect_content:
            with patch(
                "aim2_project.aim2_ontology.parsers.get_parser_for_format"
            ) as mock_get_parser:
                mock_detect_content.return_value = "jsonld"
                mock_parser_class = Mock()
                mock_parser_instance = Mock(spec=AbstractParser)
                mock_parser_class.return_value = mock_parser_instance
                mock_get_parser.return_value = mock_parser_class

                result = auto_detect_parser(content=sample_content)

                assert result == mock_parser_instance
                mock_detect_content.assert_called_once_with(sample_content)
                mock_get_parser.assert_called_once_with("jsonld")

    def test_file_reading_fallback(self):
        """Test auto-detection by reading file when extension detection fails."""
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".data", delete=False
        ) as temp_file:
            temp_file.write("name,age\nJohn,30")
            temp_file.flush()
            temp_path = temp_file.name

        try:
            with patch(
                "aim2_project.aim2_ontology.parsers.detect_format_from_extension"
            ) as mock_detect_ext:
                with patch(
                    "aim2_project.aim2_ontology.parsers.detect_format_from_content"
                ) as mock_detect_content:
                    with patch(
                        "aim2_project.aim2_ontology.parsers.get_parser_for_format"
                    ) as mock_get_parser:
                        mock_detect_ext.return_value = None  # Extension detection fails
                        mock_detect_content.return_value = (
                            "csv"  # Content detection succeeds
                        )
                        mock_parser_class = Mock()
                        mock_parser_instance = Mock(spec=AbstractParser)
                        mock_parser_class.return_value = mock_parser_instance
                        mock_get_parser.return_value = mock_parser_class

                        result = auto_detect_parser(file_path=temp_path)

                        assert result == mock_parser_instance
                        mock_detect_ext.assert_called_once()
                        mock_detect_content.assert_called_once()
        finally:
            Path(temp_path).unlink()

    def test_combined_detection_priority(self):
        """Test that extension detection takes priority over content detection."""
        sample_content = '{"@context": "test"}'

        with patch(
            "aim2_project.aim2_ontology.parsers.detect_format_from_extension"
        ) as mock_detect_ext:
            with patch(
                "aim2_project.aim2_ontology.parsers.detect_format_from_content"
            ) as mock_detect_content:
                with patch(
                    "aim2_project.aim2_ontology.parsers.get_parser_for_format"
                ) as mock_get_parser:
                    mock_detect_ext.return_value = "owl"  # Extension says OWL
                    mock_detect_content.return_value = "jsonld"  # Content says JSON-LD
                    mock_parser_class = Mock()
                    mock_parser_instance = Mock(spec=AbstractParser)
                    mock_parser_class.return_value = mock_parser_instance
                    mock_get_parser.return_value = mock_parser_class

                    result = auto_detect_parser(
                        file_path="test.owl", content=sample_content
                    )

                    # Should use extension-based detection (OWL)
                    mock_get_parser.assert_called_once_with("owl")
                    # Content detection should not be called when extension succeeds
                    mock_detect_content.assert_not_called()

    def test_no_parameters_error(self):
        """Test that ValueError is raised when neither file_path nor content is provided."""
        with pytest.raises(
            ValueError, match="Either file_path or content must be provided"
        ):
            auto_detect_parser()

    def test_detection_metadata(self):
        """Test that detection metadata is added to parser instance."""
        with patch(
            "aim2_project.aim2_ontology.parsers.detect_format_from_extension"
        ) as mock_detect_ext:
            with patch(
                "aim2_project.aim2_ontology.parsers.get_parser_for_format"
            ) as mock_get_parser:
                mock_detect_ext.return_value = "owl"
                mock_parser_class = Mock()
                mock_parser_instance = Mock(spec=AbstractParser)
                mock_parser_instance._detection_metadata = {}
                mock_parser_class.return_value = mock_parser_instance
                mock_get_parser.return_value = mock_parser_class

                result = auto_detect_parser(file_path="test.owl")

                # Check that detection metadata was set
                expected_metadata = {
                    "detected_format": "owl",
                    "detection_method": "extension",
                    "auto_detected": True,
                }
                assert result._detection_metadata == expected_metadata

    def test_no_format_detected(self):
        """Test handling when no format can be detected."""
        with patch(
            "aim2_project.aim2_ontology.parsers.detect_format_from_extension"
        ) as mock_detect_ext:
            with patch(
                "aim2_project.aim2_ontology.parsers.detect_format_from_content"
            ) as mock_detect_content:
                mock_detect_ext.return_value = None
                mock_detect_content.return_value = None

                result = auto_detect_parser(
                    file_path="test.unknown", content="unrecognizable"
                )

                assert result is None

    def test_no_parser_for_format(self):
        """Test handling when format is detected but no parser is available."""
        with patch(
            "aim2_project.aim2_ontology.parsers.detect_format_from_extension"
        ) as mock_detect_ext:
            with patch(
                "aim2_project.aim2_ontology.parsers.get_parser_for_format"
            ) as mock_get_parser:
                mock_detect_ext.return_value = "unsupported_format"
                mock_get_parser.return_value = None

                result = auto_detect_parser(file_path="test.unsupported")

                assert result is None

    def test_parser_instantiation_error(self):
        """Test handling of parser instantiation errors."""
        with patch(
            "aim2_project.aim2_ontology.parsers.detect_format_from_extension"
        ) as mock_detect_ext:
            with patch(
                "aim2_project.aim2_ontology.parsers.get_parser_for_format"
            ) as mock_get_parser:
                mock_detect_ext.return_value = "owl"
                mock_parser_class = Mock()
                mock_parser_class.side_effect = Exception("Parser instantiation failed")
                mock_get_parser.return_value = mock_parser_class

                result = auto_detect_parser(file_path="test.owl")

                # Should handle exception gracefully and return None
                assert result is None

    @patch("aim2_project.aim2_ontology.parsers.get_logger")
    def test_logging_behavior(self, mock_get_logger):
        """Test that appropriate logging occurs during auto-detection."""
        mock_logger = Mock()
        mock_get_logger.return_value = mock_logger

        with patch(
            "aim2_project.aim2_ontology.parsers.detect_format_from_extension"
        ) as mock_detect_ext:
            with patch(
                "aim2_project.aim2_ontology.parsers.get_parser_for_format"
            ) as mock_get_parser:
                mock_detect_ext.return_value = "owl"
                mock_parser_class = Mock()
                mock_parser_instance = Mock(spec=AbstractParser)
                mock_parser_class.return_value = mock_parser_instance
                mock_get_parser.return_value = mock_parser_class

                auto_detect_parser(file_path="test.owl")

                # Verify logging calls
                mock_logger.debug.assert_called()
                mock_logger.info.assert_called()


class TestFormatAutoDetectionIntegration:
    """Integration tests for format auto-detection workflow."""

    def test_owl_file_detection_workflow(self):
        """Test complete workflow for OWL file detection."""
        owl_content = """<?xml version="1.0"?>
<rdf:RDF xmlns:owl="http://www.w3.org/2002/07/owl#">
    <owl:Ontology rdf:about="http://example.org/test"/>
</rdf:RDF>"""

        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".owl", delete=False
        ) as temp_file:
            temp_file.write(owl_content)
            temp_file.flush()
            temp_path = temp_file.name

        try:
            # Test extension-based detection
            format_from_ext = detect_format_from_extension(temp_path)
            assert format_from_ext == "owl"

            # Test content-based detection
            detect_format_from_content(owl_content)
            # This might return different values depending on parser implementation

            # Test parser mapping
            parser_class = get_parser_for_format("owl")
            assert parser_class == OWLParser

            # Test full auto-detection
            parser = auto_detect_parser(file_path=temp_path)
            assert parser is not None
            assert isinstance(parser, AbstractParser)

        finally:
            Path(temp_path).unlink()

    def test_csv_file_detection_workflow(self):
        """Test complete workflow for CSV file detection."""
        csv_content = """name,age,city
John,30,NYC
Jane,25,LA"""

        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".csv", delete=False
        ) as temp_file:
            temp_file.write(csv_content)
            temp_file.flush()
            temp_path = temp_file.name

        try:
            # Test extension-based detection
            format_from_ext = detect_format_from_extension(temp_path)
            assert format_from_ext == "csv"

            # Test parser mapping
            parser_class = get_parser_for_format("csv")
            assert parser_class == CSVParser

            # Test full auto-detection
            parser = auto_detect_parser(file_path=temp_path)
            assert parser is not None
            assert isinstance(parser, AbstractParser)

        finally:
            Path(temp_path).unlink()

    def test_ambiguous_extension_workflow(self):
        """Test workflow when file extension is ambiguous."""
        json_content = (
            '{"@context": {"@vocab": "http://example.org/"}, "@type": "Thing"}'
        )

        # Use .txt extension which maps to txt format but content is JSON-LD
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".txt", delete=False
        ) as temp_file:
            temp_file.write(json_content)
            temp_file.flush()
            temp_path = temp_file.name

        try:
            # Extension detection should return 'txt'
            format_from_ext = detect_format_from_extension(temp_path)
            assert format_from_ext == "txt"

            # But auto-detection should potentially detect JSON-LD from content
            # when extension detection leads to wrong parser
            parser = auto_detect_parser(file_path=temp_path)
            # Result depends on implementation - might get CSVParser from extension
            # or might read file and detect JSON-LD
            assert parser is not None

        finally:
            Path(temp_path).unlink()

    def test_error_handling_workflow(self):
        """Test error handling throughout the detection workflow."""
        # Test with non-existent file
        parser = auto_detect_parser(file_path="/nonexistent/file.owl")
        # Should handle gracefully - might return OWLParser from extension or None

        # Test with malformed content
        malformed_content = "not valid content for any format"
        parser = auto_detect_parser(content=malformed_content)
        # Should handle gracefully and return None or best-guess parser


class TestFormatAutoDetectionEdgeCases:
    """Test edge cases and error conditions."""

    def test_unicode_file_paths(self):
        """Test handling of Unicode characters in file paths."""
        unicode_path = "tëst_ñämé.owl"
        result = detect_format_from_extension(unicode_path)
        assert result == "owl"

    def test_very_long_file_paths(self):
        """Test handling of very long file paths."""
        long_path = "a" * 1000 + ".csv"
        result = detect_format_from_extension(long_path)
        assert result == "csv"

    def test_multiple_extensions(self):
        """Test handling of files with multiple extensions."""
        multi_ext_path = "file.backup.owl"
        result = detect_format_from_extension(multi_ext_path)
        # Should detect based on last extension (.owl)
        assert result == "owl"  # The function detects based on the final extension

        # This should also work
        correct_path = "file.backup.csv"
        result = detect_format_from_extension(correct_path)
        assert result == "csv"

    def test_hidden_files(self):
        """Test handling of hidden files (starting with dot)."""
        hidden_file = ".hidden.owl"
        result = detect_format_from_extension(hidden_file)
        assert result == "owl"

    def test_binary_content_handling(self):
        """Test handling of binary content."""
        binary_content = b"\x00\x01\x02\x03\x04".decode("latin1")

        # Mock all parsers to return unknown/None
        with patch(
            "aim2_project.aim2_ontology.parsers.OWLParser"
        ) as mock_owl_parser_class:
            with patch(
                "aim2_project.aim2_ontology.parsers.JSONLDParser"
            ) as mock_jsonld_parser_class:
                with patch(
                    "aim2_project.aim2_ontology.parsers.CSVParser"
                ) as mock_csv_parser_class:
                    # Set up mocks to return "unknown" for binary content
                    for parser_class in [
                        mock_owl_parser_class,
                        mock_jsonld_parser_class,
                        mock_csv_parser_class,
                    ]:
                        mock_parser = Mock()
                        mock_parser.detect_format.return_value = "unknown"
                        parser_class.return_value = mock_parser

                    result = detect_format_from_content(binary_content)
                    # Should handle gracefully and return None
                    assert result is None

    def test_extremely_large_content(self):
        """Test handling of extremely large content."""
        # Create content larger than the 10KB sampling limit
        large_content = "test line\n" * 10000  # Much larger than 10KB

        with patch(
            "aim2_project.aim2_ontology.parsers.OWLParser"
        ) as mock_owl_parser_class:
            mock_parser = Mock()
            mock_parser.detect_format.return_value = "owl"
            mock_owl_parser_class.return_value = mock_parser

            result = detect_format_from_content(large_content)

            # Should handle by sampling
            assert result == "owl"

            # Verify parser was called with sampled content
            called_content = mock_parser.detect_format.call_args[0][0]
            assert len(called_content) <= 10000

    def test_concurrent_access_safety(self):
        """Test thread safety of detection functions."""
        import threading
        import time

        results = []
        errors = []

        def detect_worker():
            try:
                for i in range(10):
                    result = detect_format_from_extension(f"test_{i}.owl")
                    results.append(result)
                    time.sleep(0.001)  # Small delay to encourage race conditions
            except Exception as e:
                errors.append(e)

        # Start multiple threads
        threads = [threading.Thread(target=detect_worker) for _ in range(5)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()

        # Check results
        assert len(errors) == 0, f"Errors occurred: {errors}"
        assert len(results) == 50  # 5 threads * 10 iterations
        assert all(result == "owl" for result in results)
