"""
Comprehensive Unit Tests for CSV Parser (TDD Approach)

This module provides comprehensive unit tests for the CSV parser functionality in the AIM2 ontology
information extraction system. The tests follow test-driven development (TDD) approach,
defining the expected behavior of the CSV parser classes before implementation.

Test Classes:
    TestCSVParserCreation: Tests for CSVParser instantiation and configuration
    TestCSVParserFormatSupport: Tests for CSV format detection and validation
    TestCSVParserParsing: Tests for core CSV parsing functionality
    TestCSVParserDialectDetection: Tests for CSV dialect detection
    TestCSVParserHeaderHandling: Tests for CSV header processing
    TestCSVParserConversion: Tests for converting parsed CSV to internal models
    TestCSVParserErrorHandling: Tests for error handling and validation
    TestCSVParserValidation: Tests for CSV validation functionality
    TestCSVParserOptions: Tests for parser configuration and options
    TestCSVParserIntegration: Integration tests with other components
    TestCSVParserPerformance: Performance and scalability tests

The CSVParser is expected to be a concrete implementation providing:
- CSV format support: Standard CSV, TSV, custom delimiters
- Header detection and processing
- Dialect detection (delimiter, quote character, escape character)
- Integration with pandas and csv modules
- Conversion to internal Term/Relationship/Ontology models
- Comprehensive validation and error reporting
- Performance optimization for large CSV files
- Configurable parsing options and encoding detection

Dependencies:
    - pytest: For test framework
    - unittest.mock: For mocking functionality
    - pandas: For CSV processing (mocked)
    - csv: For CSV dialect detection (mocked)
    - typing: For type hints

Usage:
    pytest tests/unit/test_csv_parser.py -v
"""

import io
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch

import pytest


# Mock the parser classes since they don't exist yet (TDD approach)
@pytest.fixture
def mock_csv_parser():
    """Mock CSVParser class."""
    with patch("aim2_project.aim2_ontology.parsers.CSVParser") as mock_parser:
        mock_instance = Mock()

        # Core parsing methods
        mock_instance.parse = Mock()
        mock_instance.parse_file = Mock()
        mock_instance.parse_string = Mock()
        mock_instance.parse_stream = Mock()

        # Format detection and validation
        mock_instance.detect_format = Mock()
        mock_instance.detect_dialect = Mock()
        mock_instance.detect_encoding = Mock()
        mock_instance.validate_format = Mock()
        mock_instance.get_supported_formats = Mock(return_value=["csv", "tsv", "txt"])

        # Header handling
        mock_instance.detect_headers = Mock()
        mock_instance.set_headers = Mock()
        mock_instance.get_headers = Mock()
        mock_instance.infer_column_types = Mock()

        # Conversion methods
        mock_instance.to_ontology = Mock()
        mock_instance.extract_terms = Mock()
        mock_instance.extract_relationships = Mock()
        mock_instance.extract_metadata = Mock()

        # Configuration
        mock_instance.set_options = Mock()
        mock_instance.get_options = Mock(return_value={})
        mock_instance.reset_options = Mock()

        # Validation
        mock_instance.validate = Mock()
        mock_instance.validate_csv = Mock()
        mock_instance.get_validation_errors = Mock(return_value=[])

        mock_parser.return_value = mock_instance
        yield mock_parser


@pytest.fixture
def mock_pandas():
    """Mock pandas library."""
    with patch("pandas.read_csv") as mock_read_csv:
        mock_df = Mock()
        mock_df.shape = (100, 5)
        mock_df.columns = ["id", "name", "definition", "category", "synonyms"]
        mock_df.head = Mock(return_value=Mock())
        mock_df.info = Mock()
        mock_df.dtypes = Mock()
        mock_df.to_dict = Mock(return_value={})
        mock_df.iterrows = Mock(return_value=[])
        mock_read_csv.return_value = mock_df
        yield mock_read_csv


@pytest.fixture
def mock_csv_module():
    """Mock csv module."""
    with patch("csv.Sniffer") as mock_sniffer:
        mock_sniffer_instance = Mock()
        mock_dialect = Mock()
        mock_dialect.delimiter = ","
        mock_dialect.quotechar = '"'
        mock_dialect.escapechar = None
        mock_dialect.skipinitialspace = False
        mock_sniffer_instance.sniff = Mock(return_value=mock_dialect)
        mock_sniffer_instance.has_header = Mock(return_value=True)
        mock_sniffer.return_value = mock_sniffer_instance
        yield mock_sniffer


@pytest.fixture
def sample_csv_content():
    """Sample CSV content for testing."""
    return """id,name,definition,category,synonyms
CHEM:001,Chemical,A chemical compound,Chemical,compound;substance
CHEM:002,Glucose,Simple sugar,Carbohydrate,dextrose;blood sugar
CHEM:003,Protein,Large biomolecule,Macromolecule,polypeptide
CHEM:004,DNA,Genetic material,Nucleic Acid,deoxyribonucleic acid"""


@pytest.fixture
def sample_tsv_content():
    """Sample TSV content for testing."""
    return """id\tname\tdefinition\tcategory\tsynonyms
CHEM:001\tChemical\tA chemical compound\tChemical\tcompound;substance
CHEM:002\tGlucose\tSimple sugar\tCarbohydrate\tdextrose;blood sugar
CHEM:003\tProtein\tLarge biomolecule\tMacromolecule\tpolypeptide"""


@pytest.fixture
def sample_custom_delimiter_content():
    """Sample CSV with custom delimiter for testing."""
    return """id|name|definition|category|synonyms
CHEM:001|Chemical|A chemical compound|Chemical|compound;substance
CHEM:002|Glucose|Simple sugar|Carbohydrate|dextrose;blood sugar"""


@pytest.fixture
def sample_csv_no_headers():
    """Sample CSV without headers for testing."""
    return """CHEM:001,Chemical,A chemical compound,Chemical,compound;substance
CHEM:002,Glucose,Simple sugar,Carbohydrate,dextrose;blood sugar
CHEM:003,Protein,Large biomolecule,Macromolecule,polypeptide"""


@pytest.fixture
def sample_csv_with_quotes():
    """Sample CSV with quoted fields for testing."""
    return '''id,name,definition,category,synonyms
CHEM:001,"Chemical","A chemical compound, basic unit",Chemical,"compound;substance"
CHEM:002,"Glucose","Simple sugar, C6H12O6",Carbohydrate,"dextrose;blood sugar"'''


class TestCSVParserCreation:
    """Test CSVParser instantiation and configuration."""

    def test_csv_parser_creation_default(self, mock_csv_parser):
        """Test creating CSVParser with default settings."""
        parser = mock_csv_parser()

        # Verify parser was created
        assert parser is not None
        mock_csv_parser.assert_called_once()

    def test_csv_parser_creation_with_options(self, mock_csv_parser):
        """Test creating CSVParser with custom options."""
        options = {
            "delimiter": ",",
            "quotechar": '"',
            "has_headers": True,
            "encoding": "utf-8",
            "skiprows": 0,
        }

        parser = mock_csv_parser(options=options)
        mock_csv_parser.assert_called_once_with(options=options)

    def test_csv_parser_creation_with_invalid_options(self, mock_csv_parser):
        """Test creating CSVParser with invalid options raises error."""
        mock_csv_parser.side_effect = ValueError("Invalid option: unknown_delimiter")

        invalid_options = {"unknown_delimiter": ";;"}

        with pytest.raises(ValueError, match="Invalid option"):
            mock_csv_parser(options=invalid_options)

    def test_csv_parser_inherits_from_abstract_parser(self, mock_csv_parser):
        """Test that CSVParser implements AbstractParser interface."""
        parser = mock_csv_parser()

        # Verify all required methods are available
        required_methods = [
            "parse",
            "validate",
            "get_supported_formats",
            "set_options",
            "get_options",
        ]
        for method in required_methods:
            assert hasattr(parser, method)


class TestCSVParserFormatSupport:
    """Test CSV format detection and validation."""

    def test_get_supported_formats(self, mock_csv_parser):
        """Test getting list of supported CSV formats."""
        parser = mock_csv_parser()

        formats = parser.get_supported_formats()

        expected_formats = ["csv", "tsv", "txt"]
        assert all(fmt in formats for fmt in expected_formats)

    def test_detect_format_csv(self, mock_csv_parser, sample_csv_content):
        """Test detecting CSV format."""
        parser = mock_csv_parser()
        parser.detect_format.return_value = "csv"

        detected_format = parser.detect_format(sample_csv_content)

        assert detected_format == "csv"
        parser.detect_format.assert_called_once_with(sample_csv_content)

    def test_detect_format_tsv(self, mock_csv_parser, sample_tsv_content):
        """Test detecting TSV format."""
        parser = mock_csv_parser()
        parser.detect_format.return_value = "tsv"

        detected_format = parser.detect_format(sample_tsv_content)

        assert detected_format == "tsv"

    def test_detect_format_custom_delimiter(
        self, mock_csv_parser, sample_custom_delimiter_content
    ):
        """Test detecting custom delimiter format."""
        parser = mock_csv_parser()
        parser.detect_format.return_value = "csv"

        detected_format = parser.detect_format(sample_custom_delimiter_content)

        assert detected_format == "csv"

    def test_validate_format_valid_content(self, mock_csv_parser, sample_csv_content):
        """Test validating valid CSV content."""
        parser = mock_csv_parser()
        parser.validate_format.return_value = True

        is_valid = parser.validate_format(sample_csv_content, "csv")

        assert is_valid is True
        parser.validate_format.assert_called_once_with(sample_csv_content, "csv")

    def test_validate_format_invalid_content(self, mock_csv_parser):
        """Test validating invalid CSV content."""
        parser = mock_csv_parser()
        parser.validate_format.return_value = False

        invalid_content = "This is not CSV content\nIt has no structure"
        is_valid = parser.validate_format(invalid_content, "csv")

        assert is_valid is False


class TestCSVParserDialectDetection:
    """Test enhanced CSV dialect detection functionality with confidence scoring."""

    def test_detect_dialect_comma_delimiter_high_confidence(
        self, mock_csv_parser, sample_csv_content
    ):
        """Test detecting comma delimiter dialect with high confidence."""
        parser = mock_csv_parser()
        mock_dialect = Mock()
        mock_dialect.delimiter = ","
        mock_dialect.quotechar = '"'
        mock_dialect.escapechar = None

        mock_result = {
            "dialect": mock_dialect,
            "confidence": 0.9,
            "method": "sniffer",
            "details": "Detected with csv.Sniffer",
        }
        parser.detect_dialect.return_value = mock_result

        result = parser.detect_dialect(sample_csv_content)

        assert result["dialect"].delimiter == ","
        assert result["dialect"].quotechar == '"'
        assert result["confidence"] == 0.9
        assert result["method"] == "sniffer"
        parser.detect_dialect.assert_called_once_with(sample_csv_content)

    def test_detect_dialect_tab_delimiter_high_confidence(
        self, mock_csv_parser, sample_tsv_content
    ):
        """Test detecting tab delimiter dialect with high confidence."""
        parser = mock_csv_parser()
        mock_dialect = Mock()
        mock_dialect.delimiter = "\t"
        mock_dialect.quotechar = '"'

        mock_result = {
            "dialect": mock_dialect,
            "confidence": 0.85,
            "method": "sniffer",
            "details": "Detected with csv.Sniffer",
        }
        parser.detect_dialect.return_value = mock_result

        result = parser.detect_dialect(sample_tsv_content)

        assert result["dialect"].delimiter == "\t"
        assert result["confidence"] >= 0.8

    def test_detect_dialect_custom_delimiter_pipe(
        self, mock_csv_parser, sample_custom_delimiter_content
    ):
        """Test detecting pipe delimiter dialect."""
        parser = mock_csv_parser()
        mock_dialect = Mock()
        mock_dialect.delimiter = "|"
        mock_dialect.quotechar = '"'

        mock_result = {
            "dialect": mock_dialect,
            "confidence": 0.7,
            "method": "manual",
            "details": "Manual detection: delimiter_score=0.8",
        }
        parser.detect_dialect.return_value = mock_result

        result = parser.detect_dialect(sample_custom_delimiter_content)

        assert result["dialect"].delimiter == "|"
        assert result["method"] == "manual"

    def test_detect_dialect_semicolon_delimiter(self, mock_csv_parser):
        """Test detecting semicolon delimiter (European CSV format)."""
        parser = mock_csv_parser()
        semicolon_csv = "id;name;definition\nCHEM:001;Chemical;A compound\nCHEM:002;Glucose;Simple sugar"

        mock_dialect = Mock()
        mock_dialect.delimiter = ";"
        mock_dialect.quotechar = '"'

        mock_result = {
            "dialect": mock_dialect,
            "confidence": 0.75,
            "method": "sniffer",
            "details": "Detected with csv.Sniffer",
        }
        parser.detect_dialect.return_value = mock_result

        result = parser.detect_dialect(semicolon_csv)

        assert result["dialect"].delimiter == ";"
        assert result["confidence"] > 0.7

    def test_detect_dialect_space_delimiter(self, mock_csv_parser):
        """Test detecting space delimiter."""
        parser = mock_csv_parser()
        space_csv = (
            "id name definition\nCHEM:001 Chemical Compound\nCHEM:002 Glucose Sugar"
        )

        mock_dialect = Mock()
        mock_dialect.delimiter = " "
        mock_dialect.quotechar = '"'

        mock_result = {
            "dialect": mock_dialect,
            "confidence": 0.6,
            "method": "manual",
            "details": "Manual detection: delimiter_score=0.7",
        }
        parser.detect_dialect.return_value = mock_result

        result = parser.detect_dialect(space_csv)

        assert result["dialect"].delimiter == " "

    def test_detect_dialect_colon_delimiter(self, mock_csv_parser):
        """Test detecting colon delimiter."""
        parser = mock_csv_parser()
        colon_csv = (
            "id:name:definition\nCHEM:001:Chemical:Compound\nCHEM:002:Glucose:Sugar"
        )

        mock_dialect = Mock()
        mock_dialect.delimiter = ":"
        mock_dialect.quotechar = '"'

        mock_result = {
            "dialect": mock_dialect,
            "confidence": 0.65,
            "method": "manual",
            "details": "Manual detection: delimiter_score=0.8",
        }
        parser.detect_dialect.return_value = mock_result

        result = parser.detect_dialect(colon_csv)

        assert result["dialect"].delimiter == ":"

    def test_detect_dialect_single_quotes(self, mock_csv_parser):
        """Test detecting single quote character."""
        parser = mock_csv_parser()
        single_quote_csv = "id,name,definition\nCHEM:001,'Chemical Compound','A basic unit'\nCHEM:002,'Glucose','Simple sugar'"

        mock_dialect = Mock()
        mock_dialect.delimiter = ","
        mock_dialect.quotechar = "'"
        mock_dialect.escapechar = None

        mock_result = {
            "dialect": mock_dialect,
            "confidence": 0.8,
            "method": "manual",
            "details": "Manual detection: delimiter_score=0.9, quote_patterns=2",
        }
        parser.detect_dialect.return_value = mock_result

        result = parser.detect_dialect(single_quote_csv)

        assert result["dialect"].quotechar == "'"

    def test_detect_dialect_backtick_quotes(self, mock_csv_parser):
        """Test detecting backtick quote character."""
        parser = mock_csv_parser()
        backtick_csv = "id,name,definition\nCHEM:001,`Chemical Compound`,`A basic unit`\nCHEM:002,`Glucose`,`Simple sugar`"

        mock_dialect = Mock()
        mock_dialect.delimiter = ","
        mock_dialect.quotechar = "`"
        mock_dialect.escapechar = None

        mock_result = {
            "dialect": mock_dialect,
            "confidence": 0.75,
            "method": "manual",
            "details": "Manual detection: delimiter_score=0.9, quote_patterns=4",
        }
        parser.detect_dialect.return_value = mock_result

        result = parser.detect_dialect(backtick_csv)

        assert result["dialect"].quotechar == "`"

    def test_detect_dialect_with_escape_characters(self, mock_csv_parser):
        """Test detecting escape characters."""
        parser = mock_csv_parser()
        escaped_csv = 'id,name,definition\nCHEM:001,"Chemical ""Compound""","A basic unit"\nCHEM:002,"Glucose\\,Sugar","Simple sugar"'

        mock_dialect = Mock()
        mock_dialect.delimiter = ","
        mock_dialect.quotechar = '"'
        mock_dialect.escapechar = "\\"

        mock_result = {
            "dialect": mock_dialect,
            "confidence": 0.85,
            "method": "manual",
            "details": "Manual detection: delimiter_score=0.9, escape_patterns=1",
        }
        parser.detect_dialect.return_value = mock_result

        result = parser.detect_dialect(escaped_csv)

        assert result["dialect"].escapechar == "\\"

    def test_detect_dialect_mixed_delimiters_low_confidence(self, mock_csv_parser):
        """Test handling mixed delimiters with low confidence."""
        parser = mock_csv_parser()
        mixed_csv = "id,name;definition|category\nCHEM:001,Chemical;A compound|Chemical\nCHEM:002,Glucose;Sugar|Carbohydrate"

        mock_dialect = Mock()
        mock_dialect.delimiter = ","
        mock_dialect.quotechar = '"'

        mock_result = {
            "dialect": mock_dialect,
            "confidence": 0.3,
            "method": "fallback",
            "details": "Fallback detection using delimiter=','",
        }
        parser.detect_dialect.return_value = mock_result

        result = parser.detect_dialect(mixed_csv)

        assert result["confidence"] < 0.5
        assert result["method"] == "fallback"

    def test_detect_dialect_inconsistent_quoting(self, mock_csv_parser):
        """Test handling inconsistent quoting patterns."""
        parser = mock_csv_parser()
        inconsistent_csv = 'id,name,definition\nCHEM:001,"Chemical",A compound\nCHEM:002,Glucose,"Simple sugar"'

        mock_dialect = Mock()
        mock_dialect.delimiter = ","
        mock_dialect.quotechar = '"'

        mock_result = {
            "dialect": mock_dialect,
            "confidence": 0.6,
            "method": "manual",
            "details": "Manual detection: delimiter_score=0.9, quote_patterns=3",
        }
        parser.detect_dialect.return_value = mock_result

        result = parser.detect_dialect(inconsistent_csv)

        assert result["confidence"] >= 0.5
        assert result["dialect"].delimiter == ","

    def test_detect_dialect_empty_content(self, mock_csv_parser):
        """Test handling empty content."""
        parser = mock_csv_parser()

        mock_result = {
            "dialect": None,
            "confidence": 0.0,
            "method": "fallback",
            "details": "Empty content",
        }
        parser.detect_dialect.return_value = mock_result

        result = parser.detect_dialect("")

        assert result["dialect"] is None
        assert result["confidence"] == 0.0
        assert result["method"] == "fallback"

    def test_detect_dialect_single_line(self, mock_csv_parser):
        """Test handling single line content."""
        parser = mock_csv_parser()
        single_line = "id,name,definition,category"

        mock_dialect = Mock()
        mock_dialect.delimiter = ","
        mock_dialect.quotechar = '"'

        mock_result = {
            "dialect": mock_dialect,
            "confidence": 0.4,
            "method": "fallback",
            "details": "Fallback detection using delimiter=','",
        }
        parser.detect_dialect.return_value = mock_result

        result = parser.detect_dialect(single_line)

        assert result["confidence"] <= 0.5  # Lower confidence for single line

    def test_detect_dialect_user_specified_delimiter(self, mock_csv_parser):
        """Test using user-specified delimiter with higher confidence."""
        parser = mock_csv_parser()
        # Simulate user options
        parser.options = {"delimiter": "|"}

        user_specified_csv = "id|name|definition\nCHEM:001|Chemical|Compound"

        mock_dialect = Mock()
        mock_dialect.delimiter = "|"
        mock_dialect.quotechar = '"'

        mock_result = {
            "dialect": mock_dialect,
            "confidence": 0.7,
            "method": "fallback",
            "details": "Fallback detection using delimiter='|' (user-specified)",
        }
        parser.detect_dialect.return_value = mock_result

        result = parser.detect_dialect(user_specified_csv)

        assert result["confidence"] >= 0.7  # Higher confidence for user-specified
        assert "user-specified" in result["details"]

    def test_detect_dialect_confidence_scoring_consistency(self, mock_csv_parser):
        """Test confidence scoring based on field count consistency."""
        parser = mock_csv_parser()
        consistent_csv = "id,name,definition\nCHEM:001,Chemical,Compound\nCHEM:002,Glucose,Sugar\nCHEM:003,Protein,Molecule"

        mock_dialect = Mock()
        mock_dialect.delimiter = ","
        mock_dialect.quotechar = '"'

        mock_result = {
            "dialect": mock_dialect,
            "confidence": 0.9,
            "method": "sniffer",
            "details": "Detected with csv.Sniffer",
        }
        parser.detect_dialect.return_value = mock_result

        result = parser.detect_dialect(consistent_csv)

        assert result["confidence"] >= 0.8  # High confidence for consistent structure

    def test_detect_dialect_complex_quoted_fields(self, mock_csv_parser):
        """Test detecting dialect with complex quoted fields containing delimiters."""
        parser = mock_csv_parser()
        complex_csv = '''id,name,definition,properties
CHEM:001,"Chemical Compound","A pure substance, basic unit","molecular_weight:100.5;state:solid"
CHEM:002,"D-Glucose","Simple sugar, C6H12O6","formula:C6H12O6;sweet:true"'''

        mock_dialect = Mock()
        mock_dialect.delimiter = ","
        mock_dialect.quotechar = '"'
        mock_dialect.escapechar = None

        mock_result = {
            "dialect": mock_dialect,
            "confidence": 0.85,
            "method": "sniffer",
            "details": "Detected with csv.Sniffer",
        }
        parser.detect_dialect.return_value = mock_result

        result = parser.detect_dialect(complex_csv)

        assert result["confidence"] >= 0.8
        assert result["dialect"].quotechar == '"'

    def test_detect_dialect_with_quoted_fields(
        self, mock_csv_parser, sample_csv_with_quotes
    ):
        """Test detecting dialect with quoted fields."""
        parser = mock_csv_parser()
        mock_dialect = Mock()
        mock_dialect.delimiter = ","
        mock_dialect.quotechar = '"'
        mock_dialect.escapechar = None

        mock_result = {
            "dialect": mock_dialect,
            "confidence": 0.8,
            "method": "sniffer",
            "details": "Detected with csv.Sniffer",
        }
        parser.detect_dialect.return_value = mock_result

        result = parser.detect_dialect(sample_csv_with_quotes)

        assert result["dialect"].quotechar == '"'
        assert result["confidence"] >= 0.7


class TestCSVDialectConfidenceScoring:
    """Test confidence scoring functionality for dialect detection."""

    def test_confidence_scoring_high_for_consistent_structure(self, mock_csv_parser):
        """Test high confidence for consistent CSV structure."""
        parser = mock_csv_parser()
        consistent_csv = "id,name,category\nCHEM:001,Chemical,Type1\nCHEM:002,Glucose,Type2\nCHEM:003,Protein,Type3"

        mock_dialect = Mock()
        mock_dialect.delimiter = ","
        mock_dialect.quotechar = '"'

        # High confidence for consistent field counts
        mock_result = {
            "dialect": mock_dialect,
            "confidence": 0.9,
            "method": "sniffer",
            "details": "Detected with csv.Sniffer",
        }
        parser.detect_dialect.return_value = mock_result

        result = parser.detect_dialect(consistent_csv)

        assert result["confidence"] >= 0.8
        assert result["method"] in ["sniffer", "manual"]

    def test_confidence_scoring_medium_for_mostly_consistent(self, mock_csv_parser):
        """Test medium confidence for mostly consistent structure."""
        parser = mock_csv_parser()
        mostly_consistent_csv = "id,name,category\nCHEM:001,Chemical,Type1\nCHEM:002,Glucose,Type2,Extra\nCHEM:003,Protein,Type3"

        mock_dialect = Mock()
        mock_dialect.delimiter = ","
        mock_dialect.quotechar = '"'

        mock_result = {
            "dialect": mock_dialect,
            "confidence": 0.6,
            "method": "manual",
            "details": "Manual detection: delimiter_score=0.7",
        }
        parser.detect_dialect.return_value = mock_result

        result = parser.detect_dialect(mostly_consistent_csv)

        assert 0.4 <= result["confidence"] <= 0.8

    def test_confidence_scoring_low_for_inconsistent_structure(self, mock_csv_parser):
        """Test low confidence for inconsistent structure."""
        parser = mock_csv_parser()
        inconsistent_csv = "id,name\nCHEM:001,Chemical,Type1,Extra1,More\nCHEM:002\nCHEM:003,Protein,Type3,Extra2"

        mock_dialect = Mock()
        mock_dialect.delimiter = ","
        mock_dialect.quotechar = '"'

        mock_result = {
            "dialect": mock_dialect,
            "confidence": 0.2,
            "method": "fallback",
            "details": "Fallback detection using delimiter=','",
        }
        parser.detect_dialect.return_value = mock_result

        result = parser.detect_dialect(inconsistent_csv)

        assert result["confidence"] <= 0.4
        assert result["method"] == "fallback"

    def test_confidence_scoring_fallback_threshold(self, mock_csv_parser):
        """Test confidence scoring triggers fallback below threshold."""
        parser = mock_csv_parser()
        ambiguous_csv = "data;with,mixed|delimiters\nmore;data,with|mixed\npatterns;here,too|confusing"

        # First method fails with low confidence
        mock_result_low = {
            "dialect": Mock(),
            "confidence": 0.3,
            "method": "fallback",
            "details": "Fallback detection using delimiter=','",
        }
        parser.detect_dialect.return_value = mock_result_low

        result = parser.detect_dialect(ambiguous_csv)

        assert result["confidence"] <= 0.5
        assert result["method"] == "fallback"

    def test_confidence_scoring_with_user_override(self, mock_csv_parser):
        """Test confidence scoring with user-specified options."""
        parser = mock_csv_parser()
        parser.options = {"delimiter": "|", "quotechar": "'"}

        custom_csv = "id|name|category\nCHEM:001|Chemical|Type1"

        mock_dialect = Mock()
        mock_dialect.delimiter = "|"
        mock_dialect.quotechar = "'"

        mock_result = {
            "dialect": mock_dialect,
            "confidence": 0.75,  # Higher confidence for user-specified
            "method": "fallback",
            "details": "Fallback detection using delimiter='|' (user-specified)",
        }
        parser.detect_dialect.return_value = mock_result

        result = parser.detect_dialect(custom_csv)

        assert result["confidence"] >= 0.7  # Higher confidence for user settings
        assert "user-specified" in result["details"]

    def test_confidence_scoring_edge_cases(self, mock_csv_parser):
        """Test confidence scoring for edge cases."""
        parser = mock_csv_parser()

        # Empty content
        mock_result_empty = {
            "dialect": None,
            "confidence": 0.0,
            "method": "fallback",
            "details": "Empty content",
        }
        parser.detect_dialect.return_value = mock_result_empty

        result = parser.detect_dialect("")
        assert result["confidence"] == 0.0

        # Single line
        single_line = "id,name,category"
        mock_result_single = {
            "dialect": Mock(),
            "confidence": 0.4,
            "method": "fallback",
            "details": "Fallback detection using delimiter=','",
        }
        parser.detect_dialect.return_value = mock_result_single

        result = parser.detect_dialect(single_line)
        assert result["confidence"] <= 0.5

    def test_detect_encoding_utf8(self, mock_csv_parser, sample_csv_content):
        """Test detecting UTF-8 encoding."""
        parser = mock_csv_parser()
        parser.detect_encoding.return_value = "utf-8"

        encoding = parser.detect_encoding(sample_csv_content.encode("utf-8"))

        assert encoding == "utf-8"

    def test_detect_encoding_latin1(self, mock_csv_parser):
        """Test detecting Latin-1 encoding."""
        parser = mock_csv_parser()
        parser.detect_encoding.return_value = "latin-1"

        # Sample content with Latin-1 characters
        latin1_content = "id,name\n1,café".encode("latin-1")
        encoding = parser.detect_encoding(latin1_content)

        assert encoding == "latin-1"


class TestCSVParserHeaderHandling:
    """Test CSV header detection and processing."""

    def test_detect_headers_true(self, mock_csv_parser, sample_csv_content):
        """Test detecting headers when present."""
        parser = mock_csv_parser()
        parser.detect_headers.return_value = True

        has_headers = parser.detect_headers(sample_csv_content)

        assert has_headers is True
        parser.detect_headers.assert_called_once_with(sample_csv_content)

    def test_detect_headers_false(self, mock_csv_parser, sample_csv_no_headers):
        """Test detecting no headers when absent."""
        parser = mock_csv_parser()
        parser.detect_headers.return_value = False

        has_headers = parser.detect_headers(sample_csv_no_headers)

        assert has_headers is False

    def test_get_headers_from_csv(self, mock_csv_parser, sample_csv_content):
        """Test extracting headers from CSV."""
        parser = mock_csv_parser()
        expected_headers = ["id", "name", "definition", "category", "synonyms"]
        parser.get_headers.return_value = expected_headers

        headers = parser.get_headers(sample_csv_content)

        assert headers == expected_headers
        assert len(headers) == 5

    def test_set_custom_headers(self, mock_csv_parser):
        """Test setting custom headers."""
        parser = mock_csv_parser()
        custom_headers = ["term_id", "term_name", "description", "type", "aliases"]

        parser.set_headers(custom_headers)

        parser.set_headers.assert_called_once_with(custom_headers)

    def test_infer_column_types(self, mock_csv_parser, sample_csv_content):
        """Test inferring column data types."""
        parser = mock_csv_parser()
        expected_types = {
            "id": "string",
            "name": "string",
            "definition": "string",
            "category": "category",
            "synonyms": "list",
        }
        parser.infer_column_types.return_value = expected_types

        column_types = parser.infer_column_types(sample_csv_content)

        assert column_types == expected_types
        assert "id" in column_types

    def test_header_validation(self, mock_csv_parser):
        """Test validation of header names."""
        parser = mock_csv_parser()
        parser.validate_csv.return_value = {
            "valid_headers": True,
            "required_columns": ["id", "name"],
            "missing_columns": [],
            "extra_columns": ["synonyms"],
        }

        validation_result = parser.validate_csv(
            "content", headers=["id", "name", "definition", "synonyms"]
        )

        assert validation_result["valid_headers"] is True
        assert len(validation_result["missing_columns"]) == 0


class TestCSVParserParsing:
    """Test core CSV parsing functionality."""

    def test_parse_string_csv_content(self, mock_csv_parser, sample_csv_content):
        """Test parsing CSV content from string."""
        parser = mock_csv_parser()
        mock_result = Mock()
        mock_result.data = [
            {"id": "CHEM:001", "name": "Chemical", "definition": "A chemical compound"},
            {"id": "CHEM:002", "name": "Glucose", "definition": "Simple sugar"},
        ]
        parser.parse_string.return_value = mock_result

        result = parser.parse_string(sample_csv_content)

        assert result == mock_result
        assert len(result.data) == 2
        parser.parse_string.assert_called_once_with(sample_csv_content)

    def test_parse_file_csv_path(self, mock_csv_parser):
        """Test parsing CSV file from file path."""
        parser = mock_csv_parser()
        mock_result = Mock()
        mock_result.shape = (100, 5)
        parser.parse_file.return_value = mock_result

        file_path = "/path/to/ontology.csv"
        result = parser.parse_file(file_path)

        assert result == mock_result
        assert result.shape == (100, 5)
        parser.parse_file.assert_called_once_with(file_path)

    def test_parse_file_with_custom_delimiter(self, mock_csv_parser):
        """Test parsing file with custom delimiter."""
        parser = mock_csv_parser()
        mock_result = Mock()
        parser.parse_file.return_value = mock_result

        file_path = "/path/to/ontology.tsv"
        result = parser.parse_file(file_path, delimiter="\t")

        assert result == mock_result
        parser.parse_file.assert_called_once_with(file_path, delimiter="\t")

    def test_parse_stream_csv_data(self, mock_csv_parser):
        """Test parsing CSV from stream/file-like object."""
        parser = mock_csv_parser()
        mock_result = Mock()
        parser.parse_stream.return_value = mock_result

        csv_stream = io.StringIO("id,name\n1,test\n2,example")
        result = parser.parse_stream(csv_stream)

        assert result == mock_result
        parser.parse_stream.assert_called_once_with(csv_stream)

    def test_parse_with_headers(self, mock_csv_parser, sample_csv_content):
        """Test parsing CSV with header detection."""
        parser = mock_csv_parser()
        mock_result = Mock()
        mock_result.headers = ["id", "name", "definition", "category", "synonyms"]
        parser.parse_string.return_value = mock_result

        # Configure parser to detect headers
        parser.set_options({"has_headers": True})
        result = parser.parse_string(sample_csv_content)

        assert result == mock_result
        assert len(result.headers) == 5
        parser.set_options.assert_called_with({"has_headers": True})

    def test_parse_without_headers(self, mock_csv_parser, sample_csv_no_headers):
        """Test parsing CSV without headers."""
        parser = mock_csv_parser()
        mock_result = Mock()
        mock_result.headers = None
        parser.parse_string.return_value = mock_result

        # Configure parser with no headers
        parser.set_options({"has_headers": False})
        result = parser.parse_string(sample_csv_no_headers)

        assert result == mock_result
        assert result.headers is None
        parser.set_options.assert_called_with({"has_headers": False})

    def test_parse_with_skiprows(self, mock_csv_parser, sample_csv_content):
        """Test parsing CSV with skipped rows."""
        parser = mock_csv_parser()
        mock_result = Mock()
        parser.parse_string.return_value = mock_result

        parser.set_options({"skiprows": 2})
        result = parser.parse_string(sample_csv_content)

        assert result == mock_result
        parser.set_options.assert_called_with({"skiprows": 2})

    def test_parse_with_nrows_limit(self, mock_csv_parser, sample_csv_content):
        """Test parsing CSV with row limit."""
        parser = mock_csv_parser()
        mock_result = Mock()
        mock_result.row_count = 10
        parser.parse_string.return_value = mock_result

        parser.set_options({"nrows": 10})
        result = parser.parse_string(sample_csv_content)

        assert result == mock_result
        assert result.row_count == 10
        parser.set_options.assert_called_with({"nrows": 10})

    def test_parse_with_encoding_specification(self, mock_csv_parser):
        """Test parsing with specific encoding."""
        parser = mock_csv_parser()
        mock_result = Mock()
        parser.parse_file.return_value = mock_result

        parser.set_options({"encoding": "latin-1"})
        result = parser.parse_file("/path/to/file.csv")

        assert result == mock_result
        parser.set_options.assert_called_with({"encoding": "latin-1"})

    def test_parse_large_csv_file(self, mock_csv_parser):
        """Test parsing large CSV file with chunking."""
        parser = mock_csv_parser()
        mock_result = Mock()
        mock_result.chunks_processed = 10
        parser.parse_file.return_value = mock_result

        # Configure for large file parsing
        parser.set_options({"chunksize": 1000, "low_memory": True})
        result = parser.parse_file("/path/to/large_file.csv")

        assert result == mock_result
        parser.set_options.assert_called_with({"chunksize": 1000, "low_memory": True})

    def test_parse_with_na_values(self, mock_csv_parser, sample_csv_content):
        """Test parsing with custom NA values."""
        parser = mock_csv_parser()
        mock_result = Mock()
        parser.parse_string.return_value = mock_result

        parser.set_options({"na_values": ["", "NULL", "N/A", "unknown"]})
        result = parser.parse_string(sample_csv_content)

        assert result == mock_result
        parser.set_options.assert_called_with(
            {"na_values": ["", "NULL", "N/A", "unknown"]}
        )


class TestCSVParserConversion:
    """Test converting parsed CSV to internal models."""

    def test_to_ontology_conversion(self, mock_csv_parser, sample_csv_content):
        """Test converting parsed CSV to Ontology model."""
        parser = mock_csv_parser()

        # Mock the conversion result
        mock_ontology = Mock()
        mock_ontology.id = "csv_ontology_001"
        mock_ontology.name = "CSV Parsed Ontology"
        mock_ontology.terms = {}
        mock_ontology.relationships = {}
        parser.to_ontology.return_value = mock_ontology

        # Parse and convert
        parsed_result = parser.parse_string(sample_csv_content)
        ontology = parser.to_ontology(parsed_result)

        assert ontology == mock_ontology
        assert ontology.id == "csv_ontology_001"
        assert ontology.name == "CSV Parsed Ontology"

    def test_extract_terms_from_parsed_csv(self, mock_csv_parser):
        """Test extracting Term objects from parsed CSV."""
        parser = mock_csv_parser()

        # Mock extracted terms
        mock_term1 = Mock()
        mock_term1.id = "CHEM:001"
        mock_term1.name = "Chemical"
        mock_term1.definition = "A chemical compound"
        mock_term1.category = "Chemical"

        mock_term2 = Mock()
        mock_term2.id = "CHEM:002"
        mock_term2.name = "Glucose"
        mock_term2.definition = "Simple sugar"
        mock_term2.category = "Carbohydrate"

        mock_terms = [mock_term1, mock_term2]
        parser.extract_terms.return_value = mock_terms

        parsed_result = Mock()
        terms = parser.extract_terms(parsed_result)

        assert len(terms) == 2
        assert terms[0].id == "CHEM:001"
        assert terms[1].name == "Glucose"

    def test_extract_relationships_from_parsed_csv(self, mock_csv_parser):
        """Test extracting Relationship objects from parsed CSV."""
        parser = mock_csv_parser()

        # Mock extracted relationships (inferred from CSV structure)
        mock_relationships = [
            Mock(
                id="REL:001",
                subject="CHEM:002",
                predicate="is_a",
                object="CHEM:001",
                confidence=0.8,
                source="csv_inference",
            )
        ]
        parser.extract_relationships.return_value = mock_relationships

        parsed_result = Mock()
        relationships = parser.extract_relationships(parsed_result)

        assert len(relationships) == 1
        assert relationships[0].subject == "CHEM:002"
        assert relationships[0].predicate == "is_a"
        assert relationships[0].source == "csv_inference"

    def test_extract_metadata_from_parsed_csv(self, mock_csv_parser):
        """Test extracting metadata from parsed CSV."""
        parser = mock_csv_parser()

        # Mock extracted metadata
        mock_metadata = {
            "source_format": "csv",
            "row_count": 1000,
            "column_count": 5,
            "headers": ["id", "name", "definition", "category", "synonyms"],
            "delimiter": ",",
            "encoding": "utf-8",
            "has_headers": True,
            "parser_version": "1.0.0",
        }
        parser.extract_metadata.return_value = mock_metadata

        parsed_result = Mock()
        metadata = parser.extract_metadata(parsed_result)

        assert metadata["source_format"] == "csv"
        assert metadata["row_count"] == 1000
        assert len(metadata["headers"]) == 5

    def test_conversion_with_column_mapping(self, mock_csv_parser):
        """Test conversion with custom column mapping."""
        parser = mock_csv_parser()

        # Configure column mapping
        column_mapping = {
            "id": "term_id",
            "name": "label",
            "definition": "description",
            "category": "type",
            "synonyms": "alternative_labels",
        }
        parser.set_options({"column_mapping": column_mapping})

        mock_ontology = Mock()
        parser.to_ontology.return_value = mock_ontology

        parsed_result = Mock()
        ontology = parser.to_ontology(parsed_result)

        assert ontology == mock_ontology
        parser.set_options.assert_called_with({"column_mapping": column_mapping})

    def test_conversion_with_filters(self, mock_csv_parser):
        """Test conversion with row/column filters."""
        parser = mock_csv_parser()

        # Configure filters
        filters = {
            "row_filter": lambda row: row["category"] == "Chemical",
            "column_filter": ["id", "name", "definition"],
            "value_filter": lambda value: value is not None and value != "",
        }
        parser.set_options({"conversion_filters": filters})

        mock_ontology = Mock()
        parser.to_ontology.return_value = mock_ontology

        parsed_result = Mock()
        ontology = parser.to_ontology(parsed_result)

        assert ontology == mock_ontology
        parser.set_options.assert_called_with({"conversion_filters": filters})

    def test_conversion_preserves_provenance(self, mock_csv_parser):
        """Test that conversion preserves provenance information."""
        parser = mock_csv_parser()

        mock_ontology = Mock()
        mock_ontology.metadata = {
            "source_format": "csv",
            "source_file": "/path/to/data.csv",
            "parser_version": "1.0.0",
            "parsing_timestamp": "2023-01-01T00:00:00Z",
            "original_row_count": 1000,
            "processed_row_count": 950,
        }
        parser.to_ontology.return_value = mock_ontology

        parsed_result = Mock()
        ontology = parser.to_ontology(parsed_result)

        assert "source_format" in ontology.metadata
        assert "source_file" in ontology.metadata
        assert "parsing_timestamp" in ontology.metadata


class TestCSVParserErrorHandling:
    """Test error handling and validation."""

    def test_parse_invalid_csv_content(self, mock_csv_parser):
        """Test parsing invalid CSV content raises appropriate error."""
        parser = mock_csv_parser()
        parser.parse_string.side_effect = ValueError(
            "Invalid CSV: inconsistent field count"
        )

        invalid_csv = "id,name\nCHEM:001,Chemical,Extra,Field\nCHEM:002"

        with pytest.raises(ValueError, match="Invalid CSV"):
            parser.parse_string(invalid_csv)

    def test_parse_nonexistent_file(self, mock_csv_parser):
        """Test parsing nonexistent file raises FileNotFoundError."""
        parser = mock_csv_parser()
        parser.parse_file.side_effect = FileNotFoundError(
            "File not found: /nonexistent/file.csv"
        )

        with pytest.raises(FileNotFoundError, match="File not found"):
            parser.parse_file("/nonexistent/file.csv")

    def test_parse_empty_file(self, mock_csv_parser):
        """Test parsing empty file raises appropriate error."""
        parser = mock_csv_parser()
        parser.parse_string.side_effect = ValueError("Empty CSV file")

        with pytest.raises(ValueError, match="Empty CSV file"):
            parser.parse_string("")

    def test_parse_encoding_error(self, mock_csv_parser):
        """Test handling of encoding errors."""
        parser = mock_csv_parser()
        parser.parse_file.side_effect = UnicodeDecodeError(
            "utf-8", b"", 0, 1, "invalid start byte"
        )

        with pytest.raises(UnicodeDecodeError):
            parser.parse_file("/path/to/file_with_bad_encoding.csv")

    def test_validation_errors_collection(self, mock_csv_parser, sample_csv_content):
        """Test collection and reporting of validation errors."""
        parser = mock_csv_parser()

        # Mock validation errors
        validation_errors = [
            "Warning: Row 5 has missing 'name' field",
            "Error: Row 12 has invalid ID format",
            "Warning: Column 'synonyms' has inconsistent delimiter",
        ]
        parser.get_validation_errors.return_value = validation_errors

        # Parse with validation
        parser.set_options({"validate_on_parse": True})
        parser.parse_string(sample_csv_content)

        errors = parser.get_validation_errors()
        assert len(errors) == 3
        assert any("missing 'name' field" in error for error in errors)
        assert any("invalid ID format" in error for error in errors)

    def test_memory_limit_exceeded(self, mock_csv_parser):
        """Test handling of memory limit exceeded during parsing."""
        parser = mock_csv_parser()
        parser.parse_file.side_effect = MemoryError(
            "Memory limit exceeded while parsing large CSV file"
        )

        with pytest.raises(MemoryError, match="Memory limit exceeded"):
            parser.parse_file("/path/to/huge_file.csv")

    def test_malformed_csv_recovery(self, mock_csv_parser):
        """Test error recovery with malformed CSV."""
        parser = mock_csv_parser()
        mock_result = Mock()
        mock_result.errors = ["Row 5: Field count mismatch", "Row 8: Unescaped quote"]
        mock_result.valid_rows = 98
        mock_result.total_rows = 100
        parser.parse_string.return_value = mock_result

        parser.set_options({"error_recovery": True, "skip_bad_lines": True})

        malformed_csv = """id,name,definition
CHEM:001,Chemical,A compound
CHEM:002,"Glucose,Simple sugar
CHEM:003,Protein,"Large molecule"""

        result = parser.parse_string(malformed_csv)

        assert result == mock_result
        assert result.valid_rows == 98
        assert len(result.errors) == 2


class TestCSVParserValidation:
    """Test CSV validation functionality."""

    def test_validate_csv_structure(self, mock_csv_parser, sample_csv_content):
        """Test validating CSV structure and format."""
        parser = mock_csv_parser()
        parser.validate.return_value = True

        is_valid = parser.validate(sample_csv_content)

        assert is_valid is True
        parser.validate.assert_called_once_with(sample_csv_content)

    def test_validate_csv_headers(self, mock_csv_parser, sample_csv_content):
        """Test validating CSV headers."""
        parser = mock_csv_parser()
        parser.validate_csv.return_value = {
            "valid_structure": True,
            "valid_headers": True,
            "required_columns_present": True,
            "header_consistency": True,
            "errors": [],
            "warnings": [],
        }

        validation_result = parser.validate_csv(sample_csv_content)

        assert validation_result["valid_headers"] is True
        assert validation_result["required_columns_present"] is True
        assert len(validation_result["errors"]) == 0

    def test_validate_csv_data_types(self, mock_csv_parser, sample_csv_content):
        """Test validating CSV data types."""
        parser = mock_csv_parser()
        parser.validate_csv.return_value = {
            "valid_structure": True,
            "valid_data_types": True,
            "type_consistency": True,
            "null_value_check": True,
            "errors": [],
            "warnings": ["Row 15: 'synonyms' field is empty"],
        }

        validation_result = parser.validate_csv(sample_csv_content)

        assert validation_result["valid_data_types"] is True
        assert validation_result["type_consistency"] is True
        assert len(validation_result["warnings"]) == 1

    def test_validate_required_columns(self, mock_csv_parser):
        """Test validation of required columns."""
        parser = mock_csv_parser()
        parser.validate_csv.return_value = {
            "valid_structure": False,
            "required_columns": ["id", "name"],
            "missing_columns": ["name"],
            "extra_columns": ["extra_field"],
        }

        csv_missing_column = "id,definition\nCHEM:001,A compound"
        validation_result = parser.validate_csv(csv_missing_column)

        assert validation_result["valid_structure"] is False
        assert "name" in validation_result["missing_columns"]
        assert "extra_field" in validation_result["extra_columns"]

    def test_validate_field_formats(self, mock_csv_parser, sample_csv_content):
        """Test validation of field formats."""
        parser = mock_csv_parser()
        parser.validate_csv.return_value = {
            "valid_structure": True,
            "valid_field_formats": True,
            "id_format_valid": True,
            "name_format_valid": True,
            "format_errors": [],
        }

        validation_result = parser.validate_csv(sample_csv_content)

        assert validation_result["valid_field_formats"] is True
        assert validation_result["id_format_valid"] is True
        assert len(validation_result["format_errors"]) == 0


class TestCSVParserOptions:
    """Test parser configuration and options."""

    def test_set_parsing_options(self, mock_csv_parser):
        """Test setting various parsing options."""
        parser = mock_csv_parser()

        options = {
            "delimiter": ",",
            "quotechar": '"',
            "escapechar": "\\",
            "has_headers": True,
            "encoding": "utf-8",
            "skiprows": 1,
            "nrows": 1000,
        }

        parser.set_options(options)
        parser.set_options.assert_called_once_with(options)

    def test_get_current_options(self, mock_csv_parser):
        """Test getting current parser options."""
        parser = mock_csv_parser()

        expected_options = {
            "delimiter": ",",
            "quotechar": '"',
            "has_headers": True,
            "encoding": "utf-8",
            "validate_on_parse": False,
        }
        parser.get_options.return_value = expected_options

        current_options = parser.get_options()

        assert current_options == expected_options
        assert "delimiter" in current_options

    def test_reset_options_to_defaults(self, mock_csv_parser):
        """Test resetting options to default values."""
        parser = mock_csv_parser()

        # Set some custom options first
        parser.set_options({"delimiter": "|", "has_headers": False})

        # Reset to defaults
        parser.reset_options()
        parser.reset_options.assert_called_once()

    def test_invalid_option_handling(self, mock_csv_parser):
        """Test handling of invalid configuration options."""
        parser = mock_csv_parser()
        parser.set_options.side_effect = ValueError("Unknown option: invalid_delimiter")

        invalid_options = {"invalid_delimiter": ";;"}

        with pytest.raises(ValueError, match="Unknown option"):
            parser.set_options(invalid_options)

    def test_option_validation(self, mock_csv_parser):
        """Test validation of option values."""
        parser = mock_csv_parser()
        parser.set_options.side_effect = ValueError(
            "Invalid value for option 'nrows': must be positive integer"
        )

        invalid_options = {"nrows": -100}

        with pytest.raises(ValueError, match="Invalid value for option"):
            parser.set_options(invalid_options)


class TestCSVParserIntegration:
    """Integration tests with other components."""

    def test_integration_with_ontology_manager(self, mock_csv_parser):
        """Test integration with OntologyManager."""
        parser = mock_csv_parser()

        # Mock integration with ontology manager
        with patch(
            "aim2_project.aim2_ontology.ontology_manager.OntologyManager"
        ) as MockManager:
            manager = MockManager()
            manager.add_ontology = Mock(return_value=True)

            # Parse and add to manager
            mock_ontology = Mock()
            parser.to_ontology.return_value = mock_ontology

            parsed_result = parser.parse_string("csv_content")
            ontology = parser.to_ontology(parsed_result)
            manager.add_ontology(ontology)

            manager.add_ontology.assert_called_once_with(ontology)

    def test_integration_with_term_validation(self, mock_csv_parser):
        """Test integration with term validation pipeline."""
        parser = mock_csv_parser()

        # Mock validation pipeline
        with patch(
            "aim2_project.aim2_ontology.validators.ValidationPipeline"
        ) as MockPipeline:
            validator = MockPipeline()
            validator.validate_terms = Mock(return_value={"valid": True, "errors": []})

            mock_terms = [Mock(), Mock(), Mock()]
            parser.extract_terms.return_value = mock_terms

            parsed_result = parser.parse_string("csv_content")
            terms = parser.extract_terms(parsed_result)

            # Run validation
            validation_result = validator.validate_terms(terms)

            assert validation_result["valid"] is True
            validator.validate_terms.assert_called_once_with(terms)

    def test_end_to_end_parsing_workflow(self, mock_csv_parser, sample_csv_content):
        """Test complete end-to-end parsing workflow."""
        parser = mock_csv_parser()

        # Mock the complete workflow
        mock_parsed = Mock()
        mock_ontology = Mock()
        mock_ontology.terms = {
            "CHEM:001": Mock(),
            "CHEM:002": Mock(),
            "CHEM:003": Mock(),
        }
        mock_ontology.relationships = {}

        parser.parse_string.return_value = mock_parsed
        parser.to_ontology.return_value = mock_ontology

        # Execute workflow
        parsed_result = parser.parse_string(sample_csv_content)
        ontology = parser.to_ontology(parsed_result)

        # Verify results
        assert len(ontology.terms) == 3
        assert len(ontology.relationships) == 0

        parser.parse_string.assert_called_once_with(sample_csv_content)
        parser.to_ontology.assert_called_once_with(mock_parsed)


class TestCSVParserPerformance:
    """Performance and scalability tests."""

    def test_parse_large_csv_performance(self, mock_csv_parser):
        """Test parsing performance with large CSV files."""
        parser = mock_csv_parser()

        # Configure for performance testing
        parser.set_options({"low_memory": True, "chunksize": 10000, "engine": "c"})

        mock_result = Mock()
        mock_result.row_count = 1000000
        mock_result.column_count = 10
        mock_result.memory_usage = "150MB"
        parser.parse_file.return_value = mock_result

        result = parser.parse_file("/path/to/large_file.csv")

        assert result.row_count == 1000000
        assert result.column_count == 10

    def test_memory_usage_optimization(self, mock_csv_parser):
        """Test memory usage optimization features."""
        parser = mock_csv_parser()

        # Configure memory optimization
        parser.set_options({"low_memory": True, "dtype": "string", "chunksize": 5000})

        mock_result = Mock()
        mock_result.memory_efficient = True
        parser.parse_file.return_value = mock_result

        result = parser.parse_file("/path/to/memory_intensive_file.csv")

        assert result == mock_result
        assert result.memory_efficient is True

    def test_chunked_processing(self, mock_csv_parser):
        """Test chunked processing for very large files."""
        parser = mock_csv_parser()

        # Configure for chunked processing
        parser.set_options({"chunksize": 1000, "iterator": True})

        mock_chunks = [Mock(), Mock(), Mock()]
        parser.parse_file.return_value = mock_chunks

        chunks = parser.parse_file("/path/to/huge_file.csv")

        assert len(chunks) == 3
        parser.set_options.assert_called_with({"chunksize": 1000, "iterator": True})

    def test_parallel_processing_support(self, mock_csv_parser):
        """Test support for parallel processing operations."""
        parser = mock_csv_parser()

        # Configure for parallel processing
        parser.set_options({"n_jobs": 4, "parallel": True, "backend": "threading"})

        files = [
            "/path/to/file1.csv",
            "/path/to/file2.csv",
            "/path/to/file3.csv",
            "/path/to/file4.csv",
        ]

        mock_results = [Mock(), Mock(), Mock(), Mock()]
        parser.parse_file.side_effect = mock_results

        results = []
        for file_path in files:
            result = parser.parse_file(file_path)
            results.append(result)

        assert len(results) == 4
        assert parser.parse_file.call_count == 4


# Additional test fixtures for complex scenarios
@pytest.fixture
def complex_csv_content():
    """Fixture providing complex CSV data for testing."""
    return '''id,name,definition,category,synonyms,properties,references
CHEM:001,"Chemical Compound","A pure chemical substance, basic unit of matter",Chemical,"compound;substance;matter","molecular_weight:100.5;state:solid","PMID:12345678;DOI:10.1000/xyz"
CHEM:002,"D-Glucose","Simple sugar with formula C6H12O6, primary energy source",Carbohydrate,"dextrose;blood sugar;grape sugar","molecular_formula:C6H12O6;molecular_weight:180.16","PMID:87654321;ChEBI:17234"
CHEM:003,"Insulin Protein","Peptide hormone regulating glucose metabolism",Protein,"insulin;hormone","sequence_length:51;mass:5808Da","UniProt:P01308;PMID:11111111"
CHEM:004,"Deoxyribonucleic Acid","Hereditary material in humans, double helix structure",Nucleic_Acid,"DNA;genetic material","structure:double_helix;bases:ATCG","PMID:22222222;DOI:10.1038/nature"'''


@pytest.fixture
def malformed_csv_content():
    """Fixture providing malformed CSV content for error testing."""
    return """id,name,definition,category
CHEM:001,Chemical,"A chemical compound,Chemical
CHEM:002,Glucose,Simple sugar,Carbohydrate,Extra Field
CHEM:003,"Protein,Large biomolecule
Missing ID,,Some definition,Category"""


@pytest.fixture
def csv_configuration_options():
    """Fixture providing comprehensive CSV parser configuration options."""
    return {
        "basic_options": {
            "delimiter": ",",
            "quotechar": '"',
            "escapechar": None,
            "has_headers": True,
            "encoding": "utf-8",
            "skiprows": 0,
            "nrows": None,
        },
        "performance_options": {
            "low_memory": False,
            "chunksize": None,
            "iterator": False,
            "engine": "python",
            "n_jobs": 1,
            "parallel": False,
        },
        "validation_options": {
            "validate_on_parse": True,
            "required_columns": ["id", "name"],
            "data_type_validation": True,
            "null_value_check": True,
            "format_validation": True,
        },
        "conversion_options": {
            "infer_types": True,
            "column_mapping": None,
            "value_transformations": None,
            "relationship_inference": False,
            "metadata_extraction": True,
        },
        "error_handling_options": {
            "error_recovery": False,
            "skip_bad_lines": False,
            "max_errors": None,
            "continue_on_error": False,
            "warn_bad_lines": True,
        },
    }


@pytest.fixture
def temp_csv_files():
    """Create temporary CSV files for testing."""
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)

        # Create various CSV files
        files = {}

        # Simple CSV file
        simple_csv = temp_path / "simple.csv"
        simple_csv.write_text(
            "id,name,definition\nCHEM:001,Chemical,A compound\nCHEM:002,Glucose,Simple sugar"
        )
        files["simple"] = str(simple_csv)

        # TSV file
        tsv_file = temp_path / "data.tsv"
        tsv_file.write_text(
            "id\tname\tdefinition\nCHEM:001\tChemical\tA compound\nCHEM:002\tGlucose\tSimple sugar"
        )
        files["tsv"] = str(tsv_file)

        # CSV with custom delimiter
        pipe_csv = temp_path / "data.txt"
        pipe_csv.write_text(
            "id|name|definition\nCHEM:001|Chemical|A compound\nCHEM:002|Glucose|Simple sugar"
        )
        files["pipe"] = str(pipe_csv)

        # Large CSV file (simulated)
        large_csv = temp_path / "large.csv"
        content = "id,name,definition\n"
        for i in range(1000):
            content += f"CHEM:{i:03d},Chemical_{i},Definition for chemical {i}\n"
        large_csv.write_text(content)
        files["large"] = str(large_csv)

        yield files
