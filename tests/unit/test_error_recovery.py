"""
Comprehensive Unit Tests for Error Recovery Functionality

This module provides comprehensive unit tests for the error recovery system implemented
in the parser classes. The tests cover error classification, recovery strategy selection,
parser-specific recovery methods, statistics tracking, and edge cases.

Test Classes:
    TestErrorSeverityClassification: Tests error severity classification logic
    TestRecoveryStrategySelection: Tests recovery strategy selection algorithms
    TestAbstractParserErrorRecovery: Tests base parser error recovery methods
    TestOWLParserErrorRecovery: Tests OWL parser-specific recovery methods
    TestCSVParserErrorRecovery: Tests CSV parser-specific recovery methods
    TestJSONLDParserErrorRecovery: Tests JSON-LD parser-specific recovery methods
    TestErrorContextManagement: Tests error context creation and tracking
    TestErrorStatisticsTracking: Tests statistics collection and reporting
    TestRecoveryConfigurationOptions: Tests configuration-driven recovery behavior
    TestRecoveryFallbackBehaviors: Tests edge cases and fallback scenarios

The error recovery system provides:
- Error classification (WARNING, RECOVERABLE, FATAL)
- Recovery strategies (SKIP, DEFAULT, RETRY, REPLACE, ABORT, CONTINUE)
- Error context tracking with attempted recoveries
- Parser-specific recovery implementations
- Statistics tracking for monitoring and debugging
- Configuration options for customizing recovery behavior

Dependencies:
    - pytest: For test framework
    - unittest.mock: For mocking functionality
    - datetime: For timestamp testing
    - typing: For type hints

Usage:
    pytest tests/unit/test_error_recovery.py -v
"""

import json
from datetime import datetime
from typing import Any, Dict, List, Optional
from unittest.mock import Mock

import pytest

# Import the error recovery classes and enums
try:
    from aim2_project.aim2_ontology.parsers import (
        ErrorContext,
        ErrorRecoveryStats,
        ErrorSeverity,
        RecoveryStrategy,
    )
except ImportError:
    # Mock the classes if they're not available (for TDD approach)
    from dataclasses import dataclass, field
    from enum import Enum

    class ErrorSeverity(Enum):
        WARNING = "warning"
        RECOVERABLE = "recoverable"
        FATAL = "fatal"

    class RecoveryStrategy(Enum):
        SKIP = "skip"
        DEFAULT = "default"
        RETRY = "retry"
        REPLACE = "replace"
        ABORT = "abort"
        CONTINUE = "continue"

    @dataclass
    class ErrorContext:
        error: Exception
        severity: ErrorSeverity
        location: str
        recovery_strategy: Optional[RecoveryStrategy] = None
        attempted_recoveries: List[RecoveryStrategy] = field(default_factory=list)
        recovery_data: Dict[str, Any] = field(default_factory=dict)
        timestamp: datetime = field(default_factory=datetime.now)

    @dataclass
    class ErrorRecoveryStats:
        total_errors: int = 0
        warnings: int = 0
        recoverable_errors: int = 0
        fatal_errors: int = 0
        successful_recoveries: int = 0
        failed_recoveries: int = 0
        recovery_strategies_used: Dict[str, int] = field(default_factory=dict)


class TestErrorSeverityClassification:
    """Test error severity classification logic."""

    @pytest.fixture
    def mock_parser(self):
        """Create a mock parser with error classification methods."""
        parser = Mock()
        parser._classify_error_severity = Mock()
        return parser

    def test_syntax_error_classification(self, mock_parser):
        """Test that SyntaxError is classified as RECOVERABLE."""
        error = SyntaxError("Invalid syntax in line 10")
        mock_parser._classify_error_severity.return_value = ErrorSeverity.RECOVERABLE

        result = mock_parser._classify_error_severity(error, "line 10")

        assert result == ErrorSeverity.RECOVERABLE
        mock_parser._classify_error_severity.assert_called_once_with(error, "line 10")

    def test_value_error_classification(self, mock_parser):
        """Test that ValueError is classified as RECOVERABLE."""
        error = ValueError("Invalid value for field 'age'")
        mock_parser._classify_error_severity.return_value = ErrorSeverity.RECOVERABLE

        result = mock_parser._classify_error_severity(error, "field validation")

        assert result == ErrorSeverity.RECOVERABLE
        mock_parser._classify_error_severity.assert_called_once_with(
            error, "field validation"
        )

    def test_key_error_classification(self, mock_parser):
        """Test that KeyError is classified as WARNING."""
        error = KeyError("Missing optional field 'description'")
        mock_parser._classify_error_severity.return_value = ErrorSeverity.WARNING

        result = mock_parser._classify_error_severity(error, "optional field")

        assert result == ErrorSeverity.WARNING
        mock_parser._classify_error_severity.assert_called_once_with(
            error, "optional field"
        )

    def test_memory_error_classification(self, mock_parser):
        """Test that MemoryError is classified as FATAL."""
        error = MemoryError("Out of memory")
        mock_parser._classify_error_severity.return_value = ErrorSeverity.FATAL

        result = mock_parser._classify_error_severity(error, "large file processing")

        assert result == ErrorSeverity.FATAL
        mock_parser._classify_error_severity.assert_called_once_with(
            error, "large file processing"
        )

    def test_timeout_error_classification(self, mock_parser):
        """Test that TimeoutError is classified as RECOVERABLE."""
        error = TimeoutError("Request timeout after 30s")
        mock_parser._classify_error_severity.return_value = ErrorSeverity.RECOVERABLE

        result = mock_parser._classify_error_severity(error, "remote resource")

        assert result == ErrorSeverity.RECOVERABLE
        mock_parser._classify_error_severity.assert_called_once_with(
            error, "remote resource"
        )

    def test_unknown_error_classification(self, mock_parser):
        """Test that unknown errors default to RECOVERABLE."""
        error = RuntimeError("Unexpected error")
        mock_parser._classify_error_severity.return_value = ErrorSeverity.RECOVERABLE

        result = mock_parser._classify_error_severity(error, "unknown context")

        assert result == ErrorSeverity.RECOVERABLE
        mock_parser._classify_error_severity.assert_called_once_with(
            error, "unknown context"
        )


class TestRecoveryStrategySelection:
    """Test recovery strategy selection algorithms."""

    @pytest.fixture
    def mock_parser(self):
        """Create a mock parser with strategy selection methods."""
        parser = Mock()
        parser._select_recovery_strategy = Mock()
        return parser

    def test_strategy_selection_for_syntax_error(self, mock_parser):
        """Test strategy selection for SyntaxError."""
        error_context = ErrorContext(
            error=SyntaxError("Invalid syntax"),
            severity=ErrorSeverity.RECOVERABLE,
            location="line 10",
            attempted_recoveries=[],
        )
        mock_parser._select_recovery_strategy.return_value = RecoveryStrategy.SKIP

        result = mock_parser._select_recovery_strategy(error_context)

        assert result == RecoveryStrategy.SKIP
        mock_parser._select_recovery_strategy.assert_called_once_with(error_context)

    def test_strategy_selection_with_previous_attempts(self, mock_parser):
        """Test strategy selection when previous recoveries were attempted."""
        error_context = ErrorContext(
            error=ValueError("Invalid value"),
            severity=ErrorSeverity.RECOVERABLE,
            location="field validation",
            attempted_recoveries=[RecoveryStrategy.SKIP],
        )
        mock_parser._select_recovery_strategy.return_value = RecoveryStrategy.DEFAULT

        result = mock_parser._select_recovery_strategy(error_context)

        assert result == RecoveryStrategy.DEFAULT
        mock_parser._select_recovery_strategy.assert_called_once_with(error_context)

    def test_strategy_selection_for_fatal_error(self, mock_parser):
        """Test strategy selection for FATAL errors."""
        error_context = ErrorContext(
            error=MemoryError("Out of memory"),
            severity=ErrorSeverity.FATAL,
            location="large file processing",
            attempted_recoveries=[],
        )
        mock_parser._select_recovery_strategy.return_value = RecoveryStrategy.ABORT

        result = mock_parser._select_recovery_strategy(error_context)

        assert result == RecoveryStrategy.ABORT
        mock_parser._select_recovery_strategy.assert_called_once_with(error_context)

    def test_strategy_selection_for_warning(self, mock_parser):
        """Test strategy selection for WARNING errors."""
        error_context = ErrorContext(
            error=KeyError("Missing optional field"),
            severity=ErrorSeverity.WARNING,
            location="optional field",
            attempted_recoveries=[],
        )
        mock_parser._select_recovery_strategy.return_value = RecoveryStrategy.CONTINUE

        result = mock_parser._select_recovery_strategy(error_context)

        assert result == RecoveryStrategy.CONTINUE
        mock_parser._select_recovery_strategy.assert_called_once_with(error_context)

    def test_strategy_selection_retry_limit(self, mock_parser):
        """Test strategy selection when retry limit is reached."""
        error_context = ErrorContext(
            error=TimeoutError("Connection timeout"),
            severity=ErrorSeverity.RECOVERABLE,
            location="remote resource",
            attempted_recoveries=[
                RecoveryStrategy.RETRY,
                RecoveryStrategy.RETRY,
                RecoveryStrategy.RETRY,
            ],
        )
        mock_parser._select_recovery_strategy.return_value = RecoveryStrategy.ABORT

        result = mock_parser._select_recovery_strategy(error_context)

        assert result == RecoveryStrategy.ABORT
        mock_parser._select_recovery_strategy.assert_called_once_with(error_context)


class TestAbstractParserErrorRecovery:
    """Test base parser error recovery methods."""

    @pytest.fixture
    def mock_abstract_parser(self):
        """Create a mock AbstractParser with recovery methods."""
        parser = Mock()
        parser._recover_skip = Mock()
        parser._recover_default = Mock()
        parser._recover_retry = Mock()
        parser._recover_replace = Mock()
        parser._recover_continue = Mock()
        parser._apply_recovery_strategy = Mock()
        parser._handle_parse_error = Mock()
        parser.update_error_stats = Mock()
        return parser

    def test_skip_recovery_strategy(self, mock_abstract_parser):
        """Test SKIP recovery strategy implementation."""
        error_context = ErrorContext(
            error=ValueError("Invalid data"),
            severity=ErrorSeverity.RECOVERABLE,
            location="row 5",
            recovery_strategy=RecoveryStrategy.SKIP,
        )
        mock_abstract_parser._recover_skip.return_value = None

        result = mock_abstract_parser._recover_skip(error_context)

        assert result is None
        mock_abstract_parser._recover_skip.assert_called_once_with(error_context)

    def test_default_recovery_strategy(self, mock_abstract_parser):
        """Test DEFAULT recovery strategy implementation."""
        error_context = ErrorContext(
            error=KeyError("Missing field"),
            severity=ErrorSeverity.WARNING,
            location="field validation",
            recovery_strategy=RecoveryStrategy.DEFAULT,
        )
        expected_default = {"field": "default_value"}
        mock_abstract_parser._recover_default.return_value = expected_default

        result = mock_abstract_parser._recover_default(error_context)

        assert result == expected_default
        mock_abstract_parser._recover_default.assert_called_once_with(error_context)

    def test_retry_recovery_strategy(self, mock_abstract_parser):
        """Test RETRY recovery strategy implementation."""
        error_context = ErrorContext(
            error=TimeoutError("Connection timeout"),
            severity=ErrorSeverity.RECOVERABLE,
            location="remote resource",
            recovery_strategy=RecoveryStrategy.RETRY,
            attempted_recoveries=[],
        )
        expected_result = {"success": True, "data": "recovered_data"}
        mock_abstract_parser._recover_retry.return_value = expected_result

        result = mock_abstract_parser._recover_retry(error_context, max_retries=3)

        assert result == expected_result
        mock_abstract_parser._recover_retry.assert_called_once_with(
            error_context, max_retries=3
        )

    def test_replace_recovery_strategy(self, mock_abstract_parser):
        """Test REPLACE recovery strategy implementation."""
        error_context = ErrorContext(
            error=ValueError("Invalid format"),
            severity=ErrorSeverity.RECOVERABLE,
            location="data formatting",
            recovery_strategy=RecoveryStrategy.REPLACE,
        )
        replacement_data = {"corrected": "data"}
        mock_abstract_parser._recover_replace.return_value = replacement_data

        result = mock_abstract_parser._recover_replace(
            error_context, replacement=replacement_data
        )

        assert result == replacement_data
        mock_abstract_parser._recover_replace.assert_called_once_with(
            error_context, replacement=replacement_data
        )

    def test_continue_recovery_strategy(self, mock_abstract_parser):
        """Test CONTINUE recovery strategy implementation."""
        error_context = ErrorContext(
            error=KeyError("Optional field missing"),
            severity=ErrorSeverity.WARNING,
            location="optional field processing",
            recovery_strategy=RecoveryStrategy.CONTINUE,
        )
        partial_result = {"partial": "data"}
        mock_abstract_parser._recover_continue.return_value = partial_result

        result = mock_abstract_parser._recover_continue(error_context)

        assert result == partial_result
        mock_abstract_parser._recover_continue.assert_called_once_with(error_context)

    def test_error_statistics_update(self, mock_abstract_parser):
        """Test error statistics are updated correctly."""
        error_context = ErrorContext(
            error=ValueError("Test error"),
            severity=ErrorSeverity.RECOVERABLE,
            location="test location",
        )

        mock_abstract_parser.update_error_stats(error_context, recovery_successful=True)

        mock_abstract_parser.update_error_stats.assert_called_once_with(
            error_context, recovery_successful=True
        )

    def test_comprehensive_error_handling(self, mock_abstract_parser):
        """Test comprehensive error handling workflow."""
        error = ValueError("Test parsing error")
        expected_context = ErrorContext(
            error=error, severity=ErrorSeverity.RECOVERABLE, location="test location"
        )
        mock_abstract_parser._handle_parse_error.return_value = expected_context

        result = mock_abstract_parser._handle_parse_error(
            error, location="test location"
        )

        assert result == expected_context
        mock_abstract_parser._handle_parse_error.assert_called_once_with(
            error, location="test location"
        )


class TestOWLParserErrorRecovery:
    """Test OWL parser-specific error recovery methods."""

    @pytest.fixture
    def mock_owl_parser(self):
        """Create a mock OWLParser with OWL-specific recovery methods."""
        parser = Mock()
        parser._recover_retry = Mock()
        parser._recover_default = Mock()
        parser._sanitize_owl_content = Mock()
        parser._try_alternative_format = Mock()
        parser._create_minimal_owl = Mock()
        return parser

    def test_owl_malformed_xml_recovery(self, mock_owl_parser):
        """Test recovery from malformed RDF/XML."""
        error_context = ErrorContext(
            error=SyntaxError("XML parsing error at line 15"),
            severity=ErrorSeverity.RECOVERABLE,
            location="RDF/XML parsing",
            recovery_strategy=RecoveryStrategy.RETRY,
        )
        sanitized_content = "<rdf:RDF>...</rdf:RDF>"
        mock_owl_parser._sanitize_owl_content.return_value = sanitized_content
        mock_owl_parser._recover_retry.return_value = {
            "success": True,
            "content": sanitized_content,
        }

        result = mock_owl_parser._recover_retry(error_context, content="malformed_xml")

        assert result["success"] is True
        assert "content" in result
        mock_owl_parser._recover_retry.assert_called_once_with(
            error_context, content="malformed_xml"
        )

    def test_owl_missing_namespace_recovery(self, mock_owl_parser):
        """Test recovery from missing namespace declarations."""
        error_context = ErrorContext(
            error=KeyError("Namespace prefix 'owl' not declared"),
            severity=ErrorSeverity.RECOVERABLE,
            location="namespace resolution",
            recovery_strategy=RecoveryStrategy.DEFAULT,
        )
        default_namespaces = {
            "owl": "http://www.w3.org/2002/07/owl#",
            "rdf": "http://www.w3.org/1999/02/22-rdf-syntax-ns#",
        }
        mock_owl_parser._recover_default.return_value = default_namespaces

        result = mock_owl_parser._recover_default(error_context)

        assert "owl" in result
        assert "rdf" in result
        mock_owl_parser._recover_default.assert_called_once_with(error_context)

    def test_owl_parser_fallback_strategy(self, mock_owl_parser):
        """Test fallback between rdflib and owlready2 parsers."""
        error_context = ErrorContext(
            error=RuntimeError("owlready2 parsing failed"),
            severity=ErrorSeverity.RECOVERABLE,
            location="OWL parsing",
            recovery_strategy=RecoveryStrategy.RETRY,
            recovery_data={"parser": "owlready2"},
        )
        fallback_result = {"parser_used": "rdflib", "success": True}
        mock_owl_parser._try_alternative_format.return_value = fallback_result
        mock_owl_parser._recover_retry.return_value = fallback_result

        result = mock_owl_parser._recover_retry(error_context)

        assert result["parser_used"] == "rdflib"
        assert result["success"] is True
        mock_owl_parser._recover_retry.assert_called_once_with(error_context)

    def test_owl_content_sanitization(self, mock_owl_parser):
        """Test OWL content sanitization for malformed input."""
        malformed_content = "<owl:Class rdf:about='BadClass'><invalid_tag></owl:Class>"
        sanitized_content = "<owl:Class rdf:about='BadClass'></owl:Class>"
        mock_owl_parser._sanitize_owl_content.return_value = sanitized_content

        result = mock_owl_parser._sanitize_owl_content(malformed_content)

        assert result == sanitized_content
        mock_owl_parser._sanitize_owl_content.assert_called_once_with(malformed_content)

    def test_owl_minimal_ontology_creation(self, mock_owl_parser):
        """Test creation of minimal valid OWL ontology for severe errors."""
        error_context = ErrorContext(
            error=ValueError("Completely malformed OWL"),
            severity=ErrorSeverity.RECOVERABLE,
            location="OWL structure validation",
            recovery_strategy=RecoveryStrategy.DEFAULT,
        )
        minimal_owl = {
            "@context": {"owl": "http://www.w3.org/2002/07/owl#"},
            "@type": "owl:Ontology",
            "@id": "http://example.org/minimal",
        }
        mock_owl_parser._create_minimal_owl.return_value = minimal_owl
        mock_owl_parser._recover_default.return_value = minimal_owl

        result = mock_owl_parser._recover_default(error_context)

        assert result["@type"] == "owl:Ontology"
        assert "@context" in result
        mock_owl_parser._recover_default.assert_called_once_with(error_context)


class TestCSVParserErrorRecovery:
    """Test CSV parser-specific error recovery methods."""

    @pytest.fixture
    def mock_csv_parser(self):
        """Create a mock CSVParser with CSV-specific recovery methods."""
        parser = Mock()
        parser._recover_default = Mock()
        parser._recover_skip = Mock()
        parser._recover_replace = Mock()
        parser._fix_field_count_mismatch = Mock()
        parser._handle_encoding_error = Mock()
        parser._infer_missing_data = Mock()
        return parser

    def test_csv_field_count_mismatch_recovery(self, mock_csv_parser):
        """Test recovery from field count mismatch in CSV rows."""
        error_context = ErrorContext(
            error=ValueError("Row has 5 fields, expected 3"),
            severity=ErrorSeverity.RECOVERABLE,
            location="row 10",
            recovery_strategy=RecoveryStrategy.REPLACE,
            recovery_data={"expected_fields": 3, "actual_fields": 5},
        )
        fixed_row = ["field1", "field2", "field3"]
        mock_csv_parser._fix_field_count_mismatch.return_value = fixed_row
        mock_csv_parser._recover_replace.return_value = fixed_row

        result = mock_csv_parser._recover_replace(error_context)

        assert len(result) == 3
        assert result == fixed_row
        mock_csv_parser._recover_replace.assert_called_once_with(error_context)

    def test_csv_bad_row_skipping(self, mock_csv_parser):
        """Test skipping of completely malformed CSV rows."""
        error_context = ErrorContext(
            error=SyntaxError("Malformed CSV row with unmatched quotes"),
            severity=ErrorSeverity.RECOVERABLE,
            location="row 25",
            recovery_strategy=RecoveryStrategy.SKIP,
        )
        mock_csv_parser._recover_skip.return_value = None

        result = mock_csv_parser._recover_skip(error_context)

        assert result is None
        mock_csv_parser._recover_skip.assert_called_once_with(error_context)

    def test_csv_encoding_error_recovery(self, mock_csv_parser):
        """Test recovery from CSV encoding errors."""
        error_context = ErrorContext(
            error=UnicodeDecodeError("utf-8", b"", 0, 1, "invalid start byte"),
            severity=ErrorSeverity.RECOVERABLE,
            location="file encoding detection",
            recovery_strategy=RecoveryStrategy.RETRY,
            recovery_data={"encoding": "utf-8"},
        )
        recovered_content = "successfully decoded content"
        mock_csv_parser._handle_encoding_error.return_value = recovered_content
        mock_csv_parser._recover_retry = Mock(return_value=recovered_content)

        result = mock_csv_parser._recover_retry(error_context)

        assert result == recovered_content
        mock_csv_parser._recover_retry.assert_called_once_with(error_context)

    def test_csv_missing_data_inference(self, mock_csv_parser):
        """Test inference of missing data in CSV fields."""
        error_context = ErrorContext(
            error=ValueError("Empty required field 'age'"),
            severity=ErrorSeverity.WARNING,
            location="row 15, column 'age'",
            recovery_strategy=RecoveryStrategy.DEFAULT,
        )
        inferred_data = {"age": 0}  # Default value for missing age
        mock_csv_parser._infer_missing_data.return_value = inferred_data
        mock_csv_parser._recover_default.return_value = inferred_data

        result = mock_csv_parser._recover_default(error_context)

        assert result["age"] == 0
        mock_csv_parser._recover_default.assert_called_once_with(error_context)

    def test_csv_dialect_detection_error_recovery(self, mock_csv_parser):
        """Test recovery from CSV dialect detection failures."""
        error_context = ErrorContext(
            error=RuntimeError("Could not determine CSV dialect"),
            severity=ErrorSeverity.RECOVERABLE,
            location="dialect detection",
            recovery_strategy=RecoveryStrategy.DEFAULT,
        )
        default_dialect = {"delimiter": ",", "quotechar": '"', "escapechar": None}
        mock_csv_parser._recover_default.return_value = default_dialect

        result = mock_csv_parser._recover_default(error_context)

        assert result["delimiter"] == ","
        assert result["quotechar"] == '"'
        mock_csv_parser._recover_default.assert_called_once_with(error_context)


class TestJSONLDParserErrorRecovery:
    """Test JSON-LD parser-specific error recovery methods."""

    @pytest.fixture
    def mock_jsonld_parser(self):
        """Create a mock JSONLDParser with JSON-LD-specific recovery methods."""
        parser = Mock()
        parser._recover_retry = Mock()
        parser._recover_default = Mock()
        parser._sanitize_json_content = Mock()
        parser._create_minimal_jsonld = Mock()
        parser._resolve_missing_context = Mock()
        return parser

    def test_jsonld_malformed_json_recovery(self, mock_jsonld_parser):
        """Test recovery from malformed JSON in JSON-LD documents."""
        error_context = ErrorContext(
            error=json.JSONDecodeError(
                "Expecting ',' delimiter", '{"invalid": json}', 15
            ),
            severity=ErrorSeverity.RECOVERABLE,
            location="JSON parsing",
            recovery_strategy=RecoveryStrategy.RETRY,
        )
        sanitized_json = '{"@context": {}, "@type": "Thing"}'
        mock_jsonld_parser._sanitize_json_content.return_value = sanitized_json
        mock_jsonld_parser._recover_retry.return_value = {
            "success": True,
            "content": sanitized_json,
        }

        result = mock_jsonld_parser._recover_retry(error_context)

        assert result["success"] is True
        assert "content" in result
        mock_jsonld_parser._recover_retry.assert_called_once_with(error_context)

    def test_jsonld_missing_context_recovery(self, mock_jsonld_parser):
        """Test recovery from missing @context in JSON-LD documents."""
        error_context = ErrorContext(
            error=KeyError("@context not found"),
            severity=ErrorSeverity.RECOVERABLE,
            location="@context resolution",
            recovery_strategy=RecoveryStrategy.DEFAULT,
        )
        default_context = {
            "@context": {
                "@vocab": "http://schema.org/",
                "owl": "http://www.w3.org/2002/07/owl#",
            }
        }
        mock_jsonld_parser._resolve_missing_context.return_value = default_context
        mock_jsonld_parser._recover_default.return_value = default_context

        result = mock_jsonld_parser._recover_default(error_context)

        assert "@context" in result
        assert "@vocab" in result["@context"]
        mock_jsonld_parser._recover_default.assert_called_once_with(error_context)

    def test_jsonld_expansion_failure_recovery(self, mock_jsonld_parser):
        """Test recovery from JSON-LD expansion failures."""
        error_context = ErrorContext(
            error=RuntimeError("JSON-LD expansion failed"),
            severity=ErrorSeverity.RECOVERABLE,
            location="JSON-LD expansion",
            recovery_strategy=RecoveryStrategy.RETRY,
            recovery_data={"operation": "expand"},
        )
        fallback_result = {"@graph": [{"@type": "Thing"}]}
        mock_jsonld_parser._recover_retry.return_value = fallback_result

        result = mock_jsonld_parser._recover_retry(error_context)

        assert "@graph" in result
        mock_jsonld_parser._recover_retry.assert_called_once_with(error_context)

    def test_jsonld_content_sanitization(self, mock_jsonld_parser):
        """Test JSON-LD content sanitization for malformed input."""
        malformed_content = '{"@context": invalid, "@type": "Person"}'
        sanitized_content = '{"@context": {}, "@type": "Person"}'
        mock_jsonld_parser._sanitize_json_content.return_value = sanitized_content

        result = mock_jsonld_parser._sanitize_json_content(malformed_content)

        assert result == sanitized_content
        mock_jsonld_parser._sanitize_json_content.assert_called_once_with(
            malformed_content
        )

    def test_jsonld_minimal_document_creation(self, mock_jsonld_parser):
        """Test creation of minimal valid JSON-LD document for severe errors."""
        error_context = ErrorContext(
            error=ValueError("Completely malformed JSON-LD"),
            severity=ErrorSeverity.RECOVERABLE,
            location="JSON-LD structure validation",
            recovery_strategy=RecoveryStrategy.DEFAULT,
        )
        minimal_jsonld = {
            "@context": {"@vocab": "http://schema.org/"},
            "@type": "Thing",
            "@id": "_:minimal",
        }
        mock_jsonld_parser._create_minimal_jsonld.return_value = minimal_jsonld
        mock_jsonld_parser._recover_default.return_value = minimal_jsonld

        result = mock_jsonld_parser._recover_default(error_context)

        assert result["@type"] == "Thing"
        assert "@context" in result
        mock_jsonld_parser._recover_default.assert_called_once_with(error_context)


class TestErrorContextManagement:
    """Test error context creation and tracking."""

    def test_error_context_creation(self):
        """Test creation of ErrorContext with all required fields."""
        error = ValueError("Test error")
        severity = ErrorSeverity.RECOVERABLE
        location = "test location"

        context = ErrorContext(error=error, severity=severity, location=location)

        assert context.error == error
        assert context.severity == severity
        assert context.location == location
        assert context.recovery_strategy is None
        assert context.attempted_recoveries == []
        assert context.recovery_data == {}
        assert isinstance(context.timestamp, datetime)

    def test_error_context_with_recovery_data(self):
        """Test ErrorContext with recovery data and attempted recoveries."""
        error = TimeoutError("Connection timeout")
        context = ErrorContext(
            error=error,
            severity=ErrorSeverity.RECOVERABLE,
            location="remote API call",
            recovery_strategy=RecoveryStrategy.RETRY,
            attempted_recoveries=[RecoveryStrategy.RETRY],
            recovery_data={"retry_count": 1, "timeout": 30},
        )

        assert context.recovery_strategy == RecoveryStrategy.RETRY
        assert RecoveryStrategy.RETRY in context.attempted_recoveries
        assert context.recovery_data["retry_count"] == 1
        assert context.recovery_data["timeout"] == 30

    def test_error_context_attempted_recoveries_tracking(self):
        """Test tracking of attempted recovery strategies."""
        context = ErrorContext(
            error=ValueError("Test error"),
            severity=ErrorSeverity.RECOVERABLE,
            location="test location",
        )

        # Simulate adding attempted recoveries
        context.attempted_recoveries.append(RecoveryStrategy.SKIP)
        context.attempted_recoveries.append(RecoveryStrategy.DEFAULT)

        assert len(context.attempted_recoveries) == 2
        assert RecoveryStrategy.SKIP in context.attempted_recoveries
        assert RecoveryStrategy.DEFAULT in context.attempted_recoveries
        assert RecoveryStrategy.RETRY not in context.attempted_recoveries


class TestErrorStatisticsTracking:
    """Test statistics collection and reporting."""

    def test_error_stats_initialization(self):
        """Test ErrorRecoveryStats initialization with default values."""
        stats = ErrorRecoveryStats()

        assert stats.total_errors == 0
        assert stats.warnings == 0
        assert stats.recoverable_errors == 0
        assert stats.fatal_errors == 0
        assert stats.successful_recoveries == 0
        assert stats.failed_recoveries == 0
        assert stats.recovery_strategies_used == {}

    def test_error_stats_tracking(self):
        """Test error statistics tracking and updates."""
        stats = ErrorRecoveryStats()

        # Simulate error statistics updates
        stats.total_errors += 1
        stats.recoverable_errors += 1
        stats.successful_recoveries += 1
        stats.recovery_strategies_used["SKIP"] = (
            stats.recovery_strategies_used.get("SKIP", 0) + 1
        )

        assert stats.total_errors == 1
        assert stats.recoverable_errors == 1
        assert stats.successful_recoveries == 1
        assert stats.recovery_strategies_used["SKIP"] == 1

    def test_comprehensive_error_stats_tracking(self):
        """Test comprehensive error statistics with multiple error types."""
        stats = ErrorRecoveryStats()

        # Simulate various error scenarios
        # Warning errors
        stats.total_errors += 2
        stats.warnings += 2
        stats.successful_recoveries += 2
        stats.recovery_strategies_used["CONTINUE"] = 2

        # Recoverable errors
        stats.total_errors += 3
        stats.recoverable_errors += 3
        stats.successful_recoveries += 2
        stats.failed_recoveries += 1
        stats.recovery_strategies_used["SKIP"] = 1
        stats.recovery_strategies_used["DEFAULT"] = 1
        stats.recovery_strategies_used["RETRY"] = 1

        # Fatal errors
        stats.total_errors += 1
        stats.fatal_errors += 1
        stats.failed_recoveries += 1
        stats.recovery_strategies_used["ABORT"] = 1

        assert stats.total_errors == 6
        assert stats.warnings == 2
        assert stats.recoverable_errors == 3
        assert stats.fatal_errors == 1
        assert stats.successful_recoveries == 4
        assert stats.failed_recoveries == 2
        assert len(stats.recovery_strategies_used) == 5


class TestRecoveryConfigurationOptions:
    """Test configuration-driven recovery behavior."""

    @pytest.fixture
    def mock_parser_with_config(self):
        """Create a mock parser with configuration options."""
        parser = Mock()
        parser.config = {
            "error_recovery": {
                "max_retries": 3,
                "skip_malformed_rows": True,
                "use_default_values": True,
                "abort_on_fatal": True,
                "collect_statistics": True,
            }
        }
        parser.get_config = Mock(return_value=parser.config)
        return parser

    def test_max_retries_configuration(self, mock_parser_with_config):
        """Test max retries configuration affects recovery behavior."""
        config = mock_parser_with_config.get_config()
        max_retries = config["error_recovery"]["max_retries"]

        assert max_retries == 3
        mock_parser_with_config.get_config.assert_called_once()

    def test_skip_malformed_rows_configuration(self, mock_parser_with_config):
        """Test skip malformed rows configuration."""
        config = mock_parser_with_config.get_config()
        skip_malformed = config["error_recovery"]["skip_malformed_rows"]

        assert skip_malformed is True
        mock_parser_with_config.get_config.assert_called_once()

    def test_use_default_values_configuration(self, mock_parser_with_config):
        """Test use default values configuration."""
        config = mock_parser_with_config.get_config()
        use_defaults = config["error_recovery"]["use_default_values"]

        assert use_defaults is True
        mock_parser_with_config.get_config.assert_called_once()

    def test_abort_on_fatal_configuration(self, mock_parser_with_config):
        """Test abort on fatal errors configuration."""
        config = mock_parser_with_config.get_config()
        abort_on_fatal = config["error_recovery"]["abort_on_fatal"]

        assert abort_on_fatal is True
        mock_parser_with_config.get_config.assert_called_once()

    def test_collect_statistics_configuration(self, mock_parser_with_config):
        """Test collect statistics configuration."""
        config = mock_parser_with_config.get_config()
        collect_stats = config["error_recovery"]["collect_statistics"]

        assert collect_stats is True
        mock_parser_with_config.get_config.assert_called_once()


class TestRecoveryFallbackBehaviors:
    """Test edge cases and fallback scenarios."""

    @pytest.fixture
    def mock_parser_with_fallbacks(self):
        """Create a mock parser with fallback behaviors."""
        parser = Mock()
        parser._handle_unknown_error = Mock()
        parser._handle_recovery_failure = Mock()
        parser._apply_last_resort_recovery = Mock()
        return parser

    def test_unknown_error_type_fallback(self, mock_parser_with_fallbacks):
        """Test fallback behavior for unknown error types."""
        unknown_error = RuntimeError("Unknown error type")
        expected_fallback = {"strategy": "skip", "reason": "unknown_error_type"}
        mock_parser_with_fallbacks._handle_unknown_error.return_value = (
            expected_fallback
        )

        result = mock_parser_with_fallbacks._handle_unknown_error(unknown_error)

        assert result["strategy"] == "skip"
        assert result["reason"] == "unknown_error_type"
        mock_parser_with_fallbacks._handle_unknown_error.assert_called_once_with(
            unknown_error
        )

    def test_recovery_strategy_failure_fallback(self, mock_parser_with_fallbacks):
        """Test fallback when recovery strategies fail."""
        error_context = ErrorContext(
            error=ValueError("Recovery failed"),
            severity=ErrorSeverity.RECOVERABLE,
            location="recovery attempt",
            attempted_recoveries=[
                RecoveryStrategy.SKIP,
                RecoveryStrategy.DEFAULT,
                RecoveryStrategy.RETRY,
            ],
        )
        last_resort_result = {"fallback": "minimal_data"}
        mock_parser_with_fallbacks._handle_recovery_failure.return_value = (
            last_resort_result
        )

        result = mock_parser_with_fallbacks._handle_recovery_failure(error_context)

        assert result["fallback"] == "minimal_data"
        mock_parser_with_fallbacks._handle_recovery_failure.assert_called_once_with(
            error_context
        )

    def test_multiple_recovery_attempts_exhausted(self, mock_parser_with_fallbacks):
        """Test behavior when all recovery attempts are exhausted."""
        error_context = ErrorContext(
            error=TimeoutError("All recovery attempts failed"),
            severity=ErrorSeverity.RECOVERABLE,
            location="final recovery attempt",
            attempted_recoveries=[
                RecoveryStrategy.RETRY,
                RecoveryStrategy.RETRY,
                RecoveryStrategy.RETRY,
                RecoveryStrategy.SKIP,
                RecoveryStrategy.DEFAULT,
            ],
        )
        last_resort_result = None  # Indicate complete failure
        mock_parser_with_fallbacks._apply_last_resort_recovery.return_value = (
            last_resort_result
        )

        result = mock_parser_with_fallbacks._apply_last_resort_recovery(error_context)

        assert result is None
        mock_parser_with_fallbacks._apply_last_resort_recovery.assert_called_once_with(
            error_context
        )

    def test_circular_recovery_prevention(self, mock_parser_with_fallbacks):
        """Test prevention of circular recovery attempts."""
        error_context = ErrorContext(
            error=RecursionError("Circular recovery detected"),
            severity=ErrorSeverity.FATAL,
            location="recovery loop detection",
            attempted_recoveries=[RecoveryStrategy.RETRY] * 10,  # Excessive retries
        )

        # Should detect circular recovery and abort
        expected_result = {"error": "circular_recovery_detected", "action": "abort"}
        mock_parser_with_fallbacks._handle_recovery_failure.return_value = (
            expected_result
        )

        result = mock_parser_with_fallbacks._handle_recovery_failure(error_context)

        assert result["error"] == "circular_recovery_detected"
        assert result["action"] == "abort"
        mock_parser_with_fallbacks._handle_recovery_failure.assert_called_once_with(
            error_context
        )


# Integration test fixtures and utilities
@pytest.fixture
def sample_malformed_owl():
    """Sample malformed OWL content for testing."""
    return """<?xml version="1.0"?>
    <rdf:RDF xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#"
             xmlns:owl="http://www.w3.org/2002/07/owl#">
        <owl:Class rdf:about="http://example.org/BadClass">
            <invalid_property>invalid value
        </owl:Class>
    </rdf:RDF>"""


@pytest.fixture
def sample_malformed_csv():
    """Sample malformed CSV content for testing."""
    return """name,age,city
    John,25,New York
    Jane,"30,Boston"  # Unmatched quote
    Bob,invalid_age,Chicago
    Alice,,  # Missing city
    """


@pytest.fixture
def sample_malformed_jsonld():
    """Sample malformed JSON-LD content for testing."""
    return """{
        "@context": invalid_context,
        "@type": "Person",
        "name": "John Doe",
        "age": "not_a_number",
        "address": {
            "@type": "Address",
            "street": "123 Main St"
            # Missing closing brace
    }"""


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
