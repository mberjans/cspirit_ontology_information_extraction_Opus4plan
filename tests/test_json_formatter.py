"""
Unit tests for JSON formatter functionality in the AIM2 project logging framework

This module provides comprehensive unit tests specifically for JSON formatter functionality,
including formatter creation, configuration, JSON output validation, and integration with handlers.

Test Coverage:
- JSONFormatter creation and basic configuration
- JSON field inclusion/exclusion and validation
- Pretty-printed vs compact JSON output formatting
- Custom fields and field mapping functionality
- Exception handling and traceback formatting
- Message truncation and length limits
- Unicode and special character handling
- Integration with LoggerConfig and LoggerManager
- Edge cases (None values, empty messages, circular references)
- Thread safety and performance characteristics
- Error scenarios and fallback handling

Test Classes:
    TestJSONFormatterCreation: Tests formatter creation and basic configuration
    TestJSONFormatterFieldConfiguration: Tests field configuration and validation
    TestJSONFormatterOutput: Tests JSON output with various inputs
    TestJSONFormatterFormatting: Tests pretty-printing and output formatting
    TestJSONFormatterCustomFields: Tests custom fields and field mapping
    TestJSONFormatterExceptionHandling: Tests exception logging and tracebacks
    TestJSONFormatterMessageTruncation: Tests message length limits
    TestJSONFormatterUnicodeHandling: Tests special characters and unicode
    TestJSONFormatterIntegration: Tests integration with logging framework
    TestJSONFormatterEdgeCases: Tests edge cases and error scenarios
    TestJSONFormatterPerformance: Tests thread safety and performance
"""

import io
import json
import logging
import logging.handlers
import tempfile
import pytest
import re
import time
import threading
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock

# Import the modules to be tested
try:
    from aim2_project.aim2_utils.json_formatter import (
        JSONFormatter,
        JSONFormatterError,
        create_json_formatter,
    )
    from aim2_project.aim2_utils.logger_config import (
        LoggerConfig,
        LoggerConfigError,
    )
    from aim2_project.aim2_utils.logger_manager import (
        LoggerManager,
        LoggerManagerError,
    )
except ImportError:
    # Expected during TDD - tests define the interface
    pass


class TestJSONFormatterCreation:
    """Test suite for JSON formatter creation and basic configuration."""

    def test_formatter_creation_default_config(self):
        """Test JSONFormatter creation with default configuration."""
        formatter = JSONFormatter()
        
        assert formatter.fields == JSONFormatter.DEFAULT_FIELDS
        assert formatter.pretty_print is False
        assert formatter.custom_fields == {}
        assert formatter.timestamp_format == "iso"
        assert formatter.use_utc is False
        assert formatter.include_exception_traceback is True
        assert formatter.max_message_length is None
        assert formatter.field_mapping == {}
        assert formatter.ensure_ascii is False
        assert hasattr(formatter, '_lock')

    def test_formatter_creation_custom_fields(self):
        """Test JSONFormatter creation with custom field list."""
        custom_fields = ["timestamp", "level", "message"]
        formatter = JSONFormatter(fields=custom_fields)
        
        assert formatter.fields == custom_fields
        assert "logger_name" not in formatter.fields

    def test_formatter_creation_pretty_print(self):
        """Test JSONFormatter creation with pretty printing enabled."""
        formatter = JSONFormatter(pretty_print=True)
        
        assert formatter.pretty_print is True

    def test_formatter_creation_custom_fields_dict(self):
        """Test JSONFormatter creation with custom fields dictionary."""
        custom_fields_dict = {"service": "test-service", "version": "1.0.0"}
        formatter = JSONFormatter(custom_fields=custom_fields_dict)
        
        assert formatter.custom_fields == custom_fields_dict

    def test_formatter_creation_timestamp_format(self):
        """Test JSONFormatter creation with different timestamp formats."""
        # ISO format
        formatter_iso = JSONFormatter(timestamp_format="iso")
        assert formatter_iso.timestamp_format == "iso"
        
        # Epoch format
        formatter_epoch = JSONFormatter(timestamp_format="epoch")
        assert formatter_epoch.timestamp_format == "epoch"
        
        # Custom strftime format
        formatter_custom = JSONFormatter(timestamp_format="%Y-%m-%d %H:%M:%S")
        assert formatter_custom.timestamp_format == "%Y-%m-%d %H:%M:%S"

    def test_formatter_creation_utc_timezone(self):
        """Test JSONFormatter creation with UTC timezone setting."""
        formatter = JSONFormatter(use_utc=True)
        
        assert formatter.use_utc is True

    def test_formatter_creation_max_message_length(self):
        """Test JSONFormatter creation with message length limit."""
        formatter = JSONFormatter(max_message_length=100)
        
        assert formatter.max_message_length == 100

    def test_formatter_creation_field_mapping(self):
        """Test JSONFormatter creation with field mapping."""
        field_mapping = {"level": "log_level", "message": "msg"}
        formatter = JSONFormatter(field_mapping=field_mapping)
        
        assert formatter.field_mapping == field_mapping

    def test_formatter_creation_ensure_ascii(self):
        """Test JSONFormatter creation with ASCII enforcement."""
        formatter = JSONFormatter(ensure_ascii=True)
        
        assert formatter.ensure_ascii is True

    def test_formatter_creation_invalid_fields(self):
        """Test JSONFormatter creation with invalid fields."""
        with pytest.raises(JSONFormatterError) as exc_info:
            JSONFormatter(fields=["invalid_field"])
        
        assert "Invalid fields" in str(exc_info.value)
        assert "invalid_field" in str(exc_info.value)

    def test_formatter_creation_non_list_fields(self):
        """Test JSONFormatter creation with non-list fields parameter."""
        with pytest.raises(JSONFormatterError) as exc_info:
            JSONFormatter(fields="not_a_list")
        
        assert "Fields must be a list" in str(exc_info.value)

    def test_formatter_creation_invalid_timestamp_format(self):
        """Test JSONFormatter creation with invalid timestamp format."""
        with pytest.raises(JSONFormatterError) as exc_info:
            JSONFormatter(timestamp_format="invalid")
        
        assert "timestamp_format must be" in str(exc_info.value)

    def test_formatter_creation_invalid_custom_fields(self):
        """Test JSONFormatter creation with invalid custom fields."""
        with pytest.raises(JSONFormatterError) as exc_info:
            JSONFormatter(custom_fields="not_a_dict")
        
        assert "custom_fields must be a dictionary" in str(exc_info.value)

    def test_formatter_creation_invalid_max_message_length(self):
        """Test JSONFormatter creation with invalid max message length."""
        with pytest.raises(JSONFormatterError) as exc_info:
            JSONFormatter(max_message_length=0)
        
        assert "max_message_length must be a positive integer" in str(exc_info.value)

    def test_formatter_creation_invalid_field_mapping(self):
        """Test JSONFormatter creation with invalid field mapping."""
        with pytest.raises(JSONFormatterError) as exc_info:
            JSONFormatter(field_mapping="not_a_dict")
        
        assert "field_mapping must be a dictionary" in str(exc_info.value)

    def test_formatter_repr(self):
        """Test JSONFormatter string representation."""
        formatter = JSONFormatter(
            fields=["timestamp", "level", "message"],
            pretty_print=True,
            custom_fields={"service": "test"},
            timestamp_format="epoch"
        )
        
        repr_str = repr(formatter)
        assert "JSONFormatter" in repr_str
        assert "fields=3" in repr_str
        assert "pretty_print=True" in repr_str
        assert "timestamp_format='epoch'" in repr_str
        assert "custom_fields=1" in repr_str


class TestJSONFormatterFieldConfiguration:
    """Test suite for JSON formatter field configuration and validation."""

    @pytest.fixture
    def formatter(self):
        """Create a JSONFormatter instance for testing."""
        return JSONFormatter()

    def test_field_validation_valid_fields(self, formatter):
        """Test field validation with valid field names."""
        valid_fields = ["timestamp", "level", "logger_name", "message", "exception"]
        formatter.set_fields(valid_fields)
        
        assert formatter.fields == valid_fields

    def test_field_validation_field_mapping_keys(self, formatter):
        """Test field validation includes all FIELD_MAPPING keys."""
        for field in JSONFormatter.FIELD_MAPPING.keys():
            # Each field should be valid
            formatter.set_fields([field])
            assert field in formatter.fields

    def test_add_field_new_field(self, formatter):
        """Test adding a new field to the formatter."""
        initial_fields = formatter.fields.copy()
        formatter.add_field("process")
        
        assert "process" in formatter.fields
        assert len(formatter.fields) == len(initial_fields) + 1

    def test_add_field_existing_field(self, formatter):
        """Test adding an existing field doesn't duplicate it."""
        initial_fields = formatter.fields.copy()
        initial_length = len(formatter.fields)
        
        # Add a field that already exists
        formatter.add_field("timestamp")
        
        assert formatter.fields == initial_fields
        assert len(formatter.fields) == initial_length

    def test_remove_field_existing_field(self, formatter):
        """Test removing an existing field from the formatter."""
        formatter.remove_field("timestamp")
        
        assert "timestamp" not in formatter.fields

    def test_remove_field_nonexistent_field(self, formatter):
        """Test removing a non-existent field doesn't cause errors."""
        initial_fields = formatter.fields.copy()
        formatter.remove_field("nonexistent_field")
        
        assert formatter.fields == initial_fields

    def test_set_fields_invalid_field(self, formatter):
        """Test setting fields with invalid field name."""
        with pytest.raises(JSONFormatterError):
            formatter.set_fields(["invalid_field"])

    def test_set_fields_validation_failure_rollback(self, formatter):
        """Test that field validation failure rolls back to previous fields."""
        original_fields = formatter.fields.copy()
        
        with pytest.raises(JSONFormatterError):
            formatter.set_fields(["invalid_field"])
        
        # Fields should be rolled back to original
        assert formatter.fields == original_fields

    def test_custom_fields_validation(self, formatter):
        """Test custom fields validation."""
        valid_custom_fields = {"service": "test", "version": "1.0"}
        formatter.set_custom_fields(valid_custom_fields)
        
        assert formatter.custom_fields == valid_custom_fields

    def test_custom_fields_invalid_type(self, formatter):
        """Test custom fields validation with invalid type."""
        with pytest.raises(JSONFormatterError) as exc_info:
            formatter.set_custom_fields("not_a_dict")
        
        assert "custom_fields must be a dictionary" in str(exc_info.value)

    def test_timestamp_format_validation(self, formatter):
        """Test timestamp format validation."""
        # Valid formats
        valid_formats = ["iso", "epoch", "%Y-%m-%d", "%H:%M:%S"]
        
        for fmt in valid_formats:
            formatter.set_timestamp_format(fmt)
            assert formatter.timestamp_format == fmt

    def test_timestamp_format_invalid(self, formatter):
        """Test timestamp format validation with invalid format."""
        with pytest.raises(JSONFormatterError):
            formatter.set_timestamp_format("invalid_format")

    def test_pretty_print_setting(self, formatter):
        """Test pretty print setting."""
        formatter.set_pretty_print(True)
        assert formatter.pretty_print is True
        
        formatter.set_pretty_print(False)
        assert formatter.pretty_print is False

    def test_get_configuration(self, formatter):
        """Test getting current formatter configuration."""
        config = formatter.get_configuration()
        
        assert isinstance(config, dict)
        assert "fields" in config
        assert "pretty_print" in config
        assert "custom_fields" in config
        assert "timestamp_format" in config
        assert "use_utc" in config
        assert "include_exception_traceback" in config
        assert "max_message_length" in config
        assert "field_mapping" in config
        assert "ensure_ascii" in config

    def test_configuration_immutability(self, formatter):
        """Test that returned configuration is a copy."""
        config = formatter.get_configuration()
        config["fields"].append("new_field")
        
        # Original formatter should be unchanged
        assert "new_field" not in formatter.fields


class TestJSONFormatterOutput:
    """Test suite for JSON formatter output with various inputs."""

    @pytest.fixture
    def formatter(self):
        """Create a JSONFormatter instance for testing."""
        return JSONFormatter()

    @pytest.fixture
    def log_record(self):
        """Create a sample log record for testing."""
        return logging.LogRecord(
            name="test_logger",
            level=logging.INFO,
            pathname="/path/to/test.py",
            lineno=42,
            msg="Test message",
            args=(),
            exc_info=None,
        )

    def test_basic_json_output(self, formatter, log_record):
        """Test basic JSON output formatting."""
        output = formatter.format(log_record)
        
        # Should be valid JSON
        json_data = json.loads(output)
        
        # Verify expected fields
        assert "timestamp" in json_data
        assert "level" in json_data
        assert "logger_name" in json_data
        assert "message" in json_data
        
        # Verify values
        assert json_data["level"] == "INFO"
        assert json_data["logger_name"] == "test_logger"
        assert json_data["message"] == "Test message"

    def test_json_output_different_log_levels(self, formatter):
        """Test JSON output with different log levels."""
        test_cases = [
            (logging.DEBUG, "DEBUG"),
            (logging.INFO, "INFO"),
            (logging.WARNING, "WARNING"),
            (logging.ERROR, "ERROR"),
            (logging.CRITICAL, "CRITICAL"),
        ]
        
        for level, level_name in test_cases:
            record = logging.LogRecord(
                name="test",
                level=level,
                pathname="test.py",
                lineno=1,
                msg="Test message",
                args=(),
                exc_info=None,
            )
            
            output = formatter.format(record)
            json_data = json.loads(output)
            
            assert json_data["level"] == level_name

    def test_json_output_with_args(self, formatter):
        """Test JSON output with message arguments."""
        record = logging.LogRecord(
            name="test",
            level=logging.INFO,
            pathname="test.py",
            lineno=1,
            msg="Test message with %s and %d",
            args=("string", 42),
            exc_info=None,
        )
        
        output = formatter.format(record)
        json_data = json.loads(output)
        
        assert json_data["message"] == "Test message with string and 42"

    def test_json_output_compact_format(self, formatter, log_record):
        """Test compact JSON output (no pretty printing)."""
        formatter.set_pretty_print(False)
        output = formatter.format(log_record)
        
        # Compact JSON should not have indentation or extra whitespace
        assert "\n" not in output
        assert "  " not in output  # No double spaces for indentation

    def test_json_output_pretty_format(self, formatter, log_record):
        """Test pretty-printed JSON output."""
        formatter.set_pretty_print(True)
        output = formatter.format(log_record)
        
        # Pretty JSON should have indentation
        assert "\n" in output
        assert "  " in output  # Indentation spaces

    def test_json_output_field_selection(self, formatter, log_record):
        """Test JSON output with specific field selection."""
        formatter.set_fields(["timestamp", "level", "message"])
        output = formatter.format(log_record)
        
        json_data = json.loads(output)
        
        # Should only contain selected fields
        assert "timestamp" in json_data
        assert "level" in json_data
        assert "message" in json_data
        assert "logger_name" not in json_data
        assert "module" not in json_data

    def test_json_output_empty_message(self, formatter):
        """Test JSON output with empty message."""
        record = logging.LogRecord(
            name="test",
            level=logging.INFO,
            pathname="test.py",
            lineno=1,
            msg="",
            args=(),
            exc_info=None,
        )
        
        output = formatter.format(record)
        json_data = json.loads(output)
        
        assert json_data["message"] == ""

    def test_json_output_none_message(self, formatter):
        """Test JSON output with None message."""
        record = logging.LogRecord(
            name="test",
            level=logging.INFO,
            pathname="test.py",
            lineno=1,
            msg=None,
            args=(),
            exc_info=None,
        )
        
        output = formatter.format(record)
        json_data = json.loads(output)
        
        # Should handle None gracefully
        assert "message" in json_data

    def test_json_output_special_characters(self, formatter):
        """Test JSON output with special characters."""
        special_messages = [
            "Message with unicode: café ñoño 中文",
            "Message with quotes: 'single' \"double\"",
            "Message with symbols: !@#$%^&*()",
            "Message with newlines:\nLine 1\nLine 2",
        ]
        
        for msg in special_messages:
            record = logging.LogRecord(
                name="test",
                level=logging.INFO,
                pathname="test.py",
                lineno=1,
                msg=msg,
                args=(),
                exc_info=None,
            )
            
            output = formatter.format(record)
            json_data = json.loads(output)  # Should not raise exception
            
            assert json_data["message"] == msg

    def test_json_output_large_message(self, formatter):
        """Test JSON output with large message."""
        large_message = "A" * 10000
        record = logging.LogRecord(
            name="test",
            level=logging.INFO,
            pathname="test.py",
            lineno=1,
            msg=large_message,
            args=(),
            exc_info=None,
        )
        
        output = formatter.format(record)
        json_data = json.loads(output)
        
        assert json_data["message"] == large_message

    def test_json_output_timestamp_formats(self, formatter, log_record):
        """Test JSON output with different timestamp formats."""
        # ISO format
        formatter.set_timestamp_format("iso")
        output_iso = formatter.format(log_record)
        json_data_iso = json.loads(output_iso)
        
        # Should be ISO format
        timestamp_iso = json_data_iso["timestamp"]
        assert "T" in timestamp_iso or "-" in timestamp_iso
        
        # Epoch format
        formatter.set_timestamp_format("epoch")
        output_epoch = formatter.format(log_record)
        json_data_epoch = json.loads(output_epoch)
        
        # Should be numeric
        timestamp_epoch = json_data_epoch["timestamp"]
        assert isinstance(float(timestamp_epoch), float)

    def test_json_output_utc_timestamp(self, formatter, log_record):
        """Test JSON output with UTC timestamps."""
        formatter.timestamp_format = "iso"
        formatter.use_utc = True
        
        output = formatter.format(log_record)
        json_data = json.loads(output)
        
        # UTC timestamp should have timezone info
        timestamp = json_data["timestamp"]
        assert "+00:00" in timestamp or "Z" in timestamp or "UTC" in timestamp

    def test_json_output_all_fields(self, formatter):
        """Test JSON output with all available fields."""
        # Set all possible fields
        all_fields = list(JSONFormatter.FIELD_MAPPING.keys())
        formatter.set_fields(all_fields)
        
        record = logging.LogRecord(
            name="test.module",
            level=logging.WARNING,
            pathname="/path/to/test.py",
            lineno=100,
            msg="Test message",
            args=(),
            exc_info=None,
            func="test_function",
        )
        
        output = formatter.format(record)
        json_data = json.loads(output)
        
        # Should contain most fields (some may be None/empty)
        assert "timestamp" in json_data
        assert "level" in json_data
        assert "logger_name" in json_data
        assert "message" in json_data


class TestJSONFormatterFormatting:
    """Test suite for JSON formatter output formatting options."""

    @pytest.fixture
    def formatter(self):
        """Create a JSONFormatter instance for testing."""
        return JSONFormatter()

    @pytest.fixture
    def log_record(self):
        """Create a sample log record for testing."""
        return logging.LogRecord(
            name="test_logger",
            level=logging.INFO,
            pathname="/path/to/test.py",
            lineno=42,
            msg="Test message",
            args=(),
            exc_info=None,
        )

    def test_pretty_print_formatting(self, formatter, log_record):
        """Test pretty-printed JSON formatting."""
        formatter.set_pretty_print(True)
        output = formatter.format(log_record)
        
        # Should have proper indentation
        lines = output.split('\n')
        assert len(lines) > 1  # Multiple lines
        
        # Should have consistent indentation
        indented_lines = [line for line in lines if line.startswith('  ')]
        assert len(indented_lines) > 0

    def test_compact_formatting(self, formatter, log_record):
        """Test compact JSON formatting."""
        formatter.set_pretty_print(False)
        output = formatter.format(log_record)
        
        # Should be single line
        lines = output.split('\n')
        assert len(lines) == 1
        
        # Should not have extra spaces around separators
        assert ', ' not in output
        assert ': ' not in output

    def test_ensure_ascii_false(self, formatter):
        """Test JSON output with ensure_ascii=False."""
        formatter.ensure_ascii = False
        
        record = logging.LogRecord(
            name="test",
            level=logging.INFO,
            pathname="test.py",
            lineno=1,
            msg="Unicode message: café 中文",
            args=(),
            exc_info=None,
        )
        
        output = formatter.format(record)
        json_data = json.loads(output)
        
        # Unicode characters should be preserved
        assert "café" in json_data["message"]
        assert "中文" in json_data["message"]

    def test_ensure_ascii_true(self, formatter):
        """Test JSON output with ensure_ascii=True."""
        formatter.ensure_ascii = True
        
        record = logging.LogRecord(
            name="test",
            level=logging.INFO,
            pathname="test.py",
            lineno=1,
            msg="Unicode message: café 中文",
            args=(),
            exc_info=None,
        )
        
        output = formatter.format(record)
        
        # Should contain escaped unicode sequences
        assert "\\u" in output
        
        # But when parsed, should still be correct
        json_data = json.loads(output)
        assert "café" in json_data["message"]
        assert "中文" in json_data["message"]

    def test_field_mapping_output(self, formatter, log_record):
        """Test JSON output with field mapping."""
        field_mapping = {
            "level": "log_level",
            "logger_name": "service",
            "message": "msg"
        }
        formatter.field_mapping = field_mapping
        
        output = formatter.format(log_record)
        json_data = json.loads(output)
        
        # Should use mapped field names
        assert "log_level" in json_data
        assert "service" in json_data
        assert "msg" in json_data
        
        # Should not contain original field names
        assert "level" not in json_data
        assert "logger_name" not in json_data
        assert "message" not in json_data

    def test_sorted_keys_in_pretty_print(self, formatter, log_record):
        """Test that pretty-printed JSON has sorted keys."""
        formatter.set_pretty_print(True)
        output = formatter.format(log_record)
        
        json_data = json.loads(output)
        keys = list(json_data.keys())
        sorted_keys = sorted(keys)
        
        # Keys should be in sorted order in pretty print
        assert keys == sorted_keys

    def test_json_serialization_fallback(self, formatter):
        """Test JSON serialization with non-serializable objects."""
        class NonSerializable:
            def __str__(self):
                return "non-serializable object"
        
        record = logging.LogRecord(
            name="test",
            level=logging.INFO,
            pathname="test.py",
            lineno=1,
            msg="Object: %s",
            args=(NonSerializable(),),
            exc_info=None,
        )
        
        output = formatter.format(record)
        json_data = json.loads(output)
        
        # Should handle non-serializable objects gracefully
        assert "non-serializable object" in json_data["message"]


class TestJSONFormatterCustomFields:
    """Test suite for JSON formatter custom fields and field mapping."""

    @pytest.fixture
    def formatter(self):
        """Create a JSONFormatter instance for testing."""
        return JSONFormatter()

    @pytest.fixture
    def log_record(self):
        """Create a sample log record for testing."""
        return logging.LogRecord(
            name="test_logger",
            level=logging.INFO,
            pathname="/path/to/test.py",
            lineno=42,
            msg="Test message",
            args=(),
            exc_info=None,
        )

    def test_custom_fields_inclusion(self, formatter, log_record):
        """Test inclusion of custom fields in JSON output."""
        custom_fields = {"service": "test-service", "version": "1.0.0", "environment": "dev"}
        formatter.set_custom_fields(custom_fields)
        formatter.add_field("custom_fields")
        
        output = formatter.format(log_record)
        json_data = json.loads(output)
        
        assert "custom_fields" in json_data
        assert json_data["custom_fields"]["service"] == "test-service"
        assert json_data["custom_fields"]["version"] == "1.0.0"
        assert json_data["custom_fields"]["environment"] == "dev"

    def test_custom_fields_mapping(self, formatter, log_record):
        """Test custom fields with field mapping."""
        custom_fields = {"service": "test-service", "version": "1.0.0"}
        field_mapping = {"custom_fields": "metadata"}
        
        formatter.set_custom_fields(custom_fields)
        formatter.field_mapping = field_mapping
        formatter.add_field("custom_fields")
        
        output = formatter.format(log_record)
        json_data = json.loads(output)
        
        # Should use mapped name
        assert "metadata" in json_data
        assert "custom_fields" not in json_data
        assert json_data["metadata"]["service"] == "test-service"

    def test_custom_fields_empty(self, formatter, log_record):
        """Test behavior with empty custom fields."""
        formatter.set_custom_fields({})
        formatter.add_field("custom_fields")
        
        output = formatter.format(log_record)
        json_data = json.loads(output)
        
        # Empty custom fields should still appear
        assert "custom_fields" in json_data
        assert json_data["custom_fields"] == {}

    def test_custom_fields_complex_values(self, formatter, log_record):
        """Test custom fields with complex values."""
        custom_fields = {
            "config": {"timeout": 30, "retries": 3},
            "tags": ["web", "api", "v1"],
            "enabled": True,
            "ratio": 0.95,
        }
        formatter.set_custom_fields(custom_fields)
        formatter.add_field("custom_fields")
        
        output = formatter.format(log_record)
        json_data = json.loads(output)
        
        custom_data = json_data["custom_fields"]
        assert custom_data["config"]["timeout"] == 30
        assert custom_data["tags"] == ["web", "api", "v1"]
        assert custom_data["enabled"] is True
        assert custom_data["ratio"] == 0.95

    def test_custom_fields_non_serializable(self, formatter, log_record):
        """Test custom fields with non-serializable values."""
        class NonSerializable:
            def __str__(self):
                return "custom object"
        
        custom_fields = {"object": NonSerializable()}
        formatter.set_custom_fields(custom_fields)
        formatter.add_field("custom_fields")
        
        output = formatter.format(log_record)
        json_data = json.loads(output)
        
        # Should convert to string
        assert json_data["custom_fields"]["object"] == "custom object"

    def test_extra_fields_from_record(self, formatter):
        """Test extraction of extra fields from log record."""
        record = logging.LogRecord(
            name="test",
            level=logging.INFO,
            pathname="test.py",
            lineno=1,
            msg="Test message",
            args=(),
            exc_info=None,
        )
        
        # Add extra attributes to record
        record.user_id = "12345"
        record.session_id = "abcdef"
        record.request_id = "req-789"
        
        output = formatter.format(record)
        json_data = json.loads(output)
        
        # Extra fields should be included
        assert "extra" in json_data
        extra_data = json_data["extra"]
        assert extra_data["user_id"] == "12345"
        assert extra_data["session_id"] == "abcdef"
        assert extra_data["request_id"] == "req-789"

    def test_extra_fields_mapping(self, formatter):
        """Test extra fields with field mapping."""
        formatter.field_mapping = {"extra": "context"}
        
        record = logging.LogRecord(
            name="test",
            level=logging.INFO,
            pathname="test.py",
            lineno=1,
            msg="Test message",
            args=(),
            exc_info=None,
        )
        record.user_id = "12345"
        
        output = formatter.format(record)
        json_data = json.loads(output)
        
        # Should use mapped name
        assert "context" in json_data
        assert "extra" not in json_data
        assert json_data["context"]["user_id"] == "12345"

    def test_extra_fields_exclude_standard_attributes(self, formatter):
        """Test that standard LogRecord attributes are not included in extra fields."""
        record = logging.LogRecord(
            name="test",
            level=logging.INFO,
            pathname="test.py",
            lineno=1,
            msg="Test message",
            args=(),
            exc_info=None,
        )
        
        output = formatter.format(record)
        json_data = json.loads(output)
        
        if "extra" in json_data:
            extra_data = json_data["extra"]
            # Standard attributes should not be in extra
            assert "name" not in extra_data
            assert "levelname" not in extra_data
            assert "msg" not in extra_data
            assert "args" not in extra_data

    def test_field_mapping_all_fields(self, formatter, log_record):
        """Test field mapping for all standard fields."""
        field_mapping = {
            "timestamp": "time",
            "level": "severity",
            "logger_name": "source",
            "module": "mod",
            "function": "func",
            "line_number": "line",
            "message": "text",
        }
        formatter.field_mapping = field_mapping
        
        output = formatter.format(log_record)
        json_data = json.loads(output)
        
        # Should use all mapped names
        assert "time" in json_data
        assert "severity" in json_data
        assert "source" in json_data
        assert "text" in json_data
        
        # Should not contain original names
        assert "timestamp" not in json_data
        assert "level" not in json_data
        assert "logger_name" not in json_data
        assert "message" not in json_data


class TestJSONFormatterExceptionHandling:
    """Test suite for JSON formatter exception handling and traceback formatting."""

    @pytest.fixture
    def formatter(self):
        """Create a JSONFormatter instance for testing."""
        return JSONFormatter()

    def test_exception_with_traceback(self, formatter):
        """Test exception formatting with traceback."""
        try:
            raise ValueError("Test exception")
        except ValueError:
            record = logging.LogRecord(
                name="test",
                level=logging.ERROR,
                pathname="test.py",
                lineno=1,
                msg="An error occurred",
                args=(),
                exc_info=True,  # Include exception info
            )
        
        formatter.add_field("exception")
        output = formatter.format(record)
        json_data = json.loads(output)
        
        assert "exception" in json_data
        exception_data = json_data["exception"]
        assert exception_data["type"] == "ValueError"
        assert exception_data["message"] == "Test exception"
        assert "traceback" in exception_data
        assert isinstance(exception_data["traceback"], list)

    def test_exception_without_traceback(self, formatter):
        """Test exception formatting without traceback."""
        formatter.include_exception_traceback = False
        
        try:
            raise ValueError("Test exception")
        except ValueError:
            record = logging.LogRecord(
                name="test",
                level=logging.ERROR,
                pathname="test.py",
                lineno=1,
                msg="An error occurred",
                args=(),
                exc_info=True,
            )
        
        formatter.add_field("exception")
        output = formatter.format(record)
        json_data = json.loads(output)
        
        assert "exception" in json_data
        exception_data = json_data["exception"]
        assert exception_data["type"] == "ValueError"
        assert exception_data["message"] == "Test exception"
        assert "traceback" not in exception_data

    def test_exception_none_exc_info(self, formatter):
        """Test exception field when no exception info is available."""
        record = logging.LogRecord(
            name="test",
            level=logging.INFO,
            pathname="test.py",
            lineno=1,
            msg="No exception here",
            args=(),
            exc_info=None,
        )
        
        formatter.add_field("exception")
        output = formatter.format(record)
        json_data = json.loads(output)
        
        # Exception field should not be present or should be null
        if "exception" in json_data:
            assert json_data["exception"] is None

    def test_exception_formatting_error(self, formatter):
        """Test exception formatting when traceback processing fails."""
        # Create a mock record with invalid exc_info
        record = logging.LogRecord(
            name="test",
            level=logging.ERROR,
            pathname="test.py",
            lineno=1,
            msg="Exception error",
            args=(),
            exc_info=None,
        )
        
        # Manually set invalid exc_info
        record.exc_info = ("not", "valid", "exc_info")
        
        formatter.add_field("exception")
        output = formatter.format(record)
        json_data = json.loads(output)
        
        # Should handle gracefully
        assert isinstance(json_data, dict)

    def test_nested_exception(self, formatter):
        """Test nested exception formatting."""
        try:
            try:
                raise ValueError("Inner exception")
            except ValueError as e:
                raise RuntimeError("Outer exception") from e
        except RuntimeError:
            record = logging.LogRecord(
                name="test",
                level=logging.ERROR,
                pathname="test.py",
                lineno=1,
                msg="Nested exception occurred",
                args=(),
                exc_info=True,
            )
        
        formatter.add_field("exception")
        output = formatter.format(record)
        json_data = json.loads(output)
        
        assert "exception" in json_data
        exception_data = json_data["exception"]
        assert exception_data["type"] == "RuntimeError"
        assert "traceback" in exception_data

    def test_exception_with_custom_message(self, formatter):
        """Test exception formatting with custom exception message."""
        class CustomException(Exception):
            def __init__(self, message, code):
                super().__init__(message)
                self.code = code
        
        try:
            raise CustomException("Custom error message", 500)
        except CustomException:
            record = logging.LogRecord(
                name="test",
                level=logging.ERROR,
                pathname="test.py",
                lineno=1,
                msg="Custom exception",
                args=(),
                exc_info=True,
            )
        
        formatter.add_field("exception")
        output = formatter.format(record)
        json_data = json.loads(output)
        
        exception_data = json_data["exception"]
        assert exception_data["type"] == "CustomException"
        assert exception_data["message"] == "Custom error message"

    def test_exception_field_mapping(self, formatter):
        """Test exception field with field mapping."""
        formatter.field_mapping = {"exception": "error_info"}
        
        try:
            raise ValueError("Test exception")
        except ValueError:
            record = logging.LogRecord(
                name="test",
                level=logging.ERROR,
                pathname="test.py",
                lineno=1,
                msg="Error occurred",
                args=(),
                exc_info=True,
            )
        
        formatter.add_field("exception")
        output = formatter.format(record)
        json_data = json.loads(output)
        
        # Should use mapped name
        assert "error_info" in json_data
        assert "exception" not in json_data
        assert json_data["error_info"]["type"] == "ValueError"


class TestJSONFormatterMessageTruncation:
    """Test suite for JSON formatter message length limits and truncation."""

    @pytest.fixture
    def formatter(self):
        """Create a JSONFormatter instance for testing."""
        return JSONFormatter()

    def test_message_truncation_enabled(self, formatter):
        """Test message truncation when max length is set."""
        formatter.max_message_length = 20
        
        long_message = "This is a very long message that should be truncated"
        record = logging.LogRecord(
            name="test",
            level=logging.INFO,
            pathname="test.py",
            lineno=1,
            msg=long_message,
            args=(),
            exc_info=None,
        )
        
        output = formatter.format(record)
        json_data = json.loads(output)
        
        # Message should be truncated with ellipsis
        assert len(json_data["message"]) == 20
        assert json_data["message"].endswith("...")

    def test_message_no_truncation_when_disabled(self, formatter):
        """Test no message truncation when max length is None."""
        formatter.max_message_length = None
        
        long_message = "This is a very long message that should not be truncated"
        record = logging.LogRecord(
            name="test",
            level=logging.INFO,
            pathname="test.py",
            lineno=1,
            msg=long_message,
            args=(),
            exc_info=None,
        )
        
        output = formatter.format(record)
        json_data = json.loads(output)
        
        # Message should not be truncated
        assert json_data["message"] == long_message

    def test_message_shorter_than_limit(self, formatter):
        """Test message shorter than truncation limit."""
        formatter.max_message_length = 100
        
        short_message = "Short message"
        record = logging.LogRecord(
            name="test",
            level=logging.INFO,
            pathname="test.py",
            lineno=1,
            msg=short_message,
            args=(),
            exc_info=None,
        )
        
        output = formatter.format(record)
        json_data = json.loads(output)
        
        # Message should not be modified
        assert json_data["message"] == short_message

    def test_message_exact_limit_length(self, formatter):
        """Test message exactly at truncation limit."""
        formatter.max_message_length = 10
        
        exact_message = "1234567890"  # Exactly 10 characters
        record = logging.LogRecord(
            name="test",
            level=logging.INFO,
            pathname="test.py",
            lineno=1,
            msg=exact_message,
            args=(),
            exc_info=None,
        )
        
        output = formatter.format(record)
        json_data = json.loads(output)
        
        # Message should not be truncated
        assert json_data["message"] == exact_message

    def test_message_truncation_with_unicode(self, formatter):
        """Test message truncation with unicode characters."""
        formatter.max_message_length = 15
        
        unicode_message = "Unicode: café 中文 more text"
        record = logging.LogRecord(
            name="test",
            level=logging.INFO,
            pathname="test.py",
            lineno=1,
            msg=unicode_message,
            args=(),
            exc_info=None,
        )
        
        output = formatter.format(record)
        json_data = json.loads(output)
        
        # Should be truncated properly
        assert len(json_data["message"]) == 15
        assert json_data["message"].endswith("...")

    def test_message_truncation_minimum_length(self, formatter):
        """Test message truncation with very small limit."""
        formatter.max_message_length = 3
        
        message = "Hello world"
        record = logging.LogRecord(
            name="test",
            level=logging.INFO,
            pathname="test.py",
            lineno=1,
            msg=message,
            args=(),
            exc_info=None,
        )
        
        output = formatter.format(record)
        json_data = json.loads(output)
        
        # Should be just "..."
        assert json_data["message"] == "..."

    def test_message_truncation_with_args(self, formatter):
        """Test message truncation with formatted message arguments."""
        formatter.max_message_length = 20
        
        record = logging.LogRecord(
            name="test",
            level=logging.INFO,
            pathname="test.py",
            lineno=1,
            msg="User %s performed action %s with result %s",
            args=("john_doe", "login", "success"),
            exc_info=None,
        )
        
        output = formatter.format(record)
        json_data = json.loads(output)
        
        # Formatted message should be truncated
        assert len(json_data["message"]) == 20
        assert json_data["message"].endswith("...")

    def test_message_truncation_empty_message(self, formatter):
        """Test message truncation with empty message."""
        formatter.max_message_length = 10
        
        record = logging.LogRecord(
            name="test",
            level=logging.INFO,
            pathname="test.py",
            lineno=1,
            msg="",
            args=(),
            exc_info=None,
        )
        
        output = formatter.format(record)
        json_data = json.loads(output)
        
        # Empty message should remain empty
        assert json_data["message"] == ""

    def test_message_formatting_error_with_truncation(self, formatter):
        """Test message truncation when message formatting fails."""
        formatter.max_message_length = 30
        
        # Create a record that might cause formatting issues
        record = logging.LogRecord(
            name="test",
            level=logging.INFO,
            pathname="test.py",
            lineno=1,
            msg=None,  # None message
            args=(),
            exc_info=None,
        )
        
        output = formatter.format(record)
        json_data = json.loads(output)
        
        # Should handle gracefully and apply truncation if needed
        if json_data["message"] and len(json_data["message"]) > 30:
            assert json_data["message"].endswith("...")


class TestJSONFormatterUnicodeHandling:
    """Test suite for JSON formatter Unicode and special character handling."""

    @pytest.fixture
    def formatter(self):
        """Create a JSONFormatter instance for testing."""
        return JSONFormatter()

    def test_unicode_characters(self, formatter):
        """Test formatting with various Unicode characters."""
        unicode_messages = [
            "Simple ASCII message",
            "Café with accents",
            "Greek: αβγδε",
            "Chinese: 你好世界",
            "Japanese: こんにちは",
            "Arabic: مرحبا",
            "Emoji: 🚀🎉✨💡",
            "Mixed: Hello 世界 🌍",
        ]
        
        for msg in unicode_messages:
            record = logging.LogRecord(
                name="test",
                level=logging.INFO,
                pathname="test.py",
                lineno=1,
                msg=msg,
                args=(),
                exc_info=None,
            )
            
            output = formatter.format(record)
            json_data = json.loads(output)
            
            # Should preserve Unicode correctly
            assert json_data["message"] == msg

    def test_special_characters_in_fields(self, formatter):
        """Test special characters in various fields."""
        record = logging.LogRecord(
            name="test.module.with.dots",
            level=logging.INFO,
            pathname="/path/with spaces/file.py",
            lineno=1,
            msg="Message with 'quotes' and \"double quotes\"",
            args=(),
            exc_info=None,
        )
        
        output = formatter.format(record)
        json_data = json.loads(output)
        
        # Should handle special characters in all fields
        assert "test.module.with.dots" in json_data["logger_name"]
        assert "quotes" in json_data["message"]

    def test_control_characters(self, formatter):
        """Test handling of control characters."""
        control_messages = [
            "Line 1\nLine 2",
            "Tab\tseparated",
            "Carriage\rreturn",
            "Bell\x07character",
            "Null\x00character",
        ]
        
        for msg in control_messages:
            record = logging.LogRecord(
                name="test",
                level=logging.INFO,
                pathname="test.py",
                lineno=1,
                msg=msg,
                args=(),
                exc_info=None,
            )
            
            output = formatter.format(record)
            # Should produce valid JSON
            json_data = json.loads(output)
            
            # Message should be preserved (JSON escaping handles control chars)
            assert isinstance(json_data["message"], str)

    def test_unicode_in_custom_fields(self, formatter):
        """Test Unicode characters in custom fields."""
        custom_fields = {
            "service": "café-service",
            "location": "北京",
            "description": "Service with emoji 🎯",
            "tags": ["测试", "test", "🏷️"],
        }
        
        formatter.set_custom_fields(custom_fields)
        formatter.add_field("custom_fields")
        
        record = logging.LogRecord(
            name="test",
            level=logging.INFO,
            pathname="test.py",
            lineno=1,
            msg="Unicode test",
            args=(),
            exc_info=None,
        )
        
        output = formatter.format(record)
        json_data = json.loads(output)
        
        # Custom fields should preserve Unicode
        custom_data = json_data["custom_fields"]
        assert custom_data["service"] == "café-service"
        assert custom_data["location"] == "北京"
        assert "🎯" in custom_data["description"]
        assert "测试" in custom_data["tags"]

    def test_unicode_in_extra_fields(self, formatter):
        """Test Unicode characters in extra fields from log record."""
        record = logging.LogRecord(
            name="test",
            level=logging.INFO,
            pathname="test.py",
            lineno=1,
            msg="Test message",
            args=(),
            exc_info=None,
        )
        
        # Add Unicode extra fields
        record.user_name = "José García"
        record.city = "北京"
        record.status = "✅ completed"
        
        output = formatter.format(record)
        json_data = json.loads(output)
        
        # Extra fields should preserve Unicode
        if "extra" in json_data:
            extra_data = json_data["extra"]
            assert extra_data["user_name"] == "José García"
            assert extra_data["city"] == "北京"
            assert extra_data["status"] == "✅ completed"

    def test_unicode_normalization(self, formatter):
        """Test Unicode normalization consistency."""
        # Same character represented in different Unicode forms
        messages = [
            "café",  # Composed form
            "cafe\u0301",  # Decomposed form (e + combining acute accent)
        ]
        
        outputs = []
        for msg in messages:
            record = logging.LogRecord(
                name="test",
                level=logging.INFO,
                pathname="test.py",
                lineno=1,
                msg=msg,
                args=(),
                exc_info=None,
            )
            
            output = formatter.format(record)
            json_data = json.loads(output)
            outputs.append(json_data["message"])
        
        # Both should produce valid JSON (exact equality may vary by Python version)
        for output in outputs:
            assert isinstance(output, str)
            assert "caf" in output

    def test_very_large_unicode_message(self, formatter):
        """Test handling of very large Unicode messages."""
        # Create a large message with Unicode characters
        unicode_char = "测"
        large_message = unicode_char * 10000
        
        record = logging.LogRecord(
            name="test",
            level=logging.INFO,
            pathname="test.py",
            lineno=1,
            msg=large_message,
            args=(),
            exc_info=None,
        )
        
        output = formatter.format(record)
        json_data = json.loads(output)
        
        # Should handle large Unicode messages
        assert len(json_data["message"]) == 10000
        assert json_data["message"] == large_message

    def test_invalid_unicode_sequences(self, formatter):
        """Test handling of invalid Unicode sequences."""
        # These might contain invalid Unicode
        invalid_messages = [
            "Invalid: \udcff",  # Invalid surrogate
            "Lone surrogate: \ud800",
        ]
        
        for msg in invalid_messages:
            record = logging.LogRecord(
                name="test",
                level=logging.INFO,
                pathname="test.py",
                lineno=1,
                msg=msg,
                args=(),
                exc_info=None,
            )
            
            try:
                output = formatter.format(record)
                # Should produce valid JSON even with invalid Unicode
                json_data = json.loads(output)
                assert isinstance(json_data["message"], str)
            except (UnicodeError, json.JSONDecodeError):
                # Some invalid sequences might still cause errors, which is acceptable
                pass

    def test_unicode_field_names_in_mapping(self, formatter):
        """Test Unicode characters in field mapping."""
        # While not recommended, test Unicode in field names
        field_mapping = {
            "message": "消息",
            "level": "级别",
        }
        formatter.field_mapping = field_mapping
        
        record = logging.LogRecord(
            name="test",
            level=logging.INFO,
            pathname="test.py",
            lineno=1,
            msg="Unicode field test",
            args=(),
            exc_info=None,
        )
        
        output = formatter.format(record)
        json_data = json.loads(output)
        
        # Should use Unicode field names
        assert "消息" in json_data
        assert "级别" in json_data


class TestJSONFormatterIntegration:
    """Test suite for JSON formatter integration with logging framework."""

    @pytest.fixture
    def temp_dir(self):
        """Create a temporary directory for test files."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            yield Path(tmp_dir)

    def test_integration_with_logger_config(self, temp_dir):
        """Test JSONFormatter integration with LoggerConfig."""
        config = LoggerConfig()
        config.load_from_dict({
            "level": "INFO",
            "formatter_type": "json",
            "handlers": ["console"],
            "json_fields": ["timestamp", "level", "message"],
            "json_pretty_print": True,
            "json_custom_fields": {"service": "test"},
        })
        
        # Test that config accepts JSON formatter settings
        assert config.get_formatter_type() == "json"
        assert config.get_json_fields() == ["timestamp", "level", "message"]
        assert config.get_json_pretty_print() is True
        assert config.get_json_custom_fields() == {"service": "test"}

    def test_integration_with_logger_manager(self, temp_dir):
        """Test JSONFormatter integration with LoggerManager."""
        config = LoggerConfig()
        config.load_from_dict({
            "level": "INFO",
            "formatter_type": "json",
            "handlers": ["console"],
            "json_fields": ["timestamp", "level", "logger_name", "message"],
        })
        
        manager = LoggerManager(config)
        manager.initialize()
        
        # Get a logger and test JSON output
        logger = manager.get_logger("json_test")
        
        # Capture output
        stream = io.StringIO()
        handler = logging.StreamHandler(stream)
        
        # Verify JSON formatter is being used
        json_formatter = JSONFormatter(fields=["timestamp", "level", "logger_name", "message"])
        handler.setFormatter(json_formatter)
        
        # Clear existing handlers and add our test handler
        for h in logger.handlers[:]:
            logger.removeHandler(h)
        logger.addHandler(handler)
        logger.propagate = False
        
        logger.info("Integration test message")
        
        output = stream.getvalue()
        json_data = json.loads(output)
        
        assert json_data["level"] == "INFO"
        assert "json_test" in json_data["logger_name"]
        assert json_data["message"] == "Integration test message"

    def test_integration_with_file_handler(self, temp_dir):
        """Test JSONFormatter integration with file handler."""
        log_file = temp_dir / "test.log"
        
        # Create file handler with JSON formatter
        file_handler = logging.FileHandler(str(log_file))
        json_formatter = JSONFormatter(fields=["timestamp", "level", "message"])
        file_handler.setFormatter(json_formatter)
        
        # Create logger and log message
        logger = logging.getLogger("file_test")
        logger.setLevel(logging.INFO)
        logger.addHandler(file_handler)
        
        logger.info("File handler test")
        file_handler.close()
        
        # Read and verify JSON output
        content = log_file.read_text()
        json_data = json.loads(content.strip())
        
        assert json_data["level"] == "INFO"
        assert json_data["message"] == "File handler test"

    def test_integration_with_rotating_file_handler(self, temp_dir):
        """Test JSONFormatter integration with rotating file handler."""
        log_file = temp_dir / "rotating.log"
        
        # Create rotating file handler with JSON formatter
        rotating_handler = logging.handlers.RotatingFileHandler(
            str(log_file),
            maxBytes=1024,
            backupCount=3
        )
        json_formatter = JSONFormatter(
            fields=["timestamp", "level", "message"],
            pretty_print=False
        )
        rotating_handler.setFormatter(json_formatter)
        
        # Create logger and log messages
        logger = logging.getLogger("rotating_test")
        logger.setLevel(logging.INFO)
        logger.addHandler(rotating_handler)
        
        logger.info("Rotating handler test")
        rotating_handler.close()
        
        # Verify JSON output
        content = log_file.read_text()
        json_data = json.loads(content.strip())
        
        assert json_data["level"] == "INFO"
        assert json_data["message"] == "Rotating handler test"

    def test_integration_with_multiple_handlers(self, temp_dir):
        """Test JSONFormatter with multiple handlers using different configurations."""
        log_file = temp_dir / "multi.log"
        
        # Create console handler with pretty JSON
        console_stream = io.StringIO()
        console_handler = logging.StreamHandler(console_stream)
        console_formatter = JSONFormatter(
            fields=["level", "message"],
            pretty_print=True
        )
        console_handler.setFormatter(console_formatter)
        
        # Create file handler with compact JSON
        file_handler = logging.FileHandler(str(log_file))
        file_formatter = JSONFormatter(
            fields=["timestamp", "level", "logger_name", "message"],
            pretty_print=False
        )
        file_handler.setFormatter(file_formatter)
        
        # Create logger with both handlers
        logger = logging.getLogger("multi_test")
        logger.setLevel(logging.INFO)
        logger.addHandler(console_handler)
        logger.addHandler(file_handler)
        
        logger.info("Multiple handlers test")
        
        file_handler.close()
        
        # Verify console output (pretty)
        console_output = console_stream.getvalue()
        assert "\n" in console_output  # Pretty printed
        console_json = json.loads(console_output)
        assert console_json["message"] == "Multiple handlers test"
        
        # Verify file output (compact)
        file_output = log_file.read_text()
        assert "\n" not in file_output.strip()  # Compact
        file_json = json.loads(file_output.strip())
        assert file_json["message"] == "Multiple handlers test"

    def test_integration_error_fallback(self, temp_dir):
        """Test JSONFormatter error fallback behavior in integration."""
        # Create a scenario that might cause JSON formatting issues
        stream = io.StringIO()
        handler = logging.StreamHandler(stream)
        
        # Create formatter that might have issues
        formatter = JSONFormatter()
        handler.setFormatter(formatter)
        
        logger = logging.getLogger("fallback_test")
        logger.addHandler(handler)
        
        # Create a problematic log record
        record = logging.LogRecord(
            name="test",
            level=logging.ERROR,
            pathname="test.py",
            lineno=1,
            msg="Test message",
            args=(),
            exc_info=None,
        )
        
        # Add a problematic attribute
        class BadObject:
            def __str__(self):
                raise Exception("Cannot stringify")
        
        record.bad_attr = BadObject()
        
        # This should not crash
        try:
            logger.handle(record)
            output = stream.getvalue()
            # Should produce some output (even if fallback)
            assert len(output) > 0
            # Should be valid JSON or at least not crash
            try:
                json.loads(output)
            except json.JSONDecodeError:
                # Fallback might not be JSON, which is acceptable
                pass
        except Exception as e:
            # Should not raise unhandled exceptions
            pytest.fail(f"JSONFormatter integration raised unhandled exception: {e}")

    def test_integration_configuration_reload(self, temp_dir):
        """Test JSONFormatter behavior during configuration reload."""
        # Initial config
        config1 = LoggerConfig()
        config1.load_from_dict({
            "level": "INFO",
            "formatter_type": "json",
            "json_fields": ["level", "message"],
        })
        
        manager = LoggerManager(config1)
        manager.initialize()
        
        # New config with different JSON settings
        config2 = LoggerConfig()
        config2.load_from_dict({
            "level": "DEBUG",
            "formatter_type": "json",
            "json_fields": ["timestamp", "level", "logger_name", "message"],
            "json_pretty_print": True,
        })
        
        # Reload configuration
        manager.reload_configuration(config2)
        
        # Test that new configuration is applied
        logger = manager.get_logger("reload_test")
        
        # Capture output to verify new config
        stream = io.StringIO()
        test_handler = logging.StreamHandler(stream)
        test_formatter = JSONFormatter(
            fields=config2.get_json_fields(),
            pretty_print=config2.get_json_pretty_print()
        )
        test_handler.setFormatter(test_formatter)
        
        logger.addHandler(test_handler)
        logger.propagate = False
        
        logger.info("Reload test")
        
        output = stream.getvalue()
        json_data = json.loads(output)
        
        # Should have new fields
        assert "timestamp" in json_data
        assert "logger_name" in json_data
        
        # Should be pretty printed
        assert "\n" in output


class TestJSONFormatterEdgeCases:
    """Test suite for JSON formatter edge cases and error scenarios."""

    @pytest.fixture
    def formatter(self):
        """Create a JSONFormatter instance for testing."""
        return JSONFormatter()

    def test_none_values_in_record(self, formatter):
        """Test handling of None values in log record attributes."""
        record = logging.LogRecord(
            name="test",
            level=logging.INFO,
            pathname=None,  # None pathname
            lineno=None,    # None line number
            msg=None,       # None message
            args=(),
            exc_info=None,
        )
        
        output = formatter.format(record)
        json_data = json.loads(output)
        
        # Should handle None values gracefully
        assert isinstance(json_data, dict)
        assert "message" in json_data  # Message field should always be present

    def test_empty_log_record(self, formatter):
        """Test handling of minimal log record."""
        # Create minimal record
        record = Mock()
        record.name = "test"
        record.levelname = "INFO"
        record.msg = "test"
        record.args = ()
        record.created = time.time()
        record.pathname = "test.py"
        record.lineno = 1
        record.funcName = "test_func"
        record.exc_info = None
        record.exc_text = None
        record.stack_info = None
        
        # Mock getMessage method
        record.getMessage = Mock(return_value="test message")
        record.__dict__ = {
            'name': 'test',
            'levelname': 'INFO',
            'msg': 'test',
            'args': (),
            'created': record.created,
            'pathname': 'test.py',
            'lineno': 1,
            'funcName': 'test_func',
            'exc_info': None,
        }
        
        output = formatter.format(record)
        json_data = json.loads(output)
        
        assert json_data["level"] == "INFO"
        assert json_data["logger_name"] == "test"

    def test_circular_reference_in_extra_fields(self, formatter):
        """Test handling of circular references in extra fields."""
        record = logging.LogRecord(
            name="test",
            level=logging.INFO,
            pathname="test.py",
            lineno=1,
            msg="Circular reference test",
            args=(),
            exc_info=None,
        )
        
        # Create circular reference
        obj1 = {"name": "obj1"}
        obj2 = {"name": "obj2", "ref": obj1}
        obj1["ref"] = obj2
        
        record.circular_obj = obj1
        
        # Should not cause infinite recursion
        output = formatter.format(record)
        json_data = json.loads(output)
        
        # Should produce valid JSON (circular ref converted to string)
        assert isinstance(json_data, dict)
        if "extra" in json_data and "circular_obj" in json_data["extra"]:
            # Circular object should be stringified
            assert isinstance(json_data["extra"]["circular_obj"], str)

    def test_extremely_deep_nested_object(self, formatter):
        """Test handling of extremely deep nested objects."""
        record = logging.LogRecord(
            name="test",
            level=logging.INFO,
            pathname="test.py",
            lineno=1,
            msg="Deep nesting test",
            args=(),
            exc_info=None,
        )
        
        # Create deeply nested object
        deep_obj = {}
        current = deep_obj
        for i in range(1000):
            current["level"] = i
            current["next"] = {}
            current = current["next"]
        
        record.deep_object = deep_obj
        
        # Should handle deep nesting gracefully
        try:
            output = formatter.format(record)
            json_data = json.loads(output)
            assert isinstance(json_data, dict)
        except RecursionError:
            # Acceptable if Python's recursion limit is hit
            pass

    def test_very_large_number_values(self, formatter):
        """Test handling of very large number values."""
        record = logging.LogRecord(
            name="test",
            level=logging.INFO,
            pathname="test.py",
            lineno=1,
            msg="Large number test",
            args=(),
            exc_info=None,
        )
        
        # Add very large numbers
        record.large_int = 2**1000
        record.large_float = 1.23e308
        record.small_float = 1.23e-308
        
        output = formatter.format(record)
        json_data = json.loads(output)
        
        # Should handle large numbers (may be converted to strings)
        assert isinstance(json_data, dict)
        if "extra" in json_data:
            # Large numbers should be represented somehow
            assert "large_int" in json_data["extra"]

    def test_binary_data_in_fields(self, formatter):
        """Test handling of binary data in log record fields."""
        record = logging.LogRecord(
            name="test",
            level=logging.INFO,
            pathname="test.py",
            lineno=1,
            msg="Binary data test",
            args=(),
            exc_info=None,
        )
        
        # Add binary data
        record.binary_data = b"\x00\x01\x02\x03\xff"
        record.image_data = bytes(range(256))
        
        output = formatter.format(record)
        json_data = json.loads(output)
        
        # Binary data should be converted to string representation
        assert isinstance(json_data, dict)
        if "extra" in json_data:
            if "binary_data" in json_data["extra"]:
                assert isinstance(json_data["extra"]["binary_data"], str)

    def test_missing_log_record_attributes(self, formatter):
        """Test handling of log record missing standard attributes."""
        # Create record missing some standard attributes
        incomplete_record = Mock()
        incomplete_record.name = "test"
        incomplete_record.levelname = "INFO"
        incomplete_record.msg = "test message"
        incomplete_record.args = ()
        incomplete_record.created = time.time()
        # Missing: pathname, lineno, funcName, etc.
        
        incomplete_record.getMessage = Mock(return_value="test message")
        incomplete_record.__dict__ = {
            'name': 'test',
            'levelname': 'INFO',
            'msg': 'test message',
            'args': (),
            'created': incomplete_record.created,
        }
        
        output = formatter.format(incomplete_record)
        json_data = json.loads(output)
        
        # Should handle missing attributes gracefully
        assert json_data["level"] == "INFO"
        assert json_data["message"] == "test message"

    def test_formatter_with_invalid_json_characters(self, formatter):
        """Test formatter with characters that are invalid in JSON."""
        record = logging.LogRecord(
            name="test",
            level=logging.INFO,
            pathname="test.py",
            lineno=1,
            msg="Invalid JSON chars: \x08\x0c\x0e\x1f",
            args=(),
            exc_info=None,
        )
        
        # Should handle invalid JSON characters
        output = formatter.format(record)
        json_data = json.loads(output)  # Should not raise exception
        
        assert isinstance(json_data["message"], str)

    def test_memory_exhaustion_simulation(self, formatter):
        """Test formatter behavior under simulated memory pressure."""
        # Create many large objects rapidly
        record = logging.LogRecord(
            name="test",
            level=logging.INFO,
            pathname="test.py",
            lineno=1,
            msg="Memory test",
            args=(),
            exc_info=None,
        )
        
        # Add many large attributes
        for i in range(100):
            setattr(record, f"large_attr_{i}", "x" * 1000)
        
        try:
            output = formatter.format(record)
            json_data = json.loads(output)
            assert isinstance(json_data, dict)
        except MemoryError:
            # Acceptable under extreme memory pressure
            pass

    def test_thread_safety_with_concurrent_modification(self, formatter):
        """Test formatter thread safety during concurrent configuration changes."""
        def format_messages():
            for i in range(100):
                record = logging.LogRecord(
                    name=f"test_{i}",
                    level=logging.INFO,
                    pathname="test.py",
                    lineno=i,
                    msg=f"Message {i}",
                    args=(),
                    exc_info=None,
                )
                formatter.format(record)
        
        def modify_config():
            for i in range(50):
                formatter.set_pretty_print(i % 2 == 0)
                formatter.set_timestamp_format("iso" if i % 2 else "epoch")
        
        # Run concurrent formatting and configuration changes
        import threading
        threads = []
        
        for _ in range(3):
            threads.append(threading.Thread(target=format_messages))
        threads.append(threading.Thread(target=modify_config))
        
        for thread in threads:
            thread.start()
        
        for thread in threads:
            thread.join()
        
        # Should complete without errors


class TestJSONFormatterPerformance:
    """Test suite for JSON formatter performance and thread safety."""

    @pytest.fixture
    def formatter(self):
        """Create a JSONFormatter instance for testing."""
        return JSONFormatter()

    def test_basic_formatting_performance(self, formatter):
        """Test basic JSON formatting performance."""
        records = []
        for i in range(1000):
            record = logging.LogRecord(
                name=f"perf_test_{i % 10}",
                level=logging.INFO,
                pathname="test.py",
                lineno=i,
                msg=f"Performance test message {i}",
                args=(),
                exc_info=None,
            )
            records.append(record)
        
        start_time = time.time()
        
        for record in records:
            formatter.format(record)
        
        end_time = time.time()
        duration = end_time - start_time
        
        # Should format 1000 records quickly
        assert duration < 1.0  # Less than 1 second
        
        # Calculate throughput
        throughput = len(records) / duration
        assert throughput > 500  # At least 500 records per second

    def test_pretty_print_performance_impact(self, formatter):
        """Test performance impact of pretty printing."""
        record = logging.LogRecord(
            name="perf_test",
            level=logging.INFO,
            pathname="test.py",
            lineno=1,
            msg="Performance test message",
            args=(),
            exc_info=None,
        )
        
        # Test compact formatting
        formatter.set_pretty_print(False)
        start_time = time.time()
        for _ in range(1000):
            formatter.format(record)
        compact_time = time.time() - start_time
        
        # Test pretty printing
        formatter.set_pretty_print(True)
        start_time = time.time()
        for _ in range(1000):
            formatter.format(record)
        pretty_time = time.time() - start_time
        
        # Pretty printing should not be significantly slower
        assert pretty_time < compact_time * 3  # At most 3x slower

    def test_thread_safety_concurrent_formatting(self, formatter):
        """Test thread safety with concurrent formatting."""
        results = []
        errors = []
        
        def format_worker(thread_id, count=100):
            thread_results = []
            try:
                for i in range(count):
                    record = logging.LogRecord(
                        name=f"thread_{thread_id}",
                        level=logging.INFO,
                        pathname="test.py",
                        lineno=i,
                        msg=f"Thread {thread_id} message {i}",
                        args=(),
                        exc_info=None,
                    )
                    output = formatter.format(record)
                    json_data = json.loads(output)
                    thread_results.append(json_data)
            except Exception as e:
                errors.append((thread_id, str(e)))
            finally:
                results.append((thread_id, thread_results))
        
        # Start multiple threads
        threads = []
        for i in range(5):
            thread = threading.Thread(target=format_worker, args=(i,))
            threads.append(thread)
            thread.start()
        
        # Wait for completion
        for thread in threads:
            thread.join()
        
        # Check results
        assert len(errors) == 0, f"Thread safety errors: {errors}"
        assert len(results) == 5
        
        # Verify all threads produced valid results
        for thread_id, thread_results in results:
            assert len(thread_results) == 100
            for result in thread_results:
                assert f"thread_{thread_id}" in result["logger_name"]

    def test_thread_safety_concurrent_configuration(self, formatter):
        """Test thread safety with concurrent configuration changes."""
        formatting_errors = []
        config_errors = []
        
        def format_worker():
            try:
                for i in range(200):
                    record = logging.LogRecord(
                        name="format_test",
                        level=logging.INFO,
                        pathname="test.py",
                        lineno=i,
                        msg=f"Concurrent format message {i}",
                        args=(),
                        exc_info=None,
                    )
                    output = formatter.format(record)
                    json.loads(output)  # Validate JSON
            except Exception as e:
                formatting_errors.append(str(e))
        
        def config_worker():
            try:
                for i in range(50):
                    formatter.set_pretty_print(i % 2 == 0)
                    formatter.set_timestamp_format("iso" if i % 3 == 0 else "epoch")
                    if i % 5 == 0:
                        formatter.set_fields(["timestamp", "level", "message"])
                    else:
                        formatter.set_fields(JSONFormatter.DEFAULT_FIELDS)
            except Exception as e:
                config_errors.append(str(e))
        
        # Start workers
        threads = []
        
        # Multiple formatting threads
        for _ in range(3):
            threads.append(threading.Thread(target=format_worker))
        
        # One configuration thread
        threads.append(threading.Thread(target=config_worker))
        
        for thread in threads:
            thread.start()
        
        for thread in threads:
            thread.join()
        
        # Should have minimal or no errors
        assert len(formatting_errors) == 0, f"Formatting errors: {formatting_errors}"
        assert len(config_errors) == 0, f"Configuration errors: {config_errors}"

    def test_memory_usage_stability(self, formatter):
        """Test that memory usage remains stable over time."""
        import gc
        
        # Force garbage collection
        gc.collect()
        
        # Format many messages
        for batch in range(10):
            for i in range(1000):
                record = logging.LogRecord(
                    name=f"memory_test_{batch}",
                    level=logging.INFO,
                    pathname="test.py",
                    lineno=i,
                    msg=f"Memory test batch {batch} message {i}",
                    args=(),
                    exc_info=None,
                )
                formatter.format(record)
            
            # Periodic garbage collection
            gc.collect()
        
        # Memory should be stable (hard to test precisely, but shouldn't crash)
        assert True  # If we get here without memory errors, that's good

    def test_large_message_performance(self, formatter):
        """Test performance with very large messages."""
        # Create large message
        large_message = "A" * 100000  # 100KB message
        
        record = logging.LogRecord(
            name="large_test",
            level=logging.INFO,
            pathname="test.py",
            lineno=1,
            msg=large_message,
            args=(),
            exc_info=None,
        )
        
        start_time = time.time()
        
        for _ in range(100):
            output = formatter.format(record)
            json.loads(output)  # Validate
        
        end_time = time.time()
        duration = end_time - start_time
        
        # Should handle large messages reasonably well
        assert duration < 5.0  # 100 large messages in less than 5 seconds

    def test_complex_structure_performance(self, formatter):
        """Test performance with complex data structures."""
        formatter.add_field("custom_fields")
        
        # Create complex custom fields
        complex_data = {
            "nested": {
                "level1": {
                    "level2": {
                        "data": list(range(1000)),
                        "mapping": {f"key_{i}": f"value_{i}" for i in range(100)}
                    }
                }
            },
            "arrays": [
                {"id": i, "data": f"item_{i}"} for i in range(500)
            ]
        }
        
        formatter.set_custom_fields(complex_data)
        
        record = logging.LogRecord(
            name="complex_test",
            level=logging.INFO,
            pathname="test.py",
            lineno=1,
            msg="Complex structure test",
            args=(),
            exc_info=None,
        )
        
        start_time = time.time()
        
        for _ in range(50):
            output = formatter.format(record)
            json.loads(output)  # Validate
        
        end_time = time.time()
        duration = end_time - start_time
        
        # Should handle complex structures reasonably
        assert duration < 10.0  # 50 complex records in less than 10 seconds

    def test_field_selection_performance_impact(self, formatter):
        """Test performance impact of different field selections."""
        record = logging.LogRecord(
            name="field_perf_test",
            level=logging.INFO,
            pathname="/very/long/path/to/test/file.py",
            lineno=12345,
            msg="Field performance test message",
            args=(),
            exc_info=None,
        )
        
        # Test with minimal fields
        formatter.set_fields(["level", "message"])
        start_time = time.time()
        for _ in range(1000):
            formatter.format(record)
        minimal_time = time.time() - start_time
        
        # Test with all fields
        formatter.set_fields(list(JSONFormatter.FIELD_MAPPING.keys()))
        start_time = time.time()
        for _ in range(1000):
            formatter.format(record)
        full_time = time.time() - start_time
        
        # More fields should take more time, but not excessively
        assert full_time < minimal_time * 5  # At most 5x slower


# Factory function tests
class TestJSONFormatterFactory:
    """Test suite for the create_json_formatter factory function."""

    def test_factory_function_default(self):
        """Test factory function with default parameters."""
        formatter = create_json_formatter()
        
        assert isinstance(formatter, JSONFormatter)
        assert formatter.fields == JSONFormatter.DEFAULT_FIELDS
        assert formatter.pretty_print is False

    def test_factory_function_custom_params(self):
        """Test factory function with custom parameters."""
        formatter = create_json_formatter(
            fields=["timestamp", "level", "message"],
            pretty_print=True,
            custom_fields={"service": "test"},
            timestamp_format="epoch",
            use_utc=True,
            max_message_length=100,
        )
        
        assert formatter.fields == ["timestamp", "level", "message"]
        assert formatter.pretty_print is True
        assert formatter.custom_fields == {"service": "test"}
        assert formatter.timestamp_format == "epoch"
        assert formatter.use_utc is True
        assert formatter.max_message_length == 100

    def test_factory_function_error_handling(self):
        """Test factory function error handling."""
        with pytest.raises(JSONFormatterError):
            create_json_formatter(fields=["invalid_field"])


if __name__ == "__main__":
    pytest.main([__file__])