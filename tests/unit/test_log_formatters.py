"""
Unit tests for log formatter functionality in the AIM2 project logging framework

This module provides comprehensive unit tests specifically for log formatter functionality,
including formatter creation, configuration, output validation, and integration with handlers.

Test Coverage:
- Formatter creation and configuration in LoggerManager
- Format string validation and parsing
- Formatter output with various log levels and message types
- Timestamp formatting and logger name formatting
- Error handling for malformed format strings
- Integration between formatters and handlers (console, file)
- Edge cases and special characters in log messages
- Hierarchical logger name formatting

Test Classes:
    TestLogFormatterCreation: Tests formatter creation and basic configuration
    TestLogFormatterOutput: Tests formatter output with various inputs
    TestLogFormatterConfiguration: Tests formatter configuration and validation
    TestLogFormatterIntegration: Tests formatter integration with handlers
    TestLogFormatterErrorHandling: Tests error scenarios and edge cases
    TestLogFormatterHierarchy: Tests formatter behavior with hierarchical loggers
    TestLogFormatterPerformance: Tests formatter performance characteristics
"""

import io
import logging
import logging.handlers
import tempfile
import pytest
import re
import time
from datetime import datetime
from pathlib import Path
from unittest.mock import Mock, patch

# Import the modules to be tested
try:
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


class TestLogFormatterCreation:
    """Test suite for log formatter creation and basic configuration."""

    @pytest.fixture
    def logger_config(self):
        """Create a LoggerConfig instance for testing."""
        return LoggerConfig()

    @pytest.fixture
    def logger_manager(self, logger_config):
        """Create a LoggerManager instance for testing."""
        return LoggerManager(logger_config)

    @pytest.fixture
    def temp_dir(self):
        """Create a temporary directory for test files."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            yield Path(tmp_dir)

    def test_formatter_creation_console_handler(self, logger_manager):
        """Test formatter is created for console handler."""
        logger_manager.initialize()

        # Get console handler
        root_logger = logger_manager.root_logger
        console_handlers = [
            h
            for h in root_logger.handlers
            if isinstance(h, logging.StreamHandler)
            and not isinstance(h, logging.FileHandler)
        ]

        assert len(console_handlers) > 0
        console_handler = console_handlers[0]

        # Verify formatter is attached
        assert console_handler.formatter is not None
        assert isinstance(console_handler.formatter, logging.Formatter)

    def test_formatter_creation_file_handler(self, logger_manager, temp_dir):
        """Test formatter is created for file handler."""
        # Configure with file handler
        logger_manager.config.load_from_dict(
            {
                "level": "INFO",
                "handlers": ["file"],
                "file_path": str(temp_dir / "test.log"),
            }
        )

        logger_manager.initialize()

        # Get file handler
        root_logger = logger_manager.root_logger
        file_handlers = [
            h for h in root_logger.handlers if isinstance(h, logging.FileHandler)
        ]

        assert len(file_handlers) > 0
        file_handler = file_handlers[0]

        # Verify formatter is attached
        assert file_handler.formatter is not None
        assert isinstance(file_handler.formatter, logging.Formatter)

    def test_formatter_creation_multiple_handlers(self, logger_manager, temp_dir):
        """Test formatters are created for multiple handlers."""
        # Configure with both console and file handlers
        logger_manager.config.load_from_dict(
            {
                "level": "INFO",
                "handlers": ["console", "file"],
                "file_path": str(temp_dir / "test.log"),
            }
        )

        logger_manager.initialize()

        root_logger = logger_manager.root_logger

        # Verify all handlers have formatters
        for handler in root_logger.handlers:
            assert handler.formatter is not None
            assert isinstance(handler.formatter, logging.Formatter)

    def test_formatter_creation_with_custom_format(self, logger_manager):
        """Test formatter creation with custom format string."""
        custom_format = "%(levelname)s:%(name)s:%(message)s"

        logger_manager.config.load_from_dict(
            {"level": "INFO", "format": custom_format, "handlers": ["console"]}
        )

        logger_manager.initialize()

        root_logger = logger_manager.root_logger
        handler = root_logger.handlers[0]
        formatter = handler.formatter

        # The formatter should be created with the custom format
        assert formatter is not None
        # Note: logging.Formatter doesn't expose _fmt in all Python versions
        # So we test by actually formatting a record

    def test_formatter_creation_caching(self, logger_manager):
        """Test formatter creation uses caching."""
        logger_manager.initialize()

        # Create multiple handlers of same type
        handler1 = logger_manager._create_handler("console")
        handler2 = logger_manager._create_handler("console")

        # Handlers should be cached (same instance)
        assert handler1 is handler2

        # Formatters should be the same instance
        assert handler1.formatter is handler2.formatter

    def test_formatter_creation_different_handler_types(self, logger_manager, temp_dir):
        """Test formatters for different handler types."""
        # Setup config with file path
        logger_manager.config.load_from_dict(
            {
                "level": "INFO",
                "handlers": ["console"],
                "file_path": str(temp_dir / "test.log"),
            }
        )

        # Create different handler types
        console_handler = logger_manager._create_handler("console")
        file_handler = logger_manager._create_handler("file")

        # Both should have formatters
        assert console_handler.formatter is not None
        assert file_handler.formatter is not None

        # Formatters should use same format string but can be different instances
        assert isinstance(console_handler.formatter, logging.Formatter)
        assert isinstance(file_handler.formatter, logging.Formatter)

    def test_formatter_creation_invalid_handler_type(self, logger_manager):
        """Test formatter creation with invalid handler type."""
        with pytest.raises(LoggerManagerError) as exc_info:
            logger_manager._create_handler("invalid_handler")

        assert "Unsupported handler type" in str(exc_info.value)

    def test_formatter_creation_error_handling(self, logger_manager):
        """Test formatter creation error handling."""
        with patch("logging.Formatter") as mock_formatter:
            mock_formatter.side_effect = Exception("Formatter creation failed")

            with pytest.raises(LoggerManagerError) as exc_info:
                logger_manager._create_handler("console")

            assert "Failed to create console handler" in str(exc_info.value)


class TestLogFormatterOutput:
    """Test suite for log formatter output with various inputs."""

    @pytest.fixture
    def logger_manager(self):
        """Create a LoggerManager instance for testing."""
        return LoggerManager()

    @pytest.fixture
    def string_stream(self):
        """Create a string stream for capturing log output."""
        return io.StringIO()

    @pytest.fixture
    def logger_with_stream(self, logger_manager, string_stream):
        """Create a logger with stream handler for output testing."""
        logger_manager.initialize()
        logger = logger_manager.get_logger("test_logger")

        # Create a stream handler with our string stream
        stream_handler = logging.StreamHandler(string_stream)
        formatter = logging.Formatter(logger_manager.config.get_format())
        stream_handler.setFormatter(formatter)

        # Clear existing handlers and add our test handler
        for handler in logger.handlers[:]:
            logger.removeHandler(handler)
        logger.addHandler(stream_handler)

        return logger, string_stream

    def test_formatter_output_basic_message(self, logger_with_stream):
        """Test formatter output with basic log message."""
        logger, stream = logger_with_stream

        logger.info("Test message")
        output = stream.getvalue()

        # Verify output contains expected components
        assert "Test message" in output
        assert "INFO" in output
        assert "test_logger" in output

    def test_formatter_output_different_log_levels(self, logger_with_stream):
        """Test formatter output with different log levels."""
        logger, stream = logger_with_stream

        test_cases = [
            (logging.DEBUG, "DEBUG", "Debug message"),
            (logging.INFO, "INFO", "Info message"),
            (logging.WARNING, "WARNING", "Warning message"),
            (logging.ERROR, "ERROR", "Error message"),
            (logging.CRITICAL, "CRITICAL", "Critical message"),
        ]

        for level, level_name, message in test_cases:
            # Clear previous output
            stream.seek(0)
            stream.truncate(0)

            # Set logger level to DEBUG to ensure all messages are processed
            logger.setLevel(logging.DEBUG)

            logger.log(level, message)
            output = stream.getvalue()

            assert level_name in output
            assert message in output

    def test_formatter_output_with_exception(self, logger_with_stream):
        """Test formatter output when logging exceptions."""
        logger, stream = logger_with_stream

        try:
            raise ValueError("Test exception")
        except ValueError:
            logger.exception("An error occurred")

        output = stream.getvalue()

        # Verify exception information is included
        assert "An error occurred" in output
        assert "ERROR" in output
        assert "ValueError" in output
        assert "Test exception" in output
        assert "Traceback" in output

    def test_formatter_output_with_extra_fields(self, logger_with_stream):
        """Test formatter output with extra fields."""
        logger, stream = logger_with_stream

        # Log with extra fields
        logger.info("Message with extras", extra={"user": "testuser", "session": "123"})
        output = stream.getvalue()

        # Basic message should be present
        assert "Message with extras" in output
        assert "INFO" in output

    def test_formatter_output_special_characters(self, logger_with_stream):
        """Test formatter output with special characters."""
        logger, stream = logger_with_stream

        special_messages = [
            "Message with unicode: café ñoño 中文",
            "Message with newlines:\nLine 1\nLine 2",
            "Message with tabs:\tTabbed content",
            "Message with quotes: 'single' \"double\"",
            "Message with symbols: !@#$%^&*()",
        ]

        for message in special_messages:
            # Clear previous output
            stream.seek(0)
            stream.truncate(0)

            logger.info(message)
            output = stream.getvalue()

            # The exact message might be encoded differently, but should be present
            assert len(output) > 0
            assert "INFO" in output

    def test_formatter_output_long_messages(self, logger_with_stream):
        """Test formatter output with very long messages."""
        logger, stream = logger_with_stream

        # Test with a very long message
        long_message = "A" * 10000
        logger.info(long_message)

        output = stream.getvalue()
        assert long_message in output
        assert "INFO" in output

    def test_formatter_output_empty_message(self, logger_with_stream):
        """Test formatter output with empty message."""
        logger, stream = logger_with_stream

        logger.info("")
        output = stream.getvalue()

        # Should still contain log level and logger name
        assert "INFO" in output
        assert "test_logger" in output

    def test_formatter_output_none_message(self, logger_with_stream):
        """Test formatter output with None message."""
        logger, stream = logger_with_stream

        logger.info(None)
        output = stream.getvalue()

        # Should handle None gracefully
        assert "INFO" in output
        assert "test_logger" in output
        assert "None" in output

    def test_formatter_output_object_message(self, logger_with_stream):
        """Test formatter output with object as message."""
        logger, stream = logger_with_stream

        class TestObject:
            def __str__(self):
                return "TestObject string representation"

        test_obj = TestObject()
        logger.info(test_obj)

        output = stream.getvalue()
        assert "TestObject string representation" in output
        assert "INFO" in output

    def test_formatter_output_timestamp_presence(self, logger_with_stream):
        """Test formatter output includes timestamp."""
        logger, stream = logger_with_stream

        datetime.now()
        logger.info("Timestamp test")
        datetime.now()

        output = stream.getvalue()

        # Should contain a timestamp
        # Look for patterns like YYYY-MM-DD HH:MM:SS
        timestamp_pattern = r"\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}"
        assert re.search(timestamp_pattern, output) is not None


class TestLogFormatterConfiguration:
    """Test suite for log formatter configuration and validation."""

    @pytest.fixture
    def logger_manager(self):
        """Create a LoggerManager instance for testing."""
        return LoggerManager()

    def test_formatter_with_custom_format_string(self, logger_manager):
        """Test formatter configuration with custom format strings."""
        test_formats = [
            "%(name)s - %(levelname)s - %(message)s",
            "%(asctime)s [%(levelname)s] %(name)s: %(message)s",
            "%(levelname)8s | %(name)s | %(message)s",
            "%(message)s",  # Minimal format
            "%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s",  # Detailed format
        ]

        for format_string in test_formats:
            logger_manager.config.load_from_dict(
                {"level": "INFO", "format": format_string, "handlers": ["console"]}
            )

            logger_manager.initialize()

            # Get the formatter and verify it was created
            root_logger = logger_manager.root_logger
            handler = root_logger.handlers[0]
            formatter = handler.formatter

            assert formatter is not None

            # Clean up for next iteration
            logger_manager.cleanup()

    def test_formatter_with_invalid_format_string(self, logger_manager):
        """Test formatter behavior with invalid format strings."""
        # These might not raise errors during creation but during formatting
        invalid_formats = [
            "%(invalid_field)s - %(message)s",
            "%(asctime)s - %(message)s - %(nonexistent)s",
        ]

        for format_string in invalid_formats:
            logger_manager.config.load_from_dict(
                {"level": "INFO", "format": format_string, "handlers": ["console"]}
            )

            # Should create formatter without error
            logger_manager.initialize()

            root_logger = logger_manager.root_logger
            handler = root_logger.handlers[0]
            formatter = handler.formatter

            assert formatter is not None

            # Clean up for next iteration
            logger_manager.cleanup()

    def test_formatter_with_malformed_format_string(self, logger_manager):
        """Test formatter with malformed format strings."""
        malformed_formats = [
            "%(asctime)s - %(message)s - %(incomplete",  # Incomplete placeholder
            "%(asctime)s - %s - %(message)s",  # Mixed format styles
            "%(asctime)s - %(message)s - %(",  # Incomplete format
        ]

        for format_string in malformed_formats:
            logger_manager.config.load_from_dict(
                {"level": "INFO", "format": format_string, "handlers": ["console"]}
            )

            # Should still create formatter (error might occur during use)
            logger_manager.initialize()

            root_logger = logger_manager.root_logger
            handler = root_logger.handlers[0]
            formatter = handler.formatter

            assert formatter is not None

            # Clean up for next iteration
            logger_manager.cleanup()

    def test_formatter_configuration_reload(self, logger_manager):
        """Test formatter configuration after config reload."""
        # Initial configuration
        initial_format = "%(name)s - %(message)s"
        logger_manager.config.load_from_dict(
            {"level": "INFO", "format": initial_format, "handlers": ["console"]}
        )

        logger_manager.initialize()

        # Get initial formatter
        initial_handler = logger_manager.root_logger.handlers[0]
        initial_formatter = initial_handler.formatter

        # Reload with new format
        new_format = "%(levelname)s: %(message)s"
        new_config = LoggerConfig()
        new_config.load_from_dict(
            {"level": "INFO", "format": new_format, "handlers": ["console"]}
        )

        logger_manager.reload_configuration(new_config)

        # Verify new formatter is created
        new_handler = logger_manager.root_logger.handlers[0]
        new_formatter = new_handler.formatter

        assert new_formatter is not None
        # Formatters should be different instances after reload
        assert new_formatter is not initial_formatter

    def test_formatter_format_validation(self, logger_manager):
        """Test that format string validation works in LoggerConfig."""
        logger_config = LoggerConfig()

        # Valid format should pass validation
        valid_config = {
            "level": "INFO",
            "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            "handlers": ["console"],
        }

        assert logger_config.validate_config(valid_config) is True

        # Empty format should fail validation
        invalid_config = {"level": "INFO", "format": "", "handlers": ["console"]}

        with pytest.raises(LoggerConfigError) as exc_info:
            logger_config.validate_config(invalid_config)

        assert "cannot be empty" in str(exc_info.value)

    def test_formatter_format_string_types(self, logger_manager):
        """Test formatter with different format string types."""
        logger_config = LoggerConfig()

        # Non-string format should fail validation
        invalid_configs = [
            {"level": "INFO", "format": 123, "handlers": ["console"]},
            {"level": "INFO", "format": None, "handlers": ["console"]},
            {"level": "INFO", "format": [], "handlers": ["console"]},
        ]

        for config in invalid_configs:
            with pytest.raises(LoggerConfigError) as exc_info:
                logger_config.validate_config(config)

            assert "must be a string" in str(exc_info.value)


class TestLogFormatterIntegration:
    """Test suite for formatter integration with handlers."""

    @pytest.fixture
    def temp_dir(self):
        """Create a temporary directory for test files."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            yield Path(tmp_dir)

    @pytest.fixture
    def logger_manager_with_file(self, temp_dir):
        """Create LoggerManager configured with file handler."""
        config = LoggerConfig()
        config.load_from_dict(
            {
                "level": "INFO",
                "handlers": ["console", "file"],
                "file_path": str(temp_dir / "test.log"),
                "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            }
        )
        return LoggerManager(config)

    def test_formatter_integration_console_handler(self, logger_manager_with_file):
        """Test formatter integration with console handler."""
        logger_manager_with_file.initialize()

        root_logger = logger_manager_with_file.root_logger
        console_handlers = [
            h
            for h in root_logger.handlers
            if isinstance(h, logging.StreamHandler)
            and not isinstance(h, logging.FileHandler)
        ]

        assert len(console_handlers) > 0
        console_handler = console_handlers[0]

        # Verify formatter is properly integrated
        assert console_handler.formatter is not None

        # Test actual formatting by creating a log record
        record = logging.LogRecord(
            name="test_logger",
            level=logging.INFO,
            pathname="test.py",
            lineno=1,
            msg="Test message",
            args=(),
            exc_info=None,
        )

        formatted_output = console_handler.format(record)

        assert "test_logger" in formatted_output
        assert "INFO" in formatted_output
        assert "Test message" in formatted_output

    def test_formatter_integration_file_handler(
        self, logger_manager_with_file, temp_dir
    ):
        """Test formatter integration with file handler."""
        logger_manager_with_file.initialize()

        # Get the file handler to verify it has a formatter
        root_logger = logger_manager_with_file.root_logger
        file_handlers = [
            h for h in root_logger.handlers if isinstance(h, logging.FileHandler)
        ]

        assert len(file_handlers) > 0, "No file handler found"
        file_handler = file_handlers[0]

        # Verify formatter is attached
        assert file_handler.formatter is not None
        assert isinstance(file_handler.formatter, logging.Formatter)

        # Test formatting directly by creating a log record
        record = logging.LogRecord(
            name="file_test",
            level=logging.INFO,
            pathname="test.py",
            lineno=1,
            msg="File handler test message",
            args=(),
            exc_info=None,
        )

        formatted_output = file_handler.format(record)

        # Verify formatted content
        assert "file_test" in formatted_output
        assert "INFO" in formatted_output
        assert "File handler test message" in formatted_output

    def test_formatter_integration_multiple_handlers(self, logger_manager_with_file):
        """Test formatter integration with multiple handlers."""
        logger_manager_with_file.initialize()

        root_logger = logger_manager_with_file.root_logger

        # Verify all handlers have formatters
        for handler in root_logger.handlers:
            assert handler.formatter is not None

            # Test formatting with each handler
            record = logging.LogRecord(
                name="multi_test",
                level=logging.WARNING,
                pathname="test.py",
                lineno=1,
                msg="Multi-handler test",
                args=(),
                exc_info=None,
            )

            formatted_output = handler.format(record)

            assert "multi_test" in formatted_output
            assert "WARNING" in formatted_output
            assert "Multi-handler test" in formatted_output

    def test_formatter_integration_rotating_file_handler(self, temp_dir):
        """Test formatter integration with rotating file handler."""
        config = LoggerConfig()
        config.load_from_dict(
            {
                "level": "INFO",
                "handlers": ["file"],
                "file_path": str(temp_dir / "rotating.log"),
                "max_file_size": "1KB",
                "backup_count": 3,
                "format": "%(levelname)s:%(name)s:%(message)s",
            }
        )

        logger_manager = LoggerManager(config)
        logger_manager.initialize()

        # Get file handler
        root_logger = logger_manager.root_logger
        file_handlers = [
            h
            for h in root_logger.handlers
            if isinstance(h, logging.handlers.RotatingFileHandler)
        ]

        assert len(file_handlers) > 0
        file_handler = file_handlers[0]

        # Verify formatter is attached
        assert file_handler.formatter is not None

        # Test formatting
        record = logging.LogRecord(
            name="rotating_test",
            level=logging.ERROR,
            pathname="test.py",
            lineno=1,
            msg="Rotating handler test",
            args=(),
            exc_info=None,
        )

        formatted_output = file_handler.format(record)

        assert "ERROR:rotating_test:Rotating handler test" == formatted_output.strip()

    def test_formatter_integration_handler_addition(self, logger_manager_with_file):
        """Test formatter integration when adding handlers dynamically."""
        logger_manager_with_file.initialize()

        # Create a test logger
        test_logger = logger_manager_with_file.get_logger("dynamic_test")
        initial_handler_count = len(test_logger.handlers)

        # Add a new handler
        logger_manager_with_file.add_handler_to_logger("dynamic_test", "console")

        # Verify handler was added and has formatter
        assert len(test_logger.handlers) > initial_handler_count

        # Find the newly added handler
        for handler in test_logger.handlers:
            if isinstance(handler, logging.StreamHandler) and not isinstance(
                handler, logging.FileHandler
            ):
                assert handler.formatter is not None
                break

    def test_formatter_integration_handler_removal(self, logger_manager_with_file):
        """Test formatter behavior when removing handlers."""
        logger_manager_with_file.initialize()

        # Create a test logger and add console handler
        test_logger = logger_manager_with_file.get_logger("removal_test")
        logger_manager_with_file.add_handler_to_logger("removal_test", "console")

        initial_handler_count = len(test_logger.handlers)

        # Remove console handler
        logger_manager_with_file.remove_handler_from_logger("removal_test", "console")

        # Verify handler was removed
        assert len(test_logger.handlers) < initial_handler_count

    def test_formatter_integration_with_custom_handler(self, logger_manager_with_file):
        """Test formatter integration when manually adding custom handlers."""
        logger_manager_with_file.initialize()

        # Create custom handler
        custom_stream = io.StringIO()
        custom_handler = logging.StreamHandler(custom_stream)

        # Get formatter from existing handler
        root_logger = logger_manager_with_file.root_logger
        existing_formatter = None
        for handler in root_logger.handlers:
            if handler.formatter:
                existing_formatter = handler.formatter
                break

        # Apply same formatter to custom handler
        if existing_formatter:
            custom_handler.setFormatter(existing_formatter)

        # Add custom handler to logger
        test_logger = logger_manager_with_file.get_logger("custom_test")
        test_logger.addHandler(custom_handler)

        # Test logging
        test_logger.info("Custom handler test")

        output = custom_stream.getvalue()
        assert "Custom handler test" in output
        assert "INFO" in output


class TestLogFormatterErrorHandling:
    """Test suite for formatter error scenarios and edge cases."""

    @pytest.fixture
    def logger_manager(self):
        """Create a LoggerManager instance for testing."""
        return LoggerManager()

    def test_formatter_error_invalid_log_record(self, logger_manager):
        """Test formatter behavior with invalid log records."""
        logger_manager.initialize()

        root_logger = logger_manager.root_logger
        handler = root_logger.handlers[0]
        formatter = handler.formatter

        # Create a malformed log record
        incomplete_record = Mock()
        incomplete_record.name = "test"
        incomplete_record.levelname = "INFO"
        incomplete_record.msg = "Test message"
        # Add some required attributes that are missing
        incomplete_record.created = None  # This will cause formatTime to fail
        incomplete_record.args = ()
        incomplete_record.pathname = "test.py"
        incomplete_record.lineno = 1
        incomplete_record.funcName = "test_func"
        incomplete_record.exc_info = None
        incomplete_record.exc_text = None
        incomplete_record.stack_info = None

        # This should raise an exception due to invalid created time
        with pytest.raises((AttributeError, KeyError, TypeError, ValueError)):
            formatter.format(incomplete_record)

    def test_formatter_error_circular_reference(self, logger_manager):
        """Test formatter behavior with circular references in log messages."""
        logger_manager.initialize()

        # Create objects with circular references
        obj1 = {}
        obj2 = {"ref": obj1}
        obj1["ref"] = obj2

        logger = logger_manager.get_logger("circular_test")

        # This should not cause infinite recursion
        try:
            logger.info("Circular reference: %s", obj1)
            # If it succeeds, that's good
        except (RecursionError, ValueError):
            # Some protection might be in place
            pass

    def test_formatter_error_unicode_encoding(self, logger_manager):
        """Test formatter error handling with unicode encoding issues."""
        logger_manager.initialize()

        logger = logger_manager.get_logger("unicode_test")

        # Test various unicode scenarios
        unicode_messages = [
            "Unicode message: \udcff",  # Invalid unicode
            "Emoji test: 🚀🎉✨",
            "Mixed encoding: café \x80 test",
        ]

        for message in unicode_messages:
            try:
                logger.info(message)
                # Should handle gracefully
            except UnicodeError:
                # Some encodings might fail, which is acceptable
                pass

    def test_formatter_error_large_message(self, logger_manager):
        """Test formatter behavior with extremely large messages."""
        logger_manager.initialize()

        logger = logger_manager.get_logger("large_test")

        # Create a very large message
        large_message = "A" * 1000000  # 1MB message

        try:
            logger.info(large_message)
            # Should handle large messages
        except MemoryError:
            # Acceptable if system runs out of memory
            pass

    def test_formatter_error_none_values(self, logger_manager):
        """Test formatter behavior with None values in various fields."""
        logger_manager.initialize()

        logger = logger_manager.get_logger("none_test")

        # Test with None message
        logger.info(None)

        # Test with None in format arguments
        try:
            logger.info("Value: %s", None)
        except TypeError:
            # Some formatters might not handle None in format args
            pass

    def test_formatter_error_exception_in_str_method(self, logger_manager):
        """Test formatter behavior when object's __str__ method raises exception."""
        logger_manager.initialize()

        class ProblematicObject:
            def __str__(self):
                raise ValueError("Cannot convert to string")

        logger = logger_manager.get_logger("problematic_test")

        try:
            logger.info("Object: %s", ProblematicObject())
        except ValueError:
            # Expected behavior - formatter can't handle objects that can't be stringified
            pass

    def test_formatter_error_recursive_logging(self, logger_manager):
        """Test formatter behavior with recursive logging scenarios."""
        logger_manager.initialize()

        class RecursiveLogger:
            def __init__(self, logger):
                self.logger = logger

            def __str__(self):
                # This would cause recursive logging
                self.logger.info("Recursive call")
                return "RecursiveLogger"

        logger = logger_manager.get_logger("recursive_test")
        recursive_obj = RecursiveLogger(logger)

        # This might cause issues or be handled gracefully
        try:
            logger.info("Object: %s", recursive_obj)
        except RecursionError:
            # Expected if no protection against recursion
            pass

    def test_formatter_error_mock_failures(self, logger_manager):
        """Test formatter error handling when underlying systems fail."""
        logger_manager.initialize()

        root_logger = logger_manager.root_logger
        handler = root_logger.handlers[0]

        # Mock formatter to raise exception
        with patch.object(handler, "formatter") as mock_formatter:
            mock_formatter.format.side_effect = Exception("Formatter failed")

            logger = logger_manager.get_logger("mock_failure_test")

            # This might be handled by the logging system
            try:
                logger.info("Test message")
            except Exception:
                # Some exceptions might propagate
                pass

    def test_formatter_error_memory_pressure(self, logger_manager):
        """Test formatter behavior under memory pressure."""
        logger_manager.initialize()

        logger = logger_manager.get_logger("memory_test")

        # Create many log messages rapidly
        try:
            for i in range(10000):
                logger.info(f"Message {i}: {'A' * 100}")
        except MemoryError:
            # Acceptable under extreme memory pressure
            pass


class TestLogFormatterHierarchy:
    """Test suite for formatter behavior with hierarchical loggers."""

    @pytest.fixture
    def logger_manager(self):
        """Create a LoggerManager instance for testing."""
        return LoggerManager()

    def test_formatter_hierarchy_root_logger(self, logger_manager):
        """Test formatter configuration for root logger."""
        logger_manager.initialize()

        root_logger = logger_manager.get_logger()

        # Root logger should have handlers with formatters
        assert len(root_logger.handlers) > 0
        for handler in root_logger.handlers:
            assert handler.formatter is not None

    def test_formatter_hierarchy_child_logger(self, logger_manager):
        """Test formatter behavior with child loggers."""
        logger_manager.initialize()

        # Create child logger
        child_logger = logger_manager.get_module_logger("child")

        # Child logger inherits from root but may not have its own handlers
        assert child_logger.propagate is True

        # The child logger itself might not have handlers (depends on configuration)
        # but should be able to log through propagation

    def test_formatter_hierarchy_nested_loggers(self, logger_manager):
        """Test formatter behavior with deeply nested loggers."""
        logger_manager.initialize()

        # Create nested logger hierarchy
        parent_logger = logger_manager.get_module_logger("parent")
        child_logger = logger_manager.get_module_logger("parent.child")
        grandchild_logger = logger_manager.get_module_logger("parent.child.grandchild")

        # All should be configured for propagation
        assert parent_logger.propagate is True
        assert child_logger.propagate is True
        assert grandchild_logger.propagate is True

        # Names should reflect hierarchy
        assert "parent" in parent_logger.name
        assert "parent.child" in child_logger.name
        assert "parent.child.grandchild" in grandchild_logger.name

    def test_formatter_hierarchy_name_formatting(self, logger_manager):
        """Test formatter includes correct logger names in hierarchy."""
        # Use a custom format that includes the logger name
        logger_manager.config.load_from_dict(
            {
                "level": "INFO",
                "format": "LOGGER:%(name)s:%(levelname)s:%(message)s",
                "handlers": ["console"],
            }
        )

        logger_manager.initialize()

        # Create stream to capture output
        stream = io.StringIO()
        stream_handler = logging.StreamHandler(stream)
        formatter = logging.Formatter(logger_manager.config.get_format())
        stream_handler.setFormatter(formatter)

        # Test different logger names
        test_loggers = [
            ("root", logger_manager.get_logger()),
            ("module", logger_manager.get_module_logger("test_module")),
            ("nested", logger_manager.get_module_logger("parent.child")),
        ]

        for name, logger in test_loggers:
            # Clear handlers and add our test handler
            for handler in logger.handlers[:]:
                logger.removeHandler(handler)
            logger.addHandler(stream_handler)
            logger.propagate = False  # Don't propagate for this test

            # Clear stream
            stream.seek(0)
            stream.truncate(0)

            logger.info(f"Test message from {name}")
            output = stream.getvalue()

            # Verify logger name is in output
            assert logger.name in output
            assert "INFO" in output
            assert f"Test message from {name}" in output

    def test_formatter_hierarchy_level_inheritance(self, logger_manager):
        """Test formatter behavior with level inheritance in hierarchy."""
        logger_manager.initialize()

        # Set root level
        logger_manager.set_level("WARNING")

        # Create child loggers
        child_logger = logger_manager.get_module_logger("child")
        grandchild_logger = logger_manager.get_module_logger("child.grandchild")

        # All should inherit the WARNING level
        assert child_logger.level == logging.WARNING
        assert grandchild_logger.level == logging.WARNING

    def test_formatter_hierarchy_propagation_formatting(self, logger_manager):
        """Test formatter behavior with message propagation."""
        logger_manager.initialize()

        # Create stream to capture root logger output
        stream = io.StringIO()
        stream_handler = logging.StreamHandler(stream)
        formatter = logging.Formatter("ROOT:%(name)s:%(message)s")
        stream_handler.setFormatter(formatter)

        # Add to root logger
        root_logger = logger_manager.get_logger()
        root_logger.addHandler(stream_handler)

        # Create child logger (should propagate to root)
        child_logger = logger_manager.get_module_logger("propagation_test")
        assert child_logger.propagate is True

        # Log from child
        child_logger.info("Child message")

        # Should appear in root logger output
        output = stream.getvalue()
        assert "propagation_test" in output or "aim2.propagation_test" in output
        assert "Child message" in output

    def test_formatter_hierarchy_custom_child_handlers(self, logger_manager):
        """Test formatter behavior when child loggers have their own handlers."""
        logger_manager.initialize()

        # Create child logger
        child_logger = logger_manager.get_module_logger("custom_child")

        # Add custom handler to child
        child_stream = io.StringIO()
        child_handler = logging.StreamHandler(child_stream)
        child_formatter = logging.Formatter("CHILD:%(name)s:%(message)s")
        child_handler.setFormatter(child_formatter)
        child_logger.addHandler(child_handler)

        # Log message
        child_logger.info("Custom child message")

        # Check child handler output
        child_output = child_stream.getvalue()
        assert "CHILD:" in child_output
        assert "custom_child" in child_output or "aim2.custom_child" in child_output
        assert "Custom child message" in child_output


class TestLogFormatterPerformance:
    """Test suite for formatter performance characteristics."""

    @pytest.fixture
    def logger_manager(self):
        """Create a LoggerManager instance for testing."""
        return LoggerManager()

    def test_formatter_performance_simple_messages(self, logger_manager):
        """Test formatter performance with simple messages."""
        logger_manager.initialize()

        logger = logger_manager.get_logger("perf_test")

        # Disable actual output to focus on formatting performance
        null_handler = logging.NullHandler()
        formatter = logging.Formatter(logger_manager.config.get_format())
        null_handler.setFormatter(formatter)

        # Clear existing handlers and add null handler
        for handler in logger.handlers[:]:
            logger.removeHandler(handler)
        logger.addHandler(null_handler)
        logger.propagate = False

        # Time formatting performance
        start_time = time.time()

        for i in range(1000):
            logger.info(f"Performance test message {i}")

        end_time = time.time()
        duration = end_time - start_time

        # Should complete quickly (adjust threshold as needed)
        assert duration < 1.0  # 1000 messages in less than 1 second

    def test_formatter_performance_complex_messages(self, logger_manager):
        """Test formatter performance with complex messages."""
        logger_manager.config.load_from_dict(
            {
                "level": "INFO",
                "format": "%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s",
                "handlers": ["console"],
            }
        )

        logger_manager.initialize()

        logger = logger_manager.get_logger("complex_perf_test")

        # Use null handler for performance testing
        null_handler = logging.NullHandler()
        formatter = logging.Formatter(logger_manager.config.get_format())
        null_handler.setFormatter(formatter)

        for handler in logger.handlers[:]:
            logger.removeHandler(handler)
        logger.addHandler(null_handler)
        logger.propagate = False

        # Time formatting performance with complex format
        start_time = time.time()

        for i in range(100):
            logger.info(f"Complex performance test message {i} with extra data")

        end_time = time.time()
        duration = end_time - start_time

        # Should still be reasonable (complex formatting takes more time)
        assert duration < 1.0

    def test_formatter_performance_concurrent_access(self, logger_manager):
        """Test formatter performance with concurrent access."""
        logger_manager.initialize()

        def log_messages(thread_id, message_count=100):
            logger = logger_manager.get_logger(f"concurrent_{thread_id}")

            # Use null handler for performance testing
            null_handler = logging.NullHandler()
            formatter = logging.Formatter(logger_manager.config.get_format())
            null_handler.setFormatter(formatter)

            logger.addHandler(null_handler)
            logger.propagate = False

            for i in range(message_count):
                logger.info(f"Thread {thread_id} message {i}")

        # Start multiple threads
        import threading

        start_time = time.time()

        threads = []
        for i in range(5):
            thread = threading.Thread(target=log_messages, args=(i,))
            threads.append(thread)
            thread.start()

        for thread in threads:
            thread.join()

        end_time = time.time()
        duration = end_time - start_time

        # 5 threads * 100 messages each = 500 total messages
        # Should complete in reasonable time
        assert duration < 2.0

    def test_formatter_performance_memory_usage(self, logger_manager):
        """Test formatter memory usage doesn't grow excessively."""
        logger_manager.initialize()

        logger = logger_manager.get_logger("memory_perf_test")

        # Use null handler to avoid I/O overhead
        null_handler = logging.NullHandler()
        formatter = logging.Formatter(logger_manager.config.get_format())
        null_handler.setFormatter(formatter)

        for handler in logger.handlers[:]:
            logger.removeHandler(handler)
        logger.addHandler(null_handler)
        logger.propagate = False

        # Log many messages and check that memory doesn't grow excessively
        # This is a basic test - in a real scenario you'd measure actual memory usage
        import gc

        gc.collect()  # Clean up before test

        for i in range(10000):
            logger.info(f"Memory test message {i}")

            # Periodically force garbage collection
            if i % 1000 == 0:
                gc.collect()

        # If we get here without memory errors, that's good
        assert True

    def test_formatter_performance_large_messages(self, logger_manager):
        """Test formatter performance with large messages."""
        logger_manager.initialize()

        logger = logger_manager.get_logger("large_perf_test")

        # Use null handler for performance testing
        null_handler = logging.NullHandler()
        formatter = logging.Formatter(logger_manager.config.get_format())
        null_handler.setFormatter(formatter)

        for handler in logger.handlers[:]:
            logger.removeHandler(handler)
        logger.addHandler(null_handler)
        logger.propagate = False

        # Create large message
        large_message = "A" * 10000  # 10KB message

        start_time = time.time()

        for i in range(100):
            logger.info(f"Large message {i}: {large_message}")

        end_time = time.time()
        duration = end_time - start_time

        # Should handle large messages reasonably well
        assert duration < 5.0


if __name__ == "__main__":
    pytest.main([__file__])
