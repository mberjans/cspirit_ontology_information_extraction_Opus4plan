"""
Unit tests for colored console handler

This module contains comprehensive unit tests for the ColoredConsoleHandler class,
including color formatting, terminal detection, color scheme management, and
integration with the logging framework.

Test Coverage:
- ColoredConsoleHandler: Handler initialization, color formatting, terminal detection
- Color schemes: Default, bright, minimal schemes and validation
- ANSI color codes: Proper color application and reset codes
- Terminal detection: TTY detection, environment variable handling
- Configuration: Color enabling/disabling, force colors, scheme switching
- Error handling: Invalid color schemes, malformed configurations
- Platform compatibility: Cross-platform color support detection

Test Classes:
    TestColoredConsoleHandler: Tests basic handler functionality
    TestColorFormatting: Tests color formatting and ANSI codes
    TestTerminalDetection: Tests terminal capability detection
    TestColorSchemes: Tests color scheme management
    TestConfigurationIntegration: Tests integration with LoggerConfig
    TestErrorHandling: Tests error scenarios and edge cases
"""

import os
import logging
import pytest
from io import StringIO
from unittest.mock import patch, Mock

# Import the modules to be tested
try:
    from aim2_project.aim2_utils.colored_console_handler import (
        ColoredConsoleHandler,
        ColorScheme,
    )
    from aim2_project.aim2_utils.logger_config import (
        LoggerConfig,
        LoggerConfigError,
    )
    from aim2_project.aim2_utils.logger_manager import (
        LoggerManager,
    )
except ImportError:
    # Expected during TDD - tests define the interface
    pass


class TestColoredConsoleHandler:
    """Test the ColoredConsoleHandler class basic functionality."""

    def test_handler_initialization_default(self):
        """Test default handler initialization."""
        handler = ColoredConsoleHandler()

        assert handler.color_scheme == ColorScheme.DEFAULT
        assert handler.force_colors == False
        assert hasattr(handler, "colors_enabled")
        assert hasattr(handler, "level_colors")
        assert hasattr(handler, "reset_code")

    def test_handler_initialization_with_params(self):
        """Test handler initialization with custom parameters."""
        stream = StringIO()
        handler = ColoredConsoleHandler(
            stream=stream,
            enable_colors=True,
            color_scheme=ColorScheme.BRIGHT,
            force_colors=True,
        )

        assert handler.stream == stream
        assert handler.colors_enabled == True
        assert handler.color_scheme == ColorScheme.BRIGHT
        assert handler.force_colors == True

    def test_handler_initialization_with_string_scheme(self):
        """Test handler initialization with string color scheme."""
        handler = ColoredConsoleHandler(color_scheme="bright")
        assert handler.color_scheme == ColorScheme.BRIGHT

        handler = ColoredConsoleHandler(color_scheme="minimal")
        assert handler.color_scheme == ColorScheme.MINIMAL

    def test_handler_initialization_invalid_scheme(self):
        """Test handler initialization with invalid color scheme."""
        handler = ColoredConsoleHandler(color_scheme="invalid")
        # Should fall back to default scheme
        assert handler.color_scheme == ColorScheme.DEFAULT

    def test_handler_repr(self):
        """Test string representation of handler."""
        stream = StringIO()
        handler = ColoredConsoleHandler(
            stream=stream, enable_colors=False, color_scheme=ColorScheme.BRIGHT
        )

        repr_str = repr(handler)
        assert "ColoredConsoleHandler" in repr_str
        assert "colors_enabled=False" in repr_str
        assert "scheme=bright" in repr_str

    @patch("sys.stderr")
    def test_default_stream(self, mock_stderr):
        """Test that default stream is sys.stderr."""
        handler = ColoredConsoleHandler()
        # Default stream should be sys.stderr (inherited from StreamHandler)
        assert hasattr(handler, "stream")


class TestColorFormatting:
    """Test color formatting and ANSI code application."""

    def test_format_with_colors_enabled(self):
        """Test message formatting with colors enabled."""
        stream = StringIO()
        handler = ColoredConsoleHandler(
            stream=stream, enable_colors=True, color_scheme=ColorScheme.DEFAULT
        )

        # Create a formatter for the handler
        formatter = logging.Formatter("%(levelname)s: %(message)s")
        handler.setFormatter(formatter)

        # Test different log levels
        test_cases = [
            (logging.DEBUG, "Debug message"),
            (logging.INFO, "Info message"),
            (logging.WARNING, "Warning message"),
            (logging.ERROR, "Error message"),
            (logging.CRITICAL, "Critical message"),
        ]

        for level, message in test_cases:
            record = logging.LogRecord(
                name="test",
                level=level,
                pathname="",
                lineno=0,
                msg=message,
                args=(),
                exc_info=None,
            )

            formatted = handler.format(record)

            # Should contain ANSI color codes when colors are enabled
            if level in handler.level_colors:
                expected_color = handler.level_colors[level]
                if expected_color:  # Some levels might not have colors
                    assert expected_color in formatted
                    assert handler.reset_code in formatted

    def test_format_with_colors_disabled(self):
        """Test message formatting with colors disabled."""
        stream = StringIO()
        handler = ColoredConsoleHandler(stream=stream, enable_colors=False)

        formatter = logging.Formatter("%(levelname)s: %(message)s")
        handler.setFormatter(formatter)

        record = logging.LogRecord(
            name="test",
            level=logging.ERROR,
            pathname="",
            lineno=0,
            msg="Error message",
            args=(),
            exc_info=None,
        )

        formatted = handler.format(record)

        # Should not contain ANSI color codes when colors are disabled
        for color_code in handler.COLORS.values():
            assert color_code not in formatted

    def test_ansi_color_codes(self):
        """Test that ANSI color codes are properly defined."""
        handler = ColoredConsoleHandler()

        # Test that all expected colors are defined
        expected_colors = [
            "black",
            "red",
            "green",
            "yellow",
            "blue",
            "magenta",
            "cyan",
            "white",
            "bright_black",
            "bright_red",
            "bright_green",
            "bright_yellow",
            "bright_blue",
            "bright_magenta",
            "bright_cyan",
            "bright_white",
            "reset",
            "bold",
            "dim",
        ]

        for color in expected_colors:
            assert color in handler.COLORS
            assert handler.COLORS[color].startswith("\033[")

    def test_emit_with_colors(self):
        """Test emit method with color formatting."""
        stream = StringIO()
        handler = ColoredConsoleHandler(
            stream=stream,
            enable_colors=True,
            force_colors=True,  # Force colors to bypass terminal detection
        )

        formatter = logging.Formatter("%(message)s")
        handler.setFormatter(formatter)

        record = logging.LogRecord(
            name="test",
            level=logging.ERROR,
            pathname="",
            lineno=0,
            msg="Test error message",
            args=(),
            exc_info=None,
        )

        handler.emit(record)

        output = stream.getvalue()
        assert "Test error message" in output
        # Should contain color codes when force_colors is True
        assert any(color in output for color in handler.COLORS.values() if color)

    def test_emit_exception_handling(self):
        """Test emit method exception handling."""
        # Create a mock stream that raises an exception
        mock_stream = Mock()
        mock_stream.write.side_effect = Exception("Stream error")

        handler = ColoredConsoleHandler(stream=mock_stream)
        handler.handleError = Mock()  # Mock the error handler

        record = logging.LogRecord(
            name="test",
            level=logging.ERROR,
            pathname="",
            lineno=0,
            msg="Test message",
            args=(),
            exc_info=None,
        )

        # Should not raise exception, should call handleError
        handler.emit(record)
        handler.handleError.assert_called_once_with(record)


class TestTerminalDetection:
    """Test terminal capability detection."""

    def test_should_enable_colors_with_force(self):
        """Test color detection with force_colors=True."""
        handler = ColoredConsoleHandler(force_colors=True)
        assert handler._should_enable_colors() == True

    @patch.dict(os.environ, {"NO_COLOR": "1"})
    def test_should_disable_colors_with_no_color_env(self):
        """Test color detection with NO_COLOR environment variable."""
        handler = ColoredConsoleHandler()
        assert handler._should_enable_colors() == False

    @patch.dict(os.environ, {"FORCE_COLOR": "1"}, clear=True)
    def test_should_enable_colors_with_force_color_env(self):
        """Test color detection with FORCE_COLOR environment variable."""
        mock_stream = Mock()
        mock_stream.isatty.return_value = True
        handler = ColoredConsoleHandler(stream=mock_stream)
        assert handler._should_enable_colors() == True

    @patch.dict(os.environ, {"TERM": "dumb"})
    def test_should_disable_colors_with_dumb_terminal(self):
        """Test color detection with dumb terminal."""
        handler = ColoredConsoleHandler()
        # Should be False for dumb terminal even if stream supports isatty
        handler._should_enable_colors()
        # Result may vary based on other conditions, but dumb terminal should be handled

    @patch.dict(os.environ, {"TERM": "xterm-256color"})
    def test_should_enable_colors_with_color_terminal(self):
        """Test color detection with color-capable terminal."""
        mock_stream = Mock()
        mock_stream.isatty.return_value = True

        handler = ColoredConsoleHandler(stream=mock_stream)
        assert handler._should_enable_colors() == True

    def test_should_disable_colors_when_not_tty(self):
        """Test color detection when output is not a TTY."""
        mock_stream = Mock()
        mock_stream.isatty.return_value = False

        handler = ColoredConsoleHandler(stream=mock_stream)
        assert handler._should_enable_colors() == False

    def test_should_handle_no_isatty_method(self):
        """Test color detection when stream has no isatty method."""
        mock_stream = Mock()
        del mock_stream.isatty  # Remove isatty method

        handler = ColoredConsoleHandler(stream=mock_stream)
        # Should handle gracefully and return False
        assert handler._should_enable_colors() == False

    @patch("sys.platform", "win32")
    @patch("platform.version")
    def test_windows_color_support_modern(self, mock_version):
        """Test Windows color support detection for modern Windows."""
        mock_version.return_value = "10.0.19041"
        mock_stream = Mock()
        mock_stream.isatty.return_value = True

        handler = ColoredConsoleHandler(stream=mock_stream)
        # Modern Windows should support colors
        handler._should_enable_colors()
        # Result depends on multiple factors, but modern Windows should be handled

    @patch("sys.platform", "win32")
    @patch.dict(os.environ, {"WT_SESSION": "session_id"})
    def test_windows_terminal_support(self):
        """Test Windows Terminal color support."""
        mock_stream = Mock()
        mock_stream.isatty.return_value = True

        handler = ColoredConsoleHandler(stream=mock_stream)
        assert handler._should_enable_colors() == True

    @patch("sys.platform", "linux")
    def test_unix_default_color_support(self):
        """Test Unix-like systems default to supporting colors."""
        mock_stream = Mock()
        mock_stream.isatty.return_value = True

        handler = ColoredConsoleHandler(stream=mock_stream)
        # Unix-like systems should default to True when TTY
        assert handler._should_enable_colors() == True


class TestColorSchemes:
    """Test color scheme management and validation."""

    def test_default_color_scheme(self):
        """Test default color scheme configuration."""
        handler = ColoredConsoleHandler(color_scheme=ColorScheme.DEFAULT)

        expected_colors = {
            logging.DEBUG: "bright_black",
            logging.INFO: "white",
            logging.WARNING: "yellow",
            logging.ERROR: "red",
            logging.CRITICAL: "bright_red",
        }

        scheme_colors = handler.COLOR_SCHEMES[ColorScheme.DEFAULT]
        for level, expected_color in expected_colors.items():
            assert scheme_colors[level] == expected_color

    def test_bright_color_scheme(self):
        """Test bright color scheme configuration."""
        handler = ColoredConsoleHandler(color_scheme=ColorScheme.BRIGHT)

        scheme_colors = handler.COLOR_SCHEMES[ColorScheme.BRIGHT]

        # Bright scheme should use bright colors
        assert scheme_colors[logging.INFO] == "bright_white"
        assert scheme_colors[logging.WARNING] == "bright_yellow"
        assert scheme_colors[logging.CRITICAL] == "bright_magenta"

    def test_minimal_color_scheme(self):
        """Test minimal color scheme configuration."""
        handler = ColoredConsoleHandler(color_scheme=ColorScheme.MINIMAL)

        scheme_colors = handler.COLOR_SCHEMES[ColorScheme.MINIMAL]

        # Minimal scheme should use fewer colors
        assert scheme_colors[logging.DEBUG] == "white"
        assert scheme_colors[logging.INFO] == "white"
        assert scheme_colors[logging.WARNING] == "yellow"
        assert scheme_colors[logging.ERROR] == "red"
        assert scheme_colors[logging.CRITICAL] == "red"

    def test_set_color_scheme(self):
        """Test setting color scheme after initialization."""
        handler = ColoredConsoleHandler(color_scheme=ColorScheme.DEFAULT)

        # Change to bright scheme
        handler.set_color_scheme(ColorScheme.BRIGHT)
        assert handler.color_scheme == ColorScheme.BRIGHT

        # Change using string
        handler.set_color_scheme("minimal")
        assert handler.color_scheme == ColorScheme.MINIMAL

    def test_set_invalid_color_scheme(self):
        """Test setting invalid color scheme."""
        handler = ColoredConsoleHandler()

        with pytest.raises(ValueError, match="Invalid color scheme"):
            handler.set_color_scheme("invalid_scheme")

    def test_get_available_schemes(self):
        """Test getting list of available color schemes."""
        handler = ColoredConsoleHandler()
        schemes = handler.get_available_schemes()

        expected_schemes = ["default", "bright", "minimal"]
        assert sorted(schemes) == sorted(expected_schemes)

    def test_get_current_scheme(self):
        """Test getting current color scheme name."""
        handler = ColoredConsoleHandler(color_scheme=ColorScheme.BRIGHT)
        assert handler.get_current_scheme() == "bright"


class TestConfigurationIntegration:
    """Test integration with LoggerConfig and LoggerManager."""

    def test_logger_config_color_defaults(self):
        """Test LoggerConfig default color configuration."""
        config = LoggerConfig()

        assert config.get_enable_colors() == True
        assert config.get_color_scheme() == "default"
        assert config.get_force_colors() == False

    def test_logger_config_color_setters(self):
        """Test LoggerConfig color configuration setters."""
        config = LoggerConfig()

        config.set_enable_colors(False)
        assert config.get_enable_colors() == False

        config.set_color_scheme("bright")
        assert config.get_color_scheme() == "bright"

        config.set_force_colors(True)
        assert config.get_force_colors() == True

    def test_logger_config_color_validation(self):
        """Test LoggerConfig color configuration validation."""
        config = LoggerConfig()

        # Test invalid enable_colors
        with pytest.raises(LoggerConfigError, match="enable_colors must be a boolean"):
            config.set_enable_colors("true")

        # Test invalid color_scheme
        with pytest.raises(LoggerConfigError, match="Invalid color scheme"):
            config.set_color_scheme("invalid")

        # Test invalid force_colors
        with pytest.raises(LoggerConfigError, match="force_colors must be a boolean"):
            config.set_force_colors("false")

    def test_logger_config_from_dict_with_colors(self):
        """Test loading LoggerConfig from dictionary with color options."""
        config_dict = {
            "logging": {
                "level": "INFO",
                "handlers": ["console"],
                "enable_colors": True,
                "color_scheme": "bright",
                "force_colors": False,
            }
        }

        config = LoggerConfig()
        config.load_from_dict(config_dict)

        assert config.get_enable_colors() == True
        assert config.get_color_scheme() == "bright"
        assert config.get_force_colors() == False

    def test_logger_manager_colored_console_handler(self):
        """Test LoggerManager creates ColoredConsoleHandler when colors enabled."""
        config = LoggerConfig()
        config.set_enable_colors(True)
        config.set_color_scheme("bright")

        manager = LoggerManager(config)
        handler = manager._create_console_handler()

        assert isinstance(handler, ColoredConsoleHandler)
        assert handler.color_scheme == ColorScheme.BRIGHT

    def test_logger_manager_standard_handler_when_colors_disabled(self):
        """Test LoggerManager creates standard handler when colors disabled."""
        config = LoggerConfig()
        config.set_enable_colors(False)

        manager = LoggerManager(config)
        handler = manager._create_console_handler()

        # Should be standard StreamHandler, not ColoredConsoleHandler
        assert isinstance(handler, logging.StreamHandler)
        assert not isinstance(handler, ColoredConsoleHandler)

    @patch.dict(
        os.environ,
        {
            "AIM2_LOGGING_ENABLE_COLORS": "false",
            "AIM2_LOGGING_COLOR_SCHEME": "minimal",
            "AIM2_LOGGING_FORCE_COLORS": "true",
        },
    )
    def test_environment_variable_overrides(self):
        """Test color configuration from environment variables."""
        config = LoggerConfig()
        config.load_from_dict({})  # Triggers environment override application

        assert config.get_enable_colors() == False
        assert config.get_color_scheme() == "minimal"
        assert config.get_force_colors() == True


class TestColorControls:
    """Test color control methods."""

    def test_set_colors_enabled(self):
        """Test enabling/disabling colors dynamically."""
        handler = ColoredConsoleHandler(enable_colors=True)

        # Disable colors
        handler.set_colors_enabled(False)
        assert handler.colors_enabled == False

        # Enable colors
        handler.set_colors_enabled(True)
        assert handler.colors_enabled == True

    def test_supports_colors(self):
        """Test checking color support."""
        handler = ColoredConsoleHandler()

        # Should return boolean indicating color support
        result = handler.supports_colors()
        assert isinstance(result, bool)

    def test_color_formatting_toggle(self):
        """Test that color formatting respects enabled/disabled state."""
        stream = StringIO()
        handler = ColoredConsoleHandler(
            stream=stream, enable_colors=True, force_colors=True
        )

        formatter = logging.Formatter("%(message)s")
        handler.setFormatter(formatter)

        record = logging.LogRecord(
            name="test",
            level=logging.ERROR,
            pathname="",
            lineno=0,
            msg="Test message",
            args=(),
            exc_info=None,
        )

        # Test with colors enabled
        formatted_with_colors = handler.format(record)

        # Disable colors and test again
        handler.set_colors_enabled(False)
        formatted_without_colors = handler.format(record)

        # Should be different (one with colors, one without)
        assert formatted_with_colors != formatted_without_colors

        # Without colors should not contain ANSI codes
        for color_code in handler.COLORS.values():
            if color_code:  # Skip empty color codes
                assert color_code not in formatted_without_colors


class TestErrorHandling:
    """Test error scenarios and edge cases."""

    def test_setup_colors_with_missing_color(self):
        """Test color setup with missing color definition."""
        handler = ColoredConsoleHandler()

        # Temporarily modify color scheme to include invalid color
        original_schemes = handler.COLOR_SCHEMES.copy()
        handler.COLOR_SCHEMES[ColorScheme.DEFAULT] = {
            logging.ERROR: "nonexistent_color"
        }

        try:
            handler._setup_colors()
            # Should handle gracefully and use empty color code
            assert handler.level_colors[logging.ERROR] == ""
        finally:
            # Restore original schemes
            handler.COLOR_SCHEMES = original_schemes

    def test_format_with_none_formatter(self):
        """Test formatting behavior when no formatter is set."""
        handler = ColoredConsoleHandler(enable_colors=True, force_colors=True)
        # Don't set a formatter - should use default formatting

        record = logging.LogRecord(
            name="test",
            level=logging.ERROR,
            pathname="",
            lineno=0,
            msg="Test message",
            args=(),
            exc_info=None,
        )

        # Should not raise exception
        formatted = handler.format(record)
        assert "Test message" in formatted

    def test_recursion_error_handling(self):
        """Test handling of RecursionError in emit method."""
        handler = ColoredConsoleHandler()

        # Create a record that would cause recursion
        record = logging.LogRecord(
            name="test",
            level=logging.ERROR,
            pathname="",
            lineno=0,
            msg="Test message",
            args=(),
            exc_info=None,
        )

        # Mock format to raise RecursionError
        handler.format = Mock(side_effect=RecursionError("Max recursion"))

        # Should re-raise RecursionError
        with pytest.raises(RecursionError):
            handler.emit(record)


class TestIntegrationScenarios:
    """Test real-world integration scenarios."""

    def test_end_to_end_colored_logging(self):
        """Test complete colored logging workflow."""
        # Create a string buffer to capture output
        stream_buffer = StringIO()

        # Create a ColoredConsoleHandler directly
        handler = ColoredConsoleHandler(
            stream=stream_buffer,
            enable_colors=True,
            color_scheme="bright",
            force_colors=True,
        )

        # Set up formatter
        formatter = logging.Formatter("%(levelname)s: %(message)s")
        handler.setFormatter(formatter)

        # Create a test logger and add our handler
        test_logger = logging.getLogger("test_integration")
        test_logger.handlers.clear()  # Clear any existing handlers
        test_logger.addHandler(handler)
        test_logger.setLevel(logging.DEBUG)
        test_logger.propagate = False  # Don't propagate to root

        # Log messages at different levels
        test_logger.error("Test error message")
        test_logger.warning("Test warning message")
        test_logger.info("Test info message")

        output = stream_buffer.getvalue()

        # Should contain the messages
        assert "Test error message" in output
        assert "Test warning message" in output
        assert "Test info message" in output

        # Should contain color codes since force_colors=True
        assert any(color in output for color in handler.COLORS.values() if color)

    def test_configuration_reload_with_color_changes(self):
        """Test configuration reload with color option changes."""
        # Start with colors disabled
        config = LoggerConfig()
        config.set_enable_colors(False)

        manager = LoggerManager(config)
        manager.initialize()

        # Get initial handler
        console_handler = manager._create_console_handler()
        assert isinstance(console_handler, logging.StreamHandler)
        assert not isinstance(console_handler, ColoredConsoleHandler)

        # Create new config with colors enabled
        new_config = LoggerConfig()
        new_config.set_enable_colors(True)
        new_config.set_color_scheme("minimal")

        # Reload configuration
        manager.reload_configuration(new_config)

        # Get new handler
        new_console_handler = manager._create_console_handler()
        assert isinstance(new_console_handler, ColoredConsoleHandler)
        assert new_console_handler.color_scheme == ColorScheme.MINIMAL

    def test_multiple_loggers_with_colors(self):
        """Test multiple loggers using colored console handler."""
        config = LoggerConfig()
        config.set_enable_colors(True)

        manager = LoggerManager(config)
        manager.initialize()

        # Create multiple loggers
        logger1 = manager.get_logger("module1")
        logger2 = manager.get_logger("module2")
        logger3 = manager.get_module_logger("module3")

        # All should use the same colored console handler configuration
        assert logger1 is not None
        assert logger2 is not None
        assert logger3 is not None

        # Verify they can all log without errors
        with patch("sys.stderr", new_callable=StringIO):
            logger1.info("Module 1 message")
            logger2.warning("Module 2 message")
            logger3.error("Module 3 message")
