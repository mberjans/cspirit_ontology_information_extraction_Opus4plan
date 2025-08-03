"""
Colored Console Handler Module for AIM2 Project

This module provides a colored console handler for the logging framework that:
- Adds ANSI color codes to log messages based on their level
- Automatically detects terminal capabilities for color support
- Provides configurable color schemes
- Maintains backward compatibility with standard console handlers
- Handles cross-platform color differences

Classes:
    ColorScheme: Enum defining available color schemes
    ColoredConsoleHandler: Enhanced StreamHandler with color support

Dependencies:
    - logging: For base handler functionality
    - sys: For stdout/stderr and platform detection
    - os: For environment variable access
    - typing: For type hints
    - enum: For color scheme enumeration
"""

import logging
import sys
import os
from typing import Optional, Union
from enum import Enum


class ColorScheme(Enum):
    """
    Available color schemes for console output.

    Each scheme defines ANSI color codes for different log levels.
    """

    DEFAULT = "default"
    BRIGHT = "bright"
    MINIMAL = "minimal"


class ColoredConsoleHandler(logging.StreamHandler):
    """
    Enhanced console handler that adds ANSI color codes to log messages.

    This handler extends the standard StreamHandler to provide colored output
    based on log levels. It automatically detects terminal capabilities and
    gracefully falls back to uncolored output when colors are not supported.

    Attributes:
        colors_enabled (bool): Whether colors are currently enabled
        color_scheme (ColorScheme): Current color scheme being used
        level_colors (Dict[int, str]): Mapping of log levels to ANSI color codes
        reset_code (str): ANSI reset code to clear formatting
    """

    # ANSI color codes
    COLORS = {
        "black": "\033[30m",
        "red": "\033[31m",
        "green": "\033[32m",
        "yellow": "\033[33m",
        "blue": "\033[34m",
        "magenta": "\033[35m",
        "cyan": "\033[36m",
        "white": "\033[37m",
        "bright_black": "\033[90m",
        "bright_red": "\033[91m",
        "bright_green": "\033[92m",
        "bright_yellow": "\033[93m",
        "bright_blue": "\033[94m",
        "bright_magenta": "\033[95m",
        "bright_cyan": "\033[96m",
        "bright_white": "\033[97m",
        "reset": "\033[0m",
        "bold": "\033[1m",
        "dim": "\033[2m",
    }

    # Color schemes
    COLOR_SCHEMES = {
        ColorScheme.DEFAULT: {
            logging.DEBUG: "bright_black",  # Gray/dim for debug
            logging.INFO: "white",  # Default/white for info
            logging.WARNING: "yellow",  # Yellow for warnings
            logging.ERROR: "red",  # Red for errors
            logging.CRITICAL: "bright_red",  # Bright red for critical
        },
        ColorScheme.BRIGHT: {
            logging.DEBUG: "cyan",
            logging.INFO: "bright_white",
            logging.WARNING: "bright_yellow",
            logging.ERROR: "bright_red",
            logging.CRITICAL: "bright_magenta",
        },
        ColorScheme.MINIMAL: {
            logging.DEBUG: "white",
            logging.INFO: "white",
            logging.WARNING: "yellow",
            logging.ERROR: "red",
            logging.CRITICAL: "red",
        },
    }

    def __init__(
        self,
        stream=None,
        enable_colors: Optional[bool] = None,
        color_scheme: Union[ColorScheme, str] = ColorScheme.DEFAULT,
        force_colors: bool = False,
    ):
        """
        Initialize the ColoredConsoleHandler.

        Args:
            stream: Output stream (defaults to sys.stderr)
            enable_colors: Whether to enable colors (auto-detected if None)
            color_scheme: Color scheme to use
            force_colors: Force colors even if terminal doesn't support them
        """
        super().__init__(stream)

        # Convert string color scheme to enum if needed
        if isinstance(color_scheme, str):
            try:
                color_scheme = ColorScheme(color_scheme.lower())
            except ValueError:
                color_scheme = ColorScheme.DEFAULT

        self.color_scheme = color_scheme
        self.force_colors = force_colors

        # Determine if colors should be enabled
        if enable_colors is None:
            self.colors_enabled = self._should_enable_colors()
        else:
            self.colors_enabled = enable_colors

        # Set up color mappings
        self.level_colors = {}
        self.reset_code = self.COLORS["reset"]
        self._setup_colors()

    def _should_enable_colors(self) -> bool:
        """
        Determine if colors should be enabled based on terminal capabilities.

        Returns:
            bool: True if colors should be enabled
        """
        # Force colors if explicitly requested
        if self.force_colors:
            return True

        # Check if output is being redirected
        if not hasattr(self.stream, "isatty") or not self.stream.isatty():
            return False

        # Check environment variables
        # NO_COLOR environment variable disables colors
        if os.environ.get("NO_COLOR"):
            return False

        # FORCE_COLOR environment variable forces colors
        if os.environ.get("FORCE_COLOR"):
            return True

        # Check TERM environment variable
        term = os.environ.get("TERM", "").lower()
        if term in ("dumb", ""):
            return False

        # Check for common terminals that support colors
        if "color" in term or term in ("xterm", "xterm-256color", "screen", "tmux"):
            return True

        # On Windows, check for modern terminal support
        if sys.platform == "win32":
            # Windows 10 version 1511 and later support ANSI colors
            try:
                import platform

                version = platform.version()
                if version:
                    # Parse version string like "10.0.19041"
                    parts = version.split(".")
                    if len(parts) >= 3:
                        major = int(parts[0])
                        int(parts[1])
                        build = int(parts[2])
                        # Windows 10 build 10586 (version 1511) added ANSI support
                        if major >= 10 and build >= 10586:
                            return True
            except (ImportError, ValueError, IndexError):
                pass

            # Check for Windows Terminal or ConEmu
            if os.environ.get("WT_SESSION") or os.environ.get("ConEmuPID"):
                return True

            return False

        # Default to True on Unix-like systems
        return True

    def _setup_colors(self) -> None:
        """Set up color mappings based on the current color scheme."""
        scheme_colors = self.COLOR_SCHEMES.get(
            self.color_scheme, self.COLOR_SCHEMES[ColorScheme.DEFAULT]
        )

        self.level_colors = {}
        for level, color_name in scheme_colors.items():
            if color_name in self.COLORS:
                self.level_colors[level] = self.COLORS[color_name]
            else:
                # Fallback to no color for unknown color names
                self.level_colors[level] = ""

    def format(self, record: logging.LogRecord) -> str:
        """
        Format the log record with colors if enabled.

        Args:
            record: The log record to format

        Returns:
            str: Formatted log message with optional colors
        """
        # Get the formatted message from the base formatter
        formatted_message = super().format(record)

        # Return uncolored message if colors are disabled
        if not self.colors_enabled:
            return formatted_message

        # Get color for this log level
        color_code = self.level_colors.get(record.levelno, "")

        # Return colored message
        if color_code:
            return f"{color_code}{formatted_message}{self.reset_code}"
        else:
            return formatted_message

    def emit(self, record: logging.LogRecord) -> None:
        """
        Emit a log record with color formatting.

        Args:
            record: The log record to emit
        """
        try:
            msg = self.format(record)
            stream = self.stream
            stream.write(msg + self.terminator)
            self.flush()
        except RecursionError:
            raise
        except Exception:
            self.handleError(record)

    def set_colors_enabled(self, enabled: bool) -> None:
        """
        Enable or disable color output.

        Args:
            enabled: Whether to enable colors
        """
        self.colors_enabled = enabled

    def set_color_scheme(self, scheme: Union[ColorScheme, str]) -> None:
        """
        Change the color scheme.

        Args:
            scheme: New color scheme to use
        """
        if isinstance(scheme, str):
            try:
                scheme = ColorScheme(scheme.lower())
            except ValueError:
                raise ValueError(
                    f"Invalid color scheme: {scheme}. Valid options: {[s.value for s in ColorScheme]}"
                )

        self.color_scheme = scheme
        self._setup_colors()

    def supports_colors(self) -> bool:
        """
        Check if the current terminal supports colors.

        Returns:
            bool: True if colors are supported
        """
        return self._should_enable_colors()

    def get_available_schemes(self) -> list:
        """
        Get list of available color schemes.

        Returns:
            list: List of available color scheme names
        """
        return [scheme.value for scheme in ColorScheme]

    def get_current_scheme(self) -> str:
        """
        Get the current color scheme name.

        Returns:
            str: Current color scheme name
        """
        return self.color_scheme.value

    def __repr__(self) -> str:
        """String representation of the handler."""
        return (
            f"<ColoredConsoleHandler "
            f"colors_enabled={self.colors_enabled} "
            f"scheme={self.color_scheme.value} "
            f"stream={self.stream}>"
        )
