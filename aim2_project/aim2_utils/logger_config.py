"""
Logger Configuration Module for AIM2 Project

This module provides comprehensive logging configuration management including:
- Loading logging configuration from dict/YAML
- Validating logging configuration parameters
- Support for environment variable overrides
- Handle size format parsing (e.g., "10MB" → bytes)
- Configuration merging and updates
- Integration with ConfigManager validation system

Classes:
    LoggerConfigError: Custom exception for logger configuration-related errors
    LoggerConfig: Main logger configuration management class

Dependencies:
    - os: For environment variable access
    - re: For size format parsing
    - logging: For logging level validation
    - typing: For type hints
    - pathlib: For file path operations
"""

import os
import re
import logging
from typing import Any, Dict, List, Optional
from pathlib import Path


class LoggerConfigError(Exception):
    """
    Custom exception for logger configuration-related errors.

    This exception is raised when logger configuration loading, validation,
    or processing encounters errors.

    Args:
        message (str): Error message describing the issue
        cause (Exception, optional): Original exception that caused this error
    """

    def __init__(self, message: str, cause: Exception = None):
        super().__init__(message)
        self.message = message
        self.cause = cause
        if cause:
            self.__cause__ = cause


class LoggerConfig:
    """
    Logger configuration management class.

    Provides comprehensive logging configuration management including loading from
    various sources, validation, environment variable overrides, and size format
    parsing. Integrates with the existing AIM2 configuration system.

    Attributes:
        config (dict): The current logging configuration dictionary
        env_prefix (str): Prefix for environment variable overrides
        default_config (dict): Default logging configuration values
        size_units (dict): Mapping of size unit suffixes to byte multipliers
    """

    # Default configuration values
    DEFAULT_CONFIG = {
        "level": "INFO",
        "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        "handlers": ["console"],
        "file_path": None,
        "max_file_size": "10MB",
        "backup_count": 3,
        "enable_colors": True,
        "color_scheme": "default",
        "force_colors": False,
    }

    # Valid logging levels
    VALID_LEVELS = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]

    # Valid handler types
    VALID_HANDLERS = ["console", "file"]

    # Valid color schemes
    VALID_COLOR_SCHEMES = ["default", "bright", "minimal"]

    # Size unit multipliers (case-insensitive)
    SIZE_UNITS = {
        "B": 1,
        "KB": 1024,
        "MB": 1024 * 1024,
        "GB": 1024 * 1024 * 1024,
        "TB": 1024 * 1024 * 1024 * 1024,
    }

    def __init__(self, env_prefix: str = "AIM2"):
        """
        Initialize the LoggerConfig.

        Args:
            env_prefix (str): Prefix for environment variable overrides (default: "AIM2")
        """
        self.config = self.DEFAULT_CONFIG.copy()
        self.env_prefix = env_prefix.rstrip("_")

    def load_from_dict(self, config_dict: Dict[str, Any]) -> None:
        """
        Load logging configuration from a dictionary.

        Args:
            config_dict (Dict[str, Any]): Configuration dictionary

        Raises:
            LoggerConfigError: If configuration is invalid
        """
        if not isinstance(config_dict, dict):
            raise LoggerConfigError("Configuration must be a dictionary")

        try:
            # Extract logging configuration from nested structure if needed
            logging_config = config_dict.get("logging", config_dict)

            # Validate the configuration
            self.validate_config(logging_config)

            # Merge with defaults
            self.config = self._merge_configs(self.DEFAULT_CONFIG, logging_config)

            # Apply environment variable overrides
            self._apply_env_overrides()

            # Parse size formats
            self._parse_size_formats()

        except Exception as e:
            if isinstance(e, LoggerConfigError):
                raise
            raise LoggerConfigError(
                f"Failed to load configuration from dict: {str(e)}", e
            )

    def load_from_config_manager(self, config_manager) -> None:
        """
        Load logging configuration from a ConfigManager instance.

        Args:
            config_manager: ConfigManager instance with loaded configuration

        Raises:
            LoggerConfigError: If configuration loading fails
        """
        try:
            logging_config = config_manager.get("logging", {})
            self.load_from_dict({"logging": logging_config})
        except Exception as e:
            if isinstance(e, LoggerConfigError):
                raise
            raise LoggerConfigError(
                f"Failed to load configuration from ConfigManager: {str(e)}", e
            )

    def validate_config(self, config: Dict[str, Any]) -> bool:
        """
        Validate logging configuration.

        Args:
            config (Dict[str, Any]): Configuration to validate

        Returns:
            bool: True if configuration is valid

        Raises:
            LoggerConfigError: If configuration is invalid
        """
        if not isinstance(config, dict):
            raise LoggerConfigError("Logging configuration must be a dictionary")

        errors = []

        # Validate level
        level = config.get("level", self.DEFAULT_CONFIG["level"])
        if not isinstance(level, str):
            errors.append("Logging level must be a string")
        elif level.upper() not in self.VALID_LEVELS:
            errors.append(
                f"Invalid logging level '{level}'. Must be one of: {', '.join(self.VALID_LEVELS)}"
            )

        # Validate format
        format_str = config.get("format", self.DEFAULT_CONFIG["format"])
        if not isinstance(format_str, str):
            errors.append("Logging format must be a string")
        elif not format_str.strip():
            errors.append("Logging format cannot be empty")

        # Validate handlers
        handlers = config.get("handlers", self.DEFAULT_CONFIG["handlers"])
        if not isinstance(handlers, list):
            errors.append("Handlers must be a list")
        else:
            for handler in handlers:
                if not isinstance(handler, str):
                    errors.append(f"Handler '{handler}' must be a string")
                elif handler not in self.VALID_HANDLERS:
                    errors.append(
                        f"Invalid handler '{handler}'. Must be one of: {', '.join(self.VALID_HANDLERS)}"
                    )

        # Validate file_path (if provided)
        file_path = config.get("file_path")
        if file_path is not None:
            if not isinstance(file_path, str):
                errors.append("File path must be a string or null")
            elif not file_path.strip():
                errors.append("File path cannot be empty string")
            else:
                # Check if directory exists or can be created
                try:
                    path = Path(file_path)
                    parent_dir = path.parent
                    if not parent_dir.exists():
                        # Try to create the directory
                        parent_dir.mkdir(parents=True, exist_ok=True)
                except Exception as e:
                    errors.append(f"Invalid file path '{file_path}': {str(e)}")

        # Validate max_file_size
        max_file_size = config.get(
            "max_file_size", self.DEFAULT_CONFIG["max_file_size"]
        )
        if isinstance(max_file_size, str):
            try:
                self._parse_size_string(max_file_size)
            except ValueError as e:
                errors.append(f"Invalid max_file_size format: {str(e)}")
        elif isinstance(max_file_size, int):
            if max_file_size < 1024:
                errors.append("max_file_size must be at least 1024 bytes")
        else:
            errors.append(
                "max_file_size must be a string (e.g., '10MB') or integer (bytes)"
            )

        # Validate backup_count
        backup_count = config.get("backup_count", self.DEFAULT_CONFIG["backup_count"])
        if not isinstance(backup_count, int):
            errors.append("backup_count must be an integer")
        elif backup_count < 0:
            errors.append("backup_count must be non-negative")
        elif backup_count > 100:
            errors.append("backup_count cannot exceed 100")

        # Validate enable_colors
        enable_colors = config.get(
            "enable_colors", self.DEFAULT_CONFIG["enable_colors"]
        )
        if not isinstance(enable_colors, bool):
            errors.append("enable_colors must be a boolean")

        # Validate color_scheme
        color_scheme = config.get("color_scheme", self.DEFAULT_CONFIG["color_scheme"])
        if not isinstance(color_scheme, str):
            errors.append("color_scheme must be a string")
        elif color_scheme not in self.VALID_COLOR_SCHEMES:
            errors.append(
                f"Invalid color scheme '{color_scheme}'. Must be one of: {', '.join(self.VALID_COLOR_SCHEMES)}"
            )

        # Validate force_colors
        force_colors = config.get("force_colors", self.DEFAULT_CONFIG["force_colors"])
        if not isinstance(force_colors, bool):
            errors.append("force_colors must be a boolean")

        # Check for file handler requirements
        if "file" in handlers:
            if file_path is None:
                errors.append("file_path must be specified when using file handler")

        if errors:
            error_msg = "Logging configuration validation failed:\n" + "\n".join(
                f"  - {err}" for err in errors
            )
            raise LoggerConfigError(error_msg)

        return True

    def get_level(self) -> str:
        """
        Get the logging level.

        Returns:
            str: Logging level (uppercase)
        """
        return self.config["level"].upper()

    def get_level_int(self) -> int:
        """
        Get the logging level as an integer value.

        Returns:
            int: Logging level integer value
        """
        return getattr(logging, self.get_level())

    def get_format(self) -> str:
        """
        Get the logging format string.

        Returns:
            str: Logging format string
        """
        return self.config["format"]

    def get_handlers(self) -> List[str]:
        """
        Get the list of enabled handlers.

        Returns:
            List[str]: List of handler names
        """
        return self.config["handlers"].copy()

    def get_file_path(self) -> Optional[str]:
        """
        Get the log file path.

        Returns:
            Optional[str]: Log file path or None if not configured
        """
        return self.config["file_path"]

    def get_max_file_size_bytes(self) -> int:
        """
        Get the maximum file size in bytes.

        Returns:
            int: Maximum file size in bytes
        """
        max_size = self.config["max_file_size"]
        if isinstance(max_size, int):
            return max_size
        return self._parse_size_string(max_size)

    def get_backup_count(self) -> int:
        """
        Get the backup count for rotating file handler.

        Returns:
            int: Number of backup files to keep
        """
        return self.config["backup_count"]

    def has_console_handler(self) -> bool:
        """
        Check if console handler is enabled.

        Returns:
            bool: True if console handler is enabled
        """
        return "console" in self.config["handlers"]

    def has_file_handler(self) -> bool:
        """
        Check if file handler is enabled.

        Returns:
            bool: True if file handler is enabled
        """
        return (
            "file" in self.config["handlers"] and self.config["file_path"] is not None
        )

    def get_enable_colors(self) -> bool:
        """
        Get whether colors are enabled for console output.

        Returns:
            bool: True if colors are enabled
        """
        return self.config["enable_colors"]

    def get_color_scheme(self) -> str:
        """
        Get the color scheme for console output.

        Returns:
            str: Color scheme name
        """
        return self.config["color_scheme"]

    def get_force_colors(self) -> bool:
        """
        Get whether colors should be forced even if terminal doesn't support them.

        Returns:
            bool: True if colors should be forced
        """
        return self.config["force_colors"]

    def update_config(self, updates: Dict[str, Any]) -> None:
        """
        Update the logging configuration with new values.

        Args:
            updates (Dict[str, Any]): Configuration updates to apply

        Raises:
            LoggerConfigError: If updated configuration is invalid
        """
        # Create a temporary config with updates applied
        temp_config = self.config.copy()
        temp_config.update(updates)

        # Validate the updated configuration
        self.validate_config(temp_config)

        # Apply the updates
        self.config.update(updates)
        self._parse_size_formats()

    def set_level(self, level: str) -> None:
        """
        Set the logging level.

        Args:
            level (str): Logging level to set

        Raises:
            LoggerConfigError: If level is invalid
        """
        if level.upper() not in self.VALID_LEVELS:
            raise LoggerConfigError(
                f"Invalid logging level '{level}'. Must be one of: {', '.join(self.VALID_LEVELS)}"
            )
        self.config["level"] = level.upper()

    def set_file_path(self, file_path: Optional[str]) -> None:
        """
        Set the log file path.

        Args:
            file_path (Optional[str]): Log file path or None to disable file logging

        Raises:
            LoggerConfigError: If file path is invalid
        """
        if file_path is not None:
            if not isinstance(file_path, str) or not file_path.strip():
                raise LoggerConfigError("File path must be a non-empty string or None")

            # Validate path
            try:
                path = Path(file_path)
                parent_dir = path.parent
                if not parent_dir.exists():
                    parent_dir.mkdir(parents=True, exist_ok=True)
            except Exception as e:
                raise LoggerConfigError(f"Invalid file path '{file_path}': {str(e)}", e)

        self.config["file_path"] = file_path

    def set_enable_colors(self, enable_colors: bool) -> None:
        """
        Set whether colors are enabled for console output.

        Args:
            enable_colors (bool): Whether to enable colors

        Raises:
            LoggerConfigError: If enable_colors is not a boolean
        """
        if not isinstance(enable_colors, bool):
            raise LoggerConfigError("enable_colors must be a boolean")
        self.config["enable_colors"] = enable_colors

    def set_color_scheme(self, color_scheme: str) -> None:
        """
        Set the color scheme for console output.

        Args:
            color_scheme (str): Color scheme name to set

        Raises:
            LoggerConfigError: If color scheme is invalid
        """
        if not isinstance(color_scheme, str):
            raise LoggerConfigError("color_scheme must be a string")
        if color_scheme not in self.VALID_COLOR_SCHEMES:
            raise LoggerConfigError(
                f"Invalid color scheme '{color_scheme}'. Must be one of: {', '.join(self.VALID_COLOR_SCHEMES)}"
            )
        self.config["color_scheme"] = color_scheme

    def set_force_colors(self, force_colors: bool) -> None:
        """
        Set whether colors should be forced even if terminal doesn't support them.

        Args:
            force_colors (bool): Whether to force colors

        Raises:
            LoggerConfigError: If force_colors is not a boolean
        """
        if not isinstance(force_colors, bool):
            raise LoggerConfigError("force_colors must be a boolean")
        self.config["force_colors"] = force_colors

    def add_handler(self, handler: str) -> None:
        """
        Add a handler to the configuration.

        Args:
            handler (str): Handler name to add

        Raises:
            LoggerConfigError: If handler is invalid or already exists
        """
        if handler not in self.VALID_HANDLERS:
            raise LoggerConfigError(
                f"Invalid handler '{handler}'. Must be one of: {', '.join(self.VALID_HANDLERS)}"
            )

        if handler not in self.config["handlers"]:
            self.config["handlers"].append(handler)

    def remove_handler(self, handler: str) -> None:
        """
        Remove a handler from the configuration.

        Args:
            handler (str): Handler name to remove
        """
        if handler in self.config["handlers"]:
            self.config["handlers"].remove(handler)

    def to_dict(self) -> Dict[str, Any]:
        """
        Export the configuration as a dictionary.

        Returns:
            Dict[str, Any]: Configuration dictionary
        """
        return self.config.copy()

    def reset_to_defaults(self) -> None:
        """Reset the configuration to default values."""
        self.config = self.DEFAULT_CONFIG.copy()

    # Private helper methods

    def _merge_configs(
        self, base: Dict[str, Any], override: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Merge two configuration dictionaries.

        Args:
            base (Dict[str, Any]): Base configuration
            override (Dict[str, Any]): Override configuration

        Returns:
            Dict[str, Any]: Merged configuration
        """
        merged = base.copy()
        for key, value in override.items():
            if (
                key in merged
                and isinstance(merged[key], dict)
                and isinstance(value, dict)
            ):
                merged[key] = self._merge_configs(merged[key], value)
            else:
                merged[key] = value
        return merged

    def _apply_env_overrides(self) -> None:
        """Apply environment variable overrides to logging configuration."""
        env_prefix = f"{self.env_prefix}_LOGGING_"

        # Map environment variable suffixes to config keys
        env_mappings = {
            "LEVEL": "level",
            "FORMAT": "format",
            "FILE_PATH": "file_path",
            "MAX_FILE_SIZE": "max_file_size",
            "BACKUP_COUNT": "backup_count",
            "ENABLE_COLORS": "enable_colors",
            "COLOR_SCHEME": "color_scheme",
            "FORCE_COLORS": "force_colors",
        }

        for env_suffix, config_key in env_mappings.items():
            env_key = f"{env_prefix}{env_suffix}"
            if env_key in os.environ:
                env_value = os.environ[env_key]

                # Convert value to appropriate type
                if config_key in ["backup_count"]:
                    try:
                        self.config[config_key] = int(env_value)
                    except ValueError:
                        raise LoggerConfigError(
                            f"Invalid integer value for {env_key}: {env_value}"
                        )
                elif config_key in ["enable_colors", "force_colors"]:
                    # Convert string to boolean
                    bool_value = env_value.lower() in ["true", "1", "yes", "on"]
                    self.config[config_key] = bool_value
                elif config_key == "file_path" and env_value.lower() in [
                    "null",
                    "none",
                    "",
                ]:
                    self.config[config_key] = None
                else:
                    self.config[config_key] = env_value

        # Handle handlers (comma-separated list)
        handlers_env = f"{env_prefix}HANDLERS"
        if handlers_env in os.environ:
            handlers_str = os.environ[handlers_env]
            if handlers_str:
                handlers = [h.strip() for h in handlers_str.split(",") if h.strip()]
                self.config["handlers"] = handlers

    def _parse_size_formats(self) -> None:
        """Parse size format strings in the configuration."""
        max_size = self.config.get("max_file_size")
        if isinstance(max_size, str):
            try:
                # Keep the original string format but validate it
                self._parse_size_string(max_size)
            except ValueError as e:
                raise LoggerConfigError(f"Invalid max_file_size format: {str(e)}", e)

    def _parse_size_string(self, size_str: str) -> int:
        """
        Parse a size string (e.g., "10MB") to bytes.

        Args:
            size_str (str): Size string to parse

        Returns:
            int: Size in bytes

        Raises:
            ValueError: If size string format is invalid
        """
        if not isinstance(size_str, str):
            raise ValueError("Size must be a string")

        size_str = size_str.strip().upper()

        # Pattern to match number + optional unit
        pattern = r"^(\d+(?:\.\d+)?)\s*([A-Z]*B?)$"
        match = re.match(pattern, size_str)

        if not match:
            raise ValueError(
                f"Invalid size format: {size_str}. Expected format like '10MB', '1.5GB', or '1024'"
            )

        number_str, unit = match.groups()

        try:
            number = float(number_str)
        except ValueError:
            raise ValueError(f"Invalid number in size string: {number_str}")

        if number <= 0:
            raise ValueError("Size must be positive")

        # Default to bytes if no unit specified
        if not unit or unit == "B":
            multiplier = 1
        else:
            if unit not in self.SIZE_UNITS:
                valid_units = ", ".join(self.SIZE_UNITS.keys())
                raise ValueError(
                    f"Invalid size unit: {unit}. Valid units are: {valid_units}"
                )
            multiplier = self.SIZE_UNITS[unit]

        result = int(number * multiplier)

        # Minimum size check
        if result < 1024:
            raise ValueError("Minimum file size is 1024 bytes (1KB)")

        return result
