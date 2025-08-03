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
        "formatter_type": "standard",
        "handlers": ["console"],
        "file_path": None,
        "max_file_size": "10MB",
        "backup_count": 3,
        "rotation_type": "size",
        "time_interval": "daily",
        "time_when": "midnight",
        "enable_colors": True,
        "color_scheme": "default",
        "force_colors": False,
        # JSON formatter specific configuration
        "json_fields": ["timestamp", "level", "logger_name", "message"],
        "json_pretty_print": False,
        "json_custom_fields": {},
        "json_timestamp_format": "iso",
        "json_use_utc": False,
        "json_include_exception_traceback": True,
        "json_max_message_length": None,
        "json_field_mapping": {},
        "json_ensure_ascii": False,
        # Context injection configuration
        "enable_context_injection": True,
        "context_injection_method": "filter",
        "auto_generate_request_id": True,
        "request_id_field": "request_id",
        "context_fields": [
            "request_id",
            "user_id",
            "session_id",
            "correlation_id",
            "trace_id",
            "operation",
            "component",
            "service",
            "version",
            "environment",
        ],
        "include_context_in_json": True,
        "context_prefix": "",
        "default_context": {},
        # Module-specific log level configuration
        "module_levels": {},
        # Performance logging configuration
        "enable_performance_logging": True,
        "performance_thresholds": {
            "warning_seconds": 1.0,
            "critical_seconds": 5.0,
            "memory_warning_mb": 100,
            "memory_critical_mb": 500,
        },
        "performance_profiles": {
            "fast": {"warning_seconds": 0.1, "critical_seconds": 0.5},
            "normal": {"warning_seconds": 1.0, "critical_seconds": 5.0},
            "slow": {"warning_seconds": 10.0, "critical_seconds": 30.0},
        },
        "performance_metrics": {
            "include_memory_usage": False,
            "include_cpu_usage": False,
            "include_function_args": False,
            "include_function_result": False,
            "max_arg_length": 200,
            "max_result_length": 200,
        },
        "performance_logging_level": {
            "success": "INFO",
            "warning": "WARNING",
            "critical": "CRITICAL",
            "error": "ERROR",
        },
        "performance_context_prefix": "perf_",
        "performance_operation_field": "operation",
        "performance_auto_operation_naming": True,
    }

    # Valid logging levels
    VALID_LEVELS = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]

    # Valid handler types
    VALID_HANDLERS = ["console", "file"]

    # Valid formatter types
    VALID_FORMATTER_TYPES = ["standard", "json"]

    # Valid color schemes
    VALID_COLOR_SCHEMES = ["default", "bright", "minimal"]

    # Valid rotation types
    VALID_ROTATION_TYPES = ["size", "time", "both"]

    # Valid time intervals
    VALID_TIME_INTERVALS = ["hourly", "daily", "weekly", "midnight"]

    # Valid context injection methods
    VALID_CONTEXT_INJECTION_METHODS = ["filter", "replace", "disabled"]

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

        # Validate formatter_type
        formatter_type = config.get(
            "formatter_type", self.DEFAULT_CONFIG["formatter_type"]
        )
        if not isinstance(formatter_type, str):
            errors.append("Formatter type must be a string")
        elif formatter_type not in self.VALID_FORMATTER_TYPES:
            errors.append(
                f"Invalid formatter type '{formatter_type}'. Must be one of: {', '.join(self.VALID_FORMATTER_TYPES)}"
            )

        # Validate JSON formatter specific configuration if JSON formatter is selected
        if formatter_type == "json":
            self._validate_json_formatter_config(config, errors)

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

        # Validate rotation_type
        rotation_type = config.get(
            "rotation_type", self.DEFAULT_CONFIG["rotation_type"]
        )
        if not isinstance(rotation_type, str):
            errors.append("rotation_type must be a string")
        elif rotation_type not in self.VALID_ROTATION_TYPES:
            errors.append(
                f"Invalid rotation_type '{rotation_type}'. Must be one of: {', '.join(self.VALID_ROTATION_TYPES)}"
            )

        # Validate time_interval
        time_interval = config.get(
            "time_interval", self.DEFAULT_CONFIG["time_interval"]
        )
        if not isinstance(time_interval, str):
            errors.append("time_interval must be a string")
        elif time_interval not in self.VALID_TIME_INTERVALS:
            errors.append(
                f"Invalid time_interval '{time_interval}'. Must be one of: {', '.join(self.VALID_TIME_INTERVALS)}"
            )

        # Validate time_when
        time_when = config.get("time_when", self.DEFAULT_CONFIG["time_when"])
        if not isinstance(time_when, str):
            errors.append("time_when must be a string")
        elif time_interval in ["daily", "midnight"] and time_when != "midnight":
            # Validate HH:MM format for daily rotation
            try:
                hour, minute = map(int, time_when.split(":"))
                if not (0 <= hour <= 23 and 0 <= minute <= 59):
                    raise ValueError()
            except (ValueError, IndexError):
                errors.append(
                    f"Invalid time_when '{time_when}'. Must be 'midnight' or 'HH:MM' format (e.g., '02:30')"
                )

        # Check for file handler requirements
        if "file" in handlers:
            if file_path is None:
                errors.append("file_path must be specified when using file handler")

        # Validate context injection configuration
        self._validate_context_injection_config(config, errors)

        # Validate module-specific levels configuration
        self._validate_module_levels_config(config, errors)

        # Validate performance logging configuration
        self._validate_performance_config(config, errors)

        if errors:
            error_msg = "Logging configuration validation failed:\n" + "\n".join(
                f"  - {err}" for err in errors
            )
            raise LoggerConfigError(error_msg)

        return True

    def _validate_json_formatter_config(
        self, config: Dict[str, Any], errors: List[str]
    ) -> None:
        """
        Validate JSON formatter specific configuration.

        Args:
            config: Configuration dictionary to validate
            errors: List to append errors to
        """
        # Validate json_fields
        json_fields = config.get("json_fields", self.DEFAULT_CONFIG["json_fields"])
        if not isinstance(json_fields, list):
            errors.append("json_fields must be a list")
        else:
            valid_fields = [
                "timestamp",
                "level",
                "level_number",
                "logger_name",
                "module",
                "function",
                "line_number",
                "message",
                "pathname",
                "filename",
                "thread",
                "thread_name",
                "process",
                "process_name",
                "exception",
                "stack_info",
                "extra",
                "custom_fields",
            ]
            for field in json_fields:
                if not isinstance(field, str):
                    errors.append(f"JSON field '{field}' must be a string")
                elif field not in valid_fields:
                    errors.append(
                        f"Invalid JSON field '{field}'. Valid fields: {', '.join(valid_fields)}"
                    )

        # Validate json_pretty_print
        json_pretty_print = config.get(
            "json_pretty_print", self.DEFAULT_CONFIG["json_pretty_print"]
        )
        if not isinstance(json_pretty_print, bool):
            errors.append("json_pretty_print must be a boolean")

        # Validate json_custom_fields
        json_custom_fields = config.get(
            "json_custom_fields", self.DEFAULT_CONFIG["json_custom_fields"]
        )
        if not isinstance(json_custom_fields, dict):
            errors.append("json_custom_fields must be a dictionary")

        # Validate json_timestamp_format
        json_timestamp_format = config.get(
            "json_timestamp_format", self.DEFAULT_CONFIG["json_timestamp_format"]
        )
        if not isinstance(json_timestamp_format, str):
            errors.append("json_timestamp_format must be a string")
        elif json_timestamp_format not in [
            "iso",
            "epoch",
        ] and not json_timestamp_format.startswith("%"):
            errors.append(
                "json_timestamp_format must be 'iso', 'epoch', or a valid strftime format"
            )

        # Validate json_use_utc
        json_use_utc = config.get("json_use_utc", self.DEFAULT_CONFIG["json_use_utc"])
        if not isinstance(json_use_utc, bool):
            errors.append("json_use_utc must be a boolean")

        # Validate json_include_exception_traceback
        json_include_exception_traceback = config.get(
            "json_include_exception_traceback",
            self.DEFAULT_CONFIG["json_include_exception_traceback"],
        )
        if not isinstance(json_include_exception_traceback, bool):
            errors.append("json_include_exception_traceback must be a boolean")

        # Validate json_max_message_length
        json_max_message_length = config.get(
            "json_max_message_length", self.DEFAULT_CONFIG["json_max_message_length"]
        )
        if json_max_message_length is not None:
            if (
                not isinstance(json_max_message_length, int)
                or json_max_message_length < 1
            ):
                errors.append(
                    "json_max_message_length must be a positive integer or None"
                )

        # Validate json_field_mapping
        json_field_mapping = config.get(
            "json_field_mapping", self.DEFAULT_CONFIG["json_field_mapping"]
        )
        if not isinstance(json_field_mapping, dict):
            errors.append("json_field_mapping must be a dictionary")

        # Validate json_ensure_ascii
        json_ensure_ascii = config.get(
            "json_ensure_ascii", self.DEFAULT_CONFIG["json_ensure_ascii"]
        )
        if not isinstance(json_ensure_ascii, bool):
            errors.append("json_ensure_ascii must be a boolean")

    def _validate_context_injection_config(
        self, config: Dict[str, Any], errors: List[str]
    ) -> None:
        """
        Validate context injection specific configuration.

        Args:
            config: Configuration dictionary to validate
            errors: List to append errors to
        """
        # Validate enable_context_injection
        enable_context_injection = config.get(
            "enable_context_injection", self.DEFAULT_CONFIG["enable_context_injection"]
        )
        if not isinstance(enable_context_injection, bool):
            errors.append("enable_context_injection must be a boolean")

        # Validate context_injection_method
        context_injection_method = config.get(
            "context_injection_method", self.DEFAULT_CONFIG["context_injection_method"]
        )
        if not isinstance(context_injection_method, str):
            errors.append("context_injection_method must be a string")
        elif context_injection_method not in self.VALID_CONTEXT_INJECTION_METHODS:
            errors.append(
                f"Invalid context_injection_method '{context_injection_method}'. "
                f"Must be one of: {', '.join(self.VALID_CONTEXT_INJECTION_METHODS)}"
            )

        # Validate auto_generate_request_id
        auto_generate_request_id = config.get(
            "auto_generate_request_id", self.DEFAULT_CONFIG["auto_generate_request_id"]
        )
        if not isinstance(auto_generate_request_id, bool):
            errors.append("auto_generate_request_id must be a boolean")

        # Validate request_id_field
        request_id_field = config.get(
            "request_id_field", self.DEFAULT_CONFIG["request_id_field"]
        )
        if not isinstance(request_id_field, str):
            errors.append("request_id_field must be a string")
        elif not request_id_field.strip():
            errors.append("request_id_field cannot be empty")

        # Validate context_fields
        context_fields = config.get(
            "context_fields", self.DEFAULT_CONFIG["context_fields"]
        )
        if not isinstance(context_fields, list):
            errors.append("context_fields must be a list")
        else:
            for field in context_fields:
                if not isinstance(field, str):
                    errors.append(f"Context field '{field}' must be a string")
                elif not field.strip():
                    errors.append("Context fields cannot be empty strings")

        # Validate include_context_in_json
        include_context_in_json = config.get(
            "include_context_in_json", self.DEFAULT_CONFIG["include_context_in_json"]
        )
        if not isinstance(include_context_in_json, bool):
            errors.append("include_context_in_json must be a boolean")

        # Validate context_prefix
        context_prefix = config.get(
            "context_prefix", self.DEFAULT_CONFIG["context_prefix"]
        )
        if not isinstance(context_prefix, str):
            errors.append("context_prefix must be a string")

        # Validate default_context
        default_context = config.get(
            "default_context", self.DEFAULT_CONFIG["default_context"]
        )
        if not isinstance(default_context, dict):
            errors.append("default_context must be a dictionary")

    def _validate_module_levels_config(
        self, config: Dict[str, Any], errors: List[str]
    ) -> None:
        """
        Validate module-specific levels configuration.

        Args:
            config: Configuration dictionary to validate
            errors: List to append errors to
        """
        # Validate module_levels
        module_levels = config.get(
            "module_levels", self.DEFAULT_CONFIG["module_levels"]
        )
        if not isinstance(module_levels, dict):
            errors.append("module_levels must be a dictionary")
        else:
            for module_name, level in module_levels.items():
                # Validate module name
                if not isinstance(module_name, str):
                    errors.append(f"Module name '{module_name}' must be a string")
                elif not module_name.strip():
                    errors.append("Module names cannot be empty strings")
                elif not self._is_valid_module_name(module_name):
                    errors.append(
                        f"Invalid module name '{module_name}'. Must follow hierarchical pattern (e.g., 'aim2.ontology', 'aim2.extraction.pipeline')"
                    )

                # Validate level
                if not isinstance(level, str):
                    errors.append(f"Level for module '{module_name}' must be a string")
                elif level.upper() not in self.VALID_LEVELS:
                    errors.append(
                        f"Invalid level '{level}' for module '{module_name}'. Must be one of: {', '.join(self.VALID_LEVELS)}"
                    )

    def _is_valid_module_name(self, module_name: str) -> bool:
        """
        Check if a module name follows the valid hierarchical pattern.

        Args:
            module_name: Module name to validate

        Returns:
            bool: True if module name is valid
        """
        # Allow hierarchical names like "aim2.ontology", "aim2.extraction.pipeline"
        # Module names should contain only alphanumeric characters, dots, and underscores
        import re

        pattern = r"^[a-zA-Z][a-zA-Z0-9_]*(?:\.[a-zA-Z][a-zA-Z0-9_]*)*$"
        return bool(re.match(pattern, module_name))

    def _validate_performance_config(
        self, config: Dict[str, Any], errors: List[str]
    ) -> None:
        """
        Validate performance logging configuration.

        Args:
            config: Configuration dictionary to validate
            errors: List to append errors to
        """
        # Validate enable_performance_logging
        enable_perf = config.get("enable_performance_logging", True)
        if not isinstance(enable_perf, bool):
            errors.append("enable_performance_logging must be a boolean")

        # Validate performance_thresholds
        thresholds = config.get("performance_thresholds", {})
        if not isinstance(thresholds, dict):
            errors.append("performance_thresholds must be a dictionary")
        else:
            self._validate_thresholds(thresholds, errors)

        # Validate performance_profiles
        profiles = config.get("performance_profiles", {})
        if not isinstance(profiles, dict):
            errors.append("performance_profiles must be a dictionary")
        else:
            self._validate_profiles(profiles, errors)

        # Validate performance_metrics
        metrics_config = config.get("performance_metrics", {})
        if not isinstance(metrics_config, dict):
            errors.append("performance_metrics must be a dictionary")
        else:
            self._validate_metrics_config(metrics_config, errors)

        # Validate performance_logging_level
        logging_levels = config.get("performance_logging_level", {})
        if not isinstance(logging_levels, dict):
            errors.append("performance_logging_level must be a dictionary")
        else:
            for level_type, level_name in logging_levels.items():
                if level_type not in ["success", "warning", "critical", "error"]:
                    errors.append(
                        f"Invalid performance logging level type: {level_type}"
                    )
                if level_name not in self.VALID_LEVELS:
                    errors.append(
                        f"Invalid performance logging level '{level_name}' for {level_type}"
                    )

        # Validate performance_context_prefix
        context_prefix = config.get("performance_context_prefix", "perf_")
        if not isinstance(context_prefix, str):
            errors.append("performance_context_prefix must be a string")

        # Validate performance_auto_operation_naming
        auto_naming = config.get("performance_auto_operation_naming", True)
        if not isinstance(auto_naming, bool):
            errors.append("performance_auto_operation_naming must be a boolean")

    def _validate_thresholds(
        self, thresholds: Dict[str, Any], errors: List[str]
    ) -> None:
        """Validate threshold configuration."""
        for key, value in thresholds.items():
            if key.endswith("_seconds") or key.endswith("_mb"):
                if not isinstance(value, (int, float)) or value < 0:
                    errors.append(f"Threshold '{key}' must be a non-negative number")

        # Check logical relationships
        if (
            "warning_seconds" in thresholds
            and "critical_seconds" in thresholds
            and thresholds["warning_seconds"] >= thresholds["critical_seconds"]
        ):
            errors.append("warning_seconds must be less than critical_seconds")

    def _validate_profiles(self, profiles: Dict[str, Any], errors: List[str]) -> None:
        """Validate performance profile configuration."""
        for profile_name, profile_config in profiles.items():
            if not isinstance(profile_config, dict):
                errors.append(f"Profile '{profile_name}' must be a dictionary")
                continue

            # Validate each profile's thresholds
            temp_errors = []
            self._validate_thresholds(profile_config, temp_errors)
            for error in temp_errors:
                errors.append(f"Profile '{profile_name}': {error}")

    def _validate_metrics_config(
        self, metrics_config: Dict[str, Any], errors: List[str]
    ) -> None:
        """Validate metrics configuration."""
        # Boolean fields
        bool_fields = [
            "include_memory_usage",
            "include_cpu_usage",
            "include_function_args",
            "include_function_result",
        ]
        for field in bool_fields:
            if field in metrics_config and not isinstance(metrics_config[field], bool):
                errors.append(f"Metrics config '{field}' must be a boolean")

        # Integer fields
        int_fields = ["max_arg_length", "max_result_length"]
        for field in int_fields:
            if field in metrics_config:
                value = metrics_config[field]
                if not isinstance(value, int) or value < 1:
                    errors.append(
                        f"Metrics config '{field}' must be a positive integer"
                    )

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

    def get_rotation_type(self) -> str:
        """
        Get the rotation type for file handlers.

        Returns:
            str: Rotation type ("size", "time", or "both")
        """
        return self.config["rotation_type"]

    def get_time_interval(self) -> str:
        """
        Get the time interval for time-based rotation.

        Returns:
            str: Time interval ("hourly", "daily", "weekly", or "midnight")
        """
        return self.config["time_interval"]

    def get_time_when(self) -> str:
        """
        Get when to rotate for daily rotation.

        Returns:
            str: Time specification ("midnight" or "HH:MM" format)
        """
        return self.config["time_when"]

    def get_formatter_type(self) -> str:
        """
        Get the formatter type.

        Returns:
            str: Formatter type ("standard" or "json")
        """
        return self.config["formatter_type"]

    def get_json_fields(self) -> List[str]:
        """
        Get the JSON formatter fields.

        Returns:
            List[str]: List of fields to include in JSON output
        """
        return self.config["json_fields"].copy()

    def get_json_pretty_print(self) -> bool:
        """
        Get whether JSON output should be pretty-printed.

        Returns:
            bool: True if JSON should be pretty-printed
        """
        return self.config["json_pretty_print"]

    def get_json_custom_fields(self) -> Dict[str, Any]:
        """
        Get the JSON formatter custom fields.

        Returns:
            Dict[str, Any]: Custom fields to include in JSON output
        """
        return self.config["json_custom_fields"].copy()

    def get_json_timestamp_format(self) -> str:
        """
        Get the JSON formatter timestamp format.

        Returns:
            str: Timestamp format ("iso", "epoch", or strftime format)
        """
        return self.config["json_timestamp_format"]

    def get_json_use_utc(self) -> bool:
        """
        Get whether JSON formatter should use UTC timestamps.

        Returns:
            bool: True if UTC timestamps should be used
        """
        return self.config["json_use_utc"]

    def get_json_include_exception_traceback(self) -> bool:
        """
        Get whether JSON formatter should include exception tracebacks.

        Returns:
            bool: True if exception tracebacks should be included
        """
        return self.config["json_include_exception_traceback"]

    def get_json_max_message_length(self) -> Optional[int]:
        """
        Get the JSON formatter maximum message length.

        Returns:
            Optional[int]: Maximum message length or None for no limit
        """
        return self.config["json_max_message_length"]

    def get_json_field_mapping(self) -> Dict[str, str]:
        """
        Get the JSON formatter field mapping.

        Returns:
            Dict[str, str]: Field name mapping
        """
        return self.config["json_field_mapping"].copy()

    def get_json_ensure_ascii(self) -> bool:
        """
        Get whether JSON formatter should ensure ASCII output.

        Returns:
            bool: True if ASCII-only output should be ensured
        """
        return self.config["json_ensure_ascii"]

    def get_enable_context_injection(self) -> bool:
        """
        Get whether context injection is enabled.

        Returns:
            bool: True if context injection is enabled
        """
        return self.config["enable_context_injection"]

    def get_context_injection_method(self) -> str:
        """
        Get the context injection method.

        Returns:
            str: Context injection method ("filter", "replace", or "disabled")
        """
        return self.config["context_injection_method"]

    def get_auto_generate_request_id(self) -> bool:
        """
        Get whether request IDs should be auto-generated.

        Returns:
            bool: True if request IDs should be auto-generated
        """
        return self.config["auto_generate_request_id"]

    def get_request_id_field(self) -> str:
        """
        Get the field name for request IDs in context.

        Returns:
            str: Request ID field name
        """
        return self.config["request_id_field"]

    def get_context_fields(self) -> List[str]:
        """
        Get the list of allowed context fields.

        Returns:
            List[str]: List of allowed context field names
        """
        return self.config["context_fields"].copy()

    def get_include_context_in_json(self) -> bool:
        """
        Get whether context should be included in JSON output.

        Returns:
            bool: True if context should be included in JSON
        """
        return self.config["include_context_in_json"]

    def get_context_prefix(self) -> str:
        """
        Get the prefix for context fields.

        Returns:
            str: Context field prefix
        """
        return self.config["context_prefix"]

    def get_default_context(self) -> Dict[str, Any]:
        """
        Get the default context values.

        Returns:
            Dict[str, Any]: Default context values
        """
        return self.config["default_context"].copy()

    def get_module_levels(self) -> Dict[str, str]:
        """
        Get the module-specific log levels.

        Returns:
            Dict[str, str]: Dictionary mapping module names to log levels
        """
        return self.config["module_levels"].copy()

    def get_module_level(self, module_name: str) -> Optional[str]:
        """
        Get the log level for a specific module.

        Args:
            module_name (str): Name of the module

        Returns:
            Optional[str]: Log level for the module or None if not configured
        """
        return self.config["module_levels"].get(module_name)

    def get_effective_level(self, module_name: str) -> str:
        """
        Get the effective log level for a module, considering hierarchy.

        This method checks for module-specific levels, starting with the full
        module name and working up the hierarchy. If no specific level is found,
        returns the global level.

        Args:
            module_name (str): Name of the module

        Returns:
            str: Effective log level for the module
        """
        # Check for exact match first
        if module_name in self.config["module_levels"]:
            return self.config["module_levels"][module_name].upper()

        # Check parent modules in hierarchy
        parts = module_name.split(".")
        for i in range(len(parts) - 1, 0, -1):
            parent_module = ".".join(parts[:i])
            if parent_module in self.config["module_levels"]:
                return self.config["module_levels"][parent_module].upper()

        # Fall back to global level
        return self.get_level()

    def get_enable_performance_logging(self) -> bool:
        """
        Get whether performance logging is enabled.

        Returns:
            bool: True if performance logging is enabled
        """
        return self.config.get("enable_performance_logging", True)

    def get_performance_thresholds(self) -> Dict[str, float]:
        """
        Get the performance thresholds configuration.

        Returns:
            Dict[str, float]: Performance thresholds
        """
        return self.config.get("performance_thresholds", {}).copy()

    def get_performance_profiles(self) -> Dict[str, Dict[str, float]]:
        """
        Get the performance profiles configuration.

        Returns:
            Dict[str, Dict[str, float]]: Performance profiles
        """
        return self.config.get("performance_profiles", {}).copy()

    def get_performance_metrics(self) -> Dict[str, Any]:
        """
        Get the performance metrics configuration.

        Returns:
            Dict[str, Any]: Performance metrics configuration
        """
        return self.config.get("performance_metrics", {}).copy()

    def get_performance_logging_level(self) -> Dict[str, str]:
        """
        Get the performance logging level mapping.

        Returns:
            Dict[str, str]: Performance logging level mapping
        """
        return self.config.get("performance_logging_level", {}).copy()

    def get_performance_context_prefix(self) -> str:
        """
        Get the performance context prefix.

        Returns:
            str: Performance context prefix
        """
        return self.config.get("performance_context_prefix", "perf_")

    def get_performance_auto_operation_naming(self) -> bool:
        """
        Get whether automatic operation naming is enabled.

        Returns:
            bool: True if automatic operation naming is enabled
        """
        return self.config.get("performance_auto_operation_naming", True)

    def set_module_level(self, module_name: str, level: str) -> None:
        """
        Set the log level for a specific module.

        Args:
            module_name (str): Name of the module
            level (str): Log level to set

        Raises:
            LoggerConfigError: If module name or level is invalid
        """
        if not isinstance(module_name, str) or not module_name.strip():
            raise LoggerConfigError("Module name must be a non-empty string")

        if not self._is_valid_module_name(module_name):
            raise LoggerConfigError(
                f"Invalid module name '{module_name}'. Must follow hierarchical pattern (e.g., 'aim2.ontology')"
            )

        if not isinstance(level, str) or level.upper() not in self.VALID_LEVELS:
            raise LoggerConfigError(
                f"Invalid level '{level}'. Must be one of: {', '.join(self.VALID_LEVELS)}"
            )

        self.config["module_levels"][module_name] = level.upper()

    def remove_module_level(self, module_name: str) -> bool:
        """
        Remove the specific log level configuration for a module.

        Args:
            module_name (str): Name of the module

        Returns:
            bool: True if module level was removed, False if it wasn't configured
        """
        if module_name in self.config["module_levels"]:
            del self.config["module_levels"][module_name]
            return True
        return False

    def clear_module_levels(self) -> None:
        """Clear all module-specific log level configurations."""
        self.config["module_levels"].clear()

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

    def set_rotation_type(self, rotation_type: str) -> None:
        """
        Set the rotation type for file handlers.

        Args:
            rotation_type (str): Rotation type to set

        Raises:
            LoggerConfigError: If rotation_type is invalid
        """
        if not isinstance(rotation_type, str):
            raise LoggerConfigError("rotation_type must be a string")
        if rotation_type not in self.VALID_ROTATION_TYPES:
            raise LoggerConfigError(
                f"Invalid rotation_type '{rotation_type}'. Must be one of: {', '.join(self.VALID_ROTATION_TYPES)}"
            )
        self.config["rotation_type"] = rotation_type

    def set_time_interval(self, time_interval: str) -> None:
        """
        Set the time interval for time-based rotation.

        Args:
            time_interval (str): Time interval to set

        Raises:
            LoggerConfigError: If time_interval is invalid
        """
        if not isinstance(time_interval, str):
            raise LoggerConfigError("time_interval must be a string")
        if time_interval not in self.VALID_TIME_INTERVALS:
            raise LoggerConfigError(
                f"Invalid time_interval '{time_interval}'. Must be one of: {', '.join(self.VALID_TIME_INTERVALS)}"
            )
        self.config["time_interval"] = time_interval

    def set_time_when(self, time_when: str) -> None:
        """
        Set when to rotate for daily rotation.

        Args:
            time_when (str): Time specification to set

        Raises:
            LoggerConfigError: If time_when is invalid
        """
        if not isinstance(time_when, str):
            raise LoggerConfigError("time_when must be a string")

        # Validate format if not "midnight"
        if time_when != "midnight":
            try:
                hour, minute = map(int, time_when.split(":"))
                if not (0 <= hour <= 23 and 0 <= minute <= 59):
                    raise ValueError()
            except (ValueError, IndexError):
                raise LoggerConfigError(
                    f"Invalid time_when '{time_when}'. Must be 'midnight' or 'HH:MM' format (e.g., '02:30')"
                )

        self.config["time_when"] = time_when

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
            "ROTATION_TYPE": "rotation_type",
            "TIME_INTERVAL": "time_interval",
            "TIME_WHEN": "time_when",
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

        # Handle module-specific levels
        # Format: AIM2_LOGGING_MODULE_LEVELS="aim2.ontology=DEBUG,aim2.extraction=INFO"
        module_levels_env = f"{env_prefix}MODULE_LEVELS"
        if module_levels_env in os.environ:
            module_levels_str = os.environ[module_levels_env]
            if module_levels_str:
                module_levels = {}
                try:
                    pairs = [
                        pair.strip()
                        for pair in module_levels_str.split(",")
                        if pair.strip()
                    ]
                    for pair in pairs:
                        if "=" in pair:
                            module_name, level = pair.split("=", 1)
                            module_name = module_name.strip()
                            level = level.strip().upper()
                            if module_name and level:
                                module_levels[module_name] = level

                    if module_levels:
                        self.config["module_levels"] = module_levels
                except Exception as e:
                    raise LoggerConfigError(
                        f"Invalid format for {module_levels_env}. Expected format: 'module1=LEVEL1,module2=LEVEL2'. Error: {str(e)}"
                    )

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
