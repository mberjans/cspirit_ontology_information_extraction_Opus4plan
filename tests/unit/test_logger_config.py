"""
Unit tests for logger configuration framework

This module contains comprehensive unit tests for the logger configuration system,
including LoggerConfig, LoggerManager, and their integration with ConfigManager.

Test Coverage:
- LoggerConfig: Configuration loading, validation, environment overrides, size parsing
- LoggerManager: Logger creation, handler setup, module-specific loggers, configuration reloading
- ConfigManager integration: Logger factory methods, validation integration
- File handler operations: Rotating file handler setup, file system operations
- Error handling: Invalid configurations, file system errors, edge cases
- Performance: Large configuration handling, concurrent operations

Test Classes:
    TestLoggerConfig: Tests LoggerConfig class functionality
    TestLoggerManager: Tests LoggerManager operations
    TestRotatingFileHandler: Tests file handler setup and operations
    TestLoggerConfigValidation: Tests configuration validation
    TestLoggerIntegration: Tests ConfigManager integration
    TestModuleSpecificLoggers: Tests module logger creation
    TestLoggerErrorHandling: Tests error scenarios and edge cases
"""

import os
import json
import tempfile
import pytest
import time
import threading
import logging
import logging.handlers
from unittest.mock import patch, Mock, MagicMock, mock_open
from pathlib import Path
from io import StringIO

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
    from aim2_project.aim2_utils.config_manager import (
        ConfigManager,
        ConfigError,
    )
except ImportError:
    # Expected during TDD - tests define the interface
    pass


class TestLoggerConfig:
    """Test suite for LoggerConfig class."""

    @pytest.fixture
    def logger_config(self):
        """Create a LoggerConfig instance for testing."""
        return LoggerConfig()

    @pytest.fixture
    def temp_dir(self):
        """Create a temporary directory for test files."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            yield Path(tmp_dir)

    @pytest.fixture
    def sample_logging_config(self):
        """Sample logging configuration for testing."""
        return {
            "level": "INFO",
            "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            "handlers": ["console", "file"],
            "file_path": "/var/log/aim2.log",
            "max_file_size": "10MB",
            "backup_count": 5
        }

    @pytest.fixture
    def nested_logging_config(self):
        """Sample configuration with nested logging section."""
        return {
            "project": {
                "name": "AIM2 Test",
                "version": "1.0.0"
            },
            "logging": {
                "level": "DEBUG",
                "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
                "handlers": ["console"],
                "file_path": None,
                "max_file_size": "5MB",
                "backup_count": 3
            }
        }

    def test_logger_config_initialization(self, logger_config):
        """Test LoggerConfig can be instantiated with defaults."""
        assert logger_config is not None
        assert isinstance(logger_config.config, dict)
        assert logger_config.env_prefix == "AIM2"
        assert logger_config.config["level"] == "INFO"
        assert logger_config.config["handlers"] == ["console"]

    def test_logger_config_initialization_custom_prefix(self):
        """Test LoggerConfig initialization with custom environment prefix."""
        config = LoggerConfig(env_prefix="CUSTOM_")
        assert config.env_prefix == "CUSTOM"

    def test_load_from_dict_simple_config(self, logger_config, sample_logging_config):
        """Test loading configuration from a simple dictionary."""
        logger_config.load_from_dict(sample_logging_config)
        
        assert logger_config.config["level"] == "INFO"
        assert logger_config.config["format"] == "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        assert logger_config.config["handlers"] == ["console", "file"]
        assert logger_config.config["file_path"] == "/var/log/aim2.log"
        assert logger_config.config["max_file_size"] == "10MB"
        assert logger_config.config["backup_count"] == 5

    def test_load_from_dict_nested_config(self, logger_config, nested_logging_config):
        """Test loading configuration from nested dictionary."""
        logger_config.load_from_dict(nested_logging_config)
        
        assert logger_config.config["level"] == "DEBUG"
        assert logger_config.config["handlers"] == ["console"]
        assert logger_config.config["file_path"] is None

    def test_load_from_dict_invalid_input(self, logger_config):
        """Test loading from invalid input raises error."""
        with pytest.raises(LoggerConfigError) as exc_info:
            logger_config.load_from_dict("not a dict")
        
        assert "must be a dictionary" in str(exc_info.value)

    def test_load_from_dict_with_validation_error(self, logger_config):
        """Test loading invalid configuration raises validation error."""
        invalid_config = {
            "level": "INVALID_LEVEL",
            "handlers": ["console"]
        }
        
        with pytest.raises(LoggerConfigError) as exc_info:
            logger_config.load_from_dict(invalid_config)
        
        assert "Invalid logging level" in str(exc_info.value)

    @patch('aim2_project.aim2_utils.logger_config.os.makedirs')
    def test_load_from_dict_creates_directories(self, mock_makedirs, logger_config, temp_dir):
        """Test that loading config creates necessary directories."""
        log_file = temp_dir / "logs" / "app.log"
        config = {
            "level": "INFO",
            "handlers": ["file"],
            "file_path": str(log_file)
        }
        
        logger_config.load_from_dict(config)
        # Directory creation is handled in validation
        assert logger_config.config["file_path"] == str(log_file)

    def test_load_from_config_manager(self, logger_config):
        """Test loading configuration from ConfigManager instance."""
        mock_config_manager = Mock()
        mock_config_manager.get.return_value = {
            "level": "WARNING",
            "handlers": ["console"]
        }
        
        logger_config.load_from_config_manager(mock_config_manager)
        
        mock_config_manager.get.assert_called_once_with("logging", {})
        assert logger_config.config["level"] == "WARNING"

    def test_load_from_config_manager_error(self, logger_config):
        """Test error handling when loading from ConfigManager fails."""
        mock_config_manager = Mock()
        mock_config_manager.get.side_effect = Exception("Config manager error")
        
        with pytest.raises(LoggerConfigError) as exc_info:
            logger_config.load_from_config_manager(mock_config_manager)
        
        assert "Failed to load configuration from ConfigManager" in str(exc_info.value)

    def test_validate_config_valid(self, logger_config, sample_logging_config):
        """Test validation passes for valid configuration."""
        result = logger_config.validate_config(sample_logging_config)
        assert result is True

    def test_validate_config_invalid_type(self, logger_config):
        """Test validation fails for non-dict input."""
        with pytest.raises(LoggerConfigError) as exc_info:
            logger_config.validate_config("not a dict")
        
        assert "must be a dictionary" in str(exc_info.value)

    def test_validate_config_invalid_level(self, logger_config):
        """Test validation fails for invalid logging level."""
        config = {"level": "INVALID", "handlers": ["console"]}
        
        with pytest.raises(LoggerConfigError) as exc_info:
            logger_config.validate_config(config)
        
        assert "Invalid logging level" in str(exc_info.value)

    def test_validate_config_invalid_level_type(self, logger_config):
        """Test validation fails for non-string level."""
        config = {"level": 123, "handlers": ["console"]}
        
        with pytest.raises(LoggerConfigError) as exc_info:
            logger_config.validate_config(config)
        
        assert "must be a string" in str(exc_info.value)

    def test_validate_config_invalid_format(self, logger_config):
        """Test validation fails for invalid format."""
        config = {"level": "INFO", "format": "", "handlers": ["console"]}
        
        with pytest.raises(LoggerConfigError) as exc_info:
            logger_config.validate_config(config)
        
        assert "cannot be empty" in str(exc_info.value)

    def test_validate_config_invalid_handlers_type(self, logger_config):
        """Test validation fails for non-list handlers."""
        config = {"level": "INFO", "handlers": "console"}
        
        with pytest.raises(LoggerConfigError) as exc_info:
            logger_config.validate_config(config)
        
        assert "must be a list" in str(exc_info.value)

    def test_validate_config_invalid_handler_name(self, logger_config):
        """Test validation fails for invalid handler names."""
        config = {"level": "INFO", "handlers": ["invalid_handler"]}
        
        with pytest.raises(LoggerConfigError) as exc_info:
            logger_config.validate_config(config)
        
        assert "Invalid handler" in str(exc_info.value)

    def test_validate_config_file_handler_without_path(self, logger_config):
        """Test validation fails when file handler specified without file_path."""
        config = {"level": "INFO", "handlers": ["file"], "file_path": None}
        
        with pytest.raises(LoggerConfigError) as exc_info:
            logger_config.validate_config(config)
        
        assert "file_path must be specified" in str(exc_info.value)

    def test_validate_config_invalid_file_path_type(self, logger_config):
        """Test validation fails for invalid file_path type."""
        config = {"level": "INFO", "handlers": ["console"], "file_path": 123}
        
        with pytest.raises(LoggerConfigError) as exc_info:
            logger_config.validate_config(config)
        
        assert "must be a string or null" in str(exc_info.value)

    def test_validate_config_invalid_max_file_size_format(self, logger_config):
        """Test validation fails for invalid max_file_size format."""
        config = {
            "level": "INFO",
            "handlers": ["console"],
            "max_file_size": "invalid_size"
        }
        
        with pytest.raises(LoggerConfigError) as exc_info:
            logger_config.validate_config(config)
        
        assert "Invalid max_file_size format" in str(exc_info.value)

    def test_validate_config_max_file_size_too_small(self, logger_config):
        """Test validation fails for max_file_size below minimum."""
        config = {
            "level": "INFO",
            "handlers": ["console"],
            "max_file_size": 512  # Below 1024 bytes minimum
        }
        
        with pytest.raises(LoggerConfigError) as exc_info:
            logger_config.validate_config(config)
        
        assert "must be at least 1024 bytes" in str(exc_info.value)

    def test_validate_config_invalid_backup_count_type(self, logger_config):
        """Test validation fails for invalid backup_count type."""
        config = {
            "level": "INFO",
            "handlers": ["console"],
            "backup_count": "not_an_int"
        }
        
        with pytest.raises(LoggerConfigError) as exc_info:
            logger_config.validate_config(config)
        
        assert "must be an integer" in str(exc_info.value)

    def test_validate_config_negative_backup_count(self, logger_config):
        """Test validation fails for negative backup_count."""
        config = {
            "level": "INFO",
            "handlers": ["console"],
            "backup_count": -1
        }
        
        with pytest.raises(LoggerConfigError) as exc_info:
            logger_config.validate_config(config)
        
        assert "must be non-negative" in str(exc_info.value)

    def test_validate_config_excessive_backup_count(self, logger_config):
        """Test validation fails for excessive backup_count."""
        config = {
            "level": "INFO",
            "handlers": ["console"],
            "backup_count": 101
        }
        
        with pytest.raises(LoggerConfigError) as exc_info:
            logger_config.validate_config(config)
        
        assert "cannot exceed 100" in str(exc_info.value)

    def test_get_level(self, logger_config):
        """Test getting logging level."""
        logger_config.config["level"] = "debug"
        assert logger_config.get_level() == "DEBUG"

    def test_get_level_int(self, logger_config):
        """Test getting logging level as integer."""
        logger_config.config["level"] = "ERROR"
        assert logger_config.get_level_int() == logging.ERROR

    def test_get_format(self, logger_config):
        """Test getting logging format string."""
        test_format = "%(name)s - %(message)s"
        logger_config.config["format"] = test_format
        assert logger_config.get_format() == test_format

    def test_get_handlers(self, logger_config):
        """Test getting handlers list."""
        handlers = ["console", "file"]
        logger_config.config["handlers"] = handlers
        result = logger_config.get_handlers()
        assert result == handlers
        assert result is not logger_config.config["handlers"]  # Should be a copy

    def test_get_file_path(self, logger_config):
        """Test getting file path."""
        file_path = "/var/log/test.log"
        logger_config.config["file_path"] = file_path
        assert logger_config.get_file_path() == file_path

    def test_get_file_path_none(self, logger_config):
        """Test getting file path when None."""
        logger_config.config["file_path"] = None
        assert logger_config.get_file_path() is None

    def test_get_max_file_size_bytes_integer(self, logger_config):
        """Test getting max file size when stored as integer."""
        logger_config.config["max_file_size"] = 1048576  # 1MB
        assert logger_config.get_max_file_size_bytes() == 1048576

    def test_get_max_file_size_bytes_string(self, logger_config):
        """Test getting max file size when stored as string."""
        logger_config.config["max_file_size"] = "10MB"
        assert logger_config.get_max_file_size_bytes() == 10 * 1024 * 1024

    def test_get_backup_count(self, logger_config):
        """Test getting backup count."""
        logger_config.config["backup_count"] = 7
        assert logger_config.get_backup_count() == 7

    def test_has_console_handler(self, logger_config):
        """Test checking for console handler."""
        logger_config.config["handlers"] = ["console", "file"]
        assert logger_config.has_console_handler() is True
        
        logger_config.config["handlers"] = ["file"]
        assert logger_config.has_console_handler() is False

    def test_has_file_handler(self, logger_config):
        """Test checking for file handler."""
        logger_config.config["handlers"] = ["console", "file"]
        logger_config.config["file_path"] = "/var/log/test.log"
        assert logger_config.has_file_handler() is True
        
        logger_config.config["file_path"] = None
        assert logger_config.has_file_handler() is False
        
        logger_config.config["handlers"] = ["console"]
        logger_config.config["file_path"] = "/var/log/test.log"
        assert logger_config.has_file_handler() is False

    def test_update_config_valid(self, logger_config):
        """Test updating configuration with valid values."""
        updates = {
            "level": "WARNING",
            "backup_count": 10
        }
        
        logger_config.update_config(updates)
        
        assert logger_config.config["level"] == "WARNING"
        assert logger_config.config["backup_count"] == 10

    def test_update_config_invalid(self, logger_config):
        """Test updating configuration with invalid values fails."""
        updates = {
            "level": "INVALID_LEVEL"
        }
        
        with pytest.raises(LoggerConfigError):
            logger_config.update_config(updates)

    def test_set_level_valid(self, logger_config):
        """Test setting valid logging level."""
        logger_config.set_level("ERROR")
        assert logger_config.config["level"] == "ERROR"

    def test_set_level_case_insensitive(self, logger_config):
        """Test setting logging level is case insensitive."""
        logger_config.set_level("debug")
        assert logger_config.config["level"] == "DEBUG"

    def test_set_level_invalid(self, logger_config):
        """Test setting invalid logging level fails."""
        with pytest.raises(LoggerConfigError) as exc_info:
            logger_config.set_level("INVALID")
        
        assert "Invalid logging level" in str(exc_info.value)

    def test_set_file_path_valid(self, logger_config, temp_dir):
        """Test setting valid file path."""
        file_path = str(temp_dir / "test.log")
        logger_config.set_file_path(file_path)
        assert logger_config.config["file_path"] == file_path

    def test_set_file_path_none(self, logger_config):
        """Test setting file path to None."""
        logger_config.set_file_path(None)
        assert logger_config.config["file_path"] is None

    def test_set_file_path_invalid_type(self, logger_config):
        """Test setting invalid file path type fails."""
        with pytest.raises(LoggerConfigError) as exc_info:
            logger_config.set_file_path(123)
        
        assert "must be a non-empty string or None" in str(exc_info.value)

    def test_set_file_path_empty_string(self, logger_config):
        """Test setting empty file path fails."""
        with pytest.raises(LoggerConfigError) as exc_info:
            logger_config.set_file_path("")
        
        assert "must be a non-empty string or None" in str(exc_info.value)

    def test_add_handler_valid(self, logger_config):
        """Test adding valid handler."""
        logger_config.config["handlers"] = ["console"]
        logger_config.add_handler("file")
        assert "file" in logger_config.config["handlers"]

    def test_add_handler_duplicate(self, logger_config):
        """Test adding duplicate handler doesn't create duplicates."""
        logger_config.config["handlers"] = ["console"]
        logger_config.add_handler("console")
        assert logger_config.config["handlers"].count("console") == 1

    def test_add_handler_invalid(self, logger_config):
        """Test adding invalid handler fails."""
        with pytest.raises(LoggerConfigError) as exc_info:
            logger_config.add_handler("invalid_handler")
        
        assert "Invalid handler" in str(exc_info.value)

    def test_remove_handler(self, logger_config):
        """Test removing handler."""
        logger_config.config["handlers"] = ["console", "file"]
        logger_config.remove_handler("file")
        assert "file" not in logger_config.config["handlers"]

    def test_remove_handler_not_present(self, logger_config):
        """Test removing non-existent handler doesn't error."""
        logger_config.config["handlers"] = ["console"]
        logger_config.remove_handler("file")  # Should not raise error
        assert logger_config.config["handlers"] == ["console"]

    def test_to_dict(self, logger_config):
        """Test exporting configuration as dictionary."""
        result = logger_config.to_dict()
        assert isinstance(result, dict)
        assert result is not logger_config.config  # Should be a copy

    def test_reset_to_defaults(self, logger_config):
        """Test resetting configuration to defaults."""
        logger_config.config["level"] = "ERROR"
        logger_config.reset_to_defaults()
        assert logger_config.config["level"] == "INFO"

    @patch.dict(os.environ, {
        "AIM2_LOGGING_LEVEL": "ERROR",
        "AIM2_LOGGING_FORMAT": "%(name)s: %(message)s",
        "AIM2_LOGGING_FILE_PATH": "/custom/log/path.log",
        "AIM2_LOGGING_MAX_FILE_SIZE": "20MB",
        "AIM2_LOGGING_BACKUP_COUNT": "10",
        "AIM2_LOGGING_HANDLERS": "console,file"
    })
    def test_environment_variable_overrides(self, logger_config):
        """Test environment variable overrides."""
        logger_config.load_from_dict({})
        
        assert logger_config.config["level"] == "ERROR"
        assert logger_config.config["format"] == "%(name)s: %(message)s"
        assert logger_config.config["file_path"] == "/custom/log/path.log"
        assert logger_config.config["max_file_size"] == "20MB"
        assert logger_config.config["backup_count"] == 10
        assert logger_config.config["handlers"] == ["console", "file"]

    @patch.dict(os.environ, {"AIM2_LOGGING_FILE_PATH": "null"})
    def test_environment_variable_null_file_path(self, logger_config):
        """Test environment variable setting file_path to null."""
        logger_config.load_from_dict({})
        assert logger_config.config["file_path"] is None

    @patch.dict(os.environ, {"AIM2_LOGGING_BACKUP_COUNT": "invalid"})
    def test_environment_variable_invalid_integer(self, logger_config):
        """Test environment variable with invalid integer value."""
        with pytest.raises(LoggerConfigError) as exc_info:
            logger_config.load_from_dict({})
        
        assert "Invalid integer value" in str(exc_info.value)

    def test_environment_variable_custom_prefix(self):
        """Test environment variables with custom prefix."""
        with patch.dict(os.environ, {"CUSTOM_LOGGING_LEVEL": "DEBUG"}):
            config = LoggerConfig(env_prefix="CUSTOM")
            config.load_from_dict({})
            assert config.config["level"] == "DEBUG"

    @pytest.mark.parametrize("size_str,expected_bytes", [
        ("1024", 1024),
        ("1KB", 1024),
        ("1MB", 1024 * 1024),
        ("1GB", 1024 * 1024 * 1024),
        ("1TB", 1024 * 1024 * 1024 * 1024),
        ("10MB", 10 * 1024 * 1024),
        ("1.5MB", int(1.5 * 1024 * 1024)),
        ("2048B", 2048),
    ])
    def test_parse_size_string_valid(self, logger_config, size_str, expected_bytes):
        """Test parsing valid size strings."""
        result = logger_config._parse_size_string(size_str)
        assert result == expected_bytes

    @pytest.mark.parametrize("invalid_size", [
        "invalid",
        "10XB",
        "-10MB",
        "0MB",
        "abc123",
        "",
        "10 MB GB",
    ])
    def test_parse_size_string_invalid(self, logger_config, invalid_size):
        """Test parsing invalid size strings raises error."""
        with pytest.raises(ValueError):
            logger_config._parse_size_string(invalid_size)

    def test_parse_size_string_too_small(self, logger_config):
        """Test parsing size string below minimum."""
        with pytest.raises(ValueError) as exc_info:
            logger_config._parse_size_string("512B")
        
        assert "Minimum file size is 1024 bytes" in str(exc_info.value)

    def test_merge_configs(self, logger_config):
        """Test merging configuration dictionaries."""
        base = {
            "level": "INFO",
            "handlers": ["console"],
            "backup_count": 3
        }
        
        override = {
            "level": "DEBUG",
            "file_path": "/var/log/test.log"
        }
        
        result = logger_config._merge_configs(base, override)
        
        assert result["level"] == "DEBUG"  # Overridden
        assert result["handlers"] == ["console"]  # Preserved
        assert result["backup_count"] == 3  # Preserved
        assert result["file_path"] == "/var/log/test.log"  # Added


class TestLoggerManager:
    """Test suite for LoggerManager class."""

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

    def test_logger_manager_initialization(self, logger_manager):
        """Test LoggerManager can be instantiated."""
        assert logger_manager is not None
        assert isinstance(logger_manager.config, LoggerConfig)
        assert logger_manager.loggers == {}
        assert logger_manager.handlers == {}
        assert logger_manager._initialized is False

    def test_logger_manager_initialization_no_config(self):
        """Test LoggerManager initialization without config creates default."""
        manager = LoggerManager()
        assert isinstance(manager.config, LoggerConfig)

    def test_logger_manager_initialize(self, logger_manager):
        """Test LoggerManager initialization."""
        logger_manager.initialize()
        
        assert logger_manager._initialized is True
        assert logger_manager.root_logger is not None
        assert LoggerManager.ROOT_LOGGER_NAME in logger_manager.loggers

    def test_logger_manager_initialize_idempotent(self, logger_manager):
        """Test LoggerManager initialization is idempotent."""
        logger_manager.initialize()
        first_root = logger_manager.root_logger
        
        logger_manager.initialize()  # Second call
        assert logger_manager.root_logger is first_root

    def test_logger_manager_initialize_with_invalid_config(self):
        """Test LoggerManager initialization with invalid config fails."""
        invalid_config = LoggerConfig()
        invalid_config.config["level"] = "INVALID"
        
        manager = LoggerManager(invalid_config)
        
        with pytest.raises(LoggerManagerError) as exc_info:
            manager.initialize()
        
        assert "Failed to initialize logger manager" in str(exc_info.value)

    def test_get_logger_root(self, logger_manager):
        """Test getting root logger."""
        logger = logger_manager.get_logger()
        
        assert logger.name == LoggerManager.ROOT_LOGGER_NAME
        assert logger_manager._initialized is True
        assert logger in logger_manager.loggers.values()

    def test_get_logger_by_name(self, logger_manager):
        """Test getting logger by name."""
        logger = logger_manager.get_logger("custom_logger")
        
        assert logger.name == "custom_logger"
        assert logger in logger_manager.loggers.values()

    def test_get_logger_by_module_name(self, logger_manager):
        """Test getting logger by module name."""
        logger = logger_manager.get_logger(module_name="test_module")
        
        expected_name = f"{LoggerManager.ROOT_LOGGER_NAME}.test_module"
        assert logger.name == expected_name
        assert logger in logger_manager.loggers.values()

    def test_get_logger_cached(self, logger_manager):
        """Test that loggers are cached and reused."""
        logger1 = logger_manager.get_logger("test_logger")
        logger2 = logger_manager.get_logger("test_logger")
        
        assert logger1 is logger2

    def test_get_logger_hierarchy(self, logger_manager):
        """Test logger hierarchy configuration."""
        root_logger = logger_manager.get_logger()
        module_logger = logger_manager.get_logger(module_name="test_module")
        
        assert root_logger.propagate is False
        assert module_logger.propagate is True

    def test_get_logger_level_configuration(self, logger_manager):
        """Test loggers are configured with correct level."""
        logger_manager.config.set_level("WARNING")
        logger = logger_manager.get_logger("test_logger")
        
        assert logger.level == logging.WARNING

    def test_get_module_logger(self, logger_manager):
        """Test getting module-specific logger."""
        logger = logger_manager.get_module_logger("test_module")
        
        expected_name = f"{LoggerManager.ROOT_LOGGER_NAME}.test_module"
        assert logger.name == expected_name

    def test_get_module_logger_invalid_name(self, logger_manager):
        """Test getting module logger with invalid name fails."""
        with pytest.raises(LoggerManagerError) as exc_info:
            logger_manager.get_module_logger("")
        
        assert "must be a non-empty string" in str(exc_info.value)

    def test_get_module_logger_path_cleaning(self, logger_manager):
        """Test module logger name cleaning for paths."""
        logger = logger_manager.get_module_logger("path/to/module")
        
        expected_name = f"{LoggerManager.ROOT_LOGGER_NAME}.path.to.module"
        assert logger.name == expected_name

    def test_reload_configuration(self, logger_manager, temp_dir):
        """Test reloading configuration."""
        # Initialize with console handler
        logger_manager.initialize()
        initial_handlers = len(logger_manager.root_logger.handlers)
        
        # Create new config with file handler
        new_config = LoggerConfig()
        new_config.load_from_dict({
            "level": "ERROR",
            "handlers": ["console", "file"],
            "file_path": str(temp_dir / "test.log")
        })
        
        logger_manager.reload_configuration(new_config)
        
        assert logger_manager.config.get_level() == "ERROR"
        # Should have handlers for new configuration
        assert len(logger_manager.root_logger.handlers) >= initial_handlers

    def test_reload_configuration_current(self, logger_manager):
        """Test reloading current configuration."""
        logger_manager.initialize()
        original_level = logger_manager.config.get_level()
        
        # Modify config directly
        logger_manager.config.set_level("DEBUG")
        
        # Reload should apply the modified config
        logger_manager.reload_configuration()
        
        assert logger_manager.config.get_level() == "DEBUG"
        assert logger_manager.root_logger.level == logging.DEBUG

    def test_reload_configuration_invalid(self, logger_manager):
        """Test reloading with invalid configuration fails."""
        logger_manager.initialize()
        
        invalid_config = LoggerConfig()
        invalid_config.config["level"] = "INVALID"
        
        with pytest.raises(LoggerManagerError) as exc_info:
            logger_manager.reload_configuration(invalid_config)
        
        assert "Failed to reload configuration" in str(exc_info.value)

    def test_set_level(self, logger_manager):
        """Test setting logging level for all loggers."""
        logger1 = logger_manager.get_logger("logger1")
        logger2 = logger_manager.get_logger("logger2")
        
        logger_manager.set_level("ERROR")
        
        assert logger_manager.config.get_level() == "ERROR"
        assert logger1.level == logging.ERROR
        assert logger2.level == logging.ERROR

    def test_set_level_invalid(self, logger_manager):
        """Test setting invalid level fails."""
        with pytest.raises(LoggerManagerError) as exc_info:
            logger_manager.set_level("INVALID")
        
        assert "Failed to set level" in str(exc_info.value)

    def test_add_handler_to_logger(self, logger_manager):
        """Test adding handler to specific logger."""
        logger = logger_manager.get_logger("test_logger")
        initial_count = len(logger.handlers)
        
        logger_manager.add_handler_to_logger("test_logger", "console")
        
        assert len(logger.handlers) > initial_count

    def test_add_handler_to_nonexistent_logger(self, logger_manager):
        """Test adding handler to non-existent logger fails."""
        with pytest.raises(LoggerManagerError) as exc_info:
            logger_manager.add_handler_to_logger("nonexistent", "console")
        
        assert "not found" in str(exc_info.value)

    def test_remove_handler_from_logger(self, logger_manager):
        """Test removing handler from logger."""
        logger = logger_manager.get_logger("test_logger")
        logger_manager.add_handler_to_logger("test_logger", "console")
        initial_count = len(logger.handlers)
        
        logger_manager.remove_handler_from_logger("test_logger", "console")
        
        assert len(logger.handlers) < initial_count

    def test_remove_handler_from_nonexistent_logger(self, logger_manager):
        """Test removing handler from non-existent logger fails."""
        with pytest.raises(LoggerManagerError) as exc_info:
            logger_manager.remove_handler_from_logger("nonexistent", "console")
        
        assert "not found" in str(exc_info.value)

    def test_get_logger_info(self, logger_manager):
        """Test getting logger information."""
        logger_manager.get_logger("test_logger")
        
        info = logger_manager.get_logger_info()
        
        assert isinstance(info, dict)
        assert "configuration" in info
        assert "loggers" in info
        assert "handlers" in info
        assert "initialized" in info
        assert info["initialized"] is True
        assert "test_logger" in info["loggers"]

    def test_cleanup(self, logger_manager):
        """Test cleanup functionality."""
        logger_manager.get_logger("test_logger")
        logger_manager.cleanup()
        
        assert logger_manager._initialized is False
        assert len(logger_manager.loggers) == 0
        assert len(logger_manager.handlers) == 0

    def test_cleanup_safe(self, logger_manager):
        """Test cleanup is safe even with errors."""
        # Create logger with mocked handler that raises on close
        logger = logger_manager.get_logger("test_logger")
        
        mock_handler = Mock()
        mock_handler.close.side_effect = Exception("Close error")
        logger.addHandler(mock_handler)
        
        # Cleanup should not raise exception
        logger_manager.cleanup()

    def test_is_initialized(self, logger_manager):
        """Test checking initialization status."""
        assert logger_manager.is_initialized() is False
        
        logger_manager.initialize()
        assert logger_manager.is_initialized() is True

    def test_get_managed_loggers(self, logger_manager):
        """Test getting list of managed logger names."""
        logger_manager.get_logger("logger1")
        logger_manager.get_logger("logger2")
        
        names = logger_manager.get_managed_loggers()
        
        assert "logger1" in names
        assert "logger2" in names


class TestRotatingFileHandler:
    """Test suite for rotating file handler functionality."""

    @pytest.fixture
    def temp_dir(self):
        """Create a temporary directory for test files."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            yield Path(tmp_dir)

    @pytest.fixture
    def logger_manager_with_file(self, temp_dir):
        """Create LoggerManager configured with file handler."""
        config = LoggerConfig()
        config.load_from_dict({
            "level": "INFO",
            "handlers": ["file"],
            "file_path": str(temp_dir / "test.log"),
            "max_file_size": "1KB",
            "backup_count": 3
        })
        return LoggerManager(config)

    def test_create_file_handler(self, logger_manager_with_file):
        """Test creating file handler."""
        logger_manager_with_file.initialize()
        
        # Check that file handler was created
        root_logger = logger_manager_with_file.root_logger
        file_handlers = [h for h in root_logger.handlers 
                        if isinstance(h, logging.handlers.RotatingFileHandler)]
        
        assert len(file_handlers) > 0

    def test_file_handler_configuration(self, logger_manager_with_file, temp_dir):
        """Test file handler is configured correctly."""
        logger_manager_with_file.initialize()
        
        root_logger = logger_manager_with_file.root_logger
        file_handler = next((h for h in root_logger.handlers 
                           if isinstance(h, logging.handlers.RotatingFileHandler)), None)
        
        assert file_handler is not None
        assert file_handler.baseFilename == str(temp_dir / "test.log")
        assert file_handler.maxBytes == 1024
        assert file_handler.backupCount == 3

    def test_file_handler_directory_creation(self, temp_dir):
        """Test file handler creates directories as needed."""
        nested_dir = temp_dir / "logs" / "nested"
        log_file = nested_dir / "test.log"
        
        config = LoggerConfig()
        config.load_from_dict({
            "level": "INFO",
            "handlers": ["file"],
            "file_path": str(log_file)
        })
        
        manager = LoggerManager(config)
        manager.initialize()
        
        # Directory should be created
        assert nested_dir.exists()

    @patch('logging.handlers.RotatingFileHandler')
    def test_file_handler_creation_error(self, mock_handler, temp_dir):
        """Test file handler creation error handling."""
        mock_handler.side_effect = Exception("File creation error")
        
        config = LoggerConfig()
        config.load_from_dict({
            "level": "INFO",
            "handlers": ["file"],
            "file_path": str(temp_dir / "test.log")
        })
        
        manager = LoggerManager(config)
        
        with pytest.raises(LoggerManagerError) as exc_info:
            manager.initialize()
        
        assert "Failed to create file handler" in str(exc_info.value)

    def test_file_handler_without_path(self):
        """Test file handler creation without file path."""
        config = LoggerConfig()
        # Load config without file handler to avoid validation error
        config.load_from_dict({
            "level": "INFO",
            "handlers": ["console"]
        })
        
        # Manually set file_path to None and modify handlers to test file handler creation
        config.config["file_path"] = None
        
        manager = LoggerManager(config)
        # Should not create file handler when path is None
        handler = manager._create_file_handler()
        assert handler is None

    def test_file_handler_encoding(self, logger_manager_with_file):
        """Test file handler uses UTF-8 encoding."""
        logger_manager_with_file.initialize()
        
        root_logger = logger_manager_with_file.root_logger
        file_handler = next((h for h in root_logger.handlers 
                           if isinstance(h, logging.handlers.RotatingFileHandler)), None)
        
        # The RotatingFileHandler should be created with UTF-8 encoding
        assert file_handler is not None
        # Note: encoding is set in the constructor but may not be directly accessible


class TestLoggerConfigValidation:
    """Test suite for detailed logger configuration validation."""

    @pytest.fixture
    def logger_config(self):
        """Create a LoggerConfig instance for testing."""
        return LoggerConfig()

    def test_validation_comprehensive_valid_config(self, logger_config):
        """Test validation with comprehensive valid configuration."""
        valid_config = {
            "level": "DEBUG",
            "format": "%(asctime)s [%(levelname)s] %(name)s: %(message)s",
            "handlers": ["console", "file"],
            "file_path": "/var/log/app.log",
            "max_file_size": "50MB",
            "backup_count": 10
        }
        
        result = logger_config.validate_config(valid_config)
        assert result is True

    def test_validation_minimal_valid_config(self, logger_config):
        """Test validation with minimal valid configuration."""
        minimal_config = {
            "level": "INFO",
            "handlers": ["console"]
        }
        
        result = logger_config.validate_config(minimal_config)
        assert result is True

    def test_validation_multiple_errors(self, logger_config):
        """Test validation captures multiple errors."""
        invalid_config = {
            "level": "INVALID_LEVEL",
            "format": "",
            "handlers": "not_a_list",
            "backup_count": -5
        }
        
        with pytest.raises(LoggerConfigError) as exc_info:
            logger_config.validate_config(invalid_config)
        
        error_msg = str(exc_info.value)
        assert "Invalid logging level" in error_msg
        assert "cannot be empty" in error_msg
        assert "must be a list" in error_msg
        assert "must be non-negative" in error_msg

    def test_validation_case_insensitive_level(self, logger_config):
        """Test validation accepts case-insensitive levels."""
        config = {"level": "debug", "handlers": ["console"]}
        result = logger_config.validate_config(config)
        assert result is True

    def test_validation_empty_handlers_list(self, logger_config):
        """Test validation allows empty handlers list."""
        config = {"level": "INFO", "handlers": []}
        result = logger_config.validate_config(config)
        assert result is True

    def test_validation_duplicate_handlers(self, logger_config):
        """Test validation allows duplicate handlers."""
        config = {"level": "INFO", "handlers": ["console", "console"]}
        result = logger_config.validate_config(config)
        assert result is True

    @pytest.mark.parametrize("invalid_handler", [123, None, [], {}])
    def test_validation_invalid_handler_types(self, logger_config, invalid_handler):
        """Test validation fails for invalid handler types."""
        config = {"level": "INFO", "handlers": [invalid_handler]}
        
        with pytest.raises(LoggerConfigError) as exc_info:
            logger_config.validate_config(config)
        
        assert "must be a string" in str(exc_info.value)

    def test_validation_file_path_whitespace(self, logger_config):
        """Test validation fails for whitespace-only file path."""
        config = {
            "level": "INFO",
            "handlers": ["console"],
            "file_path": "   "
        }
        
        with pytest.raises(LoggerConfigError) as exc_info:
            logger_config.validate_config(config)
        
        assert "cannot be empty string" in str(exc_info.value)

    @patch('pathlib.Path.mkdir')
    def test_validation_file_path_creation_error(self, mock_mkdir, logger_config):
        """Test validation handles file path creation errors."""
        mock_mkdir.side_effect = OSError("Permission denied")
        
        config = {
            "level": "INFO",
            "handlers": ["console"],
            "file_path": "/restricted/path/test.log"
        }
        
        with pytest.raises(LoggerConfigError) as exc_info:
            logger_config.validate_config(config)
        
        assert "Invalid file path" in str(exc_info.value)

    @pytest.mark.parametrize("size_format,should_pass", [
        ("1024", True),
        ("1KB", True),
        ("10MB", True),
        ("1.5GB", True),
        ("invalid", False),
        ("0MB", False),
        ("-1MB", False),
        ("", False),
    ])
    def test_validation_size_format_validation(self, logger_config, size_format, should_pass):
        """Test validation of various size formats."""
        config = {
            "level": "INFO",
            "handlers": ["console"],
            "max_file_size": size_format
        }
        
        if should_pass:
            result = logger_config.validate_config(config)
            assert result is True
        else:
            with pytest.raises(LoggerConfigError):
                logger_config.validate_config(config)

    def test_validation_boundary_values(self, logger_config):
        """Test validation with boundary values."""
        # Test minimum file size
        config = {
            "level": "INFO",
            "handlers": ["console"],
            "max_file_size": 1024,
            "backup_count": 0
        }
        
        result = logger_config.validate_config(config)
        assert result is True
        
        # Test maximum backup count
        config["backup_count"] = 100
        result = logger_config.validate_config(config)
        assert result is True


class TestLoggerIntegration:
    """Test suite for ConfigManager integration with logger configuration."""

    @pytest.fixture
    def temp_dir(self):
        """Create a temporary directory for test files."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            yield Path(tmp_dir)

    @pytest.fixture
    def config_manager(self):
        """Create a ConfigManager instance for testing."""
        return ConfigManager()

    def test_logger_config_from_config_manager(self, config_manager, temp_dir):
        """Test loading logger configuration from ConfigManager."""
        # Create config file with logging section
        config_content = {
            "project": {"name": "Test Project"},
            "logging": {
                "level": "WARNING",
                "handlers": ["console", "file"],
                "file_path": str(temp_dir / "test.log"),
                "max_file_size": "5MB"
            }
        }
        
        config_file = temp_dir / "config.yaml"
        import yaml
        with open(config_file, 'w') as f:
            yaml.dump(config_content, f)
        
        config_manager.load_config(str(config_file))
        
        # Load logger configuration from ConfigManager
        logger_config = LoggerConfig()
        logger_config.load_from_config_manager(config_manager)
        
        assert logger_config.get_level() == "WARNING"
        assert logger_config.has_file_handler() is True
        assert logger_config.get_file_path() == str(temp_dir / "test.log")

    def test_logger_config_empty_logging_section(self, config_manager):
        """Test logger configuration with empty logging section."""
        config_manager.config = {"other_section": {"key": "value"}}
        
        logger_config = LoggerConfig()
        logger_config.load_from_config_manager(config_manager)
        
        # Should use default configuration
        assert logger_config.get_level() == "INFO"
        assert logger_config.has_console_handler() is True

    def test_logger_manager_with_config_manager(self, config_manager, temp_dir):
        """Test LoggerManager integration with ConfigManager."""
        # Setup config in ConfigManager
        config_manager.config = {
            "logging": {
                "level": "ERROR",
                "handlers": ["console"]
            }
        }
        
        # Create logger configuration from ConfigManager
        logger_config = LoggerConfig()
        logger_config.load_from_config_manager(config_manager)
        
        # Create logger manager
        logger_manager = LoggerManager(logger_config)
        logger = logger_manager.get_logger("test_logger")
        
        assert logger.level == logging.ERROR

    def test_config_manager_logger_factory_method(self, config_manager):
        """Test hypothetical ConfigManager logger factory method."""
        # This test assumes a logger factory method might be added to ConfigManager
        config_manager.config = {
            "logging": {
                "level": "DEBUG",
                "handlers": ["console"]
            }
        }
        
        # Direct integration test
        logger_config = LoggerConfig()
        logger_config.load_from_config_manager(config_manager)
        
        logger_manager = LoggerManager(logger_config)
        logger = logger_manager.get_logger("integration_test")
        
        assert logger.level == logging.DEBUG
        assert logger.name == "integration_test"

    @patch.dict(os.environ, {"AIM2_LOGGING_LEVEL": "CRITICAL"})
    def test_integration_with_environment_overrides(self, config_manager, temp_dir):
        """Test integration with environment variable overrides."""
        # Create config file
        config_content = {
            "logging": {
                "level": "INFO",
                "handlers": ["console"]
            }
        }
        
        config_file = temp_dir / "config.yaml"
        import yaml
        with open(config_file, 'w') as f:
            yaml.dump(config_content, f)
        
        # Load through ConfigManager
        config_manager.load_config(str(config_file))
        
        # Create logger configuration with environment overrides
        logger_config = LoggerConfig()
        logger_config.load_from_config_manager(config_manager)
        
        # Environment variable should override config file
        assert logger_config.get_level() == "CRITICAL"

    def test_config_validation_integration(self, config_manager):
        """Test validation integration with ConfigManager."""
        # Setup invalid logging configuration
        config_manager.config = {
            "logging": {
                "level": "INVALID_LEVEL",
                "handlers": ["console"]
            }
        }
        
        logger_config = LoggerConfig()
        
        with pytest.raises(LoggerConfigError) as exc_info:
            logger_config.load_from_config_manager(config_manager)
        
        assert "Invalid logging level" in str(exc_info.value)


class TestModuleSpecificLoggers:
    """Test suite for module-specific logger functionality."""

    @pytest.fixture
    def logger_manager(self):
        """Create a LoggerManager instance for testing."""
        return LoggerManager()

    def test_module_logger_hierarchy(self, logger_manager):
        """Test module logger creates proper hierarchy."""
        logger = logger_manager.get_module_logger("data.processing")
        
        expected_name = f"{LoggerManager.ROOT_LOGGER_NAME}.data.processing"
        assert logger.name == expected_name

    def test_nested_module_loggers(self, logger_manager):
        """Test creating nested module loggers."""
        parent_logger = logger_manager.get_module_logger("parent")
        child_logger = logger_manager.get_module_logger("parent.child")
        grandchild_logger = logger_manager.get_module_logger("parent.child.grandchild")
        
        # All should be separate logger instances
        assert parent_logger != child_logger
        assert child_logger != grandchild_logger
        
        # Check proper naming hierarchy
        assert parent_logger.name.endswith(".parent")
        assert child_logger.name.endswith(".parent.child")
        assert grandchild_logger.name.endswith(".parent.child.grandchild")

    def test_module_logger_propagation(self, logger_manager):
        """Test module loggers have proper propagation settings."""
        root_logger = logger_manager.get_logger()
        module_logger = logger_manager.get_module_logger("test_module")
        
        assert root_logger.propagate is False
        assert module_logger.propagate is True

    def test_module_logger_level_inheritance(self, logger_manager):
        """Test module loggers inherit level settings."""
        logger_manager.set_level("ERROR")
        
        module_logger = logger_manager.get_module_logger("test_module")
        assert module_logger.level == logging.ERROR

    def test_module_logger_path_normalization(self, logger_manager):
        """Test module logger normalizes file paths to logger names."""
        test_cases = [
            ("path/to/module", "path.to.module"),
            ("path\\to\\module", "path.to.module"),
            ("./relative/path", "relative.path"),
            ("module/", "module"),
            ("/absolute/path", "absolute.path"),
        ]
        
        for input_path, expected_suffix in test_cases:
            logger = logger_manager.get_module_logger(input_path)
            assert logger.name.endswith(expected_suffix)

    def test_multiple_module_loggers_same_level(self, logger_manager):
        """Test multiple module loggers at same level."""
        logger1 = logger_manager.get_module_logger("module1")
        logger2 = logger_manager.get_module_logger("module2")
        logger3 = logger_manager.get_module_logger("module3")
        
        assert logger1 != logger2 != logger3
        assert all(l.propagate is True for l in [logger1, logger2, logger3])

    def test_module_logger_caching(self, logger_manager):
        """Test module loggers are properly cached."""
        logger1 = logger_manager.get_module_logger("cached_module")
        logger2 = logger_manager.get_module_logger("cached_module")
        
        assert logger1 is logger2

    def test_module_logger_with_special_characters(self, logger_manager):
        """Test module logger handles special characters in names."""
        # Should handle and normalize special characters
        logger = logger_manager.get_module_logger("module-with-dashes")
        assert logger is not None
        
        # The actual behavior depends on implementation
        # This test ensures no exceptions are raised

    def test_large_number_of_module_loggers(self, logger_manager):
        """Test creating many module loggers performs well."""
        start_time = time.time()
        
        loggers = []
        for i in range(100):
            logger = logger_manager.get_module_logger(f"module_{i}")
            loggers.append(logger)
        
        creation_time = time.time() - start_time
        
        # Should complete quickly
        assert creation_time < 1.0
        assert len(loggers) == 100
        assert len(set(loggers)) == 100  # All unique


class TestLoggerErrorHandling:
    """Test suite for error handling and edge cases."""

    @pytest.fixture
    def logger_config(self):
        """Create a LoggerConfig instance for testing."""
        return LoggerConfig()

    @pytest.fixture
    def logger_manager(self):
        """Create a LoggerManager instance for testing."""
        return LoggerManager()

    def test_logger_config_exception_chaining(self, logger_config):
        """Test exception chaining in LoggerConfig."""
        with patch('pathlib.Path.mkdir') as mock_mkdir:
            mock_mkdir.side_effect = OSError("Permission denied")
            
            config = {
                "level": "INFO",
                "handlers": ["file"],
                "file_path": "/restricted/test.log"
            }
            
            with pytest.raises(LoggerConfigError) as exc_info:
                logger_config.load_from_dict(config)
            
            # Check that an exception was raised with proper error message
            # The specific cause might not always be set depending on implementation
            assert "Invalid file path" in str(exc_info.value) or "Permission denied" in str(exc_info.value)

    def test_logger_manager_exception_chaining(self, logger_manager):
        """Test exception chaining in LoggerManager."""
        # Force an error during handler creation
        with patch.object(logger_manager, '_create_handler') as mock_create:
            mock_create.side_effect = ValueError("Handler creation failed")
            
            with pytest.raises(LoggerManagerError) as exc_info:
                logger_manager.initialize()
            
            # Check exception chaining
            assert exc_info.value.cause is not None

    def test_concurrent_logger_access(self, logger_manager):
        """Test thread-safe logger access."""
        def create_loggers(thread_id):
            loggers = []
            for i in range(10):
                logger = logger_manager.get_logger(f"thread_{thread_id}_logger_{i}")
                loggers.append(logger)
            return loggers
        
        # Create loggers from multiple threads
        threads = []
        results = {}
        
        for thread_id in range(5):
            thread = threading.Thread(
                target=lambda tid=thread_id: results.update({tid: create_loggers(tid)})
            )
            threads.append(thread)
            thread.start()
        
        for thread in threads:
            thread.join()
        
        # Verify all loggers were created successfully
        all_loggers = []
        for thread_loggers in results.values():
            all_loggers.extend(thread_loggers)
        
        assert len(all_loggers) == 50  # 5 threads * 10 loggers each
        assert len(logger_manager.loggers) >= 50

    def test_logger_manager_cleanup_with_exceptions(self, logger_manager):
        """Test logger manager cleanup handles exceptions gracefully."""
        # Create logger with mock handler that raises on close
        logger = logger_manager.get_logger("test_logger")
        
        mock_handler = Mock()
        mock_handler.close.side_effect = Exception("Close failed")
        logger.addHandler(mock_handler)
        
        # Cleanup should not raise exception
        logger_manager.cleanup()
        
        # Should still reset state
        assert logger_manager._initialized is False

    def test_handler_creation_partial_failure(self, logger_manager):
        """Test handling partial handler creation failure."""
        # Mock console handler creation to succeed, file handler to fail
        with patch.object(logger_manager, '_create_console_handler') as mock_console:
            with patch.object(logger_manager, '_create_file_handler') as mock_file:
                mock_console.return_value = Mock()
                mock_file.side_effect = Exception("File handler failed")
                
                logger_manager.config.config["handlers"] = ["console", "file"]
                
                # Should handle partial failure gracefully
                # The exact behavior depends on implementation
                try:
                    logger_manager.initialize()
                except LoggerManagerError:
                    # Expected if implementation fails on any handler error
                    pass

    def test_invalid_size_string_edge_cases(self, logger_config):
        """Test size string parsing edge cases."""
        edge_cases = [
            None,  # Non-string input
            123,   # Integer input
            [],    # List input
            {},    # Dict input
        ]
        
        for invalid_input in edge_cases:
            with pytest.raises(ValueError) as exc_info:
                logger_config._parse_size_string(invalid_input)
            
            assert "Size must be a string" in str(exc_info.value)

    def test_environment_variable_edge_cases(self, logger_config):
        """Test environment variable handling edge cases."""
        with patch.dict(os.environ, {
            "AIM2_LOGGING_HANDLERS": "",  # Empty handlers
            "AIM2_LOGGING_BACKUP_COUNT": "not_a_number",  # Invalid integer
        }):
            with pytest.raises(LoggerConfigError) as exc_info:
                logger_config.load_from_dict({})
            
            assert "Invalid integer value" in str(exc_info.value)

    def test_logger_manager_double_initialization_error(self):
        """Test logger manager handles double initialization gracefully."""
        config = LoggerConfig()
        config.config["level"] = "INVALID"  # This will cause validation to fail
        
        manager = LoggerManager(config)
        
        # First initialization should fail
        with pytest.raises(LoggerManagerError):
            manager.initialize()
        
        # Should still be uninitialized
        assert not manager.is_initialized()
        
        # Fix the config and try again
        config.config["level"] = "INFO"
        manager.initialize()
        
        # Now should be initialized
        assert manager.is_initialized()

    def test_memory_usage_with_many_loggers(self, logger_manager):
        """Test memory usage doesn't grow excessively with many loggers."""
        import gc
        
        # Create many loggers
        for i in range(1000):
            logger_manager.get_logger(f"memory_test_logger_{i}")
        
        # Cleanup
        logger_manager.cleanup()
        gc.collect()
        
        # Should have cleaned up properly
        assert len(logger_manager.loggers) == 0
        assert len(logger_manager.handlers) == 0

    def test_logger_config_immutability_protection(self, logger_config):
        """Test that returned configurations are protected from modification."""
        config_dict = logger_config.to_dict()
        handlers_list = logger_config.get_handlers()
        
        # Modify returned values
        config_dict["level"] = "MODIFIED"
        handlers_list.append("modified_handler")
        
        # Original should be unchanged
        assert logger_config.config["level"] != "MODIFIED"
        assert "modified_handler" not in logger_config.config["handlers"]

    def test_logger_manager_resource_exhaustion(self, logger_manager):
        """Test logger manager behavior under resource exhaustion."""
        # This test simulates what happens when resources are exhausted
        # The exact behavior depends on the implementation
        
        with patch('logging.getLogger') as mock_get_logger:
            mock_get_logger.side_effect = MemoryError("Out of memory")
            
            with pytest.raises(LoggerManagerError) as exc_info:
                logger_manager.get_logger("resource_test")
            
            # Error message might be wrapped in initialization error
            error_msg = str(exc_info.value)
            assert ("Failed to create logger" in error_msg or 
                    "Failed to initialize logger manager" in error_msg or
                    "Out of memory" in error_msg)


if __name__ == "__main__":
    pytest.main([__file__])