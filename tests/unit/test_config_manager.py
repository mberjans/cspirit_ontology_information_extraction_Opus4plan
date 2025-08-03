"""
Unit tests for config_manager.py

This module contains comprehensive unit tests for the configuration management system,
following Test-Driven Development (TDD) approach. These tests define the expected
interface and behavior of the ConfigManager class before implementation.

Test Coverage:
- YAML file loading and parsing
- JSON file loading support
- Environment variable override logic
- Config validation and error handling
- Default config loading
- Malformed config file handling
- Config merging and validation
- Multiple config sources
"""

import os
import json
import tempfile
import pytest
import time
import threading
from unittest.mock import patch, Mock, MagicMock
from pathlib import Path

# Import the module to be tested (will fail initially in TDD)
try:
    from aim2_project.aim2_utils.config_manager import ConfigManager, ConfigError, ConfigFileWatcher
    from watchdog.observers import Observer
    from watchdog.events import FileModifiedEvent, FileCreatedEvent
except ImportError:
    # Expected during TDD - tests define the interface
    pass


class TestConfigManager:
    """Test suite for ConfigManager class."""

    @pytest.fixture
    def temp_dir(self):
        """Create a temporary directory for test files."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            yield Path(tmp_dir)

    @pytest.fixture
    def sample_yaml_config(self):
        """Sample YAML configuration for testing."""
        return """
# Sample configuration file
database:
  host: localhost
  port: 5432
  name: test_db
  username: test_user
  password: test_pass

api:
  base_url: https://api.example.com
  timeout: 30
  retry_attempts: 3

logging:
  level: INFO
  format: "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
  handlers:
    - console
    - file

features:
  enable_caching: true
  max_cache_size: 1000
  debug_mode: false

nlp:
  model_name: "bert-base-uncased"
  max_sequence_length: 512
  batch_size: 32
"""

    @pytest.fixture
    def sample_json_config(self):
        """Sample JSON configuration for testing."""
        return {
            "database": {
                "host": "localhost",
                "port": 5432,
                "name": "test_db",
                "username": "test_user",
                "password": "test_pass",
            },
            "api": {
                "base_url": "https://api.example.com",
                "timeout": 30,
                "retry_attempts": 3,
            },
            "logging": {
                "level": "INFO",
                "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
                "handlers": ["console", "file"],
            },
            "features": {
                "enable_caching": True,
                "max_cache_size": 1000,
                "debug_mode": False,
            },
        }

    @pytest.fixture
    def malformed_yaml_config(self):
        """Malformed YAML configuration for error testing."""
        return """
database:
  host: localhost
  port: 5432
  name: test_db
    username: test_user  # Invalid indentation
  password: test_pass
    invalid_structure:
  - item1
    - item2  # Invalid YAML structure
"""

    @pytest.fixture
    def config_manager(self):
        """Create a ConfigManager instance for testing."""
        return ConfigManager()

    def test_config_manager_initialization(self, config_manager):
        """Test ConfigManager can be instantiated."""
        assert config_manager is not None
        assert hasattr(config_manager, "config")
        assert isinstance(config_manager.config, dict)

    def test_load_yaml_config_file(self, config_manager, temp_dir, sample_yaml_config):
        """Test loading a valid YAML configuration file."""
        # Create test YAML file
        yaml_file = temp_dir / "test_config.yaml"
        yaml_file.write_text(sample_yaml_config)

        # Load configuration
        config_manager.load_config(str(yaml_file))

        # Verify config loaded correctly
        assert config_manager.config["database"]["host"] == "localhost"
        assert config_manager.config["database"]["port"] == 5432
        assert config_manager.config["api"]["timeout"] == 30
        assert config_manager.config["features"]["enable_caching"] is True
        assert config_manager.config["nlp"]["model_name"] == "bert-base-uncased"

    def test_load_json_config_file(self, config_manager, temp_dir, sample_json_config):
        """Test loading a valid JSON configuration file."""
        # Create test JSON file
        json_file = temp_dir / "test_config.json"
        json_file.write_text(json.dumps(sample_json_config, indent=2))

        # Load configuration
        config_manager.load_config(str(json_file))

        # Verify config loaded correctly
        assert config_manager.config["database"]["host"] == "localhost"
        assert config_manager.config["database"]["port"] == 5432
        assert config_manager.config["api"]["timeout"] == 30
        assert config_manager.config["features"]["enable_caching"] is True

    def test_load_nonexistent_file_raises_error(self, config_manager):
        """Test that loading a non-existent file raises ConfigError."""
        with pytest.raises(ConfigError) as exc_info:
            config_manager.load_config("/nonexistent/path/config.yaml")

        assert "File not found" in str(exc_info.value) or "does not exist" in str(
            exc_info.value
        )

    def test_load_malformed_yaml_raises_error(
        self, config_manager, temp_dir, malformed_yaml_config
    ):
        """Test that loading malformed YAML raises ConfigError."""
        # Create malformed YAML file
        yaml_file = temp_dir / "malformed_config.yaml"
        yaml_file.write_text(malformed_yaml_config)

        with pytest.raises(ConfigError) as exc_info:
            config_manager.load_config(str(yaml_file))

        assert "YAML" in str(exc_info.value) or "parse" in str(exc_info.value).lower()

    def test_load_malformed_json_raises_error(self, config_manager, temp_dir):
        """Test that loading malformed JSON raises ConfigError."""
        # Create malformed JSON file
        malformed_json = (
            '{"database": {"host": "localhost", "port": 5432,}}'  # Trailing comma
        )
        json_file = temp_dir / "malformed_config.json"
        json_file.write_text(malformed_json)

        with pytest.raises(ConfigError) as exc_info:
            config_manager.load_config(str(json_file))

        assert "JSON" in str(exc_info.value) or "parse" in str(exc_info.value).lower()

    def test_load_default_config(self, config_manager):
        """Test loading default configuration."""
        config_manager.load_default_config()

        # Default config should contain basic structure
        assert isinstance(config_manager.config, dict)
        assert len(config_manager.config) > 0

    def test_environment_variable_override(
        self, config_manager, temp_dir, sample_yaml_config
    ):
        """Test that environment variables override config file values."""
        # Create test YAML file
        yaml_file = temp_dir / "test_config.yaml"
        yaml_file.write_text(sample_yaml_config)

        # Set environment variables
        env_vars = {
            "AIM2_DATABASE_HOST": "env-host",
            "AIM2_DATABASE_PORT": "9999",
            "AIM2_API_TIMEOUT": "60",
            "AIM2_FEATURES_DEBUG_MODE": "true",
        }

        with patch.dict(os.environ, env_vars):
            config_manager.load_config(str(yaml_file))

        # Verify environment variables override file values
        assert config_manager.config["database"]["host"] == "env-host"
        assert config_manager.config["database"]["port"] == 9999
        assert config_manager.config["api"]["timeout"] == 60
        assert config_manager.config["features"]["debug_mode"] is True

    def test_nested_environment_variable_override(
        self, config_manager, temp_dir, sample_yaml_config
    ):
        """Test environment variable override for deeply nested config values."""
        yaml_file = temp_dir / "test_config.yaml"
        yaml_file.write_text(sample_yaml_config)

        env_vars = {
            "AIM2_NLP_MODEL_NAME": "roberta-base",
            "AIM2_NLP_MAX_SEQUENCE_LENGTH": "256",
        }

        with patch.dict(os.environ, env_vars):
            config_manager.load_config(str(yaml_file))

        assert config_manager.config["nlp"]["model_name"] == "roberta-base"
        assert config_manager.config["nlp"]["max_sequence_length"] == 256

    def test_environment_variable_type_conversion(
        self, config_manager, temp_dir, sample_yaml_config
    ):
        """Test automatic type conversion for environment variables."""
        yaml_file = temp_dir / "test_config.yaml"
        yaml_file.write_text(sample_yaml_config)

        env_vars = {
            "AIM2_DATABASE_PORT": "8080",  # String that should become int
            "AIM2_FEATURES_ENABLE_CACHING": "false",  # String that should become bool
            "AIM2_FEATURES_MAX_CACHE_SIZE": "2000",  # String that should become int
        }

        with patch.dict(os.environ, env_vars):
            config_manager.load_config(str(yaml_file))

        assert config_manager.config["database"]["port"] == 8080
        assert isinstance(config_manager.config["database"]["port"], int)
        assert config_manager.config["features"]["enable_caching"] is False
        assert isinstance(config_manager.config["features"]["enable_caching"], bool)
        assert config_manager.config["features"]["max_cache_size"] == 2000
        assert isinstance(config_manager.config["features"]["max_cache_size"], int)

    def test_get_config_value(self, config_manager, temp_dir, sample_yaml_config):
        """Test getting configuration values using dot notation."""
        yaml_file = temp_dir / "test_config.yaml"
        yaml_file.write_text(sample_yaml_config)
        config_manager.load_config(str(yaml_file))

        # Test getting nested values
        assert config_manager.get("database.host") == "localhost"
        assert config_manager.get("database.port") == 5432
        assert config_manager.get("api.base_url") == "https://api.example.com"
        assert config_manager.get("features.enable_caching") is True

    def test_get_config_value_with_default(
        self, config_manager, temp_dir, sample_yaml_config
    ):
        """Test getting configuration values with default fallback."""
        yaml_file = temp_dir / "test_config.yaml"
        yaml_file.write_text(sample_yaml_config)
        config_manager.load_config(str(yaml_file))

        # Test getting non-existent key with default
        assert config_manager.get("nonexistent.key", "default_value") == "default_value"
        assert config_manager.get("database.nonexistent", 42) == 42

    def test_get_config_value_nonexistent_without_default(
        self, config_manager, temp_dir, sample_yaml_config
    ):
        """Test that getting non-existent config value without default raises KeyError."""
        yaml_file = temp_dir / "test_config.yaml"
        yaml_file.write_text(sample_yaml_config)
        config_manager.load_config(str(yaml_file))

        with pytest.raises(KeyError):
            config_manager.get("nonexistent.key")

    def test_config_validation_required_fields(self, config_manager):
        """Test configuration validation for required fields."""
        # Test with incomplete config missing required fields
        incomplete_config = {
            "database": {
                "host": "localhost"
                # Missing required fields like port, name, etc.
            }
        }

        with pytest.raises(ConfigError) as exc_info:
            config_manager.validate_config(incomplete_config)

        assert (
            "required" in str(exc_info.value).lower()
            or "missing" in str(exc_info.value).lower()
        )

    def test_config_validation_type_checking(self, config_manager):
        """Test configuration validation for type checking."""
        # Test with config having wrong types
        invalid_config = {
            "database": {
                "host": "localhost",
                "port": "not_a_number",  # Should be int
                "name": "test_db",
                "username": "test_user",
                "password": "test_pass",
            }
        }

        with pytest.raises(ConfigError) as exc_info:
            config_manager.validate_config(invalid_config)

        assert (
            "type" in str(exc_info.value).lower()
            or "invalid" in str(exc_info.value).lower()
        )

    def test_merge_configs(self, config_manager):
        """Test merging multiple configuration dictionaries."""
        base_config = {
            "database": {"host": "localhost", "port": 5432, "name": "base_db"},
            "api": {"timeout": 30},
        }

        override_config = {
            "database": {
                "port": 8080,  # Override existing
                "username": "new_user",  # Add new field
            },
            "logging": {"level": "DEBUG"},  # Add new section
        }

        merged = config_manager.merge_configs(base_config, override_config)

        # Verify merge results
        assert merged["database"]["host"] == "localhost"  # Unchanged
        assert merged["database"]["port"] == 8080  # Overridden
        assert merged["database"]["name"] == "base_db"  # Unchanged
        assert merged["database"]["username"] == "new_user"  # Added
        assert merged["api"]["timeout"] == 30  # Unchanged
        assert merged["logging"]["level"] == "DEBUG"  # Added

    def test_load_multiple_config_sources(
        self, config_manager, temp_dir, sample_yaml_config, sample_json_config
    ):
        """Test loading and merging multiple configuration sources."""
        # Create YAML file
        yaml_file = temp_dir / "base_config.yaml"
        yaml_file.write_text(sample_yaml_config)

        # Create JSON override file
        override_config = {
            "database": {"port": 9999, "ssl": True},
            "new_section": {"key": "value"},
        }
        json_file = temp_dir / "override_config.json"
        json_file.write_text(json.dumps(override_config, indent=2))

        # Load multiple configs
        config_manager.load_configs([str(yaml_file), str(json_file)])

        # Verify merged configuration
        assert config_manager.config["database"]["host"] == "localhost"  # From YAML
        assert config_manager.config["database"]["port"] == 9999  # Overridden by JSON
        assert config_manager.config["database"]["ssl"] is True  # Added by JSON
        assert config_manager.config["new_section"]["key"] == "value"  # From JSON

    def test_config_schema_validation(self, config_manager):
        """Test configuration against a predefined schema."""
        # Define a simple schema
        schema = {
            "database": {
                "required": ["host", "port", "name"],
                "types": {"host": str, "port": int, "name": str},
            }
        }

        # Valid config
        valid_config = {
            "database": {"host": "localhost", "port": 5432, "name": "test_db"}
        }

        # Should not raise exception
        config_manager.validate_config_schema(valid_config, schema)

        # Invalid config (missing required field)
        invalid_config = {
            "database": {
                "host": "localhost",
                "port": 5432
                # Missing 'name'
            }
        }

        with pytest.raises(ConfigError):
            config_manager.validate_config_schema(invalid_config, schema)

    def test_config_interpolation(self, config_manager, temp_dir):
        """Test configuration value interpolation/substitution."""
        config_with_interpolation = """
base_url: https://api.example.com
api:
  endpoint: ${base_url}/v1/data
  auth_url: ${base_url}/auth
database:
  connection_string: postgresql://${database.username}:${database.password}@${database.host}:${database.port}/${database.name}
  host: localhost
  port: 5432
  name: test_db
  username: test_user
  password: test_pass
"""

        yaml_file = temp_dir / "interpolation_config.yaml"
        yaml_file.write_text(config_with_interpolation)

        config_manager.load_config(str(yaml_file))

        # Verify interpolated values
        assert (
            config_manager.config["api"]["endpoint"]
            == "https://api.example.com/v1/data"
        )
        assert (
            config_manager.config["api"]["auth_url"] == "https://api.example.com/auth"
        )
        expected_conn_str = "postgresql://test_user:test_pass@localhost:5432/test_db"
        assert (
            config_manager.config["database"]["connection_string"] == expected_conn_str
        )

    def test_config_export(self, config_manager, temp_dir, sample_yaml_config):
        """Test exporting configuration to file."""
        # Load config
        yaml_file = temp_dir / "test_config.yaml"
        yaml_file.write_text(sample_yaml_config)
        config_manager.load_config(str(yaml_file))

        # Export to YAML
        export_yaml = temp_dir / "exported_config.yaml"
        config_manager.export_config(str(export_yaml), format="yaml")
        assert export_yaml.exists()

        # Export to JSON
        export_json = temp_dir / "exported_config.json"
        config_manager.export_config(str(export_json), format="json")
        assert export_json.exists()

        # Verify exported content can be loaded back
        new_manager = ConfigManager()
        new_manager.load_config(str(export_yaml))
        assert new_manager.config == config_manager.config

    def test_config_reload(self, config_manager, temp_dir, sample_yaml_config):
        """Test reloading configuration from file."""
        yaml_file = temp_dir / "test_config.yaml"
        yaml_file.write_text(sample_yaml_config)
        config_manager.load_config(str(yaml_file))

        original_port = config_manager.config["database"]["port"]

        # Modify the file
        modified_config = sample_yaml_config.replace("port: 5432", "port: 8080")
        yaml_file.write_text(modified_config)

        # Reload
        config_manager.reload_config()

        # Verify changes
        assert config_manager.config["database"]["port"] == 8080
        assert config_manager.config["database"]["port"] != original_port

    def test_config_watch_mode(self, config_manager, temp_dir, sample_yaml_config):
        """Test configuration file watching for automatic reloads."""
        yaml_file = temp_dir / "test_config.yaml"
        yaml_file.write_text(sample_yaml_config)

        # Enable watch mode
        config_manager.enable_watch_mode(str(yaml_file))

        # Verify watch mode is enabled
        assert config_manager.is_watching() is True

        # Disable watch mode
        config_manager.disable_watch_mode()
        assert config_manager.is_watching() is False

    @pytest.mark.parametrize(
        "file_extension,content_type",
        [
            (".yaml", "yaml"),
            (".yml", "yaml"),
            (".json", "json"),
        ],
    )
    def test_file_format_detection(
        self, config_manager, temp_dir, file_extension, content_type
    ):
        """Test automatic file format detection based on extension."""
        if content_type == "yaml":
            content = "test: value\nnumber: 42"
        else:  # json
            content = '{"test": "value", "number": 42}'

        config_file = temp_dir / f"test_config{file_extension}"
        config_file.write_text(content)

        config_manager.load_config(str(config_file))

        assert config_manager.config["test"] == "value"
        assert config_manager.config["number"] == 42

    def test_config_backup_and_restore(
        self, config_manager, temp_dir, sample_yaml_config
    ):
        """Test configuration backup and restore functionality."""
        yaml_file = temp_dir / "test_config.yaml"
        yaml_file.write_text(sample_yaml_config)
        config_manager.load_config(str(yaml_file))

        # Create backup
        backup_id = config_manager.create_backup()
        assert backup_id is not None

        # Modify config
        config_manager.config["database"]["port"] = 9999

        # Restore from backup
        config_manager.restore_backup(backup_id)

        # Verify restoration
        assert config_manager.config["database"]["port"] == 5432

    def test_environment_prefix_configuration(
        self, config_manager, temp_dir, sample_yaml_config
    ):
        """Test configuring custom environment variable prefix."""
        yaml_file = temp_dir / "test_config.yaml"
        yaml_file.write_text(sample_yaml_config)

        # Set custom prefix
        config_manager.set_env_prefix("CUSTOM_PREFIX")

        env_vars = {
            "CUSTOM_PREFIX_DATABASE_HOST": "custom-host",
            "CUSTOM_PREFIX_API_TIMEOUT": "120",
        }

        with patch.dict(os.environ, env_vars):
            config_manager.load_config(str(yaml_file))

        assert config_manager.config["database"]["host"] == "custom-host"
        assert config_manager.config["api"]["timeout"] == 120

    def test_config_encryption_support(self, config_manager, temp_dir):
        """Test support for encrypted configuration values."""
        encrypted_config = """
database:
  host: localhost
  port: 5432
  password: "encrypted:AES256:base64encodedciphertext"
api:
  secret_key: "encrypted:AES256:anotherciphertext"
"""

        yaml_file = temp_dir / "encrypted_config.yaml"
        yaml_file.write_text(encrypted_config)

        # Mock decryption
        with patch.object(config_manager, "decrypt_value") as mock_decrypt:
            mock_decrypt.side_effect = lambda x: x.replace(
                "encrypted:AES256:", "decrypted_"
            )
            config_manager.load_config(str(yaml_file))

        assert (
            config_manager.config["database"]["password"]
            == "decrypted_base64encodedciphertext"
        )
        assert (
            config_manager.config["api"]["secret_key"] == "decrypted_anotherciphertext"
        )

    # ===== Enhanced File Watching Tests =====

    def test_enable_watchdog_file_watching(self, config_manager, temp_dir, sample_yaml_config):
        """Test enabling watchdog-based file watching for a single file."""
        yaml_file = temp_dir / "test_config.yaml"
        yaml_file.write_text(sample_yaml_config)
        config_manager.load_config(str(yaml_file))

        # Enable watchdog mode (default)
        config_manager.enable_watch_mode(str(yaml_file), use_legacy=False)

        # Verify watch mode is enabled
        assert config_manager.is_watching() is True
        assert config_manager._observer is not None
        assert config_manager._file_watcher is not None
        assert config_manager._observer.is_alive() is True

        # Verify the file is being watched
        watched_paths = config_manager.get_watched_paths()
        assert str(yaml_file.resolve()) in watched_paths

        # Clean up
        config_manager.disable_watch_mode()

    def test_enable_legacy_file_watching(self, config_manager, temp_dir, sample_yaml_config):
        """Test enabling legacy threading-based file watching."""
        yaml_file = temp_dir / "test_config.yaml"
        yaml_file.write_text(sample_yaml_config)
        config_manager.load_config(str(yaml_file))

        # Enable legacy mode explicitly
        config_manager.enable_watch_mode(str(yaml_file), use_legacy=True)

        # Verify watch mode is enabled with legacy components
        assert config_manager.is_watching() is True
        assert config_manager._watch_thread is not None
        assert config_manager._watch_thread.is_alive() is True
        assert config_manager._observer is None  # Should not use watchdog

        # Verify the file is in watchers dict
        assert str(yaml_file) in config_manager.watchers

        # Clean up
        config_manager.disable_watch_mode()

    def test_watch_config_directory(self, config_manager, temp_dir, sample_yaml_config):
        """Test watching an entire directory for config file changes."""
        # Create multiple config files in directory
        yaml_file1 = temp_dir / "config1.yaml"
        yaml_file1.write_text(sample_yaml_config)
        yaml_file2 = temp_dir / "config2.yml"
        yaml_file2.write_text(sample_yaml_config)
        json_file = temp_dir / "config.json"
        json_file.write_text('{"test": "value"}')

        # Watch the entire directory
        config_manager.watch_config_directory(str(temp_dir))

        # Verify watching is enabled
        assert config_manager.is_watching() is True
        assert config_manager._observer is not None
        assert config_manager._observer.is_alive() is True

        # Verify directory is being watched
        watched_paths = config_manager.get_watched_paths()
        assert str(temp_dir.resolve()) in watched_paths

        config_manager.disable_watch_mode()

    def test_watch_config_directory_recursive(self, config_manager, temp_dir, sample_yaml_config):
        """Test recursive directory watching."""
        # Create nested directory structure
        subdir = temp_dir / "configs" / "subdirectory"
        subdir.mkdir(parents=True)
        
        config_file = subdir / "nested_config.yaml"
        config_file.write_text(sample_yaml_config)

        # Watch directory recursively
        config_manager.watch_config_directory(str(temp_dir), recursive=True)

        # Verify recursive watching is set up
        assert config_manager.is_watching() is True
        assert f"{str(temp_dir)}:recursive" in config_manager._watched_directories

        config_manager.disable_watch_mode()

    def test_add_watch_path_multiple_files(self, config_manager, temp_dir, sample_yaml_config):
        """Test adding multiple files to watch list."""
        # Create multiple config files
        yaml_file1 = temp_dir / "config1.yaml"
        yaml_file1.write_text(sample_yaml_config)
        yaml_file2 = temp_dir / "config2.yaml"
        yaml_file2.write_text(sample_yaml_config)

        # Start watching with one file
        config_manager.enable_watch_mode(str(yaml_file1))

        # Add second file to watch list
        config_manager.add_watch_path(str(yaml_file2))

        # Verify both files are being watched
        watched_paths = config_manager.get_watched_paths()
        assert str(yaml_file1.resolve()) in watched_paths
        assert str(yaml_file2.resolve()) in watched_paths

        config_manager.disable_watch_mode()

    def test_add_watch_path_directory(self, config_manager, temp_dir, sample_yaml_config):
        """Test adding a directory to watch list."""
        yaml_file = temp_dir / "config.yaml"
        yaml_file.write_text(sample_yaml_config)
        
        # Create separate directory
        config_dir = temp_dir / "configs"
        config_dir.mkdir()
        dir_config = config_dir / "dir_config.yaml"
        dir_config.write_text(sample_yaml_config)

        # Start watching file
        config_manager.enable_watch_mode(str(yaml_file))

        # Add directory to watch list
        config_manager.add_watch_path(str(config_dir))

        # Verify both file and directory are watched
        watched_paths = config_manager.get_watched_paths()
        assert str(yaml_file.resolve()) in watched_paths
        assert str(config_dir.resolve()) in watched_paths

        config_manager.disable_watch_mode()

    def test_remove_watch_path(self, config_manager, temp_dir, sample_yaml_config):
        """Test removing paths from watch list."""
        yaml_file1 = temp_dir / "config1.yaml"
        yaml_file1.write_text(sample_yaml_config)
        yaml_file2 = temp_dir / "config2.yaml"
        yaml_file2.write_text(sample_yaml_config)

        # Start watching with multiple files
        config_manager.enable_watch_mode(str(yaml_file1))
        config_manager.add_watch_path(str(yaml_file2))

        # Verify both are watched
        watched_paths = config_manager.get_watched_paths()
        assert len(watched_paths) == 2

        # Remove one file
        config_manager.remove_watch_path(str(yaml_file2))

        # Verify only one file remains
        watched_paths = config_manager.get_watched_paths()
        assert str(yaml_file1.resolve()) in watched_paths
        assert str(yaml_file2.resolve()) not in watched_paths

        config_manager.disable_watch_mode()

    def test_set_debounce_delay(self, config_manager, temp_dir, sample_yaml_config):
        """Test setting debounce delay for file change events."""
        yaml_file = temp_dir / "config.yaml"
        yaml_file.write_text(sample_yaml_config)

        # Set debounce delay before enabling watching
        config_manager.set_debounce_delay(1.0)
        assert config_manager._debounce_delay == 1.0

        # Enable watching
        config_manager.enable_watch_mode(str(yaml_file))

        # Verify debounce delay is applied to file watcher
        assert config_manager._file_watcher.debounce_delay == 1.0

        # Change debounce delay while watching
        config_manager.set_debounce_delay(2.0)
        assert config_manager._file_watcher.debounce_delay == 2.0

        config_manager.disable_watch_mode()

    def test_set_debounce_delay_invalid_values(self, config_manager):
        """Test setting invalid debounce delay values."""
        # Test too small value
        with pytest.raises(ConfigError) as exc_info:
            config_manager.set_debounce_delay(0.05)
        assert "between 0.1 and 5.0" in str(exc_info.value)

        # Test too large value
        with pytest.raises(ConfigError) as exc_info:
            config_manager.set_debounce_delay(10.0)
        assert "between 0.1 and 5.0" in str(exc_info.value)

    def test_get_watched_paths_empty(self, config_manager):
        """Test getting watched paths when nothing is being watched."""
        watched_paths = config_manager.get_watched_paths()
        assert watched_paths == []

    def test_get_watched_paths_legacy_fallback(self, config_manager, temp_dir, sample_yaml_config):
        """Test getting watched paths with legacy watcher fallback."""
        yaml_file = temp_dir / "config.yaml"
        yaml_file.write_text(sample_yaml_config)

        # Enable legacy watching
        config_manager.enable_watch_mode(str(yaml_file), use_legacy=True)

        # Should fallback to legacy watchers dict
        watched_paths = config_manager.get_watched_paths()
        assert str(yaml_file) in watched_paths

        config_manager.disable_watch_mode()

    def test_file_watcher_debouncing(self, config_manager, temp_dir, sample_yaml_config):
        """Test debouncing mechanism prevents rapid successive reloads."""
        yaml_file = temp_dir / "config.yaml"
        yaml_file.write_text(sample_yaml_config)
        config_manager.load_config(str(yaml_file))

        # Mock reload_config to track calls
        original_reload = config_manager.reload_config
        reload_calls = []
        
        def mock_reload():
            reload_calls.append(time.time())
            original_reload()
        
        config_manager.reload_config = mock_reload

        # Set short debounce delay for testing
        config_manager.set_debounce_delay(0.1)
        config_manager.enable_watch_mode(str(yaml_file))

        # Simulate rapid file changes
        file_watcher = config_manager._file_watcher
        file_path = str(yaml_file.resolve())
        
        # Add the file to watched paths
        file_watcher.add_watched_path(file_path)

        # Trigger multiple rapid changes
        start_time = time.time()
        for i in range(5):
            file_watcher._handle_file_change(file_path, "modified")
            time.sleep(0.02)  # Very short interval

        # Wait for debounce period plus buffer
        time.sleep(0.2)

        # Should have only one reload call due to debouncing
        assert len(reload_calls) <= 1, f"Expected at most 1 reload call, got {len(reload_calls)}"

        config_manager.disable_watch_mode()

    def test_watchdog_observer_error_handling(self, config_manager, temp_dir, sample_yaml_config):
        """Test error handling when watchdog observer fails."""
        yaml_file = temp_dir / "config.yaml"
        yaml_file.write_text(sample_yaml_config)

        # Patch the Observer after config manager creates it
        with patch.object(config_manager, '_observer', None):
            # Mock observer that fails to start
            mock_observer = Mock()
            mock_observer.start.side_effect = Exception("Observer failed to start")
            mock_observer.is_alive.return_value = False
            
            config_manager._observer = mock_observer
            
            # Manually trigger the error path by calling _enable_watchdog_mode
            with pytest.raises(ConfigError) as exc_info:
                config_manager._enable_watchdog_mode(str(yaml_file))
            
            assert "Failed to enable watchdog mode" in str(exc_info.value)

    def test_add_watch_path_without_watching_enabled(self, config_manager, temp_dir, sample_yaml_config):
        """Test adding watch path when watching is not enabled."""
        yaml_file = temp_dir / "config.yaml"
        yaml_file.write_text(sample_yaml_config)

        # Try to add watch path without enabling watching first
        with pytest.raises(ConfigError) as exc_info:
            config_manager.add_watch_path(str(yaml_file))
        
        assert "watching is not enabled" in str(exc_info.value)

    def test_add_watch_path_nonexistent_file(self, config_manager, temp_dir, sample_yaml_config):
        """Test adding non-existent file to watch list."""
        yaml_file = temp_dir / "config.yaml"
        yaml_file.write_text(sample_yaml_config)
        nonexistent_file = temp_dir / "nonexistent.yaml"

        # Enable watching
        config_manager.enable_watch_mode(str(yaml_file))

        # Try to add non-existent file
        with pytest.raises(ConfigError) as exc_info:
            config_manager.add_watch_path(str(nonexistent_file))
        
        assert "Path does not exist" in str(exc_info.value)

        config_manager.disable_watch_mode()

    def test_disable_watch_mode_cleanup(self, config_manager, temp_dir, sample_yaml_config):
        """Test that disable_watch_mode properly cleans up all resources."""
        yaml_file = temp_dir / "config.yaml"
        yaml_file.write_text(sample_yaml_config)

        # Enable watching
        config_manager.enable_watch_mode(str(yaml_file))
        
        # Verify resources are created
        assert config_manager._observer is not None
        assert config_manager._file_watcher is not None
        assert len(config_manager._watched_directories) > 0
        assert len(config_manager.watchers) > 0

        # Disable watching
        config_manager.disable_watch_mode()

        # Verify complete cleanup
        assert config_manager.is_watching() is False
        assert config_manager._observer is None
        assert config_manager._file_watcher is None
        assert len(config_manager._watched_directories) == 0
        assert len(config_manager.watchers) == 0

    def test_disable_watch_mode_observer_error(self, config_manager, temp_dir, sample_yaml_config):
        """Test disable_watch_mode handles observer stop errors gracefully."""
        yaml_file = temp_dir / "config.yaml"
        yaml_file.write_text(sample_yaml_config)

        # Enable watching
        config_manager.enable_watch_mode(str(yaml_file))

        # Mock observer to raise exception on stop
        config_manager._observer.stop = Mock(side_effect=Exception("Stop failed"))
        config_manager._observer.join = Mock()

        # Disable watching should not raise exception
        config_manager.disable_watch_mode()
        
        # Should still clean up other resources
        assert config_manager.is_watching() is False
        assert config_manager._observer is None

    def test_hot_reload_file_modification(self, config_manager, temp_dir, sample_yaml_config):
        """Test that file modifications trigger config reload."""
        yaml_file = temp_dir / "config.yaml"
        yaml_file.write_text(sample_yaml_config)
        config_manager.load_config(str(yaml_file))

        original_port = config_manager.config["database"]["port"]

        # Track reload calls
        reload_called = threading.Event()
        original_reload = config_manager.reload_config
        
        def mock_reload():
            original_reload()
            reload_called.set()
        
        config_manager.reload_config = mock_reload

        # Enable watching with short debounce
        config_manager.set_debounce_delay(0.1)
        config_manager.enable_watch_mode(str(yaml_file))

        # Modify config file
        modified_config = sample_yaml_config.replace("port: 5432", "port: 8080")
        yaml_file.write_text(modified_config)

        # Trigger file modification event manually
        if config_manager._file_watcher:
            config_manager._file_watcher._handle_file_change(str(yaml_file), "modified")

        # Wait for reload to complete
        assert reload_called.wait(timeout=1.0), "Config reload was not triggered"

        # Verify config was reloaded
        assert config_manager.config["database"]["port"] == 8080

        config_manager.disable_watch_mode()

    def test_config_file_watcher_is_config_file(self):
        """Test ConfigFileWatcher correctly identifies config files."""
        watcher = ConfigFileWatcher(None, 0.5)

        # Test valid config file extensions
        assert watcher._is_config_file("/path/to/config.yaml") is True
        assert watcher._is_config_file("/path/to/config.yml") is True
        assert watcher._is_config_file("/path/to/config.json") is True
        assert watcher._is_config_file("/path/to/Config.YAML") is True  # Case insensitive

        # Test invalid extensions
        assert watcher._is_config_file("/path/to/config.txt") is False
        assert watcher._is_config_file("/path/to/config.py") is False
        assert watcher._is_config_file("/path/to/config") is False

    def test_config_file_watcher_add_remove_paths(self):
        """Test ConfigFileWatcher path management."""
        watcher = ConfigFileWatcher(None, 0.5)

        # Add watched paths
        watcher.add_watched_path("/path/to/config.yaml")
        watcher.add_watched_path("/another/config.json")

        assert len(watcher.watched_paths) == 2

        # Remove watched path
        watcher.remove_watched_path("/path/to/config.yaml")
        assert len(watcher.watched_paths) == 1

        # Remove non-existent path (should not error)
        watcher.remove_watched_path("/nonexistent/path")
        assert len(watcher.watched_paths) == 1

    @patch('aim2_project.aim2_utils.config_manager.time.time')
    def test_config_file_watcher_debouncing_logic(self, mock_time):
        """Test ConfigFileWatcher debouncing logic in detail."""
        mock_config_manager = Mock()
        mock_config_manager.reload_config = Mock()
        
        watcher = ConfigFileWatcher(mock_config_manager, 0.5)
        test_path = "/path/to/config.yaml"
        
        # Mock time progression
        current_time = 1000.0
        mock_time.return_value = current_time
        
        # Add path to watched paths
        watcher.add_watched_path(test_path)

        # First change
        watcher._handle_file_change(test_path, "modified")
        assert test_path in watcher.pending_reloads

        # Second change within debounce period
        current_time += 0.1
        mock_time.return_value = current_time
        watcher._handle_file_change(test_path, "modified")

        # Should update pending reload time
        assert watcher.pending_reloads[test_path] == current_time

    def test_enable_watch_mode_switches_from_legacy_to_watchdog(self, config_manager, temp_dir, sample_yaml_config):
        """Test switching from legacy to watchdog mode."""
        yaml_file = temp_dir / "config.yaml"
        yaml_file.write_text(sample_yaml_config)

        # First enable legacy mode
        config_manager.enable_watch_mode(str(yaml_file), use_legacy=True)
        assert config_manager._watch_thread is not None
        assert config_manager._observer is None

        # Then switch to watchdog mode
        config_manager.enable_watch_mode(str(yaml_file), use_legacy=False)
        assert config_manager._observer is not None
        assert config_manager._file_watcher is not None

        config_manager.disable_watch_mode()

    def test_enable_watch_mode_nonexistent_path(self, config_manager, temp_dir):
        """Test enabling watch mode with non-existent path."""
        nonexistent_file = temp_dir / "nonexistent.yaml"

        # Should raise ConfigError for non-existent path
        with pytest.raises(ConfigError) as exc_info:
            config_manager.enable_watch_mode(str(nonexistent_file))
        
        assert "Path does not exist" in str(exc_info.value)

    def test_watch_mode_backward_compatibility(self, config_manager, temp_dir, sample_yaml_config):
        """Test that watchers dict is maintained for backward compatibility."""
        yaml_file = temp_dir / "config.yaml"
        yaml_file.write_text(sample_yaml_config)

        # Enable watchdog mode
        config_manager.enable_watch_mode(str(yaml_file))

        # Verify watchers dict is populated for backward compatibility
        assert str(yaml_file) in config_manager.watchers
        assert config_manager.watchers[str(yaml_file)]["path"] == str(yaml_file)
        assert config_manager.watchers[str(yaml_file)]["type"] == "watchdog"

        config_manager.disable_watch_mode()

    def test_file_watcher_handles_directory_events(self, config_manager, temp_dir, sample_yaml_config):
        """Test that ConfigFileWatcher handles events from watched directories."""
        config_dir = temp_dir / "configs"
        config_dir.mkdir()
        config_file = config_dir / "new_config.yaml"

        # Start watching directory
        config_manager.watch_config_directory(str(config_dir))
        
        # Simulate creating a new config file in the directory
        config_file.write_text(sample_yaml_config)
        
        # The file watcher should recognize this as a config file change
        file_watcher = config_manager._file_watcher
        assert file_watcher._is_config_file(str(config_file)) is True

        config_manager.disable_watch_mode()

    def test_multiple_observer_schedules_same_directory(self, config_manager, temp_dir, sample_yaml_config):
        """Test that adding multiple files in same directory doesn't create duplicate observers."""
        # Create multiple files in same directory
        config1 = temp_dir / "config1.yaml"
        config2 = temp_dir / "config2.yaml"
        config1.write_text(sample_yaml_config)
        config2.write_text(sample_yaml_config)

        # Enable watching for first file
        config_manager.enable_watch_mode(str(config1))
        initial_dir_count = len(config_manager._watched_directories)

        # Add second file in same directory
        config_manager.add_watch_path(str(config2))
        
        # Should not add duplicate directory to observer
        assert len(config_manager._watched_directories) == initial_dir_count

        config_manager.disable_watch_mode()

    def test_config_file_watcher_error_handling(self):
        """Test ConfigFileWatcher error handling in file change processing."""
        mock_config_manager = Mock()
        mock_config_manager.reload_config = Mock(side_effect=Exception("Reload failed"))
        
        watcher = ConfigFileWatcher(mock_config_manager, 0.1)
        
        # Should not raise exception even if reload fails
        watcher._debounced_reload("/path/to/config", "modified", time.time())
        
        # Reload should have been attempted
        mock_config_manager.reload_config.assert_called_once()

    def test_legacy_watch_mode_file_modification_detection(self, config_manager, temp_dir, sample_yaml_config):
        """Test legacy watch mode detects file modifications."""
        yaml_file = temp_dir / "config.yaml"
        yaml_file.write_text(sample_yaml_config)
        config_manager.load_config(str(yaml_file))

        # Enable legacy watching
        config_manager.enable_watch_mode(str(yaml_file), use_legacy=True)
        
        # Give the watch thread a moment to start
        time.sleep(0.1)

        original_mtime = config_manager.watchers[str(yaml_file)]["last_modified"]

        # Modify file to update mtime
        time.sleep(0.1)  # Ensure mtime difference
        yaml_file.touch()
        
        # Give watch thread time to detect change
        time.sleep(1.1)  # Slightly longer than watch interval

        # Check that mtime was updated (indicates change was detected)
        new_mtime = config_manager.watchers[str(yaml_file)]["last_modified"]
        assert new_mtime > original_mtime

        config_manager.disable_watch_mode()


class TestConfigError:
    """Test suite for ConfigError exception class."""

    def test_config_error_instantiation(self):
        """Test ConfigError can be instantiated with message."""
        error = ConfigError("Test error message")
        assert str(error) == "Test error message"

    def test_config_error_inheritance(self):
        """Test ConfigError inherits from Exception."""
        error = ConfigError("Test error")
        assert isinstance(error, Exception)

    def test_config_error_with_nested_exception(self):
        """Test ConfigError can wrap other exceptions."""
        original_error = ValueError("Original error")
        config_error = ConfigError("Config loading failed", original_error)

        assert "Config loading failed" in str(config_error)
        assert hasattr(config_error, "__cause__") or hasattr(config_error, "args")


class TestConfigFileWatcher:
    """Test suite for ConfigFileWatcher class."""

    def test_config_file_watcher_initialization(self):
        """Test ConfigFileWatcher initialization."""
        mock_manager = Mock()
        watcher = ConfigFileWatcher(mock_manager, 1.0)
        
        assert watcher.config_manager is mock_manager
        assert watcher.debounce_delay == 1.0
        assert len(watcher.watched_paths) == 0
        assert len(watcher.pending_reloads) == 0

    def test_config_file_watcher_on_modified_event(self):
        """Test ConfigFileWatcher handles file modification events."""
        mock_manager = Mock()
        watcher = ConfigFileWatcher(mock_manager, 0.1)
        
        # Create mock file modification event
        mock_event = Mock()
        mock_event.is_directory = False
        mock_event.src_path = "/test/config.yaml"
        
        # Add path to watched paths
        watcher.add_watched_path("/test/config.yaml")
        
        # Mock the _handle_file_change method to verify it's called
        watcher._handle_file_change = Mock()
        
        # Trigger event
        watcher.on_modified(mock_event)
        
        # Verify handler was called
        watcher._handle_file_change.assert_called_once_with("/test/config.yaml", "modified")

    def test_config_file_watcher_on_created_event(self):
        """Test ConfigFileWatcher handles file creation events."""
        mock_manager = Mock()
        watcher = ConfigFileWatcher(mock_manager, 0.1)
        
        # Create mock file creation event
        mock_event = Mock()
        mock_event.is_directory = False
        mock_event.src_path = "/test/new_config.yaml"
        
        # Mock the _handle_file_change method to verify it's called
        watcher._handle_file_change = Mock()
        
        # Trigger event
        watcher.on_created(mock_event)
        
        # Verify handler was called
        watcher._handle_file_change.assert_called_once_with("/test/new_config.yaml", "created")

    def test_config_file_watcher_ignores_directory_events(self):
        """Test ConfigFileWatcher ignores directory events."""
        mock_manager = Mock()
        watcher = ConfigFileWatcher(mock_manager, 0.1)
        
        # Create mock directory event
        mock_event = Mock()
        mock_event.is_directory = True
        mock_event.src_path = "/test/directory"
        
        # Mock the _handle_file_change method
        watcher._handle_file_change = Mock()
        
        # Trigger events
        watcher.on_modified(mock_event)
        watcher.on_created(mock_event)
        
        # Verify handler was not called
        watcher._handle_file_change.assert_not_called()

    @patch('aim2_project.aim2_utils.config_manager.threading.Timer')
    def test_config_file_watcher_debounce_timer_creation(self, mock_timer):
        """Test ConfigFileWatcher creates debounce timer correctly."""
        mock_manager = Mock()
        watcher = ConfigFileWatcher(mock_manager, 0.5)
        
        test_path = "/test/config.yaml"
        watcher.add_watched_path(test_path)
        
        # Trigger file change
        watcher._handle_file_change(test_path, "modified")
        
        # Verify timer was created with correct parameters
        mock_timer.assert_called_once()
        args, kwargs = mock_timer.call_args
        assert args[0] == 0.5  # debounce delay
        assert args[1] == watcher._debounced_reload  # callback function

    def test_config_file_watcher_handle_file_change_not_watched(self):
        """Test ConfigFileWatcher ignores changes to non-watched files."""
        mock_manager = Mock()
        watcher = ConfigFileWatcher(mock_manager, 0.1)
        
        # Don't add the file to watched paths
        test_path = "/test/unwatched.yaml"
        
        # Should not create any pending reloads for unwatched files
        watcher._handle_file_change(test_path, "modified")
        
        assert len(watcher.pending_reloads) == 0

    def test_config_file_watcher_handle_file_change_non_config_file(self):
        """Test ConfigFileWatcher ignores non-config files."""
        mock_manager = Mock()
        watcher = ConfigFileWatcher(mock_manager, 0.1)
        
        # Test with non-config file extension
        test_path = "/test/document.txt"
        
        # Should not create any pending reloads for non-config files
        watcher._handle_file_change(test_path, "modified")
        
        assert len(watcher.pending_reloads) == 0

    def test_config_file_watcher_handle_file_change_path_error(self):
        """Test ConfigFileWatcher handles path resolution errors gracefully."""
        mock_manager = Mock()
        watcher = ConfigFileWatcher(mock_manager, 0.1)
        
        test_path = "/test/config.yaml"
        watcher.add_watched_path(test_path)
        
        # Mock Path.resolve to raise exception during _handle_file_change
        with patch('pathlib.Path.resolve', side_effect=Exception("Path error")):
            # Should not raise exception even with path errors
            watcher._handle_file_change(test_path, "modified")
            
            # Should not create pending reloads due to error
            assert len(watcher.pending_reloads) == 0

    def test_config_file_watcher_debounced_reload_success(self):
        """Test ConfigFileWatcher debounced reload executes successfully."""
        mock_manager = Mock()
        mock_manager.reload_config = Mock()
        
        watcher = ConfigFileWatcher(mock_manager, 0.1)
        test_path = "/test/config.yaml"
        change_time = time.time()
        
        # Add to pending reloads
        watcher.pending_reloads[test_path] = change_time
        
        # Execute debounced reload
        watcher._debounced_reload(test_path, "modified", change_time)
        
        # Verify reload was called and pending reload was cleared
        mock_manager.reload_config.assert_called_once()
        assert test_path not in watcher.pending_reloads

    def test_config_file_watcher_debounced_reload_stale_change(self):
        """Test ConfigFileWatcher skips stale debounced reloads."""
        mock_manager = Mock()
        mock_manager.reload_config = Mock()
        
        watcher = ConfigFileWatcher(mock_manager, 0.1)
        test_path = "/test/config.yaml"
        old_change_time = time.time()
        new_change_time = old_change_time + 1.0
        
        # Set newer change time in pending reloads
        watcher.pending_reloads[test_path] = new_change_time
        
        # Execute debounced reload with older change time
        watcher._debounced_reload(test_path, "modified", old_change_time)
        
        # Should not reload due to stale change
        mock_manager.reload_config.assert_not_called()
        assert test_path in watcher.pending_reloads  # Should remain

    def test_config_file_watcher_debounced_reload_manager_missing_method(self):
        """Test ConfigFileWatcher handles missing reload_config method."""
        mock_manager = Mock()
        # Don't set reload_config method
        delattr(mock_manager, 'reload_config')
        
        watcher = ConfigFileWatcher(mock_manager, 0.1)
        test_path = "/test/config.yaml"
        change_time = time.time()
        
        # Add to pending reloads
        watcher.pending_reloads[test_path] = change_time
        
        # Execute debounced reload - should not raise exception
        watcher._debounced_reload(test_path, "modified", change_time)
        
        # Should still clear pending reload
        assert test_path not in watcher.pending_reloads


class TestConfigManagerEdgeCases:
    """Test suite for ConfigManager edge cases and error conditions."""

    @pytest.fixture
    def config_manager(self):
        """Create a ConfigManager instance for testing."""
        return ConfigManager()

    @pytest.fixture
    def temp_dir(self):
        """Create a temporary directory for test files."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            yield Path(tmp_dir)

    @pytest.fixture
    def sample_yaml_config(self):
        """Sample YAML configuration for testing."""
        return """
# Sample configuration file
database:
  host: localhost
  port: 5432
  name: test_db
  username: test_user
  password: test_pass

api:
  base_url: https://api.example.com
  timeout: 30
  retry_attempts: 3

logging:
  level: INFO
  format: "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
  handlers:
    - console
    - file

features:
  enable_caching: true
  max_cache_size: 1000
  debug_mode: false

nlp:
  model_name: "bert-base-uncased"
  max_sequence_length: 512
  batch_size: 32
"""

    def test_watch_config_directory_nonexistent_directory(self, config_manager, temp_dir):
        """Test watching non-existent directory raises error."""
        nonexistent_dir = temp_dir / "nonexistent"
        
        with pytest.raises(ConfigError) as exc_info:
            config_manager.watch_config_directory(str(nonexistent_dir))
        
        assert "Directory does not exist" in str(exc_info.value)

    def test_enable_watch_mode_already_watching(self, config_manager, temp_dir, sample_yaml_config):
        """Test enabling watch mode when already watching disables previous watcher."""
        yaml_file1 = temp_dir / "config1.yaml"
        yaml_file1.write_text(sample_yaml_config)
        yaml_file2 = temp_dir / "config2.yaml"
        yaml_file2.write_text(sample_yaml_config)

        # Enable watching for first file
        config_manager.enable_watch_mode(str(yaml_file1))
        first_observer = config_manager._observer

        # Enable watching for second file (should disable first)
        config_manager.enable_watch_mode(str(yaml_file2))
        second_observer = config_manager._observer

        # Should have different observer instances
        assert first_observer is not second_observer
        assert config_manager.is_watching() is True

        config_manager.disable_watch_mode()

    def test_remove_watch_path_when_not_watching(self, config_manager):
        """Test removing watch path when not watching does nothing."""
        # Should not raise exception
        config_manager.remove_watch_path("/some/path")
        
        # Should remain not watching
        assert not config_manager.is_watching()

    def test_set_debounce_delay_updates_existing_watcher(self, config_manager, temp_dir, sample_yaml_config):
        """Test setting debounce delay updates existing file watcher."""
        yaml_file = temp_dir / "config.yaml"
        yaml_file.write_text(sample_yaml_config)

        # Enable watching with default delay
        config_manager.enable_watch_mode(str(yaml_file))
        original_delay = config_manager._file_watcher.debounce_delay

        # Change delay
        new_delay = 2.0
        config_manager.set_debounce_delay(new_delay)

        # Verify both manager and watcher are updated
        assert config_manager._debounce_delay == new_delay
        assert config_manager._file_watcher.debounce_delay == new_delay
        assert config_manager._file_watcher.debounce_delay != original_delay

        config_manager.disable_watch_mode()

    def test_get_watched_paths_with_watchdog_and_legacy_paths(self, config_manager, temp_dir, sample_yaml_config):
        """Test get_watched_paths returns correct paths with mixed watching modes."""
        yaml_file = temp_dir / "config.yaml"
        yaml_file.write_text(sample_yaml_config)

        # Enable watchdog mode
        config_manager.enable_watch_mode(str(yaml_file))
        
        # Should return watchdog paths
        watched_paths = config_manager.get_watched_paths()
        assert str(yaml_file.resolve()) in watched_paths

        config_manager.disable_watch_mode()

    def test_enable_watchdog_mode_graceful_failure_handling(self, config_manager, temp_dir, sample_yaml_config):
        """Test watchdog mode handles various failure scenarios gracefully."""
        yaml_file = temp_dir / "config.yaml"
        yaml_file.write_text(sample_yaml_config)

        # Test that normal operation works
        config_manager.enable_watch_mode(str(yaml_file))
        assert config_manager.is_watching()
        config_manager.disable_watch_mode()
        assert not config_manager.is_watching()

    def test_add_watch_path_handles_errors_gracefully(self, config_manager, temp_dir, sample_yaml_config):
        """Test add_watch_path handles various error conditions gracefully."""
        yaml_file1 = temp_dir / "config1.yaml"
        yaml_file1.write_text(sample_yaml_config)
        yaml_file2 = temp_dir / "config2.yaml"
        yaml_file2.write_text(sample_yaml_config)

        # Enable watching for first file
        config_manager.enable_watch_mode(str(yaml_file1))

        # Normal add_watch_path should work
        config_manager.add_watch_path(str(yaml_file2))
        
        # Verify both files are watched
        watched_paths = config_manager.get_watched_paths()
        assert len(watched_paths) >= 2

        config_manager.disable_watch_mode()

    def test_watch_config_directory_with_existing_watching(self, config_manager, temp_dir, sample_yaml_config):
        """Test watch_config_directory adds to existing watch configuration."""
        yaml_file = temp_dir / "config.yaml"
        yaml_file.write_text(sample_yaml_config)
        
        config_dir = temp_dir / "configs"
        config_dir.mkdir()

        # Start watching file first
        config_manager.enable_watch_mode(str(yaml_file))
        initial_paths = len(config_manager.get_watched_paths())

        # Watch directory (should add to existing)
        config_manager.watch_config_directory(str(config_dir))
        
        # Should have more watched paths
        final_paths = len(config_manager.get_watched_paths())
        assert final_paths > initial_paths

        config_manager.disable_watch_mode()

    def test_file_watcher_handles_directory_watching_correctly(self, config_manager, temp_dir, sample_yaml_config):
        """Test file watcher correctly handles directory vs file watching."""
        config_dir = temp_dir / "configs"
        config_dir.mkdir()
        config_file = config_dir / "config.yaml"
        config_file.write_text(sample_yaml_config)

        # Watch directory
        config_manager.watch_config_directory(str(config_dir))
        
        # Verify directory is in watched paths
        watched_paths = config_manager.get_watched_paths()
        assert str(config_dir.resolve()) in watched_paths

        # File watcher should recognize config files in watched directories
        file_watcher = config_manager._file_watcher
        new_config = config_dir / "new_config.yml"
        
        # Should identify this as a config file in a watched directory
        assert file_watcher._is_config_file(str(new_config)) is True

        config_manager.disable_watch_mode()


class TestConfigManagerIntegration:
    """Integration tests for ConfigManager with real file system operations."""

    def test_full_config_workflow(self, temp_dir):
        """Test complete configuration workflow from loading to validation."""
        # Create a comprehensive config file
        config_content = """
# AIM2 Project Configuration
project:
  name: "AIM2 Ontology Information Extraction"
  version: "1.0.0"
  description: "Comprehensive ontology information extraction system"

database:
  host: localhost
  port: 5432
  name: aim2_db
  username: aim2_user
  password: secure_password
  ssl_mode: require
  pool_size: 10

api:
  base_url: https://api.aim2.example.com
  version: v1
  timeout: 30
  retry_attempts: 3
  rate_limit: 1000

logging:
  level: INFO
  format: "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
  handlers:
    - console
    - file
  file_path: /var/log/aim2.log
  max_file_size: 10MB
  backup_count: 5

nlp:
  models:
    ner: "bert-base-uncased"
    relationship: "roberta-large"
  max_sequence_length: 512
  batch_size: 32
  cache_dir: /tmp/aim2_models

ontology:
  default_namespace: "http://aim2.example.com/ontology#"
  import_paths:
    - /data/ontologies/base
    - /data/ontologies/domain
  export_formats:
    - owl
    - rdf
    - json-ld

features:
  enable_caching: true
  cache_ttl: 3600
  enable_metrics: true
  debug_mode: false
  async_processing: true
  max_workers: 4
"""

        config_file = temp_dir / "full_config.yaml"
        config_file.write_text(config_content)

        # Test complete workflow
        manager = ConfigManager()

        # Load config
        manager.load_config(str(config_file))

        # Test environment override
        with patch.dict(
            os.environ, {"AIM2_DATABASE_PORT": "8080", "AIM2_NLP_BATCH_SIZE": "64"}
        ):
            manager.load_config(str(config_file))

        # Verify final configuration
        assert manager.config["database"]["port"] == 8080
        assert manager.config["nlp"]["batch_size"] == 64
        assert (
            manager.config["project"]["name"] == "AIM2 Ontology Information Extraction"
        )
        assert manager.get("features.enable_caching") is True
        assert (
            manager.get("ontology.default_namespace")
            == "http://aim2.example.com/ontology#"
        )

    def test_config_performance_large_file(self, temp_dir):
        """Test configuration loading performance with large config files."""
        import time

        # Generate a large configuration
        large_config = {"sections": {}}

        # Create 100 sections with 50 keys each
        for i in range(100):
            section = {}
            for j in range(50):
                section[f"key_{j}"] = f"value_{i}_{j}"
            large_config["sections"][f"section_{i}"] = section

        large_config_file = temp_dir / "large_config.json"
        large_config_file.write_text(json.dumps(large_config, indent=2))

        manager = ConfigManager()

        # Measure loading time
        start_time = time.time()
        manager.load_config(str(large_config_file))
        load_time = time.time() - start_time

        # Verify config loaded and performance is reasonable (< 1 second)
        assert len(manager.config["sections"]) == 100
        assert load_time < 1.0, f"Config loading took too long: {load_time:.2f}s"
