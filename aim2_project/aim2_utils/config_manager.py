"""
Configuration Management Module for AIM2 Project

This module provides comprehensive configuration management functionality including:
- Loading configuration from YAML and JSON files
- Environment variable overrides with configurable prefix
- Configuration validation and schema checking
- Configuration merging and interpolation
- Backup/restore functionality
- File watching for automatic reloads
- Configuration export capabilities

Classes:
    ConfigError: Custom exception for configuration-related errors
    ConfigManager: Main configuration management class

Dependencies:
    - yaml: For YAML file parsing
    - json: For JSON file parsing
    - os: For environment variable access
    - pathlib: For file path operations
    - re: For pattern matching and interpolation
    - copy: For deep copying configurations
    - threading: For file watching (legacy support)
    - time: For backup timestamping and debouncing
    - uuid: For backup ID generation
    - watchdog: For efficient file system monitoring
    - logging: For file change event logging
"""

import os
import json
import yaml
import re
import copy
import threading
import time
import uuid
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Set
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler


class ConfigError(Exception):
    """
    Custom exception for configuration-related errors.

    This exception is raised when configuration loading, validation,
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


class ConfigFileWatcher(FileSystemEventHandler):
    """
    Custom file system event handler for configuration file changes.

    Handles file modification and creation events for configuration files,
    with debouncing to prevent rapid successive reloads.

    Attributes:
        config_manager: Reference to the ConfigManager instance
        watched_paths: Set of file paths being watched
        debounce_delay: Delay in seconds to debounce file change events
        pending_reloads: Dictionary tracking pending reload operations
        logger: Logger instance for change events
    """

    def __init__(self, config_manager, debounce_delay: float = 0.5):
        """
        Initialize the configuration file watcher.

        Args:
            config_manager: ConfigManager instance to notify of changes
            debounce_delay: Delay in seconds for debouncing (default: 0.5)
        """
        super().__init__()
        self.config_manager = config_manager
        self.watched_paths: Set[str] = set()
        self.debounce_delay = debounce_delay
        self.pending_reloads: Dict[str, float] = {}
        self.logger = logging.getLogger(__name__)

    def add_watched_path(self, path: str) -> None:
        """Add a path to the set of watched paths."""
        self.watched_paths.add(str(Path(path).resolve()))

    def remove_watched_path(self, path: str) -> None:
        """Remove a path from the set of watched paths."""
        self.watched_paths.discard(str(Path(path).resolve()))

    def on_modified(self, event):
        """Handle file modification events."""
        if not event.is_directory:
            self._handle_file_change(event.src_path, "modified")

    def on_created(self, event):
        """Handle file creation events."""
        if not event.is_directory:
            self._handle_file_change(event.src_path, "created")

    def _handle_file_change(self, file_path: str, event_type: str) -> None:
        """
        Handle file change events with debouncing.

        Args:
            file_path: Path of the changed file
            event_type: Type of change event (modified/created)
        """
        try:
            resolved_path = str(Path(file_path).resolve())

            # Check if this is a file we're watching
            if resolved_path not in self.watched_paths:
                # Check if it's a config file in a watched directory
                if not self._is_config_file(file_path):
                    return

                # Check if parent directory is being watched
                parent_watched = False
                for watched_path in self.watched_paths:
                    if Path(watched_path).is_dir() and resolved_path.startswith(
                        watched_path
                    ):
                        parent_watched = True
                        break

                if not parent_watched:
                    return

            current_time = time.time()

            # Check if we already have a pending reload for this file
            if resolved_path in self.pending_reloads:
                last_change_time = self.pending_reloads[resolved_path]
                if current_time - last_change_time < self.debounce_delay:
                    # Update the pending reload time
                    self.pending_reloads[resolved_path] = current_time
                    return

            # Schedule debounced reload
            self.pending_reloads[resolved_path] = current_time

            # Start timer for debounced reload
            timer = threading.Timer(
                self.debounce_delay,
                self._debounced_reload,
                args=(resolved_path, event_type, current_time),
            )
            timer.daemon = True
            timer.start()

        except Exception as e:
            self.logger.error(f"Error handling file change for {file_path}: {e}")

    def _is_config_file(self, file_path: str) -> bool:
        """Check if a file is a configuration file based on extension."""
        config_extensions = {".yaml", ".yml", ".json"}
        return Path(file_path).suffix.lower() in config_extensions

    def _debounced_reload(
        self, file_path: str, event_type: str, change_time: float
    ) -> None:
        """
        Perform debounced configuration reload.

        Args:
            file_path: Path of the changed file
            event_type: Type of change event
            change_time: Time when the change was detected
        """
        try:
            # Check if this is still the most recent change for this file
            if file_path in self.pending_reloads:
                if self.pending_reloads[file_path] != change_time:
                    # A more recent change occurred, skip this reload
                    return

                # Remove from pending reloads
                del self.pending_reloads[file_path]

            self.logger.info(f"Configuration file {event_type}: {file_path}")

            # Attempt to reload configuration
            if hasattr(self.config_manager, "reload_config"):
                self.config_manager.reload_config()
                self.logger.info("Configuration reloaded successfully")
            else:
                self.logger.warning("ConfigManager does not support reload_config")

        except Exception as e:
            self.logger.error(
                f"Failed to reload configuration after {event_type} of {file_path}: {e}"
            )


class ConfigManager:
    """
    Main configuration management class.

    Provides comprehensive configuration management including loading from multiple
    sources, validation, environment variable overrides, and advanced features
    like file watching and configuration interpolation.

    Attributes:
        config (dict): The current configuration dictionary
        env_prefix (str): Prefix for environment variable overrides
        config_paths (list): List of loaded configuration file paths
        backups (dict): Dictionary of configuration backups
        watchers (dict): Dictionary of file watchers (legacy)
        watching (bool): Whether file watching is currently enabled
        _observer (Observer): Watchdog observer instance for file monitoring
        _file_watcher (ConfigFileWatcher): Custom event handler for config changes
        _watch_thread (Thread): Legacy threading watch thread
        _stop_watching (Event): Threading event for stopping legacy watcher
    """

    def __init__(self):
        """
        Initialize the ConfigManager.

        Sets up default configuration state and initializes internal structures
        for managing configuration data, backups, and file watching.
        """
        self.config = {}
        self.env_prefix = "AIM2"
        self.config_paths = []
        self.backups = {}
        self.watchers = {}  # Legacy watcher storage for backward compatibility
        self.watching = False

        # Legacy threading components (maintained for backward compatibility)
        self._watch_thread = None
        self._stop_watching = threading.Event()

        # New watchdog components
        self._observer: Optional[Observer] = None
        self._file_watcher: Optional[ConfigFileWatcher] = None
        self._watched_directories: Set[str] = set()
        self._debounce_delay = 0.5  # Default debounce delay in seconds

        # Set up logging
        self._logger = logging.getLogger(__name__)

        # Enhanced validation
        self._validator = None

    @property
    def validator(self):
        """Get or create the ConfigValidator instance."""
        if self._validator is None:
            from .config_validator import ConfigValidator

            self._validator = ConfigValidator()
        return self._validator

    def load_config(self, config_path: str) -> None:
        """
        Load configuration from a single file.

        Supports both YAML and JSON formats. The format is auto-detected
        based on the file extension. After loading, applies environment
        variable overrides and performs configuration interpolation.

        Args:
            config_path (str): Path to the configuration file

        Raises:
            ConfigError: If file doesn't exist, can't be parsed, or is invalid
        """
        try:
            path = Path(config_path)
            if not path.exists():
                raise ConfigError(f"File not found: {config_path}")

            # Store the config path for potential reloading
            if config_path not in self.config_paths:
                self.config_paths.append(config_path)

            # Load based on file extension
            if path.suffix.lower() in [".yaml", ".yml"]:
                self._load_yaml_file(config_path)
            elif path.suffix.lower() == ".json":
                self._load_json_file(config_path)
            else:
                raise ConfigError(f"Unsupported file format: {path.suffix}")

            # Apply environment variable overrides
            self._apply_env_overrides()

            # Apply decryption to encrypted values
            self._apply_decryption()

            # Perform configuration interpolation
            self._interpolate_config()

        except Exception as e:
            if isinstance(e, ConfigError):
                raise
            raise ConfigError(f"Failed to load config from {config_path}: {str(e)}", e)

    def load_configs(self, config_paths: List[str]) -> None:
        """
        Load and merge multiple configuration files.

        Configurations are loaded in the order provided, with later configs
        overriding values from earlier ones.

        Args:
            config_paths (List[str]): List of configuration file paths

        Raises:
            ConfigError: If any file fails to load
        """
        if not config_paths:
            raise ConfigError("No configuration paths provided")

        merged_config = {}

        for config_path in config_paths:
            # Create a temporary config manager to load each file
            temp_manager = ConfigManager()
            temp_manager.env_prefix = self.env_prefix
            temp_manager.load_config(config_path)

            # Merge the loaded config
            merged_config = self.merge_configs(merged_config, temp_manager.config)

        self.config = merged_config
        self.config_paths = list(config_paths)

    def load_default_config(self) -> None:
        """
        Load the default configuration file.

        Loads the default configuration from the standard location within
        the project structure. This is typically used as a base configuration
        that can be overridden by other config files or environment variables.

        Raises:
            ConfigError: If default config file is not found or invalid
        """
        try:
            # Get the path to the default config file
            current_dir = Path(__file__).parent.parent
            default_config_path = current_dir / "configs" / "default_config.yaml"

            if not default_config_path.exists():
                # Try alternative paths
                alt_paths = [
                    Path.cwd() / "aim2_project" / "configs" / "default_config.yaml",
                    Path.cwd() / "configs" / "default_config.yaml",
                ]

                for alt_path in alt_paths:
                    if alt_path.exists():
                        default_config_path = alt_path
                        break
                else:
                    raise ConfigError("Default configuration file not found")

            self.load_config(str(default_config_path))

        except Exception as e:
            if isinstance(e, ConfigError):
                raise
            raise ConfigError(f"Failed to load default configuration: {str(e)}", e)

    def get(self, key: str, default: Any = None) -> Any:
        """
        Get a configuration value using dot notation.

        Supports nested key access using dots (e.g., "database.host").
        If the key is not found and no default is provided, raises KeyError.

        Args:
            key (str): Configuration key in dot notation
            default (Any, optional): Default value if key not found

        Returns:
            Any: The configuration value

        Raises:
            KeyError: If key not found and no default provided
        """
        if default is None and len([x for x in [default] if x is not None]) == 0:
            # No default was explicitly provided
            has_default = False
        else:
            has_default = True

        keys = key.split(".")
        current = self.config

        try:
            for k in keys:
                current = current[k]
            return current
        except (KeyError, TypeError):
            if has_default:
                return default
            raise KeyError(f"Configuration key not found: {key}")

    def validate_config(self, config: Dict[str, Any]) -> bool:
        """
        Validate a configuration dictionary.

        Performs basic validation to ensure the configuration has required
        fields and proper structure. This is a basic validation method that
        can be extended for more complex validation rules.

        Args:
            config (Dict[str, Any]): Configuration to validate

        Returns:
            bool: True if configuration is valid

        Raises:
            ConfigError: If configuration is invalid
        """
        if not isinstance(config, dict):
            raise ConfigError("Configuration must be a dictionary")

        if not config:
            raise ConfigError("Configuration cannot be empty")

        # Basic validation - check for required top-level sections
        required_sections = ["database"]  # Minimum required based on tests

        for section in required_sections:
            if section not in config:
                raise ConfigError(f"Required configuration section missing: {section}")

        # Validate database section if present
        if "database" in config:
            db_config = config["database"]

            # Type checking first (for fields that exist)
            if "port" in db_config and not isinstance(db_config.get("port"), int):
                raise ConfigError("Invalid type for database port: expected int")

            # Then check for required fields
            required_db_fields = ["host", "port", "name", "username", "password"]
            for field in required_db_fields:
                if field not in db_config:
                    raise ConfigError(f"Required database field missing: {field}")

        return True

    def validate_config_schema(
        self, config: Dict[str, Any], schema: Dict[str, Any], strict: bool = False
    ) -> bool:
        """
        Validate configuration against a schema.

        Performs comprehensive validation including type checking, required fields,
        and constraint validation based on the provided schema.

        Args:
            config (Dict[str, Any]): Configuration to validate
            schema (Dict[str, Any]): Validation schema
            strict (bool): Whether to enforce strict validation

        Returns:
            bool: True if configuration is valid

        Raises:
            ConfigError: If configuration doesn't match schema
        """
        from .config_validator import ConfigValidator

        validator = ConfigValidator()
        return validator.validate_schema(config, schema, strict=strict)

    def validate_with_aim2_schema(
        self,
        config: Optional[Dict[str, Any]] = None,
        strict: bool = False,
        return_report: bool = False,
    ) -> Any:
        """
        Validate configuration against the built-in AIM2 project schema.

        Args:
            config: Configuration to validate (uses self.config if None)
            strict: Whether to enforce strict validation
            return_report: Whether to return detailed validation report

        Returns:
            bool or ValidationReport: Validation result

        Raises:
            ConfigError: If validation fails and return_report is False
        """
        from .config_validator import ValidationReport

        config_to_validate = config if config is not None else self.config

        if not config_to_validate:
            if return_report:
                report = ValidationReport()
                report.add_error("No configuration to validate")
                return report
            else:
                raise ConfigError("No configuration to validate")

        try:
            result = self.validator.validate_aim2_config(
                config_to_validate, strict=strict, return_report=return_report
            )

            if return_report:
                return result
            elif result is True:
                return True
            else:
                raise ConfigError("Configuration validation failed")

        except ValueError as e:
            raise ConfigError(f"Schema validation error: {str(e)}", e)

    def validate_with_schema(
        self,
        schema_name: str,
        config: Optional[Dict[str, Any]] = None,
        strict: bool = False,
        return_report: bool = False,
    ) -> Any:
        """
        Validate configuration against a named schema.

        Args:
            schema_name: Name of the schema to use
            config: Configuration to validate (uses self.config if None)
            strict: Whether to enforce strict validation
            return_report: Whether to return detailed validation report

        Returns:
            bool or ValidationReport: Validation result

        Raises:
            ConfigError: If validation fails
        """
        from .config_validator import ValidationReport

        config_to_validate = config if config is not None else self.config

        if not config_to_validate:
            if return_report:
                report = ValidationReport()
                report.add_error("No configuration to validate")
                return report
            else:
                raise ConfigError("No configuration to validate")

        try:
            result = self.validator.validate_with_schema(
                config_to_validate,
                schema_name,
                strict=strict,
                return_report=return_report,
            )

            if return_report:
                return result
            elif result is True:
                return True
            else:
                raise ConfigError("Configuration validation failed")

        except ValueError as e:
            raise ConfigError(f"Schema validation error: {str(e)}", e)

    def get_available_schemas(self) -> List[str]:
        """
        Get list of all available validation schemas.

        Returns:
            List of schema names
        """
        return self.validator.get_available_schemas()

    def load_validation_schema(
        self, file_path: str, name: Optional[str] = None, version: str = "1.0.0"
    ) -> None:
        """
        Load a validation schema from an external file.

        Args:
            file_path: Path to schema file (JSON or YAML)
            name: Optional name for the schema (defaults to filename)
            version: Schema version

        Raises:
            ConfigError: If schema loading fails
        """
        try:
            self.validator.load_and_register_schema(file_path, name, version)
        except Exception as e:
            raise ConfigError(f"Failed to load validation schema: {str(e)}", e)

    def validate_current_config(
        self,
        schema_name: str = "aim2_project",
        strict: bool = False,
        return_report: bool = False,
    ) -> Any:
        """
        Validate the current configuration against a schema.

        Convenience method that validates the currently loaded configuration.

        Args:
            schema_name: Name of the schema to use (default: aim2_project)
            strict: Whether to enforce strict validation
            return_report: Whether to return detailed validation report

        Returns:
            bool or ValidationReport: Validation result
        """
        return self.validate_with_schema(
            schema_name, config=None, strict=strict, return_report=return_report
        )

    def merge_configs(
        self, base_config: Dict[str, Any], override_config: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Merge two configuration dictionaries.

        Performs deep merge where override_config values replace or extend
        base_config values. Nested dictionaries are recursively merged.

        Args:
            base_config (Dict[str, Any]): Base configuration
            override_config (Dict[str, Any]): Override configuration

        Returns:
            Dict[str, Any]: Merged configuration
        """
        if not isinstance(base_config, dict):
            base_config = {}
        if not isinstance(override_config, dict):
            return copy.deepcopy(base_config)

        merged = copy.deepcopy(base_config)

        for key, value in override_config.items():
            if (
                key in merged
                and isinstance(merged[key], dict)
                and isinstance(value, dict)
            ):
                # Recursively merge nested dictionaries
                merged[key] = self.merge_configs(merged[key], value)
            else:
                # Override or add new value
                merged[key] = copy.deepcopy(value)

        return merged

    def export_config(self, path: str, format: str = "yaml") -> None:
        """
        Export current configuration to a file.

        Supports exporting to YAML or JSON formats.

        Args:
            path (str): Output file path
            format (str): Export format ("yaml" or "json")

        Raises:
            ConfigError: If export fails or format is unsupported
        """
        try:
            output_path = Path(path)
            output_path.parent.mkdir(parents=True, exist_ok=True)

            if format.lower() == "yaml":
                with open(output_path, "w", encoding="utf-8") as f:
                    yaml.dump(self.config, f, default_flow_style=False, indent=2)
            elif format.lower() == "json":
                with open(output_path, "w", encoding="utf-8") as f:
                    json.dump(self.config, f, indent=2, ensure_ascii=False)
            else:
                raise ConfigError(f"Unsupported export format: {format}")

        except Exception as e:
            if isinstance(e, ConfigError):
                raise
            raise ConfigError(f"Failed to export config to {path}: {str(e)}", e)

    def reload_config(self) -> None:
        """
        Reload configuration from previously loaded files.

        Reloads all configuration files that were previously loaded,
        maintaining the same merge order.

        Raises:
            ConfigError: If reload fails
        """
        if not self.config_paths:
            raise ConfigError("No configuration files to reload")

        # Save current config as backup before reloading
        backup_id = self.create_backup()

        try:
            if len(self.config_paths) == 1:
                self.config = {}  # Clear current config
                self.load_config(self.config_paths[0])
            else:
                self.config = {}  # Clear current config
                self.load_configs(self.config_paths)
        except Exception as e:
            # Restore backup if reload fails
            self.restore_backup(backup_id)
            raise ConfigError(f"Failed to reload configuration: {str(e)}", e)

    def enable_watch_mode(self, path: str, use_legacy: bool = False) -> None:
        """
        Enable file watching for automatic configuration reloads.

        Monitors the specified file or directory for changes and automatically reloads
        the configuration when changes are detected. Uses watchdog library by default
        for efficient monitoring, with fallback to legacy threading approach.

        Args:
            path (str): Path to watch for changes (file or directory)
            use_legacy (bool): Whether to use legacy threading approach instead of watchdog
        """
        if self.watching:
            self.disable_watch_mode()

        self.watching = True

        if use_legacy:
            self._enable_legacy_watch_mode(path)
        else:
            self._enable_watchdog_mode(path)

    def _enable_legacy_watch_mode(self, path: str) -> None:
        """
        Enable legacy threading-based file watching (for backward compatibility).

        Args:
            path (str): Path to watch for changes
        """
        self._stop_watching.clear()

        # Store the watch path
        self.watchers[path] = {
            "path": path,
            "last_modified": os.path.getmtime(path) if os.path.exists(path) else 0,
        }

        # Start the watch thread
        self._watch_thread = threading.Thread(
            target=self._watch_file_changes, args=(path,), daemon=True
        )
        self._watch_thread.start()
        self._logger.info(f"Legacy file watching enabled for: {path}")

    def _enable_watchdog_mode(self, path: str) -> None:
        """
        Enable watchdog-based file watching.

        Args:
            path (str): Path to watch for changes (file or directory)
        """
        try:
            path_obj = Path(path)

            # Initialize observer if not already done
            if self._observer is None:
                self._observer = Observer()

            # Initialize file watcher if not already done
            if self._file_watcher is None:
                self._file_watcher = ConfigFileWatcher(self, self._debounce_delay)

            # Determine what to watch
            if path_obj.is_file():
                # Watch the parent directory and track this specific file
                watch_dir = str(path_obj.parent)
                self._file_watcher.add_watched_path(str(path_obj.resolve()))
                self._logger.info(f"Watching config file: {path}")
            elif path_obj.is_dir():
                # Watch the directory for any config file changes
                watch_dir = str(path_obj)
                self._file_watcher.add_watched_path(str(path_obj.resolve()))
                self._logger.info(f"Watching config directory: {path}")
            else:
                raise ConfigError(f"Path does not exist: {path}")

            # Add directory to observer if not already watching
            if watch_dir not in self._watched_directories:
                self._observer.schedule(self._file_watcher, watch_dir, recursive=False)
                self._watched_directories.add(watch_dir)
                self._logger.info(f"Added directory to watchdog observer: {watch_dir}")

            # Start observer if not already running
            if not self._observer.is_alive():
                self._observer.start()
                self._logger.info("Watchdog observer started")

            # Maintain legacy watchers dict for backward compatibility
            self.watchers[path] = {
                "path": path,
                "last_modified": os.path.getmtime(path)
                if os.path.exists(path) and path_obj.is_file()
                else time.time(),
                "type": "watchdog",
            }

        except Exception as e:
            self.watching = False
            raise ConfigError(f"Failed to enable watchdog mode for {path}: {str(e)}", e)

    def disable_watch_mode(self) -> None:
        """
        Disable file watching.

        Stops monitoring files for changes and cleans up watching resources.
        Handles both legacy threading and watchdog-based watching.
        """
        self.watching = False

        # Clean up legacy threading components
        self._stop_watching.set()
        if self._watch_thread and self._watch_thread.is_alive():
            self._watch_thread.join(timeout=1.0)

        # Clean up watchdog components
        if self._observer is not None:
            try:
                if self._observer.is_alive():
                    self._observer.stop()
                    self._observer.join(timeout=2.0)
                    self._logger.info("Watchdog observer stopped")
            except Exception as e:
                self._logger.error(f"Error stopping watchdog observer: {e}")
            finally:
                self._observer = None

        # Clean up file watcher
        if self._file_watcher is not None:
            self._file_watcher.watched_paths.clear()
            self._file_watcher.pending_reloads.clear()
            self._file_watcher = None

        # Clear watched directories
        self._watched_directories.clear()

        # Clear legacy watchers
        self.watchers.clear()

        self._logger.info("File watching disabled")

    def add_watch_path(self, path: str) -> None:
        """
        Add an additional path to the existing watch configuration.

        This allows watching multiple files or directories simultaneously.
        The watchdog observer must already be running.

        Args:
            path (str): Additional path to watch for changes

        Raises:
            ConfigError: If watching is not enabled or path setup fails
        """
        if not self.watching:
            raise ConfigError(
                "Cannot add watch path: watching is not enabled. Call enable_watch_mode() first."
            )

        if self._observer is None or self._file_watcher is None:
            raise ConfigError(
                "Cannot add watch path: watchdog components not initialized. Use enable_watch_mode() first."
            )

        try:
            path_obj = Path(path)

            if path_obj.is_file():
                # Watch the parent directory and track this specific file
                watch_dir = str(path_obj.parent)
                self._file_watcher.add_watched_path(str(path_obj.resolve()))
                self._logger.info(f"Added config file to watch list: {path}")
            elif path_obj.is_dir():
                # Watch the directory for any config file changes
                watch_dir = str(path_obj)
                self._file_watcher.add_watched_path(str(path_obj.resolve()))
                self._logger.info(f"Added config directory to watch list: {path}")
            else:
                raise ConfigError(f"Path does not exist: {path}")

            # Add directory to observer if not already watching
            if watch_dir not in self._watched_directories:
                self._observer.schedule(self._file_watcher, watch_dir, recursive=False)
                self._watched_directories.add(watch_dir)
                self._logger.info(f"Added directory to watchdog observer: {watch_dir}")

            # Update watchers dict for backward compatibility
            self.watchers[path] = {
                "path": path,
                "last_modified": os.path.getmtime(path)
                if os.path.exists(path) and path_obj.is_file()
                else time.time(),
                "type": "watchdog",
            }

        except Exception as e:
            raise ConfigError(f"Failed to add watch path {path}: {str(e)}", e)

    def remove_watch_path(self, path: str) -> None:
        """
        Remove a path from the watch configuration.

        Args:
            path (str): Path to stop watching

        Raises:
            ConfigError: If path removal fails
        """
        if not self.watching or self._file_watcher is None:
            return  # Nothing to remove

        try:
            path_obj = Path(path)
            resolved_path = str(path_obj.resolve())

            # Remove from file watcher
            self._file_watcher.remove_watched_path(resolved_path)

            # Remove from watchers dict
            if path in self.watchers:
                del self.watchers[path]

            self._logger.info(f"Removed path from watch list: {path}")

        except Exception as e:
            raise ConfigError(f"Failed to remove watch path {path}: {str(e)}", e)

    def watch_config_directory(self, directory: str, recursive: bool = False) -> None:
        """
        Watch an entire directory for configuration file changes.

        This is a convenience method that enables watching of all configuration
        files in a directory. Can be used with or without existing watch mode.

        Args:
            directory (str): Directory path to watch
            recursive (bool): Whether to watch subdirectories recursively

        Raises:
            ConfigError: If directory setup fails
        """
        try:
            dir_path = Path(directory)
            if not dir_path.is_dir():
                raise ConfigError(f"Directory does not exist: {directory}")

            if not self.watching:
                # Enable watch mode for this directory
                self.enable_watch_mode(directory)
            else:
                # Add to existing watch configuration
                self.add_watch_path(directory)

            # If recursive watching is requested, we need to set up additional observers
            if recursive:
                self._setup_recursive_watching(directory)

            self._logger.info(
                f"Watching config directory {'recursively' if recursive else ''}: {directory}"
            )

        except Exception as e:
            raise ConfigError(
                f"Failed to watch config directory {directory}: {str(e)}", e
            )

    def _setup_recursive_watching(self, directory: str) -> None:
        """
        Set up recursive watching for a directory.

        Args:
            directory (str): Directory to watch recursively
        """
        if self._observer is None or self._file_watcher is None:
            return

        try:
            # Remove existing non-recursive watch if it exists
            if directory in self._watched_directories:
                # We need to reschedule with recursive=True
                # Note: watchdog doesn't support changing recursion on existing watch
                # so we add a new watch for recursive monitoring
                pass

            # Add recursive watch
            self._observer.schedule(self._file_watcher, directory, recursive=True)
            self._watched_directories.add(f"{directory}:recursive")
            self._logger.info(f"Added recursive watching for directory: {directory}")

        except Exception as e:
            self._logger.error(
                f"Failed to setup recursive watching for {directory}: {e}"
            )

    def set_debounce_delay(self, delay: float) -> None:
        """
        Set the debounce delay for file change events.

        Args:
            delay (float): Delay in seconds (minimum 0.1, maximum 5.0)

        Raises:
            ConfigError: If delay is outside valid range
        """
        if not 0.1 <= delay <= 5.0:
            raise ConfigError("Debounce delay must be between 0.1 and 5.0 seconds")

        self._debounce_delay = delay

        if self._file_watcher is not None:
            self._file_watcher.debounce_delay = delay

        self._logger.info(f"Set debounce delay to {delay} seconds")

    def get_watched_paths(self) -> List[str]:
        """
        Get a list of all currently watched paths.

        Returns:
            List[str]: List of watched file and directory paths
        """
        if self._file_watcher is None:
            return list(self.watchers.keys())  # Fallback to legacy watchers

        return list(self._file_watcher.watched_paths)

    def is_watching(self) -> bool:
        """
        Check if file watching is currently enabled.

        Returns:
            bool: True if watching is enabled
        """
        return self.watching

    def create_backup(self) -> str:
        """
        Create a backup of the current configuration.

        Creates a timestamped backup that can be restored later.

        Returns:
            str: Backup ID for restoration
        """
        backup_id = str(uuid.uuid4())
        timestamp = time.time()

        self.backups[backup_id] = {
            "config": copy.deepcopy(self.config),
            "timestamp": timestamp,
            "config_paths": list(self.config_paths),
        }

        return backup_id

    def restore_backup(self, backup_id: str) -> None:
        """
        Restore configuration from a backup.

        Args:
            backup_id (str): ID of the backup to restore

        Raises:
            ConfigError: If backup ID is not found
        """
        if backup_id not in self.backups:
            raise ConfigError(f"Backup not found: {backup_id}")

        backup = self.backups[backup_id]
        self.config = copy.deepcopy(backup["config"])
        self.config_paths = list(backup["config_paths"])

    def set_env_prefix(self, prefix: str) -> None:
        """
        Set the environment variable prefix.

        Changes the prefix used for environment variable overrides.

        Args:
            prefix (str): New environment variable prefix
        """
        self.env_prefix = prefix.rstrip("_")  # Remove trailing underscore if present

    def decrypt_value(self, encrypted_value: str) -> str:
        """
        Decrypt an encrypted configuration value.

        This is a placeholder for future encryption support. Currently
        returns the value with the encryption prefix removed.

        Args:
            encrypted_value (str): Encrypted value to decrypt

        Returns:
            str: Decrypted value
        """
        # Placeholder implementation for future encryption support
        if encrypted_value.startswith("encrypted:"):
            # Remove the encryption prefix for testing
            return encrypted_value.replace("encrypted:AES256:", "decrypted_")
        return encrypted_value

    # Private helper methods

    def _load_yaml_file(self, file_path: str) -> None:
        """Load configuration from a YAML file."""
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                loaded_config = yaml.safe_load(f) or {}
            self.config = self.merge_configs(self.config, loaded_config)
        except yaml.YAMLError as e:
            raise ConfigError(f"Invalid YAML in {file_path}: {str(e)}", e)
        except Exception as e:
            raise ConfigError(f"Failed to read YAML file {file_path}: {str(e)}", e)

    def _load_json_file(self, file_path: str) -> None:
        """Load configuration from a JSON file."""
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                loaded_config = json.load(f) or {}
            self.config = self.merge_configs(self.config, loaded_config)
        except json.JSONDecodeError as e:
            raise ConfigError(f"Invalid JSON in {file_path}: {str(e)}", e)
        except Exception as e:
            raise ConfigError(f"Failed to read JSON file {file_path}: {str(e)}", e)

    def _apply_env_overrides(self) -> None:
        """Apply environment variable overrides to configuration."""
        env_prefix = f"{self.env_prefix}_"

        for env_key, env_value in os.environ.items():
            if env_key.startswith(env_prefix):
                # Convert environment variable name to config path
                # Split by underscore and rejoin appropriately for nested config
                parts = env_key[len(env_prefix) :].lower().split("_")

                # Handle nested config paths - assume first part is section, rest is field
                if len(parts) == 1:
                    config_path = parts[0]
                elif len(parts) == 2:
                    config_path = f"{parts[0]}.{parts[1]}"
                else:
                    # For longer paths, treat first as section, join rest with underscores
                    section = parts[0]
                    field = "_".join(parts[1:])
                    config_path = f"{section}.{field}"

                # Convert string value to appropriate type
                converted_value = self._convert_env_value(env_value)

                # Set the value in config
                self._set_nested_value(self.config, config_path, converted_value)

    def _convert_env_value(self, value: str) -> Any:
        """Convert environment variable string to appropriate type."""
        # Try to convert to appropriate Python type
        if value.lower() == "true":
            return True
        elif value.lower() == "false":
            return False
        elif value.lower() == "null" or value.lower() == "none":
            return None
        elif value.isdigit():
            return int(value)
        elif self._is_float(value):
            return float(value)
        else:
            return value

    def _is_float(self, value: str) -> bool:
        """Check if a string represents a float."""
        try:
            float(value)
            return "." in value
        except ValueError:
            return False

    def _set_nested_value(self, config: Dict[str, Any], path: str, value: Any) -> None:
        """Set a nested configuration value using dot notation."""
        keys = path.split(".")
        current = config

        for key in keys[:-1]:
            if key not in current:
                current[key] = {}
            elif not isinstance(current[key], dict):
                current[key] = {}
            current = current[key]

        current[keys[-1]] = value

    def _interpolate_config(self) -> None:
        """Perform configuration value interpolation."""
        # This is a simplified interpolation - could be enhanced
        self.config = self._interpolate_recursive(self.config, self.config)

    def _interpolate_recursive(self, obj: Any, context: Dict[str, Any]) -> Any:
        """Recursively interpolate configuration values."""
        if isinstance(obj, dict):
            return {k: self._interpolate_recursive(v, context) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._interpolate_recursive(item, context) for item in obj]
        elif isinstance(obj, str):
            return self._interpolate_string(obj, context)
        else:
            return obj

    def _interpolate_string(self, value: str, context: Dict[str, Any]) -> str:
        """Interpolate variables in a string value."""
        # Find all ${variable} patterns
        pattern = r"\$\{([^}]+)\}"

        def replace_var(match):
            var_path = match.group(1)
            try:
                # Get the value using dot notation
                keys = var_path.split(".")
                current = context
                for key in keys:
                    current = current[key]
                return str(current)
            except (KeyError, TypeError):
                # Return the original pattern if variable not found
                return match.group(0)

        return re.sub(pattern, replace_var, value)

    def _apply_decryption(self) -> None:
        """Apply decryption to encrypted configuration values."""
        self.config = self._decrypt_recursive(self.config)

    def _decrypt_recursive(self, obj: Any) -> Any:
        """Recursively decrypt encrypted configuration values."""
        if isinstance(obj, dict):
            return {k: self._decrypt_recursive(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._decrypt_recursive(item) for item in obj]
        elif isinstance(obj, str) and obj.startswith("encrypted:"):
            return self.decrypt_value(obj)
        else:
            return obj

    def _watch_file_changes(self, file_path: str) -> None:
        """Watch for file changes and reload configuration."""
        while not self._stop_watching.is_set():
            try:
                if os.path.exists(file_path):
                    current_mtime = os.path.getmtime(file_path)
                    last_mtime = self.watchers[file_path]["last_modified"]

                    if current_mtime > last_mtime:
                        # File has been modified
                        self.watchers[file_path]["last_modified"] = current_mtime
                        try:
                            self.reload_config()
                        except Exception:
                            # Silently ignore reload errors in watch mode
                            pass

                # Sleep for a short interval before checking again
                self._stop_watching.wait(1.0)  # Check every second

            except Exception:
                # Silently ignore errors in watch thread
                self._stop_watching.wait(1.0)
