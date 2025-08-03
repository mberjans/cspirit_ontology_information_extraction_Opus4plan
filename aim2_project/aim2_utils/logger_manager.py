"""
Logger Manager Module for AIM2 Project

This module provides centralized logger management functionality including:
- Centralized logger creation and management
- Setup console and rotating file handlers
- Module-specific logger creation with hierarchical naming
- Configuration reloading with graceful handler management
- Thread-safe operations and cleanup
- Integration with LoggerConfig for configuration management

Classes:
    LoggerManagerError: Custom exception for logger manager-related errors
    LoggerManager: Main logger management class

Dependencies:
    - logging: For Python logging system
    - logging.handlers: For rotating file handler
    - threading: For thread-safe operations
    - pathlib: For file path operations
    - typing: For type hints
    - atexit: For cleanup registration
"""

import logging
import logging.handlers
import threading
import atexit
from pathlib import Path
from typing import Any, Dict, List, Optional

from .logger_config import LoggerConfig, LoggerConfigError
from .colored_console_handler import ColoredConsoleHandler
from .json_formatter import JSONFormatter
from .request_context import RequestContextManager
from .context_injection import (
    ContextInjectionFilter,
    ContextInjectingLogger,
)


class LoggerManagerError(Exception):
    """
    Custom exception for logger manager-related errors.

    This exception is raised when logger management operations encounter errors
    such as configuration issues, handler setup failures, or cleanup problems.

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


class LoggerManager:
    """
    Centralized logger management class.

    Provides comprehensive logger management including creation, configuration,
    handler management, and cleanup. Supports module-specific loggers with
    hierarchical naming and thread-safe operations.

    Attributes:
        config (LoggerConfig): Logger configuration instance
        root_logger (logging.Logger): Root logger for the project
        loggers (Dict[str, logging.Logger]): Registry of created loggers
        handlers (Dict[str, logging.Handler]): Registry of created handlers
        _lock (threading.RLock): Thread lock for safe operations
        _initialized (bool): Whether the manager has been initialized
        _cleanup_registered (bool): Whether cleanup has been registered
    """

    # Default root logger name for the project
    ROOT_LOGGER_NAME = "aim2"

    def __init__(self, config: Optional[LoggerConfig] = None):
        """
        Initialize the LoggerManager.

        Args:
            config (Optional[LoggerConfig]): Logger configuration instance.
                                           If None, creates default configuration.
        """
        self.config = config or LoggerConfig()
        self.root_logger: Optional[logging.Logger] = None
        self.loggers: Dict[str, logging.Logger] = {}
        self.handlers: Dict[str, logging.Handler] = {}
        self._lock = threading.RLock()
        self._initialized = False
        self._cleanup_registered = False

        # Context injection support
        self.context_manager: Optional[RequestContextManager] = None
        self.context_injection_filter: Optional[ContextInjectionFilter] = None

    def initialize(self) -> None:
        """
        Initialize the logger manager and set up the root logger.

        This method sets up the root logger with configured handlers and
        registers cleanup functions.

        Raises:
            LoggerManagerError: If initialization fails
        """
        with self._lock:
            if self._initialized:
                return

            try:
                # Validate configuration
                self.config.validate_config(self.config.to_dict())

                # Set up root logger
                self._setup_root_logger()

                # Register cleanup
                if not self._cleanup_registered:
                    atexit.register(self.cleanup)
                    self._cleanup_registered = True

                self._initialized = True

            except Exception as e:
                if isinstance(e, (LoggerConfigError, LoggerManagerError)):
                    raise LoggerManagerError(
                        f"Failed to initialize logger manager: {str(e)}", e
                    )
                raise LoggerManagerError(
                    f"Unexpected error during initialization: {str(e)}", e
                )

    def get_logger(
        self, name: Optional[str] = None, module_name: Optional[str] = None
    ) -> logging.Logger:
        """
        Get or create a logger with the specified name.

        Creates a hierarchical logger name based on the project root and
        optional module name. Loggers are cached for reuse.

        Args:
            name (Optional[str]): Logger name. If None, uses module_name or root name.
            module_name (Optional[str]): Module name for hierarchical naming.

        Returns:
            logging.Logger: Configured logger instance

        Raises:
            LoggerManagerError: If logger creation fails
        """
        with self._lock:
            if not self._initialized:
                self.initialize()

            try:
                # Determine the logger name
                if name is not None:
                    logger_name = name
                elif module_name is not None:
                    logger_name = f"{self.ROOT_LOGGER_NAME}.{module_name}"
                else:
                    logger_name = self.ROOT_LOGGER_NAME

                # Return cached logger if it exists
                if logger_name in self.loggers:
                    return self.loggers[logger_name]

                # Create new logger
                logger = logging.getLogger(logger_name)

                # Configure logger level and propagation
                # Use module-specific level if available, otherwise use global level
                effective_level = self._get_effective_level_for_logger(logger_name)
                logger.setLevel(getattr(logging, effective_level))

                # For non-root loggers, enable propagation to root
                if logger_name != self.ROOT_LOGGER_NAME:
                    logger.propagate = True
                else:
                    logger.propagate = False

                # Add context injection to individual loggers if enabled
                self._setup_logger_context_injection(logger, logger_name)

                # Cache the logger
                self.loggers[logger_name] = logger

                return logger

            except Exception as e:
                raise LoggerManagerError(
                    f"Failed to create logger '{name or module_name}': {str(e)}", e
                )

    def get_module_logger(self, module_name: str) -> logging.Logger:
        """
        Get a logger for a specific module.

        Convenience method for creating module-specific loggers with
        hierarchical naming.

        Args:
            module_name (str): Name of the module

        Returns:
            logging.Logger: Module-specific logger

        Raises:
            LoggerManagerError: If module name is invalid or logger creation fails
        """
        if not module_name or not isinstance(module_name, str):
            raise LoggerManagerError("Module name must be a non-empty string")

        # Clean module name for logger hierarchy
        clean_name = module_name.replace("/", ".").replace("\\", ".").strip(".")
        return self.get_logger(module_name=clean_name)

    def reload_configuration(self, new_config: Optional[LoggerConfig] = None) -> None:
        """
        Reload logger configuration and update all handlers.

        This method gracefully updates the logging configuration by removing
        old handlers and creating new ones based on the updated configuration.

        Args:
            new_config (Optional[LoggerConfig]): New configuration to apply.
                                               If None, reloads current configuration.

        Raises:
            LoggerManagerError: If configuration reload fails
        """
        with self._lock:
            try:
                # Update configuration if provided
                if new_config is not None:
                    self.config = new_config

                # Validate new configuration
                self.config.validate_config(self.config.to_dict())

                # Remove existing handlers
                self._cleanup_handlers()

                # Recreate root logger with new configuration
                self._setup_root_logger()

                # Update existing loggers with new levels (considering module-specific levels)
                for logger_name, logger in self.loggers.items():
                    effective_level = self._get_effective_level_for_logger(logger_name)
                    logger.setLevel(getattr(logging, effective_level))

                # Clear handler cache to force recreation
                self.handlers.clear()

            except Exception as e:
                if isinstance(e, (LoggerConfigError, LoggerManagerError)):
                    raise LoggerManagerError(
                        f"Failed to reload configuration: {str(e)}", e
                    )
                raise LoggerManagerError(
                    f"Unexpected error during configuration reload: {str(e)}", e
                )

    def set_level(self, level: str) -> None:
        """
        Set the logging level for all managed loggers.

        Args:
            level (str): Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)

        Raises:
            LoggerManagerError: If level is invalid
        """
        with self._lock:
            try:
                # Update configuration
                self.config.set_level(level)

                # Update all existing loggers with their effective levels
                for logger_name, logger in self.loggers.items():
                    effective_level = self._get_effective_level_for_logger(logger_name)
                    logger.setLevel(getattr(logging, effective_level))

            except Exception as e:
                if isinstance(e, LoggerConfigError):
                    raise LoggerManagerError(f"Failed to set level: {str(e)}", e)
                raise LoggerManagerError(f"Unexpected error setting level: {str(e)}", e)

    def set_module_level(self, module_name: str, level: str) -> None:
        """
        Set the logging level for a specific module.

        Args:
            module_name (str): Name of the module
            level (str): Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)

        Raises:
            LoggerManagerError: If module name or level is invalid
        """
        with self._lock:
            try:
                # Update configuration
                self.config.set_module_level(module_name, level)

                # Update all existing loggers that might be affected
                self._apply_module_level_updates()

            except Exception as e:
                if isinstance(e, LoggerConfigError):
                    raise LoggerManagerError(f"Failed to set module level: {str(e)}", e)
                raise LoggerManagerError(
                    f"Unexpected error setting module level: {str(e)}", e
                )

    def get_module_level(self, module_name: str) -> Optional[str]:
        """
        Get the specific logging level configured for a module.

        Args:
            module_name (str): Name of the module

        Returns:
            Optional[str]: Module-specific level or None if not configured
        """
        return self.config.get_module_level(module_name)

    def get_effective_module_level(self, module_name: str) -> str:
        """
        Get the effective logging level for a module (considering hierarchy).

        Args:
            module_name (str): Name of the module

        Returns:
            str: Effective logging level for the module
        """
        return self.config.get_effective_level(module_name)

    def remove_module_level(self, module_name: str) -> bool:
        """
        Remove the specific logging level configuration for a module.

        Args:
            module_name (str): Name of the module

        Returns:
            bool: True if module level was removed, False if it wasn't configured

        Raises:
            LoggerManagerError: If an error occurs during removal
        """
        with self._lock:
            try:
                removed = self.config.remove_module_level(module_name)

                if removed:
                    # Update all existing loggers that might be affected
                    self._apply_module_level_updates()

                return removed

            except Exception as e:
                raise LoggerManagerError(f"Failed to remove module level: {str(e)}", e)

    def clear_module_levels(self) -> None:
        """
        Clear all module-specific logging level configurations.

        Raises:
            LoggerManagerError: If an error occurs during clearing
        """
        with self._lock:
            try:
                self.config.clear_module_levels()

                # Update all existing loggers to use global level
                self._apply_module_level_updates()

            except Exception as e:
                raise LoggerManagerError(f"Failed to clear module levels: {str(e)}", e)

    def get_module_levels(self) -> Dict[str, str]:
        """
        Get all configured module-specific logging levels.

        Returns:
            Dict[str, str]: Dictionary mapping module names to log levels
        """
        return self.config.get_module_levels()

    def _apply_module_level_updates(self) -> None:
        """
        Apply module-specific level updates to all existing loggers.

        This method should be called after module-specific level changes
        to update all affected loggers.
        """
        for logger_name, logger in self.loggers.items():
            effective_level = self._get_effective_level_for_logger(logger_name)
            logger.setLevel(getattr(logging, effective_level))

    def add_handler_to_logger(self, logger_name: str, handler_type: str) -> None:
        """
        Add a specific handler to a logger.

        Args:
            logger_name (str): Name of the logger
            handler_type (str): Type of handler to add ("console" or "file")

        Raises:
            LoggerManagerError: If handler addition fails
        """
        with self._lock:
            try:
                if logger_name not in self.loggers:
                    raise LoggerManagerError(f"Logger '{logger_name}' not found")

                logger = self.loggers[logger_name]
                handler = self._create_handler(handler_type)

                if handler:
                    logger.addHandler(handler)

            except Exception as e:
                if isinstance(e, LoggerManagerError):
                    raise
                raise LoggerManagerError(
                    f"Failed to add handler to logger: {str(e)}", e
                )

    def remove_handler_from_logger(self, logger_name: str, handler_type: str) -> None:
        """
        Remove a specific handler from a logger.

        Args:
            logger_name (str): Name of the logger
            handler_type (str): Type of handler to remove ("console" or "file")

        Raises:
            LoggerManagerError: If handler removal fails
        """
        with self._lock:
            try:
                if logger_name not in self.loggers:
                    raise LoggerManagerError(f"Logger '{logger_name}' not found")

                logger = self.loggers[logger_name]

                # Find and remove handler of specified type
                handlers_to_remove = []
                for handler in logger.handlers:
                    if (
                        handler_type == "console"
                        and isinstance(handler, logging.StreamHandler)
                        and not isinstance(handler, logging.FileHandler)
                    ):
                        handlers_to_remove.append(handler)
                    elif handler_type == "file" and isinstance(
                        handler, logging.FileHandler
                    ):
                        handlers_to_remove.append(handler)

                for handler in handlers_to_remove:
                    logger.removeHandler(handler)
                    handler.close()

            except Exception as e:
                if isinstance(e, LoggerManagerError):
                    raise
                raise LoggerManagerError(
                    f"Failed to remove handler from logger: {str(e)}", e
                )

    def get_logger_info(self) -> Dict[str, Any]:
        """
        Get information about all managed loggers and handlers.

        Returns:
            Dict[str, Any]: Dictionary containing logger and handler information
        """
        with self._lock:
            info = {
                "configuration": self.config.to_dict(),
                "loggers": {},
                "handlers": list(self.handlers.keys()),
                "initialized": self._initialized,
                "module_levels": self.config.get_module_levels(),
            }

            for name, logger in self.loggers.items():
                effective_level = self._get_effective_level_for_logger(name)
                info["loggers"][name] = {
                    "level": logging.getLevelName(logger.level),
                    "effective_level": effective_level,
                    "handlers": [type(h).__name__ for h in logger.handlers],
                    "propagate": logger.propagate,
                }

            return info

    def cleanup(self) -> None:
        """
        Clean up all loggers and handlers.

        This method should be called before application shutdown to ensure
        proper cleanup of file handles and other resources.
        """
        with self._lock:
            try:
                # Close and remove all handlers
                self._cleanup_handlers()

                # Clear logger cache
                self.loggers.clear()

                # Reset root logger
                if self.root_logger:
                    # Remove all handlers from root logger
                    for handler in self.root_logger.handlers[:]:
                        self.root_logger.removeHandler(handler)
                        handler.close()

                self._initialized = False

            except Exception as e:
                # Log cleanup errors but don't raise (cleanup should be safe)
                print(f"Warning: Error during logger cleanup: {e}")

    def is_initialized(self) -> bool:
        """
        Check if the logger manager has been initialized.

        Returns:
            bool: True if initialized
        """
        return self._initialized

    def get_managed_loggers(self) -> List[str]:
        """
        Get a list of all managed logger names.

        Returns:
            List[str]: List of logger names
        """
        with self._lock:
            return list(self.loggers.keys())

    # Private helper methods

    def _setup_root_logger(self) -> None:
        """Set up the root logger with configured handlers."""
        try:
            # Get or create root logger
            self.root_logger = logging.getLogger(self.ROOT_LOGGER_NAME)

            # Clear existing handlers
            for handler in self.root_logger.handlers[:]:
                self.root_logger.removeHandler(handler)
                handler.close()

            # Set level
            self.root_logger.setLevel(self.config.get_level_int())
            self.root_logger.propagate = False

            # Add configured handlers
            for handler_type in self.config.get_handlers():
                handler = self._create_handler(handler_type)
                if handler:
                    self.root_logger.addHandler(handler)

            # Set up context injection if enabled
            self._setup_context_injection()

            # Cache the root logger
            self.loggers[self.ROOT_LOGGER_NAME] = self.root_logger

        except Exception as e:
            raise LoggerManagerError(f"Failed to setup root logger: {str(e)}", e)

    def _create_handler(self, handler_type: str) -> Optional[logging.Handler]:
        """
        Create a logging handler of the specified type.

        Args:
            handler_type (str): Type of handler to create

        Returns:
            Optional[logging.Handler]: Created handler or None if type not supported

        Raises:
            LoggerManagerError: If handler creation fails
        """
        try:
            handler_key = f"{handler_type}_{id(self.config)}"

            # Return cached handler if it exists
            if handler_key in self.handlers:
                return self.handlers[handler_key]

            handler = None

            if handler_type == "console":
                handler = self._create_console_handler()
            elif handler_type == "file":
                handler = self._create_file_handler()
            else:
                raise LoggerManagerError(f"Unsupported handler type: {handler_type}")

            if handler:
                # Set formatter based on configured type
                formatter = self._create_formatter()
                handler.setFormatter(formatter)

                # Cache the handler
                self.handlers[handler_key] = handler

            return handler

        except Exception as e:
            # Always wrap with handler-specific error message
            raise LoggerManagerError(
                f"Failed to create {handler_type} handler: {str(e)}", e
            )

    def _create_console_handler(self) -> logging.StreamHandler:
        """Create a console handler with optional color support."""
        # Check if colors are enabled in configuration
        enable_colors = self.config.get_enable_colors()
        color_scheme = self.config.get_color_scheme()
        force_colors = self.config.get_force_colors()

        # Create colored console handler if colors are enabled
        if enable_colors:
            return ColoredConsoleHandler(
                enable_colors=enable_colors,
                color_scheme=color_scheme,
                force_colors=force_colors,
            )
        else:
            # Fall back to standard StreamHandler when colors are disabled
            return logging.StreamHandler()

    def _create_file_handler(self) -> Optional[logging.Handler]:
        """Create a rotating file handler."""
        file_path = self.config.get_file_path()
        if not file_path:
            return None

        try:
            # Ensure directory exists
            path = Path(file_path)
            path.parent.mkdir(parents=True, exist_ok=True)

            # Get configuration for rotating file handler
            max_bytes = self.config.get_max_file_size_bytes()
            backup_count = self.config.get_backup_count()

            # Create standard rotating file handler for compatibility with tests
            handler = logging.handlers.RotatingFileHandler(
                filename=file_path,
                maxBytes=max_bytes,
                backupCount=backup_count,
                encoding="utf-8",
            )

            # Set handler level to DEBUG to allow all messages through
            # Logger-level filtering will handle module-specific levels
            handler.setLevel(logging.DEBUG)

            return handler

        except Exception as e:
            raise LoggerManagerError(
                f"Failed to create file handler for {file_path}: {str(e)}",
                e,
            )

    def _create_formatter(self) -> logging.Formatter:
        """
        Create a logging formatter based on the configured type.

        Returns:
            logging.Formatter: Configured formatter instance

        Raises:
            LoggerManagerError: If formatter creation fails
        """
        try:
            formatter_type = self.config.get_formatter_type()

            if formatter_type == "json":
                # Create JSON formatter with configuration
                return JSONFormatter(
                    fields=self.config.get_json_fields(),
                    pretty_print=self.config.get_json_pretty_print(),
                    custom_fields=self.config.get_json_custom_fields(),
                    timestamp_format=self.config.get_json_timestamp_format(),
                    use_utc=self.config.get_json_use_utc(),
                    include_exception_traceback=self.config.get_json_include_exception_traceback(),
                    max_message_length=self.config.get_json_max_message_length(),
                    field_mapping=self.config.get_json_field_mapping(),
                    ensure_ascii=self.config.get_json_ensure_ascii(),
                )
            elif formatter_type == "standard":
                # Create standard formatter
                return logging.Formatter(self.config.get_format())
            else:
                raise LoggerManagerError(
                    f"Unsupported formatter type: {formatter_type}"
                )

        except Exception as e:
            if isinstance(e, LoggerManagerError):
                raise
            raise LoggerManagerError(f"Failed to create formatter: {str(e)}", e)

    def _setup_context_injection(self) -> None:
        """
        Set up context injection for the root logger based on configuration.
        """
        try:
            # Check if context injection is enabled
            if not self.config.get_enable_context_injection():
                return

            injection_method = self.config.get_context_injection_method()

            if injection_method == "disabled":
                return

            # Initialize context manager if not already done
            if self.context_manager is None:
                self.context_manager = RequestContextManager(
                    auto_generate_request_id=self.config.get_auto_generate_request_id(),
                    request_id_field=self.config.get_request_id_field(),
                    default_context=self.config.get_default_context(),
                    allowed_context_fields=set(self.config.get_context_fields()),
                )

            # Apply context injection based on method
            if injection_method == "filter":
                # Create and add context injection filter
                self.context_injection_filter = ContextInjectionFilter(
                    context_manager=self.context_manager,
                    context_prefix=self.config.get_context_prefix(),
                    include_full_context=self.config.get_include_context_in_json(),
                )
                self.root_logger.addFilter(self.context_injection_filter)

            elif injection_method == "replace":
                # Replace logger class (more comprehensive but invasive)
                if self.root_logger:
                    # Convert existing logger to ContextInjectingLogger
                    self.root_logger.__class__ = ContextInjectingLogger
                    self.root_logger.context_manager = self.context_manager

        except Exception as e:
            # Don't fail initialization if context injection fails
            # Log the error if we have a working logger
            if self.root_logger:
                self.root_logger.warning(
                    f"Failed to set up context injection: {str(e)}"
                )

    def get_context_manager(self) -> Optional[RequestContextManager]:
        """
        Get the request context manager instance.

        Returns:
            Optional[RequestContextManager]: Context manager or None if not initialized
        """
        return self.context_manager

    def set_context_manager(self, context_manager: RequestContextManager) -> None:
        """
        Set a custom context manager instance.

        Args:
            context_manager: RequestContextManager instance to use

        Raises:
            LoggerManagerError: If context manager is invalid
        """
        if not isinstance(context_manager, RequestContextManager):
            raise LoggerManagerError(
                "Context manager must be a RequestContextManager instance"
            )

        with self._lock:
            self.context_manager = context_manager

            # Update context injection if already initialized
            if self._initialized and self.config.get_enable_context_injection():
                self._setup_context_injection()

    def update_context_injection(self) -> None:
        """
        Update context injection configuration.

        This method can be called after configuration changes to apply
        new context injection settings.
        """
        with self._lock:
            if not self._initialized:
                return

            # Remove existing context injection
            if self.context_injection_filter and self.root_logger:
                self.root_logger.removeFilter(self.context_injection_filter)
                self.context_injection_filter = None

            # Re-setup context injection with new configuration
            self._setup_context_injection()

    def _setup_logger_context_injection(
        self, logger: logging.Logger, logger_name: str
    ) -> None:
        """
        Set up context injection for an individual logger.

        Args:
            logger: Logger instance to set up context injection for
            logger_name: Name of the logger
        """
        try:
            # Only set up context injection if enabled and we have a context manager
            if (
                not self.config.get_enable_context_injection()
                or not self.context_manager
            ):
                return

            injection_method = self.config.get_context_injection_method()

            if injection_method == "disabled":
                return

            # For filter method, add a context injection filter to individual loggers
            # This ensures context injection works even when propagate=False
            if injection_method == "filter":
                # Check if logger already has a context injection filter
                has_context_filter = any(
                    isinstance(f, ContextInjectionFilter) for f in logger.filters
                )

                if not has_context_filter:
                    context_filter = ContextInjectionFilter(
                        context_manager=self.context_manager,
                        context_prefix=self.config.get_context_prefix(),
                        include_full_context=self.config.get_include_context_in_json(),
                    )
                    logger.addFilter(context_filter)

        except Exception as e:
            # Don't fail logger creation if context injection setup fails
            if logger:
                logger.warning(
                    f"Failed to set up context injection for logger '{logger_name}': {str(e)}"
                )

    def _get_effective_level_for_logger(self, logger_name: str) -> str:
        """
        Get the effective log level for a logger based on module-specific configuration.

        Args:
            logger_name (str): Name of the logger

        Returns:
            str: Effective log level for the logger
        """
        # For root logger, always use global level
        if logger_name == self.ROOT_LOGGER_NAME:
            return self.config.get_level()

        # Extract module name from logger name
        # Logger names follow pattern: "aim2.module.submodule"
        if logger_name.startswith(f"{self.ROOT_LOGGER_NAME}."):
            module_name = logger_name[len(f"{self.ROOT_LOGGER_NAME}.") :]
            return self.config.get_effective_level(module_name)

        # For non-hierarchical logger names, use global level
        return self.config.get_level()

    def _cleanup_handlers(self) -> None:
        """Clean up all managed handlers."""
        try:
            # Close and remove handlers from all loggers
            for logger in self.loggers.values():
                for handler in logger.handlers[:]:
                    logger.removeHandler(handler)
                    handler.close()

            # Close cached handlers
            for handler in self.handlers.values():
                try:
                    handler.close()
                except:
                    pass  # Ignore errors during cleanup

            self.handlers.clear()

        except Exception as e:
            # Don't raise during cleanup, just log the error
            print(f"Warning: Error during handler cleanup: {e}")

    def create_performance_logger(
        self, name: str, enable_performance_features: bool = True, **perf_kwargs
    ) -> logging.Logger:
        """
        Create a logger optimized for performance monitoring.

        Args:
            name: Logger name
            enable_performance_features: Whether to enable performance-specific features
            **perf_kwargs: Additional performance configuration options

        Returns:
            logging.Logger: Logger configured for performance monitoring

        Raises:
            LoggerManagerError: If logger creation fails
        """
        with self._lock:
            try:
                # Get or create the logger
                logger = self.get_logger(name)

                # Configure for performance monitoring if enabled
                if (
                    enable_performance_features
                    and self.config.get_enable_performance_logging()
                ):
                    # Ensure JSON formatter is used if performance logging is enabled
                    # Performance data is best captured in structured JSON format
                    if self.config.get_formatter_type() != "json":
                        # Log a warning that performance data will be limited in non-JSON format
                        if hasattr(logger, "warning"):
                            logger.warning(
                                "Performance logging works best with JSON formatter. "
                                "Consider setting formatter_type to 'json' for full performance data capture."
                            )

                    # Performance loggers benefit from context injection
                    if self.config.get_enable_context_injection():
                        # Context injection is already set up during initialization
                        pass

                return logger

            except Exception as e:
                raise LoggerManagerError(
                    f"Failed to create performance logger '{name}': {str(e)}", e
                )

    def get_performance_config_summary(self) -> Dict[str, Any]:
        """
        Get a summary of performance logging configuration.

        Returns:
            Dict[str, Any]: Performance configuration summary
        """
        try:
            return {
                "enabled": self.config.get_enable_performance_logging(),
                "thresholds": self.config.get_performance_thresholds(),
                "profiles": self.config.get_performance_profiles(),
                "metrics": self.config.get_performance_metrics(),
                "context_prefix": self.config.get_performance_context_prefix(),
                "auto_operation_naming": self.config.get_performance_auto_operation_naming(),
                "formatter_type": self.config.get_formatter_type(),
                "context_injection_enabled": self.config.get_enable_context_injection(),
            }
        except Exception as e:
            return {"error": f"Failed to get performance config: {str(e)}"}
