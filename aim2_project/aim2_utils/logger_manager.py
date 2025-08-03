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
                logger.setLevel(self.config.get_level_int())

                # For non-root loggers, enable propagation to root
                if logger_name != self.ROOT_LOGGER_NAME:
                    logger.propagate = True
                else:
                    logger.propagate = False

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

                # Update existing loggers with new level
                new_level = self.config.get_level_int()
                for logger in self.loggers.values():
                    logger.setLevel(new_level)

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

                # Update all existing loggers
                new_level_int = self.config.get_level_int()
                for logger in self.loggers.values():
                    logger.setLevel(new_level_int)

            except Exception as e:
                if isinstance(e, LoggerConfigError):
                    raise LoggerManagerError(f"Failed to set level: {str(e)}", e)
                raise LoggerManagerError(f"Unexpected error setting level: {str(e)}", e)

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
            }

            for name, logger in self.loggers.items():
                info["loggers"][name] = {
                    "level": logging.getLevelName(logger.level),
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
                # Set formatter
                formatter = logging.Formatter(self.config.get_format())
                handler.setFormatter(formatter)

                # Cache the handler
                self.handlers[handler_key] = handler

            return handler

        except Exception as e:
            if isinstance(e, LoggerManagerError):
                raise
            raise LoggerManagerError(
                f"Failed to create {handler_type} handler: {str(e)}", e
            )

    def _create_console_handler(self) -> logging.StreamHandler:
        """Create a console handler."""
        return logging.StreamHandler()

    def _create_file_handler(self) -> Optional[logging.Handler]:
        """Create a file handler (rotating if configured)."""
        file_path = self.config.get_file_path()
        if not file_path:
            return None

        try:
            # Ensure directory exists
            path = Path(file_path)
            path.parent.mkdir(parents=True, exist_ok=True)

            # Create rotating file handler
            max_bytes = self.config.get_max_file_size_bytes()
            backup_count = self.config.get_backup_count()

            handler = logging.handlers.RotatingFileHandler(
                filename=file_path,
                maxBytes=max_bytes,
                backupCount=backup_count,
                encoding="utf-8",
            )

            return handler

        except Exception as e:
            raise LoggerManagerError(
                f"Failed to create file handler for {file_path}: {str(e)}", e
            )

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
