"""
Logger Factory Module for AIM2 Project

This module provides a factory pattern implementation for creating loggers with
a simple, clean interface that hides the complexity of LoggerManager and LoggerConfig.
The factory supports singleton pattern for global access and automatic initialization.

Classes:
    LoggerFactoryError: Custom exception for logger factory-related errors
    LoggerFactory: Main factory class for creating loggers

Dependencies:
    - logging: For Python logging system
    - threading: For thread-safe operations
    - typing: For type hints
    - inspect: For automatic module name detection
"""

import logging
import threading
import inspect
from typing import Any, Dict, Optional, Union
from .logger_config import LoggerConfig, LoggerConfigError
from .logger_manager import LoggerManager, LoggerManagerError


class LoggerFactoryError(Exception):
    """
    Custom exception for logger factory-related errors.

    This exception is raised when logger factory operations encounter errors
    such as configuration issues, initialization failures, or logger creation problems.

    Args:
        message (str): Error message describing the issue
        cause (Exception, optional): Original exception that caused this error
    """

    def __init__(self, message: str, cause: Optional[Exception] = None):
        super().__init__(message)
        self.message = message
        self.cause = cause
        if cause:
            self.__cause__ = cause


class LoggerFactory:
    """
    Factory class for creating and managing loggers.

    Provides a simple, clean interface for logger creation while hiding the complexity
    of LoggerManager and LoggerConfig. Supports singleton pattern for global access,
    automatic initialization, and module-specific logger creation.

    Features:
    - Simple interface for getting loggers without direct LoggerManager interaction
    - Singleton pattern for global access
    - Automatic module name detection from calling context
    - Thread-safe operations
    - Integration with existing LoggerConfig and LoggerManager
    - Module-specific logger creation with hierarchical naming
    - Automatic initialization and configuration management

    Attributes:
        _instance (Optional[LoggerFactory]): Singleton instance
        _lock (threading.RLock): Class-level lock for singleton creation
        _manager (Optional[LoggerManager]): Logger manager instance
        _config (Optional[LoggerConfig]): Logger configuration instance
        _instance_lock (threading.RLock): Instance-level lock for thread safety
        _initialized (bool): Whether the factory has been initialized
    """

    # Class-level singleton management
    _instance: Optional["LoggerFactory"] = None
    _lock: threading.RLock = threading.RLock()

    def __init__(self, config: Optional[LoggerConfig] = None):
        """
        Initialize the LoggerFactory.

        Note: Direct instantiation is discouraged. Use get_instance() for singleton access.

        Args:
            config (Optional[LoggerConfig]): Logger configuration instance.
                                           If None, creates default configuration.
        """
        self._config = config or LoggerConfig()
        self._manager: Optional[LoggerManager] = None
        self._instance_lock = threading.RLock()
        self._initialized = False

    @classmethod
    def get_instance(cls, config: Optional[LoggerConfig] = None) -> "LoggerFactory":
        """
        Get or create the singleton LoggerFactory instance.

        This is the recommended way to access the LoggerFactory. The first call
        will create the singleton instance with the provided configuration.
        Subsequent calls will return the same instance (config parameter ignored).

        Args:
            config (Optional[LoggerConfig]): Logger configuration for initial creation.
                                           Ignored if instance already exists.

        Returns:
            LoggerFactory: Singleton LoggerFactory instance

        Raises:
            LoggerFactoryError: If factory creation fails
        """
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    try:
                        cls._instance = cls(config)
                        cls._instance.initialize()
                    except Exception as e:
                        raise LoggerFactoryError(
                            f"Failed to create LoggerFactory singleton: {str(e)}", e
                        )
        return cls._instance

    @classmethod
    def reset_instance(cls) -> None:
        """
        Reset the singleton instance.

        This method is primarily for testing purposes. It cleans up the current
        instance and allows a new one to be created with different configuration.

        Warning: This will affect all code using the factory singleton.
        """
        with cls._lock:
            if cls._instance is not None:
                try:
                    cls._instance.cleanup()
                except Exception:
                    pass  # Ignore cleanup errors during reset
                cls._instance = None

    def initialize(self) -> None:
        """
        Initialize the factory and underlying logger manager.

        This method sets up the LoggerManager with the current configuration
        and ensures the factory is ready for use.

        Raises:
            LoggerFactoryError: If initialization fails
        """
        with self._instance_lock:
            if self._initialized:
                return

            try:
                # Create and initialize the logger manager
                self._manager = LoggerManager(self._config)
                self._manager.initialize()
                self._initialized = True

            except Exception as e:
                if isinstance(e, (LoggerConfigError, LoggerManagerError)):
                    raise LoggerFactoryError(
                        f"Failed to initialize LoggerFactory: {str(e)}", e
                    )
                raise LoggerFactoryError(
                    f"Unexpected error during LoggerFactory initialization: {str(e)}", e
                )

    def get_logger(
        self,
        name: Optional[str] = None,
        module_name: Optional[str] = None,
        auto_detect_module: bool = True,
    ) -> logging.Logger:
        """
        Get or create a logger with the specified name.

        This is the primary interface for getting loggers. It supports automatic
        module name detection from the calling context and creates hierarchical
        logger names based on the project structure.

        Args:
            name (Optional[str]): Explicit logger name. If provided, used as-is.
            module_name (Optional[str]): Module name for hierarchical naming.
            auto_detect_module (bool): Whether to automatically detect module name
                                     from calling context if not provided.

        Returns:
            logging.Logger: Configured logger instance

        Raises:
            LoggerFactoryError: If logger creation fails

        Examples:
            # Get root logger
            logger = factory.get_logger()

            # Get logger with explicit name
            logger = factory.get_logger("my_component")

            # Get module-specific logger
            logger = factory.get_logger(module_name="data_processing")

            # Auto-detect module from calling context (default)
            logger = factory.get_logger()  # Will use calling module name
        """
        with self._instance_lock:
            if not self._initialized:
                self.initialize()

            try:
                # Determine the effective module name
                effective_module_name = self._determine_module_name(
                    name, module_name, auto_detect_module
                )

                # Delegate to the logger manager
                return self._manager.get_logger(
                    name=name, module_name=effective_module_name
                )

            except Exception as e:
                if isinstance(e, LoggerManagerError):
                    raise LoggerFactoryError(f"Failed to create logger: {str(e)}", e)
                raise LoggerFactoryError(
                    f"Unexpected error creating logger: {str(e)}", e
                )

    def get_module_logger(self, module_name: Optional[str] = None) -> logging.Logger:
        """
        Get a logger for a specific module.

        Convenience method for creating module-specific loggers. If module_name
        is not provided, attempts to auto-detect from calling context.

        Args:
            module_name (Optional[str]): Name of the module. If None, auto-detects.

        Returns:
            logging.Logger: Module-specific logger

        Raises:
            LoggerFactoryError: If module logger creation fails

        Examples:
            # Get logger for specific module
            logger = factory.get_module_logger("data_processor")

            # Auto-detect module from calling context
            logger = factory.get_module_logger()
        """
        if module_name is None:
            module_name = self._detect_caller_module()

        if not module_name:
            raise LoggerFactoryError(
                "Could not determine module name and none was provided"
            )

        with self._instance_lock:
            if not self._initialized:
                self.initialize()

            try:
                return self._manager.get_module_logger(module_name)
            except Exception as e:
                if isinstance(e, LoggerManagerError):
                    raise LoggerFactoryError(
                        f"Failed to create module logger: {str(e)}", e
                    )
                raise LoggerFactoryError(
                    f"Unexpected error creating module logger: {str(e)}", e
                )

    def configure(
        self,
        config: Optional[Union[LoggerConfig, Dict[str, Any]]] = None,
        **kwargs: Any,
    ) -> None:
        """
        Configure or reconfigure the logging system.

        This method allows updating the logging configuration after the factory
        has been created. It supports both LoggerConfig objects and dictionary
        configurations.

        Args:
            config (Optional[Union[LoggerConfig, Dict[str, Any]]]): New configuration.
                Can be a LoggerConfig instance or a configuration dictionary.
            **kwargs: Additional configuration parameters to update.

        Raises:
            LoggerFactoryError: If configuration update fails

        Examples:
            # Configure with LoggerConfig object
            new_config = LoggerConfig()
            factory.configure(new_config)

            # Configure with dictionary
            factory.configure({"level": "DEBUG", "handlers": ["console", "file"]})

            # Configure with keyword arguments
            factory.configure(level="INFO", file_path="/var/log/app.log")
        """
        with self._instance_lock:
            try:
                if config is None and not kwargs:
                    return  # Nothing to configure

                # Handle different configuration types
                if isinstance(config, LoggerConfig):
                    new_config = config
                elif isinstance(config, dict):
                    new_config = LoggerConfig()
                    new_config.load_from_dict(config)
                elif config is None:
                    # Use current config and apply kwargs
                    new_config = self._config
                else:
                    raise LoggerFactoryError(
                        "Configuration must be LoggerConfig instance, dict, or None"
                    )

                # Apply keyword argument updates
                if kwargs:
                    new_config.update_config(kwargs)

                # Update factory configuration
                self._config = new_config

                # Reload manager configuration if initialized
                if self._initialized and self._manager:
                    self._manager.reload_configuration(new_config)
                elif not self._initialized:
                    # If not initialized, initialize with new config
                    self.initialize()

            except Exception as e:
                if isinstance(e, (LoggerConfigError, LoggerManagerError)):
                    raise LoggerFactoryError(
                        f"Failed to configure LoggerFactory: {str(e)}", e
                    )
                raise LoggerFactoryError(
                    f"Unexpected error during configuration: {str(e)}", e
                )

    def set_level(self, level: str) -> None:
        """
        Set the logging level for all managed loggers.

        Args:
            level (str): Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)

        Raises:
            LoggerFactoryError: If level setting fails
        """
        with self._instance_lock:
            if not self._initialized:
                self.initialize()

            try:
                self._manager.set_level(level)
            except Exception as e:
                if isinstance(e, LoggerManagerError):
                    raise LoggerFactoryError(f"Failed to set level: {str(e)}", e)
                raise LoggerFactoryError(f"Unexpected error setting level: {str(e)}", e)

    def get_info(self) -> Dict[str, Any]:
        """
        Get information about the factory and managed loggers.

        Returns:
            Dict[str, Any]: Dictionary containing factory and logger information

        Examples:
            info = factory.get_info()
            print(f"Initialized: {info['initialized']}")
            print(f"Loggers: {list(info['loggers'].keys())}")
        """
        with self._instance_lock:
            info = {
                "initialized": self._initialized,
                "singleton_active": self.__class__._instance is not None,
                "config": self._config.to_dict() if self._config else None,
            }

            if self._initialized and self._manager:
                manager_info = self._manager.get_logger_info()
                info.update(manager_info)
            else:
                info.update({"loggers": {}, "handlers": []})

            return info

    def cleanup(self) -> None:
        """
        Clean up the factory and all managed resources.

        This method should be called before application shutdown to ensure
        proper cleanup of file handles and other resources.
        """
        with self._instance_lock:
            try:
                if self._manager:
                    self._manager.cleanup()
                self._initialized = False
            except Exception as e:
                # Use logging if available, otherwise fall back to stderr
                try:
                    import logging

                    logger = logging.getLogger(__name__)
                    logger.warning(f"Error during LoggerFactory cleanup: {e}")
                except:
                    # If logging is not available or fails, use stderr
                    import sys

                    print(
                        f"Warning: Error during LoggerFactory cleanup: {e}",
                        file=sys.stderr,
                    )

    def is_initialized(self) -> bool:
        """
        Check if the factory has been initialized.

        Returns:
            bool: True if the factory is initialized and ready for use
        """
        return self._initialized

    def get_managed_loggers(self) -> Dict[str, logging.Logger]:
        """
        Get all managed loggers.

        Returns:
            Dict[str, logging.Logger]: Dictionary mapping logger names to logger instances

        Raises:
            LoggerFactoryError: If factory is not initialized
        """
        with self._instance_lock:
            if not self._initialized:
                raise LoggerFactoryError("Factory not initialized")

            try:
                logger_names = self._manager.get_managed_loggers()
                return {name: self._manager.get_logger(name) for name in logger_names}
            except Exception as e:
                raise LoggerFactoryError(f"Failed to get managed loggers: {str(e)}", e)

    # Private helper methods

    def _determine_module_name(
        self,
        name: Optional[str],
        module_name: Optional[str],
        auto_detect_module: bool,
    ) -> Optional[str]:
        """
        Determine the effective module name for logger creation.

        Args:
            name: Explicit logger name
            module_name: Explicit module name
            auto_detect_module: Whether to auto-detect module name

        Returns:
            Optional[str]: Effective module name to use
        """
        # If explicit name is provided, don't use module name
        if name is not None:
            return None

        # If module name is explicitly provided, use it
        if module_name is not None:
            return module_name

        # Auto-detect if enabled
        if auto_detect_module:
            return self._detect_caller_module()

        return None

    def _detect_caller_module(self) -> Optional[str]:
        """
        Detect the module name of the calling code.

        Uses the call stack to determine the module name of the code that
        called the factory method.

        Returns:
            Optional[str]: Detected module name or None if detection fails
        """
        frame = None
        try:
            # Get the call stack
            frame = inspect.currentframe()

            # Walk up the stack to find the first frame outside this module
            while frame:
                frame = frame.f_back
                if frame is None:
                    break

                # Get the module name from the frame
                module = inspect.getmodule(frame)
                if module and module.__name__:
                    module_name = module.__name__

                    # Skip frames from this logging package
                    if not module_name.startswith("aim2_project.aim2_utils.logger"):
                        # Clean up the module name for logger hierarchy
                        if module_name.startswith("aim2_project."):
                            # Remove project prefix for cleaner names
                            module_name = module_name[len("aim2_project.") :]

                        # Convert path separators to dots for proper hierarchy
                        module_name = module_name.replace("/", ".").replace("\\", ".")
                        return module_name

            return None

        except Exception:
            # If module detection fails, return None
            return None
        finally:
            # Explicitly clean up frame reference to prevent memory leaks
            del frame


# Convenience functions for easy access


def get_logger(
    name: Optional[str] = None,
    module_name: Optional[str] = None,
    auto_detect_module: bool = True,
) -> logging.Logger:
    """
    Convenience function to get a logger using the singleton factory.

    This is the recommended way for most code to get loggers. It automatically
    uses the singleton LoggerFactory instance and provides a clean interface.

    Args:
        name (Optional[str]): Explicit logger name
        module_name (Optional[str]): Module name for hierarchical naming
        auto_detect_module (bool): Whether to auto-detect module name from context

    Returns:
        logging.Logger: Configured logger instance

    Raises:
        LoggerFactoryError: If logger creation fails

    Examples:
        # Simple usage - auto-detects module
        logger = get_logger()
        logger.info("Hello from my module")

        # Explicit module name
        logger = get_logger(module_name="data_processor")

        # Explicit logger name
        logger = get_logger("my_component")
    """
    factory = LoggerFactory.get_instance()
    return factory.get_logger(name, module_name, auto_detect_module)


def get_module_logger(module_name: Optional[str] = None) -> logging.Logger:
    """
    Convenience function to get a module-specific logger.

    Args:
        module_name (Optional[str]): Module name. If None, auto-detects from context.

    Returns:
        logging.Logger: Module-specific logger

    Raises:
        LoggerFactoryError: If logger creation fails

    Examples:
        # Auto-detect module from calling context
        logger = get_module_logger()

        # Explicit module name
        logger = get_module_logger("data_processor")
    """
    factory = LoggerFactory.get_instance()
    return factory.get_module_logger(module_name)


def configure_logging(
    config: Optional[Union[LoggerConfig, Dict[str, Any]]] = None, **kwargs: Any
) -> None:
    """
    Convenience function to configure the logging system.

    Args:
        config (Optional[Union[LoggerConfig, Dict[str, Any]]]): Configuration to apply
        **kwargs: Additional configuration parameters

    Raises:
        LoggerFactoryError: If configuration fails

    Examples:
        # Configure with dictionary
        configure_logging({"level": "DEBUG", "handlers": ["console"]})

        # Configure with keyword arguments
        configure_logging(level="INFO", file_path="/var/log/app.log")
    """
    factory = LoggerFactory.get_instance()
    factory.configure(config, **kwargs)


def set_logging_level(level: str) -> None:
    """
    Convenience function to set the logging level.

    Args:
        level (str): Logging level to set

    Raises:
        LoggerFactoryError: If level setting fails

    Examples:
        set_logging_level("DEBUG")
        set_logging_level("INFO")
    """
    factory = LoggerFactory.get_instance()
    factory.set_level(level)
