"""
Unit tests for LoggerFactory implementation in the AIM2 project logging framework

This module provides comprehensive unit tests for the LoggerFactory class,
including singleton pattern behavior, logger creation, configuration management,
thread safety, and integration with LoggerConfig and LoggerManager.

Test Coverage:
- LoggerFactory singleton pattern implementation and thread safety
- Factory initialization and cleanup operations
- Logger creation with different naming patterns and configurations
- Module name detection and hierarchical logger naming
- Configuration management and reloading
- Error handling and edge cases
- Integration with LoggerConfig and LoggerManager
- Convenience functions (get_logger, configure_logging, etc.)
- Performance characteristics under concurrent access
- Resource management and memory usage

Test Classes:
    TestLoggerFactorySingleton: Tests singleton pattern behavior and thread safety
    TestLoggerFactoryInitialization: Tests factory initialization and setup
    TestLoggerFactoryLoggerCreation: Tests logger creation functionality
    TestLoggerFactoryModuleDetection: Tests module name detection capabilities
    TestLoggerFactoryConfiguration: Tests configuration management
    TestLoggerFactoryConvenienceFunctions: Tests convenience function implementations
    TestLoggerFactoryErrorHandling: Tests error scenarios and edge cases
    TestLoggerFactoryIntegration: Tests integration with other components
    TestLoggerFactoryPerformance: Tests performance and threading characteristics
    TestLoggerFactoryResourceManagement: Tests resource cleanup and management
"""

import logging
import tempfile
import threading
import time
import pytest
import os
from pathlib import Path
from unittest.mock import Mock, patch
from concurrent.futures import ThreadPoolExecutor, as_completed

# Import the modules to be tested
try:
    from aim2_project.aim2_utils.logger_factory import (
        LoggerFactory,
        LoggerFactoryError,
        get_logger,
        get_module_logger,
        configure_logging,
        set_logging_level,
    )
    from aim2_project.aim2_utils.logger_config import (
        LoggerConfig,
        LoggerConfigError,
    )
    from aim2_project.aim2_utils.logger_manager import (
        LoggerManager,
        LoggerManagerError,
    )
except ImportError:
    # Expected during TDD - tests define the interface
    pass


class TestLoggerFactorySingleton:
    """Test suite for LoggerFactory singleton pattern behavior."""

    def teardown_method(self):
        """Clean up singleton instance after each test."""
        LoggerFactory.reset_instance()

    def test_singleton_creation(self):
        """Test singleton instance creation."""
        factory1 = LoggerFactory.get_instance()
        factory2 = LoggerFactory.get_instance()

        assert factory1 is factory2
        assert isinstance(factory1, LoggerFactory)

    def test_singleton_with_config(self):
        """Test singleton creation with custom configuration."""
        config = LoggerConfig()
        config.set_level("DEBUG")

        factory1 = LoggerFactory.get_instance(config)
        factory2 = LoggerFactory.get_instance()

        assert factory1 is factory2
        assert factory1._config.get_level() == "DEBUG"

    def test_singleton_config_ignored_on_subsequent_calls(self):
        """Test that config parameter is ignored on subsequent get_instance calls."""
        config1 = LoggerConfig()
        config1.set_level("DEBUG")

        config2 = LoggerConfig()
        config2.set_level("ERROR")

        factory1 = LoggerFactory.get_instance(config1)
        factory2 = LoggerFactory.get_instance(config2)

        assert factory1 is factory2
        assert factory1._config.get_level() == "DEBUG"  # First config should be used

    def test_singleton_reset(self):
        """Test singleton instance reset functionality."""
        factory1 = LoggerFactory.get_instance()
        factory1_id = id(factory1)

        LoggerFactory.reset_instance()

        factory2 = LoggerFactory.get_instance()
        factory2_id = id(factory2)

        assert factory1_id != factory2_id
        assert factory1 is not factory2

    def test_singleton_thread_safety(self):
        """Test singleton creation is thread-safe."""
        factories = []
        num_threads = 10

        def create_factory(thread_id, barrier):
            # Wait for all threads to be ready
            barrier.wait()
            factory = LoggerFactory.get_instance()
            factories.append((thread_id, factory))

        # Create barrier to synchronize thread starts
        barrier = threading.Barrier(num_threads)

        # Create and start threads
        threads = []
        for i in range(num_threads):
            thread = threading.Thread(target=create_factory, args=(i, barrier))
            threads.append(thread)
            thread.start()

        # Wait for all threads to complete
        for thread in threads:
            thread.join()

        # Verify all threads got the same instance
        assert len(factories) == num_threads
        first_factory = factories[0][1]
        for thread_id, factory in factories:
            assert factory is first_factory

    def test_singleton_initialization_exception_handling(self):
        """Test singleton handles initialization exceptions."""
        with patch.object(LoggerFactory, "initialize") as mock_init:
            mock_init.side_effect = Exception("Initialization failed")

            with pytest.raises(LoggerFactoryError) as exc_info:
                LoggerFactory.get_instance()

            assert "Failed to create LoggerFactory singleton" in str(exc_info.value)
            assert exc_info.value.cause is not None

    def test_singleton_concurrent_reset_and_creation(self):
        """Test concurrent reset and creation operations."""

        def reset_and_create():
            LoggerFactory.reset_instance()
            return LoggerFactory.get_instance()

        # Run multiple reset/create operations concurrently
        with ThreadPoolExecutor(max_workers=5) as executor:
            futures = [executor.submit(reset_and_create) for _ in range(10)]
            factories = [future.result() for future in as_completed(futures)]

        # All factories should be valid instances (though may be different)
        for factory in factories:
            assert isinstance(factory, LoggerFactory)

    def test_singleton_double_initialization_protection(self):
        """Test singleton protects against double initialization."""
        factory = LoggerFactory.get_instance()

        # Initialize manually
        factory.initialize()
        assert factory.is_initialized()

        # Second initialization should be idempotent
        factory.initialize()
        assert factory.is_initialized()

    def test_singleton_cleanup_during_reset(self):
        """Test that reset properly cleans up the singleton instance."""
        factory = LoggerFactory.get_instance()

        # Create a logger to ensure manager is set up
        logger = factory.get_logger("test_logger")
        assert logger is not None
        assert factory.is_initialized()

        # Reset should clean up (even if cleanup has errors)
        LoggerFactory.reset_instance()

        # New instance should be fresh (but get_instance auto-initializes)
        new_factory = LoggerFactory.get_instance()
        assert new_factory is not factory
        # New factory is auto-initialized by get_instance
        assert new_factory.is_initialized()


class TestLoggerFactoryInitialization:
    """Test suite for LoggerFactory initialization and setup."""

    def teardown_method(self):
        """Clean up singleton instance after each test."""
        LoggerFactory.reset_instance()

    @pytest.fixture
    def factory(self):
        """Create a LoggerFactory instance for testing."""
        return LoggerFactory()

    def test_factory_initialization_default_config(self, factory):
        """Test factory initialization with default configuration."""
        assert not factory.is_initialized()
        assert factory._config is not None
        assert isinstance(factory._config, LoggerConfig)
        assert factory._manager is None

    def test_factory_initialization_custom_config(self):
        """Test factory initialization with custom configuration."""
        config = LoggerConfig()
        config.set_level("ERROR")

        factory = LoggerFactory(config)

        assert factory._config is config
        assert factory._config.get_level() == "ERROR"

    def test_factory_initialize_method(self, factory):
        """Test explicit factory initialization."""
        factory.initialize()

        assert factory.is_initialized()
        assert factory._manager is not None
        assert isinstance(factory._manager, LoggerManager)

    def test_factory_initialize_idempotent(self, factory):
        """Test that initialization is idempotent."""
        factory.initialize()
        manager1 = factory._manager

        factory.initialize()  # Second call
        manager2 = factory._manager

        assert manager1 is manager2
        assert factory.is_initialized()

    def test_factory_initialize_with_invalid_config(self):
        """Test initialization with invalid configuration."""
        config = LoggerConfig()
        config.config["level"] = "INVALID_LEVEL"

        factory = LoggerFactory(config)

        with pytest.raises(LoggerFactoryError) as exc_info:
            factory.initialize()

        assert "Failed to initialize LoggerFactory" in str(exc_info.value)
        assert exc_info.value.cause is not None

    def test_factory_lazy_initialization(self, factory):
        """Test that factory initializes lazily when getting loggers."""
        assert not factory.is_initialized()

        # Getting a logger should trigger initialization
        logger = factory.get_logger("test_logger")

        assert factory.is_initialized()
        assert logger is not None

    def test_factory_initialization_thread_safety(self):
        """Test that initialization is thread-safe."""
        factory = LoggerFactory()
        initialization_results = []

        def initialize_factory(thread_id):
            try:
                factory.initialize()
                initialization_results.append(
                    (thread_id, True, factory.is_initialized())
                )
            except Exception as e:
                initialization_results.append((thread_id, False, str(e)))

        # Start multiple initialization threads
        threads = []
        for i in range(5):
            thread = threading.Thread(target=initialize_factory, args=(i,))
            threads.append(thread)
            thread.start()

        # Wait for all threads
        for thread in threads:
            thread.join()

        # All should succeed and factory should be initialized
        assert len(initialization_results) == 5
        for thread_id, success, result in initialization_results:
            assert success, f"Thread {thread_id} failed: {result}"
            assert result is True, f"Thread {thread_id} factory not initialized"

    def test_factory_cleanup(self, factory):
        """Test factory cleanup functionality."""
        factory.initialize()
        logger = factory.get_logger("cleanup_test")

        assert factory.is_initialized()
        assert logger is not None

        factory.cleanup()

        assert not factory.is_initialized()

    def test_factory_cleanup_with_errors(self, factory):
        """Test factory cleanup handles manager cleanup errors."""
        factory.initialize()

        # Mock manager cleanup to raise an exception
        with patch.object(factory._manager, "cleanup") as mock_cleanup:
            mock_cleanup.side_effect = Exception("Cleanup failed")

            # Cleanup should not raise exception but prints warning
            factory.cleanup()

            # Factory should still be initialized because cleanup failed in try block
            assert factory.is_initialized()

    def test_factory_manager_creation_error(self, factory):
        """Test handling of LoggerManager creation errors."""
        with patch(
            "aim2_project.aim2_utils.logger_factory.LoggerManager"
        ) as mock_manager_class:
            mock_manager_class.side_effect = Exception("Manager creation failed")

            with pytest.raises(LoggerFactoryError) as exc_info:
                factory.initialize()

            assert "Unexpected error during LoggerFactory initialization" in str(
                exc_info.value
            )


class TestLoggerFactoryLoggerCreation:
    """Test suite for LoggerFactory logger creation functionality."""

    def teardown_method(self):
        """Clean up singleton instance after each test."""
        LoggerFactory.reset_instance()

    @pytest.fixture
    def factory(self):
        """Create a LoggerFactory instance for testing."""
        return LoggerFactory()

    def test_get_logger_root(self, factory):
        """Test getting root logger."""
        logger = factory.get_logger()

        assert logger is not None
        assert isinstance(logger, logging.Logger)
        assert factory.is_initialized()

    def test_get_logger_with_explicit_name(self, factory):
        """Test getting logger with explicit name."""
        logger = factory.get_logger("explicit_logger")

        assert logger is not None
        assert logger.name == "explicit_logger"

    def test_get_logger_with_module_name(self, factory):
        """Test getting logger with module name."""
        logger = factory.get_logger(module_name="test_module")

        assert logger is not None
        # Logger name should include module hierarchy
        assert "test_module" in logger.name

    def test_get_logger_explicit_name_overrides_module(self, factory):
        """Test that explicit name overrides module name."""
        logger = factory.get_logger("explicit_name", module_name="ignored_module")

        assert logger.name == "explicit_name"

    def test_get_logger_caching(self, factory):
        """Test that loggers are cached and reused."""
        logger1 = factory.get_logger("cached_logger")
        logger2 = factory.get_logger("cached_logger")

        assert logger1 is logger2

    def test_get_logger_auto_detect_module_disabled(self, factory):
        """Test getting logger with auto-detection disabled."""
        logger = factory.get_logger(auto_detect_module=False)

        assert logger is not None
        # Should get root logger when no name/module specified and auto-detect disabled

    def test_get_logger_initialization_error(self, factory):
        """Test get_logger handles initialization errors."""
        with patch.object(factory, "initialize") as mock_init:
            mock_init.side_effect = LoggerFactoryError("Init failed")

            with pytest.raises(LoggerFactoryError):
                factory.get_logger("test_logger")

    def test_get_logger_manager_error(self, factory):
        """Test get_logger handles LoggerManager errors."""
        factory.initialize()

        with patch.object(factory._manager, "get_logger") as mock_get_logger:
            mock_get_logger.side_effect = LoggerManagerError("Manager failed")

            with pytest.raises(LoggerFactoryError) as exc_info:
                factory.get_logger("test_logger")

            assert "Failed to create logger" in str(exc_info.value)

    def test_get_logger_unexpected_error(self, factory):
        """Test get_logger handles unexpected errors."""
        factory.initialize()

        with patch.object(factory._manager, "get_logger") as mock_get_logger:
            mock_get_logger.side_effect = RuntimeError("Unexpected error")

            with pytest.raises(LoggerFactoryError) as exc_info:
                factory.get_logger("test_logger")

            assert "Unexpected error creating logger" in str(exc_info.value)

    def test_get_module_logger_explicit_name(self, factory):
        """Test get_module_logger with explicit module name."""
        logger = factory.get_module_logger("explicit_module")

        assert logger is not None
        assert "explicit_module" in logger.name

    def test_get_module_logger_auto_detect(self, factory):
        """Test get_module_logger with auto-detection."""
        # This will use the current module name
        logger = factory.get_module_logger()

        assert logger is not None
        # Logger name should reflect the test module
        assert "test_logger_factory" in logger.name or "unit" in logger.name

    def test_get_module_logger_no_module_name(self, factory):
        """Test get_module_logger when module name cannot be determined."""
        with patch.object(factory, "_detect_caller_module") as mock_detect:
            mock_detect.return_value = None

            with pytest.raises(LoggerFactoryError) as exc_info:
                factory.get_module_logger()

            assert "Could not determine module name" in str(exc_info.value)

    def test_get_module_logger_manager_error(self, factory):
        """Test get_module_logger handles LoggerManager errors."""
        factory.initialize()

        with patch.object(
            factory._manager, "get_module_logger"
        ) as mock_get_module_logger:
            mock_get_module_logger.side_effect = LoggerManagerError("Manager failed")

            with pytest.raises(LoggerFactoryError) as exc_info:
                factory.get_module_logger("test_module")

            assert "Failed to create module logger" in str(exc_info.value)

    def test_get_managed_loggers(self, factory):
        """Test getting all managed loggers."""
        # Create some loggers
        logger1 = factory.get_logger("logger1")
        logger2 = factory.get_logger("logger2")

        managed_loggers = factory.get_managed_loggers()

        assert isinstance(managed_loggers, dict)
        assert "logger1" in managed_loggers
        assert "logger2" in managed_loggers
        assert managed_loggers["logger1"] is logger1
        assert managed_loggers["logger2"] is logger2

    def test_get_managed_loggers_not_initialized(self, factory):
        """Test get_managed_loggers when factory not initialized."""
        with pytest.raises(LoggerFactoryError) as exc_info:
            factory.get_managed_loggers()

        assert "Factory not initialized" in str(exc_info.value)

    def test_get_managed_loggers_error(self, factory):
        """Test get_managed_loggers handles errors."""
        factory.initialize()

        with patch.object(factory._manager, "get_managed_loggers") as mock_get_managed:
            mock_get_managed.side_effect = Exception("Manager error")

            with pytest.raises(LoggerFactoryError) as exc_info:
                factory.get_managed_loggers()

            assert "Failed to get managed loggers" in str(exc_info.value)


class TestLoggerFactoryModuleDetection:
    """Test suite for LoggerFactory module name detection capabilities."""

    def teardown_method(self):
        """Clean up singleton instance after each test."""
        LoggerFactory.reset_instance()

    @pytest.fixture
    def factory(self):
        """Create a LoggerFactory instance for testing."""
        return LoggerFactory()

    def test_detect_caller_module_from_test(self, factory):
        """Test module detection from test context."""
        detected_module = factory._detect_caller_module()

        assert detected_module is not None
        # Should detect this test module
        assert "test_logger_factory" in detected_module or "unit" in detected_module

    def test_detect_caller_module_with_aim2_project_prefix(self, factory):
        """Test module detection removes aim2_project prefix."""
        with patch("inspect.currentframe") as mock_frame:
            # Create mock frame stack
            mock_current_frame = Mock()
            mock_parent_frame = Mock()
            mock_module = Mock()
            mock_module.__name__ = "aim2_project.data.processor"

            mock_current_frame.f_back = mock_parent_frame
            mock_parent_frame.f_back = None

            mock_frame.return_value = mock_current_frame

            with patch("inspect.getmodule") as mock_getmodule:
                mock_getmodule.return_value = mock_module

                detected_module = factory._detect_caller_module()

                assert detected_module == "data.processor"

    def test_detect_caller_module_skips_logger_modules(self, factory):
        """Test module detection skips logging framework modules."""
        with patch("inspect.currentframe") as mock_frame:
            # Create mock frame stack with logging module
            mock_current_frame = Mock()
            mock_logger_frame = Mock()
            mock_app_frame = Mock()

            mock_logger_module = Mock()
            mock_logger_module.__name__ = "aim2_project.aim2_utils.logger_factory"

            mock_app_module = Mock()
            mock_app_module.__name__ = "aim2_project.my_app"

            mock_current_frame.f_back = mock_logger_frame
            mock_logger_frame.f_back = mock_app_frame
            mock_app_frame.f_back = None

            mock_frame.return_value = mock_current_frame

            with patch("inspect.getmodule") as mock_getmodule:

                def side_effect(frame):
                    if frame is mock_logger_frame:
                        return mock_logger_module
                    elif frame is mock_app_frame:
                        return mock_app_module
                    return None

                mock_getmodule.side_effect = side_effect

                detected_module = factory._detect_caller_module()

                assert detected_module == "my_app"

    def test_detect_caller_module_path_separator_conversion(self, factory):
        """Test module detection converts path separators to dots."""
        with patch("inspect.currentframe") as mock_frame:
            mock_current_frame = Mock()
            mock_parent_frame = Mock()
            mock_module = Mock()
            mock_module.__name__ = "aim2_project.data/processing\\utilities"

            mock_current_frame.f_back = mock_parent_frame
            mock_parent_frame.f_back = None

            mock_frame.return_value = mock_current_frame

            with patch("inspect.getmodule") as mock_getmodule:
                mock_getmodule.return_value = mock_module

                detected_module = factory._detect_caller_module()

                assert detected_module == "data.processing.utilities"

    def test_detect_caller_module_exception_handling(self, factory):
        """Test module detection handles exceptions gracefully."""
        with patch("inspect.currentframe") as mock_frame:
            mock_frame.side_effect = Exception("Frame error")

            detected_module = factory._detect_caller_module()

            assert detected_module is None

    def test_detect_caller_module_no_frames(self, factory):
        """Test module detection when no frames available."""
        with patch("inspect.currentframe") as mock_frame:
            mock_frame.return_value = None

            detected_module = factory._detect_caller_module()

            assert detected_module is None

    def test_detect_caller_module_no_module(self, factory):
        """Test module detection when frame has no module."""
        with patch("inspect.currentframe") as mock_frame:
            mock_current_frame = Mock()
            mock_parent_frame = Mock()

            mock_current_frame.f_back = mock_parent_frame
            mock_parent_frame.f_back = None

            mock_frame.return_value = mock_current_frame

            with patch("inspect.getmodule") as mock_getmodule:
                mock_getmodule.return_value = None

                detected_module = factory._detect_caller_module()

                assert detected_module is None

    def test_determine_module_name_with_explicit_name(self, factory):
        """Test _determine_module_name with explicit logger name."""
        result = factory._determine_module_name(
            name="explicit_logger",
            module_name="ignored_module",
            auto_detect_module=True,
        )

        assert result is None  # Explicit name should override module detection

    def test_determine_module_name_with_explicit_module(self, factory):
        """Test _determine_module_name with explicit module name."""
        result = factory._determine_module_name(
            name=None, module_name="explicit_module", auto_detect_module=True
        )

        assert result == "explicit_module"

    def test_determine_module_name_with_auto_detect(self, factory):
        """Test _determine_module_name with auto-detection enabled."""
        with patch.object(factory, "_detect_caller_module") as mock_detect:
            mock_detect.return_value = "detected_module"

            result = factory._determine_module_name(
                name=None, module_name=None, auto_detect_module=True
            )

            assert result == "detected_module"

    def test_determine_module_name_no_auto_detect(self, factory):
        """Test _determine_module_name with auto-detection disabled."""
        result = factory._determine_module_name(
            name=None, module_name=None, auto_detect_module=False
        )

        assert result is None


class TestLoggerFactoryConfiguration:
    """Test suite for LoggerFactory configuration management."""

    def teardown_method(self):
        """Clean up singleton instance after each test."""
        LoggerFactory.reset_instance()

    @pytest.fixture
    def factory(self):
        """Create a LoggerFactory instance for testing."""
        return LoggerFactory()

    @pytest.fixture
    def temp_dir(self):
        """Create a temporary directory for test files."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            yield Path(tmp_dir)

    def test_configure_with_logger_config(self, factory):
        """Test configuration with LoggerConfig object."""
        new_config = LoggerConfig()
        new_config.set_level("ERROR")

        factory.configure(new_config)

        assert factory._config is new_config
        assert factory._config.get_level() == "ERROR"

    def test_configure_with_dictionary(self, factory):
        """Test configuration with dictionary."""
        config_dict = {
            "level": "WARNING",
            "handlers": ["console"],
            "format": "%(name)s - %(message)s",
        }

        factory.configure(config_dict)

        assert factory._config.get_level() == "WARNING"

    def test_configure_with_kwargs(self, factory):
        """Test configuration with keyword arguments."""
        factory.configure(level="DEBUG", handlers=["console"])

        assert factory._config.get_level() == "DEBUG"

    def test_configure_with_dict_and_kwargs(self, factory, temp_dir):
        """Test configuration with both dict and kwargs."""
        config_dict = {"level": "INFO"}
        file_path = str(temp_dir / "test.log")

        factory.configure(config_dict, file_path=file_path)

        assert factory._config.get_level() == "INFO"
        assert factory._config.get_file_path() == file_path

    def test_configure_none_parameters(self, factory):
        """Test configure with no parameters does nothing."""
        original_config = factory._config

        factory.configure()

        assert factory._config is original_config

    def test_configure_invalid_config_type(self, factory):
        """Test configure with invalid configuration type."""
        with pytest.raises(LoggerFactoryError) as exc_info:
            factory.configure("invalid_config")

        assert "Configuration must be LoggerConfig instance, dict, or None" in str(
            exc_info.value
        )

    def test_configure_with_invalid_values(self, factory):
        """Test configure with invalid configuration values."""
        with pytest.raises(LoggerFactoryError):
            factory.configure({"level": "INVALID_LEVEL"})

    def test_configure_reloads_manager_if_initialized(self, factory):
        """Test configure reloads manager configuration if initialized."""
        factory.initialize()
        original_manager = factory._manager

        new_config = LoggerConfig()
        new_config.set_level("ERROR")
        factory.configure(new_config)

        assert factory._manager is original_manager
        # Manager should have been reloaded with new config

    def test_configure_initializes_if_not_initialized(self, factory):
        """Test configure initializes factory if not initialized."""
        assert not factory.is_initialized()

        factory.configure({"level": "DEBUG"})

        assert factory.is_initialized()

    def test_configure_error_handling(self, factory):
        """Test configure handles configuration errors."""
        factory.initialize()

        with patch.object(factory._manager, "reload_configuration") as mock_reload:
            mock_reload.side_effect = LoggerManagerError("Reload failed")

            with pytest.raises(LoggerFactoryError) as exc_info:
                factory.configure({"level": "DEBUG"})

            assert "Failed to configure LoggerFactory" in str(exc_info.value)

    def test_set_level(self, factory):
        """Test setting logging level."""
        factory.set_level("ERROR")

        assert factory.is_initialized()
        # Level should be set through manager

    def test_set_level_not_initialized(self, factory):
        """Test set_level initializes factory if needed."""
        assert not factory.is_initialized()

        factory.set_level("WARNING")

        assert factory.is_initialized()

    def test_set_level_error_handling(self, factory):
        """Test set_level handles errors."""
        factory.initialize()

        with patch.object(factory._manager, "set_level") as mock_set_level:
            mock_set_level.side_effect = LoggerManagerError("Set level failed")

            with pytest.raises(LoggerFactoryError) as exc_info:
                factory.set_level("DEBUG")

            assert "Failed to set level" in str(exc_info.value)

    def test_get_info(self, factory):
        """Test getting factory information."""
        info = factory.get_info()

        assert isinstance(info, dict)
        assert "initialized" in info
        assert "singleton_active" in info
        assert "config" in info

    def test_get_info_initialized(self, factory):
        """Test get_info when factory is initialized."""
        factory.initialize()
        factory.get_logger("test_logger")

        info = factory.get_info()

        assert info["initialized"] is True
        assert "loggers" in info
        assert "handlers" in info

    def test_get_info_not_initialized(self, factory):
        """Test get_info when factory is not initialized."""
        info = factory.get_info()

        assert info["initialized"] is False
        assert info["loggers"] == {}
        assert info["handlers"] == []


class TestLoggerFactoryConvenienceFunctions:
    """Test suite for LoggerFactory convenience functions."""

    def teardown_method(self):
        """Clean up singleton instance after each test."""
        LoggerFactory.reset_instance()

    def test_get_logger_convenience_function(self):
        """Test get_logger convenience function."""
        logger = get_logger("convenience_test")

        assert logger is not None
        assert isinstance(logger, logging.Logger)
        assert logger.name == "convenience_test"

    def test_get_logger_convenience_with_module_name(self):
        """Test get_logger convenience function with module name."""
        logger = get_logger(module_name="convenience_module")

        assert logger is not None
        assert "convenience_module" in logger.name

    def test_get_logger_convenience_auto_detect(self):
        """Test get_logger convenience function with auto-detection."""
        logger = get_logger()

        assert logger is not None
        # Should detect current module context

    def test_get_module_logger_convenience_function(self):
        """Test get_module_logger convenience function."""
        logger = get_module_logger("convenience_module")

        assert logger is not None
        assert "convenience_module" in logger.name

    def test_get_module_logger_convenience_auto_detect(self):
        """Test get_module_logger convenience function with auto-detection."""
        logger = get_module_logger()

        assert logger is not None
        # Should detect current module context

    def test_configure_logging_convenience_function(self):
        """Test configure_logging convenience function."""
        config_dict = {"level": "ERROR", "handlers": ["console"]}

        configure_logging(config_dict)

        # Get factory and verify configuration
        factory = LoggerFactory.get_instance()
        assert factory._config.get_level() == "ERROR"

    def test_configure_logging_convenience_with_kwargs(self):
        """Test configure_logging convenience function with kwargs."""
        configure_logging(level="DEBUG", handlers=["console"])

        factory = LoggerFactory.get_instance()
        assert factory._config.get_level() == "DEBUG"

    def test_set_logging_level_convenience_function(self):
        """Test set_logging_level convenience function."""
        set_logging_level("CRITICAL")

        factory = LoggerFactory.get_instance()
        assert factory.is_initialized()

    def test_convenience_functions_use_singleton(self):
        """Test that convenience functions use the same singleton instance."""
        # Use different convenience functions
        logger1 = get_logger("test1")
        configure_logging(level="WARNING")
        get_module_logger("test2")
        set_logging_level("ERROR")

        # All should use the same factory instance
        factory = LoggerFactory.get_instance()
        assert factory.is_initialized()

        # Verify loggers come from same factory
        managed_loggers = factory.get_managed_loggers()
        assert "test1" in managed_loggers
        assert managed_loggers["test1"] is logger1

    def test_convenience_functions_error_propagation(self):
        """Test that convenience functions properly propagate errors."""
        with pytest.raises(LoggerFactoryError):
            configure_logging({"level": "INVALID_LEVEL"})

    def test_convenience_functions_thread_safety(self):
        """Test that convenience functions are thread-safe."""

        def use_convenience_functions(thread_id):
            logger = get_logger(f"thread_{thread_id}")
            configure_logging(level="INFO")
            module_logger = get_module_logger(f"module_{thread_id}")
            set_logging_level("DEBUG")
            return logger, module_logger

        # Run from multiple threads
        with ThreadPoolExecutor(max_workers=5) as executor:
            futures = [executor.submit(use_convenience_functions, i) for i in range(5)]
            results = [future.result() for future in as_completed(futures)]

        # All should succeed
        assert len(results) == 5
        for logger, module_logger in results:
            assert isinstance(logger, logging.Logger)
            assert isinstance(module_logger, logging.Logger)


class TestLoggerFactoryErrorHandling:
    """Test suite for LoggerFactory error scenarios and edge cases."""

    def teardown_method(self):
        """Clean up singleton instance after each test."""
        LoggerFactory.reset_instance()

    @pytest.fixture
    def factory(self):
        """Create a LoggerFactory instance for testing."""
        return LoggerFactory()

    def test_factory_exception_chaining(self, factory):
        """Test proper exception chaining in LoggerFactory."""
        with patch.object(LoggerConfig, "load_from_dict") as mock_load:
            original_error = LoggerConfigError("Config validation failed")
            mock_load.side_effect = original_error

            with pytest.raises(LoggerFactoryError) as exc_info:
                factory.configure({"level": "DEBUG"})

            assert exc_info.value.cause is not None

    def test_factory_concurrent_access_safety(self):
        """Test factory handles concurrent access safely."""
        factory = LoggerFactory()
        errors = []
        loggers = []

        def concurrent_operations(thread_id):
            try:
                # Mix different operations
                if thread_id % 2 == 0:
                    logger = factory.get_logger(f"concurrent_{thread_id}")
                    loggers.append(logger)
                else:
                    factory.configure(level="INFO")
                    logger = factory.get_module_logger(f"module_{thread_id}")
                    loggers.append(logger)
            except Exception as e:
                errors.append((thread_id, e))

        # Run concurrent operations
        threads = []
        for i in range(10):
            thread = threading.Thread(target=concurrent_operations, args=(i,))
            threads.append(thread)
            thread.start()

        for thread in threads:
            thread.join()

        # Should not have errors and should have created loggers
        assert len(errors) == 0, f"Concurrent access errors: {errors}"
        assert len(loggers) == 10

    def test_factory_resource_exhaustion_handling(self, factory):
        """Test factory behavior under resource exhaustion."""
        with patch(
            "aim2_project.aim2_utils.logger_factory.LoggerManager"
        ) as mock_manager_class:
            mock_manager_class.side_effect = MemoryError("Out of memory")

            with pytest.raises(LoggerFactoryError) as exc_info:
                factory.initialize()

            assert "Unexpected error during LoggerFactory initialization" in str(
                exc_info.value
            )

    def test_factory_cleanup_with_partial_initialization(self, factory):
        """Test cleanup when factory is partially initialized."""
        # Partially initialize
        factory._config = LoggerConfig()
        factory._initialized = True
        factory._manager = None  # Simulate partial initialization

        # Cleanup should not fail
        factory.cleanup()
        assert not factory.is_initialized()

    def test_factory_invalid_module_detection(self, factory):
        """Test factory handles invalid module detection gracefully."""
        with patch("inspect.currentframe") as mock_frame:
            mock_frame.side_effect = Exception("Detection failed")

            # Should not raise exception, just return None
            result = factory._detect_caller_module()
            assert result is None

    def test_factory_circular_dependency_prevention(self, factory):
        """Test factory prevents circular dependencies during logger creation."""
        # This is more of a design test - the factory should not create
        # circular dependencies between loggers
        logger1 = factory.get_logger("parent")
        logger2 = factory.get_logger("parent.child")
        logger3 = factory.get_logger("parent.child.grandchild")

        # All should be separate instances
        assert logger1 is not logger2
        assert logger2 is not logger3
        assert logger1 is not logger3

    def test_factory_memory_leak_prevention(self, factory):
        """Test factory doesn't create memory leaks with many loggers."""
        import gc

        # Create many loggers
        loggers = []
        for i in range(100):
            logger = factory.get_logger(f"memory_test_{i}")
            loggers.append(logger)

        # Clear local references
        del loggers
        gc.collect()

        # Factory should still work
        new_logger = factory.get_logger("after_gc")
        assert new_logger is not None

    def test_factory_config_validation_errors(self, factory):
        """Test factory handles configuration validation errors."""
        with patch.object(LoggerConfig, "load_from_dict") as mock_load:
            mock_load.side_effect = LoggerConfigError("Validation failed")

            with pytest.raises(LoggerFactoryError) as exc_info:
                factory.configure({"level": "DEBUG"})

            assert "Failed to configure LoggerFactory" in str(exc_info.value)

    def test_factory_manager_operation_errors(self, factory):
        """Test factory handles LoggerManager operation errors."""
        factory.initialize()

        with patch.object(factory._manager, "get_logger") as mock_get_logger:
            mock_get_logger.side_effect = RuntimeError("Unexpected manager error")

            with pytest.raises(LoggerFactoryError) as exc_info:
                factory.get_logger("error_test")

            assert "Unexpected error creating logger" in str(exc_info.value)

    def test_factory_state_consistency_after_errors(self, factory):
        """Test factory maintains consistent state after errors."""
        # Initialize factory
        factory.initialize()
        assert factory.is_initialized()

        # Cause an error during logger creation
        with patch.object(factory._manager, "get_logger") as mock_get_logger:
            mock_get_logger.side_effect = Exception("Error")

            with pytest.raises(LoggerFactoryError):
                factory.get_logger("error_test")

        # Factory should still be initialized and functional
        assert factory.is_initialized()

        # Should be able to create loggers after error
        logger = factory.get_logger("recovery_test")
        assert logger is not None


class TestLoggerFactoryIntegration:
    """Test suite for LoggerFactory integration with other components."""

    def teardown_method(self):
        """Clean up singleton instance after each test."""
        LoggerFactory.reset_instance()

    @pytest.fixture
    def temp_dir(self):
        """Create a temporary directory for test files."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            yield Path(tmp_dir)

    def test_integration_with_logger_config(self, temp_dir):
        """Test LoggerFactory integration with LoggerConfig."""
        config = LoggerConfig()
        config.load_from_dict(
            {
                "level": "DEBUG",
                "handlers": ["console", "file"],
                "file_path": str(temp_dir / "integration.log"),
                "format": "%(name)s - %(levelname)s - %(message)s",
            }
        )

        factory = LoggerFactory(config)
        logger = factory.get_logger("integration_test")

        assert logger is not None
        assert factory._config is config
        assert factory._config.get_level() == "DEBUG"

    def test_integration_with_logger_manager(self):
        """Test LoggerFactory integration with LoggerManager."""
        factory = LoggerFactory()
        logger = factory.get_logger("manager_integration")

        assert factory._manager is not None
        assert isinstance(factory._manager, LoggerManager)
        assert logger is not None

    def test_integration_logger_hierarchy(self):
        """Test LoggerFactory creates proper logger hierarchy."""
        factory = LoggerFactory()

        root_logger = factory.get_logger()
        module_logger = factory.get_module_logger("integration.test")
        nested_logger = factory.get_module_logger("integration.test.nested")

        # Verify hierarchy relationships
        assert root_logger is not module_logger
        assert module_logger is not nested_logger

        # Names should reflect hierarchy
        assert "integration.test" in module_logger.name
        assert "integration.test.nested" in nested_logger.name

    def test_integration_configuration_propagation(self):
        """Test configuration changes propagate through the system."""
        factory = LoggerFactory()
        logger1 = factory.get_logger("config_test1")

        # Change configuration
        factory.configure(level="ERROR")

        # Create new logger after config change
        logger2 = factory.get_logger("config_test2")

        # Both loggers should reflect new configuration
        assert logger1.level == logging.ERROR
        assert logger2.level == logging.ERROR

    def test_integration_with_environment_variables(self):
        """Test LoggerFactory integration with environment variable overrides."""
        with patch.dict(
            os.environ,
            {
                "AIM2_LOGGING_LEVEL": "WARNING",
                "AIM2_LOGGING_FORMAT": "%(name)s: %(message)s",
            },
        ):
            factory = LoggerFactory()
            factory.configure({})  # Load empty config to trigger env var reading

            assert factory._config.get_level() == "WARNING"

    def test_integration_multiple_factories(self):
        """Test behavior with multiple factory instances (non-singleton)."""
        # Create non-singleton instances directly
        config1 = LoggerConfig()
        config1.set_level("DEBUG")

        config2 = LoggerConfig()
        config2.set_level("ERROR")

        factory1 = LoggerFactory(config1)
        factory2 = LoggerFactory(config2)

        logger1 = factory1.get_logger("multi_test1")
        logger2 = factory2.get_logger("multi_test2")

        # Loggers should have different configurations
        assert logger1.level == logging.DEBUG
        assert logger2.level == logging.ERROR

    def test_integration_cleanup_propagation(self):
        """Test cleanup propagates through all components."""
        factory = LoggerFactory()
        logger = factory.get_logger("cleanup_test")

        # Verify components are set up
        assert factory.is_initialized()
        assert factory._manager is not None
        assert logger is not None

        # Cleanup should propagate
        factory.cleanup()

        assert not factory.is_initialized()

    def test_integration_error_propagation(self):
        """Test errors propagate correctly through component layers."""
        factory = LoggerFactory()

        # Create a configuration error at the LoggerConfig level
        with patch.object(LoggerConfig, "validate_config") as mock_validate:
            mock_validate.side_effect = LoggerConfigError("Config error")

            with pytest.raises(LoggerFactoryError) as exc_info:
                factory.configure({"level": "DEBUG"})

            # Error should be wrapped and chained
            assert "Failed to configure LoggerFactory" in str(exc_info.value)
            assert isinstance(exc_info.value.cause, LoggerConfigError)

    def test_integration_thread_safety_across_components(self):
        """Test thread safety across all integrated components."""
        factory = LoggerFactory()
        results = []

        def threaded_operations(thread_id):
            try:
                # Perform various operations
                logger = factory.get_logger(f"thread_{thread_id}")
                factory.configure(level="INFO")
                module_logger = factory.get_module_logger(f"module_{thread_id}")
                info = factory.get_info()

                results.append(
                    {
                        "thread_id": thread_id,
                        "logger": logger,
                        "module_logger": module_logger,
                        "info": info,
                        "success": True,
                    }
                )
            except Exception as e:
                results.append(
                    {"thread_id": thread_id, "error": str(e), "success": False}
                )

        # Run multiple threads
        threads = []
        for i in range(5):
            thread = threading.Thread(target=threaded_operations, args=(i,))
            threads.append(thread)
            thread.start()

        for thread in threads:
            thread.join()

        # All operations should succeed
        assert len(results) == 5
        for result in results:
            assert result[
                "success"
            ], f"Thread {result.get('thread_id')} failed: {result.get('error')}"


class TestLoggerFactoryPerformance:
    """Test suite for LoggerFactory performance and threading characteristics."""

    def teardown_method(self):
        """Clean up singleton instance after each test."""
        LoggerFactory.reset_instance()

    def test_logger_creation_performance(self):
        """Test logger creation performance."""
        factory = LoggerFactory()

        start_time = time.time()

        # Create many loggers
        loggers = []
        for i in range(100):
            logger = factory.get_logger(f"perf_test_{i}")
            loggers.append(logger)

        end_time = time.time()
        duration = end_time - start_time

        # Should create loggers quickly
        assert duration < 1.0  # 100 loggers in less than 1 second
        assert len(loggers) == 100

    def test_singleton_access_performance(self):
        """Test singleton access performance."""
        start_time = time.time()

        # Access singleton many times
        factories = []
        for i in range(1000):
            factory = LoggerFactory.get_instance()
            factories.append(factory)

        end_time = time.time()
        duration = end_time - start_time

        # Should be very fast
        assert duration < 0.1  # 1000 accesses in less than 0.1 seconds

        # All should be the same instance
        first_factory = factories[0]
        for factory in factories:
            assert factory is first_factory

    def test_concurrent_logger_creation_performance(self):
        """Test performance of concurrent logger creation."""
        factory = LoggerFactory()
        all_loggers = []

        def create_loggers(thread_id, count=50):
            loggers = []
            for i in range(count):
                logger = factory.get_logger(f"thread_{thread_id}_logger_{i}")
                loggers.append(logger)
            return loggers

        start_time = time.time()

        # Use thread pool for concurrent creation
        with ThreadPoolExecutor(max_workers=4) as executor:
            futures = [executor.submit(create_loggers, i) for i in range(4)]

            for future in as_completed(futures):
                loggers = future.result()
                all_loggers.extend(loggers)

        end_time = time.time()
        duration = end_time - start_time

        # Should complete reasonably quickly
        assert (
            duration < 2.0
        )  # 200 loggers (4 threads * 50 each) in less than 2 seconds
        assert len(all_loggers) == 200

    def test_configuration_change_performance(self):
        """Test performance of configuration changes."""
        factory = LoggerFactory()

        # Create some loggers first
        [factory.get_logger(f"config_perf_{i}") for i in range(10)]

        start_time = time.time()

        # Perform multiple configuration changes
        for i in range(50):
            level = ["DEBUG", "INFO", "WARNING", "ERROR"][i % 4]
            factory.configure(level=level)

        end_time = time.time()
        duration = end_time - start_time

        # Should handle config changes efficiently
        assert duration < 1.0  # 50 config changes in less than 1 second

    def test_memory_usage_with_many_loggers(self):
        """Test memory usage doesn't grow excessively with many loggers."""
        import gc

        factory = LoggerFactory()

        # Create many loggers
        for i in range(1000):
            factory.get_logger(f"memory_test_{i}")

            # Periodically force garbage collection
            if i % 100 == 0:
                gc.collect()

        # Get info about managed loggers
        managed_loggers = factory.get_managed_loggers()

        # Should have all the loggers
        assert len(managed_loggers) >= 1000

        # Cleanup and verify memory is freed
        factory.cleanup()
        gc.collect()

    def test_deep_module_hierarchy_performance(self):
        """Test performance with deep module hierarchies."""
        factory = LoggerFactory()

        start_time = time.time()

        # Create loggers with deep hierarchies
        loggers = []
        for i in range(100):
            module_name = f"level1.level2.level3.level4.module_{i}"
            logger = factory.get_module_logger(module_name)
            loggers.append(logger)

        end_time = time.time()
        duration = end_time - start_time

        # Should handle deep hierarchies efficiently
        assert duration < 1.0
        assert len(loggers) == 100

    def test_repeated_operations_performance(self):
        """Test performance of repeated operations."""
        factory = LoggerFactory()

        start_time = time.time()

        # Perform repeated operations
        for i in range(1000):
            # Mix of operations
            if i % 3 == 0:
                factory.get_logger(f"repeat_{i}")
            elif i % 3 == 1:
                factory.get_info()
            else:
                factory.set_level("INFO")

        end_time = time.time()
        duration = end_time - start_time

        # Should handle repeated operations efficiently
        assert (
            duration < 10.0
        )  # 1000 operations in less than 10 seconds (more lenient for CI)

    def test_large_configuration_performance(self):
        """Test performance with large configurations."""
        factory = LoggerFactory()

        # Create a large configuration dictionary
        large_config = {
            "level": "DEBUG",
            "format": "%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s",
            "handlers": ["console"],
        }

        start_time = time.time()

        # Apply large configuration multiple times
        for i in range(100):
            factory.configure(large_config)

        end_time = time.time()
        duration = end_time - start_time

        # Should handle large configs efficiently
        assert duration < 1.0


class TestLoggerFactoryResourceManagement:
    """Test suite for LoggerFactory resource cleanup and management."""

    def teardown_method(self):
        """Clean up singleton instance after each test."""
        LoggerFactory.reset_instance()

    @pytest.fixture
    def temp_dir(self):
        """Create a temporary directory for test files."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            yield Path(tmp_dir)

    def test_cleanup_releases_resources(self, temp_dir):
        """Test that cleanup properly releases all resources."""
        factory = LoggerFactory()

        # Configure with file logging
        factory.configure(
            {
                "level": "INFO",
                "handlers": ["console", "file"],
                "file_path": str(temp_dir / "resource_test.log"),
            }
        )

        # Create some loggers
        loggers = [factory.get_logger(f"resource_{i}") for i in range(5)]

        assert factory.is_initialized()
        assert len(loggers) == 5

        # Cleanup should release everything
        factory.cleanup()

        assert not factory.is_initialized()

    def test_cleanup_handles_missing_manager(self):
        """Test cleanup handles case where manager is None."""
        factory = LoggerFactory()
        factory._initialized = True
        factory._manager = None

        # Should not raise exception
        factory.cleanup()
        assert not factory.is_initialized()

    def test_cleanup_handles_manager_errors(self):
        """Test cleanup handles manager cleanup errors gracefully."""
        factory = LoggerFactory()
        factory.initialize()

        with patch.object(factory._manager, "cleanup") as mock_cleanup:
            mock_cleanup.side_effect = Exception("Cleanup error")

            # Should not raise exception, just print warning
            factory.cleanup()
            # Factory remains initialized because cleanup failed inside try block
            assert factory.is_initialized()

    def test_atexit_cleanup_registration(self):
        """Test that cleanup is properly registered with atexit."""
        factory = LoggerFactory()
        factory.initialize()

        # Verify that factory has cleanup registered
        # This is hard to test directly, but we can verify the manager exists
        assert factory._manager is not None

    def test_resource_cleanup_after_errors(self):
        """Test resources are cleaned up properly after errors."""
        factory = LoggerFactory()

        # Cause an error during operation
        with patch.object(factory, "_manager", None):
            factory._initialized = True

            try:
                factory.get_managed_loggers()
            except LoggerFactoryError:
                pass

        # Cleanup should still work
        factory.cleanup()
        assert not factory.is_initialized()

    def test_memory_cleanup_with_circular_references(self):
        """Test memory cleanup with potential circular references."""
        import gc

        factory = LoggerFactory()

        # Create loggers that might have circular references
        loggers = []
        for i in range(10):
            logger = factory.get_logger(f"circular_{i}")
            # Simulate adding references that might create cycles
            logger.custom_ref = factory  # This could create a cycle
            loggers.append(logger)

        # Clear local references
        del loggers

        # Cleanup factory
        factory.cleanup()
        gc.collect()

        # Factory should still be functional after cleanup
        new_logger = factory.get_logger("after_cleanup")
        assert new_logger is not None

    def test_file_handle_cleanup(self, temp_dir):
        """Test that file handles are properly closed during cleanup."""
        factory = LoggerFactory()
        log_file = temp_dir / "handle_test.log"

        factory.configure(
            {"level": "INFO", "handlers": ["file"], "file_path": str(log_file)}
        )

        # Create logger and write to it
        logger = factory.get_logger("file_test")
        logger.info("Test message")

        # Flush handlers to ensure content is written
        for handler in logger.handlers:
            if hasattr(handler, "flush"):
                handler.flush()

        # File should exist and have content
        assert log_file.exists()
        # Content might be buffered, so check if file was created

        # Cleanup should close file handles
        factory.cleanup()

        # File should still exist
        assert log_file.exists()

    def test_cleanup_thread_safety(self):
        """Test that cleanup operations are thread-safe."""
        factory = LoggerFactory()
        factory.initialize()

        # Create loggers from multiple threads
        def create_and_cleanup(thread_id):
            logger = factory.get_logger(f"thread_cleanup_{thread_id}")
            if thread_id == 0:
                # One thread performs cleanup
                factory.cleanup()
            return logger

        threads = []
        results = []

        for i in range(5):
            thread = threading.Thread(
                target=lambda tid=i: results.append(create_and_cleanup(tid))
            )
            threads.append(thread)
            thread.start()

        for thread in threads:
            thread.join()

        # Should complete without errors
        assert len(results) == 5

    def test_resource_limits_handling(self):
        """Test factory handles resource limits gracefully."""
        factory = LoggerFactory()

        # Simulate resource exhaustion during logger creation
        with patch("logging.getLogger") as mock_get_logger:
            mock_get_logger.side_effect = OSError("Too many open files")

            with pytest.raises(LoggerFactoryError):
                factory.get_logger("resource_limit_test")

        # Factory should still be cleanable
        factory.cleanup()

    def test_repeated_cleanup_safety(self):
        """Test that repeated cleanup calls are safe."""
        factory = LoggerFactory()
        factory.initialize()

        # Multiple cleanup calls should be safe
        factory.cleanup()
        factory.cleanup()
        factory.cleanup()

        assert not factory.is_initialized()

        # Should be able to initialize again after multiple cleanups
        factory.initialize()
        assert factory.is_initialized()


if __name__ == "__main__":
    pytest.main([__file__])
