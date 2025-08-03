"""
Unit tests for context injection functionality in the AIM2 project

This module provides comprehensive unit tests for context injection functionality,
including RequestContextManager, context injection filters, context-aware loggers,
and integration with the existing logging framework.

Test Coverage:
- RequestContextManager creation and context management
- Context injection filter functionality
- ContextInjectingLogger automatic context inclusion
- Thread safety and concurrent context management
- Integration with LoggerConfig and LoggerManager
- Context field validation and serialization
- Request context decorators and utilities
- Error handling and fallback scenarios

Test Classes:
    TestRequestContextManager: Tests for RequestContextManager class
    TestContextInjection: Tests for context injection mechanisms
    TestContextInjectingLogger: Tests for context-aware logger
    TestContextInjectionFilter: Tests for context injection filter
    TestContextIntegration: Tests for integration with logging framework
    TestContextThreadSafety: Tests for thread safety and concurrency
    TestContextValidation: Tests for context validation and error handling
    TestContextUtilities: Tests for utility functions and decorators
"""

import io
import json
import logging
import threading
import time
import pytest
from unittest.mock import Mock

# Import the modules to be tested
try:
    from aim2_project.aim2_utils.request_context import (
        RequestContextManager,
        RequestContextError,
        get_global_context_manager,
        set_global_context_manager,
        reset_global_context_manager,
        set_request_context,
        get_request_context,
        clear_request_context,
        get_current_request_id,
        request_context,
    )
    from aim2_project.aim2_utils.context_injection import (
        ContextInjectingLogRecord,
        ContextInjectingLogger,
        ContextInjectionFilter,
        ContextAwareFormatter,
        install_context_injection,
        create_context_logger,
        add_context_to_all_loggers,
        with_request_context,
        inject_request_id,
    )
    from aim2_project.aim2_utils.logger_config import LoggerConfig
    from aim2_project.aim2_utils.logger_manager import LoggerManager
    from aim2_project.aim2_utils.json_formatter import JSONFormatter
except ImportError:
    # Expected during TDD - tests define the interface
    pass


class TestRequestContextManager:
    """Test suite for RequestContextManager functionality."""

    def test_context_manager_creation_default(self):
        """Test RequestContextManager creation with default settings."""
        manager = RequestContextManager()

        assert manager._auto_generate_request_id is True
        assert manager._request_id_field == "request_id"
        assert isinstance(manager._context_fields, set)
        assert "request_id" in manager._context_fields
        assert "user_id" in manager._context_fields

    def test_context_manager_creation_custom(self):
        """Test RequestContextManager creation with custom settings."""
        custom_fields = {"custom_field", "another_field"}
        manager = RequestContextManager(
            auto_generate_request_id=False,
            request_id_field="custom_req_id",
            default_context={"service": "test"},
            allowed_context_fields=custom_fields,
        )

        assert manager._auto_generate_request_id is False
        assert manager._request_id_field == "custom_req_id"
        assert manager._default_context == {"service": "test"}
        assert (
            "custom_req_id" in manager._context_fields
        )  # Should be added automatically

    def test_set_context_basic(self):
        """Test basic context setting functionality."""
        manager = RequestContextManager()

        manager.set_context(request_id="test-123", user_id="user456")

        assert manager.get_context("request_id") == "test-123"
        assert manager.get_context("user_id") == "user456"

    def test_set_context_auto_generate_request_id(self):
        """Test automatic request ID generation."""
        manager = RequestContextManager(auto_generate_request_id=True)

        manager.set_context(user_id="user456")
        request_id = manager.get_context("request_id")

        assert request_id is not None
        assert len(request_id) > 0
        assert isinstance(request_id, str)

    def test_set_context_invalid_field(self):
        """Test setting invalid context field."""
        manager = RequestContextManager(
            allowed_context_fields={"request_id", "user_id"}
        )

        with pytest.raises(RequestContextError) as exc_info:
            manager.set_context(invalid_field="value")

        assert "invalid_field" in str(exc_info.value)
        assert "not allowed" in str(exc_info.value)

    def test_get_context_full(self):
        """Test getting full context dictionary."""
        manager = RequestContextManager()
        manager.set_context(request_id="test-123", user_id="user456")

        context = manager.get_context()

        assert isinstance(context, dict)
        assert context["request_id"] == "test-123"
        assert context["user_id"] == "user456"

    def test_get_context_with_default(self):
        """Test getting context with default values."""
        manager = RequestContextManager(
            default_context={"service": "test-service", "version": "1.0"}
        )
        manager.set_context(request_id="test-123")

        context = manager.get_context()

        assert context["request_id"] == "test-123"
        assert context["service"] == "test-service"
        assert context["version"] == "1.0"

    def test_clear_context(self):
        """Test clearing context."""
        manager = RequestContextManager()
        manager.set_context(request_id="test-123", user_id="user456")

        manager.clear_context()

        # Should only have default context now
        context = manager.get_context()
        assert context.get("request_id") is None
        assert context.get("user_id") is None

    def test_update_context(self):
        """Test updating context with multiple values."""
        manager = RequestContextManager()
        manager.set_context(request_id="test-123")

        updates = {"user_id": "user456", "session_id": "session789"}
        manager.update_context(updates)

        assert manager.get_context("user_id") == "user456"
        assert manager.get_context("session_id") == "session789"
        assert manager.get_context("request_id") == "test-123"  # Should be preserved

    def test_remove_context(self):
        """Test removing specific context keys."""
        manager = RequestContextManager()
        manager.set_context(
            request_id="test-123", user_id="user456", session_id="session789"
        )

        manager.remove_context("user_id", "session_id")

        assert manager.get_context("request_id") == "test-123"
        assert manager.get_context("user_id") is None
        assert manager.get_context("session_id") is None

    def test_has_context(self):
        """Test checking if context key exists."""
        manager = RequestContextManager()
        manager.set_context(request_id="test-123")

        assert manager.has_context("request_id") is True
        assert manager.has_context("nonexistent") is False

    def test_get_request_id(self):
        """Test getting request ID specifically."""
        manager = RequestContextManager()
        manager.set_request_id("custom-request-id")

        assert manager.get_request_id() == "custom-request-id"

    def test_request_context_manager(self):
        """Test request context manager functionality."""
        manager = RequestContextManager()

        with manager.request_context(user_id="user123") as req_id:
            assert req_id is not None
            assert manager.get_context("user_id") == "user123"
            assert manager.get_request_id() == req_id

        # Context should be cleared after exiting
        assert manager.get_context("user_id") is None

    def test_request_context_manager_nested(self):
        """Test nested request context managers."""
        manager = RequestContextManager()
        manager.set_context(user_id="outer_user")

        with manager.request_context(user_id="inner_user", operation="test") as req_id:
            assert manager.get_context("user_id") == "inner_user"
            assert manager.get_context("operation") == "test"
            assert manager.get_request_id() == req_id

        # Should restore outer context
        assert manager.get_context("user_id") == "outer_user"
        assert manager.get_context("operation") is None

    def test_get_context_for_logging(self):
        """Test getting context formatted for logging."""
        manager = RequestContextManager()
        manager.set_context(request_id="test-123", user_id="user456")

        logging_context = manager.get_context_for_logging()

        assert isinstance(logging_context, dict)
        assert logging_context["request_id"] == "test-123"
        assert logging_context["user_id"] == "user456"

    def test_get_context_for_logging_serialization(self):
        """Test context serialization for logging."""
        manager = RequestContextManager()

        # Add non-serializable object
        class NonSerializable:
            def __str__(self):
                return "non-serializable"

        manager.set_context(request_id="test-123")
        # Manually add non-serializable object (bypassing validation)
        context = manager._get_context_dict()
        context["non_serializable"] = NonSerializable()

        logging_context = manager.get_context_for_logging()

        # Should convert to string
        assert logging_context["non_serializable"] == "non-serializable"

    def test_context_manager_thread_safety(self):
        """Test thread safety of context manager."""
        manager = RequestContextManager()
        results = {}
        errors = []

        def worker(thread_id):
            try:
                with manager.request_context(user_id=f"user_{thread_id}") as req_id:
                    time.sleep(0.01)  # Simulate work
                    results[thread_id] = {
                        "request_id": req_id,
                        "user_id": manager.get_context("user_id"),
                    }
            except Exception as e:
                errors.append((thread_id, str(e)))

        # Start multiple threads
        threads = []
        for i in range(10):
            thread = threading.Thread(target=worker, args=(i,))
            threads.append(thread)
            thread.start()

        # Wait for completion
        for thread in threads:
            thread.join()

        # Check results
        assert len(errors) == 0, f"Thread safety errors: {errors}"
        assert len(results) == 10

        # Each thread should have different context
        request_ids = {result["request_id"] for result in results.values()}
        assert len(request_ids) == 10  # All unique


class TestContextInjection:
    """Test suite for context injection mechanisms."""

    def test_context_injecting_log_record(self):
        """Test ContextInjectingLogRecord creation and context injection."""
        manager = RequestContextManager()
        manager.set_context(request_id="test-123", user_id="user456")

        record = ContextInjectingLogRecord(
            name="test_logger",
            level=logging.INFO,
            pathname="test.py",
            lineno=42,
            msg="Test message",
            args=(),
            exc_info=None,
            context_manager=manager,
        )

        assert hasattr(record, "request_id")
        assert record.request_id == "test-123"
        assert hasattr(record, "user_id")
        assert record.user_id == "user456"
        assert hasattr(record, "logging_context")
        assert record.logging_context["request_id"] == "test-123"

    def test_context_injecting_logger(self):
        """Test ContextInjectingLogger automatic context injection."""
        manager = RequestContextManager()
        manager.set_context(request_id="test-123", user_id="user456")

        logger = ContextInjectingLogger("test_logger", context_manager=manager)

        # Capture log output
        stream = io.StringIO()
        handler = logging.StreamHandler(stream)
        formatter = JSONFormatter()
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)

        logger.info("Test message")

        output = stream.getvalue()
        json_data = json.loads(output)

        # Should include context in JSON output
        assert "context" in json_data
        assert json_data["context"]["request_id"] == "test-123"
        assert json_data["context"]["user_id"] == "user456"

    def test_context_injection_filter(self):
        """Test ContextInjectionFilter functionality."""
        manager = RequestContextManager()
        manager.set_context(request_id="test-123", user_id="user456")

        # Create standard logger with context filter
        logger = logging.getLogger("test_filter_logger")
        context_filter = ContextInjectionFilter(context_manager=manager)
        logger.addFilter(context_filter)

        # Create log record
        record = logging.LogRecord(
            name="test_filter_logger",
            level=logging.INFO,
            pathname="test.py",
            lineno=42,
            msg="Test message",
            args=(),
            exc_info=None,
        )

        # Apply filter
        result = context_filter.filter(record)

        assert result is True  # Filter should not block record
        assert hasattr(record, "request_id")
        assert record.request_id == "test-123"
        assert hasattr(record, "logging_context")

    def test_context_injection_filter_with_prefix(self):
        """Test ContextInjectionFilter with field prefix."""
        manager = RequestContextManager()
        manager.set_context(request_id="test-123", user_id="user456")

        context_filter = ContextInjectionFilter(
            context_manager=manager, context_prefix="ctx_"
        )

        record = logging.LogRecord(
            name="test",
            level=logging.INFO,
            pathname="test.py",
            lineno=1,
            msg="Test",
            args=(),
            exc_info=None,
        )

        context_filter.filter(record)

        assert hasattr(record, "ctx_request_id")
        assert record.ctx_request_id == "test-123"
        assert hasattr(record, "ctx_user_id")
        assert record.ctx_user_id == "user456"

    def test_context_aware_formatter(self):
        """Test ContextAwareFormatter functionality."""
        manager = RequestContextManager()
        manager.set_context(request_id="test-123", user_id="user456")

        formatter = ContextAwareFormatter(
            fmt="%(levelname)s - %(message)s",
            context_fields=["request_id"],
            include_context_in_message=True,
        )

        # Create record with context
        record = ContextInjectingLogRecord(
            name="test",
            level=logging.INFO,
            pathname="test.py",
            lineno=1,
            msg="Test message",
            args=(),
            exc_info=None,
            context_manager=manager,
        )

        formatted = formatter.format(record)

        assert "INFO - Test message" in formatted
        assert "[request_id=test-123]" in formatted

    def test_install_context_injection_filter_method(self):
        """Test installing context injection using filter method."""
        manager = RequestContextManager()
        logger = logging.getLogger("test_install_filter")

        install_context_injection(logger, manager, method="filter")

        # Should have context injection filter
        filters = [f for f in logger.filters if isinstance(f, ContextInjectionFilter)]
        assert len(filters) == 1

    def test_install_context_injection_replace_method(self):
        """Test installing context injection using replace method."""
        manager = RequestContextManager()
        logger = logging.getLogger("test_install_replace")

        install_context_injection(logger, manager, method="replace")

        # Logger class should be replaced
        assert isinstance(logger, ContextInjectingLogger)
        assert logger.context_manager == manager

    def test_create_context_logger(self):
        """Test creating context-injecting logger."""
        manager = RequestContextManager()
        logger = create_context_logger("test_context_logger", context_manager=manager)

        assert isinstance(logger, ContextInjectingLogger)
        assert logger.context_manager == manager

    def test_context_injection_error_handling(self):
        """Test context injection error handling."""
        # Create a context manager that will cause errors
        broken_manager = Mock()
        broken_manager.get_context_for_logging.side_effect = Exception("Context error")

        context_filter = ContextInjectionFilter(context_manager=broken_manager)

        record = logging.LogRecord(
            name="test",
            level=logging.INFO,
            pathname="test.py",
            lineno=1,
            msg="Test message",
            args=(),
            exc_info=None,
        )

        # Should not raise exception, just continue
        result = context_filter.filter(record)
        assert result is True


class TestContextUtilities:
    """Test suite for context utility functions and decorators."""

    def test_global_context_manager(self):
        """Test global context manager functions."""
        # Reset to clean state
        reset_global_context_manager()

        # Get global manager
        manager = get_global_context_manager()
        assert isinstance(manager, RequestContextManager)

        # Set context using global functions
        set_request_context(request_id="global-123", user_id="global_user")

        assert get_request_context("request_id") == "global-123"
        assert get_current_request_id() == "global-123"

        # Clear context
        clear_request_context()
        assert get_request_context("request_id") is None

    def test_set_global_context_manager(self):
        """Test setting custom global context manager."""
        custom_manager = RequestContextManager(auto_generate_request_id=False)
        set_global_context_manager(custom_manager)

        retrieved_manager = get_global_context_manager()
        assert retrieved_manager is custom_manager

    def test_global_request_context(self):
        """Test global request context manager."""
        reset_global_context_manager()

        with request_context(user_id="context_user") as req_id:
            assert req_id is not None
            assert get_request_context("user_id") == "context_user"

        # Should be cleared after exit
        assert get_request_context("user_id") is None

    def test_with_request_context_decorator(self):
        """Test with_request_context decorator."""
        reset_global_context_manager()

        @with_request_context(operation="test_operation", service="test_service")
        def test_function():
            return {
                "request_id": get_current_request_id(),
                "operation": get_request_context("operation"),
                "service": get_request_context("service"),
            }

        result = test_function()

        assert result["request_id"] is not None
        assert result["operation"] == "test_operation"
        assert result["service"] == "test_service"

        # Context should be cleared after function
        assert get_request_context("operation") is None

    def test_inject_request_id_decorator(self):
        """Test inject_request_id decorator."""
        reset_global_context_manager()

        @inject_request_id
        def test_function():
            return get_current_request_id()

        request_id = test_function()

        assert request_id is not None
        assert isinstance(request_id, str)

        # Context should be cleared after function
        assert get_current_request_id() is None

    def test_add_context_to_all_loggers(self):
        """Test adding context to all loggers."""
        manager = RequestContextManager()

        # This is a global operation, test carefully
        root_logger = logging.getLogger()
        initial_filter_count = len(root_logger.filters)

        add_context_to_all_loggers(manager)

        # Should add context filter to root logger
        assert len(root_logger.filters) == initial_filter_count + 1

        # Find the context filter
        context_filters = [
            f for f in root_logger.filters if isinstance(f, ContextInjectionFilter)
        ]
        assert len(context_filters) == 1


class TestContextIntegration:
    """Test suite for context injection integration with logging framework."""

    def test_logger_config_context_settings(self):
        """Test LoggerConfig with context injection settings."""
        config_dict = {
            "enable_context_injection": True,
            "context_injection_method": "filter",
            "auto_generate_request_id": True,
            "request_id_field": "req_id",
            "context_fields": ["req_id", "user_id", "trace_id"],
            "include_context_in_json": True,
            "context_prefix": "ctx_",
            "default_context": {"service": "test_service"},
        }

        config = LoggerConfig()
        config.load_from_dict(config_dict)

        assert config.get_enable_context_injection() is True
        assert config.get_context_injection_method() == "filter"
        assert config.get_auto_generate_request_id() is True
        assert config.get_request_id_field() == "req_id"
        assert config.get_context_fields() == ["req_id", "user_id", "trace_id"]
        assert config.get_include_context_in_json() is True
        assert config.get_context_prefix() == "ctx_"
        assert config.get_default_context() == {"service": "test_service"}

    def test_logger_config_validation_context_injection(self):
        """Test LoggerConfig validation for context injection settings."""
        config = LoggerConfig()

        # Test invalid context injection method
        with pytest.raises(Exception):  # Should be LoggerConfigError
            config.load_from_dict({"context_injection_method": "invalid_method"})

        # Test invalid context fields
        with pytest.raises(Exception):  # Should be LoggerConfigError
            config.load_from_dict({"context_fields": "not_a_list"})

    def test_logger_manager_context_integration(self):
        """Test LoggerManager integration with context injection."""
        config = LoggerConfig()
        config.load_from_dict(
            {
                "enable_context_injection": True,
                "context_injection_method": "filter",
                "auto_generate_request_id": True,
            }
        )

        manager = LoggerManager(config)
        manager.initialize()

        # Should have context manager
        context_manager = manager.get_context_manager()
        assert context_manager is not None
        assert isinstance(context_manager, RequestContextManager)

        # Logger should have context injection
        logger = manager.get_logger("test_integration")

        # Set context and test logging
        context_manager.set_context(user_id="integration_user")

        # Capture output
        stream = io.StringIO()
        handler = logging.StreamHandler(stream)
        formatter = JSONFormatter()
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.propagate = False

        logger.info("Integration test message")

        output = stream.getvalue()
        json_data = json.loads(output)

        # Should include context
        assert "context" in json_data
        assert json_data["context"]["user_id"] == "integration_user"

    def test_json_formatter_context_extraction(self):
        """Test JSONFormatter context extraction."""
        manager = RequestContextManager()
        manager.set_context(request_id="json-test-123", operation="json_test")

        # Create record with context
        record = ContextInjectingLogRecord(
            name="json_test",
            level=logging.INFO,
            pathname="test.py",
            lineno=1,
            msg="JSON test message",
            args=(),
            exc_info=None,
            context_manager=manager,
        )

        formatter = JSONFormatter()
        output = formatter.format(record)
        json_data = json.loads(output)

        # Should extract context fields
        assert "context" in json_data
        assert json_data["context"]["request_id"] == "json-test-123"
        assert json_data["context"]["operation"] == "json_test"

    def test_end_to_end_context_flow(self):
        """Test complete end-to-end context injection flow."""
        # Setup complete logging system with context injection
        config = LoggerConfig()
        config.load_from_dict(
            {
                "level": "INFO",
                "formatter_type": "json",
                "handlers": ["console"],
                "enable_context_injection": True,
                "context_injection_method": "filter",
                "auto_generate_request_id": True,
                "json_fields": ["timestamp", "level", "logger_name", "message"],
                "include_context_in_json": True,
            }
        )

        logger_manager = LoggerManager(config)
        logger_manager.initialize()

        # Get logger and set up output capture
        logger = logger_manager.get_logger("end_to_end_test")
        stream = io.StringIO()
        handler = logging.StreamHandler(stream)
        formatter = JSONFormatter()
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.propagate = False

        # Use context manager and log message
        context_manager = logger_manager.get_context_manager()
        with context_manager.request_context(
            user_id="end_to_end_user", operation="complete_test"
        ) as req_id:
            logger.info("End-to-end test message")

        # Verify output
        output = stream.getvalue()
        json_data = json.loads(output)

        assert json_data["level"] == "INFO"
        assert json_data["message"] == "End-to-end test message"
        assert "context" in json_data
        assert json_data["context"]["request_id"] == req_id
        assert json_data["context"]["user_id"] == "end_to_end_user"
        assert json_data["context"]["operation"] == "complete_test"


class TestContextThreadSafety:
    """Test suite for context injection thread safety."""

    def test_concurrent_context_operations(self):
        """Test concurrent context operations."""
        manager = RequestContextManager()
        results = {}
        errors = []

        def worker(thread_id, iterations=50):
            thread_results = []
            try:
                for i in range(iterations):
                    with manager.request_context(
                        user_id=f"user_{thread_id}", operation=f"op_{i}"
                    ) as req_id:
                        # Simulate some work
                        time.sleep(0.001)

                        # Verify context isolation
                        user_id = manager.get_context("user_id")
                        operation = manager.get_context("operation")

                        thread_results.append(
                            {
                                "request_id": req_id,
                                "user_id": user_id,
                                "operation": operation,
                                "iteration": i,
                            }
                        )

                        # Verify values are correct for this thread
                        assert user_id == f"user_{thread_id}"
                        assert operation == f"op_{i}"

            except Exception as e:
                errors.append((thread_id, str(e)))
            finally:
                results[thread_id] = thread_results

        # Start multiple threads
        threads = []
        num_threads = 5
        for i in range(num_threads):
            thread = threading.Thread(target=worker, args=(i,))
            threads.append(thread)
            thread.start()

        # Wait for completion
        for thread in threads:
            thread.join()

        # Verify results
        assert len(errors) == 0, f"Thread safety errors: {errors}"
        assert len(results) == num_threads

        # Verify each thread had correct isolated context
        for thread_id, thread_results in results.items():
            assert len(thread_results) == 50
            for result in thread_results:
                assert result["user_id"] == f"user_{thread_id}"
                assert f"op_{result['iteration']}" == result["operation"]

    def test_concurrent_logging_with_context(self):
        """Test concurrent logging with context injection."""
        manager = RequestContextManager()
        logger = ContextInjectingLogger("thread_test", context_manager=manager)

        # Capture all output
        stream = io.StringIO()
        handler = logging.StreamHandler(stream)
        formatter = JSONFormatter()
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)

        log_results = []
        errors = []
        lock = threading.Lock()

        def logging_worker(thread_id, message_count=20):
            try:
                for i in range(message_count):
                    with manager.request_context(
                        user_id=f"user_{thread_id}", message_num=str(i)
                    ) as req_id:
                        logger.info(f"Thread {thread_id} message {i}")

                        # Record what we logged
                        with lock:
                            log_results.append(
                                {
                                    "thread_id": thread_id,
                                    "message_num": i,
                                    "request_id": req_id,
                                    "user_id": f"user_{thread_id}",
                                }
                            )

            except Exception as e:
                errors.append((thread_id, str(e)))

        # Start multiple logging threads
        threads = []
        num_threads = 3
        for i in range(num_threads):
            thread = threading.Thread(target=logging_worker, args=(i,))
            threads.append(thread)
            thread.start()

        # Wait for completion
        for thread in threads:
            thread.join()

        # Verify no errors
        assert len(errors) == 0, f"Concurrent logging errors: {errors}"

        # Parse logged output
        output_lines = stream.getvalue().strip().split("\n")
        logged_messages = []
        for line in output_lines:
            if line.strip():
                try:
                    json_data = json.loads(line)
                    logged_messages.append(json_data)
                except json.JSONDecodeError:
                    pass

        # Should have correct number of messages
        assert len(logged_messages) == num_threads * 20

        # Verify each message has correct context
        for msg in logged_messages:
            assert "context" in msg
            context = msg["context"]
            assert "request_id" in context
            assert "user_id" in context

            # Extract thread info from message
            message_text = msg["message"]
            assert "Thread" in message_text
            assert "message" in message_text


class TestContextValidation:
    """Test suite for context validation and error handling."""

    def test_context_field_validation(self):
        """Test context field validation."""
        manager = RequestContextManager(
            allowed_context_fields={"request_id", "user_id"}
        )

        # Valid fields should work
        manager.set_context(request_id="test-123", user_id="user456")

        # Invalid field should raise error
        with pytest.raises(RequestContextError):
            manager.set_context(invalid_field="value")

    def test_context_serialization_handling(self):
        """Test handling of non-serializable context values."""
        manager = RequestContextManager()

        # Test with various data types
        manager.set_context(
            request_id="test-123",
            number_value=42,
            float_value=3.14,
            bool_value=True,
            list_value=[1, 2, 3],
            dict_value={"nested": "value"},
        )

        logging_context = manager.get_context_for_logging()

        # All should be present and JSON-serializable
        assert logging_context["request_id"] == "test-123"
        assert logging_context["number_value"] == 42
        assert logging_context["float_value"] == 3.14
        assert logging_context["bool_value"] is True
        assert logging_context["list_value"] == [1, 2, 3]
        assert logging_context["dict_value"] == {"nested": "value"}

        # Test JSON serialization
        import json

        json.dumps(logging_context)  # Should not raise

    def test_context_manager_error_recovery(self):
        """Test context manager error recovery."""
        manager = RequestContextManager()

        # Test that context manager continues working after errors
        try:
            manager.set_context(invalid_field="value")
        except RequestContextError:
            pass

        # Should still work after error
        manager.set_context(request_id="recovery-test")
        assert manager.get_request_id() == "recovery-test"

    def test_context_injection_fallback(self):
        """Test context injection fallback when context is unavailable."""
        # Create logger without context manager
        logger = logging.getLogger("fallback_test")

        # Add context filter with no context set
        context_filter = ContextInjectionFilter()
        logger.addFilter(context_filter)

        # Create record
        record = logging.LogRecord(
            name="fallback_test",
            level=logging.INFO,
            pathname="test.py",
            lineno=1,
            msg="Fallback test",
            args=(),
            exc_info=None,
        )

        # Should not fail
        result = context_filter.filter(record)
        assert result is True

    def test_invalid_context_manager_configuration(self):
        """Test handling of invalid context manager configuration."""
        # Test invalid request ID field
        with pytest.raises(RequestContextError):
            RequestContextManager(request_id_field="")

        # Test invalid allowed fields (should be handled gracefully)
        manager = RequestContextManager(allowed_context_fields=set())
        # Should still include request_id_field
        assert "request_id" in manager._context_fields


if __name__ == "__main__":
    pytest.main([__file__])
