#!/usr/bin/env python3
"""
Demonstration script for AIM2-004-07: Context Injection for Request IDs

This script demonstrates the complete context injection functionality including:
- Request context management with automatic request ID generation
- Context injection into log messages
- JSON formatting with context information
- Thread-safe context isolation
- Integration with existing logging framework

Usage:
    python test_context_integration_demo.py

Expected Output:
- JSON log messages with automatically injected context information
- Demonstration of request ID tracking across operations
- Thread-safe context isolation examples
"""

import json
import logging
import threading
import time

# Import our context injection modules
from aim2_project.aim2_utils.request_context import (
    RequestContextManager,
    with_request_context,
    inject_request_id,
)
from aim2_project.aim2_utils.context_injection import (
    ContextInjectingLogger,
)
from aim2_project.aim2_utils.logger_config import LoggerConfig
from aim2_project.aim2_utils.logger_manager import LoggerManager
from aim2_project.aim2_utils.json_formatter import JSONFormatter


def setup_logging_with_context():
    """Set up logging system with context injection enabled."""
    print("🔧 Setting up logging with context injection...")

    # Create configuration with context injection enabled
    config = LoggerConfig()
    config.load_from_dict(
        {
            "level": "INFO",
            "formatter_type": "json",
            "handlers": ["console"],
            "enable_context_injection": True,
            "context_injection_method": "filter",
            "auto_generate_request_id": True,
            "request_id_field": "request_id",
            "context_fields": [
                "request_id",
                "user_id",
                "session_id",
                "operation",
                "component",
                "trace_id",
            ],
            "include_context_in_json": True,
            "json_fields": ["timestamp", "level", "logger_name", "message"],
            "json_pretty_print": True,
            "default_context": {"service": "aim2-demo", "version": "1.0.0"},
        }
    )

    # Initialize logger manager
    logger_manager = LoggerManager(config)
    logger_manager.initialize()

    return logger_manager


def demonstrate_basic_context_injection(logger_manager):
    """Demonstrate basic context injection functionality."""
    print("\n📝 Demonstrating basic context injection...")

    logger = logger_manager.get_logger("demo.basic")
    context_manager = logger_manager.get_context_manager()

    # Set some context
    context_manager.set_context(
        user_id="demo_user_123", operation="basic_demo", component="context_demo"
    )

    logger.info("This message will include context automatically")
    logger.warning("Context is injected into all log levels")

    # Clear context
    context_manager.clear_context()
    logger.info("This message has no additional context")


def demonstrate_request_context_manager(logger_manager):
    """Demonstrate request context manager functionality."""
    print("\n🔄 Demonstrating request context manager...")

    logger = logger_manager.get_logger("demo.request_context")
    context_manager = logger_manager.get_context_manager()

    # Use request context manager
    with context_manager.request_context(
        user_id="context_user_456",
        operation="request_context_demo",
        session_id="session_789",
    ) as request_id:
        logger.info(f"Inside request context with ID: {request_id}")
        logger.info("Performing some operation...")

        # Nested context
        with context_manager.request_context(
            operation="nested_operation", trace_id="trace_abc123"
        ) as nested_request_id:
            logger.info(f"Nested context with ID: {nested_request_id}")
            logger.info("Nested operation completed")

        logger.info("Back to parent context")

    logger.info("Context automatically cleared after exiting")


@with_request_context(operation="decorated_function", component="demo")
def demonstrate_context_decorators(logger_manager):
    """Demonstrate context decorators."""
    print("\n🎭 Demonstrating context decorators...")

    logger = logger_manager.get_logger("demo.decorators")
    logger.info("This function is decorated with @with_request_context")
    logger.info("Context is automatically set up and cleaned up")


@inject_request_id
def demonstrate_request_id_injection(logger_manager):
    """Demonstrate automatic request ID injection."""
    print("\n🆔 Demonstrating automatic request ID injection...")

    logger = logger_manager.get_logger("demo.request_id")
    logger.info("This function automatically gets a request ID")
    logger.info("No need to manually set context")


def demonstrate_thread_safety(logger_manager):
    """Demonstrate thread-safe context isolation."""
    print("\n🧵 Demonstrating thread-safe context isolation...")

    def worker(thread_id, logger_manager):
        logger = logger_manager.get_logger(f"demo.thread_{thread_id}")
        context_manager = logger_manager.get_context_manager()

        with context_manager.request_context(
            user_id=f"thread_user_{thread_id}",
            operation=f"thread_operation_{thread_id}",
            component="thread_demo",
        ) as request_id:
            logger.info(f"Thread {thread_id} starting work")

            # Simulate some work
            time.sleep(0.1)

            logger.info(f"Thread {thread_id} doing work...")

            # More work
            time.sleep(0.1)

            logger.info(f"Thread {thread_id} completed work")

    # Start multiple threads
    threads = []
    for i in range(3):
        thread = threading.Thread(target=worker, args=(i, logger_manager))
        threads.append(thread)
        thread.start()

    # Wait for all threads to complete
    for thread in threads:
        thread.join()

    print("All threads completed - each had isolated context")


def demonstrate_context_aware_json_formatter():
    """Demonstrate context-aware JSON formatter."""
    print("\n📋 Demonstrating context-aware JSON formatter...")

    # Create a simple logger with context injection
    context_manager = RequestContextManager()
    logger = ContextInjectingLogger("demo.json", context_manager=context_manager)

    # Set up JSON formatter
    formatter = JSONFormatter(
        fields=["timestamp", "level", "logger_name", "message"], pretty_print=True
    )

    # Set up handler
    import io

    stream = io.StringIO()
    handler = logging.StreamHandler(stream)
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)

    # Set context and log
    context_manager.set_context(
        user_id="json_demo_user", operation="json_formatting", trace_id="trace_json_123"
    )

    logger.info("This message will be formatted as JSON with context")

    # Get the output
    output = stream.getvalue()
    print("Raw JSON output:")
    print(output)

    # Parse and display nicely
    try:
        json_data = json.loads(output.strip())
        print("\nParsed JSON structure:")
        print(json.dumps(json_data, indent=2))
    except json.JSONDecodeError as e:
        print(f"Error parsing JSON: {e}")


def demonstrate_error_handling(logger_manager):
    """Demonstrate error handling and fallback behavior."""
    print("\n⚠️ Demonstrating error handling...")

    logger = logger_manager.get_logger("demo.error_handling")
    context_manager = logger_manager.get_context_manager()

    # Test with valid context
    try:
        context_manager.set_context(request_id="error_demo_123", user_id="error_user")
        logger.info("Normal operation with valid context")
    except Exception as e:
        logger.error(f"Unexpected error: {e}")

    # Test with invalid context field (should be handled gracefully)
    try:
        context_manager.set_context(invalid_field="should_fail")
    except Exception as e:
        logger.warning(f"Expected error for invalid field: {e}")
        # Context should still work after error
        context_manager.set_context(user_id="recovery_user")
        logger.info("Context recovery successful")


def main():
    """Main demonstration function."""
    print("🚀 AIM2-004-07: Context Injection for Request IDs - Demo")
    print("=" * 60)

    try:
        # Set up logging system with context injection
        logger_manager = setup_logging_with_context()

        # Run demonstrations
        demonstrate_basic_context_injection(logger_manager)
        demonstrate_request_context_manager(logger_manager)
        demonstrate_context_decorators(logger_manager)
        demonstrate_request_id_injection(logger_manager)
        demonstrate_thread_safety(logger_manager)
        demonstrate_context_aware_json_formatter()
        demonstrate_error_handling(logger_manager)

        print("\n✅ All demonstrations completed successfully!")
        print("\nKey Features Demonstrated:")
        print("- ✓ Automatic request ID generation")
        print("- ✓ Context injection into log messages")
        print("- ✓ JSON formatting with context")
        print("- ✓ Thread-safe context isolation")
        print("- ✓ Request context managers")
        print("- ✓ Context decorators")
        print("- ✓ Error handling and recovery")
        print("- ✓ Integration with existing logging framework")

    except Exception as e:
        print(f"❌ Demo failed with error: {e}")
        import traceback

        traceback.print_exc()
        return 1

    return 0


if __name__ == "__main__":
    exit(main())
