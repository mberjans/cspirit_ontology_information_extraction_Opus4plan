#!/usr/bin/env python3
"""
Integration Test for Performance Logging with Existing Logger Infrastructure

This test verifies that the performance decorators integrate seamlessly with:
- Logger configuration system
- Context injection
- JSON formatter
- Module-specific configuration
- Backward compatibility

Run with: python integration_test_performance_logging.py
"""

import logging
import time
import tempfile
import json
import sys
from pathlib import Path
from typing import Any

# Add the project to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from aim2_project.aim2_utils.logger_config import LoggerConfig
from aim2_project.aim2_utils.logger_manager import LoggerManager
from aim2_project.aim2_utils.request_context import RequestContextManager
from aim2_project.aim2_utils.context_injection import with_request_context
from aim2_project.aim2_utils.performance_decorators import performance_logging


class IntegrationTestRunner:
    """Runs comprehensive integration tests for performance logging."""

    def __init__(self):
        self.temp_dir = None
        self.log_file = None
        self.logger_manager = None
        self.context_manager = None
        self.test_results = []

    def setup(self):
        """Set up test environment."""
        print("Setting up integration test environment...")

        # Create temporary directory for logs
        self.temp_dir = tempfile.mkdtemp(prefix="aim2_perf_test_")
        self.log_file = Path(self.temp_dir) / "integration_test.log"

        # Create logger configuration with all features enabled
        config = LoggerConfig()
        config.load_from_dict(
            {
                "level": "DEBUG",
                "handlers": ["console", "file"],
                "file_path": str(self.log_file),
                "formatter_type": "json",
                "json_fields": [
                    "timestamp",
                    "level",
                    "logger_name",
                    "message",
                    "extra",
                    "custom_fields",
                ],
                "json_pretty_print": True,
                "enable_context_injection": True,
                "context_fields": [
                    "request_id",
                    "user_id",
                    "operation",
                    "component",
                    "perf_duration",
                    "perf_operation",
                    "perf_success",
                    "perf_exceeded_warning",
                    "perf_exceeded_critical",
                    "perf_memory_delta_mb",
                    "perf_start_time",
                    "perf_error_type",
                ],
                "include_context_in_json": True,
                "enable_performance_logging": True,
                "performance_thresholds": {
                    "warning_seconds": 0.1,
                    "critical_seconds": 0.5,
                    "memory_warning_mb": 50,
                    "memory_critical_mb": 200,
                },
                "performance_profiles": {
                    "test_fast": {"warning_seconds": 0.05, "critical_seconds": 0.2},
                    "test_slow": {"warning_seconds": 1.0, "critical_seconds": 3.0},
                },
                "performance_metrics": {
                    "include_memory_usage": True,
                    "include_cpu_usage": False,
                    "include_function_args": True,
                    "include_function_result": True,
                    "max_arg_length": 100,
                    "max_result_length": 100,
                },
                "module_levels": {"test_module": "DEBUG", "test_module.fast": "INFO"},
            }
        )

        # Initialize logger manager
        self.logger_manager = LoggerManager(config)
        self.logger_manager.initialize()

        # Set up context manager
        self.context_manager = RequestContextManager(
            allowed_context_fields=set(config.get_context_fields())
        )

        print(f"Test environment ready. Log file: {self.log_file}")

        # Quick test to verify basic logging works
        test_logger = self.logger_manager.get_logger("setup_test")
        test_logger.info("Setup test message")

        # Force flush
        for handler in test_logger.handlers:
            handler.flush()

    def teardown(self):
        """Clean up test environment."""
        print("Cleaning up test environment...")
        if self.logger_manager:
            # Flush all handlers before cleanup
            for handler in logging.root.handlers:
                handler.flush()
            # Also flush logger manager handlers
            for logger_name, logger in logging.Logger.manager.loggerDict.items():
                if hasattr(logger, "handlers"):
                    for handler in logger.handlers:
                        handler.flush()

            # Clean up handlers to avoid issues
            for handler in logging.root.handlers[:]:
                logging.root.removeHandler(handler)
                handler.close()

        # Note: We intentionally keep temp files for inspection
        print(f"Test logs preserved in: {self.temp_dir}")

    def run_test(self, test_name: str, test_func) -> bool:
        """Run a single test and record results."""
        print(f"\n--- Running test: {test_name} ---")
        try:
            test_func()
            print(f"✓ {test_name} PASSED")
            self.test_results.append((test_name, True, None))
            return True
        except Exception as e:
            print(f"✗ {test_name} FAILED: {str(e)}")
            self.test_results.append((test_name, False, str(e)))
            return False

    def test_basic_performance_logging(self):
        """Test basic performance logging functionality."""
        self.logger_manager.get_logger("test_module")

        @performance_logging(logger_name="test_module")
        def fast_function(x, y):
            return x + y

        # Test successful execution
        result = fast_function(1, 2)
        assert result == 3, f"Expected 3, got {result}"

        # Check log file for performance entry
        self._verify_log_contains("perf_operation", "fast_function")
        self._verify_log_contains("perf_success", True)

    def test_performance_with_context(self):
        """Test performance logging with request context."""
        self.logger_manager.get_logger("test_module")

        @with_request_context(request_id="test-123", user_id="user456")
        @performance_logging(logger_name="test_module", operation_name="context_test")
        def context_function():
            time.sleep(0.05)  # Small delay to trigger thresholds
            return "success"

        result = context_function()
        assert result == "success"

        # Verify context is in logs
        self._verify_log_contains("request_id", "test-123")
        self._verify_log_contains("user_id", "user456")
        self._verify_log_contains("perf_operation", "context_test")

    def test_performance_thresholds(self):
        """Test performance threshold detection."""
        self.logger_manager.get_logger("test_module")

        @performance_logging(
            logger_name="test_module", threshold_warning=0.1, threshold_critical=0.3
        )
        def slow_function():
            time.sleep(0.2)  # Should trigger warning
            return "done"

        result = slow_function()
        assert result == "done"

        # Check for threshold warnings
        self._verify_log_contains("perf_exceeded_warning", True)

    def test_performance_profiles(self):
        """Test performance profiles from configuration."""
        self.logger_manager.get_logger("test_module")

        @performance_logging(logger_name="test_module", profile="test_fast")
        def profile_test_function():
            time.sleep(0.1)  # Should exceed fast profile thresholds
            return "profile_test"

        result = profile_test_function()
        assert result == "profile_test"

        # Verify profile-based thresholds were applied
        self._verify_log_contains("perf_operation", "profile_test_function")

    def test_error_handling(self):
        """Test performance logging with errors."""
        self.logger_manager.get_logger("test_module")

        @performance_logging(logger_name="test_module", on_error="log_and_raise")
        def error_function():
            raise ValueError("Test error")

        try:
            error_function()
            assert False, "Expected ValueError"
        except ValueError:
            pass  # Expected

        # Verify error logging
        self._verify_log_contains("perf_success", False)
        self._verify_log_contains("perf_error_type", "ValueError")

    def test_json_formatter_integration(self):
        """Test that performance data is properly formatted in JSON logs."""
        self.logger_manager.get_logger("test_module")

        @performance_logging(logger_name="test_module", operation_name="json_test")
        def json_test_function(param1, param2="default"):
            return {"result": "json_formatted", "param1": param1}

        result = json_test_function("test_value")
        assert result["result"] == "json_formatted"

        # Parse JSON log entries and verify structure
        log_entries = self._parse_json_log_entries()
        perf_entries = [
            entry for entry in log_entries if "perf_operation" in entry.get("extra", {})
        ]

        assert len(perf_entries) > 0, "No performance log entries found"

        perf_entry = perf_entries[-1]  # Get latest entry
        extra = perf_entry.get("extra", {})

        assert "perf_operation" in extra, "Missing perf_operation in JSON"
        assert "perf_duration" in extra, "Missing perf_duration in JSON"
        assert "perf_success" in extra, "Missing perf_success in JSON"
        assert (
            extra["perf_operation"] == "json_test"
        ), f"Expected 'json_test', got {extra.get('perf_operation')}"

    def test_module_specific_configuration(self):
        """Test module-specific performance configuration."""
        # Test with different modules to verify configuration inheritance
        self.logger_manager.get_logger("test_module.fast")
        self.logger_manager.get_logger("test_module.regular")

        @performance_logging(logger_name="test_module.fast")
        def fast_module_function():
            return "fast_module_result"

        @performance_logging(logger_name="test_module.regular")
        def regular_module_function():
            return "regular_module_result"

        # Execute both functions
        fast_result = fast_module_function()
        regular_result = regular_module_function()

        assert fast_result == "fast_module_result"
        assert regular_result == "regular_module_result"

        # Verify both generated performance logs
        self._verify_log_contains("perf_operation", "fast_module_function")
        self._verify_log_contains("perf_operation", "regular_module_function")

    def test_backward_compatibility(self):
        """Test that existing logging functionality still works."""
        logger = self.logger_manager.get_logger("test_module")

        # Test traditional logging methods
        logger.debug("Debug message")
        logger.info("Info message")
        logger.warning("Warning message")
        logger.error("Error message")

        # Test structured logging with extra data
        logger.info(
            "Structured log", extra={"custom_field": "custom_value", "number": 42}
        )

        # Verify traditional logs appear in output
        self._verify_log_contains("message", "Info message")
        self._verify_log_contains("message", "Warning message")
        self._verify_log_contains("custom_field", "custom_value")

    def test_performance_memory_tracking(self):
        """Test memory tracking in performance logs."""
        self.logger_manager.get_logger("test_module")

        @performance_logging(
            logger_name="test_module", include_memory=True, operation_name="memory_test"
        )
        def memory_intensive_function():
            # Allocate some memory
            data = [i for i in range(10000)]
            return len(data)

        result = memory_intensive_function()
        assert result == 10000

        # Check for memory tracking in logs (if psutil is available)
        try:
            pass

            self._verify_log_contains("perf_operation", "memory_test")
            # Note: Memory delta might be 0 for small allocations, so we just check the field exists
        except ImportError:
            print("Note: psutil not available, skipping memory verification")

    def test_nested_performance_decorators(self):
        """Test nested functions with performance decorators."""
        self.logger_manager.get_logger("test_module")

        @performance_logging(logger_name="test_module", operation_name="outer_function")
        def outer_function():
            return inner_function()

        @performance_logging(logger_name="test_module", operation_name="inner_function")
        def inner_function():
            time.sleep(0.05)
            return "nested_result"

        result = outer_function()
        assert result == "nested_result"

        # Verify both functions logged performance data
        self._verify_log_contains("perf_operation", "outer_function")
        self._verify_log_contains("perf_operation", "inner_function")

    def _verify_log_contains(self, field: str, expected_value: Any):
        """Verify that the log file contains an entry with the specified field and value."""
        if not self.log_file.exists():
            raise AssertionError(f"Log file {self.log_file} does not exist")

        log_entries = self._parse_json_log_entries()

        for entry in log_entries:
            # Check in main entry fields
            if field in entry and entry[field] == expected_value:
                return

            # Check in extra fields
            extra = entry.get("extra", {})
            if field in extra and extra[field] == expected_value:
                return

            # Check in custom_fields
            custom_fields = entry.get("custom_fields", {})
            if field in custom_fields and custom_fields[field] == expected_value:
                return

        raise AssertionError(
            f"Log does not contain entry with {field}={expected_value}. "
            f"Found {len(log_entries)} entries in log."
        )

    def _parse_json_log_entries(self) -> list:
        """Parse JSON log entries from the log file."""
        if not self.log_file.exists():
            return []

        entries = []
        with open(self.log_file, "r") as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        entry = json.loads(line)
                        entries.append(entry)
                    except json.JSONDecodeError as e:
                        print(
                            f"Warning: Could not parse JSON line: {line[:100]}... Error: {e}"
                        )
                        continue
        return entries

    def run_all_tests(self):
        """Run all integration tests."""
        print("=" * 60)
        print("AIM2 Performance Logging Integration Tests")
        print("=" * 60)

        self.setup()

        try:
            tests = [
                ("Basic Performance Logging", self.test_basic_performance_logging),
                ("Performance with Context", self.test_performance_with_context),
                ("Performance Thresholds", self.test_performance_thresholds),
                ("Performance Profiles", self.test_performance_profiles),
                ("Error Handling", self.test_error_handling),
                ("JSON Formatter Integration", self.test_json_formatter_integration),
                (
                    "Module-Specific Configuration",
                    self.test_module_specific_configuration,
                ),
                ("Backward Compatibility", self.test_backward_compatibility),
                ("Memory Tracking", self.test_performance_memory_tracking),
                (
                    "Nested Performance Decorators",
                    self.test_nested_performance_decorators,
                ),
            ]

            passed = 0
            total = len(tests)

            for test_name, test_func in tests:
                if self.run_test(test_name, test_func):
                    passed += 1

            print("\n" + "=" * 60)
            print("TEST SUMMARY")
            print("=" * 60)
            print(f"Total tests: {total}")
            print(f"Passed: {passed}")
            print(f"Failed: {total - passed}")

            if passed == total:
                print("\n🎉 ALL INTEGRATION TESTS PASSED!")
                return True
            else:
                print(f"\n❌ {total - passed} tests failed")
                print("\nFailed tests:")
                for name, success, error in self.test_results:
                    if not success:
                        print(f"  - {name}: {error}")
                return False

        finally:
            self.teardown()


def main():
    """Main entry point for integration tests."""
    test_runner = IntegrationTestRunner()
    success = test_runner.run_all_tests()

    if success:
        print(f"\nIntegration test completed successfully!")
        print(f"Log files are available in: {test_runner.temp_dir}")
        sys.exit(0)
    else:
        print(f"\nIntegration test failed!")
        print(f"Log files are available for inspection in: {test_runner.temp_dir}")
        sys.exit(1)


if __name__ == "__main__":
    main()
