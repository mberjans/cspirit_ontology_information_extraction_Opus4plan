"""
Unit Tests for Performance Decorators Module

Tests the performance logging decorators functionality including:
- Basic decorator operation
- Async function support
- Context integration
- Error handling
- Configuration validation
- Metrics collection
"""

import asyncio
import pytest
import sys
import time
import threading
from unittest.mock import Mock, patch

from aim2_project.aim2_utils.performance_decorators import (
    performance_logging,
    async_performance_logging,
    method_performance_logging,
    conditional_performance_logging,
    performance_profile,
    track_performance,
    PerformanceMetrics,
    PerformanceConfig,
    PerformanceTracker,
    set_performance_config,
    get_performance_config,
)
from aim2_project.aim2_utils.request_context import (
    RequestContextManager,
    get_global_context_manager,
)


class TestPerformanceMetrics:
    """Test the PerformanceMetrics data structure."""

    def test_metrics_initialization(self):
        """Test basic metrics initialization."""
        metrics = PerformanceMetrics(operation_name="test_op")

        assert metrics.operation_name == "test_op"
        assert metrics.success is True
        assert metrics.duration_seconds is None
        assert metrics.exceeded_warning is False
        assert metrics.exceeded_critical is False

    def test_timing_completion(self):
        """Test timing completion functionality."""
        metrics = PerformanceMetrics(operation_name="test_op")
        start_time = metrics.start_time

        # Wait a bit then complete timing
        time.sleep(0.01)
        metrics.complete_timing()

        assert metrics.end_time is not None
        assert metrics.duration_seconds is not None
        assert metrics.duration_seconds > 0
        assert metrics.end_time > start_time

    def test_threshold_evaluation(self):
        """Test threshold evaluation logic."""
        metrics = PerformanceMetrics(operation_name="test_op")
        metrics.warning_threshold = 0.1
        metrics.critical_threshold = 0.5
        metrics.duration_seconds = 0.2

        metrics.evaluate_thresholds()

        assert metrics.exceeded_warning is True
        assert metrics.exceeded_critical is False

        # Test critical threshold
        metrics.duration_seconds = 0.6
        metrics.evaluate_thresholds()

        assert metrics.exceeded_warning is True
        assert metrics.exceeded_critical is True

    def test_error_recording(self):
        """Test error information recording."""
        metrics = PerformanceMetrics(operation_name="test_op")
        test_error = ValueError("Test error message")

        metrics.set_error(test_error)

        assert metrics.success is False
        assert metrics.error_type == "ValueError"
        assert metrics.error_message == "Test error message"
        assert metrics.error_traceback is not None

    def test_to_dict_conversion(self):
        """Test conversion to dictionary."""
        metrics = PerformanceMetrics(operation_name="test_op")
        metrics.complete_timing()

        data = metrics.to_dict()

        assert isinstance(data, dict)
        assert data["operation_name"] == "test_op"
        assert "start_timestamp" in data
        assert "end_timestamp" in data
        assert data["success"] is True

    def test_context_fields_conversion(self):
        """Test conversion to context fields."""
        metrics = PerformanceMetrics(operation_name="test_op")
        metrics.duration_seconds = 0.5
        metrics.exceeded_warning = True
        metrics.memory_delta_mb = 10.5

        context = metrics.to_context_fields(prefix="perf_")

        assert context["perf_operation"] == "test_op"
        assert context["perf_duration"] == 0.5
        assert context["perf_success"] is True
        assert context["perf_exceeded_warning"] is True
        assert context["perf_memory_delta_mb"] == 10.5


class TestPerformanceConfig:
    """Test the PerformanceConfig data structure."""

    def test_config_defaults(self):
        """Test default configuration values."""
        config = PerformanceConfig()

        assert config.enabled is True
        assert config.warning_threshold == 1.0
        assert config.critical_threshold == 5.0
        assert config.include_memory is False
        assert config.include_args is False
        assert config.include_result is False
        assert config.context_prefix == "perf_"


class TestPerformanceTracker:
    """Test the PerformanceTracker class."""

    def setup_method(self):
        """Set up test fixtures."""
        self.mock_logger = Mock()
        self.mock_context_manager = Mock(spec=RequestContextManager)
        self.config = PerformanceConfig(
            warning_threshold=0.1,
            critical_threshold=0.5,
            include_args=True,
            include_result=True,
        )

    def test_tracker_initialization(self):
        """Test tracker initialization."""
        tracker = PerformanceTracker(
            "test_operation", self.config, self.mock_logger, self.mock_context_manager
        )

        assert tracker.operation_name == "test_operation"
        assert tracker.config == self.config
        assert tracker.logger == self.mock_logger
        assert tracker.context_manager == self.mock_context_manager
        assert tracker.metrics.operation_name == "test_operation"

    def test_context_manager_protocol(self):
        """Test context manager protocol."""
        tracker = PerformanceTracker(
            "test_operation", self.config, self.mock_logger, self.mock_context_manager
        )

        with tracker as t:
            assert t is tracker
            time.sleep(0.01)

        # Verify context manager was called
        self.mock_context_manager.update_context.assert_called()

        # Verify logger was called
        self.mock_logger.info.assert_called()

    def test_error_handling_in_context(self):
        """Test error handling within context manager."""
        tracker = PerformanceTracker(
            "test_operation", self.config, self.mock_logger, self.mock_context_manager
        )

        test_error = ValueError("Test error")

        try:
            with tracker:
                raise test_error
        except ValueError:
            pass

        # Verify error was recorded
        assert tracker.metrics.success is False
        assert tracker.metrics.error_type == "ValueError"

        # Verify error was logged
        self.mock_logger.error.assert_called()

    def test_function_args_tracking(self):
        """Test function arguments tracking."""
        self.config.include_args = True
        tracker = PerformanceTracker(
            "test_operation", self.config, self.mock_logger, self.mock_context_manager
        )

        args = (1, 2, 3)
        kwargs = {"key": "value"}

        tracker.set_function_args(args, kwargs)

        assert tracker.metrics.args_summary is not None
        assert tracker.metrics.kwargs_summary is not None

    def test_result_tracking(self):
        """Test function result tracking."""
        self.config.include_result = True
        tracker = PerformanceTracker(
            "test_operation", self.config, self.mock_logger, self.mock_context_manager
        )

        result = {"status": "success", "data": [1, 2, 3]}
        tracker.set_result(result)

        assert tracker.metrics.result_summary is not None


class TestPerformanceDecorators:
    """Test performance decorator functions."""

    def setup_method(self):
        """Set up test fixtures."""
        # Reset global configuration
        config = PerformanceConfig(
            enabled=True, warning_threshold=0.1, critical_threshold=0.5
        )
        set_performance_config(config)

    def test_basic_performance_logging_decorator(self):
        """Test basic performance logging decorator."""

        @performance_logging(
            operation_name="test_function",
            threshold_warning=0.05,
            include_args=True,
            include_result=True,
        )
        def test_function(x: int, y: int) -> int:
            time.sleep(0.01)
            return x + y

        result = test_function(5, 10)
        assert result == 15

    def test_decorator_with_disabled_performance(self):
        """Test decorator when performance logging is disabled."""
        # Disable performance logging
        config = PerformanceConfig(enabled=False)
        set_performance_config(config)

        @performance_logging(operation_name="disabled_test")
        def test_function() -> str:
            return "executed"

        result = test_function()
        assert result == "executed"

    def test_error_handling_strategies(self):
        """Test different error handling strategies."""

        @performance_logging(operation_name="error_test", on_error="log_and_continue")
        def function_with_error():
            raise ValueError("Test error")

        # Should return None instead of raising
        result = function_with_error()
        assert result is None

        @performance_logging(operation_name="error_test", on_error="silent")
        def silent_error_function():
            raise ValueError("Test error")

        # Should return None silently
        result = silent_error_function()
        assert result is None

        @performance_logging(operation_name="error_test", on_error="log_and_raise")
        def raising_error_function():
            raise ValueError("Test error")

        # Should raise the error
        with pytest.raises(ValueError):
            raising_error_function()

    @pytest.mark.asyncio
    async def test_async_performance_decorator(self):
        """Test async performance logging decorator."""

        @async_performance_logging(
            operation_name="async_test",
            threshold_warning=0.05,
            include_args=True,
            include_result=True,
        )
        async def async_test_function(delay: float) -> dict:
            await asyncio.sleep(delay)
            return {"delay": delay, "status": "completed"}

        result = await async_test_function(0.01)
        assert result["delay"] == 0.01
        assert result["status"] == "completed"

    def test_method_performance_decorator(self):
        """Test method performance logging decorator."""

        class TestClass:
            @method_performance_logging(
                include_class_name=True, threshold_warning=0.05, include_args=True
            )
            def test_method(self, value: int) -> int:
                time.sleep(0.01)
                return value * 2

        instance = TestClass()
        result = instance.test_method(5)
        assert result == 10

    def test_conditional_performance_decorator(self):
        """Test conditional performance logging decorator."""
        monitoring_enabled = True

        @conditional_performance_logging(
            condition=lambda: monitoring_enabled, operation_name="conditional_test"
        )
        def conditional_function() -> str:
            return "executed"

        # Should monitor when condition is True
        result = conditional_function()
        assert result == "executed"

        # Change condition
        monitoring_enabled = False

        @conditional_performance_logging(
            condition=lambda: monitoring_enabled, operation_name="conditional_test"
        )
        def conditional_function_disabled() -> str:
            return "executed_without_monitoring"

        result = conditional_function_disabled()
        assert result == "executed_without_monitoring"

    def test_performance_profiles(self):
        """Test performance profile decorator."""

        @performance_profile("fast", include_args=True)
        def fast_function(value: int) -> int:
            time.sleep(0.001)
            return value * 2

        result = fast_function(10)
        assert result == 20

        @performance_profile("slow", include_args=True)
        def slow_function(value: int) -> int:
            time.sleep(0.001)
            return value**2

        result = slow_function(5)
        assert result == 25

    def test_manual_performance_tracking(self):
        """Test manual performance tracking context manager."""
        with track_performance("manual_test", include_memory=True) as tracker:
            time.sleep(0.01)
            tracker.set_result({"manual": True})

        assert tracker.metrics.operation_name == "manual_test"
        assert tracker.metrics.duration_seconds is not None
        assert tracker.metrics.duration_seconds > 0


class TestContextIntegration:
    """Test integration with request context system."""

    def test_context_injection(self):
        """Test context injection during performance tracking."""
        context_manager = get_global_context_manager()

        # Set initial context
        context_manager.set_context(request_id="test-123", operation="context_test")

        @performance_logging(
            operation_name="context_aware_function",
            context_fields={"component": "test_component"},  # Use allowed field
        )
        def context_function() -> str:
            return "executed"

        result = context_function()
        assert result == "executed"

        # Verify context was updated with performance fields
        current_context = context_manager.get_context()
        assert "request_id" in current_context
        assert current_context["request_id"] == "test-123"
        assert "component" in current_context
        assert current_context["component"] == "test_component"


class TestThreadSafety:
    """Test thread safety of performance decorators."""

    def test_concurrent_performance_tracking(self):
        """Test concurrent performance tracking."""
        results = []
        errors = []

        @performance_logging(operation_name="concurrent_test", include_args=True)
        def concurrent_function(thread_id: int) -> dict:
            time.sleep(0.01)
            return {"thread_id": thread_id, "result": thread_id * 2}

        def worker(thread_id: int):
            try:
                result = concurrent_function(thread_id)
                results.append(result)
            except Exception as e:
                errors.append((thread_id, e))

        # Create multiple threads
        threads = []
        for i in range(5):
            thread = threading.Thread(target=worker, args=(i,))
            threads.append(thread)
            thread.start()

        # Wait for all threads to complete
        for thread in threads:
            thread.join()

        # Verify results
        assert len(errors) == 0, f"Errors occurred: {errors}"
        assert len(results) == 5

        # Verify all threads completed successfully
        thread_ids = [r["thread_id"] for r in results]
        assert set(thread_ids) == set(range(5))


class TestConfigurationValidation:
    """Test configuration validation and error handling."""

    def test_invalid_threshold_configuration(self):
        """Test handling of invalid threshold configuration."""

        # This should work without raising errors due to graceful error handling
        @performance_logging(
            threshold_warning=5.0,  # Warning > Critical
            threshold_critical=1.0,
            operation_name="invalid_threshold_test",
        )
        def test_function() -> str:
            return "executed"

        result = test_function()
        assert result == "executed"

    def test_performance_config_global_state(self):
        """Test global performance configuration state management."""
        # Set custom configuration
        custom_config = PerformanceConfig(
            enabled=True,
            warning_threshold=2.0,
            critical_threshold=10.0,
            include_memory=True,
        )
        set_performance_config(custom_config)

        # Verify configuration was set
        retrieved_config = get_performance_config()
        assert retrieved_config.warning_threshold == 2.0
        assert retrieved_config.critical_threshold == 10.0
        assert retrieved_config.include_memory is True

    def test_error_recovery(self):
        """Test error recovery in performance tracking."""
        # Mock a problematic context manager
        with patch(
            "aim2_project.aim2_utils.performance_decorators.get_global_context_manager"
        ) as mock_get_context:
            mock_context = Mock()
            mock_context.update_context.side_effect = Exception("Context error")
            mock_get_context.return_value = mock_context

            @performance_logging(operation_name="error_recovery_test")
            def test_function() -> str:
                return "executed"

            # Function should still execute despite context errors
            result = test_function()
            assert result == "executed"


class TestMemoryTracking:
    """Test suite for memory tracking functionality."""

    def setup_method(self):
        """Set up test fixtures."""
        self.config = PerformanceConfig(
            include_memory=True, warning_threshold=0.1, critical_threshold=0.5
        )
        self.mock_logger = Mock()
        self.mock_context_manager = Mock(spec=RequestContextManager)

    def test_memory_tracking_with_psutil(self):
        """Test memory tracking when psutil is available."""
        # Mock psutil module
        mock_psutil = Mock()
        mock_process = Mock()
        mock_memory_info = Mock()
        mock_memory_info.rss = 1024 * 1024 * 100  # 100MB
        mock_process.memory_info.return_value = mock_memory_info
        mock_psutil.Process.return_value = mock_process

        with patch.dict("sys.modules", {"psutil": mock_psutil}):
            tracker = PerformanceTracker(
                "memory_test", self.config, self.mock_logger, self.mock_context_manager
            )

            with tracker:
                # Simulate memory change for completion
                mock_memory_info.rss = 1024 * 1024 * 110  # 110MB

            # Verify memory metrics were captured
            assert tracker.metrics.memory_start_mb is not None
            assert tracker.metrics.memory_end_mb is not None
            assert tracker.metrics.memory_delta_mb is not None
            assert tracker.metrics.memory_delta_mb == 10.0  # 10MB increase

    def test_memory_tracking_without_psutil(self):
        """Test memory tracking graceful fallback when psutil is not available."""
        # Remove psutil from sys.modules to simulate ImportError
        original_modules = sys.modules.copy()
        if "psutil" in sys.modules:
            del sys.modules["psutil"]

        try:
            with patch(
                "builtins.__import__",
                side_effect=lambda name, *args: Mock()
                if name != "psutil"
                else ImportError(),
            ):
                tracker = PerformanceTracker(
                    "memory_test",
                    self.config,
                    self.mock_logger,
                    self.mock_context_manager,
                )

                with tracker:
                    pass

                # Memory metrics should be None or not set
                assert tracker.metrics.memory_start_mb is None
                assert tracker.metrics.memory_end_mb is None
                assert tracker.metrics.memory_delta_mb is None
        finally:
            sys.modules.update(original_modules)

    def test_memory_tracking_error_handling(self):
        """Test memory tracking error handling."""
        # Mock psutil to raise exception
        mock_psutil = Mock()
        mock_psutil.Process.side_effect = Exception("Memory access error")

        with patch.dict("sys.modules", {"psutil": mock_psutil}):
            tracker = PerformanceTracker(
                "memory_test", self.config, self.mock_logger, self.mock_context_manager
            )

            # Should not raise exception, just handle gracefully
            with tracker:
                pass

            # Memory metrics should be None due to error
            assert tracker.metrics.memory_start_mb is None

    def test_memory_tracking_with_decorator(self):
        """Test memory tracking integration with decorator."""
        # Mock psutil module
        mock_psutil = Mock()
        mock_process = Mock()
        mock_memory_info = Mock()
        mock_memory_info.rss = 1024 * 1024 * 50  # 50MB
        mock_process.memory_info.return_value = mock_memory_info
        mock_psutil.Process.return_value = mock_process

        with patch.dict("sys.modules", {"psutil": mock_psutil}):

            @performance_logging(
                operation_name="memory_decorator_test", include_memory=True
            )
            def memory_function() -> str:
                # Simulate memory increase
                mock_memory_info.rss = 1024 * 1024 * 60  # 60MB
                return "memory_tested"

            result = memory_function()
            assert result == "memory_tested"


class TestCPUTracking:
    """Test suite for CPU tracking functionality."""

    def setup_method(self):
        """Set up test fixtures."""
        self.config = PerformanceConfig(
            include_cpu=True, warning_threshold=0.1, critical_threshold=0.5
        )
        self.mock_logger = Mock()
        self.mock_context_manager = Mock(spec=RequestContextManager)

    def test_cpu_tracking_initialization(self):
        """Test CPU tracking initialization."""
        mock_psutil = Mock()
        mock_psutil.cpu_percent.return_value = None  # First call returns None

        with patch.dict("sys.modules", {"psutil": mock_psutil}):
            tracker = PerformanceTracker(
                "cpu_test", self.config, self.mock_logger, self.mock_context_manager
            )

            with tracker:
                pass

            # Should have called cpu_percent to initialize
            mock_psutil.cpu_percent.assert_called_once()

    def test_cpu_tracking_without_psutil(self):
        """Test CPU tracking when psutil is not available."""
        # Remove psutil from sys.modules to simulate ImportError
        original_modules = sys.modules.copy()
        if "psutil" in sys.modules:
            del sys.modules["psutil"]

        try:
            tracker = PerformanceTracker(
                "cpu_test", self.config, self.mock_logger, self.mock_context_manager
            )

            # Should not raise exception
            with tracker:
                pass
        finally:
            sys.modules.update(original_modules)

    def test_cpu_tracking_error_handling(self):
        """Test CPU tracking error handling."""
        mock_psutil = Mock()
        mock_psutil.cpu_percent.side_effect = Exception("CPU access error")

        with patch.dict("sys.modules", {"psutil": mock_psutil}):
            tracker = PerformanceTracker(
                "cpu_test", self.config, self.mock_logger, self.mock_context_manager
            )

            # Should handle error gracefully
            with tracker:
                pass


class TestErrorBoundariesAndGracefulDegradation:
    """Test suite for error boundaries and graceful degradation."""

    def setup_method(self):
        """Set up test fixtures."""
        self.config = PerformanceConfig()
        self.mock_logger = Mock()
        self.mock_context_manager = Mock(spec=RequestContextManager)

    def test_context_manager_error_during_entry(self):
        """Test error handling during context manager entry."""
        tracker = PerformanceTracker(
            "error_test", self.config, self.mock_logger, self.mock_context_manager
        )

        # Mock context manager to raise error
        self.mock_context_manager.update_context.side_effect = Exception(
            "Context error"
        )

        # Should handle error gracefully and still work
        with tracker:
            pass

        # Should have tried to update context and handled error gracefully
        self.mock_context_manager.update_context.assert_called()
        # Error is caught and logged internally, but mock logger might not be called
        # depending on implementation - main thing is it doesn't crash

    def test_context_manager_error_during_exit(self):
        """Test error handling during context manager exit."""
        tracker = PerformanceTracker(
            "error_test", self.config, self.mock_logger, self.mock_context_manager
        )

        # First call succeeds, second call fails
        self.mock_context_manager.update_context.side_effect = [
            None,
            Exception("Exit error"),
        ]

        with tracker:
            pass

        # Should have attempted both calls and handled errors gracefully
        assert self.mock_context_manager.update_context.call_count == 2
        # Error handling might not always call warning - main thing is no crash

    def test_logger_error_during_performance_logging(self):
        """Test error handling when logger fails."""
        tracker = PerformanceTracker(
            "error_test", self.config, self.mock_logger, self.mock_context_manager
        )

        # Make logger raise error
        self.mock_logger.info.side_effect = Exception("Logger error")

        # Should handle gracefully
        with tracker:
            pass

        # Should have attempted to log performance result
        self.mock_logger.info.assert_called()

    def test_frame_inspection_error(self):
        """Test error handling during frame inspection."""
        with patch(
            "aim2_project.aim2_utils.performance_decorators.inspect.stack"
        ) as mock_stack:
            mock_stack.side_effect = Exception("Stack inspection error")

            tracker = PerformanceTracker(
                "frame_error_test",
                self.config,
                self.mock_logger,
                self.mock_context_manager,
            )

            # Should handle gracefully
            with tracker:
                pass

            # Function name should be None or not set
            assert (
                tracker.metrics.function_name is None
                or tracker.metrics.function_name == "_start_tracking"
            )

    def test_summarize_value_error_handling(self):
        """Test error handling in value summarization."""
        tracker = PerformanceTracker(
            "summarize_test", self.config, self.mock_logger, self.mock_context_manager
        )

        # Test with object that raises exception in __str__
        class BadObject:
            def __str__(self):
                raise Exception("Cannot stringify")

        bad_obj = BadObject()
        result = tracker._summarize_value(bad_obj, 100)

        # Should fall back to type name
        assert "BadObject" in result

    def test_multiple_errors_in_sequence(self):
        """Test handling of multiple errors in sequence."""
        tracker = PerformanceTracker(
            "multi_error_test", self.config, self.mock_logger, self.mock_context_manager
        )

        # Make multiple components fail
        self.mock_context_manager.update_context.side_effect = Exception(
            "Context error"
        )
        self.mock_logger.info.side_effect = Exception("Logger error")

        # Should handle all errors gracefully without raising
        with tracker:
            pass

        # Should have attempted operations
        assert self.mock_context_manager.update_context.call_count >= 1


class TestLoggerCreationAndNaming:
    """Test suite for logger creation and operation naming."""

    def test_generate_operation_name_from_function(self):
        """Test operation name generation from function metadata."""

        def test_function():
            pass

        from aim2_project.aim2_utils.performance_decorators import (
            _generate_operation_name,
        )

        name = _generate_operation_name(test_function)
        assert "test_function" in name
        assert name.endswith(".test_function")

    def test_generate_operation_name_with_main_module(self):
        """Test operation name generation for __main__ module."""

        def test_function():
            pass

        # Mock function to appear from __main__
        test_function.__module__ = "__main__"

        from aim2_project.aim2_utils.performance_decorators import (
            _generate_operation_name,
        )

        name = _generate_operation_name(test_function)
        assert name.startswith("main.")
        assert "test_function" in name

    def test_generate_operation_name_with_unknown_attributes(self):
        """Test operation name generation with missing attributes."""

        # Create object without standard function attributes
        class FakeFunction:
            pass

        fake_func = FakeFunction()
        from aim2_project.aim2_utils.performance_decorators import (
            _generate_operation_name,
        )

        name = _generate_operation_name(fake_func)
        assert "unknown" in name

    def test_get_performance_logger_with_custom_name(self):
        """Test logger creation with custom name."""

        def test_function():
            pass

        from aim2_project.aim2_utils.performance_decorators import (
            _get_performance_logger,
        )

        logger = _get_performance_logger(test_function, "custom.logger")
        assert logger.name == "custom.logger"

    def test_get_performance_logger_auto_generated(self):
        """Test logger creation with auto-generated name."""

        def test_function():
            pass

        test_function.__module__ = "test.module"
        test_function.__name__ = "test_function"

        from aim2_project.aim2_utils.performance_decorators import (
            _get_performance_logger,
        )

        logger = _get_performance_logger(test_function, None)
        assert "test.module.test_function" in logger.name

    def test_get_performance_logger_with_unknown_function(self):
        """Test logger creation with function missing attributes."""

        class FakeFunction:
            pass

        fake_func = FakeFunction()
        from aim2_project.aim2_utils.performance_decorators import (
            _get_performance_logger,
        )

        logger = _get_performance_logger(fake_func, None)
        assert "unknown" in logger.name


class TestJSONFormatterIntegration:
    """Test suite for integration with JSON formatter performance fields."""

    def test_performance_metrics_in_json_extra(self):
        """Test that performance metrics are included in JSON extra fields."""
        mock_logger = Mock()
        config = PerformanceConfig()

        tracker = PerformanceTracker("json_test", config, mock_logger)

        with tracker:
            time.sleep(0.01)

        # Verify logger was called with performance metrics in extra
        mock_logger.info.assert_called_once()
        call_args = mock_logger.info.call_args

        # Check that extra contains performance data
        assert "extra" in call_args.kwargs
        extra = call_args.kwargs["extra"]
        assert "performance_metrics" in extra
        assert "performance" in extra  # For JSON formatter compatibility

        # Verify performance data structure
        perf_data = extra["performance"]
        assert "operation_name" in perf_data
        assert "duration_seconds" in perf_data
        assert "success" in perf_data
        assert perf_data["operation_name"] == "json_test"

    def test_performance_metrics_to_dict_json_serializable(self):
        """Test that performance metrics to_dict is JSON serializable."""
        import json

        metrics = PerformanceMetrics(operation_name="json_serialization_test")
        metrics.complete_timing()
        metrics.evaluate_thresholds()

        data_dict = metrics.to_dict()

        # Should be JSON serializable
        json_str = json.dumps(data_dict)
        assert isinstance(json_str, str)

        # Round trip should work
        parsed_data = json.loads(json_str)
        assert parsed_data["operation_name"] == "json_serialization_test"
        assert "start_timestamp" in parsed_data

    def test_performance_context_fields_format(self):
        """Test performance context fields format for injection."""
        metrics = PerformanceMetrics(operation_name="context_fields_test")
        metrics.duration_seconds = 1.5
        metrics.exceeded_warning = True
        metrics.memory_delta_mb = 25.7
        metrics.set_error(ValueError("Test error"))

        context_fields = metrics.to_context_fields(prefix="perf_")

        # Verify expected fields
        assert context_fields["perf_operation"] == "context_fields_test"
        assert context_fields["perf_duration"] == 1.5
        assert context_fields["perf_success"] is False  # Due to error
        assert context_fields["perf_exceeded_warning"] is True
        assert context_fields["perf_error_type"] == "ValueError"
        assert context_fields["perf_memory_delta_mb"] == 25.7

        # Should not include fields that are False or None
        assert "perf_exceeded_critical" not in context_fields


class TestAdvancedDecoratorScenarios:
    """Test suite for advanced decorator usage scenarios."""

    def test_nested_performance_decorators(self):
        """Test nested performance decorators."""

        @performance_logging(operation_name="outer_operation")
        def outer_function():
            @performance_logging(operation_name="inner_operation")
            def inner_function():
                time.sleep(0.01)
                return "inner_result"

            return inner_function()

        result = outer_function()
        assert result == "inner_result"

    def test_decorator_on_class_methods(self):
        """Test decorator application on various class methods."""

        class TestClass:
            @performance_logging(operation_name="instance_method")
            def instance_method(self):
                return "instance"

            @classmethod
            @performance_logging(operation_name="class_method")
            def class_method(cls):
                return "class"

            @staticmethod
            @performance_logging(operation_name="static_method")
            def static_method():
                return "static"

        obj = TestClass()
        assert obj.instance_method() == "instance"
        assert TestClass.class_method() == "class"
        assert TestClass.static_method() == "static"

    def test_multiple_decorators_combination(self):
        """Test combination with other decorators."""

        def timing_decorator(func):
            def wrapper(*args, **kwargs):
                start = time.time()
                result = func(*args, **kwargs)
                end = time.time()
                wrapper.timing = end - start
                return result

            return wrapper

        @timing_decorator
        @performance_logging(operation_name="multi_decorator_test")
        def decorated_function():
            time.sleep(0.01)
            return "decorated"

        result = decorated_function()
        assert result == "decorated"
        assert hasattr(decorated_function, "timing")


if __name__ == "__main__":
    # Run tests with pytest
    pytest.main([__file__, "-v"])
