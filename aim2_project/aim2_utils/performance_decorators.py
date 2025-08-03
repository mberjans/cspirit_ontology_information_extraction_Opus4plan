"""
Performance Logging Decorators for AIM2 Project

This module provides performance monitoring decorators that integrate with the existing
logging and context injection infrastructure. The decorators support both sync and async
functions, collect comprehensive metrics, and provide thread-safe performance tracking.

Classes:
    PerformanceError: Custom exception for performance-related errors
    PerformanceMetrics: Data structure for performance metrics
    PerformanceTracker: Core performance tracking implementation
    PerformanceConfig: Configuration for performance logging

Functions:
    performance_logging: Main decorator for performance monitoring
    async_performance_logging: Decorator for async function monitoring
    method_performance_logging: Decorator for class method monitoring
    track_performance: Context manager for manual performance tracking

Dependencies:
    - time: For performance timing
    - threading: For thread-safe operations
    - asyncio: For async function support
    - functools: For decorator implementation
    - inspect: For function metadata
    - typing: For type hints
    - contextlib: For context manager support
"""

import functools
import gc
import inspect
import logging
import os
import threading
import time
import traceback
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field
from datetime import datetime
from typing import Any, Callable, Dict, Optional, Union

from .request_context import RequestContextManager, get_global_context_manager


class PerformanceError(Exception):
    """
    Custom exception for performance-related errors.

    This exception is raised when performance monitoring operations encounter errors
    such as configuration issues, tracking failures, or metric collection problems.

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


@dataclass
class PerformanceMetrics:
    """
    Comprehensive performance metrics data structure.

    Captures timing, success/failure, function metadata, and optional resource usage
    metrics that integrate with the existing logging and JSON formatting systems.
    """

    # Core timing metrics
    operation_name: str
    start_time: float = field(default_factory=time.perf_counter)
    end_time: Optional[float] = None
    duration_seconds: Optional[float] = None

    # Success/failure tracking
    success: bool = True
    error_type: Optional[str] = None
    error_message: Optional[str] = None
    error_traceback: Optional[str] = None

    # Function metadata
    function_name: Optional[str] = None
    module_name: Optional[str] = None
    class_name: Optional[str] = None

    # Resource usage (optional)
    memory_start_mb: Optional[float] = None
    memory_end_mb: Optional[float] = None
    memory_delta_mb: Optional[float] = None
    memory_peak_mb: Optional[float] = None
    cpu_percent: Optional[float] = None

    # Function arguments and results (optional)
    args_summary: Optional[str] = None
    kwargs_summary: Optional[str] = None
    result_summary: Optional[str] = None

    # Threshold evaluation
    warning_threshold: Optional[float] = None
    critical_threshold: Optional[float] = None
    exceeded_warning: bool = False
    exceeded_critical: bool = False

    # Context information
    thread_id: Optional[int] = None
    process_id: Optional[int] = None

    def complete_timing(self) -> None:
        """Complete the timing measurement."""
        self.end_time = time.perf_counter()
        if self.end_time and hasattr(self, "start_time"):
            self.duration_seconds = self.end_time - self.start_time

    def evaluate_thresholds(self) -> None:
        """Evaluate performance against configured thresholds."""
        if self.duration_seconds is None:
            return

        if self.warning_threshold and self.duration_seconds >= self.warning_threshold:
            self.exceeded_warning = True

        if self.critical_threshold and self.duration_seconds >= self.critical_threshold:
            self.exceeded_critical = True

    def set_error(self, error: Exception) -> None:
        """Record error information."""
        self.success = False
        self.error_type = type(error).__name__
        self.error_message = str(error)
        self.error_traceback = traceback.format_exc()

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        data = asdict(self)

        # Convert timestamps to ISO format for better readability
        if self.start_time:
            data["start_timestamp"] = datetime.fromtimestamp(
                self.start_time
            ).isoformat()
        if self.end_time:
            data["end_timestamp"] = datetime.fromtimestamp(self.end_time).isoformat()

        return data

    def to_context_fields(self, prefix: str = "perf_") -> Dict[str, Any]:
        """Convert to context fields for injection."""
        context = {}

        # Core metrics
        context[f"{prefix}operation"] = self.operation_name
        context[f"{prefix}duration"] = self.duration_seconds
        context[f"{prefix}success"] = self.success

        # Threshold flags
        if self.exceeded_warning:
            context[f"{prefix}exceeded_warning"] = True
        if self.exceeded_critical:
            context[f"{prefix}exceeded_critical"] = True

        # Error information
        if not self.success:
            context[f"{prefix}error_type"] = self.error_type

        # Resource usage
        if self.memory_delta_mb is not None:
            context[f"{prefix}memory_delta_mb"] = self.memory_delta_mb

        return context


@dataclass
class PerformanceConfig:
    """Configuration for performance logging."""

    enabled: bool = True
    warning_threshold: float = 1.0
    critical_threshold: float = 5.0
    include_memory: bool = False
    include_cpu: bool = False
    include_args: bool = False
    include_result: bool = False
    max_arg_length: int = 200
    max_result_length: int = 200
    context_prefix: str = "perf_"
    auto_operation_naming: bool = True


class PerformanceTracker:
    """
    Thread-safe performance tracker with context integration.

    Core tracking logic that integrates with existing context injection
    and logging systems.
    """

    def __init__(
        self,
        operation_name: str,
        config: PerformanceConfig,
        logger,
        context_manager: Optional[RequestContextManager] = None,
    ):
        self.operation_name = operation_name
        self.config = config
        self.logger = logger
        self.context_manager = context_manager or get_global_context_manager()
        self.metrics = PerformanceMetrics(operation_name=operation_name)

        # Set thresholds
        self.metrics.warning_threshold = config.warning_threshold
        self.metrics.critical_threshold = config.critical_threshold

        # Thread safety
        self._lock = threading.RLock()

    def __enter__(self) -> "PerformanceTracker":
        """Start performance tracking."""
        with self._lock:
            try:
                self._start_tracking()
                self._inject_start_context()
            except Exception as e:
                self._handle_tracking_error(e, "start")
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Complete performance tracking."""
        with self._lock:
            try:
                if exc_type is not None:
                    self.metrics.set_error(exc_val)

                self._complete_tracking()
                self._inject_completion_context()
                self._log_performance_result()

            except Exception as e:
                self._handle_tracking_error(e, "completion")

    def _start_tracking(self) -> None:
        """Initialize performance tracking."""
        # Set function metadata
        frame = self._get_caller_frame()
        if frame:
            self.metrics.function_name = frame.f_code.co_name
            self.metrics.module_name = frame.f_globals.get("__name__")

        # Set process/thread info
        self.metrics.thread_id = threading.current_thread().ident
        self.metrics.process_id = os.getpid()

        # Initialize resource tracking
        if self.config.include_memory:
            self._start_memory_tracking()

        if self.config.include_cpu:
            self._start_cpu_tracking()

    def _complete_tracking(self) -> None:
        """Complete performance measurements."""
        self.metrics.complete_timing()

        if self.config.include_memory:
            self._complete_memory_tracking()

        self.metrics.evaluate_thresholds()

    def _start_memory_tracking(self) -> None:
        """Start memory usage tracking."""
        try:
            # Use basic memory tracking if psutil is not available
            gc.collect()  # Force garbage collection for more accurate measurement

            # Try to get memory info using psutil if available
            try:
                import psutil

                process = psutil.Process()
                memory_info = process.memory_info()
                self.metrics.memory_start_mb = memory_info.rss / 1024 / 1024
            except ImportError:
                # Fallback to basic method (not as accurate)
                pass
        except Exception:
            # Memory tracking is optional - don't fail on errors
            pass

    def _complete_memory_tracking(self) -> None:
        """Complete memory usage tracking."""
        try:
            gc.collect()  # Force garbage collection

            try:
                import psutil

                process = psutil.Process()
                memory_info = process.memory_info()
                self.metrics.memory_end_mb = memory_info.rss / 1024 / 1024

                if self.metrics.memory_start_mb is not None:
                    self.metrics.memory_delta_mb = (
                        self.metrics.memory_end_mb - self.metrics.memory_start_mb
                    )
            except ImportError:
                pass
        except Exception:
            pass

    def _start_cpu_tracking(self) -> None:
        """Start CPU usage tracking."""
        try:
            # Initialize CPU percentage tracking if psutil is available
            try:
                import psutil

                psutil.cpu_percent()  # First call initializes
            except ImportError:
                pass
        except Exception:
            pass

    def _inject_start_context(self) -> None:
        """Inject performance start context."""
        try:
            context_fields = {
                f"{self.config.context_prefix}operation": self.operation_name,
                f"{self.config.context_prefix}start_time": self.metrics.start_time,
            }
            self.context_manager.update_context(context_fields)
        except Exception:
            # Context injection is best-effort
            pass

    def _inject_completion_context(self) -> None:
        """Inject performance completion context."""
        try:
            context_fields = self.metrics.to_context_fields(self.config.context_prefix)
            self.context_manager.update_context(context_fields)
        except Exception:
            pass

    def _log_performance_result(self) -> None:
        """Log the performance result."""
        try:
            # Determine log level based on thresholds
            if not self.metrics.success:
                level = "error"
            elif self.metrics.exceeded_critical:
                level = "critical"
            elif self.metrics.exceeded_warning:
                level = "warning"
            else:
                level = "info"

            # Create log message
            message = self._create_log_message()

            # Log with performance metrics in extra
            getattr(self.logger, level)(
                message,
                extra={
                    "performance_metrics": self.metrics.to_dict(),
                    "performance": self.metrics.to_dict(),  # For JSON formatter
                },
            )

        except Exception as e:
            self._handle_tracking_error(e, "logging")

    def _create_log_message(self) -> str:
        """Create human-readable log message."""
        if not self.metrics.success:
            return f"Operation '{self.operation_name}' failed after {self.metrics.duration_seconds:.3f}s: {self.metrics.error_message}"

        status_parts = []
        if self.metrics.exceeded_critical:
            status_parts.append("CRITICAL THRESHOLD EXCEEDED")
        elif self.metrics.exceeded_warning:
            status_parts.append("WARNING THRESHOLD EXCEEDED")

        status = f" ({', '.join(status_parts)})" if status_parts else ""

        return f"Operation '{self.operation_name}' completed in {self.metrics.duration_seconds:.3f}s{status}"

    def _get_caller_frame(self):
        """Get the frame of the decorated function."""
        try:
            # Walk up the stack to find the original function
            for frame_info in inspect.stack():
                frame = frame_info.frame
                if frame.f_code.co_name not in (
                    "__enter__",
                    "__exit__",
                    "wrapper",
                    "_start_tracking",
                ):
                    return frame
        except Exception:
            pass
        return None

    def _handle_tracking_error(self, error: Exception, phase: str) -> None:
        """Handle errors in performance tracking gracefully."""
        try:
            self.logger.warning(
                f"Performance tracking error during {phase}: {error}",
                extra={"performance_tracking_error": str(error), "phase": phase},
            )
        except Exception:
            # Ultimate fallback - do nothing to avoid breaking the actual operation
            pass

    def set_function_args(self, args: tuple, kwargs: dict) -> None:
        """Set function arguments for tracking."""
        if not self.config.include_args:
            return

        try:
            args_str = self._summarize_value(args, self.config.max_arg_length)
            kwargs_str = self._summarize_value(kwargs, self.config.max_arg_length)

            self.metrics.args_summary = args_str
            self.metrics.kwargs_summary = kwargs_str
        except Exception:
            pass

    def set_result(self, result: Any) -> None:
        """Set function result for tracking."""
        if not self.config.include_result:
            return

        try:
            result_str = self._summarize_value(result, self.config.max_result_length)
            self.metrics.result_summary = result_str
        except Exception:
            pass

    def _summarize_value(self, value: Any, max_length: int) -> str:
        """Create a summary string of a value."""
        try:
            str_value = str(value)
            if len(str_value) <= max_length:
                return str_value
            else:
                return str_value[: max_length - 3] + "..."
        except Exception:
            return f"<{type(value).__name__} object>"


# Global configuration instance
_global_performance_config: Optional[PerformanceConfig] = None
_global_config_lock = threading.RLock()


def get_performance_config() -> PerformanceConfig:
    """Get global performance configuration."""
    global _global_performance_config
    if _global_performance_config is None:
        with _global_config_lock:
            if _global_performance_config is None:
                _global_performance_config = PerformanceConfig()
    return _global_performance_config


def set_performance_config(config: PerformanceConfig) -> None:
    """Set global performance configuration."""
    global _global_performance_config
    with _global_config_lock:
        _global_performance_config = config


def performance_logging(
    operation_name: Optional[str] = None,
    threshold_warning: Optional[float] = None,
    threshold_critical: Optional[float] = None,
    include_args: bool = False,
    include_result: bool = False,
    include_memory: bool = False,
    logger_name: Optional[str] = None,
    context_fields: Optional[Dict[str, Any]] = None,
    on_error: str = "log_and_raise",
    profile: Optional[str] = None,
) -> Callable:
    """
    Decorator for performance logging with automatic monitoring.

    Args:
        operation_name: Custom operation name (defaults to function name)
        threshold_warning: Warning threshold in seconds
        threshold_critical: Critical threshold in seconds
        include_args: Include function arguments in logging
        include_result: Include function result in logging
        include_memory: Include memory usage tracking
        logger_name: Custom logger name
        context_fields: Additional context fields
        on_error: Error handling strategy
        profile: Performance profile to use

    Returns:
        Decorated function with performance monitoring
    """

    def decorator(func: Callable) -> Callable:
        # Get configuration
        perf_config = get_performance_config()

        # Override with decorator parameters
        if threshold_warning is not None:
            perf_config.warning_threshold = threshold_warning
        if threshold_critical is not None:
            perf_config.critical_threshold = threshold_critical
        if include_args:
            perf_config.include_args = True
        if include_result:
            perf_config.include_result = True
        if include_memory:
            perf_config.include_memory = True

        # Determine operation name
        op_name = operation_name or _generate_operation_name(func)

        # Get logger
        logger = _get_performance_logger(func, logger_name)

        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # Skip if performance logging is disabled
            if not perf_config.enabled:
                return func(*args, **kwargs)

            # Set additional context if provided
            context_manager = get_global_context_manager()
            if context_fields:
                context_manager.update_context(context_fields)

            # Create performance tracker
            with PerformanceTracker(
                op_name, perf_config, logger, context_manager
            ) as tracker:
                try:
                    # Set function arguments for tracking
                    tracker.set_function_args(args, kwargs)

                    # Execute function
                    result = func(*args, **kwargs)

                    # Set result for tracking
                    tracker.set_result(result)

                    return result

                except Exception:
                    # Handle based on error strategy
                    if on_error == "log_and_continue":
                        return None
                    elif on_error == "silent":
                        return None
                    else:  # "log_and_raise"
                        raise

        return wrapper

    return decorator


def async_performance_logging(
    operation_name: Optional[str] = None, **kwargs: Any
) -> Callable:
    """
    Decorator for async function performance logging.

    Args:
        operation_name: Custom operation name
        **kwargs: Additional arguments passed to performance_logging

    Returns:
        Decorated async function with performance monitoring
    """

    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        async def async_wrapper(*args, **method_kwargs):
            # Get configuration
            perf_config = get_performance_config()

            # Apply parameter overrides
            config = PerformanceConfig(
                enabled=perf_config.enabled,
                warning_threshold=kwargs.get(
                    "threshold_warning", perf_config.warning_threshold
                ),
                critical_threshold=kwargs.get(
                    "threshold_critical", perf_config.critical_threshold
                ),
                include_args=kwargs.get("include_args", perf_config.include_args),
                include_result=kwargs.get("include_result", perf_config.include_result),
                include_memory=kwargs.get("include_memory", perf_config.include_memory),
                context_prefix=perf_config.context_prefix,
            )

            # Skip if performance logging is disabled
            if not config.enabled:
                return await func(*args, **method_kwargs)

            # Determine operation name
            op_name = operation_name or _generate_operation_name(func)

            # Get logger
            logger = _get_performance_logger(func, kwargs.get("logger_name"))

            # Set additional context if provided
            context_manager = get_global_context_manager()
            if kwargs.get("context_fields"):
                context_manager.update_context(kwargs["context_fields"])

            # Create performance tracker
            with PerformanceTracker(
                op_name, config, logger, context_manager
            ) as tracker:
                try:
                    # Set function arguments for tracking
                    tracker.set_function_args(args, method_kwargs)

                    # Execute async function
                    result = await func(*args, **method_kwargs)

                    # Set result for tracking
                    tracker.set_result(result)

                    return result

                except Exception:
                    # Handle based on error strategy
                    on_error = kwargs.get("on_error", "log_and_raise")
                    if on_error == "log_and_continue":
                        return None
                    elif on_error == "silent":
                        return None
                    else:  # "log_and_raise"
                        raise

        return async_wrapper

    return decorator


def method_performance_logging(
    include_class_name: bool = True, **kwargs: Any
) -> Callable:
    """
    Decorator for class method performance logging.

    Args:
        include_class_name: Include class name in operation name
        **kwargs: Additional arguments passed to performance_logging

    Returns:
        Decorated method with performance monitoring
    """

    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **method_kwargs):
            # Generate operation name including class if requested
            if include_class_name and args:
                instance = args[0]
                class_name = instance.__class__.__name__
                op_name = f"{class_name}.{func.__name__}"
            else:
                op_name = func.__name__

            # Apply performance logging with dynamic name
            perf_decorator = performance_logging(operation_name=op_name, **kwargs)
            return perf_decorator(func)(*args, **method_kwargs)

        return wrapper

    return decorator


def conditional_performance_logging(
    condition: Union[bool, Callable[[], bool]], **kwargs: Any
) -> Callable:
    """
    Decorator for conditional performance logging.

    Args:
        condition: Boolean or callable that returns boolean
        **kwargs: Additional arguments passed to performance_logging

    Returns:
        Decorated function with conditional performance monitoring
    """

    def decorator(func: Callable) -> Callable:
        # Create performance-enabled version
        perf_func = performance_logging(**kwargs)(func)

        @functools.wraps(func)
        def wrapper(*args, **method_kwargs):
            # Evaluate condition
            should_monitor = condition
            if callable(condition):
                try:
                    should_monitor = condition()
                except Exception:
                    should_monitor = False

            # Use appropriate version
            if should_monitor:
                return perf_func(*args, **method_kwargs)
            else:
                return func(*args, **method_kwargs)

        return wrapper

    return decorator


def performance_profile(profile: str, **overrides: Any) -> Callable:
    """
    Decorator using predefined performance profiles.

    Args:
        profile: Profile name ("fast", "normal", "slow", etc.)
        **overrides: Override specific profile settings

    Returns:
        Decorated function with profile-based performance monitoring
    """

    # Define performance profiles
    profiles = {
        "fast": {"threshold_warning": 0.1, "threshold_critical": 0.5},
        "normal": {"threshold_warning": 1.0, "threshold_critical": 5.0},
        "slow": {"threshold_warning": 10.0, "threshold_critical": 30.0},
    }

    def decorator(func: Callable) -> Callable:
        # Get profile settings
        profile_settings = profiles.get(profile, profiles["normal"])

        # Merge with overrides
        final_settings = {**profile_settings, **overrides}

        return performance_logging(**final_settings)(func)

    return decorator


# Helper functions


def _generate_operation_name(func: Callable) -> str:
    """Generate operation name from function metadata."""
    module_name = getattr(func, "__module__", "unknown")
    function_name = getattr(func, "__name__", "unknown")

    # Clean up module name
    if module_name.startswith("__"):
        module_name = "main"

    return f"{module_name}.{function_name}"


def _get_performance_logger(func: Callable, logger_name: Optional[str]):
    """Get logger for performance monitoring."""
    if logger_name:
        name = logger_name
    else:
        # Generate from function
        module_name = getattr(func, "__module__", "unknown")
        function_name = getattr(func, "__name__", "unknown")
        name = f"{module_name}.{function_name}"

    # Use standard logging to get logger
    return logging.getLogger(name)


# Context manager for manual performance tracking


@contextmanager
def track_performance(operation_name: str, **kwargs: Any):
    """
    Context manager for manual performance tracking.

    Args:
        operation_name: Name of the operation to track
        **kwargs: Additional configuration options

    Yields:
        PerformanceTracker: Tracker instance for manual control

    Example:
        with track_performance("manual_operation") as tracker:
            # Do work
            result = some_function()
            tracker.set_result(result)
    """
    config = get_performance_config()

    # Override with provided kwargs
    for key, value in kwargs.items():
        if hasattr(config, key):
            setattr(config, key, value)

    logger = _get_performance_logger(
        type("DummyFunc", (), {"__module__": "manual", "__name__": operation_name})(),
        None,
    )

    with PerformanceTracker(operation_name, config, logger) as tracker:
        yield tracker
