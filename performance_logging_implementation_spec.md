# Performance Logging Implementation Specification

## Technical Implementation Details for AIM2-004-10

### Module Structure

```
aim2_project/aim2_utils/
├── performance_logging.py          # Main performance decorators
├── performance_tracker.py          # Core tracking logic
├── performance_config.py           # Configuration extensions
└── performance_metrics.py          # Metrics data structures
```

## 1. Core Implementation Files

### 1.1 performance_metrics.py

```python
"""
Performance Metrics Data Structures for AIM2 Project

Defines data structures for capturing and managing performance metrics
that integrate with the existing logging and JSON formatting systems.
"""

import time
import traceback
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, Optional, Union
from datetime import datetime


@dataclass
class PerformanceMetrics:
    """Comprehensive performance metrics data structure."""

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
        if self.end_time and hasattr(self, 'start_time'):
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
            data['start_timestamp'] = datetime.fromtimestamp(self.start_time).isoformat()
        if self.end_time:
            data['end_timestamp'] = datetime.fromtimestamp(self.end_time).isoformat()

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
```

### 1.2 performance_tracker.py

```python
"""
Performance Tracker for AIM2 Project

Core tracking logic that integrates with existing context injection
and logging systems.
"""

import gc
import os
import psutil
import threading
from typing import Any, Callable, Dict, Optional
from contextlib import contextmanager

from .performance_metrics import PerformanceMetrics, PerformanceConfig
from .request_context import RequestContextManager, get_global_context_manager
from .logger_config import LoggerConfig


class PerformanceTracker:
    """Thread-safe performance tracker with context integration."""

    def __init__(
        self,
        operation_name: str,
        config: PerformanceConfig,
        logger,
        context_manager: Optional[RequestContextManager] = None
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

    def __enter__(self) -> 'PerformanceTracker':
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
            self.metrics.module_name = frame.f_globals.get('__name__')

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
            process = psutil.Process()
            memory_info = process.memory_info()
            self.metrics.memory_start_mb = memory_info.rss / 1024 / 1024
        except Exception:
            # Memory tracking is optional - don't fail on errors
            pass

    def _complete_memory_tracking(self) -> None:
        """Complete memory usage tracking."""
        try:
            process = psutil.Process()
            memory_info = process.memory_info()
            self.metrics.memory_end_mb = memory_info.rss / 1024 / 1024

            if self.metrics.memory_start_mb is not None:
                self.metrics.memory_delta_mb = (
                    self.metrics.memory_end_mb - self.metrics.memory_start_mb
                )
        except Exception:
            pass

    def _start_cpu_tracking(self) -> None:
        """Start CPU usage tracking."""
        try:
            # Initialize CPU percentage tracking
            psutil.cpu_percent()  # First call initializes
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
                    "performance": self.metrics.to_dict()  # For JSON formatter
                }
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
        import inspect
        try:
            # Walk up the stack to find the original function
            for frame_info in inspect.stack():
                frame = frame_info.frame
                if frame.f_code.co_name not in ('__enter__', '__exit__', 'wrapper', '_start_tracking'):
                    return frame
        except Exception:
            pass
        return None

    def _handle_tracking_error(self, error: Exception, phase: str) -> None:
        """Handle errors in performance tracking gracefully."""
        try:
            self.logger.warning(
                f"Performance tracking error during {phase}: {error}",
                extra={"performance_tracking_error": str(error), "phase": phase}
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
                return str_value[:max_length-3] + "..."
        except Exception:
            return f"<{type(value).__name__} object>"
```

### 1.3 performance_config.py

```python
"""
Performance Configuration Extensions for LoggerConfig

Extends the existing LoggerConfig class with performance logging
configuration options.
"""

from typing import Any, Dict, List, Optional
from .logger_config import LoggerConfig, LoggerConfigError
from .performance_metrics import PerformanceConfig


class PerformanceLoggerConfig(LoggerConfig):
    """Extended LoggerConfig with performance logging support."""

    # Performance-specific defaults
    PERFORMANCE_DEFAULTS = {
        "enable_performance_logging": True,
        "performance_thresholds": {
            "warning_seconds": 1.0,
            "critical_seconds": 5.0,
            "memory_warning_mb": 100,
            "memory_critical_mb": 500
        },
        "performance_profiles": {
            "fast": {"warning_seconds": 0.1, "critical_seconds": 0.5},
            "normal": {"warning_seconds": 1.0, "critical_seconds": 5.0},
            "slow": {"warning_seconds": 10.0, "critical_seconds": 30.0}
        },
        "performance_metrics": {
            "include_memory_usage": False,
            "include_cpu_usage": False,
            "include_function_args": False,
            "include_function_result": False,
            "max_arg_length": 200,
            "max_result_length": 200
        },
        "performance_logging_level": {
            "success": "INFO",
            "warning": "WARNING",
            "critical": "CRITICAL",
            "error": "ERROR"
        },
        "performance_context_prefix": "perf_",
        "performance_operation_field": "operation",
        "performance_auto_operation_naming": True
    }

    def __init__(self, env_prefix: str = "AIM2"):
        """Initialize with performance defaults."""
        super().__init__(env_prefix)
        # Merge performance defaults
        self.config.update(self.PERFORMANCE_DEFAULTS)

    def validate_config(self, config: Dict[str, Any]) -> bool:
        """Extended validation including performance settings."""
        # Call parent validation first
        super().validate_config(config)

        errors = []
        self._validate_performance_config(config, errors)

        if errors:
            error_msg = "Performance configuration validation failed:\n" + "\n".join(
                f"  - {err}" for err in errors
            )
            raise LoggerConfigError(error_msg)

        return True

    def _validate_performance_config(self, config: Dict[str, Any], errors: List[str]) -> None:
        """Validate performance logging configuration."""
        # Validate enable_performance_logging
        enable_perf = config.get("enable_performance_logging", True)
        if not isinstance(enable_perf, bool):
            errors.append("enable_performance_logging must be a boolean")

        # Validate performance_thresholds
        thresholds = config.get("performance_thresholds", {})
        if not isinstance(thresholds, dict):
            errors.append("performance_thresholds must be a dictionary")
        else:
            self._validate_thresholds(thresholds, errors)

        # Validate performance_profiles
        profiles = config.get("performance_profiles", {})
        if not isinstance(profiles, dict):
            errors.append("performance_profiles must be a dictionary")
        else:
            self._validate_profiles(profiles, errors)

        # Validate performance_metrics
        metrics_config = config.get("performance_metrics", {})
        if not isinstance(metrics_config, dict):
            errors.append("performance_metrics must be a dictionary")
        else:
            self._validate_metrics_config(metrics_config, errors)

    def _validate_thresholds(self, thresholds: Dict[str, Any], errors: List[str]) -> None:
        """Validate threshold configuration."""
        for key, value in thresholds.items():
            if key.endswith("_seconds") or key.endswith("_mb"):
                if not isinstance(value, (int, float)) or value < 0:
                    errors.append(f"Threshold '{key}' must be a non-negative number")

        # Check logical relationships
        if ("warning_seconds" in thresholds and "critical_seconds" in thresholds and
            thresholds["warning_seconds"] >= thresholds["critical_seconds"]):
            errors.append("warning_seconds must be less than critical_seconds")

    def _validate_profiles(self, profiles: Dict[str, Any], errors: List[str]) -> None:
        """Validate performance profile configuration."""
        for profile_name, profile_config in profiles.items():
            if not isinstance(profile_config, dict):
                errors.append(f"Profile '{profile_name}' must be a dictionary")
                continue

            # Validate each profile's thresholds
            temp_errors = []
            self._validate_thresholds(profile_config, temp_errors)
            for error in temp_errors:
                errors.append(f"Profile '{profile_name}': {error}")

    def _validate_metrics_config(self, metrics_config: Dict[str, Any], errors: List[str]) -> None:
        """Validate metrics configuration."""
        # Boolean fields
        bool_fields = [
            "include_memory_usage", "include_cpu_usage",
            "include_function_args", "include_function_result"
        ]
        for field in bool_fields:
            if field in metrics_config and not isinstance(metrics_config[field], bool):
                errors.append(f"Metrics config '{field}' must be a boolean")

        # Integer fields
        int_fields = ["max_arg_length", "max_result_length"]
        for field in int_fields:
            if field in metrics_config:
                value = metrics_config[field]
                if not isinstance(value, int) or value < 1:
                    errors.append(f"Metrics config '{field}' must be a positive integer")

    def get_performance_config(self, profile: Optional[str] = None) -> PerformanceConfig:
        """Get performance configuration for decorators."""
        base_config = self.config["performance_thresholds"].copy()
        metrics_config = self.config["performance_metrics"].copy()

        # Apply profile overrides if specified
        if profile and profile in self.config["performance_profiles"]:
            profile_config = self.config["performance_profiles"][profile]
            base_config.update(profile_config)

        return PerformanceConfig(
            enabled=self.config["enable_performance_logging"],
            warning_threshold=base_config.get("warning_seconds", 1.0),
            critical_threshold=base_config.get("critical_seconds", 5.0),
            include_memory=metrics_config.get("include_memory_usage", False),
            include_cpu=metrics_config.get("include_cpu_usage", False),
            include_args=metrics_config.get("include_function_args", False),
            include_result=metrics_config.get("include_function_result", False),
            max_arg_length=metrics_config.get("max_arg_length", 200),
            max_result_length=metrics_config.get("max_result_length", 200),
            context_prefix=self.config.get("performance_context_prefix", "perf_"),
            auto_operation_naming=self.config.get("performance_auto_operation_naming", True)
        )
```

### 1.4 performance_logging.py

```python
"""
Performance Logging Decorators for AIM2 Project

Main decorator implementations that integrate with the existing
logging and context injection infrastructure.
"""

import asyncio
import functools
import inspect
from typing import Any, Callable, Dict, Optional, Union

from .performance_tracker import PerformanceTracker
from .performance_config import PerformanceLoggerConfig
from .performance_metrics import PerformanceConfig
from .request_context import get_global_context_manager
from .logger_manager import LoggerManager


# Global configuration instance
_global_performance_config: Optional[PerformanceLoggerConfig] = None


def get_performance_config() -> PerformanceLoggerConfig:
    """Get global performance configuration."""
    global _global_performance_config
    if _global_performance_config is None:
        _global_performance_config = PerformanceLoggerConfig()
    return _global_performance_config


def set_performance_config(config: PerformanceLoggerConfig) -> None:
    """Set global performance configuration."""
    global _global_performance_config
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
    profile: Optional[str] = None
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
        global_config = get_performance_config()

        # Create performance config
        if profile:
            perf_config = global_config.get_performance_config(profile)
        else:
            perf_config = global_config.get_performance_config()

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
            with PerformanceTracker(op_name, perf_config, logger, context_manager) as tracker:
                try:
                    # Set function arguments for tracking
                    tracker.set_function_args(args, kwargs)

                    # Execute function
                    result = func(*args, **kwargs)

                    # Set result for tracking
                    tracker.set_result(result)

                    return result

                except Exception as e:
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
    operation_name: Optional[str] = None,
    **kwargs: Any
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
        # Apply standard performance logging
        perf_func = performance_logging(operation_name, **kwargs)(func)

        @functools.wraps(func)
        async def async_wrapper(*args, **kwargs):
            # For async functions, we need to handle the coroutine
            if asyncio.iscoroutinefunction(func):
                return await perf_func(*args, **kwargs)
            else:
                return perf_func(*args, **kwargs)

        return async_wrapper

    return decorator


def method_performance_logging(
    include_class_name: bool = True,
    **kwargs: Any
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
        # Generate operation name including class if requested
        def get_method_operation_name(*args, **method_kwargs):
            if include_class_name and args:
                instance = args[0]
                class_name = instance.__class__.__name__
                return f"{class_name}.{func.__name__}"
            return func.__name__

        # Create dynamic operation name
        operation_name = kwargs.pop('operation_name', None)
        if operation_name is None:
            # We'll set this dynamically in the wrapper
            pass

        @functools.wraps(func)
        def wrapper(*args, **method_kwargs):
            # Set operation name dynamically
            current_operation_name = operation_name or get_method_operation_name(*args, **method_kwargs)

            # Apply performance logging with dynamic name
            perf_decorator = performance_logging(
                operation_name=current_operation_name,
                **kwargs
            )
            return perf_decorator(func)(*args, **method_kwargs)

        return wrapper

    return decorator


def conditional_performance_logging(
    condition: Union[bool, Callable[[], bool]],
    **kwargs: Any
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


def performance_profile(
    profile: str,
    **overrides: Any
) -> Callable:
    """
    Decorator using predefined performance profiles.

    Args:
        profile: Profile name ("fast", "normal", "slow", etc.)
        **overrides: Override specific profile settings

    Returns:
        Decorated function with profile-based performance monitoring
    """

    def decorator(func: Callable) -> Callable:
        return performance_logging(profile=profile, **overrides)(func)

    return decorator


# Helper functions

def _generate_operation_name(func: Callable) -> str:
    """Generate operation name from function metadata."""
    module_name = getattr(func, '__module__', 'unknown')
    function_name = getattr(func, '__name__', 'unknown')

    # Clean up module name
    if module_name.startswith('__'):
        module_name = 'main'

    return f"{module_name}.{function_name}"


def _get_performance_logger(func: Callable, logger_name: Optional[str]):
    """Get logger for performance monitoring."""
    if logger_name:
        name = logger_name
    else:
        # Generate from function
        module_name = getattr(func, '__module__', 'unknown')
        function_name = getattr(func, '__name__', 'unknown')
        name = f"{module_name}.{function_name}"

    # Use LoggerManager if available, otherwise standard logging
    try:
        from .logger_manager import LoggerManager
        # Try to get from global logger manager
        import logging
        return logging.getLogger(name)
    except ImportError:
        import logging
        return logging.getLogger(name)


# Context manager for manual performance tracking

@contextmanager
def track_performance(
    operation_name: str,
    **kwargs: Any
):
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
    perf_config = config.get_performance_config()

    # Override with provided kwargs
    for key, value in kwargs.items():
        if hasattr(perf_config, key):
            setattr(perf_config, key, value)

    logger = _get_performance_logger(
        type('DummyFunc', (), {'__module__': 'manual', '__name__': operation_name})(),
        None
    )

    with PerformanceTracker(operation_name, perf_config, logger) as tracker:
        yield tracker
```

## 2. Integration Examples

### 2.1 LoggerManager Integration

```python
# Add to logger_manager.py

def create_performance_logger(self, name: str, **perf_kwargs) -> logging.Logger:
    """Create logger with performance monitoring enabled."""
    logger = self.get_logger(name)

    # Configure performance settings
    if self.config.get_enable_performance_logging():
        # Performance configuration is handled by decorators
        pass

    return logger
```

### 2.2 JSON Formatter Enhancement

```python
# Add to json_formatter.py

def _extract_performance_fields(self, record: logging.LogRecord) -> Dict[str, Any]:
    """Extract performance metrics from log record."""
    performance_data = {}

    # Check for performance metrics in extra
    if hasattr(record, 'performance_metrics'):
        performance_data = record.performance_metrics
    elif hasattr(record, 'performance'):
        performance_data = record.performance

    return performance_data

# Modify _build_json_data to include performance
if hasattr(record, 'performance') or hasattr(record, 'performance_metrics'):
    performance_fields = self._extract_performance_fields(record)
    if performance_fields:
        json_data['performance'] = performance_fields
```

## 3. Testing Strategy

### 3.1 Unit Tests Structure

```
tests/
├── test_performance_logging.py
├── test_performance_tracker.py
├── test_performance_config.py
└── test_performance_integration.py
```

### 3.2 Key Test Cases

1. **Basic decorator functionality**
2. **Threshold evaluation**
3. **Context injection integration**
4. **Error handling scenarios**
5. **Async function support**
6. **Thread safety**
7. **Configuration validation**
8. **JSON formatter integration**

This implementation provides a complete, robust performance logging system that integrates seamlessly with the existing AIM2 logging infrastructure while following all established patterns and maintaining backwards compatibility.
