# Performance Logging Decorator Design for AIM2-004-10

## Overview

This document outlines the design for performance logging decorators that integrate seamlessly with the existing AIM2 logging infrastructure. The design follows established patterns from `context_injection.py` and integrates with the existing `LoggerConfig`, `JSONFormatter`, and `RequestContextManager` systems.

## 1. Decorator Interface Specifications

### 1.1 Core Performance Decorator

```python
def performance_logging(
    operation_name: Optional[str] = None,
    threshold_warning: Optional[float] = None,
    threshold_critical: Optional[float] = None,
    include_args: bool = False,
    include_result: bool = False,
    include_memory: bool = False,
    logger_name: Optional[str] = None,
    context_fields: Optional[Dict[str, Any]] = None,
    on_error: str = "log_and_raise"  # "log_and_raise", "log_and_continue", "silent"
) -> Callable
```

**Parameters:**
- `operation_name`: Custom name for the operation (defaults to function name)
- `threshold_warning`: Time in seconds to trigger WARNING level (uses config default if None)
- `threshold_critical`: Time in seconds to trigger CRITICAL level (uses config default if None)
- `include_args`: Whether to include function arguments in performance metrics
- `include_result`: Whether to include function result in performance metrics
- `include_memory`: Whether to track memory usage changes
- `logger_name`: Custom logger name (defaults to module.function)
- `context_fields`: Additional context fields for this operation
- `on_error`: Error handling strategy

### 1.2 Specialized Decorators

```python
# Decorator factory for different performance profiles
def performance_profile(
    profile: str,  # "fast", "normal", "slow", "custom"
    **overrides: Any
) -> Callable

# Async function support
def async_performance_logging(
    operation_name: Optional[str] = None,
    **kwargs: Any
) -> Callable

# Class method decorator with automatic operation naming
def method_performance_logging(
    include_class_name: bool = True,
    **kwargs: Any
) -> Callable

# Conditional performance logging
def conditional_performance_logging(
    condition: Union[bool, Callable[[], bool]],
    **kwargs: Any
) -> Callable
```

## 2. Configuration Schema Extensions

### 2.1 LoggerConfig Extensions

Add the following fields to `LoggerConfig.DEFAULT_CONFIG`:

```python
# Performance logging configuration
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
```

### 2.2 Validation Extensions

Add validation methods to `LoggerConfig`:

```python
def _validate_performance_config(self, config: Dict[str, Any], errors: List[str]) -> None:
    """Validate performance logging configuration."""

def _validate_performance_thresholds(self, thresholds: Dict[str, float], errors: List[str]) -> None:
    """Validate performance threshold values."""

def _validate_performance_profiles(self, profiles: Dict[str, Dict], errors: List[str]) -> None:
    """Validate performance profile configurations."""
```

## 3. Performance Metrics Data Structure

### 3.1 Core Metrics Structure

```python
@dataclass
class PerformanceMetrics:
    """Performance metrics data structure for logging."""

    operation_name: str
    start_time: float
    end_time: float
    duration_seconds: float
    success: bool
    error_type: Optional[str] = None
    error_message: Optional[str] = None

    # Memory metrics (optional)
    memory_start_mb: Optional[float] = None
    memory_end_mb: Optional[float] = None
    memory_delta_mb: Optional[float] = None
    memory_peak_mb: Optional[float] = None

    # CPU metrics (optional)
    cpu_percent: Optional[float] = None

    # Function metadata (optional)
    function_name: str = None
    module_name: str = None
    class_name: Optional[str] = None
    args_summary: Optional[str] = None
    result_summary: Optional[str] = None

    # Threshold evaluation
    exceeded_warning: bool = False
    exceeded_critical: bool = False
    warning_threshold: Optional[float] = None
    critical_threshold: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert metrics to dictionary for JSON logging."""

    def to_context_fields(self, prefix: str = "perf_") -> Dict[str, Any]:
        """Convert metrics to context fields for injection."""
```

### 3.2 JSON Formatter Integration

Extend `JSONFormatter` to handle performance metrics:

```python
# Add to JSONFormatter.FIELD_MAPPING
"performance": None,  # Special handling for performance metrics

# Add performance-specific field extraction
def _extract_performance_fields(self, record: logging.LogRecord) -> Dict[str, Any]:
    """Extract performance metrics from log record."""
```

## 4. Integration with Existing Context Injection

### 4.1 Context Manager Integration

```python
class PerformanceContextManager:
    """Manages performance-related context injection."""

    def __init__(self,
                 context_manager: RequestContextManager,
                 config: LoggerConfig):
        self.context_manager = context_manager
        self.config = config

    def inject_performance_context(self, metrics: PerformanceMetrics) -> None:
        """Inject performance metrics into request context."""

    def create_operation_context(self, operation_name: str) -> ContextManager:
        """Create context manager for operation tracking."""
```

### 4.2 Decorator Implementation Pattern

Following the pattern from `@with_request_context`:

```python
def performance_logging(
    operation_name: Optional[str] = None,
    threshold_warning: Optional[float] = None,
    threshold_critical: Optional[float] = None,
    **kwargs: Any
) -> Callable:
    """Performance logging decorator following existing patterns."""

    def decorator(func: Callable) -> Callable:
        def wrapper(*args, **kwargs):
            # Get performance configuration
            config = get_performance_config()
            logger = get_performance_logger(func)
            context_manager = get_global_context_manager()

            # Setup performance tracking
            with PerformanceTracker(
                operation_name or func.__name__,
                config,
                logger,
                context_manager
            ) as tracker:
                try:
                    # Execute function
                    result = func(*args, **kwargs)
                    tracker.set_success(result)
                    return result
                except Exception as e:
                    tracker.set_error(e)
                    # Handle based on error strategy
                    raise

        # Preserve function metadata
        wrapper.__name__ = func.__name__
        wrapper.__doc__ = func.__doc__
        wrapper.__module__ = func.__module__

        return wrapper
    return decorator
```

## 5. Example Usage Patterns

### 5.1 Basic Usage

```python
from aim2_project.aim2_utils.performance_logging import performance_logging

@performance_logging(operation_name="data_processing")
def process_data(data: List[Dict]) -> Dict:
    """Process data with automatic performance monitoring."""
    # Function implementation
    return result

# Results in log output:
# {
#   "timestamp": "2024-01-15T10:30:45.123Z",
#   "level": "INFO",
#   "logger_name": "mymodule.process_data",
#   "message": "Operation completed successfully",
#   "context": {
#     "request_id": "req_123",
#     "operation": "data_processing"
#   },
#   "performance": {
#     "operation_name": "data_processing",
#     "duration_seconds": 0.45,
#     "success": true,
#     "exceeded_warning": false,
#     "exceeded_critical": false
#   }
# }
```

### 5.2 Advanced Usage with Custom Thresholds

```python
@performance_logging(
    operation_name="slow_database_query",
    threshold_warning=2.0,
    threshold_critical=10.0,
    include_args=True,
    include_memory=True,
    context_fields={"component": "database", "query_type": "analytical"}
)
def execute_complex_query(query: str, params: Dict) -> List[Dict]:
    """Execute complex database query with detailed monitoring."""
    return results
```

### 5.3 Class Method Decoration

```python
class DataProcessor:
    @method_performance_logging(
        include_class_name=True,
        threshold_warning=1.0
    )
    def process_batch(self, batch_data: List) -> ProcessingResult:
        """Process a batch of data."""
        return result

    # Results in operation_name: "DataProcessor.process_batch"
```

### 5.4 Async Function Support

```python
@async_performance_logging(
    operation_name="async_api_call",
    threshold_warning=3.0
)
async def fetch_external_data(endpoint: str) -> Dict:
    """Fetch data from external API."""
    async with httpx.AsyncClient() as client:
        response = await client.get(endpoint)
        return response.json()
```

### 5.5 Conditional Performance Logging

```python
# Only log performance in production
@conditional_performance_logging(
    condition=lambda: os.getenv("ENVIRONMENT") == "production",
    operation_name="critical_operation"
)
def critical_business_logic():
    """Critical business logic with conditional monitoring."""
    pass

# Or with boolean flag
DEBUG_MODE = True
@conditional_performance_logging(
    condition=not DEBUG_MODE,
    threshold_warning=0.1
)
def development_function():
    """Function with conditional performance monitoring."""
    pass
```

### 5.6 Profile-Based Configuration

```python
# Use predefined performance profiles
@performance_profile("fast", include_memory=True)
def quick_calculation(x: float, y: float) -> float:
    """Quick mathematical calculation."""
    return x * y + math.sqrt(x)

@performance_profile("slow", include_args=True, include_result=True)
def complex_analysis(dataset: pd.DataFrame) -> AnalysisResult:
    """Complex data analysis operation."""
    return analysis_result
```

## 6. Error Handling Strategy

### 6.1 Error Handling Modes

1. **"log_and_raise" (default)**: Log performance metrics including error info, then re-raise
2. **"log_and_continue"**: Log error but return None/default value
3. **"silent"**: Continue without logging on errors

### 6.2 Fallback Behavior

```python
class PerformanceTracker:
    """Handles performance tracking with robust error handling."""

    def __enter__(self):
        try:
            self._start_tracking()
        except Exception as e:
            # Fallback: minimal tracking
            self._setup_minimal_tracking()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        try:
            self._complete_tracking(exc_type, exc_val, exc_tb)
        except Exception as e:
            # Fallback: log basic completion
            self._log_minimal_completion()

    def _handle_tracking_error(self, error: Exception) -> None:
        """Handle errors in performance tracking gracefully."""
        # Never let performance tracking break the actual operation
        try:
            self.logger.warning(
                f"Performance tracking error: {error}",
                extra={"performance_tracking_error": str(error)}
            )
        except Exception:
            # Ultimate fallback - do nothing
            pass
```

### 6.3 Validation and Configuration Errors

```python
def validate_performance_decorator_config(**kwargs) -> Dict[str, Any]:
    """Validate decorator configuration with helpful error messages."""
    errors = []

    # Validate thresholds
    if 'threshold_warning' in kwargs and kwargs['threshold_warning'] < 0:
        errors.append("threshold_warning must be non-negative")

    if ('threshold_warning' in kwargs and 'threshold_critical' in kwargs and
        kwargs['threshold_warning'] > kwargs['threshold_critical']):
        errors.append("threshold_warning must be less than threshold_critical")

    if errors:
        raise ValueError(f"Performance decorator configuration errors: {'; '.join(errors)}")

    return kwargs
```

## 7. Integration Points Summary

### 7.1 Existing System Integration

- **LoggerConfig**: Extended with performance-specific configuration
- **JSONFormatter**: Enhanced to handle performance metrics in structured output
- **RequestContextManager**: Integrated for context injection of performance data
- **Context Injection**: Performance metrics automatically injected into log context
- **Logger Manager**: Factory methods for performance-enabled loggers

### 7.2 Thread Safety

- All performance tracking uses thread-local storage where needed
- Context injection maintains thread isolation
- Performance metrics collection is thread-safe
- No global state modification during tracking

### 7.3 Backwards Compatibility

- All existing functionality remains unchanged
- Performance logging is opt-in via configuration
- Decorators gracefully degrade if performance logging is disabled
- No breaking changes to existing APIs

This design provides a comprehensive, robust performance logging system that feels native to the existing codebase while adding powerful monitoring capabilities.
