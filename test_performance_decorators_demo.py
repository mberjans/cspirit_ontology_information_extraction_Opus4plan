#!/usr/bin/env python3
"""
Comprehensive Performance Decorators Demo for AIM2 Project

This educational demonstration script showcases all capabilities of the AIM2
performance logging decorators, including:

• Basic @performance_logging decorator usage
• Async function decoration with @async_performance_logging
• Class method decoration with @method_performance_logging
• Threshold-based warning and critical logging
• Performance profile usage (fast/normal/slow)
• Manual tracking with context managers
• Error handling scenarios and strategies
• Memory tracking capabilities (when psutil available)
• Context injection integration
• JSON formatted output examples
• Configuration options and customization

Each demonstration includes:
- Clear explanations of what's being tested
- Expected behavior and output descriptions
- Realistic timing scenarios
- Threshold violation examples
- Integration with existing logging features

Usage:
    python test_performance_decorators_demo.py

Expected Output:
- JSON-formatted log messages with performance metrics
- Threshold warnings and critical alerts
- Memory usage tracking (if psutil available)
- Context injection with performance data
- Error handling demonstrations

Requirements:
- All AIM2 logging infrastructure components
- Optional: psutil for enhanced memory tracking
"""

import asyncio
import gc
import time
from typing import Dict, Any, List

# Import AIM2 components
from aim2_project.aim2_utils.performance_decorators import (
    performance_logging,
    async_performance_logging,
    method_performance_logging,
    conditional_performance_logging,
    performance_profile,
    track_performance,
    PerformanceConfig,
    set_performance_config,
    get_performance_config,
)
from aim2_project.aim2_utils.logger_manager import LoggerManager
from aim2_project.aim2_utils.logger_config import LoggerConfig
from aim2_project.aim2_utils.request_context import set_request_context, request_context


def print_section_header(title: str, subtitle: str = None):
    """Print a formatted section header for the demo."""
    print(f"\n{'=' * 80}")
    print(f"🎯 {title}")
    if subtitle:
        print(f"   {subtitle}")
    print(f"{'=' * 80}")


def print_subsection(title: str):
    """Print a formatted subsection header."""
    print(f"\n📋 {title}")
    print("-" * 60)


def print_explanation(text: str):
    """Print formatted explanation text."""
    print(f"💡 {text}")


def print_expected_behavior(behavior: str):
    """Print expected behavior description."""
    print(f"🔮 Expected: {behavior}")


def print_result(result_text: str):
    """Print result with formatting."""
    print(f"✅ Result: {result_text}")


def check_psutil_availability():
    """Check if psutil is available for memory tracking."""
    try:
        pass

        print("✅ psutil is available - enhanced memory tracking enabled")
        return True
    except ImportError:
        print("⚠️  psutil not available - basic memory tracking only")
        return False


def setup_demo_logging():
    """Set up comprehensive logging configuration for the demo."""
    print_section_header(
        "LOGGING SETUP", "Configuring performance logging with JSON output"
    )

    print_explanation("Setting up LoggerConfig with performance features enabled")

    # Create logger configuration with all performance features enabled
    config = LoggerConfig()
    config.update_config(
        {
            "level": "INFO",
            "formatter_type": "json",
            "handlers": ["console"],
            "enable_performance_logging": True,
            "enable_context_injection": True,
            "context_injection": {
                "enabled": True,
                "include_timestamp": True,
                "include_thread_info": True,
            },
            "performance_thresholds": {"warning_seconds": 0.3, "critical_seconds": 1.0},
            "performance_metrics": {
                "include_memory_usage": True,
                "include_function_args": True,
                "include_function_result": True,
                "max_arg_length": 200,
                "max_result_length": 200,
            },
        }
    )

    # Initialize logger manager
    logger_manager = LoggerManager(config)
    logger_manager.initialize()

    # Configure global performance settings
    perf_config = PerformanceConfig(
        enabled=True,
        warning_threshold=0.3,  # 300ms warning threshold
        critical_threshold=1.0,  # 1 second critical threshold
        include_memory=True,
        include_cpu=False,  # CPU tracking requires psutil
        include_args=True,
        include_result=True,
        max_arg_length=200,
        max_result_length=200,
        context_prefix="perf_",
        auto_operation_naming=True,
    )
    set_performance_config(perf_config)

    print_result(
        "Logging configuration completed with JSON output and performance tracking"
    )
    return logger_manager


# ============================================================================
# DEMONSTRATION FUNCTIONS - BASIC DECORATOR USAGE
# ============================================================================


@performance_logging(
    operation_name="lightning_fast_operation",
    threshold_warning=0.1,
    threshold_critical=0.5,
    include_args=True,
    include_result=True,
)
def lightning_fast_operation(x: int, y: int) -> int:
    """Ultra-fast operation that should never exceed thresholds."""
    # Simulate very fast computation (10ms)
    time.sleep(0.01)
    result = x + y
    return result


@performance_logging(
    operation_name="warning_threshold_operation",
    threshold_warning=0.2,
    threshold_critical=1.0,
    include_args=True,
    include_result=True,
    include_memory=True,
)
def warning_threshold_operation(iterations: int) -> dict:
    """Operation designed to exceed warning threshold but not critical."""
    # Simulate processing that takes ~400ms (exceeds 200ms warning)
    time.sleep(0.4)

    # Simulate some memory allocation
    data = list(range(iterations * 1000))
    result = {
        "iterations": iterations,
        "total_items": len(data),
        "sum": sum(data[:100]) if data else 0,
    }
    return result


@performance_logging(
    operation_name="critical_threshold_operation",
    threshold_warning=0.5,
    threshold_critical=1.0,
    include_args=True,
    include_result=True,
    include_memory=True,
)
def critical_threshold_operation(complexity: str) -> dict:
    """Operation designed to exceed critical threshold."""
    if complexity == "extreme":
        # Simulate very heavy processing (1.2s - exceeds 1s critical)
        time.sleep(1.2)
    else:
        # Moderate processing
        time.sleep(0.6)

    return {"complexity": complexity, "status": "completed"}


@performance_logging(
    operation_name="error_demonstration", include_args=True, on_error="log_and_raise"
)
def error_demonstration(error_type: str, message: str = "Demo error"):
    """Function that demonstrates error handling in performance logging."""
    if error_type == "value_error":
        raise ValueError(message)
    elif error_type == "runtime_error":
        raise RuntimeError(message)
    elif error_type == "custom_error":
        raise Exception(f"Custom error: {message}")
    else:
        return f"Success: {error_type}"


# ============================================================================
# ASYNC DEMONSTRATION FUNCTIONS
# ============================================================================


@async_performance_logging(
    operation_name="fast_async_task",
    threshold_warning=0.2,
    threshold_critical=0.8,
    include_args=True,
    include_result=True,
)
async def fast_async_task(task_id: str, data_size: int) -> Dict[str, Any]:
    """Fast async operation for performance testing."""
    # Simulate I/O-bound work
    await asyncio.sleep(0.1)

    return {
        "task_id": task_id,
        "data_size": data_size,
        "processing_time": "100ms",
        "status": "completed",
    }


@async_performance_logging(
    operation_name="slow_async_task",
    threshold_warning=0.3,
    threshold_critical=1.0,
    include_args=True,
    include_result=True,
    include_memory=True,
)
async def slow_async_task(task_id: str, complexity: int = 5) -> Dict[str, Any]:
    """Slow async operation that may exceed thresholds."""
    # Simulate heavy async processing
    await asyncio.sleep(0.5)  # Will exceed warning threshold

    # Simulate some data processing
    processed_items = complexity * 100

    return {
        "task_id": task_id,
        "complexity": complexity,
        "processed_items": processed_items,
        "status": "completed",
    }


@async_performance_logging(
    operation_name="async_error_demo", include_args=True, on_error="log_and_raise"
)
async def async_error_demo(should_fail: bool = False, delay: float = 0.1):
    """Async function that demonstrates error handling."""
    await asyncio.sleep(delay)

    if should_fail:
        raise RuntimeError("Async operation failed")

    return {"status": "success", "delay": delay}


# ============================================================================
# CLASS-BASED DEMONSTRATION
# ============================================================================


class DataAnalyzer:
    """Demonstration class with various performance-monitored methods."""

    def __init__(self, name: str):
        self.name = name
        self.processed_count = 0

    @method_performance_logging(
        include_class_name=True,
        threshold_warning=0.2,
        threshold_critical=0.8,
        include_args=True,
        include_result=True,
        include_memory=True,
    )
    def analyze_dataset(
        self, dataset: List[int], analysis_type: str = "basic"
    ) -> Dict[str, Any]:
        """Analyze a dataset with performance monitoring."""
        # Simulate analysis processing time based on type
        if analysis_type == "deep":
            time.sleep(0.3)  # Will exceed warning threshold
        else:
            time.sleep(0.1)  # Should be within limits

        # Perform basic analysis
        if not dataset:
            return {"error": "empty dataset"}

        analysis_result = {
            "analysis_type": analysis_type,
            "dataset_size": len(dataset),
            "min_value": min(dataset),
            "max_value": max(dataset),
            "average": sum(dataset) / len(dataset),
            "analyzer": self.name,
        }

        self.processed_count += 1
        return analysis_result

    @performance_profile("fast", include_args=True, include_result=True)
    def quick_validation(self, data: Any) -> bool:
        """Quick validation using fast performance profile."""
        # Fast validation should complete in <100ms
        time.sleep(0.02)
        return data is not None and len(str(data)) > 0

    @performance_profile("slow", include_args=True, include_result=True)
    def comprehensive_validation(self, data: Any) -> Dict[str, Any]:
        """Comprehensive validation using slow performance profile."""
        # Slow validation can take several seconds
        time.sleep(1.5)

        return {
            "is_valid": data is not None,
            "data_type": type(data).__name__,
            "data_size": len(str(data)),
            "validation_level": "comprehensive",
        }


# ============================================================================
# CONDITIONAL AND PROFILE DEMONSTRATIONS
# ============================================================================

# Global flag for conditional monitoring
ENABLE_MONITORING = True


@conditional_performance_logging(
    condition=lambda: ENABLE_MONITORING,
    operation_name="conditional_processing",
    threshold_warning=0.15,
    include_args=True,
    include_result=True,
)
def conditional_processing(data: str, enable_detailed: bool = False) -> Dict[str, Any]:
    """Function with conditional performance monitoring."""
    processing_time = 0.2 if enable_detailed else 0.05
    time.sleep(processing_time)

    return {
        "data_length": len(data),
        "detailed_processing": enable_detailed,
        "processing_time": f"{processing_time}s",
    }


@performance_profile("fast")
def fast_computation(x: int, y: int) -> int:
    """Fast computation using fast profile (warning: 0.1s, critical: 0.5s)."""
    time.sleep(0.02)  # Well within fast profile limits
    return x * y + x + y


@performance_profile("normal")
def normal_computation(data: List[int]) -> float:
    """Normal computation using normal profile (warning: 1.0s, critical: 5.0s)."""
    time.sleep(0.3)  # Within normal profile limits
    return sum(data) / len(data) if data else 0.0


@performance_profile("slow")
def slow_computation(complexity: int) -> Dict[str, int]:
    """Slow computation using slow profile (warning: 10.0s, critical: 30.0s)."""
    time.sleep(2.0)  # Within slow profile limits but slow

    result = {}
    for i in range(complexity):
        result[f"item_{i}"] = i**2

    return result


# ============================================================================
# TEST FUNCTIONS
# ============================================================================


def test_basic_decorators():
    """Test basic performance decorator functionality with threshold violations."""
    print_section_header(
        "BASIC PERFORMANCE DECORATORS", "Testing core decorator functionality"
    )

    with request_context(
        request_id="demo-basic-001", operation="basic_performance_test"
    ):
        print_subsection("Lightning Fast Operation")
        print_explanation("Testing operation that should never exceed thresholds")
        print_expected_behavior("No threshold warnings, execution time ~10ms")

        result = lightning_fast_operation(42, 58)
        print_result(f"Computation result: {result}")

        print_subsection("Warning Threshold Violation")
        print_explanation(
            "Testing operation designed to exceed warning (200ms) but not critical (1s)"
        )
        print_expected_behavior(
            "WARNING log level due to threshold violation, ~400ms execution"
        )

        result = warning_threshold_operation(50)
        print_result(f"Processing result: {result}")

        print_subsection("Critical Threshold Violation")
        print_explanation(
            "Testing operation designed to exceed critical threshold (1s)"
        )
        print_expected_behavior(
            "CRITICAL log level due to threshold violation, ~1.2s execution"
        )

        result = critical_threshold_operation("extreme")
        print_result(f"Complex processing result: {result}")


def test_error_handling():
    """Test error handling scenarios in performance logging."""
    print_section_header(
        "ERROR HANDLING", "Testing performance logging with exceptions"
    )

    with request_context(request_id="demo-error-001", operation="error_handling_test"):
        error_scenarios = [
            ("value_error", "Invalid input value provided"),
            ("runtime_error", "Runtime execution failed"),
            ("custom_error", "Custom business logic error"),
        ]

        for error_type, message in error_scenarios:
            print_subsection(f"Error Scenario: {error_type.replace('_', ' ').title()}")
            print_explanation(
                f"Testing {error_type} handling with message: '{message}'"
            )
            print_expected_behavior(
                "ERROR log level with exception details and performance metrics"
            )

            try:
                error_demonstration(error_type, message)
            except Exception as e:
                print_result(f"Expected {type(e).__name__} caught: {e}")

        print_subsection("Successful Operation")
        print_explanation("Testing successful operation after errors")
        result = error_demonstration("success", "All good!")
        print_result(f"Success result: {result}")


async def test_async_decorators():
    """Test async performance decorator functionality."""
    print_section_header(
        "ASYNC PERFORMANCE DECORATORS", "Testing async function monitoring"
    )

    with request_context(
        request_id="demo-async-001", operation="async_performance_test"
    ):
        print_subsection("Fast Async Task")
        print_explanation("Testing fast async operation within thresholds")
        print_expected_behavior("No warnings, ~100ms execution time")

        result = await fast_async_task("task_001", 1000)
        print_result(f"Fast task result: {result}")

        print_subsection("Slow Async Task with Warning")
        print_explanation("Testing async operation that exceeds warning threshold")
        print_expected_behavior("WARNING log level, ~500ms execution time")

        result = await slow_async_task("task_002", 10)
        print_result(f"Slow task result: {result}")

        print_subsection("Async Error Handling")
        print_explanation("Testing error handling in async functions")
        print_expected_behavior("ERROR log level with async exception details")

        try:
            await async_error_demo(should_fail=True, delay=0.1)
        except RuntimeError as e:
            print_result(f"Expected async error caught: {e}")

        # Successful async operation
        result = await async_error_demo(should_fail=False, delay=0.05)
        print_result(f"Successful async result: {result}")


def test_method_decorators():
    """Test class method performance decorators."""
    print_section_header(
        "METHOD PERFORMANCE DECORATORS", "Testing class method monitoring"
    )

    analyzer = DataAnalyzer("DemoAnalyzer-001")

    with request_context(
        request_id="demo-method-001", operation="method_performance_test"
    ):
        print_subsection("Basic Dataset Analysis")
        print_explanation("Testing method decoration with basic analysis")
        print_expected_behavior("Method name includes class name, ~100ms execution")

        dataset = list(range(1, 101))  # 1 to 100
        result = analyzer.analyze_dataset(dataset, "basic")
        print_result(f"Basic analysis: {result}")

        print_subsection("Deep Dataset Analysis (Warning Threshold)")
        print_explanation("Testing method that exceeds warning threshold")
        print_expected_behavior(
            "WARNING log level due to processing time, ~300ms execution"
        )

        result = analyzer.analyze_dataset(dataset, "deep")
        print_result(f"Deep analysis: {result}")

        print_subsection("Fast Profile Validation")
        print_explanation("Testing method with fast performance profile")
        print_expected_behavior("No warnings, very quick execution")

        is_valid = analyzer.quick_validation({"test": "data", "items": [1, 2, 3]})
        print_result(f"Quick validation: {is_valid}")

        print_subsection("Slow Profile Validation")
        print_explanation(
            "Testing method with slow performance profile (allows longer execution)"
        )
        print_expected_behavior(
            "No warnings despite 1.5s execution (within slow profile limits)"
        )

        result = analyzer.comprehensive_validation([1, 2, 3, 4, 5])
        print_result(f"Comprehensive validation: {result}")


def test_conditional_and_profiles():
    """Test conditional decorators and performance profiles."""
    print_section_header(
        "CONDITIONAL & PROFILE DECORATORS", "Testing advanced decorator features"
    )

    with request_context(
        request_id="demo-conditional-001", operation="conditional_profile_test"
    ):
        print_subsection("Conditional Performance Monitoring (Enabled)")
        print_explanation("Testing conditional decorator when monitoring is enabled")
        print_expected_behavior(
            "Performance monitoring active, possible warning due to processing time"
        )

        global ENABLE_MONITORING
        ENABLE_MONITORING = True
        result = conditional_processing(
            "test data for conditional processing", enable_detailed=True
        )
        print_result(f"Conditional result (monitored): {result}")

        print_subsection("Conditional Performance Monitoring (Disabled)")
        print_explanation("Testing conditional decorator when monitoring is disabled")
        print_expected_behavior("No performance monitoring, function executes normally")

        ENABLE_MONITORING = False
        result = conditional_processing(
            "test data without monitoring", enable_detailed=False
        )
        print_result(f"Conditional result (not monitored): {result}")

        # Re-enable for rest of demo
        ENABLE_MONITORING = True

        print_subsection("Fast Performance Profile")
        print_explanation("Testing fast profile (warning: 0.1s, critical: 0.5s)")
        print_expected_behavior("No warnings, very quick execution")

        result = fast_computation(15, 25)
        print_result(f"Fast computation: {result}")

        print_subsection("Normal Performance Profile")
        print_explanation("Testing normal profile (warning: 1.0s, critical: 5.0s)")
        print_expected_behavior("No warnings, moderate execution time")

        result = normal_computation([10, 20, 30, 40, 50])
        print_result(f"Normal computation average: {result}")

        print_subsection("Slow Performance Profile")
        print_explanation("Testing slow profile (warning: 10.0s, critical: 30.0s)")
        print_expected_behavior(
            "No warnings despite 2s execution (within slow profile)"
        )

        result = slow_computation(5)
        print_result(f"Slow computation items: {len(result)}")


def test_manual_tracking():
    """Test manual performance tracking with context managers."""
    print_section_header(
        "MANUAL PERFORMANCE TRACKING", "Testing context manager approach"
    )

    with request_context(
        request_id="demo-manual-001", operation="manual_tracking_test"
    ):
        print_subsection("Basic Manual Tracking")
        print_explanation(
            "Using track_performance context manager for manual monitoring"
        )
        print_expected_behavior(
            "Manual control over performance tracking with custom results"
        )

        with track_performance(
            "manual_database_operation", include_memory=True
        ) as tracker:
            # Simulate database query
            time.sleep(0.3)

            # Create some data to affect memory
            temp_data = list(range(10000))
            query_result = {"rows": len(temp_data), "operation": "SELECT"}

            # Set custom result for tracking
            tracker.set_result(query_result)
            print_result(f"Database operation completed: {query_result}")

        print_subsection("Manual Tracking with Error")
        print_explanation("Testing manual tracking when operations fail")
        print_expected_behavior(
            "ERROR log level with exception captured in performance metrics"
        )

        try:
            with track_performance("manual_api_call", include_memory=True) as tracker:
                time.sleep(0.1)
                # Simulate API call failure
                raise ConnectionError("API endpoint unavailable")
        except ConnectionError as e:
            print_result(f"Expected error in manual tracking: {e}")

        print_subsection("Manual Tracking with Custom Configuration")
        print_explanation("Testing manual tracking with custom thresholds")
        print_expected_behavior("Custom warning threshold (100ms) should be exceeded")

        with track_performance(
            "manual_custom_operation",
            warning_threshold=0.1,
            critical_threshold=0.5,
            include_memory=True,
            include_args=True,
        ) as tracker:
            time.sleep(0.2)  # Will exceed 100ms warning

            # Set custom arguments and result
            tracker.set_function_args((42, "test"), {"option": "custom"})
            tracker.set_result({"status": "completed", "custom": True})
            print_result("Custom manual operation completed")


def test_memory_tracking():
    """Test memory tracking capabilities."""
    print_section_header("MEMORY TRACKING", "Testing memory usage monitoring")

    has_psutil = check_psutil_availability()

    with request_context(
        request_id="demo-memory-001", operation="memory_tracking_test"
    ):
        print_subsection("Memory-Intensive Operation")
        print_explanation("Testing function that allocates significant memory")
        if has_psutil:
            print_expected_behavior("Memory delta shown in performance metrics")
        else:
            print_expected_behavior("Basic memory tracking without detailed metrics")

        @performance_logging(
            operation_name="memory_intensive_task",
            include_memory=True,
            include_args=True,
            include_result=True,
        )
        def memory_intensive_task(size: int) -> Dict[str, Any]:
            # Force garbage collection before starting
            gc.collect()

            # Allocate memory
            large_data = list(range(size))
            large_dict = {i: str(i) * 10 for i in range(size // 100)}

            # Do some processing
            time.sleep(0.1)

            result = {
                "list_size": len(large_data),
                "dict_size": len(large_dict),
                "total_allocated": size,
            }

            # Clean up
            del large_data
            del large_dict
            gc.collect()

            return result

        result = memory_intensive_task(100000)
        print_result(f"Memory task result: {result}")


def test_context_integration():
    """Test integration with request context system."""
    print_section_header(
        "CONTEXT INTEGRATION", "Testing performance + context injection"
    )

    print_subsection("Rich Context Integration")
    print_explanation("Testing performance logging with rich request context")
    print_expected_behavior(
        "Performance metrics combined with request context in JSON logs"
    )

    @performance_logging(
        operation_name="context_aware_operation",
        include_args=True,
        include_result=True,
        context_fields={
            "service": "demo_service",
            "version": "2.0.0",
            "environment": "demonstration",
        },
    )
    def context_aware_operation(
        user_data: Dict[str, Any], options: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Operation that demonstrates context integration."""
        time.sleep(0.15)  # Moderate processing time

        processed_result = {
            "user_id": user_data.get("user_id"),
            "data_processed": len(str(user_data)),
            "options_applied": len(options),
            "timestamp": time.time(),
            "status": "success",
        }

        return processed_result

    # Set rich request context
    set_request_context(
        user_id="demo_user_12345",
        session_id="session_abcdef",
        component="performance_demo",
        trace_id="trace_xyz789",
        environment="demonstration",
        service="demo_service",
    )

    user_data = {
        "user_id": "demo_user_12345",
        "name": "Demo User",
        "preferences": {"theme": "dark", "language": "en"},
    }

    options = {"enable_caching": True, "compression": "gzip", "retry_count": 3}

    result = context_aware_operation(user_data, options)
    print_result(f"Context-aware result: {result}")


def test_configuration_options():
    """Test various configuration options for performance logging."""
    print_section_header(
        "CONFIGURATION OPTIONS", "Testing different performance configurations"
    )

    print_subsection("Custom Performance Configuration")
    print_explanation("Testing custom global performance configuration")
    print_expected_behavior("Modified thresholds and settings applied globally")

    # Save current config
    original_config = get_performance_config()

    # Set custom configuration
    custom_config = PerformanceConfig(
        enabled=True,
        warning_threshold=0.05,  # Very strict 50ms warning
        critical_threshold=0.2,  # 200ms critical
        include_memory=True,
        include_args=True,
        include_result=True,
        max_arg_length=50,  # Shorter arg summaries
        max_result_length=50,  # Shorter result summaries
        context_prefix="custom_perf_",
    )
    set_performance_config(custom_config)

    @performance_logging(operation_name="custom_config_test")
    def test_custom_config(data: str) -> str:
        time.sleep(0.1)  # Will exceed 50ms warning threshold
        return f"Processed: {data[:20]}..."

    with request_context(request_id="demo-config-001", operation="configuration_test"):
        result = test_custom_config(
            "This is test data for custom configuration demonstration"
        )
        print_result(f"Custom config result: {result}")

    # Restore original configuration
    set_performance_config(original_config)
    print_result("Original configuration restored")


def print_demo_summary():
    """Print a summary of what was demonstrated."""
    print_section_header("DEMO SUMMARY", "Summary of demonstrated features")

    features = [
        "✅ Basic @performance_logging decorator with threshold monitoring",
        "✅ Async function decoration with @async_performance_logging",
        "✅ Class method decoration with @method_performance_logging",
        "✅ Threshold-based warning and critical logging levels",
        "✅ Performance profiles (fast/normal/slow) for different operation types",
        "✅ Manual tracking with track_performance context manager",
        "✅ Comprehensive error handling and exception capture",
        "✅ Memory tracking capabilities (basic and enhanced with psutil)",
        "✅ Context injection integration with request tracking",
        "✅ JSON formatted output with structured performance metrics",
        "✅ Configuration options and customization",
        "✅ Conditional monitoring based on runtime conditions",
    ]

    print("\n📊 Features Demonstrated:")
    for feature in features:
        print(f"   {feature}")

    print(f"\n💡 Key Benefits:")
    print(f"   • Non-intrusive performance monitoring")
    print(f"   • Automatic threshold-based alerting")
    print(f"   • Rich context integration for debugging")
    print(f"   • Structured JSON logging for analysis")
    print(f"   • Thread-safe operation")
    print(f"   • Flexible configuration options")
    print(f"   • Error handling without breaking functionality")

    print(f"\n🔍 Log Analysis Tips:")
    print(f"   • Look for 'performance_metrics' in JSON logs")
    print(f"   • Check 'exceeded_warning' and 'exceeded_critical' flags")
    print(f"   • Monitor 'duration_seconds' for actual timing")
    print(f"   • Use 'memory_delta_mb' for memory usage patterns")
    print(f"   • Correlate with request context fields for debugging")


async def main():
    """Main demonstration function that runs all tests."""
    print_section_header(
        "AIM2 PERFORMANCE DECORATORS", "Comprehensive Demonstration Script"
    )
    print(f"🎭 This demo showcases all performance monitoring capabilities")
    print(f"📝 Each section demonstrates different aspects with explanations")
    print(f"🔍 Watch for JSON-formatted log output with performance metrics")

    # Setup logging
    logger_manager = setup_demo_logging()

    try:
        # Run all demonstration tests
        test_basic_decorators()
        test_error_handling()
        await test_async_decorators()
        test_method_decorators()
        test_conditional_and_profiles()
        test_manual_tracking()
        test_memory_tracking()
        test_context_integration()
        test_configuration_options()

        # Print summary
        print_demo_summary()

        print_section_header("DEMO COMPLETED SUCCESSFULLY! 🎉")
        print("🔍 Review the JSON log output above to see performance metrics")
        print("📊 Each operation includes timing, threshold checks, and context data")
        print("⚠️  Warning and Critical threshold violations are highlighted")
        print("💾 Memory usage deltas are shown when available")
        print("🔗 Request context is automatically injected into performance logs")
        print("\n📖 Understanding the Output:")
        print("   • Text lines show the demo progress and results")
        print("   • JSON log entries contain the actual performance data")
        print(
            "   • Look for fields like 'duration_seconds', 'exceeded_warning', 'memory_delta_mb'"
        )
        print("   • Error scenarios show how exceptions are captured with timing data")
        print(
            "   • Context fields (request_id, user_id, etc.) are automatically included"
        )

    except Exception as e:
        print(f"\n❌ Demo failed with error: {e}")
        import traceback

        traceback.print_exc()
        raise
    finally:
        # Cleanup
        logger_manager.cleanup()


if __name__ == "__main__":
    # Run the comprehensive demo
    asyncio.run(main())
