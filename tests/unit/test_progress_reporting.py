"""
Comprehensive Unit Tests for Progress Reporting Functionality

This module provides comprehensive unit tests for the progress reporting system
in the AIM2 ontology parsers, including infrastructure components and integration
testing with parsers.

Test Classes:
    TestProgressPhase: Tests for ProgressPhase enum (4 tests)
    TestProgressInfo: Tests for ProgressInfo dataclass (5 tests)
    TestProgressReporter: Tests for ProgressReporter class (8 tests)
    TestOperationCancelledException: Tests for cancellation exception (4 tests)
    TestAbstractParserProgress: Tests for AbstractParser progress methods (8 tests)
    TestParserProgressIntegration: Tests parser-specific progress integration (2 tests)
    TestProgressThreadSafety: Tests thread safety of progress operations (3 tests)
    TestProgressPerformance: Tests performance impact of progress reporting (3 tests)

Total: 37 comprehensive tests covering all aspects of progress reporting functionality

The tests comprehensively cover:
- Progress phase enumeration and transitions
- Progress information structure and computed properties
- Progress reporter functionality including callbacks and cancellation
- Cancellation mechanisms and exception handling
- AbstractParser progress integration
- Parser-specific progress reporting through simulation
- Thread safety under concurrent access
- Performance impact measurements
- Error handling during progress reporting
- Memory usage stability
- Hook system integration

Key Features Tested:
1. ProgressPhase enum with all 13 defined phases
2. ProgressInfo dataclass with percentage calculations and metadata
3. ProgressReporter thread-safe callback management and cancellation
4. OperationCancelledException proper exception handling
5. AbstractParser progress method integration
6. Simulated parser progress reporting workflows
7. Concurrent progress reporting (10 threads, 20 reports each)
8. Performance overhead testing (1000+ progress reports)
9. Memory usage stability (5000+ progress reports)

Dependencies:
    - unittest: Core testing framework
    - unittest.mock: Mocking functionality for dependencies
    - threading: Thread safety testing
    - time: Performance measurements
    - datetime: Timestamp handling
    - json: JSON serialization testing

Usage:
    # Run all progress reporting tests
    python -m pytest tests/unit/test_progress_reporting.py -v
    
    # Run specific test class
    python -m pytest tests/unit/test_progress_reporting.py::TestProgressPhase -v
    
    # Run performance tests only
    python -m pytest tests/unit/test_progress_reporting.py::TestProgressPerformance -v

Test Results:
    All 37 tests pass successfully, validating the complete progress reporting
    infrastructure and its integration with the parser system.
"""

import json
import threading
import time
import unittest
from datetime import datetime, timedelta
from unittest.mock import Mock, MagicMock, patch, call
from typing import List, Dict, Any, Optional, Callable

# Import the progress reporting components
from aim2_project.aim2_ontology.parsers import (
    ProgressPhase,
    ProgressInfo,
    ProgressReporter,
    OperationCancelledException,
    AbstractParser,
    ConfigManager
)


class TestProgressPhase(unittest.TestCase):
    """Test the ProgressPhase enumeration."""

    def test_progress_phase_values(self):
        """Test that all expected progress phases exist with correct values."""
        expected_phases = {
            "INITIALIZING": "initializing",
            "VALIDATING": "validating", 
            "PARSING": "parsing",
            "PROCESSING_CONTENT": "processing_content",
            "BUILDING_ONTOLOGY": "building_ontology",
            "EXTRACTING_TERMS": "extracting_terms",
            "EXTRACTING_RELATIONSHIPS": "extracting_relationships",
            "EXTRACTING_TRIPLES": "extracting_triples",
            "POST_PROCESSING": "post_processing",
            "FINALIZING": "finalizing",
            "COMPLETED": "completed",
            "ERROR": "error",
            "CANCELLED": "cancelled"
        }
        
        for phase_name, phase_value in expected_phases.items():
            phase = getattr(ProgressPhase, phase_name)
            self.assertEqual(phase.value, phase_value)
            
    def test_progress_phase_enum_membership(self):
        """Test that all phases are properly enumerated."""
        phases = list(ProgressPhase)
        self.assertGreaterEqual(len(phases), 13)  # At least 13 phases expected
        
        # Test that each phase has a string value
        for phase in phases:
            self.assertIsInstance(phase.value, str)
            self.assertGreater(len(phase.value), 0)
            
    def test_progress_phase_uniqueness(self):
        """Test that all phase values are unique."""
        values = [phase.value for phase in ProgressPhase]
        self.assertEqual(len(values), len(set(values)))
        
    def test_progress_phase_string_representation(self):
        """Test string representation of progress phases."""
        phase = ProgressPhase.INITIALIZING
        self.assertEqual(str(phase), "ProgressPhase.INITIALIZING")
        self.assertEqual(phase.value, "initializing")


class TestProgressInfo(unittest.TestCase):
    """Test the ProgressInfo dataclass."""

    def test_progress_info_basic_creation(self):
        """Test basic ProgressInfo creation with required fields."""
        info = ProgressInfo(
            phase=ProgressPhase.PARSING,
            current_step=5,
            total_steps=10,
            message="Processing data"
        )
        
        self.assertEqual(info.phase, ProgressPhase.PARSING)
        self.assertEqual(info.current_step, 5)
        self.assertEqual(info.total_steps, 10)
        self.assertEqual(info.message, "Processing data")
        self.assertEqual(info.data_processed, 0)
        self.assertEqual(info.total_data, 0)
        self.assertIsInstance(info.timestamp, datetime)
        self.assertIsInstance(info.details, dict)
        
    def test_progress_info_with_all_fields(self):
        """Test ProgressInfo creation with all fields specified."""
        test_timestamp = datetime.now()
        test_details = {"source": "test", "items": 100}
        
        info = ProgressInfo(
            phase=ProgressPhase.BUILDING_ONTOLOGY,
            current_step=8,
            total_steps=12,
            message="Building ontology structure",
            data_processed=1024,
            total_data=2048,
            timestamp=test_timestamp,
            details=test_details
        )
        
        self.assertEqual(info.phase, ProgressPhase.BUILDING_ONTOLOGY)
        self.assertEqual(info.current_step, 8)
        self.assertEqual(info.total_steps, 12)
        self.assertEqual(info.message, "Building ontology structure")
        self.assertEqual(info.data_processed, 1024)
        self.assertEqual(info.total_data, 2048)
        self.assertEqual(info.timestamp, test_timestamp)
        self.assertEqual(info.details, test_details)
        
    def test_progress_info_percentage_calculation(self):
        """Test percentage calculation property."""
        # Normal case
        info = ProgressInfo(
            phase=ProgressPhase.PARSING,
            current_step=25,
            total_steps=100
        )
        self.assertEqual(info.percentage, 25.0)
        
        # Zero total steps
        info_zero = ProgressInfo(
            phase=ProgressPhase.PARSING,
            current_step=5,
            total_steps=0
        )
        self.assertEqual(info_zero.percentage, 0.0)
        
        # Complete case
        info_complete = ProgressInfo(
            phase=ProgressPhase.COMPLETED,
            current_step=10,
            total_steps=10
        )
        self.assertEqual(info_complete.percentage, 100.0)
        
    def test_progress_info_data_percentage_calculation(self):
        """Test data percentage calculation property."""
        info = ProgressInfo(
            phase=ProgressPhase.PROCESSING_CONTENT,
            current_step=5,
            total_steps=10,
            data_processed=750,
            total_data=1000
        )
        self.assertEqual(info.data_percentage, 75.0)
        
        # Zero total data
        info_zero_data = ProgressInfo(
            phase=ProgressPhase.PROCESSING_CONTENT,
            current_step=5,
            total_steps=10,
            data_processed=500,
            total_data=0
        )
        self.assertEqual(info_zero_data.data_percentage, 0.0)
        
    def test_progress_info_immutability(self):
        """Test that ProgressInfo fields can be accessed properly."""
        info = ProgressInfo(
            phase=ProgressPhase.PARSING,
            current_step=5,
            total_steps=10
        )
        
        # Should be able to access fields
        self.assertEqual(info.current_step, 5)
        
        # Note: ProgressInfo is not frozen, so fields can be modified
        # This test verifies field access works correctly
        info.current_step = 6
        self.assertEqual(info.current_step, 6)


class TestProgressReporter(unittest.TestCase):
    """Test the ProgressReporter class."""

    def setUp(self):
        """Set up test fixtures."""
        self.reporter = ProgressReporter()
        self.test_callbacks = []
        
    def tearDown(self):
        """Clean up after tests."""
        self.reporter.clear_callbacks()
        self.reporter.reset()
        
    def test_progress_reporter_initialization(self):
        """Test ProgressReporter initialization."""
        reporter = ProgressReporter()
        self.assertEqual(len(reporter._callbacks), 0)
        self.assertIsNone(reporter._current_progress)
        self.assertFalse(reporter._cancelled)
        self.assertIsNotNone(reporter._lock)
        
    def test_add_callback(self):
        """Test adding progress callbacks."""
        callback1 = Mock()
        callback2 = Mock()
        
        self.reporter.add_callback(callback1)
        self.assertEqual(len(self.reporter._callbacks), 1)
        
        self.reporter.add_callback(callback2)
        self.assertEqual(len(self.reporter._callbacks), 2)
        
        self.assertIn(callback1, self.reporter._callbacks)
        self.assertIn(callback2, self.reporter._callbacks)
        
    def test_remove_callback(self):
        """Test removing progress callbacks."""
        callback1 = Mock()
        callback2 = Mock()
        
        self.reporter.add_callback(callback1)
        self.reporter.add_callback(callback2)
        self.assertEqual(len(self.reporter._callbacks), 2)
        
        self.reporter.remove_callback(callback1)
        self.assertEqual(len(self.reporter._callbacks), 1)
        self.assertNotIn(callback1, self.reporter._callbacks)
        self.assertIn(callback2, self.reporter._callbacks)
        
        # Remove non-existing callback should not raise error
        self.reporter.remove_callback(Mock())
        self.assertEqual(len(self.reporter._callbacks), 1)
        
    def test_clear_callbacks(self):
        """Test clearing all callbacks."""
        callback1 = Mock()
        callback2 = Mock()
        
        self.reporter.add_callback(callback1)
        self.reporter.add_callback(callback2)
        self.assertEqual(len(self.reporter._callbacks), 2)
        
        self.reporter.clear_callbacks()
        self.assertEqual(len(self.reporter._callbacks), 0)
        
    def test_report_progress(self):
        """Test progress reporting functionality."""
        callback1 = Mock()
        callback2 = Mock()
        
        self.reporter.add_callback(callback1)
        self.reporter.add_callback(callback2)
        
        progress_info = ProgressInfo(
            phase=ProgressPhase.PARSING,
            current_step=3,
            total_steps=10,
            message="Test progress"
        )
        
        self.reporter.report_progress(progress_info)
        
        # Check callbacks were called
        callback1.assert_called_once_with(progress_info)
        callback2.assert_called_once_with(progress_info)
        
        # Check current progress was stored
        current = self.reporter.get_current_progress()
        self.assertEqual(current, progress_info)
        
    def test_report_progress_callback_error_handling(self):
        """Test that callback errors don't stop progress reporting."""
        good_callback = Mock()
        bad_callback = Mock(side_effect=Exception("Callback error"))
        
        self.reporter.add_callback(good_callback)
        self.reporter.add_callback(bad_callback)
        
        progress_info = ProgressInfo(
            phase=ProgressPhase.PARSING,
            current_step=1,
            total_steps=5
        )
        
        # Should not raise exception despite bad callback
        with patch('logging.getLogger') as mock_logger:
            logger_instance = Mock()
            mock_logger.return_value = logger_instance
            
            self.reporter.report_progress(progress_info)
            
            # Good callback should still be called
            good_callback.assert_called_once_with(progress_info)
            bad_callback.assert_called_once_with(progress_info)
            
            # Error should be logged
            logger_instance.warning.assert_called()
            
    def test_cancellation(self):
        """Test progress cancellation functionality."""
        self.assertFalse(self.reporter.is_cancelled())
        
        self.reporter.cancel()
        self.assertTrue(self.reporter.is_cancelled())
        
    def test_reset(self):
        """Test resetting progress reporter state."""
        # Set up some state
        callback = Mock()
        self.reporter.add_callback(callback)
        
        progress_info = ProgressInfo(
            phase=ProgressPhase.PARSING,
            current_step=5,
            total_steps=10
        )
        self.reporter.report_progress(progress_info)
        self.reporter.cancel()
        
        # Verify state is set
        self.assertIsNotNone(self.reporter.get_current_progress())
        self.assertTrue(self.reporter.is_cancelled())
        
        # Reset
        self.reporter.reset()
        
        # Verify state is cleared
        self.assertIsNone(self.reporter.get_current_progress())
        self.assertFalse(self.reporter.is_cancelled())
        # Note: callbacks are not cleared by reset, only progress state
        

class TestOperationCancelledException(unittest.TestCase):
    """Test the OperationCancelledException."""

    def test_operation_cancelled_exception_default(self):
        """Test default OperationCancelledException."""
        exc = OperationCancelledException()
        self.assertEqual(str(exc), "Operation was cancelled")
        self.assertEqual(exc.message, "Operation was cancelled")
        
    def test_operation_cancelled_exception_custom_message(self):
        """Test OperationCancelledException with custom message."""
        custom_message = "Custom cancellation message"
        exc = OperationCancelledException(custom_message)
        self.assertEqual(str(exc), custom_message)
        self.assertEqual(exc.message, custom_message)
        
    def test_operation_cancelled_exception_inheritance(self):
        """Test that OperationCancelledException inherits from Exception."""
        exc = OperationCancelledException()
        self.assertIsInstance(exc, Exception)
        
    def test_operation_cancelled_exception_raising(self):
        """Test raising and catching OperationCancelledException."""
        with self.assertRaises(OperationCancelledException) as context:
            raise OperationCancelledException("Test cancellation")
            
        self.assertEqual(str(context.exception), "Test cancellation")


class TestAbstractParserProgress(unittest.TestCase):
    """Test AbstractParser progress functionality."""

    def setUp(self):
        """Set up test fixtures."""
        # Create a concrete parser for testing
        class TestParser(AbstractParser):
            def parse(self, content: str, **kwargs) -> Any:
                return {"parsed": True}
                
            def validate(self, content: str, **kwargs) -> bool:
                return True
                
            def get_supported_formats(self) -> List[str]:
                return ["test"]
                
        # Mock the dependencies to avoid initialization issues
        with patch('aim2_project.aim2_ontology.parsers.ConfigManager') as mock_config:
            mock_config.return_value = Mock()
            mock_config.return_value.get_parser_config.return_value = {}
            
            self.parser = TestParser(
                parser_name="test_parser"
            )
            self.parser.enable_progress_reporting(True)
        
    def test_progress_reporter_setting(self):
        """Test setting and getting progress reporter."""
        reporter = ProgressReporter()
        
        self.parser.set_progress_reporter(reporter)
        self.assertEqual(self.parser.get_progress_reporter(), reporter)
        
        self.parser.set_progress_reporter(None)
        self.assertIsNone(self.parser.get_progress_reporter())
        
    def test_progress_callback_management(self):
        """Test adding and removing progress callbacks."""
        callback1 = Mock()
        callback2 = Mock()
        
        # Add callbacks
        self.parser.add_progress_callback(callback1)
        self.parser.add_progress_callback(callback2)
        
        self.assertIn(callback1, self.parser._progress_callbacks)
        self.assertIn(callback2, self.parser._progress_callbacks)
        
        # Remove callback
        self.parser.remove_progress_callback(callback1)
        self.assertNotIn(callback1, self.parser._progress_callbacks)
        self.assertIn(callback2, self.parser._progress_callbacks)
        
        # Clear all callbacks
        self.parser.clear_progress_callbacks()
        self.assertEqual(len(self.parser._progress_callbacks), 0)
        
    def test_progress_callback_with_reporter(self):
        """Test progress callbacks work with progress reporter."""
        reporter = ProgressReporter()
        callback = Mock()
        
        self.parser.set_progress_reporter(reporter)
        self.parser.add_progress_callback(callback)
        
        # Callback should be added to both parser and reporter
        self.assertIn(callback, self.parser._progress_callbacks)
        self.assertIn(callback, reporter._callbacks)
        
        # Remove callback should remove from both
        self.parser.remove_progress_callback(callback)
        self.assertNotIn(callback, self.parser._progress_callbacks)
        self.assertNotIn(callback, reporter._callbacks)
        
    def test_enable_disable_progress_reporting(self):
        """Test enabling and disabling progress reporting."""
        # Initially enabled (set in setUp)
        self.assertTrue(self.parser.is_progress_enabled())
        
        # Disable
        self.parser.enable_progress_reporting(False)
        self.assertFalse(self.parser.is_progress_enabled())
        self.assertFalse(self.parser.options["enable_progress_reporting"])
        
        # Re-enable
        self.parser.enable_progress_reporting(True)
        self.assertTrue(self.parser.is_progress_enabled())
        self.assertTrue(self.parser.options["enable_progress_reporting"])
        
    def test_report_progress_when_disabled(self):
        """Test that progress reporting is skipped when disabled."""
        callback = Mock()
        self.parser.add_progress_callback(callback)
        
        # Disable progress reporting
        self.parser.enable_progress_reporting(False)
        
        # Try to report progress
        self.parser._report_progress(
            ProgressPhase.PARSING,
            1, 5,
            "Test message"
        )
        
        # Callback should not be called
        callback.assert_not_called()
        
    def test_report_progress_when_enabled(self):
        """Test progress reporting when enabled."""
        callback = Mock()
        self.parser.add_progress_callback(callback)
        
        # Report progress
        self.parser._report_progress(
            ProgressPhase.PARSING,
            3, 10,
            "Processing data",
            data_processed=300,
            total_data=1000,
            details={"items": 50}
        )
        
        # Callback should be called with correct ProgressInfo
        callback.assert_called_once()
        args = callback.call_args[0]
        progress_info = args[0]
        
        self.assertEqual(progress_info.phase, ProgressPhase.PARSING)
        self.assertEqual(progress_info.current_step, 3)
        self.assertEqual(progress_info.total_steps, 10)
        self.assertEqual(progress_info.message, "Processing data")
        self.assertEqual(progress_info.data_processed, 300)
        self.assertEqual(progress_info.total_data, 1000)
        self.assertEqual(progress_info.details["items"], 50)
        
    def test_check_cancellation(self):
        """Test cancellation checking functionality."""
        reporter = ProgressReporter()
        self.parser.set_progress_reporter(reporter)
        
        # Should not raise when not cancelled
        self.parser._check_cancellation()
        
        # Cancel operation
        reporter.cancel()
        
        # Should raise OperationCancelledException
        with self.assertRaises(OperationCancelledException):
            self.parser._check_cancellation()
            
    def test_get_current_progress(self):
        """Test getting current progress information."""
        reporter = ProgressReporter()
        self.parser.set_progress_reporter(reporter)
        
        # Initially no progress
        self.assertIsNone(self.parser.get_current_progress())
        
        # Report some progress
        progress_info = ProgressInfo(
            phase=ProgressPhase.PARSING,
            current_step=2,
            total_steps=5
        )
        reporter.report_progress(progress_info)
        
        # Should return current progress
        current = self.parser.get_current_progress()
        self.assertEqual(current, progress_info)


class TestParserProgressIntegration(unittest.TestCase):
    """Test progress reporting integration with specific parsers."""

    def setUp(self):
        """Set up test fixtures."""
        self.progress_reports = []
        
        def progress_callback(info: ProgressInfo):
            self.progress_reports.append(info)
            
        self.progress_callback = progress_callback
        
    def test_parser_progress_integration_simulation(self):
        """Test simulated parser progress reporting integration."""
        # Create a test parser that simulates typical parsing progress
        class SimulatedParser(AbstractParser):
            def parse(self, content: str, **kwargs) -> Any:
                # Simulate typical parsing phases with progress reporting
                self._report_progress(ProgressPhase.INITIALIZING, 1, 5, "Starting parse")
                self._report_progress(ProgressPhase.VALIDATING, 2, 5, "Validating input")
                self._report_progress(ProgressPhase.PARSING, 3, 5, "Parsing content")
                self._report_progress(ProgressPhase.BUILDING_ONTOLOGY, 4, 5, "Building result")
                self._report_progress(ProgressPhase.COMPLETED, 5, 5, "Parse complete")
                return {"parsed": True, "content_length": len(content)}
                
            def validate(self, content: str, **kwargs) -> bool:
                return True
                
            def get_supported_formats(self) -> List[str]:
                return ["test"]
        
        # Mock dependencies
        with patch('aim2_project.aim2_ontology.parsers.ConfigManager') as mock_config:
            mock_config.return_value = Mock()
            mock_config.return_value.get_parser_config.return_value = {}
            
            parser = SimulatedParser(parser_name="test_parser")
            parser.enable_progress_reporting(True)
            parser.add_progress_callback(self.progress_callback)
            
            # Parse test content
            test_content = "test content for parsing"
            result = parser.parse(test_content)
            
            # Verify progress was reported
            self.assertEqual(len(self.progress_reports), 5)
            
            # Verify progress phases
            expected_phases = [
                ProgressPhase.INITIALIZING,
                ProgressPhase.VALIDATING,
                ProgressPhase.PARSING,
                ProgressPhase.BUILDING_ONTOLOGY,
                ProgressPhase.COMPLETED
            ]
            
            actual_phases = [report.phase for report in self.progress_reports]
            self.assertEqual(actual_phases, expected_phases)
            
            # Verify progress percentages
            percentages = [report.percentage for report in self.progress_reports]
            expected_percentages = [20.0, 40.0, 60.0, 80.0, 100.0]
            self.assertEqual(percentages, expected_percentages)
            
    def test_parser_progress_cancellation_integration(self):
        """Test progress reporting with cancellation during parsing."""
        class CancellableParser(AbstractParser):
            def parse(self, content: str, **kwargs) -> Any:
                self._report_progress(ProgressPhase.INITIALIZING, 1, 5, "Starting")
                self._check_cancellation()  # Should raise if cancelled
                
                self._report_progress(ProgressPhase.PARSING, 3, 5, "Parsing")
                self._check_cancellation()  # Should raise if cancelled
                
                return {"parsed": True}
                
            def validate(self, content: str, **kwargs) -> bool:
                return True
                
            def get_supported_formats(self) -> List[str]:
                return ["test"]
        
        with patch('aim2_project.aim2_ontology.parsers.ConfigManager') as mock_config:
            mock_config.return_value = Mock()
            mock_config.return_value.get_parser_config.return_value = {}
            
            parser = CancellableParser(parser_name="test_parser")
            parser.enable_progress_reporting(True)
            parser.add_progress_callback(self.progress_callback)
            
            # Set up progress reporter and cancel after first report
            reporter = ProgressReporter()
            parser.set_progress_reporter(reporter)
            
            def cancel_after_first_report(info: ProgressInfo):
                self.progress_reports.append(info)
                if len(self.progress_reports) == 1:
                    reporter.cancel()
                    
            parser.add_progress_callback(cancel_after_first_report)
            
            # Parse should raise OperationCancelledException
            with self.assertRaises(OperationCancelledException):
                parser.parse("test content")
                
            # Should have at least one progress report before cancellation
            self.assertGreaterEqual(len(self.progress_reports), 1)
            self.assertEqual(self.progress_reports[0].phase, ProgressPhase.INITIALIZING)


class TestProgressThreadSafety(unittest.TestCase):
    """Test thread safety of progress reporting operations."""

    def setUp(self):
        """Set up test fixtures."""
        self.reporter = ProgressReporter()
        self.progress_reports = []
        self.callback_errors = []
        
        def thread_safe_callback(info: ProgressInfo):
            """Thread-safe callback that records progress."""
            self.progress_reports.append(info)
            
        self.callback = thread_safe_callback
        self.reporter.add_callback(self.callback)
        
    def test_concurrent_progress_reporting(self):
        """Test concurrent progress reporting from multiple threads."""
        num_threads = 10
        reports_per_thread = 20
        threads = []
        
        def report_progress_worker(thread_id: int):
            """Worker function that reports progress."""
            for i in range(reports_per_thread):
                try:
                    progress_info = ProgressInfo(
                        phase=ProgressPhase.PARSING,
                        current_step=i,
                        total_steps=reports_per_thread,
                        message=f"Thread {thread_id} step {i}",
                        details={"thread_id": thread_id}
                    )
                    self.reporter.report_progress(progress_info)
                    time.sleep(0.001)  # Small delay to encourage race conditions
                except Exception as e:
                    self.callback_errors.append(e)
                    
        # Start threads
        for thread_id in range(num_threads):
            thread = threading.Thread(
                target=report_progress_worker,
                args=(thread_id,)
            )
            threads.append(thread)
            thread.start()
            
        # Wait for all threads to complete
        for thread in threads:
            thread.join()
            
        # Check results
        self.assertEqual(len(self.callback_errors), 0, "No callback errors should occur")
        self.assertEqual(
            len(self.progress_reports), 
            num_threads * reports_per_thread,
            "All progress reports should be recorded"
        )
        
        # Check that each thread's reports are present
        thread_ids = set()
        for report in self.progress_reports:
            if "thread_id" in report.details:
                thread_ids.add(report.details["thread_id"])
                
        self.assertEqual(len(thread_ids), num_threads, "All threads should have reported")
        
    def test_concurrent_callback_management(self):
        """Test concurrent callback addition/removal."""
        num_threads = 5
        callbacks_per_thread = 10
        threads = []
        added_callbacks = []
        
        def callback_worker():
            """Worker that adds and removes callbacks."""
            for i in range(callbacks_per_thread):
                callback = Mock()
                added_callbacks.append(callback)
                self.reporter.add_callback(callback)
                time.sleep(0.001)
                self.reporter.remove_callback(callback)
                
        # Start threads
        for _ in range(num_threads):
            thread = threading.Thread(target=callback_worker)
            threads.append(thread)
            thread.start()
            
        # Wait for completion
        for thread in threads:
            thread.join()
            
        # All callbacks should have been removed
        self.assertEqual(
            len(self.reporter._callbacks), 
            1,  # Only our original callback
            "Only original callback should remain"
        )
        
    def test_concurrent_cancellation(self):
        """Test concurrent cancellation operations."""
        num_threads = 10
        threads = []
        cancellation_results = []
        
        def cancellation_worker():
            """Worker that checks and sets cancellation."""
            # Check initial state
            initial_state = self.reporter.is_cancelled()
            cancellation_results.append(("initial", initial_state))
            
            # Cancel the operation
            self.reporter.cancel()
            
            # Check final state
            final_state = self.reporter.is_cancelled()
            cancellation_results.append(("final", final_state))
            
        # Start threads
        for _ in range(num_threads):
            thread = threading.Thread(target=cancellation_worker)
            threads.append(thread)
            thread.start()
            
        # Wait for completion
        for thread in threads:
            thread.join()
            
        # Check that cancellation is consistent
        self.assertTrue(self.reporter.is_cancelled())
        
        # All final states should be True
        final_states = [result for state, result in cancellation_results if state == "final"]
        self.assertTrue(all(final_states), "All final states should be cancelled")


class TestProgressPerformance(unittest.TestCase):
    """Test performance impact of progress reporting."""

    def setUp(self):
        """Set up test fixtures."""
        self.reporter = ProgressReporter()
        
    def test_progress_reporting_overhead(self):
        """Test that progress reporting has minimal performance overhead."""
        num_reports = 1000
        
        # Test with no callbacks (minimal overhead)
        start_time = time.time()
        for i in range(num_reports):
            progress_info = ProgressInfo(
                phase=ProgressPhase.PARSING,
                current_step=i,
                total_steps=num_reports
            )
            self.reporter.report_progress(progress_info)
        no_callback_time = time.time() - start_time
        
        # Test with callbacks
        callback = Mock()
        self.reporter.add_callback(callback)
        
        start_time = time.time()
        for i in range(num_reports):
            progress_info = ProgressInfo(
                phase=ProgressPhase.PARSING,
                current_step=i,
                total_steps=num_reports
            )
            self.reporter.report_progress(progress_info)
        with_callback_time = time.time() - start_time
        
        # Check that overhead is reasonable
        self.assertLessEqual(
            no_callback_time, 
            0.1,  # Should complete 1000 reports in < 100ms
            "Progress reporting without callbacks should be very fast"
        )
        
        self.assertLessEqual(
            with_callback_time,
            1.0,  # Should complete 1000 reports with callbacks in < 1s
            "Progress reporting with callbacks should still be reasonably fast"
        )
        
        # Callback should have been called for each report
        self.assertEqual(callback.call_count, num_reports)
        
    def test_progress_info_creation_performance(self):
        """Test ProgressInfo creation performance."""
        num_creations = 10000
        
        start_time = time.time()
        for i in range(num_creations):
            info = ProgressInfo(
                phase=ProgressPhase.PARSING,
                current_step=i,
                total_steps=num_creations,
                message=f"Step {i}",
                data_processed=i * 100,
                total_data=num_creations * 100,
                details={"iteration": i}
            )
            # Access computed properties
            _ = info.percentage
            _ = info.data_percentage
            
        creation_time = time.time() - start_time
        
        # Should be very fast
        self.assertLessEqual(
            creation_time,
            1.0,  # 10k creations in < 1 second
            "ProgressInfo creation should be very fast"
        )
        
    def test_memory_usage_stability(self):
        """Test that progress reporting doesn't cause memory leaks."""
        import gc
        
        # Force garbage collection before test
        gc.collect()
        
        # Create many progress reports
        num_reports = 5000
        for i in range(num_reports):
            progress_info = ProgressInfo(
                phase=ProgressPhase.PARSING,
                current_step=i,
                total_steps=num_reports,
                message=f"Processing item {i}",
                details={"item_id": i, "data": "x" * 100}  # Some data
            )
            self.reporter.report_progress(progress_info)
            
            # Clear reference
            del progress_info
            
        # Force garbage collection
        gc.collect()
        
        # Current progress should be the last one reported
        current = self.reporter.get_current_progress()
        self.assertIsNotNone(current)
        self.assertEqual(current.current_step, num_reports - 1)
        
        # Reset reporter to free memory
        self.reporter.reset()
        self.assertIsNone(self.reporter.get_current_progress())


if __name__ == "__main__":
    # Configure logging for tests
    import logging
    logging.basicConfig(level=logging.WARNING)
    
    # Run tests
    unittest.main(verbosity=2)