"""
Comprehensive Unit Tests for Enhanced Rotating File Handler

This module provides comprehensive unit tests for the enhanced rotating file handler
implementation that supports size-based, time-based, and combined rotation.

Test Classes:
    TestRotatingFileHandlerError: Tests for the custom exception class
    TestEnhancedRotatingFileHandler: Tests for the main handler class
    TestRotatingFileHandlerFactory: Tests for the factory function

Dependencies:
    - unittest: For test framework
    - unittest.mock: For mocking functionality
    - tempfile: For temporary file operations
    - pathlib: For file path operations
    - time: For time-based testing
    - logging: For logging functionality
    - datetime: For time calculations
"""

import unittest
import unittest.mock
import tempfile
import shutil
import os
import logging
from datetime import datetime, timedelta

from aim2_project.aim2_utils.rotating_file_handler import (
    RotatingFileHandlerError,
    EnhancedRotatingFileHandler,
    create_rotating_file_handler,
)


class TestRotatingFileHandlerError(unittest.TestCase):
    """Test cases for RotatingFileHandlerError exception class."""

    def test_basic_error_creation(self):
        """Test basic error creation without cause."""
        message = "Test error message"
        error = RotatingFileHandlerError(message)

        self.assertEqual(str(error), message)
        self.assertEqual(error.message, message)
        self.assertIsNone(error.cause)

    def test_error_with_cause(self):
        """Test error creation with a cause exception."""
        message = "Test error message"
        cause = ValueError("Original error")
        error = RotatingFileHandlerError(message, cause)

        self.assertEqual(str(error), message)
        self.assertEqual(error.message, message)
        self.assertEqual(error.cause, cause)
        self.assertEqual(error.__cause__, cause)


class TestEnhancedRotatingFileHandler(unittest.TestCase):
    """Test cases for EnhancedRotatingFileHandler class."""

    def setUp(self):
        """Set up test fixtures before each test method."""
        self.temp_dir = tempfile.mkdtemp()
        self.test_file = os.path.join(self.temp_dir, "test.log")
        self.handler = None

    def tearDown(self):
        """Clean up after each test method."""
        if self.handler:
            try:
                self.handler.close()
            except:
                pass

        # Clean up temporary directory
        try:
            shutil.rmtree(self.temp_dir)
        except:
            pass

    def test_size_based_initialization(self):
        """Test initialization with size-based rotation."""
        self.handler = EnhancedRotatingFileHandler(
            filename=self.test_file,
            rotation_type="size",
            max_bytes=1024,
            backup_count=3,
        )

        self.assertEqual(self.handler.filename, self.test_file)
        self.assertEqual(self.handler.rotation_type, "size")
        self.assertEqual(self.handler.max_bytes, 1024)
        self.assertEqual(self.handler.backup_count, 3)
        self.assertIsNone(self.handler._rollover_time)

    def test_time_based_initialization(self):
        """Test initialization with time-based rotation."""
        self.handler = EnhancedRotatingFileHandler(
            filename=self.test_file,
            rotation_type="time",
            time_interval="daily",
            time_when="midnight",
        )

        self.assertEqual(self.handler.filename, self.test_file)
        self.assertEqual(self.handler.rotation_type, "time")
        self.assertEqual(self.handler.time_interval, "daily")
        self.assertEqual(self.handler.time_when, "midnight")
        self.assertIsNotNone(self.handler._rollover_time)

    def test_combined_rotation_initialization(self):
        """Test initialization with combined rotation."""
        self.handler = EnhancedRotatingFileHandler(
            filename=self.test_file,
            rotation_type="both",
            max_bytes=2048,
            backup_count=5,
            time_interval="hourly",
        )

        self.assertEqual(self.handler.rotation_type, "both")
        self.assertEqual(self.handler.max_bytes, 2048)
        self.assertEqual(self.handler.time_interval, "hourly")
        self.assertIsNotNone(self.handler._rollover_time)

    def test_invalid_rotation_type(self):
        """Test initialization with invalid rotation type."""
        with self.assertRaises(RotatingFileHandlerError) as cm:
            EnhancedRotatingFileHandler(
                filename=self.test_file,
                rotation_type="invalid",
            )

        self.assertIn("Invalid rotation_type", str(cm.exception))

    def test_invalid_max_bytes(self):
        """Test initialization with invalid max_bytes."""
        with self.assertRaises(RotatingFileHandlerError):
            EnhancedRotatingFileHandler(
                filename=self.test_file,
                rotation_type="size",
                max_bytes=100,  # Too small
            )

    def test_invalid_backup_count_negative(self):
        """Test initialization with negative backup count."""
        with self.assertRaises(RotatingFileHandlerError):
            EnhancedRotatingFileHandler(
                filename=self.test_file,
                backup_count=-1,
            )

    def test_invalid_backup_count_too_large(self):
        """Test initialization with too large backup count."""
        with self.assertRaises(RotatingFileHandlerError):
            EnhancedRotatingFileHandler(
                filename=self.test_file,
                backup_count=150,  # Too large
            )

    def test_invalid_time_interval(self):
        """Test initialization with invalid time interval."""
        with self.assertRaises(RotatingFileHandlerError):
            EnhancedRotatingFileHandler(
                filename=self.test_file,
                rotation_type="time",
                time_interval="invalid",
            )

    def test_invalid_time_when_format(self):
        """Test initialization with invalid time_when format."""
        with self.assertRaises(RotatingFileHandlerError):
            EnhancedRotatingFileHandler(
                filename=self.test_file,
                rotation_type="time",
                time_interval="daily",
                time_when="25:00",  # Invalid hour
            )

    def test_directory_creation(self):
        """Test that handler creates missing directories."""
        nested_file = os.path.join(self.temp_dir, "nested", "dir", "test.log")

        self.handler = EnhancedRotatingFileHandler(
            filename=nested_file,
            rotation_type="size",
        )

        self.assertTrue(os.path.exists(os.path.dirname(nested_file)))

    def test_basic_logging(self):
        """Test basic logging functionality."""
        self.handler = EnhancedRotatingFileHandler(
            filename=self.test_file,
            rotation_type="size",
            max_bytes=10240,
        )

        # Create a test record
        record = logging.LogRecord(
            name="test",
            level=logging.INFO,
            pathname="",
            lineno=0,
            msg="Test message",
            args=(),
            exc_info=None,
        )

        self.handler.emit(record)
        self.handler.flush()

        # Check that file was created and contains message
        self.assertTrue(os.path.exists(self.test_file))
        with open(self.test_file, "r") as f:
            content = f.read()
            self.assertIn("Test message", content)

    def test_size_based_rollover(self):
        """Test size-based rollover functionality."""
        # Create handler with small max size
        self.handler = EnhancedRotatingFileHandler(
            filename=self.test_file,
            rotation_type="size",
            max_bytes=1024,  # Minimum allowed size
            backup_count=2,
        )

        # Write multiple messages to trigger rollover
        for i in range(20):
            record = logging.LogRecord(
                name="test",
                level=logging.INFO,
                pathname="",
                lineno=0,
                msg=f"This is a long test message number {i:03d} that should trigger rollover when we write enough content to exceed the maximum file size limit",
                args=(),
                exc_info=None,
            )
            self.handler.emit(record)

        self.handler.flush()

        # Check that backup files were created
        backup1 = f"{self.test_file}.1"
        self.assertTrue(os.path.exists(backup1))

    def test_time_rollover_calculation_daily_midnight(self):
        """Test time rollover calculation for daily midnight rotation."""
        self.handler = EnhancedRotatingFileHandler(
            filename=self.test_file,
            rotation_type="time",
            time_interval="midnight",
        )

        # Check that rollover time is set for next midnight
        now = datetime.now()
        expected_rollover = now.replace(
            hour=0, minute=0, second=0, microsecond=0
        ) + timedelta(days=1)
        actual_rollover = datetime.fromtimestamp(self.handler._rollover_time)

        # Allow for small differences due to execution time
        self.assertAlmostEqual(
            expected_rollover.timestamp(),
            actual_rollover.timestamp(),
            delta=5,  # 5 seconds tolerance
        )

    def test_time_rollover_calculation_hourly(self):
        """Test time rollover calculation for hourly rotation."""
        self.handler = EnhancedRotatingFileHandler(
            filename=self.test_file,
            rotation_type="time",
            time_interval="hourly",
        )

        # Check that rollover time is set for next hour
        now = datetime.now()
        expected_rollover = now.replace(minute=0, second=0, microsecond=0) + timedelta(
            hours=1
        )
        actual_rollover = datetime.fromtimestamp(self.handler._rollover_time)

        # Allow for small differences due to execution time
        self.assertAlmostEqual(
            expected_rollover.timestamp(),
            actual_rollover.timestamp(),
            delta=5,  # 5 seconds tolerance
        )

    def test_time_rollover_calculation_custom_time(self):
        """Test time rollover calculation for custom daily time."""
        self.handler = EnhancedRotatingFileHandler(
            filename=self.test_file,
            rotation_type="time",
            time_interval="daily",
            time_when="14:30",  # 2:30 PM
        )

        # Check that rollover time is set correctly
        now = datetime.now()
        expected_time = now.replace(hour=14, minute=30, second=0, microsecond=0)

        # If the time has passed today, it should be tomorrow
        if expected_time.timestamp() <= now.timestamp():
            expected_time += timedelta(days=1)

        actual_rollover = datetime.fromtimestamp(self.handler._rollover_time)

        self.assertAlmostEqual(
            expected_time.timestamp(),
            actual_rollover.timestamp(),
            delta=5,  # 5 seconds tolerance
        )

    @unittest.mock.patch("time.time")
    def test_should_rollover_size(self, mock_time):
        """Test size-based rollover detection."""
        mock_time.return_value = 1000.0

        self.handler = EnhancedRotatingFileHandler(
            filename=self.test_file,
            rotation_type="size",
            max_bytes=1024,
        )

        # Write some content to the file first
        with open(self.test_file, "w") as f:
            f.write("x" * 900)  # 900 bytes

        # Create a record that would exceed the limit
        record = logging.LogRecord(
            name="test",
            level=logging.INFO,
            pathname="",
            lineno=0,
            msg="x" * 200,  # 200+ bytes when formatted
            args=(),
            exc_info=None,
        )

        # Open the file handler
        self.handler._open()

        # Should trigger rollover due to size
        should_rollover = self.handler._should_rollover(record)
        self.assertTrue(should_rollover)

    @unittest.mock.patch("time.time")
    def test_should_rollover_time(self, mock_time):
        """Test time-based rollover detection."""
        future_time = 2000.0
        mock_time.return_value = 1000.0

        self.handler = EnhancedRotatingFileHandler(
            filename=self.test_file,
            rotation_type="time",
            time_interval="hourly",
        )

        # Set rollover time to the past
        self.handler._rollover_time = 1500.0

        # Current time is after rollover time
        mock_time.return_value = future_time

        record = logging.LogRecord(
            name="test",
            level=logging.INFO,
            pathname="",
            lineno=0,
            msg="Test message",
            args=(),
            exc_info=None,
        )

        should_rollover = self.handler._should_rollover(record)
        self.assertTrue(should_rollover)

    def test_rotation_info(self):
        """Test getting rotation information."""
        self.handler = EnhancedRotatingFileHandler(
            filename=self.test_file,
            rotation_type="both",
            max_bytes=2097152,  # 2MB exactly
            backup_count=5,
            time_interval="daily",
            time_when="02:30",
        )

        info = self.handler.get_rotation_info()

        self.assertEqual(info["filename"], self.test_file)
        self.assertEqual(info["rotation_type"], "both")
        self.assertEqual(info["max_bytes"], 2097152)
        self.assertAlmostEqual(info["max_size_mb"], 2.0, places=2)
        self.assertEqual(info["backup_count"], 5)
        self.assertEqual(info["time_interval"], "daily")
        self.assertEqual(info["time_when"], "02:30")
        self.assertIn("next_rollover", info)

    def test_handler_repr(self):
        """Test string representation of handler."""
        self.handler = EnhancedRotatingFileHandler(
            filename=self.test_file,
            rotation_type="size",
            max_bytes=1024,
            backup_count=3,
        )

        repr_str = repr(self.handler)
        self.assertIn("EnhancedRotatingFileHandler", repr_str)
        self.assertIn(self.test_file, repr_str)
        self.assertIn("size", repr_str)

    def test_close_handler(self):
        """Test handler cleanup and closing."""
        self.handler = EnhancedRotatingFileHandler(
            filename=self.test_file,
            rotation_type="size",
        )

        # Emit a record to open the stream
        record = logging.LogRecord(
            name="test",
            level=logging.INFO,
            pathname="",
            lineno=0,
            msg="Test message",
            args=(),
            exc_info=None,
        )
        self.handler.emit(record)

        self.assertIsNotNone(self.handler.stream)

        # Close the handler
        self.handler.close()

        self.assertIsNone(self.handler.stream)

    def test_delayed_file_opening(self):
        """Test delayed file opening functionality."""
        self.handler = EnhancedRotatingFileHandler(
            filename=self.test_file,
            rotation_type="size",
            delay=True,
        )

        # Stream should not be opened initially
        self.assertIsNone(self.handler.stream)

        # Emit a record to trigger opening
        record = logging.LogRecord(
            name="test",
            level=logging.INFO,
            pathname="",
            lineno=0,
            msg="Test message",
            args=(),
            exc_info=None,
        )
        self.handler.emit(record)

        # Stream should now be opened
        self.assertIsNotNone(self.handler.stream)


class TestRotatingFileHandlerFactory(unittest.TestCase):
    """Test cases for the factory function."""

    def setUp(self):
        """Set up test fixtures before each test method."""
        self.temp_dir = tempfile.mkdtemp()
        self.test_file = os.path.join(self.temp_dir, "test.log")

    def tearDown(self):
        """Clean up after each test method."""
        # Clean up temporary directory
        try:
            shutil.rmtree(self.temp_dir)
        except:
            pass

    def test_factory_basic_creation(self):
        """Test basic handler creation using factory function."""
        handler = create_rotating_file_handler(
            filename=self.test_file,
            rotation_type="size",
            max_bytes=1024,
        )

        try:
            self.assertIsInstance(handler, EnhancedRotatingFileHandler)
            self.assertEqual(handler.filename, self.test_file)
            self.assertEqual(handler.rotation_type, "size")
            self.assertEqual(handler.max_bytes, 1024)
        finally:
            handler.close()

    def test_factory_with_formatter(self):
        """Test factory function with custom formatter."""
        formatter = logging.Formatter("%(levelname)s: %(message)s")

        handler = create_rotating_file_handler(
            filename=self.test_file,
            formatter=formatter,
        )

        try:
            self.assertEqual(handler.formatter, formatter)
        finally:
            handler.close()

    def test_factory_with_string_level(self):
        """Test factory function with string log level."""
        handler = create_rotating_file_handler(
            filename=self.test_file,
            level="DEBUG",
        )

        try:
            self.assertEqual(handler.level, logging.DEBUG)
        finally:
            handler.close()

    def test_factory_with_int_level(self):
        """Test factory function with integer log level."""
        handler = create_rotating_file_handler(
            filename=self.test_file,
            level=logging.WARNING,
        )

        try:
            self.assertEqual(handler.level, logging.WARNING)
        finally:
            handler.close()

    def test_factory_time_based_handler(self):
        """Test factory function for time-based handler."""
        handler = create_rotating_file_handler(
            filename=self.test_file,
            rotation_type="time",
            time_interval="daily",
            time_when="midnight",
        )

        try:
            self.assertEqual(handler.rotation_type, "time")
            self.assertEqual(handler.time_interval, "daily")
            self.assertEqual(handler.time_when, "midnight")
        finally:
            handler.close()

    def test_factory_combined_rotation(self):
        """Test factory function for combined rotation."""
        handler = create_rotating_file_handler(
            filename=self.test_file,
            rotation_type="both",
            max_bytes=2048,
            time_interval="hourly",
        )

        try:
            self.assertEqual(handler.rotation_type, "both")
            self.assertEqual(handler.max_bytes, 2048)
            self.assertEqual(handler.time_interval, "hourly")
        finally:
            handler.close()

    def test_factory_error_handling(self):
        """Test factory function error handling."""
        with self.assertRaises(RotatingFileHandlerError):
            create_rotating_file_handler(
                filename=self.test_file,
                rotation_type="invalid",  # Invalid rotation type
            )


class TestIntegration(unittest.TestCase):
    """Integration tests for the enhanced rotating file handler."""

    def setUp(self):
        """Set up test fixtures before each test method."""
        self.temp_dir = tempfile.mkdtemp()
        self.test_file = os.path.join(self.temp_dir, "integration.log")

    def tearDown(self):
        """Clean up after each test method."""
        # Clean up temporary directory
        try:
            shutil.rmtree(self.temp_dir)
        except:
            pass

    def test_integration_with_logger(self):
        """Test integration with Python's logging system."""
        # Create logger
        logger = logging.getLogger("test_integration")
        logger.setLevel(logging.DEBUG)

        # Clear any existing handlers
        logger.handlers.clear()

        # Create and add rotating file handler
        handler = create_rotating_file_handler(
            filename=self.test_file,
            rotation_type="size",
            max_bytes=1024,
            backup_count=2,
            level=logging.INFO,
        )

        try:
            logger.addHandler(handler)

            # Log some messages
            logger.info("Integration test message 1")
            logger.warning("Integration test message 2")
            logger.error("Integration test message 3")

            # Check that file was created and contains messages
            self.assertTrue(os.path.exists(self.test_file))

            with open(self.test_file, "r") as f:
                content = f.read()
                self.assertIn("Integration test message 1", content)
                self.assertIn("Integration test message 2", content)
                self.assertIn("Integration test message 3", content)

        finally:
            logger.removeHandler(handler)
            handler.close()

    def test_multiple_rotation_cycles(self):
        """Test multiple rotation cycles."""
        handler = create_rotating_file_handler(
            filename=self.test_file,
            rotation_type="size",
            max_bytes=1024,  # Minimum allowed size for testing
            backup_count=3,
        )

        try:
            # Generate enough logs to trigger multiple rotations
            for i in range(100):
                record = logging.LogRecord(
                    name="test",
                    level=logging.INFO,
                    pathname="",
                    lineno=0,
                    msg=f"Test message number {i:03d} with some additional content to make it longer and trigger rotation when we have enough data",
                    args=(),
                    exc_info=None,
                )
                handler.emit(record)

            handler.flush()

            # Check that backup files were created
            backup_files = []
            for i in range(1, 5):  # Check for .1, .2, .3, .4
                backup_file = f"{self.test_file}.{i}"
                if os.path.exists(backup_file):
                    backup_files.append(backup_file)

            # Should have created some backup files
            self.assertGreater(len(backup_files), 0)

            # Should not exceed backup_count
            self.assertLessEqual(len(backup_files), 3)

        finally:
            handler.close()


if __name__ == "__main__":
    # Run the tests
    unittest.main(verbosity=2)
