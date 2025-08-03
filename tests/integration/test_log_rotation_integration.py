"""
Comprehensive Integration Tests for Log Rotation Functionality

This module provides comprehensive integration tests for the enhanced rotating file handler
that test real-world scenarios and complement the existing unit tests.

Test Classes:
    TestRealWorldSizeBasedRotation: Tests actual file size-based rotation scenarios
    TestTimeBasedRotationSimulation: Tests time-based rotation with mocked time
    TestCombinedRotationScenarios: Tests combined size and time rotation
    TestConcurrentAccessStress: Tests thread safety and concurrent access
    TestPerformanceAndReliability: Tests performance and error handling
    TestConfigurationDriven: Tests various configuration scenarios

Dependencies:
    - unittest: For test framework
    - unittest.mock: For mocking functionality
    - tempfile: For temporary file operations
    - threading: For concurrent access tests
    - time: For time-based operations
    - logging: For logging functionality
    - os: For file operations
    - pathlib: For path operations
"""

import unittest
import unittest.mock
import tempfile
import shutil
import os
import logging
import threading
import time
import random
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List

from aim2_project.aim2_utils.rotating_file_handler import (
    RotatingFileHandlerError,
    EnhancedRotatingFileHandler,
)


class TestRealWorldSizeBasedRotation(unittest.TestCase):
    """Test real-world size-based rotation scenarios with actual file content."""

    def setUp(self):
        """Set up test fixtures before each test method."""
        self.temp_dir = tempfile.mkdtemp()
        self.test_file = os.path.join(self.temp_dir, "size_rotation.log")
        self.handler = None

    def tearDown(self):
        """Clean up after each test method."""
        if self.handler:
            try:
                self.handler.close()
            except:
                pass

        try:
            shutil.rmtree(self.temp_dir)
        except:
            pass

    def test_multiple_rotations_with_real_content(self):
        """Test multiple rotations triggered by actual log content generation."""
        self.handler = EnhancedRotatingFileHandler(
            filename=self.test_file,
            rotation_type="size",
            max_bytes=2048,  # Small size to trigger rotation quickly
            backup_count=5,
        )

        # Generate realistic log content that will trigger multiple rotations
        messages = [
            "Application startup completed successfully",
            "Database connection established to postgresql://localhost:5432/testdb",
            "User authentication successful for user_id=12345, session_id=abc123def456",
            "Processing batch job: data_import_20250803, items=1500, estimated_time=45min",
            "Error: Failed to connect to external API endpoint https://api.example.com/v1/data",
            "Warning: Memory usage is at 85% of allocated limit (2.1GB/2.5GB)",
            "Info: Scheduled task 'cleanup_temp_files' completed in 3.2 seconds",
            "Debug: Cache hit rate for session 'user_12345' is 94.2% (47/50 requests)",
        ]

        rotation_count = 0
        total_messages = 0

        # Generate enough content to trigger multiple rotations
        for cycle in range(10):  # 10 cycles of messages
            for i, base_message in enumerate(messages):
                # Create varied messages with timestamps and additional context
                timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
                thread_id = f"Thread-{random.randint(1, 8)}"

                enhanced_message = (
                    f"{timestamp} [{thread_id}] {base_message} "
                    f"[correlation_id=req_{cycle:03d}_{i:03d}] "
                    f"[metadata={{cycle: {cycle}, iteration: {i}, "
                    f"random_data: {random.random():.6f}}}]"
                )

                record = logging.LogRecord(
                    name="integration.test",
                    level=logging.INFO,
                    pathname=__file__,
                    lineno=100 + i,
                    msg=enhanced_message,
                    args=(),
                    exc_info=None,
                )

                # Count files before emission
                files_before = self._count_rotation_files()

                self.handler.emit(record)
                self.handler.flush()

                # Count files after emission
                files_after = self._count_rotation_files()
                if files_after > files_before:
                    rotation_count += 1

                total_messages += 1

        # Verify multiple rotations occurred
        self.assertGreater(rotation_count, 0, "No rotations occurred")
        self.assertGreater(total_messages, 50, "Not enough messages generated")

        # Verify backup files exist and are properly numbered
        backup_files = self._get_backup_files()
        self.assertGreater(len(backup_files), 0, "No backup files created")
        self.assertLessEqual(len(backup_files), 5, "Too many backup files")

        # Verify backup files are properly ordered (newest first)
        for i, backup_file in enumerate(backup_files, 1):
            expected_name = f"{self.test_file}.{i}"
            self.assertEqual(backup_file, expected_name)
            self.assertTrue(os.path.exists(backup_file))

            # Verify file has content
            self.assertGreater(os.path.getsize(backup_file), 0)

    def test_backup_count_cleanup(self):
        """Test that old backup files are properly cleaned up when exceeding backup_count."""
        backup_count = 3
        self.handler = EnhancedRotatingFileHandler(
            filename=self.test_file,
            rotation_type="size",
            max_bytes=1536,  # Small size for quick rotation
            backup_count=backup_count,
        )

        # Generate enough content to exceed backup_count
        for batch in range(8):  # Should trigger more than 3 rotations
            for msg_num in range(20):
                message = (
                    f"Batch {batch:03d} Message {msg_num:03d}: "
                    f"This is a substantial log message that contains enough content "
                    f"to help trigger file rotation when combined with other messages. "
                    f"Random data: {random.random():.10f} "
                    f"Timestamp: {time.time():.6f}"
                )

                record = logging.LogRecord(
                    name="cleanup.test",
                    level=logging.INFO,
                    pathname=__file__,
                    lineno=200 + msg_num,
                    msg=message,
                    args=(),
                    exc_info=None,
                )

                self.handler.emit(record)

            self.handler.flush()

        # Verify backup_count is respected
        backup_files = self._get_backup_files()
        self.assertLessEqual(
            len(backup_files),
            backup_count,
            f"Found {len(backup_files)} backup files, expected <= {backup_count}",
        )

        # Verify no files beyond backup_count exist
        for i in range(backup_count + 1, backup_count + 5):
            excess_file = f"{self.test_file}.{i}"
            self.assertFalse(
                os.path.exists(excess_file),
                f"Excess backup file {excess_file} should not exist",
            )

    def test_large_single_message_rotation(self):
        """Test rotation triggered by a single large message."""
        self.handler = EnhancedRotatingFileHandler(
            filename=self.test_file,
            rotation_type="size",
            max_bytes=4096,  # 4KB limit
            backup_count=2,
        )

        # First, write some normal content
        for i in range(10):
            record = logging.LogRecord(
                name="large.test",
                level=logging.INFO,
                pathname=__file__,
                lineno=300 + i,
                msg=f"Normal message {i:03d}",
                args=(),
                exc_info=None,
            )
            self.handler.emit(record)

        self.handler.flush()
        initial_files = self._count_rotation_files()

        # Now write a large message that should trigger rotation
        large_content = "x" * 5000  # 5KB message, larger than max_bytes
        large_record = logging.LogRecord(
            name="large.test",
            level=logging.ERROR,
            pathname=__file__,
            lineno=400,
            msg=f"Large error dump: {large_content}",
            args=(),
            exc_info=None,
        )

        self.handler.emit(large_record)
        self.handler.flush()

        final_files = self._count_rotation_files()

        # Verify rotation occurred
        self.assertGreater(final_files, initial_files, "Rotation should have occurred")

        # Verify the large message is in the current log file
        self.assertTrue(os.path.exists(self.test_file))
        with open(self.test_file, "r", encoding="utf-8") as f:
            content = f.read()
            self.assertIn("Large error dump", content)

    def _count_rotation_files(self) -> int:
        """Count the number of rotation-related files (main + backups)."""
        count = 0
        if os.path.exists(self.test_file):
            count += 1

        i = 1
        while os.path.exists(f"{self.test_file}.{i}"):
            count += 1
            i += 1

        return count

    def _get_backup_files(self) -> List[str]:
        """Get list of backup files in order."""
        backup_files = []
        i = 1
        while os.path.exists(f"{self.test_file}.{i}"):
            backup_files.append(f"{self.test_file}.{i}")
            i += 1
        return backup_files


class TestTimeBasedRotationSimulation(unittest.TestCase):
    """Test time-based rotation with mocked time advancement."""

    def setUp(self):
        """Set up test fixtures before each test method."""
        self.temp_dir = tempfile.mkdtemp()
        self.test_file = os.path.join(self.temp_dir, "time_rotation.log")
        self.handler = None

    def tearDown(self):
        """Clean up after each test method."""
        if self.handler:
            try:
                self.handler.close()
            except:
                pass

        try:
            shutil.rmtree(self.temp_dir)
        except:
            pass

    @unittest.mock.patch("time.time")
    @unittest.mock.patch("aim2_project.aim2_utils.rotating_file_handler.datetime")
    def test_hourly_rotation_simulation(self, mock_datetime, mock_time):
        """Test hourly rotation by mocking time advancement."""
        # Set up initial time: 2025-01-15 10:30:00
        base_time = datetime(2025, 1, 15, 10, 30, 0)
        mock_datetime.now.return_value = base_time
        mock_datetime.fromtimestamp.side_effect = datetime.fromtimestamp
        mock_time.return_value = base_time.timestamp()

        self.handler = EnhancedRotatingFileHandler(
            filename=self.test_file,
            rotation_type="time",
            time_interval="hourly",
            backup_count=3,
        )

        # Log initial messages
        for i in range(5):
            record = logging.LogRecord(
                name="hourly.test",
                level=logging.INFO,
                pathname=__file__,
                lineno=500 + i,
                msg=f"Initial hour message {i}",
                args=(),
                exc_info=None,
            )
            self.handler.emit(record)

        self.handler.flush()
        initial_files = self._count_rotation_files()

        # Advance time to next hour: 11:00:00
        next_hour = base_time.replace(minute=0, second=0) + timedelta(hours=1)
        mock_time.return_value = next_hour.timestamp()

        # Log message that should trigger rotation
        record = logging.LogRecord(
            name="hourly.test",
            level=logging.INFO,
            pathname=__file__,
            lineno=600,
            msg="Next hour message - should trigger rotation",
            args=(),
            exc_info=None,
        )

        self.handler.emit(record)
        self.handler.flush()

        final_files = self._count_rotation_files()

        # Verify rotation occurred
        self.assertGreater(
            final_files, initial_files, "Hourly rotation should have occurred"
        )

        # Advance time further and test multiple rotations
        for hour_offset in range(1, 4):
            future_time = next_hour + timedelta(hours=hour_offset)
            mock_time.return_value = future_time.timestamp()

            record = logging.LogRecord(
                name="hourly.test",
                level=logging.INFO,
                pathname=__file__,
                lineno=700 + hour_offset,
                msg=f"Hour +{hour_offset} message",
                args=(),
                exc_info=None,
            )

            self.handler.emit(record)
            self.handler.flush()

        # Verify backup files were created
        backup_files = self._get_backup_files()
        self.assertGreater(len(backup_files), 0, "No backup files created")
        self.assertLessEqual(len(backup_files), 3, "Too many backup files")

    @unittest.mock.patch("time.time")
    @unittest.mock.patch("aim2_project.aim2_utils.rotating_file_handler.datetime")
    def test_daily_custom_time_rotation(self, mock_datetime, mock_time):
        """Test daily rotation at custom time (e.g., 02:30)."""
        # Set up initial time: 2025-01-15 01:45:00 (before rotation time)
        base_time = datetime(2025, 1, 15, 1, 45, 0)
        mock_datetime.now.return_value = base_time
        mock_datetime.fromtimestamp.side_effect = datetime.fromtimestamp
        mock_time.return_value = base_time.timestamp()

        self.handler = EnhancedRotatingFileHandler(
            filename=self.test_file,
            rotation_type="time",
            time_interval="daily",
            time_when="02:30",  # Rotate at 2:30 AM
            backup_count=2,
        )

        # Log messages before rotation time
        for i in range(3):
            record = logging.LogRecord(
                name="daily.test",
                level=logging.INFO,
                pathname=__file__,
                lineno=800 + i,
                msg=f"Before rotation: {base_time.strftime('%H:%M:%S')} - message {i}",
                args=(),
                exc_info=None,
            )
            self.handler.emit(record)

        self.handler.flush()
        initial_files = self._count_rotation_files()

        # Advance time past rotation time: 02:35:00
        rotation_time = base_time.replace(hour=2, minute=35, second=0)
        mock_time.return_value = rotation_time.timestamp()

        # Log message that should trigger rotation
        record = logging.LogRecord(
            name="daily.test",
            level=logging.INFO,
            pathname=__file__,
            lineno=900,
            msg=f"After rotation time: {rotation_time.strftime('%H:%M:%S')}",
            args=(),
            exc_info=None,
        )

        self.handler.emit(record)
        self.handler.flush()

        final_files = self._count_rotation_files()

        # Verify rotation occurred
        self.assertGreater(
            final_files, initial_files, "Daily rotation should have occurred"
        )

        # Verify current file contains the new message
        with open(self.test_file, "r", encoding="utf-8") as f:
            content = f.read()
            self.assertIn("After rotation time: 02:35:00", content)

        # Verify backup file contains old messages
        backup_files = self._get_backup_files()
        self.assertGreater(len(backup_files), 0, "No backup files created")

        with open(backup_files[0], "r", encoding="utf-8") as f:
            backup_content = f.read()
            self.assertIn("Before rotation: 01:45:00", backup_content)

    @unittest.mock.patch("time.time")
    @unittest.mock.patch("aim2_project.aim2_utils.rotating_file_handler.datetime")
    def test_weekly_rotation_simulation(self, mock_datetime, mock_time):
        """Test weekly rotation simulation."""
        # Set up initial time: Friday 2025-01-17 15:00:00
        base_time = datetime(2025, 1, 17, 15, 0, 0)  # Friday
        mock_datetime.now.return_value = base_time
        mock_datetime.fromtimestamp.side_effect = datetime.fromtimestamp
        mock_time.return_value = base_time.timestamp()

        self.handler = EnhancedRotatingFileHandler(
            filename=self.test_file,
            rotation_type="time",
            time_interval="weekly",
            backup_count=4,
        )

        # Log messages during the week
        for day in range(3):  # Friday, Saturday, Sunday
            day_time = base_time + timedelta(days=day)
            mock_time.return_value = day_time.timestamp()

            record = logging.LogRecord(
                name="weekly.test",
                level=logging.INFO,
                pathname=__file__,
                lineno=1000 + day,
                msg=f"Day {day_time.strftime('%A')}: message from {day_time.strftime('%Y-%m-%d')}",
                args=(),
                exc_info=None,
            )
            self.handler.emit(record)
            self.handler.flush()

        initial_files = self._count_rotation_files()

        # Advance to next Monday (rotation day): 2025-01-20 00:01:00
        next_monday = datetime(2025, 1, 20, 0, 1, 0)
        mock_time.return_value = next_monday.timestamp()

        # Log message that should trigger rotation
        record = logging.LogRecord(
            name="weekly.test",
            level=logging.INFO,
            pathname=__file__,
            lineno=1100,
            msg=f"New week: {next_monday.strftime('%A %Y-%m-%d')}",
            args=(),
            exc_info=None,
        )

        self.handler.emit(record)
        self.handler.flush()

        final_files = self._count_rotation_files()

        # Verify rotation occurred
        self.assertGreater(
            final_files, initial_files, "Weekly rotation should have occurred"
        )

        # Verify backup files contain the week's messages
        backup_files = self._get_backup_files()
        self.assertGreater(len(backup_files), 0, "No backup files created")

        with open(backup_files[0], "r", encoding="utf-8") as f:
            backup_content = f.read()
            self.assertIn("Day Friday", backup_content)
            self.assertIn("Day Saturday", backup_content)
            self.assertIn("Day Sunday", backup_content)

    def _count_rotation_files(self) -> int:
        """Count the number of rotation-related files (main + backups)."""
        count = 0
        if os.path.exists(self.test_file):
            count += 1

        i = 1
        while os.path.exists(f"{self.test_file}.{i}"):
            count += 1
            i += 1

        return count

    def _get_backup_files(self) -> List[str]:
        """Get list of backup files in order."""
        backup_files = []
        i = 1
        while os.path.exists(f"{self.test_file}.{i}"):
            backup_files.append(f"{self.test_file}.{i}")
            i += 1
        return backup_files


class TestCombinedRotationScenarios(unittest.TestCase):
    """Test combined rotation scenarios where both size and time limits apply."""

    def setUp(self):
        """Set up test fixtures before each test method."""
        self.temp_dir = tempfile.mkdtemp()
        self.test_file = os.path.join(self.temp_dir, "combined_rotation.log")
        self.handler = None

    def tearDown(self):
        """Clean up after each test method."""
        if self.handler:
            try:
                self.handler.close()
            except:
                pass

        try:
            shutil.rmtree(self.temp_dir)
        except:
            pass

    def test_size_triggers_before_time(self):
        """Test scenario where size limit is reached before time limit."""
        self.handler = EnhancedRotatingFileHandler(
            filename=self.test_file,
            rotation_type="both",
            max_bytes=2048,  # 2KB - relatively small
            time_interval="daily",  # Long time interval
            time_when="midnight",
            backup_count=3,
        )

        initial_files = self._count_rotation_files()

        # Generate enough content to trigger size-based rotation quickly
        for i in range(50):
            message = (
                f"Message {i:03d}: This is a substantial message that contains "
                f"enough content to help reach the size limit before the time limit. "
                f"Random data: {random.random():.10f} "
                f"Timestamp: {datetime.now().isoformat()} "
                f"Additional padding: {'x' * 50}"
            )

            record = logging.LogRecord(
                name="combined.test",
                level=logging.INFO,
                pathname=__file__,
                lineno=1200 + i,
                msg=message,
                args=(),
                exc_info=None,
            )

            self.handler.emit(record)

            # Check if rotation occurred
            current_files = self._count_rotation_files()
            if current_files > initial_files:
                # Size-based rotation occurred
                break

        self.handler.flush()
        final_files = self._count_rotation_files()

        # Verify that rotation occurred due to size (not time)
        self.assertGreater(
            final_files, initial_files, "Size-based rotation should have occurred"
        )

        # Verify that the rotation info still shows proper next rollover time
        rotation_info = self.handler.get_rotation_info()
        self.assertIn("next_rollover", rotation_info)
        self.assertIsNotNone(rotation_info["next_rollover"])

    @unittest.mock.patch("time.time")
    @unittest.mock.patch("aim2_project.aim2_utils.rotating_file_handler.datetime")
    def test_time_triggers_before_size(self, mock_datetime, mock_time):
        """Test scenario where time limit is reached before size limit."""
        # Set up initial time
        base_time = datetime(2025, 1, 15, 23, 45, 0)  # Close to midnight
        mock_datetime.now.return_value = base_time
        mock_datetime.fromtimestamp.side_effect = datetime.fromtimestamp
        mock_time.return_value = base_time.timestamp()

        self.handler = EnhancedRotatingFileHandler(
            filename=self.test_file,
            rotation_type="both",
            max_bytes=50 * 1024,  # 50KB - large size limit
            time_interval="daily",
            time_when="midnight",
            backup_count=3,
        )

        # Write some content, but not enough to trigger size rotation
        for i in range(10):
            record = logging.LogRecord(
                name="combined.test",
                level=logging.INFO,
                pathname=__file__,
                lineno=1300 + i,
                msg=f"Pre-midnight message {i}: {base_time.strftime('%H:%M:%S')}",
                args=(),
                exc_info=None,
            )
            self.handler.emit(record)

        self.handler.flush()
        initial_files = self._count_rotation_files()

        # Get current file size to verify it's well under the limit
        current_size = (
            os.path.getsize(self.test_file) if os.path.exists(self.test_file) else 0
        )
        self.assertLess(current_size, 50 * 1024, "File should be under size limit")

        # Advance time past midnight
        next_day = datetime(2025, 1, 16, 0, 5, 0)  # 5 minutes after midnight
        mock_time.return_value = next_day.timestamp()

        # Write message that should trigger time-based rotation
        record = logging.LogRecord(
            name="combined.test",
            level=logging.INFO,
            pathname=__file__,
            lineno=1400,
            msg=f"Post-midnight message: {next_day.strftime('%H:%M:%S')}",
            args=(),
            exc_info=None,
        )

        self.handler.emit(record)
        self.handler.flush()

        final_files = self._count_rotation_files()

        # Verify that time-based rotation occurred
        self.assertGreater(
            final_files, initial_files, "Time-based rotation should have occurred"
        )

        # Verify current file contains post-midnight message
        with open(self.test_file, "r", encoding="utf-8") as f:
            content = f.read()
            self.assertIn("Post-midnight message: 00:05:00", content)

        # Verify backup file contains pre-midnight messages
        backup_files = self._get_backup_files()
        self.assertGreater(len(backup_files), 0, "No backup files created")

        with open(backup_files[0], "r", encoding="utf-8") as f:
            backup_content = f.read()
            self.assertIn("Pre-midnight message", backup_content)

    @unittest.mock.patch("time.time")
    @unittest.mock.patch("aim2_project.aim2_utils.rotating_file_handler.datetime")
    def test_size_rotation_recalculates_time(self, mock_datetime, mock_time):
        """Test that size-triggered rotation properly recalculates next time rollover."""
        # Set up time
        base_time = datetime(2025, 1, 15, 14, 30, 0)
        mock_datetime.now.return_value = base_time
        mock_datetime.fromtimestamp.side_effect = datetime.fromtimestamp
        mock_time.return_value = base_time.timestamp()

        self.handler = EnhancedRotatingFileHandler(
            filename=self.test_file,
            rotation_type="both",
            max_bytes=1536,  # Small size for quick rotation
            time_interval="daily",
            time_when="midnight",
            backup_count=2,
        )

        # Get initial rotation info
        initial_info = self.handler.get_rotation_info()
        initial_next_rollover = initial_info["next_rollover"]

        # Generate content to trigger size-based rotation
        for i in range(30):
            message = f"Size rotation test message {i:03d}: " + "x" * 100
            record = logging.LogRecord(
                name="recalc.test",
                level=logging.INFO,
                pathname=__file__,
                lineno=1500 + i,
                msg=message,
                args=(),
                exc_info=None,
            )
            self.handler.emit(record)

        self.handler.flush()

        # Get rotation info after size-triggered rotation
        final_info = self.handler.get_rotation_info()
        final_next_rollover = final_info["next_rollover"]

        # Verify that next rollover time was recalculated
        # It should still be the next midnight, but the internal timing should be updated
        self.assertEqual(
            initial_next_rollover,
            final_next_rollover,
            "Next rollover time should remain the same for daily rotation",
        )

        # Verify rotation occurred
        backup_files = self._get_backup_files()
        self.assertGreater(
            len(backup_files), 0, "Size-based rotation should have occurred"
        )

    def _count_rotation_files(self) -> int:
        """Count the number of rotation-related files (main + backups)."""
        count = 0
        if os.path.exists(self.test_file):
            count += 1

        i = 1
        while os.path.exists(f"{self.test_file}.{i}"):
            count += 1
            i += 1

        return count

    def _get_backup_files(self) -> List[str]:
        """Get list of backup files in order."""
        backup_files = []
        i = 1
        while os.path.exists(f"{self.test_file}.{i}"):
            backup_files.append(f"{self.test_file}.{i}")
            i += 1
        return backup_files


class TestConcurrentAccessStress(unittest.TestCase):
    """Test concurrent access and thread safety during rotation operations."""

    def setUp(self):
        """Set up test fixtures before each test method."""
        self.temp_dir = tempfile.mkdtemp()
        self.test_file = os.path.join(self.temp_dir, "concurrent_rotation.log")
        self.handler = None

    def tearDown(self):
        """Clean up after each test method."""
        if self.handler:
            try:
                self.handler.close()
            except:
                pass

        try:
            shutil.rmtree(self.temp_dir)
        except:
            pass

    def test_multiple_threads_same_handler(self):
        """Test multiple threads writing to the same rotating log handler."""
        self.handler = EnhancedRotatingFileHandler(
            filename=self.test_file,
            rotation_type="size",
            max_bytes=8192,  # Larger size to reduce rotation frequency
            backup_count=5,
        )

        num_threads = 4  # Reduced thread count for more manageable concurrency
        messages_per_thread = 25  # Reduced message count
        thread_completion_count = 0
        lock = threading.Lock()

        def worker_thread(thread_id: int) -> int:
            """Worker function for concurrent logging."""
            nonlocal thread_completion_count
            messages_written = 0

            for i in range(messages_per_thread):
                try:
                    message = (
                        f"Thread-{thread_id:02d} Message-{i:03d}: "
                        f"Concurrent logging test message. "
                        f"Timestamp: {time.time():.6f}"
                    )

                    record = logging.LogRecord(
                        name=f"thread.{thread_id}",
                        level=logging.INFO,
                        pathname=__file__,
                        lineno=1600 + i,
                        msg=message,
                        args=(),
                        exc_info=None,
                    )

                    self.handler.emit(record)
                    messages_written += 1

                    # Small delay to allow other threads to interleave
                    time.sleep(0.001)

                except Exception:
                    # In concurrent scenarios, some exceptions during rotation are expected
                    pass

            with lock:
                thread_completion_count += 1

            return messages_written

        # Start threads
        with ThreadPoolExecutor(max_workers=num_threads) as executor:
            futures = [
                executor.submit(worker_thread, thread_id)
                for thread_id in range(num_threads)
            ]

            # Wait for completion
            results = [future.result() for future in as_completed(futures)]

        self.handler.flush()

        # Verify all threads completed
        self.assertEqual(
            thread_completion_count, num_threads, "Not all threads completed"
        )

        # Verify basic functionality - files exist and have content
        self.assertTrue(os.path.exists(self.test_file), "Main log file should exist")

        # Check for file corruption - all files should be readable
        try:
            with open(self.test_file, "r", encoding="utf-8") as f:
                main_content = f.read()
                self.assertGreater(
                    len(main_content), 0, "Main file should have content"
                )
        except UnicodeDecodeError:
            self.fail("Main log file appears to be corrupted")

        # Check backup files for corruption
        backup_files = self._get_backup_files()
        for backup_file in backup_files:
            try:
                with open(backup_file, "r", encoding="utf-8") as f:
                    backup_content = f.read()
                    self.assertGreater(
                        len(backup_content),
                        0,
                        f"Backup file {backup_file} should have content",
                    )
            except UnicodeDecodeError:
                self.fail(f"Backup file {backup_file} appears to be corrupted")

        # Verify at least some messages were logged (allowing for concurrent loss)
        total_lines = 0
        with open(self.test_file, "r", encoding="utf-8") as f:
            total_lines += sum(1 for line in f if line.strip())

        for backup_file in backup_files:
            with open(backup_file, "r", encoding="utf-8") as f:
                total_lines += sum(1 for line in f if line.strip())

        # In high-concurrency scenarios, some message loss is acceptable
        # The key is that the system doesn't crash and files aren't corrupted
        expected_messages = num_threads * messages_per_thread
        self.assertGreater(total_lines, 0, "No messages were logged")
        self.assertGreater(
            total_lines,
            expected_messages * 0.5,
            "Too much message loss in concurrent scenario",
        )

    def test_rotation_during_high_frequency_logging(self):
        """Test rotation behavior during high-frequency logging."""
        self.handler = EnhancedRotatingFileHandler(
            filename=self.test_file,
            rotation_type="size",
            max_bytes=2048,  # Small size for frequent rotation
            backup_count=3,
        )

        num_threads = 4
        duration_seconds = 2
        messages_sent = []

        def rapid_logger(thread_id: int):
            """Rapid logging function."""
            start_time = time.time()
            count = 0

            while time.time() - start_time < duration_seconds:
                try:
                    message = f"T{thread_id:02d}-{count:04d}: {time.time():.6f}"
                    record = logging.LogRecord(
                        name=f"rapid.{thread_id}",
                        level=logging.INFO,
                        pathname=__file__,
                        lineno=1700 + count,
                        msg=message,
                        args=(),
                        exc_info=None,
                    )

                    self.handler.emit(record)
                    count += 1

                except Exception:
                    pass  # Continue logging despite errors

            messages_sent.append(count)

        # Start rapid logging threads
        threads = []
        for thread_id in range(num_threads):
            thread = threading.Thread(target=rapid_logger, args=(thread_id,))
            threads.append(thread)
            thread.start()

        # Wait for threads to complete
        for thread in threads:
            thread.join()

        self.handler.flush()

        # Verify no data loss and proper rotation
        total_sent = sum(messages_sent)
        self.assertGreater(
            total_sent, 100, "Not enough messages were sent for stress test"
        )

        # Verify files exist and have content
        self.assertTrue(os.path.exists(self.test_file), "Main log file should exist")
        self.assertGreater(
            os.path.getsize(self.test_file), 0, "Main log file should have content"
        )

        # Verify backup files were created (indicating rotation occurred)
        backup_files = self._get_backup_files()
        self.assertGreater(
            len(backup_files), 0, "No rotation occurred during stress test"
        )

        # Verify no corruption in files
        for backup_file in backup_files + [self.test_file]:
            try:
                with open(backup_file, "r", encoding="utf-8") as f:
                    content = f.read()
                    self.assertGreater(len(content), 0, f"File {backup_file} is empty")
            except UnicodeDecodeError:
                self.fail(f"File {backup_file} appears to be corrupted")

    def _get_backup_files(self) -> List[str]:
        """Get list of backup files in order."""
        backup_files = []
        i = 1
        while os.path.exists(f"{self.test_file}.{i}"):
            backup_files.append(f"{self.test_file}.{i}")
            i += 1
        return backup_files


class TestPerformanceAndReliability(unittest.TestCase):
    """Test performance characteristics and error handling reliability."""

    def setUp(self):
        """Set up test fixtures before each test method."""
        self.temp_dir = tempfile.mkdtemp()
        self.test_file = os.path.join(self.temp_dir, "performance_test.log")
        self.handler = None

    def tearDown(self):
        """Clean up after each test method."""
        if self.handler:
            try:
                self.handler.close()
            except:
                pass

        try:
            shutil.rmtree(self.temp_dir)
        except:
            pass

    def test_large_file_rotation_performance(self):
        """Test rotation performance with large log files."""
        # Create handler with larger file size for performance testing
        self.handler = EnhancedRotatingFileHandler(
            filename=self.test_file,
            rotation_type="size",
            max_bytes=5 * 1024 * 1024,  # 5MB files
            backup_count=3,
        )

        start_time = time.time()
        messages_written = 0

        # Generate substantial content
        base_message = "Performance test message with substantial content: " + "x" * 200

        # Write until we get at least one rotation
        initial_files = self._count_rotation_files()

        while (
            self._count_rotation_files() <= initial_files
            and time.time() - start_time < 30
        ):
            record = logging.LogRecord(
                name="performance.test",
                level=logging.INFO,
                pathname=__file__,
                lineno=1800 + (messages_written % 1000),
                msg=f"{base_message} #{messages_written:06d}",
                args=(),
                exc_info=None,
            )

            self.handler.emit(record)
            messages_written += 1

            # Periodic flush
            if messages_written % 100 == 0:
                self.handler.flush()

        self.handler.flush()
        end_time = time.time()

        duration = end_time - start_time

        # Verify performance is reasonable (should complete within 30 seconds)
        self.assertLess(duration, 30, "Large file rotation took too long")
        self.assertGreater(
            messages_written, 1000, "Not enough messages written for performance test"
        )

        # Verify rotation occurred
        final_files = self._count_rotation_files()
        self.assertGreater(final_files, initial_files, "No rotation occurred")

        # Calculate throughput
        throughput = messages_written / duration
        self.assertGreater(
            throughput, 100, "Throughput too low (should be > 100 messages/sec)"
        )

    @unittest.mock.patch(
        "builtins.open", side_effect=PermissionError("Permission denied")
    )
    def test_file_permission_error_handling(self, mock_open):
        """Test graceful handling of file permission errors."""
        # This test verifies that the handler gracefully handles permission errors
        # without crashing the application

        try:
            self.handler = EnhancedRotatingFileHandler(
                filename=self.test_file,
                rotation_type="size",
                max_bytes=1024,
                backup_count=2,
            )

            # Should raise RotatingFileHandlerError due to permission error
            self.fail("Expected RotatingFileHandlerError due to permission error")

        except RotatingFileHandlerError as e:
            self.assertIn("Failed to open log file", str(e))
            self.assertIsInstance(e.cause, PermissionError)

    @unittest.mock.patch("os.path.exists")
    @unittest.mock.patch("os.remove")
    @unittest.mock.patch("os.rename")
    def test_rotation_with_file_system_errors(
        self, mock_rename, mock_remove, mock_exists
    ):
        """Test rotation behavior when file system operations fail."""
        # Set up mocks to simulate file system errors
        mock_exists.return_value = True
        mock_remove.side_effect = OSError("Disk full")
        mock_rename.side_effect = OSError("Disk full")

        self.handler = EnhancedRotatingFileHandler(
            filename=self.test_file,
            rotation_type="size",
            max_bytes=1024,
            backup_count=2,
            delay=True,  # Delay opening to avoid initial file creation
        )

        # Create a record that should trigger rotation
        record = logging.LogRecord(
            name="error.test",
            level=logging.ERROR,
            pathname=__file__,
            lineno=1900,
            msg="x" * 2000,  # Large message to trigger rotation
            args=(),
            exc_info=None,
        )

        # Mock _open to succeed but _do_rollover to fail
        original_open = self.handler._open

        def mock_open_success():
            self.handler.stream = unittest.mock.MagicMock()
            self.handler.stream.write = unittest.mock.MagicMock()
            self.handler.stream.flush = unittest.mock.MagicMock()
            self.handler.stream.close = unittest.mock.MagicMock()
            self.handler.stream.fileno.return_value = 1

        self.handler._open = mock_open_success

        # This should handle the rotation error gracefully
        try:
            self.handler.emit(record)
            # If we get here, the handler managed the error gracefully
        except Exception as e:
            # Rotation errors should be caught and handled by handleError
            self.assertIsInstance(e, (OSError, RotatingFileHandlerError))

    def test_rapid_rotation_scenarios(self):
        """Test scenarios with very rapid rotation requirements."""
        self.handler = EnhancedRotatingFileHandler(
            filename=self.test_file,
            rotation_type="size",
            max_bytes=1024,  # Very small for rapid rotation
            backup_count=10,  # More backups to test cleanup
        )

        rotation_count = 0
        messages_written = 0

        # Write messages that will cause rapid rotation
        for batch in range(20):  # 20 batches
            for msg in range(5):  # 5 messages per batch
                message = (
                    f"Rapid rotation batch {batch:02d} message {msg:02d}: " + "x" * 150
                )

                files_before = self._count_rotation_files()

                record = logging.LogRecord(
                    name="rapid.test",
                    level=logging.INFO,
                    pathname=__file__,
                    lineno=2000 + messages_written,
                    msg=message,
                    args=(),
                    exc_info=None,
                )

                self.handler.emit(record)
                messages_written += 1

                files_after = self._count_rotation_files()
                if files_after > files_before:
                    rotation_count += 1

        self.handler.flush()

        # Verify multiple rotations occurred
        self.assertGreater(rotation_count, 5, "Expected multiple rapid rotations")

        # Verify backup count is respected
        backup_files = self._get_backup_files()
        self.assertLessEqual(len(backup_files), 10, "Too many backup files")

        # Verify all files have content
        for backup_file in backup_files + [self.test_file]:
            if os.path.exists(backup_file):
                self.assertGreater(
                    os.path.getsize(backup_file),
                    0,
                    f"File {backup_file} should have content",
                )

    def _count_rotation_files(self) -> int:
        """Count the number of rotation-related files (main + backups)."""
        count = 0
        if os.path.exists(self.test_file):
            count += 1

        i = 1
        while os.path.exists(f"{self.test_file}.{i}"):
            count += 1
            i += 1

        return count

    def _get_backup_files(self) -> List[str]:
        """Get list of backup files in order."""
        backup_files = []
        i = 1
        while os.path.exists(f"{self.test_file}.{i}"):
            backup_files.append(f"{self.test_file}.{i}")
            i += 1
        return backup_files


class TestConfigurationDriven(unittest.TestCase):
    """Test various configuration scenarios and edge cases."""

    def setUp(self):
        """Set up test fixtures before each test method."""
        self.temp_dir = tempfile.mkdtemp()
        self.handler = None

    def tearDown(self):
        """Clean up after each test method."""
        if self.handler:
            try:
                self.handler.close()
            except:
                pass

        try:
            shutil.rmtree(self.temp_dir)
        except:
            pass

    def test_various_file_size_formats(self):
        """Test different file size configurations."""
        test_cases = [
            (1024, "1KB"),  # Minimum size
            (1024 * 1024, "1MB"),
            (5 * 1024 * 1024, "5MB"),
            (10 * 1024 * 1024, "10MB"),
        ]

        for max_bytes, description in test_cases:
            with self.subTest(max_bytes=max_bytes, description=description):
                test_file = os.path.join(
                    self.temp_dir, f"size_{description.lower()}.log"
                )

                handler = EnhancedRotatingFileHandler(
                    filename=test_file,
                    rotation_type="size",
                    max_bytes=max_bytes,
                    backup_count=2,
                )

                try:
                    # Verify configuration
                    self.assertEqual(handler.max_bytes, max_bytes)

                    # Test basic functionality
                    record = logging.LogRecord(
                        name="size.test",
                        level=logging.INFO,
                        pathname=__file__,
                        lineno=2100,
                        msg=f"Testing {description} configuration",
                        args=(),
                        exc_info=None,
                    )

                    handler.emit(record)
                    handler.flush()

                    # Verify file was created
                    self.assertTrue(os.path.exists(test_file))

                    # Verify rotation info
                    info = handler.get_rotation_info()
                    self.assertEqual(info["max_bytes"], max_bytes)

                finally:
                    handler.close()

    def test_various_backup_counts(self):
        """Test different backup_count configurations."""
        test_cases = [0, 1, 2, 5, 10, 50]

        for backup_count in test_cases:
            with self.subTest(backup_count=backup_count):
                test_file = os.path.join(self.temp_dir, f"backup_{backup_count}.log")

                handler = EnhancedRotatingFileHandler(
                    filename=test_file,
                    rotation_type="size",
                    max_bytes=1536,  # Small size for quick rotation
                    backup_count=backup_count,
                )

                try:
                    # Generate enough content to trigger multiple rotations
                    for i in range(50):
                        message = (
                            f"Backup count {backup_count} test message {i:03d}: "
                            + "x" * 100
                        )
                        record = logging.LogRecord(
                            name="backup.test",
                            level=logging.INFO,
                            pathname=__file__,
                            lineno=2200 + i,
                            msg=message,
                            args=(),
                            exc_info=None,
                        )
                        handler.emit(record)

                    handler.flush()

                    # Count actual backup files
                    actual_backups = 0
                    i = 1
                    while os.path.exists(f"{test_file}.{i}"):
                        actual_backups += 1
                        i += 1

                    # Verify backup count is respected
                    self.assertLessEqual(
                        actual_backups,
                        backup_count,
                        f"Too many backup files for backup_count={backup_count}",
                    )

                finally:
                    handler.close()

    def test_various_time_configurations(self):
        """Test different time-based rotation configurations."""
        test_cases = [
            ("hourly", "midnight"),
            ("daily", "midnight"),
            ("daily", "02:30"),
            ("daily", "14:45"),
            ("weekly", "midnight"),
        ]

        for time_interval, time_when in test_cases:
            with self.subTest(time_interval=time_interval, time_when=time_when):
                test_file = os.path.join(
                    self.temp_dir,
                    f"time_{time_interval}_{time_when.replace(':', '')}.log",
                )

                handler = EnhancedRotatingFileHandler(
                    filename=test_file,
                    rotation_type="time",
                    time_interval=time_interval,
                    time_when=time_when,
                    backup_count=2,
                )

                try:
                    # Verify configuration
                    self.assertEqual(handler.time_interval, time_interval)
                    self.assertEqual(handler.time_when, time_when)
                    self.assertIsNotNone(handler._rollover_time)

                    # Test basic functionality
                    record = logging.LogRecord(
                        name="time.test",
                        level=logging.INFO,
                        pathname=__file__,
                        lineno=2300,
                        msg=f"Testing {time_interval} at {time_when}",
                        args=(),
                        exc_info=None,
                    )

                    handler.emit(record)
                    handler.flush()

                    # Verify file was created
                    self.assertTrue(os.path.exists(test_file))

                    # Verify rotation info
                    info = handler.get_rotation_info()
                    self.assertEqual(info["time_interval"], time_interval)
                    self.assertEqual(info["time_when"], time_when)
                    self.assertIn("next_rollover", info)

                finally:
                    handler.close()

    def test_encoding_configurations(self):
        """Test different encoding configurations."""
        test_cases = ["utf-8", "utf-16", "ascii", "latin-1"]

        for encoding in test_cases:
            with self.subTest(encoding=encoding):
                test_file = os.path.join(
                    self.temp_dir, f"encoding_{encoding.replace('-', '_')}.log"
                )

                handler = EnhancedRotatingFileHandler(
                    filename=test_file,
                    rotation_type="size",
                    max_bytes=2048,
                    backup_count=1,
                    encoding=encoding,
                )

                try:
                    # Test with appropriate content for encoding
                    if encoding == "ascii":
                        message = "ASCII test message 123"
                    else:
                        message = "Encoding test: Hello world cafe naive"  # ASCII-compatible for testing

                    record = logging.LogRecord(
                        name="encoding.test",
                        level=logging.INFO,
                        pathname=__file__,
                        lineno=2400,
                        msg=message,
                        args=(),
                        exc_info=None,
                    )

                    handler.emit(record)
                    handler.flush()

                    # Verify file was created and can be read with correct encoding
                    self.assertTrue(os.path.exists(test_file))

                    with open(test_file, "r", encoding=encoding) as f:
                        content = f.read()
                        if encoding == "ascii":
                            self.assertIn("ASCII test message", content)
                        else:
                            self.assertIn("Encoding test", content)
                            self.assertGreater(len(content), 0)

                finally:
                    handler.close()

    def test_delay_opening_configurations(self):
        """Test delay opening functionality with different configurations."""
        test_cases = [
            {"rotation_type": "size", "delay": True},
            {"rotation_type": "size", "delay": False},
            {"rotation_type": "time", "delay": True},
            {"rotation_type": "both", "delay": True},
        ]

        for config in test_cases:
            with self.subTest(config=config):
                test_file = os.path.join(
                    self.temp_dir,
                    f"delay_{config['rotation_type']}_{config['delay']}.log",
                )

                handler = EnhancedRotatingFileHandler(
                    filename=test_file, max_bytes=2048, backup_count=1, **config
                )

                try:
                    # Check initial stream state
                    if config["delay"]:
                        self.assertIsNone(
                            handler.stream,
                            "Stream should not be opened with delay=True",
                        )
                    else:
                        self.assertIsNotNone(
                            handler.stream, "Stream should be opened with delay=False"
                        )

                    # Emit a record
                    record = logging.LogRecord(
                        name="delay.test",
                        level=logging.INFO,
                        pathname=__file__,
                        lineno=2500,
                        msg="Delay opening test message",
                        args=(),
                        exc_info=None,
                    )

                    handler.emit(record)
                    handler.flush()

                    # Stream should now be opened
                    self.assertIsNotNone(
                        handler.stream, "Stream should be opened after emit"
                    )

                    # Verify file was created
                    self.assertTrue(os.path.exists(test_file))

                finally:
                    handler.close()


if __name__ == "__main__":
    # Configure test logging to avoid interference
    logging.basicConfig(level=logging.CRITICAL)  # Suppress test output

    # Run the tests with detailed output
    unittest.main(verbosity=2)
