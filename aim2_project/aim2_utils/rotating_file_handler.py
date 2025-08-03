"""
Enhanced Rotating File Handler for AIM2 Project

This module provides comprehensive rotating file handler implementation that supports:
- Size-based rotation (RotatingFileHandler)
- Time-based rotation (TimedRotatingFileHandler)
- Combined rotation (both size and time constraints)
- Thread-safe operations
- Proper cleanup and error handling

Classes:
    RotatingFileHandlerError: Custom exception for rotating file handler errors
    EnhancedRotatingFileHandler: Main enhanced rotating file handler class

Dependencies:
    - os: For file operations
    - logging: For standard logging handlers
    - threading: For thread safety
    - datetime: For time-based operations
    - pathlib: For file path operations
    - typing: For type hints
"""

import os
import logging
import logging.handlers
import threading
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional, Union, Literal


class RotatingFileHandlerError(Exception):
    """
    Custom exception for rotating file handler-related errors.

    This exception is raised when rotating file handler operations encounter errors
    during initialization, rotation, or cleanup.

    Args:
        message (str): Error message describing the issue
        cause (Exception, optional): Original exception that caused this error
    """

    def __init__(self, message: str, cause: Exception = None):
        super().__init__(message)
        self.message = message
        self.cause = cause
        if cause:
            self.__cause__ = cause


class EnhancedRotatingFileHandler(logging.Handler):
    """
    Enhanced rotating file handler that supports both size and time-based rotation.

    This handler combines the functionality of RotatingFileHandler and
    TimedRotatingFileHandler, allowing rotation based on:
    - File size only
    - Time intervals only
    - Both size and time (whichever condition is met first)

    Attributes:
        filename (str): The log file path
        rotation_type (str): Type of rotation ("size", "time", or "both")
        max_bytes (int): Maximum file size in bytes (for size-based rotation)
        backup_count (int): Number of backup files to keep
        time_interval (str): Time interval for rotation ("daily", "weekly", "hourly", "midnight")
        time_when (str): When to rotate for daily rotation (e.g., "midnight", "02:00")
        encoding (str): File encoding
        delay (bool): Whether to delay file opening
        utc (bool): Whether to use UTC time for time-based rotation
    """

    # Valid rotation types
    VALID_ROTATION_TYPES = ["size", "time", "both"]

    # Valid time intervals
    VALID_TIME_INTERVALS = ["hourly", "daily", "weekly", "midnight"]

    # Time interval mappings to seconds
    TIME_INTERVALS = {
        "hourly": 3600,
        "daily": 86400,
        "weekly": 604800,
        "midnight": 86400,  # Same as daily but rotates at midnight
    }

    def __init__(
        self,
        filename: str,
        rotation_type: Literal["size", "time", "both"] = "size",
        max_bytes: int = 10 * 1024 * 1024,  # 10MB default
        backup_count: int = 3,
        time_interval: Literal["hourly", "daily", "weekly", "midnight"] = "daily",
        time_when: str = "midnight",
        encoding: Optional[str] = "utf-8",
        delay: bool = False,
        utc: bool = False,
    ):
        """
        Initialize the enhanced rotating file handler.

        Args:
            filename (str): Path to the log file
            rotation_type (str): Type of rotation ("size", "time", or "both")
            max_bytes (int): Maximum file size in bytes for size-based rotation
            backup_count (int): Number of backup files to keep
            time_interval (str): Time interval for rotation
            time_when (str): When to rotate for daily rotation
            encoding (str): File encoding
            delay (bool): Whether to delay file opening
            utc (bool): Whether to use UTC time

        Raises:
            RotatingFileHandlerError: If configuration is invalid
        """
        super().__init__()

        # Validate parameters
        self._validate_parameters(
            rotation_type, max_bytes, backup_count, time_interval, time_when
        )

        self.filename = filename
        self.rotation_type = rotation_type
        self.max_bytes = max_bytes
        self.backup_count = backup_count
        self.time_interval = time_interval
        self.time_when = time_when
        self.encoding = encoding
        self.delay = delay
        self.utc = utc

        # Internal state
        self.stream = None
        self._lock = threading.RLock()
        self._rollover_time = None
        self._last_rollover_time = 0

        # Create directory if it doesn't exist
        self._ensure_directory_exists()

        # Initialize rotation timing
        if self.rotation_type in ["time", "both"]:
            self._calculate_next_rollover_time()

        # Open file if not delayed
        if not delay:
            self._open()

    def _validate_parameters(
        self,
        rotation_type: str,
        max_bytes: int,
        backup_count: int,
        time_interval: str,
        time_when: str,
    ) -> None:
        """
        Validate handler parameters.

        Args:
            rotation_type (str): Type of rotation
            max_bytes (int): Maximum file size
            backup_count (int): Number of backup files
            time_interval (str): Time interval
            time_when (str): When to rotate

        Raises:
            RotatingFileHandlerError: If any parameter is invalid
        """
        if rotation_type not in self.VALID_ROTATION_TYPES:
            raise RotatingFileHandlerError(
                f"Invalid rotation_type '{rotation_type}'. "
                f"Must be one of: {', '.join(self.VALID_ROTATION_TYPES)}"
            )

        if rotation_type in ["size", "both"]:
            if not isinstance(max_bytes, int) or max_bytes < 1024:
                raise RotatingFileHandlerError("max_bytes must be an integer >= 1024")

        if not isinstance(backup_count, int) or backup_count < 0:
            raise RotatingFileHandlerError(
                "backup_count must be a non-negative integer"
            )

        if backup_count > 100:
            raise RotatingFileHandlerError("backup_count cannot exceed 100")

        if rotation_type in ["time", "both"]:
            if time_interval not in self.VALID_TIME_INTERVALS:
                raise RotatingFileHandlerError(
                    f"Invalid time_interval '{time_interval}'. "
                    f"Must be one of: {', '.join(self.VALID_TIME_INTERVALS)}"
                )

            # Validate time_when format for daily/midnight rotation
            if time_interval in ["daily", "midnight"] and time_when != "midnight":
                # Validate HH:MM format
                try:
                    hour, minute = map(int, time_when.split(":"))
                    if not (0 <= hour <= 23 and 0 <= minute <= 59):
                        raise ValueError()
                except (ValueError, IndexError):
                    raise RotatingFileHandlerError(
                        f"Invalid time_when '{time_when}'. "
                        "Must be 'midnight' or 'HH:MM' format (e.g., '02:30')"
                    )

    def _ensure_directory_exists(self) -> None:
        """
        Ensure the log file directory exists.

        Raises:
            RotatingFileHandlerError: If directory cannot be created
        """
        try:
            log_dir = Path(self.filename).parent
            log_dir.mkdir(parents=True, exist_ok=True)
        except Exception as e:
            raise RotatingFileHandlerError(
                f"Failed to create log directory: {str(e)}", e
            )

    def _calculate_next_rollover_time(self) -> None:
        """
        Calculate the next rollover time for time-based rotation.
        """
        now = time.time()

        if self.time_interval == "hourly":
            # Next hour
            dt = datetime.fromtimestamp(now)
            next_hour = dt.replace(minute=0, second=0, microsecond=0) + timedelta(
                hours=1
            )
            self._rollover_time = next_hour.timestamp()

        elif self.time_interval in ["daily", "midnight"]:
            # Next day or specific time
            dt = datetime.fromtimestamp(now)

            if self.time_when == "midnight":
                next_day = dt.replace(
                    hour=0, minute=0, second=0, microsecond=0
                ) + timedelta(days=1)
                self._rollover_time = next_day.timestamp()
            else:
                # Parse HH:MM format
                hour, minute = map(int, self.time_when.split(":"))
                next_time = dt.replace(
                    hour=hour, minute=minute, second=0, microsecond=0
                )

                # If time has passed today, move to tomorrow
                if next_time.timestamp() <= now:
                    next_time += timedelta(days=1)

                self._rollover_time = next_time.timestamp()

        elif self.time_interval == "weekly":
            # Next Monday at midnight
            dt = datetime.fromtimestamp(now)
            days_until_monday = (7 - dt.weekday()) % 7
            if days_until_monday == 0:  # If today is Monday
                days_until_monday = 7

            next_monday = dt.replace(
                hour=0, minute=0, second=0, microsecond=0
            ) + timedelta(days=days_until_monday)
            self._rollover_time = next_monday.timestamp()

    def _open(self) -> None:
        """
        Open the log file for writing.

        Raises:
            RotatingFileHandlerError: If file cannot be opened
        """
        try:
            if self.stream:
                self.stream.close()
                self.stream = None

            self.stream = open(self.filename, "a", encoding=self.encoding)
        except Exception as e:
            raise RotatingFileHandlerError(
                f"Failed to open log file '{self.filename}': {str(e)}", e
            )

    def emit(self, record: logging.LogRecord) -> None:
        """
        Emit a log record, performing rotation if necessary.

        Args:
            record (logging.LogRecord): The log record to emit
        """
        try:
            with self._lock:
                # Check if rotation is needed
                if self._should_rollover(record):
                    self._do_rollover()

                # Emit the record
                if self.stream is None:
                    self._open()

                msg = self.format(record)
                self.stream.write(f"{msg}\n")
                self.flush()

        except Exception:
            self.handleError(record)

    def _should_rollover(self, record: logging.LogRecord) -> bool:
        """
        Determine if rollover should occur.

        Args:
            record (logging.LogRecord): The log record being processed

        Returns:
            bool: True if rollover should occur
        """
        # Size-based check
        size_rollover = False
        if self.rotation_type in ["size", "both"]:
            if self.stream is None:
                self._open()

            # Get current file size
            try:
                self.stream.flush()
                current_size = os.fstat(self.stream.fileno()).st_size

                # Add estimated size of current record
                msg = self.format(record)
                estimated_size = (
                    len(msg.encode(self.encoding or "utf-8")) + 1
                )  # +1 for newline

                size_rollover = (current_size + estimated_size) >= self.max_bytes
            except (OSError, AttributeError):
                # If we can't get size, assume no rollover needed
                size_rollover = False

        # Time-based check
        time_rollover = False
        if self.rotation_type in ["time", "both"]:
            current_time = time.time()
            time_rollover = current_time >= self._rollover_time

        # Return True if any condition is met
        return size_rollover or time_rollover

    def _do_rollover(self) -> None:
        """
        Perform the actual rollover operation.
        """
        try:
            # Close current stream
            if self.stream:
                self.stream.close()
                self.stream = None

            # Generate backup filename with timestamp
            now = datetime.now()
            now.strftime("%Y%m%d_%H%M%S")

            # Rotate existing backup files
            for i in range(self.backup_count - 1, 0, -1):
                old_name = f"{self.filename}.{i}"
                new_name = f"{self.filename}.{i + 1}"

                if os.path.exists(old_name):
                    if os.path.exists(new_name):
                        os.remove(new_name)
                    os.rename(old_name, new_name)

            # Move current file to backup
            if os.path.exists(self.filename):
                backup_name = f"{self.filename}.1"
                if os.path.exists(backup_name):
                    os.remove(backup_name)
                os.rename(self.filename, backup_name)

            # Remove excess backup files
            for i in range(self.backup_count + 1, self.backup_count + 10):
                backup_file = f"{self.filename}.{i}"
                if os.path.exists(backup_file):
                    os.remove(backup_file)
                else:
                    break

            # Update rollover time for time-based rotation
            if self.rotation_type in ["time", "both"]:
                self._last_rollover_time = time.time()
                self._calculate_next_rollover_time()

            # Open new file
            self._open()

        except Exception as e:
            raise RotatingFileHandlerError(f"Failed to perform rollover: {str(e)}", e)

    def close(self) -> None:
        """
        Close the handler and clean up resources.
        """
        try:
            with self._lock:
                if self.stream:
                    self.stream.close()
                    self.stream = None
        finally:
            super().close()

    def flush(self) -> None:
        """
        Flush the stream if it exists.
        """
        if self.stream:
            self.stream.flush()

    def get_rotation_info(self) -> dict:
        """
        Get information about the current rotation configuration.

        Returns:
            dict: Dictionary containing rotation information
        """
        info = {
            "filename": self.filename,
            "rotation_type": self.rotation_type,
            "backup_count": self.backup_count,
            "encoding": self.encoding,
        }

        if self.rotation_type in ["size", "both"]:
            info.update(
                {
                    "max_bytes": self.max_bytes,
                    "max_size_mb": round(self.max_bytes / (1024 * 1024), 2),
                }
            )

        if self.rotation_type in ["time", "both"]:
            info.update(
                {
                    "time_interval": self.time_interval,
                    "time_when": self.time_when,
                    "next_rollover": None,
                }
            )

            if self._rollover_time:
                next_rollover_dt = datetime.fromtimestamp(self._rollover_time)
                info["next_rollover"] = next_rollover_dt.strftime("%Y-%m-%d %H:%M:%S")

        return info

    def __repr__(self) -> str:
        """
        Return string representation of the handler.

        Returns:
            str: String representation
        """
        return (
            f"EnhancedRotatingFileHandler("
            f"filename='{self.filename}', "
            f"rotation_type='{self.rotation_type}', "
            f"max_bytes={self.max_bytes}, "
            f"backup_count={self.backup_count}, "
            f"time_interval='{self.time_interval}'"
            f")"
        )


# Factory function for easier integration
def create_rotating_file_handler(
    filename: str,
    rotation_type: str = "size",
    max_bytes: int = 10 * 1024 * 1024,
    backup_count: int = 3,
    time_interval: str = "daily",
    time_when: str = "midnight",
    encoding: str = "utf-8",
    level: Union[int, str] = logging.INFO,
    formatter: Optional[logging.Formatter] = None,
) -> EnhancedRotatingFileHandler:
    """
    Factory function to create and configure an enhanced rotating file handler.

    Args:
        filename (str): Path to the log file
        rotation_type (str): Type of rotation ("size", "time", or "both")
        max_bytes (int): Maximum file size in bytes
        backup_count (int): Number of backup files to keep
        time_interval (str): Time interval for rotation
        time_when (str): When to rotate for daily rotation
        encoding (str): File encoding
        level (Union[int, str]): Logging level
        formatter (Optional[logging.Formatter]): Log formatter

    Returns:
        EnhancedRotatingFileHandler: Configured handler instance

    Raises:
        RotatingFileHandlerError: If handler creation fails
    """
    try:
        handler = EnhancedRotatingFileHandler(
            filename=filename,
            rotation_type=rotation_type,
            max_bytes=max_bytes,
            backup_count=backup_count,
            time_interval=time_interval,
            time_when=time_when,
            encoding=encoding,
        )

        # Set level
        if isinstance(level, str):
            level = getattr(logging, level.upper())
        handler.setLevel(level)

        # Set formatter
        if formatter:
            handler.setFormatter(formatter)

        return handler

    except Exception as e:
        raise RotatingFileHandlerError(
            f"Failed to create rotating file handler: {str(e)}", e
        )
