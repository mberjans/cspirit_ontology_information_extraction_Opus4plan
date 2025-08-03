#!/usr/bin/env python3
"""Debug test for basic logging functionality."""

import sys
import tempfile
from pathlib import Path

# Add the project to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from aim2_project.aim2_utils.logger_config import LoggerConfig
from aim2_project.aim2_utils.logger_manager import LoggerManager


def test_basic_logging():
    """Test basic logging functionality."""

    # Create temporary log file
    temp_dir = tempfile.mkdtemp(prefix="debug_log_test_")
    log_file = Path(temp_dir) / "debug.log"

    print(f"Log file: {log_file}")

    # Create simple config
    config = LoggerConfig()
    config.load_from_dict(
        {
            "level": "DEBUG",
            "handlers": ["console", "file"],
            "file_path": str(log_file),
            "formatter_type": "json",
            "json_fields": ["timestamp", "level", "logger_name", "message"],
            "json_pretty_print": True,
        }
    )

    print("Config created successfully")

    # Initialize logger manager
    try:
        manager = LoggerManager(config)
        manager.initialize()
        print("Logger manager initialized")
    except Exception as e:
        print(f"Logger manager initialization failed: {e}")
        return False

    # Get a logger and test it
    logger = manager.get_logger("debug_test")
    print("Logger obtained")

    # Test logging
    logger.info("This is a test message")
    logger.warning("This is a warning")
    logger.error("This is an error")

    print("Messages logged")

    # Flush handlers
    for handler in logger.handlers:
        handler.flush()
        print(f"Flushed handler: {handler}")

    # Check if log file exists and has content
    if log_file.exists():
        size = log_file.stat().st_size
        print(f"Log file exists with size: {size} bytes")

        if size > 0:
            with open(log_file, "r") as f:
                content = f.read()
                print(f"Log file content:\n{content}")
                return True
        else:
            print("Log file is empty")
            return False
    else:
        print("Log file does not exist")
        return False


if __name__ == "__main__":
    success = test_basic_logging()
    print(f"Test result: {'PASS' if success else 'FAIL'}")
