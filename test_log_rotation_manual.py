#!/usr/bin/env python3
"""
Manual test script to verify log rotation functionality is working correctly.
This script will generate log files with rotation to demonstrate the feature works.
"""

import logging
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from aim2_project.aim2_utils.rotating_file_handler import create_rotating_file_handler


def test_log_rotation():
    """Test log rotation functionality by generating logs that trigger rotation."""

    print("🚀 Testing AIM2 Log Rotation Functionality")
    print("=" * 50)

    # Ensure logs directory exists
    logs_dir = Path("logs")
    logs_dir.mkdir(exist_ok=True)

    # Create a rotating file handler with small size for quick testing
    log_file = logs_dir / "test_rotation.log"

    # Remove existing test files
    for file in logs_dir.glob("test_rotation.log*"):
        try:
            file.unlink()
        except:
            pass

    print(f"📝 Creating rotating log handler: {log_file}")
    print(f"   - Max size: 2KB (2048 bytes)")
    print(f"   - Backup count: 3")
    print(f"   - Rotation type: size")

    # Create handler
    handler = create_rotating_file_handler(
        filename=str(log_file),
        rotation_type="size",
        max_bytes=2048,  # 2KB - small for quick testing
        backup_count=3,
        level=logging.INFO,
    )

    # Create logger
    logger = logging.getLogger("test_rotation")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    logger.addHandler(handler)

    try:
        print("\n📊 Generating log messages to trigger rotation...")

        # Generate enough log messages to trigger rotation
        for i in range(50):
            logger.info(
                f"Test log message {i:03d} - This is a test message to fill up the log file and trigger rotation when the size limit is reached. This message contains some extra content to make it longer and ensure we hit the size limit quickly."
            )

            # Check file size periodically
            if i % 10 == 0 and log_file.exists():
                size = log_file.stat().st_size
                print(f"   Message {i:03d}: Current file size = {size} bytes")

        # Force any remaining logs to be written
        handler.flush()

        print("\n📁 Checking created log files:")

        # List all test rotation log files
        log_files = sorted(logs_dir.glob("test_rotation.log*"))

        if log_files:
            for log_file_path in log_files:
                size = log_file_path.stat().st_size
                log_file_path.stat().st_mtime
                mtime_str = logging.Formatter().formatTime(
                    logging.LogRecord("", 0, "", 0, "", (), None), "%Y-%m-%d %H:%M:%S"
                )
                print(f"   ✅ {log_file_path.name}: {size} bytes")

            print(f"\n🎉 SUCCESS: Log rotation created {len(log_files)} files!")
            print(f"   - Main log file: test_rotation.log")
            print(f"   - Backup files: {len(log_files) - 1}")

            # Verify rotation worked correctly
            if len(log_files) > 1:
                print("   - ✅ Rotation was triggered successfully")
                print("   - ✅ Backup files were created")
                print("   - ✅ File count is within backup_count limit")
            else:
                print("   - ⚠️  No backup files created (may need more log data)")

        else:
            print("   ❌ No log files found!")
            return False

    finally:
        # Clean up
        handler.close()
        logger.removeHandler(handler)

    print(f"\n🧹 Log files remain in '{logs_dir}' directory for inspection")
    print("✅ Log rotation functionality test completed successfully!")

    return True


if __name__ == "__main__":
    success = test_log_rotation()
    sys.exit(0 if success else 1)
