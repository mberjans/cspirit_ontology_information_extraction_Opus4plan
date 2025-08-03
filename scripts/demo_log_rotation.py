#!/usr/bin/env python3
"""
Log Rotation Demonstration Script

This script demonstrates the log rotation functionality implemented in the AIM2 project.
It shows different types of rotation (size-based, time-based, and combined) in action
and provides interactive feedback to the user.

Usage:
    python scripts/demo_log_rotation.py

Features:
- Size-based rotation demo with small file size limits
- Time-based rotation demo with short intervals
- Combined rotation demo
- Real-time progress updates
- Log file inspection and cleanup
- Integration with existing configuration system

Author: AIM2 Project
"""

import os
import sys
import logging
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any

# Add project root to path for imports
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

try:
    from aim2_project.aim2_utils.rotating_file_handler import (
        create_rotating_file_handler,
    )
    from aim2_project.aim2_utils.json_formatter import JSONFormatter
    from aim2_project.aim2_utils.logger_factory import LoggerFactory
    from aim2_project.aim2_utils.logger_config import LoggerConfig
except ImportError as e:
    print(f"Error importing AIM2 modules: {e}")
    print(
        "Please ensure you're running from the project root and all dependencies are installed."
    )
    sys.exit(1)


class LogRotationDemo:
    """
    Main demonstration class for log rotation functionality.

    This class orchestrates different rotation demos and provides user interaction.
    """

    def __init__(self):
        """Initialize the demonstration class."""
        self.demo_dir = project_root / "logs"
        self.demo_dir.mkdir(exist_ok=True)

        # Demo configuration
        self.demo_configs = {
            "size_rotation": {
                "filename": str(self.demo_dir / "demo_size_rotation.log"),
                "rotation_type": "size",
                "max_bytes": 2048,  # 2KB for quick demo (minimum is 1KB)
                "backup_count": 3,
                "description": "Size-based rotation (2KB limit)",
            },
            "time_rotation": {
                "filename": str(self.demo_dir / "demo_time_rotation.log"),
                "rotation_type": "time",
                "time_interval": "hourly",
                "backup_count": 2,
                "description": "Time-based rotation (simulated hourly)",
            },
            "combined_rotation": {
                "filename": str(self.demo_dir / "demo_combined_rotation.log"),
                "rotation_type": "both",
                "max_bytes": 3072,  # 3KB for demo
                "time_interval": "hourly",
                "backup_count": 2,
                "description": "Combined size and time rotation",
            },
        }

        # Tracking for created files
        self.created_files: List[str] = []

    def print_header(self, title: str) -> None:
        """Print a formatted header."""
        print("\n" + "=" * 60)
        print(f" {title}")
        print("=" * 60)

    def print_section(self, title: str) -> None:
        """Print a formatted section header."""
        print(f"\n--- {title} ---")

    def wait_for_user(self, message: str = "Press Enter to continue...") -> None:
        """Wait for user input."""
        input(f"\n{message}")

    def display_file_info(self, file_path: str) -> Dict[str, Any]:
        """Display information about a log file and its backups."""
        file_info = {"main_file": None, "backup_files": []}

        # Check main file
        if os.path.exists(file_path):
            stat = os.stat(file_path)
            file_info["main_file"] = {
                "path": file_path,
                "size": stat.st_size,
                "modified": datetime.fromtimestamp(stat.st_mtime).strftime("%H:%M:%S"),
            }
            print(
                f"  📄 {os.path.basename(file_path)}: {stat.st_size} bytes (modified: {file_info['main_file']['modified']})"
            )

        # Check backup files
        for i in range(1, 10):  # Check up to 9 backup files
            backup_path = f"{file_path}.{i}"
            if os.path.exists(backup_path):
                stat = os.stat(backup_path)
                backup_info = {
                    "path": backup_path,
                    "size": stat.st_size,
                    "modified": datetime.fromtimestamp(stat.st_mtime).strftime(
                        "%H:%M:%S"
                    ),
                }
                file_info["backup_files"].append(backup_info)
                print(
                    f"  📋 {os.path.basename(backup_path)}: {stat.st_size} bytes (modified: {backup_info['modified']})"
                )

        if not file_info["main_file"] and not file_info["backup_files"]:
            print("  (No files found)")

        return file_info

    def generate_log_messages(
        self, logger: logging.Logger, count: int, prefix: str = "Demo"
    ) -> None:
        """Generate a series of log messages."""
        messages = [
            "System initialization started",
            "Loading configuration files",
            "Database connection established",
            "Starting data processing pipeline",
            "Processing batch of entities",
            "Performing ontology mapping",
            "Extracting relationships from text",
            "Validating extracted information",
            "Updating knowledge base",
            "Generating summary report",
            "Operation completed successfully",
            "Cleaning up temporary resources",
        ]

        for i in range(count):
            msg_idx = i % len(messages)
            logger.info(f"{prefix} message {i+1:02d}: {messages[msg_idx]}")

            # Add some variety with different log levels
            if i % 5 == 0:
                logger.warning(f"{prefix} warning {i+1:02d}: Resource usage is high")
            elif i % 7 == 0:
                logger.error(f"{prefix} error {i+1:02d}: Temporary connection timeout")
            elif i % 11 == 0:
                logger.debug(
                    f"{prefix} debug {i+1:02d}: Detailed processing information"
                )

    def demo_size_based_rotation(self) -> None:
        """Demonstrate size-based log rotation."""
        self.print_section("Size-Based Rotation Demo")

        config = self.demo_configs["size_rotation"]
        print(f"Configuration: {config['description']}")
        print(f"File: {os.path.basename(config['filename'])}")
        print(
            f"Max size: {config['max_bytes']} bytes, Backup count: {config['backup_count']}"
        )

        # Clean up any existing files
        self.cleanup_demo_files(config["filename"])

        try:
            # Create logger with rotating handler
            logger = logging.getLogger("size_demo")
            logger.setLevel(logging.DEBUG)

            # Clear any existing handlers
            for handler in logger.handlers[:]:
                logger.removeHandler(handler)

            # Create rotating file handler
            handler = create_rotating_file_handler(
                filename=config["filename"],
                rotation_type=config["rotation_type"],
                max_bytes=config["max_bytes"],
                backup_count=config["backup_count"],
                level=logging.DEBUG,
            )

            # Add JSON formatter for structured logs
            formatter = JSONFormatter()
            handler.setFormatter(formatter)
            logger.addHandler(handler)

            self.created_files.append(config["filename"])

            print("\n🚀 Starting size-based rotation demo...")
            self.wait_for_user("Ready to generate log messages?")

            # Generate messages in batches to show rotation
            for batch in range(3):
                print(f"\n📝 Generating batch {batch + 1} of log messages...")
                self.generate_log_messages(logger, 15, f"SizeBatch{batch+1}")

                print(f"\n📊 File status after batch {batch + 1}:")
                self.display_file_info(config["filename"])

                if batch < 2:
                    self.wait_for_user(
                        f"Batch {batch + 1} complete. Continue to next batch?"
                    )

            print("\n✅ Size-based rotation demo completed!")
            print("Notice how files were rotated when they exceeded 2KB in size.")

        except Exception as e:
            print(f"❌ Error in size-based rotation demo: {e}")
        finally:
            # Clean up logger
            for handler in logger.handlers[:]:
                handler.close()
                logger.removeHandler(handler)

    def demo_time_based_rotation(self) -> None:
        """Demonstrate time-based log rotation (simulated)."""
        self.print_section("Time-Based Rotation Demo")

        config = self.demo_configs["time_rotation"]
        print(f"Configuration: {config['description']}")
        print(f"File: {os.path.basename(config['filename'])}")
        print(
            f"Interval: {config['time_interval']}, Backup count: {config['backup_count']}"
        )
        print(
            "Note: For demo purposes, we'll simulate time-based rotation by manually triggering it."
        )

        # Clean up any existing files
        self.cleanup_demo_files(config["filename"])

        try:
            # Create logger with rotating handler
            logger = logging.getLogger("time_demo")
            logger.setLevel(logging.DEBUG)

            # Clear any existing handlers
            for handler in logger.handlers[:]:
                logger.removeHandler(handler)

            # Create rotating file handler
            handler = create_rotating_file_handler(
                filename=config["filename"],
                rotation_type=config["rotation_type"],
                time_interval=config["time_interval"],
                backup_count=config["backup_count"],
                level=logging.DEBUG,
            )

            # Add formatter
            formatter = logging.Formatter(
                "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)

            self.created_files.append(config["filename"])

            print("\n🚀 Starting time-based rotation demo...")
            self.wait_for_user("Ready to generate log messages?")

            # Simulate time-based rotation by writing logs and manually triggering rotation
            for period in range(3):
                print(f"\n📝 Simulating time period {period + 1}...")
                self.generate_log_messages(logger, 10, f"TimePeriod{period+1}")

                # Simulate time passing by manually triggering rotation
                if hasattr(handler, "_do_rollover"):
                    print("⏰ Simulating time-based rotation...")
                    handler._do_rollover()

                print(f"\n📊 File status after time period {period + 1}:")
                self.display_file_info(config["filename"])

                if period < 2:
                    self.wait_for_user(
                        f"Time period {period + 1} complete. Continue to next period?"
                    )

            print("\n✅ Time-based rotation demo completed!")
            print("Notice how files were rotated at simulated time intervals.")

        except Exception as e:
            print(f"❌ Error in time-based rotation demo: {e}")
        finally:
            # Clean up logger
            for handler in logger.handlers[:]:
                handler.close()
                logger.removeHandler(handler)

    def demo_combined_rotation(self) -> None:
        """Demonstrate combined size and time-based rotation."""
        self.print_section("Combined Rotation Demo")

        config = self.demo_configs["combined_rotation"]
        print(f"Configuration: {config['description']}")
        print(f"File: {os.path.basename(config['filename'])}")
        print(
            f"Max size: {config['max_bytes']} bytes, Interval: {config['time_interval']}"
        )
        print(f"Backup count: {config['backup_count']}")
        print("Rotation will occur when EITHER size limit OR time interval is reached.")

        # Clean up any existing files
        self.cleanup_demo_files(config["filename"])

        try:
            # Create logger with rotating handler
            logger = logging.getLogger("combined_demo")
            logger.setLevel(logging.DEBUG)

            # Clear any existing handlers
            for handler in logger.handlers[:]:
                logger.removeHandler(handler)

            # Create rotating file handler
            handler = create_rotating_file_handler(
                filename=config["filename"],
                rotation_type=config["rotation_type"],
                max_bytes=config["max_bytes"],
                time_interval=config["time_interval"],
                backup_count=config["backup_count"],
                level=logging.DEBUG,
            )

            # Add formatter
            formatter = logging.Formatter(
                "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)

            self.created_files.append(config["filename"])

            print("\n🚀 Starting combined rotation demo...")
            self.wait_for_user("Ready to generate log messages?")

            # Generate messages to trigger both types of rotation
            for round_num in range(2):
                print(
                    f"\n📝 Round {round_num + 1}: Generating messages to trigger rotation..."
                )

                # Generate enough messages to potentially trigger size-based rotation
                self.generate_log_messages(logger, 20, f"CombinedRound{round_num+1}")

                # Also simulate time-based rotation
                if hasattr(handler, "_do_rollover") and round_num == 0:
                    print("⏰ Simulating time-based rotation trigger...")
                    handler._do_rollover()

                print(f"\n📊 File status after round {round_num + 1}:")
                self.display_file_info(config["filename"])

                if round_num < 1:
                    self.wait_for_user(
                        f"Round {round_num + 1} complete. Continue to next round?"
                    )

            print("\n✅ Combined rotation demo completed!")
            print("Notice how rotation was triggered by both size and time conditions.")

        except Exception as e:
            print(f"❌ Error in combined rotation demo: {e}")
        finally:
            # Clean up logger
            for handler in logger.handlers[:]:
                handler.close()
                logger.removeHandler(handler)

    def demo_config_integration(self) -> None:
        """Demonstrate integration with the existing configuration system."""
        self.print_section("Configuration System Integration Demo")

        print(
            "This demo shows how log rotation integrates with the AIM2 configuration system."
        )

        try:
            # Create a custom configuration
            config = LoggerConfig()

            # Configure for file logging with rotation
            log_file = str(self.demo_dir / "demo_config_integration.log")
            config.update_config(
                {
                    "level": "DEBUG",
                    "handlers": ["file"],
                    "file_path": log_file,
                    "max_file_size": "2KB",  # Small size for demo
                    "backup_count": 3,
                    "rotation_type": "size",
                }
            )

            print(f"Configuration:")
            print(f"  File: {os.path.basename(log_file)}")
            print(f"  Max size: 2KB")
            print(f"  Rotation type: size-based")
            print(f"  Backup count: 3")

            # Clean up any existing files
            self.cleanup_demo_files(log_file)
            self.created_files.append(log_file)

            # Create logger factory with this configuration
            factory = LoggerFactory(config)
            factory.initialize()

            # Get a logger
            logger = factory.get_logger("config_integration_demo")

            print("\n🚀 Starting configuration integration demo...")
            self.wait_for_user(
                "Ready to generate log messages using the configured logger?"
            )

            # Generate messages using the configured logger
            for batch in range(3):
                print(f"\n📝 Generating batch {batch + 1} using configured logger...")
                self.generate_log_messages(logger, 12, f"ConfigBatch{batch+1}")

                print(f"\n📊 File status after batch {batch + 1}:")
                self.display_file_info(log_file)

                if batch < 2:
                    self.wait_for_user(
                        f"Batch {batch + 1} complete. Continue to next batch?"
                    )

            print("\n✅ Configuration integration demo completed!")
            print(
                "This shows how easy it is to configure log rotation through the config system."
            )

            # Clean up factory
            factory.cleanup()

        except Exception as e:
            print(f"❌ Error in configuration integration demo: {e}")

    def show_summary(self) -> None:
        """Show a summary of all created files."""
        self.print_section("Demo Summary")

        print("📋 Summary of log rotation demonstration:")
        print(f"✅ Created {len(self.created_files)} demonstration log files")
        print(f"📁 All files are in: {self.demo_dir}")

        print("\n📊 Final file status:")
        for log_file in self.created_files:
            print(f"\n{os.path.basename(log_file)}:")
            self.display_file_info(log_file)

        total_files = 0
        for log_file in self.created_files:
            # Count main file
            if os.path.exists(log_file):
                total_files += 1

            # Count backup files
            for i in range(1, 10):
                backup_path = f"{log_file}.{i}"
                if os.path.exists(backup_path):
                    total_files += 1

        print(f"\n📈 Total files created: {total_files}")

    def cleanup_demo_files(self, base_filename: str) -> None:
        """Clean up demo files for a fresh start."""
        files_to_remove = [base_filename]

        # Add backup files
        for i in range(1, 10):
            files_to_remove.append(f"{base_filename}.{i}")

        for file_path in files_to_remove:
            if os.path.exists(file_path):
                try:
                    os.remove(file_path)
                except OSError:
                    pass  # Ignore errors during cleanup

    def cleanup_all_demo_files(self) -> None:
        """Clean up all demo files."""
        print("\n🧹 Cleaning up demonstration files...")

        for log_file in self.created_files:
            self.cleanup_demo_files(log_file)

        print("✅ Cleanup completed!")

    def run_full_demo(self) -> None:
        """Run the complete log rotation demonstration."""
        self.print_header("AIM2 Log Rotation Functionality Demonstration")

        print(
            "This demonstration will show you how log rotation works in the AIM2 project."
        )
        print("You'll see:")
        print("  • Size-based rotation (files rotate when they reach a size limit)")
        print("  • Time-based rotation (files rotate at time intervals)")
        print("  • Combined rotation (files rotate on size OR time conditions)")
        print("  • Integration with the configuration system")
        print("\nLog files will be created in the 'logs' directory for inspection.")

        self.wait_for_user("Ready to start the demonstration?")

        try:
            # Run all demos
            self.demo_size_based_rotation()
            self.wait_for_user("Ready to continue to time-based rotation demo?")

            self.demo_time_based_rotation()
            self.wait_for_user("Ready to continue to combined rotation demo?")

            self.demo_combined_rotation()
            self.wait_for_user("Ready to see configuration integration demo?")

            self.demo_config_integration()

            # Show summary
            self.show_summary()

            print("\n🎉 Log rotation demonstration completed successfully!")
            print("\nKey takeaways:")
            print("  ✅ Size-based rotation keeps log files under specified size limits")
            print("  ✅ Time-based rotation creates new files at regular intervals")
            print("  ✅ Combined rotation provides flexibility with multiple triggers")
            print("  ✅ Easy integration with the AIM2 configuration system")
            print("  ✅ Automatic backup file management and cleanup")

        except KeyboardInterrupt:
            print("\n\n⚠️  Demo interrupted by user.")
        except Exception as e:
            print(f"\n❌ Unexpected error during demo: {e}")
        finally:
            # Ask about cleanup
            cleanup_choice = (
                input("\n🧹 Would you like to clean up the demo files? (y/n): ")
                .lower()
                .strip()
            )
            if cleanup_choice in ("y", "yes"):
                self.cleanup_all_demo_files()
            else:
                print(f"📁 Demo files preserved in: {self.demo_dir}")
                print("You can inspect them manually or run the cleanup later.")


def main():
    """Main entry point for the demonstration script."""
    try:
        demo = LogRotationDemo()
        demo.run_full_demo()
    except KeyboardInterrupt:
        print("\n\nDemo interrupted. Goodbye!")
    except Exception as e:
        print(f"\nFatal error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
