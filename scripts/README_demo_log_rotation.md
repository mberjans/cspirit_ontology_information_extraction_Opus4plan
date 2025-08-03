# Log Rotation Demo Script

This directory contains a demonstration script that showcases the log rotation functionality implemented in the AIM2 project.

## Overview

The `demo_log_rotation.py` script provides an interactive demonstration of:

- **Size-based rotation**: Files rotate when they reach a specified size limit
- **Time-based rotation**: Files rotate at specified time intervals  
- **Combined rotation**: Files rotate when either size or time conditions are met
- **Configuration integration**: How rotation works with the AIM2 configuration system

## Usage

Run the demo script from the project root:

```bash
python scripts/demo_log_rotation.py
```

The script will:
1. Guide you through interactive demonstrations
2. Create demo log files in the `logs/` directory
3. Show real-time file status and rotation behavior
4. Offer to clean up demo files when complete

## Demo Features

### Size-Based Rotation Demo
- Creates logs with a 2KB size limit
- Generates messages to trigger rotation
- Shows backup file creation and management

### Time-Based Rotation Demo
- Uses simulated time intervals for demonstration
- Shows rotation at specified time periods
- Demonstrates backup file naming and retention

### Combined Rotation Demo  
- Shows both size AND time-based triggers
- Demonstrates flexible rotation conditions
- Illustrates real-world usage scenarios

### Configuration Integration Demo
- Shows integration with AIM2's LoggerConfig system
- Demonstrates configuration-driven rotation setup
- Uses the LoggerFactory for managed logging

## Technical Details

The demo uses:
- `EnhancedRotatingFileHandler` for core rotation functionality
- `JSONFormatter` for structured log output
- `LoggerFactory` and `LoggerConfig` for configuration management
- Small file sizes and message counts for quick demonstration

## Demo Files

All demo files are created in the `logs/` directory:
- `demo_size_rotation.log` (and backups)
- `demo_time_rotation.log` (and backups)  
- `demo_combined_rotation.log` (and backups)
- `demo_config_integration.log` (and backups)

## Requirements

- Python 3.7+
- AIM2 project dependencies installed
- Write access to the `logs/` directory

## Notes

- The script uses small file sizes (1-3KB) for quick demonstration
- Time-based rotation is simulated for demo purposes
- All created files can be automatically cleaned up
- The demo is interactive and requires user input to proceed

## Example Output

```
============================================================
 AIM2 Log Rotation Functionality Demonstration
============================================================

--- Size-Based Rotation Demo ---
Configuration: Size-based rotation (2KB limit)
File: demo_size_rotation.log
Max size: 2048 bytes, Backup count: 3

🚀 Starting size-based rotation demo...
📝 Generating batch 1 of log messages...

📊 File status after batch 1:
  📄 demo_size_rotation.log: 1834 bytes (modified: 10:15:30)
  📋 demo_size_rotation.log.1: 1967 bytes (modified: 10:15:29)
```

This demonstration helps verify that the log rotation implementation works correctly in real scenarios and provides users with a clear understanding of the functionality.
