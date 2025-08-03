#!/bin/bash
# AIM2 Project Virtual Environment Setup Script - Unix Shell Wrapper
# This shell script provides a convenient way to run the setup script on Unix-like systems

set -e  # Exit on any error

echo "AIM2 Project Virtual Environment Setup"
echo "======================================="
echo ""

# Get the directory of this script
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Check if Python is available
if ! command -v python3 &> /dev/null && ! command -v python &> /dev/null; then
    echo "Error: Python is not installed or not in PATH"
    echo "Please install Python 3.8+ and try again"
    exit 1
fi

# Determine Python executable
if command -v python3 &> /dev/null; then
    PYTHON_CMD="python3"
else
    PYTHON_CMD="python"
fi

# Check Python version
PYTHON_VERSION=$($PYTHON_CMD -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
PYTHON_MAJOR=$(echo $PYTHON_VERSION | cut -d. -f1)
PYTHON_MINOR=$(echo $PYTHON_VERSION | cut -d. -f2)

if [ "$PYTHON_MAJOR" -lt 3 ] || ([ "$PYTHON_MAJOR" -eq 3 ] && [ "$PYTHON_MINOR" -lt 8 ]); then
    echo "Error: Python 3.8+ is required (found Python $PYTHON_VERSION)"
    exit 1
fi

echo "Using Python $PYTHON_VERSION at $(which $PYTHON_CMD)"
echo ""

# Run the setup script with all arguments passed through
$PYTHON_CMD "$SCRIPT_DIR/setup_env.py" "$@"