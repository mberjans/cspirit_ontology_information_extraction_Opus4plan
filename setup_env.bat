@echo off
REM AIM2 Project Virtual Environment Setup Script - Windows Batch Wrapper
REM This batch file provides a convenient way to run the setup script on Windows

echo AIM2 Project Virtual Environment Setup
echo ========================================

REM Check if Python is available
python --version >nul 2>&1
if errorlevel 1 (
    echo Error: Python is not installed or not in PATH
    echo Please install Python 3.8+ and try again
    pause
    exit /b 1
)

REM Run the setup script with all arguments passed through
python "%~dp0setup_env.py" %*

REM Pause to show results if run by double-clicking
if "%1"=="" pause
