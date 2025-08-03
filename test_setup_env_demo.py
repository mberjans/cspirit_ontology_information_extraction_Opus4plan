#!/usr/bin/env python3
"""
Demonstration script for setup_env.py test suite.

This script demonstrates how to run the comprehensive test suite for the
setup_env.py virtual environment setup script. It shows different test
scenarios and validates that the setup script works correctly.

Usage:
    python test_setup_env_demo.py
"""

import subprocess
import sys
from pathlib import Path


def print_header(title: str):
    """Print a formatted header."""
    print("\n" + "=" * 70)
    print(f" {title}")
    print("=" * 70)


def print_section(title: str):
    """Print a formatted section header."""
    print(f"\n{'-' * 50}")
    print(f" {title}")
    print(f"{'-' * 50}")


def run_command(cmd: list, description: str) -> int:
    """Run a command and return the exit code."""
    print(f"\n▶ {description}")
    print(f"Command: {' '.join(cmd)}")
    print()

    result = subprocess.run(cmd)

    if result.returncode == 0:
        print(f"✓ {description} - SUCCESS")
    else:
        print(f"✗ {description} - FAILED (exit code: {result.returncode})")

    return result.returncode


def main():
    """Main demonstration function."""
    print_header("Setup Environment Test Suite Demonstration")

    project_root = Path(__file__).parent.resolve()
    test_runner = project_root / "run_setup_env_tests.py"
    test_file = project_root / "tests" / "test_setup_env.py"

    # Check if files exist
    if not test_runner.exists():
        print(f"Error: Test runner not found at {test_runner}")
        return 1

    if not test_file.exists():
        print(f"Error: Test file not found at {test_file}")
        return 1

    print(f"Project root: {project_root}")
    print(f"Test runner: {test_runner}")
    print(f"Test file: {test_file}")

    total_tests = 0
    failed_tests = 0

    print_section("1. Check Test Dependencies")
    result = run_command(
        [sys.executable, str(test_runner), "--check-deps"], "Checking test dependencies"
    )

    if result != 0:
        print("\n❌ Test dependencies are missing!")
        print("Please install required dependencies first:")
        print("  pip install pytest pytest-mock pytest-cov pytest-html pytest-timeout")
        return 1

    print_section("2. List Available Test Categories")
    result = run_command(
        [sys.executable, str(test_runner), "--list-tests"],
        "Listing available test categories",
    )
    total_tests += 1
    if result != 0:
        failed_tests += 1

    print_section("3. Run Quick Unit Tests")
    result = run_command(
        [sys.executable, str(test_runner), "--quick", "--verbose"],
        "Running quick unit tests",
    )
    total_tests += 1
    if result != 0:
        failed_tests += 1

    print_section("4. Run Cross-Platform Tests")
    result = run_command(
        [sys.executable, str(test_runner), "--cross-platform", "--verbose"],
        "Running cross-platform compatibility tests",
    )
    total_tests += 1
    if result != 0:
        failed_tests += 1

    print_section("5. Run Unit Tests with Coverage")
    result = run_command(
        [sys.executable, str(test_runner), "--unit", "--coverage", "--verbose"],
        "Running unit tests with coverage report",
    )
    total_tests += 1
    if result != 0:
        failed_tests += 1

    print_section("6. Test Individual Components")

    # Test specific test classes using pytest directly
    test_classes = [
        "TestColors",
        "TestProgressIndicator",
        "TestVirtualEnvSetupInitialization",
        "TestArgumentParsing",
        "TestErrorHandling",
    ]

    for test_class in test_classes:
        result = run_command(
            [
                sys.executable,
                "-m",
                "pytest",
                f"{test_file}::{test_class}",
                "-v",
                "-x",  # Stop on first failure
            ],
            f"Testing {test_class}",
        )
        total_tests += 1
        if result != 0:
            failed_tests += 1

    print_section("7. Validate Setup Script Syntax")
    setup_script = project_root / "setup_env.py"

    if setup_script.exists():
        try:
            with open(setup_script, "r") as f:
                compile(f.read(), str(setup_script), "exec")
            print("✓ setup_env.py syntax validation - SUCCESS")
        except SyntaxError as e:
            print(f"✗ setup_env.py syntax validation - FAILED: {e}")
            failed_tests += 1
        total_tests += 1
    else:
        print(f"⚠ setup_env.py not found at {setup_script}")

    print_section("8. Test Import Functionality")
    try:
        sys.path.insert(0, str(project_root))
        import setup_env

        # Test that key classes can be instantiated
        setup_env.Colors()
        setup_env.ProgressIndicator("Test")

        print("✓ setup_env.py import and basic instantiation - SUCCESS")
    except Exception as e:
        print(f"✗ setup_env.py import failed - ERROR: {e}")
        failed_tests += 1
    total_tests += 1

    # Summary
    print_header("Test Suite Demonstration Summary")

    print(f"Total test categories: {total_tests}")
    print(f"Failed test categories: {failed_tests}")
    print(f"Success rate: {((total_tests - failed_tests) / total_tests * 100):.1f}%")

    if failed_tests == 0:
        print("\n🎉 All test categories completed successfully!")
        print("\nThe setup_env.py test suite is working correctly and provides:")
        print("  ✓ Comprehensive unit test coverage")
        print("  ✓ Cross-platform compatibility testing")
        print("  ✓ Error handling and edge case validation")
        print("  ✓ Integration testing capabilities")
        print("  ✓ Performance and timeout testing")
        print("  ✓ Mock testing for external dependencies")

        print("\nNext steps:")
        print("  1. Run full test suite: python run_setup_env_tests.py --full")
        print(
            "  2. Generate HTML report: python run_setup_env_tests.py --full --html-report"
        )
        print("  3. Test actual setup_env.py: python setup_env.py --help")

        return 0
    else:
        print(f"\n❌ {failed_tests} test categories failed.")
        print("\nTroubleshooting:")
        print("  1. Ensure all test dependencies are installed")
        print("  2. Check that pytest is properly configured")
        print("  3. Verify that setup_env.py is in the project root")
        print("  4. Run individual test categories for more details")

        return 1


if __name__ == "__main__":
    sys.exit(main())
