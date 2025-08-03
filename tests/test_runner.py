#!/usr/bin/env python3
"""
Test runner for setup.py functionality.

This script provides a convenient way to run different categories of tests
for setup.py validation and package functionality.
"""

import sys
import subprocess
import argparse
from pathlib import Path


def run_command(cmd, description):
    """Run a command and report results."""
    print(f"\n{'='*60}")
    print(f"Running: {description}")
    print(f"Command: {' '.join(cmd)}")
    print("=" * 60)

    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        print("✓ SUCCESS")
        if result.stdout:
            print("STDOUT:")
            print(result.stdout)
        return True
    except subprocess.CalledProcessError as e:
        print("✗ FAILED")
        print(f"Return code: {e.returncode}")
        if e.stdout:
            print("STDOUT:")
            print(e.stdout)
        if e.stderr:
            print("STDERR:")
            print(e.stderr)
        return False


def main():
    """Main test runner function."""
    parser = argparse.ArgumentParser(description="Run setup.py tests")
    parser.add_argument(
        "--category",
        choices=["unit", "integration", "all", "setup", "dependencies", "packaging"],
        default="all",
        help="Category of tests to run",
    )
    parser.add_argument("--fast", action="store_true", help="Skip slow tests")
    parser.add_argument(
        "--coverage", action="store_true", help="Run with coverage reporting"
    )
    parser.add_argument("--verbose", action="store_true", help="Verbose output")

    args = parser.parse_args()

    # Change to project root
    project_root = Path(__file__).parent.parent
    print(f"Running tests from: {project_root}")

    # Base pytest command
    pytest_cmd = [sys.executable, "-m", "pytest"]

    # Add verbosity
    if args.verbose:
        pytest_cmd.append("-v")
    else:
        pytest_cmd.append("-q")

    # Add coverage if requested
    if args.coverage:
        pytest_cmd.extend(["--cov=aim2_project", "--cov-report=term-missing"])

    # Skip slow tests if requested
    if args.fast:
        pytest_cmd.extend(["-m", "not slow"])

    # Determine test path based on category
    if args.category == "unit":
        test_path = "tests/unit/"
    elif args.category == "integration":
        test_path = "tests/integration/"
    elif args.category == "setup":
        test_path = "tests/unit/test_setup.py"
    elif args.category == "dependencies":
        test_path = "tests/unit/test_dependencies.py"
    elif args.category == "packaging":
        test_path = "tests/integration/test_setup_integration.py"
    else:  # all
        test_path = "tests/"

    pytest_cmd.append(test_path)

    # Run the tests
    success = run_command(pytest_cmd, f"pytest {args.category} tests")

    if not success:
        print("\n❌ Tests failed!")
        return 1

    print("\n✅ All tests passed!")

    # Run additional checks if running all tests
    if args.category == "all":
        print("\n" + "=" * 60)
        print("Running additional validation checks...")
        print("=" * 60)

        # Check if setup.py exists and is valid
        setup_py = project_root / "setup.py"
        if setup_py.exists():
            # Validate setup.py syntax
            validate_cmd = [
                sys.executable,
                "-c",
                f"compile(open('{setup_py}').read(), '{setup_py}', 'exec')",
            ]
            run_command(validate_cmd, "Validating setup.py syntax")

            # Try to build source distribution
            sdist_cmd = [sys.executable, "setup.py", "check", "--strict"]
            run_command(sdist_cmd, "Checking setup.py metadata")
        else:
            print("⚠️  setup.py not found - tests are designed for TDD")

    return 0


if __name__ == "__main__":
    sys.exit(main())
