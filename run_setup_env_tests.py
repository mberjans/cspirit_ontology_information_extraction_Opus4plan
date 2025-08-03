#!/usr/bin/env python3
"""
Test runner script for setup_env.py validation.

This script provides a convenient way to run the comprehensive test suite
for the setup_env.py virtual environment setup script. It includes options
for running different categories of tests and generating detailed reports.

Usage:
    python run_setup_env_tests.py [options]

Options:
    --quick             Run only fast unit tests
    --full              Run all tests including integration and slow tests
    --integration       Run only integration tests
    --cross-platform    Run cross-platform compatibility tests
    --coverage          Generate coverage report
    --html-report       Generate HTML test report
    --verbose           Enable verbose output
    --help              Show this help message

Examples:
    python run_setup_env_tests.py --quick
    python run_setup_env_tests.py --full --coverage
    python run_setup_env_tests.py --integration --verbose
"""

import argparse
import os
import subprocess
import sys
from pathlib import Path
from typing import List, Optional


class SetupEnvTestRunner:
    """Test runner for setup_env.py validation."""
    
    def __init__(self, project_root: Path):
        self.project_root = project_root
        self.test_file = project_root / "tests" / "test_setup_env.py"
        
        # Verify test file exists
        if not self.test_file.exists():
            raise FileNotFoundError(f"Test file not found: {self.test_file}")
    
    def run_tests(
        self,
        test_categories: List[str] = None,
        verbose: bool = False,
        coverage: bool = False,
        html_report: bool = False,
        extra_args: List[str] = None
    ) -> int:
        """
        Run the setup_env tests with specified options.
        
        Args:
            test_categories: List of test category markers to run
            verbose: Enable verbose output
            coverage: Generate coverage report
            html_report: Generate HTML test report
            extra_args: Additional pytest arguments
        
        Returns:
            Exit code from pytest
        """
        cmd = ["python", "-m", "pytest"]
        
        # Add test file
        cmd.append(str(self.test_file))
        
        # Add test category markers
        if test_categories:
            for category in test_categories:
                cmd.extend(["-m", category])
        
        # Add verbosity
        if verbose:
            cmd.append("-v")
        else:
            cmd.append("-q")
        
        # Add coverage
        if coverage:
            cmd.extend([
                "--cov=setup_env",
                "--cov-report=term-missing",
                "--cov-report=xml"
            ])
            
            if html_report:
                cmd.extend(["--cov-report=html"])
        
        # Add HTML report for test results
        if html_report:
            cmd.extend([
                "--html=test_reports/setup_env_report.html",
                "--self-contained-html"
            ])
        
        # Add extra arguments
        if extra_args:
            cmd.extend(extra_args)
        
        # Ensure test reports directory exists
        if html_report:
            reports_dir = self.project_root / "test_reports"
            reports_dir.mkdir(exist_ok=True)
        
        print(f"Running command: {' '.join(cmd)}")
        print(f"Working directory: {self.project_root}")
        print("-" * 70)
        
        # Run pytest
        result = subprocess.run(cmd, cwd=self.project_root)
        
        return result.returncode
    
    def check_dependencies(self) -> bool:
        """Check that required test dependencies are available."""
        required_packages = [
            "pytest",
            "pytest-mock", 
            "pytest-cov",
            "pytest-html",
            "pytest-timeout"
        ]
        
        missing_packages = []
        
        for package in required_packages:
            try:
                __import__(package.replace("-", "_"))
            except ImportError:
                missing_packages.append(package)
        
        if missing_packages:
            print("Missing required test dependencies:")
            for package in missing_packages:
                print(f"  - {package}")
            print("\nInstall missing dependencies with:")
            print(f"  pip install {' '.join(missing_packages)}")
            return False
        
        return True
    
    def print_test_summary(self):
        """Print a summary of available test categories."""
        print("Available test categories for setup_env.py:")
        print()
        print("  unit              - Unit tests for individual components")
        print("  integration       - Integration tests for complete workflows")
        print("  cross_platform    - Cross-platform compatibility tests")
        print("  slow              - Tests that take longer to run")
        print("  mock_external     - Tests that mock external dependencies")
        print("  timeout           - Tests that verify timeout behavior")
        print()
        print("Test scenarios covered:")
        print("  ✓ Virtual environment creation")
        print("  ✓ Dependency installation")
        print("  ✓ Python version checking")
        print("  ✓ Cross-platform compatibility")
        print("  ✓ Error handling and edge cases")
        print("  ✓ Command-line argument parsing")
        print("  ✓ Progress indicators and user feedback")
        print("  ✓ Timeout and performance handling")
        print("  ✓ Mock testing for external dependencies")


def parse_arguments() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Test runner for setup_env.py validation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_setup_env_tests.py --quick
  python run_setup_env_tests.py --full --coverage --html-report
  python run_setup_env_tests.py --integration --verbose
  python run_setup_env_tests.py --cross-platform
        """
    )
    
    # Test category options
    test_group = parser.add_mutually_exclusive_group()
    test_group.add_argument(
        "--quick",
        action="store_true",
        help="Run only fast unit tests (excludes slow and integration tests)"
    )
    test_group.add_argument(
        "--full",
        action="store_true",
        help="Run all tests including integration and slow tests"
    )
    test_group.add_argument(
        "--integration",
        action="store_true",
        help="Run only integration tests"
    )
    test_group.add_argument(
        "--cross-platform",
        action="store_true",
        help="Run cross-platform compatibility tests"
    )
    test_group.add_argument(
        "--unit",
        action="store_true",
        help="Run only unit tests"
    )
    
    # Output options
    parser.add_argument(
        "--coverage",
        action="store_true",
        help="Generate coverage report"
    )
    
    parser.add_argument(
        "--html-report",
        action="store_true",
        help="Generate HTML test report"
    )
    
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose output"
    )
    
    # Utility options
    parser.add_argument(
        "--list-tests",
        action="store_true",
        help="List available test categories and exit"
    )
    
    parser.add_argument(
        "--check-deps",
        action="store_true",
        help="Check test dependencies and exit"
    )
    
    # Additional pytest arguments
    parser.add_argument(
        "--pytest-args",
        nargs=argparse.REMAINDER,
        help="Additional arguments to pass to pytest"
    )
    
    return parser.parse_args()


def main():
    """Main entry point."""
    try:
        args = parse_arguments()
        
        # Get project root
        project_root = Path(__file__).parent.resolve()
        
        # Create test runner
        runner = SetupEnvTestRunner(project_root)
        
        # Handle utility options
        if args.list_tests:
            runner.print_test_summary()
            return 0
        
        if args.check_deps:
            if runner.check_dependencies():
                print("All test dependencies are available.")
                return 0
            else:
                return 1
        
        # Check dependencies before running tests
        if not runner.check_dependencies():
            print("\nError: Missing test dependencies. Use --check-deps for details.")
            return 1
        
        # Determine test categories to run
        test_categories = []
        
        if args.quick:
            test_categories = ["unit", "not slow", "not integration"]
        elif args.full:
            test_categories = []  # Run all tests
        elif args.integration:
            test_categories = ["integration"]
        elif args.cross_platform:
            test_categories = ["cross_platform"]
        elif args.unit:
            test_categories = ["unit"]
        else:
            # Default: run unit tests only
            test_categories = ["unit"]
        
        # Run tests
        print("Setup Environment Test Suite")
        print("=" * 70)
        
        if test_categories:
            print(f"Running test categories: {', '.join(test_categories)}")
        else:
            print("Running all tests")
        
        print()
        
        exit_code = runner.run_tests(
            test_categories=test_categories,
            verbose=args.verbose,
            coverage=args.coverage,
            html_report=args.html_report,
            extra_args=args.pytest_args or []
        )
        
        print("-" * 70)
        
        if exit_code == 0:
            print("✓ All tests passed!")
            
            if args.coverage:
                print("\nCoverage report generated.")
            
            if args.html_report:
                html_report_path = project_root / "test_reports" / "setup_env_report.html"
                print(f"\nHTML report generated: {html_report_path}")
        else:
            print("✗ Some tests failed.")
            print("\nTip: Use --verbose for more detailed output")
        
        return exit_code
        
    except KeyboardInterrupt:
        print("\nTest execution interrupted by user.")
        return 130
    except Exception as e:
        print(f"\nError running tests: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())