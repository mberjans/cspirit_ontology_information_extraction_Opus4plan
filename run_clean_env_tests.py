#!/usr/bin/env python3
"""
Test runner script for clean environment installation validation (AIM2-002-10).

This script provides a convenient command-line interface for running comprehensive
clean environment installation tests. It orchestrates the test execution, provides
different test scenarios, and generates detailed reports.

The script extends the existing testing infrastructure and integrates with the
project's Makefile targets while providing additional clean environment specific
testing capabilities.

Usage:
    python run_clean_env_tests.py [options]

Options:
    --quick             Run only essential clean environment tests
    --full              Run all clean environment tests including slow tests
    --venv-only         Test only virtual environment creation
    --install-only      Test only package installation in clean environment
    --cross-platform    Run cross-platform compatibility tests
    --precommit         Test pre-commit hooks setup
    --coverage          Generate coverage report for clean env tests
    --html-report       Generate HTML test report
    --json-output       Save test results to JSON file
    --verbose           Enable verbose output
    --keep-artifacts    Keep test artifacts for debugging
    --parallel          Run compatible tests in parallel
    --timeout SECONDS   Set custom timeout for tests
    --help              Show this help message

Examples:
    python run_clean_env_tests.py --quick
    python run_clean_env_tests.py --full --coverage --html-report
    python run_clean_env_tests.py --venv-only --verbose
    python run_clean_env_tests.py --cross-platform --json-output results.json
"""

import argparse
import json
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import List, Optional, Dict, Any

# Import the clean environment tester
sys.path.insert(0, str(Path(__file__).parent))
try:
    from test_clean_environment_installation import (
        CleanEnvironmentTester,
    )
except ImportError as e:
    print(f"Error importing clean environment tester: {e}")
    print("Make sure test_clean_environment_installation.py is in the same directory")
    sys.exit(1)


class CleanEnvTestRunner:
    """
    Advanced test runner for clean environment installation validation.

    This class orchestrates different types of clean environment tests,
    manages test execution scenarios, and provides comprehensive reporting.
    """

    def __init__(self, project_root: Path, verbose: bool = False):
        self.project_root = project_root
        self.verbose = verbose
        self.start_time = time.time()

        # Test configuration
        self.config = {
            "default_timeout": 1800,  # 30 minutes
            "parallel_workers": 2,
            "retry_attempts": 1,
            "cleanup_on_success": True,
            "preserve_logs": True,
        }

        # Test categories and their descriptions
        self.test_categories = {
            "quick": {
                "description": "Essential clean environment tests (fast)",
                "tests": ["venv_creation", "basic_installation"],
                "timeout_multiplier": 1.0,
            },
            "full": {
                "description": "Complete clean environment test suite",
                "tests": [
                    "venv_creation",
                    "package_installation",
                    "precommit_setup",
                    "cross_platform",
                ],
                "timeout_multiplier": 2.0,
            },
            "venv_only": {
                "description": "Virtual environment creation tests only",
                "tests": ["venv_creation"],
                "timeout_multiplier": 0.5,
            },
            "install_only": {
                "description": "Package installation tests only",
                "tests": ["package_installation"],
                "timeout_multiplier": 1.5,
            },
            "cross_platform": {
                "description": "Cross-platform compatibility tests",
                "tests": ["cross_platform"],
                "timeout_multiplier": 1.0,
            },
            "precommit": {
                "description": "Pre-commit hooks setup tests",
                "tests": ["precommit_setup"],
                "timeout_multiplier": 1.0,
            },
        }

        # Results storage
        self.results = {
            "execution_info": {
                "start_time": time.strftime("%Y-%m-%d %H:%M:%S"),
                "runner_version": "1.0.0",
                "project_root": str(self.project_root),
                "platform_info": self._get_platform_info(),
            },
            "test_runs": [],
            "summary": {
                "total_tests": 0,
                "passed_tests": 0,
                "failed_tests": 0,
                "skipped_tests": 0,
                "total_time": 0,
            },
        }

    def _get_platform_info(self) -> Dict[str, str]:
        """Get platform information for test context."""
        import platform

        return {
            "system": platform.system(),
            "release": platform.release(),
            "version": platform.version(),
            "machine": platform.machine(),
            "processor": platform.processor(),
            "python_version": sys.version,
        }

    def log(self, message: str, level: str = "INFO"):
        """Log a message with timestamp and level."""
        timestamp = time.strftime("%H:%M:%S")
        formatted_message = f"[{timestamp}] [{level}] {message}"

        if self.verbose or level in ["ERROR", "WARNING"]:
            print(formatted_message)

    def check_dependencies(self) -> bool:
        """Check that required dependencies are available."""
        required_modules = ["pytest", "pathlib", "subprocess", "tempfile", "json"]

        missing_modules = []
        for module in required_modules:
            try:
                __import__(module)
            except ImportError:
                missing_modules.append(module)

        if missing_modules:
            self.log(f"Missing required modules: {', '.join(missing_modules)}", "ERROR")
            self.log("Install missing dependencies with: pip install pytest", "ERROR")
            return False

        return True

    def run_pytest_clean_env_tests(
        self,
        test_markers: List[str] = None,
        coverage: bool = False,
        html_report: bool = False,
        parallel: bool = False,
        timeout: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        Run pytest-based clean environment tests.

        This method runs the pytest test suite specifically for clean environment testing.
        """
        self.log("Running pytest clean environment tests")

        result = {
            "success": False,
            "return_code": None,
            "output": "",
            "error": "",
            "test_file": "test_clean_environment_installation.py",
            "execution_time": None,
        }

        try:
            start_time = time.time()

            # Build pytest command
            cmd = [sys.executable, "-m", "pytest"]

            # Add test file
            test_file = self.project_root / "test_clean_environment_installation.py"
            if not test_file.exists():
                raise FileNotFoundError(f"Test file not found: {test_file}")

            cmd.append(str(test_file))

            # Add test markers
            if test_markers:
                for marker in test_markers:
                    cmd.extend(["-m", marker])

            # Add verbosity
            if self.verbose:
                cmd.append("-v")
            else:
                cmd.append("-q")

            # Add coverage
            if coverage:
                cmd.extend(
                    [
                        "--cov=test_clean_environment_installation",
                        "--cov-report=term-missing",
                        "--cov-report=xml:clean_env_coverage.xml",
                    ]
                )

                if html_report:
                    cmd.extend(["--cov-report=html:clean_env_htmlcov"])

            # Add HTML report
            if html_report:
                reports_dir = self.project_root / "test_reports"
                reports_dir.mkdir(exist_ok=True)
                cmd.extend(
                    [
                        "--html=test_reports/clean_env_report.html",
                        "--self-contained-html",
                    ]
                )

            # Add parallel execution
            if parallel:
                cmd.extend(["-n", "auto"])

            # Add timeout
            test_timeout = timeout or self.config["default_timeout"]
            cmd.extend(["--timeout", str(test_timeout)])

            self.log(f"Running command: {' '.join(cmd)}")

            # Execute pytest
            process_result = subprocess.run(
                cmd,
                cwd=self.project_root,
                capture_output=True,
                text=True,
                timeout=test_timeout + 60,  # Extra buffer for pytest overhead
            )

            result["return_code"] = process_result.returncode
            result["output"] = process_result.stdout
            result["error"] = process_result.stderr
            result["execution_time"] = time.time() - start_time
            result["success"] = process_result.returncode == 0

            if result["success"]:
                self.log(
                    f"Pytest tests completed successfully in {result['execution_time']:.2f} seconds"
                )
            else:
                self.log(
                    f"Pytest tests failed with return code {result['return_code']}",
                    "ERROR",
                )
                if result["error"]:
                    self.log(f"Error output: {result['error']}", "ERROR")

        except subprocess.TimeoutExpired:
            result["error"] = f"Tests timed out after {test_timeout} seconds"
            self.log(result["error"], "ERROR")

        except FileNotFoundError as e:
            result["error"] = str(e)
            self.log(result["error"], "ERROR")

        except Exception as e:
            result["error"] = f"Unexpected error: {e}"
            self.log(result["error"], "ERROR")

        return result

    def run_standalone_clean_env_test(
        self, keep_artifacts: bool = False, timeout: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Run standalone clean environment test.

        This method runs the clean environment test as a standalone function
        without pytest, useful for direct integration and debugging.
        """
        self.log("Running standalone clean environment test")

        result = {
            "success": False,
            "test_results": None,
            "execution_time": None,
            "error": None,
        }

        try:
            start_time = time.time()

            # Create tester instance
            tester = CleanEnvironmentTester(self.project_root, verbose=self.verbose)

            # Configure tester
            if keep_artifacts:
                tester.test_config["cleanup_on_failure"] = False

            if timeout:
                tester.test_config["max_install_time"] = timeout

            # Run comprehensive test
            test_results = tester.run_comprehensive_clean_environment_test()

            result["test_results"] = test_results
            result["execution_time"] = time.time() - start_time
            result["success"] = test_results["success"]

            if result["success"]:
                self.log(
                    f"Standalone test completed successfully in {result['execution_time']:.2f} seconds"
                )
            else:
                self.log("Standalone test failed", "ERROR")
                if test_results["summary"]["critical_failures"]:
                    for failure in test_results["summary"]["critical_failures"]:
                        self.log(f"Critical failure: {failure}", "ERROR")

        except Exception as e:
            result["error"] = str(e)
            self.log(f"Standalone test failed with error: {e}", "ERROR")

        return result

    def run_parallel_tests(
        self, test_configurations: List[Dict[str, Any]], max_workers: int = None
    ) -> List[Dict[str, Any]]:
        """
        Run multiple test configurations in parallel.

        This method allows running different test scenarios concurrently
        to reduce overall test execution time.
        """
        if max_workers is None:
            max_workers = self.config["parallel_workers"]

        self.log(
            f"Running {len(test_configurations)} test configurations in parallel (max_workers={max_workers})"
        )

        results = []

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit all test configurations
            future_to_config = {
                executor.submit(self._run_single_test_config, config): config
                for config in test_configurations
            }

            # Collect results as they complete
            for future in as_completed(future_to_config):
                config = future_to_config[future]
                try:
                    result = future.result()
                    result["config"] = config
                    results.append(result)

                    status = "✓" if result["success"] else "✗"
                    self.log(
                        f"{status} Completed test: {config.get('name', 'Unnamed')}"
                    )

                except Exception as e:
                    error_result = {"success": False, "error": str(e), "config": config}
                    results.append(error_result)
                    self.log(
                        f"✗ Failed test: {config.get('name', 'Unnamed')} - {e}", "ERROR"
                    )

        return results

    def _run_single_test_config(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Run a single test configuration."""
        test_type = config.get("type", "pytest")

        if test_type == "pytest":
            return self.run_pytest_clean_env_tests(
                test_markers=config.get("markers", []),
                coverage=config.get("coverage", False),
                html_report=config.get("html_report", False),
                parallel=config.get("parallel", False),
                timeout=config.get("timeout"),
            )
        elif test_type == "standalone":
            return self.run_standalone_clean_env_test(
                keep_artifacts=config.get("keep_artifacts", False),
                timeout=config.get("timeout"),
            )
        else:
            raise ValueError(f"Unknown test type: {test_type}")

    def generate_test_report(
        self, results: List[Dict[str, Any]], output_file: Path = None
    ) -> str:
        """
        Generate a comprehensive test report.

        Returns the report as a string and optionally saves to file.
        """
        report_lines = [
            "AIM2 Clean Environment Installation Test Report",
            "=" * 60,
            "",
            f"Execution Time: {self.results['execution_info']['start_time']}",
            f"Project Root: {self.results['execution_info']['project_root']}",
            f"Platform: {self.results['execution_info']['platform_info']['system']} {self.results['execution_info']['platform_info']['release']}",
            f"Python Version: {self.results['execution_info']['platform_info']['python_version'].split()[0]}",
            "",
            "Test Results Summary:",
            "-" * 30,
        ]

        total_tests = len(results)
        passed_tests = sum(1 for r in results if r["success"])
        failed_tests = total_tests - passed_tests

        report_lines.extend(
            [
                f"Total Tests: {total_tests}",
                f"Passed: {passed_tests} ✓",
                f"Failed: {failed_tests} ✗",
                f"Success Rate: {(passed_tests/total_tests*100):.1f}%",
                "",
            ]
        )

        # Detailed test results
        report_lines.append("Detailed Results:")
        report_lines.append("-" * 20)

        for i, result in enumerate(results, 1):
            status = "✓ PASS" if result["success"] else "✗ FAIL"
            config_name = result.get("config", {}).get("name", f"Test {i}")
            execution_time = result.get("execution_time", 0)

            report_lines.append(
                f"{i:2d}. {status} - {config_name} ({execution_time:.2f}s)"
            )

            if not result["success"] and "error" in result:
                report_lines.append(f"     Error: {result['error']}")

            # Add test-specific details
            if "test_results" in result and result["test_results"]:
                test_results = result["test_results"]
                if "summary" in test_results:
                    summary = test_results["summary"]
                    report_lines.append(
                        f"     Phases: {summary['passed_phases']}/{summary['total_phases']}"
                    )

                    if summary["critical_failures"]:
                        report_lines.append(
                            f"     Critical Failures: {len(summary['critical_failures'])}"
                        )
                        for failure in summary["critical_failures"]:
                            report_lines.append(f"       - {failure}")

        report_lines.extend(["", "Test Categories Executed:", "-" * 25])

        for result in results:
            config = result.get("config", {})
            if "category" in config:
                category_info = self.test_categories.get(config["category"], {})
                report_lines.append(
                    f"- {config['category']}: {category_info.get('description', 'No description')}"
                )

        # Performance metrics
        total_time = time.time() - self.start_time
        report_lines.extend(
            [
                "",
                "Performance Metrics:",
                "-" * 20,
                f"Total Execution Time: {total_time:.2f} seconds",
                f"Average Test Time: {(sum(r.get('execution_time', 0) for r in results) / len(results)):.2f} seconds",
                f"Fastest Test: {min((r.get('execution_time', 0) for r in results if r.get('execution_time')), default=0):.2f} seconds",
                f"Slowest Test: {max((r.get('execution_time', 0) for r in results if r.get('execution_time')), default=0):.2f} seconds",
            ]
        )

        # Recommendations
        if failed_tests > 0:
            report_lines.extend(
                [
                    "",
                    "Recommendations:",
                    "-" * 15,
                    "- Review failed test error messages above",
                    "- Run tests with --verbose for more detailed output",
                    "- Use --keep-artifacts to preserve test environments for debugging",
                    "- Check that all dependencies are properly installed",
                    "- Verify that the project structure is correct",
                ]
            )

        report_content = "\n".join(report_lines)

        # Save to file if requested
        if output_file:
            output_file.write_text(report_content)
            self.log(f"Test report saved to: {output_file}")

        return report_content

    def save_json_results(self, results: List[Dict[str, Any]], output_file: Path):
        """Save test results to JSON file."""
        json_data = {
            "execution_info": self.results["execution_info"],
            "results": results,
            "summary": {
                "total_tests": len(results),
                "passed_tests": sum(1 for r in results if r["success"]),
                "failed_tests": sum(1 for r in results if not r["success"]),
                "total_time": time.time() - self.start_time,
            },
        }

        with open(output_file, "w") as f:
            json.dump(json_data, f, indent=2, default=str)

        self.log(f"JSON results saved to: {output_file}")

    def run_test_category(
        self,
        category: str,
        coverage: bool = False,
        html_report: bool = False,
        parallel: bool = False,
        keep_artifacts: bool = False,
        timeout: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        """
        Run tests for a specific category.

        Returns a list of test results.
        """
        if category not in self.test_categories:
            raise ValueError(
                f"Unknown test category: {category}. Available: {list(self.test_categories.keys())}"
            )

        category_info = self.test_categories[category]
        self.log(f"Running test category: {category} - {category_info['description']}")

        # Calculate timeout
        base_timeout = timeout or self.config["default_timeout"]
        adjusted_timeout = int(base_timeout * category_info["timeout_multiplier"])

        results = []

        # For most categories, run pytest tests
        if category in [
            "quick",
            "full",
            "venv_only",
            "install_only",
            "cross_platform",
            "precommit",
        ]:
            # Map category to pytest markers
            marker_mapping = {
                "quick": ["unit"],
                "full": ["integration"],
                "venv_only": ["integration", "not slow"],
                "install_only": ["integration", "slow"],
                "cross_platform": ["cross_platform"],
                "precommit": ["integration"],
            }

            markers = marker_mapping.get(category, [])

            pytest_config = {
                "name": f"pytest_{category}",
                "type": "pytest",
                "category": category,
                "markers": markers,
                "coverage": coverage,
                "html_report": html_report,
                "parallel": parallel
                and category != "venv_only",  # venv tests can't be parallelized
                "timeout": adjusted_timeout,
            }

            pytest_result = self._run_single_test_config(pytest_config)
            pytest_result["config"] = pytest_config
            results.append(pytest_result)

            # For full category, also run standalone test
            if category == "full":
                standalone_config = {
                    "name": f"standalone_{category}",
                    "type": "standalone",
                    "category": category,
                    "keep_artifacts": keep_artifacts,
                    "timeout": adjusted_timeout,
                }

                standalone_result = self._run_single_test_config(standalone_config)
                standalone_result["config"] = standalone_config
                results.append(standalone_result)

        return results


def parse_arguments() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Clean Environment Installation Test Runner for AIM2-002-10",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_clean_env_tests.py --quick
  python run_clean_env_tests.py --full --coverage --html-report
  python run_clean_env_tests.py --venv-only --verbose --keep-artifacts
  python run_clean_env_tests.py --cross-platform --json-output results.json
  python run_clean_env_tests.py --install-only --timeout 1200
        """,
    )

    # Test category options (mutually exclusive)
    category_group = parser.add_mutually_exclusive_group()
    category_group.add_argument(
        "--quick",
        action="store_true",
        help="Run only essential clean environment tests (fast)",
    )
    category_group.add_argument(
        "--full",
        action="store_true",
        help="Run all clean environment tests including slow tests",
    )
    category_group.add_argument(
        "--venv-only",
        action="store_true",
        help="Test only virtual environment creation",
    )
    category_group.add_argument(
        "--install-only",
        action="store_true",
        help="Test only package installation in clean environment",
    )
    category_group.add_argument(
        "--cross-platform",
        action="store_true",
        help="Run cross-platform compatibility tests",
    )
    category_group.add_argument(
        "--precommit", action="store_true", help="Test pre-commit hooks setup"
    )

    # Output and reporting options
    parser.add_argument(
        "--coverage",
        action="store_true",
        help="Generate coverage report for clean env tests",
    )

    parser.add_argument(
        "--html-report", action="store_true", help="Generate HTML test report"
    )

    parser.add_argument(
        "--json-output", type=Path, help="Save test results to JSON file"
    )

    parser.add_argument(
        "--text-report", type=Path, help="Save test report to text file"
    )

    # Execution options
    parser.add_argument(
        "--verbose", "-v", action="store_true", help="Enable verbose output"
    )

    parser.add_argument(
        "--keep-artifacts",
        action="store_true",
        help="Keep test artifacts for debugging",
    )

    parser.add_argument(
        "--parallel", action="store_true", help="Run compatible tests in parallel"
    )

    parser.add_argument(
        "--timeout", type=int, help="Set custom timeout for tests (seconds)"
    )

    # Utility options
    parser.add_argument(
        "--list-categories",
        action="store_true",
        help="List available test categories and exit",
    )

    parser.add_argument(
        "--check-deps", action="store_true", help="Check dependencies and exit"
    )

    parser.add_argument(
        "--project-root",
        type=Path,
        help="Path to project root (default: parent of this script)",
    )

    return parser.parse_args()


def main():
    """Main entry point."""
    try:
        args = parse_arguments()

        # Get project root
        project_root = args.project_root or Path(__file__).parent.resolve()

        # Create test runner
        runner = CleanEnvTestRunner(project_root, verbose=args.verbose)

        # Handle utility options
        if args.list_categories:
            print("Available Test Categories:")
            print("=" * 30)
            for category, info in runner.test_categories.items():
                print(f"  {category:15} - {info['description']}")
                print(f"  {'':15}   Tests: {', '.join(info['tests'])}")
                print(f"  {'':15}   Timeout: {info['timeout_multiplier']}x normal")
                print()
            return 0

        if args.check_deps:
            print("Checking dependencies...")
            if runner.check_dependencies():
                print("✓ All dependencies are available.")
                return 0
            else:
                print("✗ Missing dependencies found.")
                return 1

        # Check dependencies before running tests
        if not runner.check_dependencies():
            print("Error: Missing required dependencies. Use --check-deps for details.")
            return 1

        # Determine test category
        if args.quick:
            category = "quick"
        elif args.full:
            category = "full"
        elif args.venv_only:
            category = "venv_only"
        elif args.install_only:
            category = "install_only"
        elif args.cross_platform:
            category = "cross_platform"
        elif args.precommit:
            category = "precommit"
        else:
            category = "quick"  # Default

        # Print header
        print("AIM2 Clean Environment Installation Test Runner")
        print("=" * 60)
        print(f"Category: {category}")
        print(f"Description: {runner.test_categories[category]['description']}")
        print(f"Project Root: {project_root}")
        print(f"Verbose: {args.verbose}")
        if args.timeout:
            print(f"Timeout: {args.timeout} seconds")
        print()

        # Run tests
        results = runner.run_test_category(
            category=category,
            coverage=args.coverage,
            html_report=args.html_report,
            parallel=args.parallel,
            keep_artifacts=args.keep_artifacts,
            timeout=args.timeout,
        )

        # Generate report
        report = runner.generate_test_report(results, args.text_report)

        # Save JSON results if requested
        if args.json_output:
            runner.save_json_results(results, args.json_output)

        # Print summary
        print()
        print("Test Execution Summary:")
        print("-" * 30)

        total_tests = len(results)
        passed_tests = sum(1 for r in results if r["success"])
        failed_tests = total_tests - passed_tests

        print(f"Total Tests: {total_tests}")
        print(f"Passed: {passed_tests} ✓")
        print(f"Failed: {failed_tests} ✗")
        print(f"Success Rate: {(passed_tests/total_tests*100):.1f}%")

        if args.coverage:
            print("\nCoverage report generated.")

        if args.html_report:
            html_report_path = project_root / "test_reports" / "clean_env_report.html"
            if html_report_path.exists():
                print(f"\nHTML report generated: {html_report_path}")

        if args.json_output:
            print(f"\nJSON results saved: {args.json_output}")

        if args.text_report:
            print(f"\nText report saved: {args.text_report}")

        # Print detailed report if verbose
        if args.verbose:
            print("\nDetailed Report:")
            print("-" * 20)
            print(report)

        # Return appropriate exit code
        return 0 if failed_tests == 0 else 1

    except KeyboardInterrupt:
        print("\nTest execution interrupted by user.")
        return 130
    except Exception as e:
        print(f"\nError running clean environment tests: {e}")
        if args.verbose:
            import traceback

            traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
