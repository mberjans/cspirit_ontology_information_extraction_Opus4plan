#!/usr/bin/env python3
"""
Error Recovery Test Runner

This script provides a convenient way to run all error recovery tests and demonstrations.
It includes options for running specific test suites, generating reports, and viewing
demonstration outputs.

Usage:
    python run_error_recovery_tests.py [options]

Options:
    --unit              Run unit tests only
    --integration       Run integration tests only
    --demo              Run demonstration script only
    --all               Run all tests and demonstrations (default)
    --verbose           Verbose output
    --report            Generate detailed test report
    --help              Show this help message

Test Suites:
    1. Unit Tests (test_error_recovery.py):
       - Error severity classification tests
       - Recovery strategy selection tests
       - Parser-specific recovery method tests
       - Error context and statistics tests
       - Configuration option tests
       - Edge case and fallback tests

    2. Integration Tests (test_error_recovery_integration.py):
       - Complete error recovery workflow tests
       - Parser-specific recovery integration tests
       - Configuration-driven recovery tests
       - Performance impact tests
       - Comprehensive reporting tests

    3. Demonstration (demo_error_recovery.py):
       - Interactive demonstration of error recovery features
       - Real-world error scenarios and recovery examples
       - Configuration impact examples
       - Statistics and reporting examples

Examples:
    # Run all tests and demonstrations
    python run_error_recovery_tests.py

    # Run only unit tests with verbose output
    python run_error_recovery_tests.py --unit --verbose

    # Run demonstration only
    python run_error_recovery_tests.py --demo

    # Generate detailed test report
    python run_error_recovery_tests.py --report
"""

import sys
import os
import subprocess
import argparse
import json
from datetime import datetime
from pathlib import Path


def run_command(command, verbose=False):
    """Run a command and return the result."""
    if verbose:
        print(f"Running: {' '.join(command)}")
    
    try:
        result = subprocess.run(
            command,
            capture_output=True,
            text=True,
            check=False
        )
        
        if verbose or result.returncode != 0:
            print(f"Exit code: {result.returncode}")
            if result.stdout:
                print("STDOUT:")
                print(result.stdout)
            if result.stderr:
                print("STDERR:")
                print(result.stderr)
        
        return result.returncode == 0, result.stdout, result.stderr
    
    except Exception as e:
        print(f"Error running command: {e}")
        return False, "", str(e)


def run_unit_tests(verbose=False):
    """Run error recovery unit tests."""
    print("\n" + "="*60)
    print("RUNNING ERROR RECOVERY UNIT TESTS")
    print("="*60)
    
    test_file = Path(__file__).parent / "tests" / "unit" / "test_error_recovery.py"
    
    if not test_file.exists():
        print(f"Unit test file not found: {test_file}")
        return False
    
    command = ["python", "-m", "pytest", str(test_file)]
    if verbose:
        command.append("-v")
    command.extend(["-x", "--tb=short"])
    
    success, stdout, stderr = run_command(command, verbose)
    
    if success:
        print("✅ Unit tests PASSED")
    else:
        print("❌ Unit tests FAILED")
    
    return success


def run_integration_tests(verbose=False):
    """Run error recovery integration tests."""
    print("\n" + "="*60)
    print("RUNNING ERROR RECOVERY INTEGRATION TESTS")
    print("="*60)
    
    test_file = Path(__file__).parent / "tests" / "unit" / "test_error_recovery_integration.py"
    
    if not test_file.exists():
        print(f"Integration test file not found: {test_file}")
        return False
    
    command = ["python", "-m", "pytest", str(test_file)]
    if verbose:
        command.append("-v")
    command.extend(["-x", "--tb=short"])
    
    success, stdout, stderr = run_command(command, verbose)
    
    if success:
        print("✅ Integration tests PASSED")
    else:
        print("❌ Integration tests FAILED")
    
    return success


def run_demonstration(verbose=False):
    """Run error recovery demonstration."""
    print("\n" + "="*60)
    print("RUNNING ERROR RECOVERY DEMONSTRATION")
    print("="*60)
    
    demo_file = Path(__file__).parent / "demo_error_recovery.py"
    
    if not demo_file.exists():
        print(f"Demonstration file not found: {demo_file}")
        return False
    
    command = ["python", str(demo_file)]
    success, stdout, stderr = run_command(command, verbose)
    
    if not verbose:
        print(stdout)
    
    if success:
        print("✅ Demonstration completed successfully")
    else:
        print("❌ Demonstration failed")
    
    return success


def generate_test_report():
    """Generate a detailed test report."""
    print("\n" + "="*60)
    print("GENERATING ERROR RECOVERY TEST REPORT")
    print("="*60)
    
    report = {
        "timestamp": datetime.now().isoformat(),
        "test_suites": {},
        "summary": {}
    }
    
    # Run unit tests with detailed output
    unit_test_file = Path(__file__).parent / "tests" / "unit" / "test_error_recovery.py"
    if unit_test_file.exists():
        command = ["python", "-m", "pytest", str(unit_test_file), "-v", "--tb=short", "--json-report", "--json-report-file=unit_test_report.json"]
        success, stdout, stderr = run_command(command)
        
        report["test_suites"]["unit_tests"] = {
            "success": success,
            "file": str(unit_test_file),
            "stdout": stdout[:1000] if stdout else "",  # Limit output size
            "stderr": stderr[:1000] if stderr else ""
        }
    
    # Run integration tests with detailed output
    integration_test_file = Path(__file__).parent / "tests" / "unit" / "test_error_recovery_integration.py"
    if integration_test_file.exists():
        command = ["python", "-m", "pytest", str(integration_test_file), "-v", "--tb=short"]
        success, stdout, stderr = run_command(command)
        
        report["test_suites"]["integration_tests"] = {
            "success": success,
            "file": str(integration_test_file),
            "stdout": stdout[:1000] if stdout else "",
            "stderr": stderr[:1000] if stderr else ""
        }
    
    # Run demonstration
    demo_file = Path(__file__).parent / "demo_error_recovery.py"
    if demo_file.exists():
        command = ["python", str(demo_file)]
        success, stdout, stderr = run_command(command)
        
        report["test_suites"]["demonstration"] = {
            "success": success,
            "file": str(demo_file),
            "stdout": stdout[:2000] if stdout else "",  # Allow more output for demo
            "stderr": stderr[:1000] if stderr else ""
        }
    
    # Generate summary
    total_suites = len(report["test_suites"])
    successful_suites = sum(1 for suite in report["test_suites"].values() if suite["success"])
    
    report["summary"] = {
        "total_suites": total_suites,
        "successful_suites": successful_suites,
        "failed_suites": total_suites - successful_suites,
        "success_rate": (successful_suites / total_suites * 100) if total_suites > 0 else 0
    }
    
    # Save report to file
    report_file = Path(__file__).parent / f"error_recovery_test_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    
    try:
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2)
        
        print(f"📄 Test report saved to: {report_file}")
        
        # Display summary
        print(f"📊 Test Summary:")
        print(f"   Total Test Suites: {report['summary']['total_suites']}")
        print(f"   Successful: {report['summary']['successful_suites']}")
        print(f"   Failed: {report['summary']['failed_suites']}")
        print(f"   Success Rate: {report['summary']['success_rate']:.1f}%")
        
        return report['summary']['failed_suites'] == 0
        
    except Exception as e:
        print(f"❌ Failed to save test report: {e}")
        return False


def print_test_overview():
    """Print an overview of available tests."""
    print("ERROR RECOVERY TEST SUITE OVERVIEW")
    print("="*60)
    print("This test suite provides comprehensive testing for the error recovery")
    print("system implemented in the parser classes.")
    print()
    print("📋 Test Components:")
    print()
    print("1. Unit Tests (test_error_recovery.py):")
    print("   • Error severity classification (WARNING, RECOVERABLE, FATAL)")
    print("   • Recovery strategy selection algorithms")
    print("   • Abstract parser error recovery methods")
    print("   • OWL parser-specific recovery methods")
    print("   • CSV parser-specific recovery methods")
    print("   • JSON-LD parser-specific recovery methods")
    print("   • Error context management and tracking")
    print("   • Error statistics collection")
    print("   • Configuration-driven recovery behavior")
    print("   • Edge cases and fallback scenarios")
    print()
    print("2. Integration Tests (test_error_recovery_integration.py):")
    print("   • Complete error recovery workflows")
    print("   • Parser-specific recovery integration")
    print("   • Configuration impact testing")
    print("   • Performance characteristics")
    print("   • Comprehensive error reporting")
    print("   • Real-world error scenarios")
    print()
    print("3. Demonstration (demo_error_recovery.py):")
    print("   • Interactive examples of error recovery")
    print("   • Real error scenarios and recovery strategies")
    print("   • Configuration impact examples")
    print("   • Statistics and reporting demonstrations")
    print("   • Parser-specific recovery showcases")
    print()
    print("🎯 Key Features Tested:")
    print("   • Error classification system")
    print("   • Recovery strategy selection")
    print("   • Parser-specific recovery methods")
    print("   • Error context tracking")
    print("   • Statistics collection and reporting")
    print("   • Configuration-driven behavior")
    print("   • Performance and scalability")
    print("   • Fallback and edge case handling")


def main():
    """Main test runner function."""
    parser = argparse.ArgumentParser(
        description="Run error recovery tests and demonstrations",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    
    parser.add_argument("--unit", action="store_true",
                       help="Run unit tests only")
    parser.add_argument("--integration", action="store_true",
                       help="Run integration tests only")
    parser.add_argument("--demo", action="store_true",
                       help="Run demonstration only")
    parser.add_argument("--all", action="store_true",
                       help="Run all tests and demonstrations (default)")
    parser.add_argument("--verbose", "-v", action="store_true",
                       help="Verbose output")
    parser.add_argument("--report", action="store_true",
                       help="Generate detailed test report")
    parser.add_argument("--overview", action="store_true",
                       help="Show test overview and exit")
    
    args = parser.parse_args()
    
    # Show overview if requested
    if args.overview:
        print_test_overview()
        return 0
    
    # Default to running all tests if no specific option is given
    if not any([args.unit, args.integration, args.demo, args.report]):
        args.all = True
    
    print("ERROR RECOVERY TEST RUNNER")
    print("="*60)
    print(f"Running at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    success_count = 0
    total_count = 0
    
    # Run tests based on options
    if args.unit or args.all:
        total_count += 1
        if run_unit_tests(args.verbose):
            success_count += 1
    
    if args.integration or args.all:
        total_count += 1
        if run_integration_tests(args.verbose):
            success_count += 1
    
    if args.demo or args.all:
        total_count += 1
        if run_demonstration(args.verbose):
            success_count += 1
    
    if args.report:
        total_count += 1
        if generate_test_report():
            success_count += 1
    
    # Final summary
    print("\n" + "="*60)
    print("TEST RUNNER SUMMARY")
    print("="*60)
    print(f"Total components run: {total_count}")
    print(f"Successful: {success_count}")
    print(f"Failed: {total_count - success_count}")
    
    if success_count == total_count:
        print("🎉 ALL TESTS PASSED!")
        return 0
    else:
        print("❌ SOME TESTS FAILED")
        return 1


if __name__ == "__main__":
    sys.exit(main())