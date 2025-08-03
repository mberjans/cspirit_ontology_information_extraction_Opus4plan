#!/usr/bin/env python3
"""
Comprehensive test suite for clean environment installation testing (AIM2-002-10).

This module provides comprehensive testing for installation in clean environments,
extending the existing testing infrastructure with isolated environment validation.
It simulates fresh installation scenarios and validates all aspects of the setup
process in isolation.

Test Categories:
- Clean virtual environment creation tests
- Isolated dependency installation tests
- Fresh package installation validation
- Cross-platform clean installation tests
- Installation source validation (local, wheel, PyPI simulation)
- Pre-commit hooks setup verification
- CLI commands functionality in clean environments
- Integration with existing setup_env.py infrastructure

Features:
- Builds upon existing test patterns from tests/test_setup_env.py
- Uses fixtures and utilities from conftest.py
- Follows established code quality standards
- Provides comprehensive error handling and logging
- Supports different test scenarios and environments
"""

import argparse
import json
import os
import platform
import shutil
import subprocess
import sys
import tempfile
import time
import uuid
from pathlib import Path
from typing import Dict, Any
import pytest

# Import the setup_env module for testing
sys.path.insert(0, str(Path(__file__).parent))

# Import test utilities from conftest if available
try:
    from conftest import TestUtilities
except ImportError:
    # Define minimal test utilities if conftest not available
    class TestUtilities:
        @staticmethod
        def validate_package_structure(package_dir: Path) -> bool:
            return (package_dir / "__init__.py").exists()

        @staticmethod
        def parse_requirements_file(requirements_file: Path) -> list:
            if not requirements_file.exists():
                return []
            with open(requirements_file, "r") as f:
                return [
                    line.strip()
                    for line in f.readlines()
                    if line.strip() and not line.strip().startswith("#")
                ]


class CleanEnvironmentTester:
    """
    Comprehensive clean environment testing framework.

    This class provides methods to test installation processes in completely
    isolated environments, simulating fresh system installations.
    """

    def __init__(self, project_root: Path, verbose: bool = False):
        self.project_root = project_root
        self.verbose = verbose
        self.test_id = str(uuid.uuid4())[:8]
        self.temp_dirs = []
        self.created_venvs = []

        # Test configuration
        self.test_config = {
            "timeout_multiplier": 2.0,
            "max_install_time": 600,  # 10 minutes
            "cleanup_on_failure": True,
            "preserve_logs": True,
        }

        self.log_file = None
        self._setup_logging()

    def _setup_logging(self):
        """Setup logging for clean environment tests."""
        logs_dir = self.project_root / "test_logs"
        logs_dir.mkdir(exist_ok=True)

        log_filename = f"clean_env_test_{self.test_id}_{int(time.time())}.log"
        self.log_file = logs_dir / log_filename

        if self.verbose:
            print(f"Clean environment test logging to: {self.log_file}")

    def log(self, message: str, level: str = "INFO"):
        """Log a message to both console (if verbose) and log file."""
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
        log_entry = f"[{timestamp}] [{level}] {message}"

        if self.verbose:
            print(log_entry)

        if self.log_file:
            with open(self.log_file, "a") as f:
                f.write(log_entry + "\n")

    def create_isolated_temp_dir(self, suffix: str = "") -> Path:
        """Create an isolated temporary directory for testing."""
        temp_dir = Path(
            tempfile.mkdtemp(suffix=f"_clean_env_test_{suffix}_{self.test_id}")
        )
        self.temp_dirs.append(temp_dir)
        self.log(f"Created isolated temp directory: {temp_dir}")
        return temp_dir

    def cleanup_temp_dirs(self):
        """Clean up all created temporary directories."""
        for temp_dir in self.temp_dirs:
            if temp_dir.exists():
                try:
                    shutil.rmtree(temp_dir)
                    self.log(f"Cleaned up temp directory: {temp_dir}")
                except OSError as e:
                    self.log(f"Warning: Could not clean up {temp_dir}: {e}", "WARNING")

        for venv_path in self.created_venvs:
            if venv_path.exists():
                try:
                    shutil.rmtree(venv_path)
                    self.log(f"Cleaned up virtual environment: {venv_path}")
                except OSError as e:
                    self.log(
                        f"Warning: Could not clean up venv {venv_path}: {e}", "WARNING"
                    )

    def create_clean_project_copy(self, target_dir: Path) -> Path:
        """
        Create a clean copy of the project for isolated testing.

        This copies essential project files while excluding build artifacts,
        virtual environments, and other temporary files.
        """
        project_copy = target_dir / "project_copy"
        project_copy.mkdir(exist_ok=True)

        # Files and directories to copy
        essential_items = [
            "aim2_project",
            "setup.py",
            "setup.cfg",
            "pyproject.toml",
            "requirements.txt",
            "requirements-dev.txt",
            "README.md",
            "LICENSE",
            "Makefile",
            ".gitignore",
            ".pre-commit-config.yaml",
        ]

        # Files and directories to exclude
        exclude_patterns = {
            "venv",
            ".venv",
            "env",
            ".env",
            "__pycache__",
            ".pytest_cache",
            ".mypy_cache",
            "build",
            "dist",
            "*.egg-info",
            ".git",
            ".tox",
            "htmlcov",
            "test_logs",
            "test_reports",
        }

        for item_name in essential_items:
            src_path = self.project_root / item_name
            if src_path.exists():
                dst_path = project_copy / item_name

                if src_path.is_file():
                    shutil.copy2(src_path, dst_path)
                    self.log(f"Copied file: {item_name}")
                elif src_path.is_dir():
                    shutil.copytree(
                        src_path,
                        dst_path,
                        ignore=shutil.ignore_patterns(*exclude_patterns),
                    )
                    self.log(f"Copied directory: {item_name}")

        self.log(f"Created clean project copy at: {project_copy}")
        return project_copy

    def validate_clean_environment(self, test_dir: Path) -> Dict[str, Any]:
        """
        Validate that the test environment is truly clean.

        Returns a dictionary with validation results.
        """
        validation = {
            "is_clean": True,
            "issues": [],
            "environment_info": {
                "platform": platform.platform(),
                "python_version": sys.version,
                "working_directory": str(test_dir),
                "environment_variables": dict(os.environ),
            },
        }

        # Check for existing virtual environments
        possible_venv_names = ["venv", ".venv", "env", ".env", "virtualenv"]
        for venv_name in possible_venv_names:
            venv_path = test_dir / venv_name
            if venv_path.exists():
                validation["is_clean"] = False
                validation["issues"].append(
                    f"Found existing virtual environment: {venv_path}"
                )

        # Check for Python cache directories
        cache_dirs = list(test_dir.rglob("__pycache__"))
        if cache_dirs:
            validation["is_clean"] = False
            validation["issues"].append(
                f"Found {len(cache_dirs)} Python cache directories"
            )

        # Check for build artifacts
        build_artifacts = ["build", "dist", "*.egg-info"]
        for pattern in build_artifacts:
            artifacts = list(test_dir.glob(pattern))
            if artifacts:
                validation["is_clean"] = False
                validation["issues"].append(
                    f"Found build artifacts: {[str(a) for a in artifacts]}"
                )

        # Check current Python environment for project packages
        try:
            pass

            validation["issues"].append(
                "WARNING: aim2_project already installed in current environment"
            )
        except ImportError:
            pass  # Good, project not installed

        return validation

    def test_clean_venv_creation(
        self, test_dir: Path, venv_name: str = "test_clean_venv"
    ) -> Dict[str, Any]:
        """
        Test creation of a virtual environment in a clean directory.

        Returns test results dictionary.
        """
        self.log(f"Testing clean virtual environment creation: {venv_name}")

        result = {
            "success": False,
            "venv_path": None,
            "creation_time": None,
            "python_version": None,
            "pip_version": None,
            "errors": [],
        }

        venv_path = test_dir / venv_name
        result["venv_path"] = str(venv_path)
        self.created_venvs.append(venv_path)

        try:
            start_time = time.time()

            # Create virtual environment
            create_cmd = [sys.executable, "-m", "venv", str(venv_path)]
            self.log(f"Creating venv with command: {' '.join(create_cmd)}")

            create_result = subprocess.run(
                create_cmd, capture_output=True, text=True, timeout=120, cwd=test_dir
            )

            if create_result.returncode != 0:
                raise subprocess.CalledProcessError(
                    create_result.returncode,
                    create_cmd,
                    create_result.stdout,
                    create_result.stderr,
                )

            creation_time = time.time() - start_time
            result["creation_time"] = creation_time
            self.log(f"Virtual environment created in {creation_time:.2f} seconds")

            # Validate venv structure
            if not venv_path.exists():
                raise Exception("Virtual environment directory not created")

            if not (venv_path / "pyvenv.cfg").exists():
                raise Exception("pyvenv.cfg not found")

            # Get Python executable path
            if platform.system() == "Windows":
                python_exe = venv_path / "Scripts" / "python.exe"
                pip_exe = venv_path / "Scripts" / "pip.exe"
            else:
                python_exe = venv_path / "bin" / "python"
                pip_exe = venv_path / "bin" / "pip"

            if not python_exe.exists():
                raise Exception(f"Python executable not found: {python_exe}")

            # Test Python version
            version_result = subprocess.run(
                [str(python_exe), "--version"],
                capture_output=True,
                text=True,
                timeout=30,
            )

            if version_result.returncode == 0:
                result["python_version"] = version_result.stdout.strip()
                self.log(
                    f"Virtual environment Python version: {result['python_version']}"
                )

            # Test pip version if available
            if pip_exe.exists():
                pip_result = subprocess.run(
                    [str(pip_exe), "--version"],
                    capture_output=True,
                    text=True,
                    timeout=30,
                )

                if pip_result.returncode == 0:
                    result["pip_version"] = pip_result.stdout.strip()
                    self.log(
                        f"Virtual environment pip version: {result['pip_version']}"
                    )

            result["success"] = True

        except subprocess.TimeoutExpired as e:
            error_msg = f"Virtual environment creation timed out: {e}"
            result["errors"].append(error_msg)
            self.log(error_msg, "ERROR")

        except subprocess.CalledProcessError as e:
            error_msg = f"Virtual environment creation failed: {e.stderr}"
            result["errors"].append(error_msg)
            self.log(error_msg, "ERROR")

        except Exception as e:
            error_msg = f"Unexpected error during venv creation: {e}"
            result["errors"].append(error_msg)
            self.log(error_msg, "ERROR")

        return result

    def test_clean_package_installation(
        self, project_copy: Path, venv_path: Path
    ) -> Dict[str, Any]:
        """
        Test clean installation of the project package and dependencies.

        Returns test results dictionary.
        """
        self.log("Testing clean package installation")

        result = {
            "success": False,
            "install_time": None,
            "installed_packages": [],
            "requirements_install": {"success": False, "errors": []},
            "dev_requirements_install": {"success": False, "errors": []},
            "project_install": {"success": False, "errors": []},
            "package_verification": {"success": False, "errors": []},
            "errors": [],
        }

        try:
            start_time = time.time()

            # Get pip executable
            if platform.system() == "Windows":
                pip_exe = venv_path / "Scripts" / "pip.exe"
            else:
                pip_exe = venv_path / "bin" / "pip"

            if not pip_exe.exists():
                raise Exception(f"Pip executable not found: {pip_exe}")

            # Upgrade pip first
            self.log("Upgrading pip...")
            upgrade_result = subprocess.run(
                [str(pip_exe), "install", "--upgrade", "pip", "setuptools", "wheel"],
                capture_output=True,
                text=True,
                timeout=300,
                cwd=project_copy,
            )

            if upgrade_result.returncode != 0:
                self.log(
                    f"Warning: pip upgrade failed: {upgrade_result.stderr}", "WARNING"
                )

            # Install requirements.txt if exists
            requirements_file = project_copy / "requirements.txt"
            if requirements_file.exists():
                self.log("Installing requirements.txt...")
                req_result = subprocess.run(
                    [str(pip_exe), "install", "-r", str(requirements_file)],
                    capture_output=True,
                    text=True,
                    timeout=600,
                    cwd=project_copy,
                )

                if req_result.returncode == 0:
                    result["requirements_install"]["success"] = True
                    self.log("Requirements.txt installation successful")
                else:
                    result["requirements_install"]["errors"].append(req_result.stderr)
                    self.log(
                        f"Requirements.txt installation failed: {req_result.stderr}",
                        "ERROR",
                    )

            # Install development requirements if exists
            dev_requirements_file = project_copy / "requirements-dev.txt"
            if dev_requirements_file.exists():
                self.log("Installing requirements-dev.txt...")
                dev_req_result = subprocess.run(
                    [str(pip_exe), "install", "-r", str(dev_requirements_file)],
                    capture_output=True,
                    text=True,
                    timeout=600,
                    cwd=project_copy,
                )

                if dev_req_result.returncode == 0:
                    result["dev_requirements_install"]["success"] = True
                    self.log("Development requirements installation successful")
                else:
                    result["dev_requirements_install"]["errors"].append(
                        dev_req_result.stderr
                    )
                    self.log(
                        f"Development requirements installation failed: {dev_req_result.stderr}",
                        "ERROR",
                    )

            # Install project package in development mode
            self.log("Installing project package in development mode...")
            project_result = subprocess.run(
                [str(pip_exe), "install", "-e", "."],
                capture_output=True,
                text=True,
                timeout=300,
                cwd=project_copy,
            )

            if project_result.returncode == 0:
                result["project_install"]["success"] = True
                self.log("Project package installation successful")
            else:
                result["project_install"]["errors"].append(project_result.stderr)
                self.log(
                    f"Project package installation failed: {project_result.stderr}",
                    "ERROR",
                )

            # Get list of installed packages
            list_result = subprocess.run(
                [str(pip_exe), "list", "--format=json"],
                capture_output=True,
                text=True,
                timeout=60,
                cwd=project_copy,
            )

            if list_result.returncode == 0:
                try:
                    packages = json.loads(list_result.stdout)
                    result["installed_packages"] = packages
                    self.log(f"Found {len(packages)} installed packages")
                except json.JSONDecodeError:
                    self.log("Could not parse pip list output", "WARNING")

            result["install_time"] = time.time() - start_time
            self.log(
                f"Package installation completed in {result['install_time']:.2f} seconds"
            )

            # Verify package installation
            result["package_verification"] = self._verify_package_installation(
                venv_path, project_copy
            )

            # Determine overall success
            result["success"] = (
                result["project_install"]["success"]
                and result["package_verification"]["success"]
            )

        except subprocess.TimeoutExpired as e:
            error_msg = f"Package installation timed out: {e}"
            result["errors"].append(error_msg)
            self.log(error_msg, "ERROR")

        except Exception as e:
            error_msg = f"Unexpected error during package installation: {e}"
            result["errors"].append(error_msg)
            self.log(error_msg, "ERROR")

        return result

    def _verify_package_installation(
        self, venv_path: Path, project_copy: Path
    ) -> Dict[str, Any]:
        """Verify that packages were installed correctly."""
        verification = {
            "success": False,
            "import_tests": {},
            "cli_tests": {},
            "errors": [],
        }

        try:
            # Get Python executable
            if platform.system() == "Windows":
                python_exe = venv_path / "Scripts" / "python.exe"
            else:
                python_exe = venv_path / "bin" / "python"

            # Test importing main package
            import_tests = [
                "import aim2_project",
                "import numpy",
                "import pandas",
                "import yaml",
                "import click",
            ]

            for import_test in import_tests:
                try:
                    import_result = subprocess.run(
                        [str(python_exe), "-c", import_test],
                        capture_output=True,
                        text=True,
                        timeout=30,
                        cwd=project_copy,
                    )

                    verification["import_tests"][import_test] = {
                        "success": import_result.returncode == 0,
                        "error": import_result.stderr
                        if import_result.returncode != 0
                        else None,
                    }

                    if import_result.returncode == 0:
                        self.log(f"Import test passed: {import_test}")
                    else:
                        self.log(
                            f"Import test failed: {import_test} - {import_result.stderr}",
                            "ERROR",
                        )

                except subprocess.TimeoutExpired:
                    verification["import_tests"][import_test] = {
                        "success": False,
                        "error": "Import test timed out",
                    }
                    self.log(f"Import test timed out: {import_test}", "ERROR")

            # Test CLI commands if they exist
            cli_commands = ["aim2-extract", "aim2-ontology", "aim2-benchmark"]

            for cli_cmd in cli_commands:
                try:
                    cli_result = subprocess.run(
                        [cli_cmd, "--help"],
                        capture_output=True,
                        text=True,
                        timeout=30,
                        cwd=project_copy,
                        env={
                            **os.environ,
                            "PATH": str(
                                venv_path
                                / (
                                    "Scripts"
                                    if platform.system() == "Windows"
                                    else "bin"
                                )
                            )
                            + os.pathsep
                            + os.environ.get("PATH", ""),
                        },
                    )

                    verification["cli_tests"][cli_cmd] = {
                        "success": cli_result.returncode == 0,
                        "error": cli_result.stderr
                        if cli_result.returncode != 0
                        else None,
                    }

                    if cli_result.returncode == 0:
                        self.log(f"CLI test passed: {cli_cmd}")
                    else:
                        self.log(
                            f"CLI test failed: {cli_cmd} - {cli_result.stderr}",
                            "WARNING",
                        )

                except (subprocess.TimeoutExpired, FileNotFoundError):
                    verification["cli_tests"][cli_cmd] = {
                        "success": False,
                        "error": "CLI command not found or timed out",
                    }
                    self.log(f"CLI test failed: {cli_cmd} not found", "WARNING")

            # Determine overall success
            successful_imports = sum(
                1 for test in verification["import_tests"].values() if test["success"]
            )
            total_imports = len(verification["import_tests"])

            verification["success"] = successful_imports >= (
                total_imports * 0.8
            )  # 80% success rate

        except Exception as e:
            error_msg = f"Package verification failed: {e}"
            verification["errors"].append(error_msg)
            self.log(error_msg, "ERROR")

        return verification

    def test_precommit_hooks_setup(
        self, project_copy: Path, venv_path: Path
    ) -> Dict[str, Any]:
        """Test pre-commit hooks setup in clean environment."""
        self.log("Testing pre-commit hooks setup")

        result = {
            "success": False,
            "precommit_config_exists": False,
            "precommit_install": {"success": False, "errors": []},
            "hook_validation": {"success": False, "errors": []},
            "errors": [],
        }

        try:
            precommit_config = project_copy / ".pre-commit-config.yaml"
            result["precommit_config_exists"] = precommit_config.exists()

            if not result["precommit_config_exists"]:
                self.log("No .pre-commit-config.yaml found, skipping pre-commit test")
                return result

            # Get pip executable
            if platform.system() == "Windows":
                pip_exe = venv_path / "Scripts" / "pip.exe"
                precommit_exe = venv_path / "Scripts" / "pre-commit.exe"
            else:
                pip_exe = venv_path / "bin" / "pip"
                precommit_exe = venv_path / "bin" / "pre-commit"

            # Install pre-commit if not already installed
            if not precommit_exe.exists():
                self.log("Installing pre-commit...")
                install_result = subprocess.run(
                    [str(pip_exe), "install", "pre-commit"],
                    capture_output=True,
                    text=True,
                    timeout=300,
                    cwd=project_copy,
                )

                if install_result.returncode != 0:
                    result["precommit_install"]["errors"].append(install_result.stderr)
                    self.log(
                        f"Pre-commit installation failed: {install_result.stderr}",
                        "ERROR",
                    )
                    return result

            result["precommit_install"]["success"] = True
            self.log("Pre-commit installation successful")

            # Install pre-commit hooks
            self.log("Installing pre-commit hooks...")
            hooks_install_result = subprocess.run(
                [str(precommit_exe), "install"],
                capture_output=True,
                text=True,
                timeout=300,
                cwd=project_copy,
            )

            if hooks_install_result.returncode == 0:
                self.log("Pre-commit hooks installation successful")
            else:
                error_msg = f"Pre-commit hooks installation failed: {hooks_install_result.stderr}"
                result["hook_validation"]["errors"].append(error_msg)
                self.log(error_msg, "ERROR")
                return result

            # Validate hooks configuration
            self.log("Validating pre-commit hooks configuration...")
            validate_result = subprocess.run(
                [str(precommit_exe), "validate-config"],
                capture_output=True,
                text=True,
                timeout=60,
                cwd=project_copy,
            )

            if validate_result.returncode == 0:
                result["hook_validation"]["success"] = True
                self.log("Pre-commit hooks validation successful")
            else:
                error_msg = (
                    f"Pre-commit hooks validation failed: {validate_result.stderr}"
                )
                result["hook_validation"]["errors"].append(error_msg)
                self.log(error_msg, "ERROR")

            result["success"] = (
                result["precommit_install"]["success"]
                and result["hook_validation"]["success"]
            )

        except subprocess.TimeoutExpired as e:
            error_msg = f"Pre-commit setup timed out: {e}"
            result["errors"].append(error_msg)
            self.log(error_msg, "ERROR")

        except Exception as e:
            error_msg = f"Unexpected error during pre-commit setup: {e}"
            result["errors"].append(error_msg)
            self.log(error_msg, "ERROR")

        return result

    def test_cross_platform_compatibility(
        self, project_copy: Path, venv_path: Path
    ) -> Dict[str, Any]:
        """Test cross-platform compatibility aspects."""
        self.log("Testing cross-platform compatibility")

        result = {
            "success": False,
            "platform_info": {
                "system": platform.system(),
                "release": platform.release(),
                "version": platform.version(),
                "machine": platform.machine(),
                "processor": platform.processor(),
            },
            "path_tests": {"success": False, "errors": []},
            "executable_tests": {"success": False, "errors": []},
            "script_tests": {"success": False, "errors": []},
            "errors": [],
        }

        try:
            # Test path handling
            self.log(f"Testing on platform: {result['platform_info']['system']}")

            # Test executable paths
            if platform.system() == "Windows":
                expected_python = venv_path / "Scripts" / "python.exe"
                expected_pip = venv_path / "Scripts" / "pip.exe"
            else:
                expected_python = venv_path / "bin" / "python"
                expected_pip = venv_path / "bin" / "pip"

            path_tests_passed = 0
            total_path_tests = 2

            if expected_python.exists():
                path_tests_passed += 1
                self.log(f"Python executable found at expected path: {expected_python}")
            else:
                result["path_tests"]["errors"].append(
                    f"Python executable not found: {expected_python}"
                )
                self.log(f"Python executable not found: {expected_python}", "ERROR")

            if expected_pip.exists():
                path_tests_passed += 1
                self.log(f"Pip executable found at expected path: {expected_pip}")
            else:
                result["path_tests"]["errors"].append(
                    f"Pip executable not found: {expected_pip}"
                )
                self.log(f"Pip executable not found: {expected_pip}", "ERROR")

            result["path_tests"]["success"] = path_tests_passed == total_path_tests

            # Test script execution
            script_tests_passed = 0
            total_script_tests = 1

            # Test setup_env.py script with the clean environment
            setup_env_script = project_copy / "setup_env.py"
            if setup_env_script.exists():
                self.log("Testing setup_env.py script...")
                script_result = subprocess.run(
                    [str(expected_python), str(setup_env_script), "--help"],
                    capture_output=True,
                    text=True,
                    timeout=60,
                    cwd=project_copy,
                )

                if script_result.returncode == 0:
                    script_tests_passed += 1
                    self.log("setup_env.py script test passed")
                else:
                    error_msg = (
                        f"setup_env.py script test failed: {script_result.stderr}"
                    )
                    result["script_tests"]["errors"].append(error_msg)
                    self.log(error_msg, "ERROR")

            result["script_tests"]["success"] = (
                script_tests_passed == total_script_tests
            )

            # Test executable functionality
            exec_tests_passed = 0
            total_exec_tests = 2

            # Test Python version
            version_result = subprocess.run(
                [str(expected_python), "--version"],
                capture_output=True,
                text=True,
                timeout=30,
            )

            if version_result.returncode == 0:
                exec_tests_passed += 1
                self.log(f"Python version test passed: {version_result.stdout.strip()}")
            else:
                error_msg = f"Python version test failed: {version_result.stderr}"
                result["executable_tests"]["errors"].append(error_msg)
                self.log(error_msg, "ERROR")

            # Test pip functionality
            pip_result = subprocess.run(
                [str(expected_pip), "--version"],
                capture_output=True,
                text=True,
                timeout=30,
            )

            if pip_result.returncode == 0:
                exec_tests_passed += 1
                self.log(f"Pip version test passed: {pip_result.stdout.strip()}")
            else:
                error_msg = f"Pip version test failed: {pip_result.stderr}"
                result["executable_tests"]["errors"].append(error_msg)
                self.log(error_msg, "ERROR")

            result["executable_tests"]["success"] = (
                exec_tests_passed == total_exec_tests
            )

            # Determine overall success
            result["success"] = all(
                [
                    result["path_tests"]["success"],
                    result["executable_tests"]["success"],
                    result["script_tests"]["success"],
                ]
            )

        except Exception as e:
            error_msg = f"Cross-platform compatibility test failed: {e}"
            result["errors"].append(error_msg)
            self.log(error_msg, "ERROR")

        return result

    def test_setup_py_functionality(
        self, project_copy: Path, venv_path: Path
    ) -> Dict[str, Any]:
        """Test setup.py validation and functionality in clean environment."""
        self.log("Testing setup.py functionality")

        result = {
            "success": False,
            "setup_check": {"success": False, "errors": []},
            "dev_install": {"success": False, "errors": []},
            "extras_install": {},
            "cli_commands": {"commands_found": 0, "commands_tested": [], "errors": []},
            "package_import": {"success": False, "errors": []},
            "errors": [],
        }

        try:
            # Get python and pip executables
            if platform.system() == "Windows":
                python_exe = venv_path / "Scripts" / "python.exe"
                pip_exe = venv_path / "Scripts" / "pip.exe"
            else:
                python_exe = venv_path / "bin" / "python"
                pip_exe = venv_path / "bin" / "pip"

            # Test 1: Validate setup.py configuration
            self.log("Running python setup.py check...")
            check_result = subprocess.run(
                [str(python_exe), "setup.py", "check"],
                capture_output=True,
                text=True,
                timeout=60,
                cwd=project_copy,
            )

            if check_result.returncode == 0:
                result["setup_check"]["success"] = True
                self.log("setup.py check passed")
            else:
                result["setup_check"]["errors"].append(check_result.stderr)
                self.log(f"setup.py check failed: {check_result.stderr}", "ERROR")

            # Test 2: Development installation
            self.log("Testing development installation...")
            dev_install_result = subprocess.run(
                [str(pip_exe), "install", "-e", "."],
                capture_output=True,
                text=True,
                timeout=300,
                cwd=project_copy,
            )

            if dev_install_result.returncode == 0:
                result["dev_install"]["success"] = True
                self.log("Development installation successful")
            else:
                result["dev_install"]["errors"].append(dev_install_result.stderr)
                self.log(
                    f"Development installation failed: {dev_install_result.stderr}",
                    "ERROR",
                )

            # Test 3: Extras installation
            extras_to_test = ["dev", "test", "lint", "docs"]
            for extra in extras_to_test:
                self.log(f"Testing installation with extras [{extra}]...")
                extra_result = {"success": False, "errors": []}

                extra_install_result = subprocess.run(
                    [str(pip_exe), "install", "-e", f".[{extra}]"],
                    capture_output=True,
                    text=True,
                    timeout=300,
                    cwd=project_copy,
                )

                if extra_install_result.returncode == 0:
                    extra_result["success"] = True
                    self.log(f"Extras [{extra}] installation successful")
                else:
                    extra_result["errors"].append(extra_install_result.stderr)
                    self.log(
                        f"Extras [{extra}] installation failed: {extra_install_result.stderr}",
                        "ERROR",
                    )

                result["extras_install"][extra] = extra_result

            # Test 4: Package import
            self.log("Testing package import...")
            import_result = subprocess.run(
                [
                    str(python_exe),
                    "-c",
                    "import aim2_project; print('Package imported successfully')",
                ],
                capture_output=True,
                text=True,
                timeout=30,
                cwd=project_copy,
            )

            if import_result.returncode == 0:
                result["package_import"]["success"] = True
                self.log("Package import successful")
            else:
                result["package_import"]["errors"].append(import_result.stderr)
                self.log(f"Package import failed: {import_result.stderr}", "ERROR")

            # Test 5: CLI commands availability
            cli_commands = ["aim2-ontology-manager", "aim2-ner-extractor"]

            for cmd in cli_commands:
                self.log(f"Testing CLI command: {cmd}")
                # First check if command is available
                if platform.system() == "Windows":
                    cmd_path = venv_path / "Scripts" / f"{cmd}.exe"
                else:
                    cmd_path = venv_path / "bin" / cmd

                if cmd_path.exists():
                    result["cli_commands"]["commands_found"] += 1
                    result["cli_commands"]["commands_tested"].append(cmd)

                    # Test help command
                    cmd_result = subprocess.run(
                        [str(cmd_path), "--help"],
                        capture_output=True,
                        text=True,
                        timeout=30,
                        cwd=project_copy,
                    )

                    if cmd_result.returncode == 0:
                        self.log(f"CLI command {cmd} works correctly")
                    else:
                        result["cli_commands"]["errors"].append(
                            f"{cmd}: {cmd_result.stderr}"
                        )
                        self.log(
                            f"CLI command {cmd} failed: {cmd_result.stderr}", "ERROR"
                        )
                else:
                    result["cli_commands"]["errors"].append(f"{cmd}: Command not found")
                    self.log(f"CLI command {cmd} not found", "ERROR")

            # Determine overall success
            setup_checks_passed = (
                result["setup_check"]["success"]
                and result["dev_install"]["success"]
                and result["package_import"]["success"]
            )

            extras_partially_successful = any(
                extra_result.get("success", False)
                for extra_result in result["extras_install"].values()
            )

            cli_commands_partially_working = (
                result["cli_commands"]["commands_found"] > 0
            )

            result["success"] = (
                setup_checks_passed
                and extras_partially_successful
                and cli_commands_partially_working
            )

            if result["success"]:
                self.log("setup.py functionality test completed successfully")
            else:
                self.log("setup.py functionality test had issues", "ERROR")

        except Exception as e:
            error_msg = f"setup.py functionality test failed: {e}"
            result["errors"].append(error_msg)
            self.log(error_msg, "ERROR")

        return result

    def run_comprehensive_clean_environment_test(self) -> Dict[str, Any]:
        """
        Run comprehensive clean environment installation test.

        This is the main test method that orchestrates all clean environment tests.
        """
        self.log("Starting comprehensive clean environment installation test")

        overall_result = {
            "success": False,
            "test_id": self.test_id,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "total_time": None,
            "phases": {
                "environment_validation": {},
                "venv_creation": {},
                "package_installation": {},
                "precommit_setup": {},
                "cross_platform_tests": {},
            },
            "summary": {"passed_phases": 0, "total_phases": 5, "critical_failures": []},
        }

        start_time = time.time()

        try:
            # Phase 1: Create isolated test environment
            self.log("Phase 1: Creating isolated test environment")
            test_dir = self.create_isolated_temp_dir("main")
            project_copy = self.create_clean_project_copy(test_dir)

            # Phase 2: Validate clean environment
            self.log("Phase 2: Validating clean environment")
            env_validation = self.validate_clean_environment(test_dir)
            overall_result["phases"]["environment_validation"] = env_validation

            if env_validation["is_clean"]:
                overall_result["summary"]["passed_phases"] += 1
                self.log("Environment validation passed")
            else:
                self.log(
                    f"Environment validation issues: {env_validation['issues']}",
                    "WARNING",
                )

            # Phase 3: Test clean virtual environment creation
            self.log("Phase 3: Testing clean virtual environment creation")
            venv_result = self.test_clean_venv_creation(test_dir, "clean_test_venv")
            overall_result["phases"]["venv_creation"] = venv_result

            if venv_result["success"]:
                overall_result["summary"]["passed_phases"] += 1
                self.log("Virtual environment creation passed")
                venv_path = Path(venv_result["venv_path"])
            else:
                overall_result["summary"]["critical_failures"].append(
                    "Virtual environment creation failed"
                )
                self.log("Virtual environment creation failed - stopping test", "ERROR")
                return overall_result

            # Phase 4: Test clean package installation
            self.log("Phase 4: Testing clean package installation")
            install_result = self.test_clean_package_installation(
                project_copy, venv_path
            )
            overall_result["phases"]["package_installation"] = install_result

            if install_result["success"]:
                overall_result["summary"]["passed_phases"] += 1
                self.log("Package installation passed")
            else:
                overall_result["summary"]["critical_failures"].append(
                    "Package installation failed"
                )
                self.log("Package installation failed", "ERROR")

            # Phase 5: Test pre-commit hooks setup
            self.log("Phase 5: Testing pre-commit hooks setup")
            precommit_result = self.test_precommit_hooks_setup(project_copy, venv_path)
            overall_result["phases"]["precommit_setup"] = precommit_result

            if precommit_result["success"]:
                overall_result["summary"]["passed_phases"] += 1
                self.log("Pre-commit hooks setup passed")
            else:
                self.log("Pre-commit hooks setup failed", "WARNING")

            # Phase 6: Test cross-platform compatibility
            self.log("Phase 6: Testing cross-platform compatibility")
            platform_result = self.test_cross_platform_compatibility(
                project_copy, venv_path
            )
            overall_result["phases"]["cross_platform_tests"] = platform_result

            if platform_result["success"]:
                overall_result["summary"]["passed_phases"] += 1
                self.log("Cross-platform compatibility tests passed")
            else:
                self.log("Cross-platform compatibility tests failed", "WARNING")

            # Determine overall success
            critical_phases_passed = (
                venv_result["success"] and install_result["success"]
            )

            overall_success_rate = (
                overall_result["summary"]["passed_phases"]
                / overall_result["summary"]["total_phases"]
            )

            overall_result["success"] = (
                critical_phases_passed and overall_success_rate >= 0.8
            )

        except Exception as e:
            error_msg = f"Comprehensive test failed with unexpected error: {e}"
            overall_result["summary"]["critical_failures"].append(error_msg)
            self.log(error_msg, "ERROR")

        finally:
            overall_result["total_time"] = time.time() - start_time
            self.log(
                f"Comprehensive test completed in {overall_result['total_time']:.2f} seconds"
            )

            # Clean up if configured to do so
            if self.test_config["cleanup_on_failure"] or overall_result["success"]:
                self.cleanup_temp_dirs()

        return overall_result


class TestCleanEnvironmentInstallation:
    """
    Pytest test class for clean environment installation testing.

    This class provides pytest-compatible test methods that use the
    CleanEnvironmentTester for comprehensive validation.
    """

    @pytest.fixture(scope="class")
    def project_root(self):
        """Get the project root directory."""
        return Path(__file__).parent.resolve()

    @pytest.fixture(scope="class")
    def clean_env_tester(self, project_root):
        """Create a CleanEnvironmentTester instance."""
        return CleanEnvironmentTester(project_root, verbose=True)

    @pytest.mark.integration
    @pytest.mark.slow
    def test_comprehensive_clean_installation(self, clean_env_tester):
        """Test comprehensive clean environment installation."""
        result = clean_env_tester.run_comprehensive_clean_environment_test()

        # Assert overall success
        assert result[
            "success"
        ], f"Clean environment test failed. Critical failures: {result['summary']['critical_failures']}"

        # Assert individual phases
        assert result["phases"]["venv_creation"][
            "success"
        ], "Virtual environment creation failed"
        assert result["phases"]["package_installation"][
            "success"
        ], "Package installation failed"

        # Check that minimum success rate is met
        success_rate = (
            result["summary"]["passed_phases"] / result["summary"]["total_phases"]
        )
        assert success_rate >= 0.8, f"Success rate {success_rate:.2f} below minimum 0.8"

    @pytest.mark.unit
    def test_clean_environment_validation(self, clean_env_tester):
        """Test clean environment validation logic."""
        # Create a temporary directory
        test_dir = clean_env_tester.create_isolated_temp_dir("validation_test")

        try:
            # Test validation of clean directory
            validation = clean_env_tester.validate_clean_environment(test_dir)

            assert isinstance(validation, dict)
            assert "is_clean" in validation
            assert "issues" in validation
            assert "environment_info" in validation

            # Should be clean initially
            assert validation[
                "is_clean"
            ], f"New directory should be clean, but found issues: {validation['issues']}"

        finally:
            clean_env_tester.cleanup_temp_dirs()

    @pytest.mark.integration
    def test_isolated_venv_creation(self, clean_env_tester):
        """Test isolated virtual environment creation."""
        test_dir = clean_env_tester.create_isolated_temp_dir("venv_test")

        try:
            result = clean_env_tester.test_clean_venv_creation(test_dir, "test_venv")

            assert isinstance(result, dict)
            assert "success" in result
            assert "venv_path" in result
            assert "creation_time" in result

            if result["success"]:
                venv_path = Path(result["venv_path"])
                assert venv_path.exists(), "Virtual environment directory should exist"
                assert (venv_path / "pyvenv.cfg").exists(), "pyvenv.cfg should exist"
            else:
                pytest.fail(f"Virtual environment creation failed: {result['errors']}")

        finally:
            clean_env_tester.cleanup_temp_dirs()

    @pytest.mark.integration
    @pytest.mark.slow
    def test_package_installation_in_clean_env(self, clean_env_tester):
        """Test package installation in clean environment."""
        test_dir = clean_env_tester.create_isolated_temp_dir("install_test")

        try:
            # Create project copy and venv
            project_copy = clean_env_tester.create_clean_project_copy(test_dir)
            venv_result = clean_env_tester.test_clean_venv_creation(
                test_dir, "install_test_venv"
            )

            if not venv_result["success"]:
                pytest.skip("Cannot test installation without successful venv creation")

            venv_path = Path(venv_result["venv_path"])
            install_result = clean_env_tester.test_clean_package_installation(
                project_copy, venv_path
            )

            assert isinstance(install_result, dict)
            assert "success" in install_result
            assert "project_install" in install_result
            assert "package_verification" in install_result

            # Project installation should succeed
            assert install_result["project_install"][
                "success"
            ], f"Project installation failed: {install_result['project_install']['errors']}"

        finally:
            clean_env_tester.cleanup_temp_dirs()

    @pytest.mark.cross_platform
    def test_cross_platform_executable_paths(self, clean_env_tester):
        """Test cross-platform executable path handling."""
        test_dir = clean_env_tester.create_isolated_temp_dir("platform_test")

        try:
            venv_result = clean_env_tester.test_clean_venv_creation(
                test_dir, "platform_test_venv"
            )

            if not venv_result["success"]:
                pytest.skip(
                    "Cannot test cross-platform paths without successful venv creation"
                )

            venv_path = Path(venv_result["venv_path"])
            project_copy = clean_env_tester.create_clean_project_copy(test_dir)

            platform_result = clean_env_tester.test_cross_platform_compatibility(
                project_copy, venv_path
            )

            assert isinstance(platform_result, dict)
            assert "success" in platform_result
            assert "platform_info" in platform_result
            assert "path_tests" in platform_result

            # Platform detection should work
            assert platform_result["platform_info"]["system"] in [
                "Windows",
                "Darwin",
                "Linux",
            ]

        finally:
            clean_env_tester.cleanup_temp_dirs()

    @pytest.mark.integration
    @pytest.mark.slow
    def test_setup_py_validation_in_clean_env(self, clean_env_tester):
        """Test setup.py validation and functionality in clean environment."""
        test_dir = clean_env_tester.create_isolated_temp_dir("setup_py_test")

        try:
            # Create project copy and venv
            project_copy = clean_env_tester.create_clean_project_copy(test_dir)
            venv_result = clean_env_tester.test_clean_venv_creation(
                test_dir, "setup_py_test_venv"
            )

            if not venv_result["success"]:
                pytest.skip("Cannot test setup.py without successful venv creation")

            venv_path = Path(venv_result["venv_path"])
            setup_py_result = clean_env_tester.test_setup_py_functionality(
                project_copy, venv_path
            )

            assert isinstance(setup_py_result, dict)
            assert "success" in setup_py_result
            assert "setup_check" in setup_py_result
            assert "dev_install" in setup_py_result
            assert "extras_install" in setup_py_result
            assert "cli_commands" in setup_py_result

            # Setup check should succeed
            assert setup_py_result["setup_check"][
                "success"
            ], f"setup.py check failed: {setup_py_result['setup_check'].get('errors', [])}"

            # Development installation should succeed
            assert setup_py_result["dev_install"][
                "success"
            ], f"Development install failed: {setup_py_result['dev_install'].get('errors', [])}"

            # At least some extras should install successfully
            extras_success = any(
                result.get("success", False)
                for result in setup_py_result["extras_install"].values()
            )
            assert extras_success, "No extras installation succeeded"

            # CLI commands should be available
            assert (
                setup_py_result["cli_commands"]["commands_found"] > 0
            ), "No CLI commands were found after installation"

        finally:
            clean_env_tester.cleanup_temp_dirs()


# Utility functions for standalone usage
def run_clean_environment_test(
    project_root: Path = None, verbose: bool = False
) -> Dict[str, Any]:
    """
    Run clean environment test as a standalone function.

    Args:
        project_root: Path to project root (defaults to current directory parent)
        verbose: Enable verbose logging

    Returns:
        Test results dictionary
    """
    if project_root is None:
        project_root = Path(__file__).parent.resolve()

    tester = CleanEnvironmentTester(project_root, verbose=verbose)

    try:
        return tester.run_comprehensive_clean_environment_test()
    finally:
        if not verbose:  # Only clean up automatically if not verbose
            tester.cleanup_temp_dirs()


if __name__ == "__main__":
    """
    Allow running the clean environment test as a standalone script.
    """
    parser = argparse.ArgumentParser(
        description="Clean Environment Installation Test for AIM2-002-10",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "--verbose", "-v", action="store_true", help="Enable verbose output"
    )

    parser.add_argument(
        "--project-root",
        type=Path,
        help="Path to project root (default: parent of this script)",
    )

    parser.add_argument("--output-json", type=Path, help="Save results to JSON file")

    args = parser.parse_args()

    print("AIM2 Clean Environment Installation Test")
    print("=" * 50)

    result = run_clean_environment_test(
        project_root=args.project_root, verbose=args.verbose
    )

    # Print summary
    print(f"\nTest Results Summary:")
    print(f"Overall Success: {'✓' if result['success'] else '✗'}")
    print(
        f"Passed Phases: {result['summary']['passed_phases']}/{result['summary']['total_phases']}"
    )
    print(f"Total Time: {result['total_time']:.2f} seconds")

    if result["summary"]["critical_failures"]:
        print(f"Critical Failures:")
        for failure in result["summary"]["critical_failures"]:
            print(f"  - {failure}")

    # Save to JSON if requested
    if args.output_json:
        with open(args.output_json, "w") as f:
            json.dump(result, f, indent=2, default=str)
        print(f"\nResults saved to: {args.output_json}")

    # Exit with appropriate code
    sys.exit(0 if result["success"] else 1)
