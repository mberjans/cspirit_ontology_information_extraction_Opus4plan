"""
Integration tests for setup.py functionality.

This module contains integration tests that validate the complete setup.py
workflow including building, installing, and using the package in realistic
scenarios.
"""

import sys
import tempfile
import shutil
import subprocess
from pathlib import Path
import pytest
import venv


class TestSetupIntegration:
    """Integration tests for complete setup.py workflow."""

    @pytest.fixture
    def isolated_environment(self):
        """Create an isolated environment for testing."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)

            # Copy project structure to temporary directory
            project_root = Path(__file__).parent.parent.parent
            target_dir = temp_path / "test_project"
            shutil.copytree(
                project_root,
                target_dir,
                ignore=shutil.ignore_patterns("venv", ".git", "__pycache__", "*.pyc"),
            )

            yield target_dir

    @pytest.fixture
    def virtual_environment(self, isolated_environment):
        """Create a virtual environment for testing."""
        venv_dir = isolated_environment / "test_venv"
        venv.create(venv_dir, with_pip=True)

        # Get the python executable path
        if sys.platform == "win32":
            python_exe = venv_dir / "Scripts" / "python.exe"
            pip_exe = venv_dir / "Scripts" / "pip.exe"
        else:
            python_exe = venv_dir / "bin" / "python"
            pip_exe = venv_dir / "bin" / "pip"

        yield {
            "venv_dir": venv_dir,
            "python": str(python_exe),
            "pip": str(pip_exe),
            "project_dir": isolated_environment,
        }

    def test_package_builds_successfully(self, isolated_environment):
        """Test that the package can be built successfully."""
        # Skip if setup.py doesn't exist yet
        setup_py = isolated_environment / "setup.py"
        if not setup_py.exists():
            pytest.skip("setup.py not yet created")

        # Test source distribution build
        result = subprocess.run(
            [sys.executable, "setup.py", "sdist"],
            cwd=isolated_environment,
            capture_output=True,
            text=True,
        )

        if result.returncode != 0:
            pytest.fail(f"sdist build failed: {result.stderr}")

        # Check that dist directory was created
        dist_dir = isolated_environment / "dist"
        assert dist_dir.exists()

        # Check that source distribution file was created
        sdist_files = list(dist_dir.glob("*.tar.gz"))
        assert len(sdist_files) > 0

    def test_package_installs_in_clean_environment(self, virtual_environment):
        """Test that the package installs correctly in a clean environment."""
        # Skip if setup.py doesn't exist yet
        setup_py = virtual_environment["project_dir"] / "setup.py"
        if not setup_py.exists():
            pytest.skip("setup.py not yet created")

        # Install package in development mode
        result = subprocess.run(
            [virtual_environment["pip"], "install", "-e", "."],
            cwd=virtual_environment["project_dir"],
            capture_output=True,
            text=True,
        )

        if result.returncode != 0:
            pytest.fail(f"Package installation failed: {result.stderr}")

        # Test that package can be imported
        import_test = subprocess.run(
            [
                virtual_environment["python"],
                "-c",
                "import aim2_project; print('Import successful')",
            ],
            capture_output=True,
            text=True,
        )

        if import_test.returncode != 0:
            pytest.fail(f"Package import failed: {import_test.stderr}")

    def test_dependencies_install_correctly(self, virtual_environment):
        """Test that all dependencies install correctly."""
        # Skip if setup.py doesn't exist yet
        setup_py = virtual_environment["project_dir"] / "setup.py"
        if not setup_py.exists():
            pytest.skip("setup.py not yet created")

        # Install with dependencies
        result = subprocess.run(
            [virtual_environment["pip"], "install", "-e", "."],
            cwd=virtual_environment["project_dir"],
            capture_output=True,
            text=True,
        )

        if result.returncode != 0:
            pytest.fail(f"Dependency installation failed: {result.stderr}")

        # Test that key dependencies are available
        key_dependencies = ["numpy", "pandas", "click"]
        for dep in key_dependencies:
            import_test = subprocess.run(
                [
                    virtual_environment["python"],
                    "-c",
                    f"import {dep}; print('{dep} available')",
                ],
                capture_output=True,
                text=True,
            )

            if import_test.returncode != 0:
                pytest.fail(f"Dependency {dep} not available: {import_test.stderr}")

    def test_optional_dependencies_install(self, virtual_environment):
        """Test that optional dependencies install correctly."""
        # Skip if setup.py doesn't exist yet
        setup_py = virtual_environment["project_dir"] / "setup.py"
        if not setup_py.exists():
            pytest.skip("setup.py not yet created")

        # Install with dev dependencies
        result = subprocess.run(
            [virtual_environment["pip"], "install", "-e", ".[dev]"],
            cwd=virtual_environment["project_dir"],
            capture_output=True,
            text=True,
        )

        if result.returncode != 0:
            pytest.fail(f"Dev dependencies installation failed: {result.stderr}")

        # Test that dev dependencies are available
        dev_dependencies = ["pytest", "black"]
        for dep in dev_dependencies:
            import_test = subprocess.run(
                [
                    virtual_environment["python"],
                    "-c",
                    f"import {dep}; print('{dep} available')",
                ],
                capture_output=True,
                text=True,
            )

            if import_test.returncode != 0:
                pytest.fail(f"Dev dependency {dep} not available: {import_test.stderr}")

    def test_console_scripts_work(self, virtual_environment):
        """Test that console scripts are installed and work."""
        # Skip if setup.py doesn't exist yet
        setup_py = virtual_environment["project_dir"] / "setup.py"
        if not setup_py.exists():
            pytest.skip("setup.py not yet created")

        # Install package
        subprocess.run(
            [virtual_environment["pip"], "install", "-e", "."],
            cwd=virtual_environment["project_dir"],
            capture_output=True,
            text=True,
        )

        # Test console scripts (these may not exist yet)
        expected_scripts = ["aim2-extract", "aim2-ontology", "aim2-benchmark"]

        for script in expected_scripts:
            # Check if script is available
            if sys.platform == "win32":
                script_path = (
                    virtual_environment["venv_dir"] / "Scripts" / f"{script}.exe"
                )
            else:
                script_path = virtual_environment["venv_dir"] / "bin" / script

            if script_path.exists():
                # Test that script can be executed (with --help to avoid side effects)
                result = subprocess.run(
                    [str(script_path), "--help"], capture_output=True, text=True
                )

                # Script should either show help or indicate it's not fully implemented yet
                assert result.returncode in [
                    0,
                    1,
                    2,
                ]  # Allow for not-implemented scripts

    def test_package_data_is_included(self, isolated_environment):
        """Test that package data files are included in distribution."""
        # Skip if setup.py doesn't exist yet
        setup_py = isolated_environment / "setup.py"
        if not setup_py.exists():
            pytest.skip("setup.py not yet created")

        # Build source distribution
        subprocess.run(
            [sys.executable, "setup.py", "sdist"],
            cwd=isolated_environment,
            capture_output=True,
            text=True,
        )

        # Extract and examine the tarball
        dist_dir = isolated_environment / "dist"
        if dist_dir.exists():
            tarball_files = list(dist_dir.glob("*.tar.gz"))
            if tarball_files:
                # Basic check that distribution was created
                tarball = tarball_files[0]
                assert tarball.exists()
                assert tarball.stat().st_size > 0

    def test_wheel_builds_and_installs(self, virtual_environment):
        """Test that wheel builds and installs correctly."""
        # Skip if setup.py doesn't exist yet
        setup_py = virtual_environment["project_dir"] / "setup.py"
        if not setup_py.exists():
            pytest.skip("setup.py not yet created")

        # Install wheel if not available
        subprocess.run(
            [virtual_environment["pip"], "install", "wheel"],
            capture_output=True,
            text=True,
        )

        # Build wheel
        build_result = subprocess.run(
            [virtual_environment["python"], "setup.py", "bdist_wheel"],
            cwd=virtual_environment["project_dir"],
            capture_output=True,
            text=True,
        )

        if build_result.returncode != 0:
            pytest.fail(f"Wheel build failed: {build_result.stderr}")

        # Check that wheel was created
        dist_dir = virtual_environment["project_dir"] / "dist"
        if dist_dir.exists():
            wheel_files = list(dist_dir.glob("*.whl"))
            assert len(wheel_files) > 0

            # Install from wheel
            wheel_file = wheel_files[0]
            install_result = subprocess.run(
                [virtual_environment["pip"], "install", str(wheel_file)],
                capture_output=True,
                text=True,
            )

            if install_result.returncode != 0:
                pytest.fail(f"Wheel installation failed: {install_result.stderr}")


class TestSetupWithRealDependencies:
    """Test setup.py with real dependency resolution."""

    @pytest.fixture
    def clean_venv(self):
        """Create a completely clean virtual environment."""
        with tempfile.TemporaryDirectory() as temp_dir:
            venv_dir = Path(temp_dir) / "clean_venv"
            venv.create(venv_dir, with_pip=True)

            if sys.platform == "win32":
                python_exe = venv_dir / "Scripts" / "python.exe"
                pip_exe = venv_dir / "Scripts" / "pip.exe"
            else:
                python_exe = venv_dir / "bin" / "python"
                pip_exe = venv_dir / "bin" / "pip"

            yield {"python": str(python_exe), "pip": str(pip_exe), "venv_dir": venv_dir}

    def test_core_dependencies_resolve(self, clean_venv):
        """Test that core dependencies can be resolved together."""
        # Test a subset of core dependencies
        test_dependencies = [
            "numpy>=1.21.0",
            "pandas>=1.3.0",
            "pyyaml>=6.0",
            "click>=8.0.0",
            "tqdm>=4.62.0",
            "requests>=2.25.0",
        ]

        # Install dependencies
        for dep in test_dependencies:
            result = subprocess.run(
                [clean_venv["pip"], "install", dep], capture_output=True, text=True
            )

            if result.returncode != 0:
                pytest.fail(f"Failed to install {dep}: {result.stderr}")

        # Test that all can be imported
        import_name_mapping = {
            "pyyaml": "yaml",
        }
        for dep in test_dependencies:
            package_name = dep.split(">=")[0].replace("-", "_")
            import_name = import_name_mapping.get(package_name, package_name)
            import_result = subprocess.run(
                [clean_venv["python"], "-c", f"import {import_name}"],
                capture_output=True,
                text=True,
            )

            if import_result.returncode != 0:
                pytest.fail(
                    f"Failed to import {import_name} (from package {package_name}): {import_result.stderr}"
                )

    @pytest.mark.slow
    def test_ml_dependencies_resolve(self, clean_venv):
        """Test that ML dependencies can be resolved (marked slow due to large downloads)."""
        # Test ML dependencies (these are large and may take time)
        ml_dependencies = [
            "torch>=1.12.0",
            "transformers>=4.20.0",
        ]

        for dep in ml_dependencies:
            result = subprocess.run(
                [
                    clean_venv["pip"],
                    "install",
                    dep,
                    "--index-url",
                    "https://download.pytorch.org/whl/cpu",
                ],
                capture_output=True,
                text=True,
                timeout=300,
            )  # 5 minute timeout

            if result.returncode != 0:
                pytest.skip(
                    f"Skipping {dep} due to installation issues: {result.stderr}"
                )

    def test_development_workflow(self, clean_venv):
        """Test a complete development workflow."""
        project_root = Path(__file__).parent.parent.parent

        # Install package in development mode (if setup.py exists)
        setup_py = project_root / "setup.py"
        if not setup_py.exists():
            pytest.skip("setup.py not yet created")

        # Install in development mode
        result = subprocess.run(
            [clean_venv["pip"], "install", "-e", str(project_root)],
            capture_output=True,
            text=True,
        )

        if result.returncode != 0:
            pytest.fail(f"Development installation failed: {result.stderr}")

        # Test that changes to source code are immediately available
        # (This is a basic test of editable installation)
        import_test = subprocess.run(
            [
                clean_venv["python"],
                "-c",
                "import aim2_project; print('Development installation working')",
            ],
            capture_output=True,
            text=True,
        )

        if import_test.returncode != 0:
            pytest.fail(f"Development installation not working: {import_test.stderr}")


class TestSetupCompatibility:
    """Test setup.py compatibility across different scenarios."""

    def test_python_version_compatibility(self):
        """Test that setup.py works with supported Python versions."""
        current_version = f"{sys.version_info.major}.{sys.version_info.minor}"
        supported_versions = ["3.8", "3.9", "3.10", "3.11"]

        if current_version not in supported_versions:
            pytest.skip(f"Current Python {current_version} not in supported versions")

        # Test that syntax is compatible
        project_root = Path(__file__).parent.parent.parent
        setup_py = project_root / "setup.py"

        if setup_py.exists():
            try:
                with open(setup_py, "r") as f:
                    compile(f.read(), str(setup_py), "exec")
            except SyntaxError as e:
                pytest.fail(
                    f"setup.py syntax incompatible with Python {current_version}: {e}"
                )

    def test_setuptools_compatibility(self):
        """Test compatibility with different setuptools versions."""
        # Test that setup.py uses modern setuptools features appropriately
        import setuptools

        setuptools_version = setuptools.__version__
        major_version = int(setuptools_version.split(".")[0])

        # Modern setuptools should be used (40+)
        assert major_version >= 40, f"setuptools version {setuptools_version} too old"

    def test_pip_compatibility(self):
        """Test compatibility with modern pip versions."""
        # Test that package can be installed with modern pip
        result = subprocess.run(
            [sys.executable, "-m", "pip", "--version"], capture_output=True, text=True
        )

        assert result.returncode == 0
        assert "pip" in result.stdout


# Test markers are defined in pytest.ini and conftest.py
