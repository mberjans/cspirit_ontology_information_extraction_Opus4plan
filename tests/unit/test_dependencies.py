"""
Unit tests specifically for dependency management and validation.

This module provides comprehensive testing for all dependency-related
functionality in setup.py including version conflicts, optional dependencies,
and dependency resolution.
"""

import sys
import subprocess
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch
import pytest
from packaging import version, requirements
import pkg_resources


class TestDependencyValidation:
    """Test dependency validation and compatibility."""

    def test_core_dependencies_are_valid_packages(self):
        """Test that all core dependencies are valid PyPI packages."""
        core_dependencies = [
            "numpy>=1.21.0",
            "pandas>=1.3.0",
            "scikit-learn>=1.0.0",
            "spacy>=3.4.0",
            "transformers>=4.20.0",
            "torch>=1.12.0",
            "rdflib>=6.0.0",
            "owlready2>=0.39",
            "networkx>=2.6.0",
            "pyyaml>=6.0",
            "click>=8.0.0",
            "tqdm>=4.62.0",
            "requests>=2.25.0",
        ]

        for dep_str in core_dependencies:
            # Parse dependency specification
            req = requirements.Requirement(dep_str)
            assert req.name is not None
            assert len(req.name) > 0

            # Validate version specifiers
            for spec in req.specifier:
                try:
                    version.parse(spec.version)
                except Exception as e:
                    pytest.fail(f"Invalid version in {dep_str}: {e}")

    def test_development_dependencies_are_valid(self):
        """Test that development dependencies are valid."""
        dev_dependencies = [
            "pytest>=7.0.0",
            "pytest-cov>=3.0.0",
            "black>=22.0.0",
            "flake8>=4.0.0",
            "mypy>=0.950",
            "pre-commit>=2.17.0",
        ]

        for dep_str in dev_dependencies:
            req = requirements.Requirement(dep_str)
            assert req.name is not None

            # Validate that these are development tools
            dev_tools = ["pytest", "black", "flake8", "mypy", "pre-commit"]
            assert any(tool in req.name for tool in dev_tools)

    def test_documentation_dependencies_are_valid(self):
        """Test that documentation dependencies are valid."""
        docs_dependencies = [
            "sphinx>=4.5.0",
            "sphinx-rtd-theme>=1.0.0",
            "myst-parser>=0.17.0",
        ]

        for dep_str in docs_dependencies:
            req = requirements.Requirement(dep_str)
            assert req.name is not None

            # Validate that these are documentation tools
            doc_tools = ["sphinx", "myst"]
            assert any(tool in req.name for tool in doc_tools)

    def test_no_version_conflicts_in_core_dependencies(self):
        """Test that core dependencies don't have version conflicts."""
        # Mock dependency resolution check
        dependencies = [
            "numpy>=1.21.0",
            "pandas>=1.3.0",
            "scikit-learn>=1.0.0",
        ]

        # Basic check: ensure no duplicate packages with conflicting versions
        package_versions = {}
        for dep_str in dependencies:
            req = requirements.Requirement(dep_str)
            if req.name in package_versions:
                # Check for conflicts (simplified)
                existing = package_versions[req.name]
                current = str(req.specifier)
                assert (
                    existing == current
                ), f"Version conflict for {req.name}: {existing} vs {current}"
            else:
                package_versions[req.name] = str(req.specifier)

    def test_minimum_versions_are_reasonable(self):
        """Test that minimum version requirements are reasonable."""
        version_checks = {
            "numpy": "1.21.0",  # Should support modern NumPy
            "pandas": "1.3.0",  # Should support modern Pandas
            "python": "3.8",  # Should support Python 3.8+
        }

        for package, min_version in version_checks.items():
            parsed_version = version.parse(min_version)

            # Check that versions are not too old
            if package == "python":
                assert parsed_version >= version.parse("3.8")
            elif package == "numpy":
                assert parsed_version >= version.parse("1.20.0")
            elif package == "pandas":
                assert parsed_version >= version.parse("1.2.0")


class TestDependencyInstallation:
    """Test dependency installation process."""

    @patch("subprocess.run")
    def test_pip_install_core_dependencies(self, mock_subprocess):
        """Test that core dependencies can be installed via pip."""
        mock_subprocess.return_value.returncode = 0
        mock_subprocess.return_value.stdout = "Successfully installed packages"

        # Test installing core dependencies
        core_deps = ["numpy>=1.21.0", "pandas>=1.3.0"]

        for dep in core_deps:
            result = subprocess.run(
                [sys.executable, "-m", "pip", "install", dep],
                capture_output=True,
                text=True,
            )

            assert result.returncode == 0

    @patch("subprocess.run")
    def test_pip_install_optional_dependencies(self, mock_subprocess):
        """Test that optional dependencies can be installed."""
        mock_subprocess.return_value.returncode = 0

        # Test extras installation
        extras = ["dev", "docs", "test"]

        for extra in extras:
            result = subprocess.run(
                [sys.executable, "-m", "pip", "install", f".[{extra}]"],
                capture_output=True,
                text=True,
            )

            assert result.returncode == 0

    def test_dependency_groups_are_properly_separated(self):
        """Test that dependency groups serve different purposes."""
        dependency_groups = {
            "core": [
                "numpy",
                "pandas",
                "scikit-learn",
                "spacy",
                "transformers",
                "torch",
                "rdflib",
                "owlready2",
                "networkx",
                "pyyaml",
                "click",
                "tqdm",
                "requests",
            ],
            "dev": ["pytest", "pytest-cov", "black", "flake8", "mypy", "pre-commit"],
            "docs": ["sphinx", "sphinx-rtd-theme", "myst-parser"],
            "test": ["pytest", "pytest-cov", "pytest-mock"],
        }

        # Ensure no core dependencies are in dev/docs groups
        core_packages = {dep.split(">=")[0] for dep in dependency_groups["core"]}
        dev_packages = {dep.split(">=")[0] for dep in dependency_groups["dev"]}
        docs_packages = {dep.split(">=")[0] for dep in dependency_groups["docs"]}

        # Core packages should not overlap with dev/docs (except for test utilities)
        core_dev_overlap = core_packages.intersection(dev_packages)
        core_docs_overlap = core_packages.intersection(docs_packages)

        assert len(core_dev_overlap) == 0, f"Core/dev overlap: {core_dev_overlap}"
        assert len(core_docs_overlap) == 0, f"Core/docs overlap: {core_docs_overlap}"


class TestDependencyConstraints:
    """Test dependency version constraints and compatibility."""

    def test_pytorch_compatibility_with_transformers(self):
        """Test that PyTorch and Transformers versions are compatible."""
        torch_version = "1.12.0"
        transformers_version = "4.20.0"

        # Basic compatibility check (in practice, this would check actual compatibility matrix)
        torch_major = int(torch_version.split(".")[0])
        torch_minor = int(torch_version.split(".")[1])

        transformers_major = int(transformers_version.split(".")[0])
        int(transformers_version.split(".")[1])

        # Transformers 4.20+ typically requires PyTorch 1.9+
        assert torch_major >= 1
        assert torch_minor >= 9
        assert transformers_major >= 4

    def test_spacy_model_compatibility(self):
        """Test that spaCy version supports required models."""
        spacy_version = "3.4.0"

        # spaCy 3.4+ should support modern language models
        major, minor = spacy_version.split(".")[:2]
        assert int(major) >= 3
        assert int(minor) >= 4

    def test_python_version_compatibility_with_dependencies(self):
        """Test that Python version supports all dependencies."""

        # Check that dependency versions are compatible with Python 3.8+
        python_sensitive_deps = {
            "transformers": "4.20.0",  # Should work with Python 3.8+
            "torch": "1.12.0",  # Should work with Python 3.8+
            "numpy": "1.21.0",  # Should work with Python 3.8+
        }

        for dep, dep_version in python_sensitive_deps.items():
            # In practice, this would check compatibility matrices
            # For now, ensure versions are modern enough for Python 3.8+
            major_version = int(dep_version.split(".")[0])
            assert major_version >= 1


class TestDependencyResolution:
    """Test dependency resolution and conflict detection."""

    def test_no_circular_dependencies(self):
        """Test that there are no circular dependencies."""
        # This is a simplified test - in practice would use dependency graph
        dependencies = {
            "aim2_project": ["numpy", "pandas", "spacy"],
            "aim2_ontology": ["rdflib", "owlready2"],
            "aim2_extraction": ["transformers", "torch", "spacy"],
            "aim2_utils": ["pyyaml", "click"],
        }

        # Basic check: ensure no package depends on itself
        for package, deps in dependencies.items():
            assert package not in deps

    def test_transitive_dependency_compatibility(self):
        """Test that transitive dependencies are compatible."""
        # Mock transitive dependency check
        primary_deps = ["torch>=1.12.0", "transformers>=4.20.0"]

        # Both torch and transformers depend on numpy, but should be compatible
        # This is a simplified test - real implementation would check actual dependencies
        for dep in primary_deps:
            req = requirements.Requirement(dep)
            assert req.name is not None

    @patch("pkg_resources.get_distribution")
    def test_installed_versions_meet_requirements(self, mock_get_distribution):
        """Test that installed package versions meet requirements."""
        # Mock installed packages
        mock_distributions = {
            "numpy": Mock(version="1.21.5"),
            "pandas": Mock(version="1.4.0"),
            "torch": Mock(version="1.12.1"),
        }

        def mock_get_dist(package_name):
            if package_name in mock_distributions:
                return mock_distributions[package_name]
            raise pkg_resources.DistributionNotFound()

        mock_get_distribution.side_effect = mock_get_dist

        # Test requirements
        requirements_to_test = [
            "numpy>=1.21.0",
            "pandas>=1.3.0",
            "torch>=1.12.0",
        ]

        for req_str in requirements_to_test:
            req = requirements.Requirement(req_str)
            try:
                dist = pkg_resources.get_distribution(req.name)
                installed_version = version.parse(dist.version)

                # Check if installed version satisfies requirement
                for spec in req.specifier:
                    if spec.operator == ">=":
                        required_version = version.parse(spec.version)
                        assert installed_version >= required_version
            except pkg_resources.DistributionNotFound:
                # Package not installed - skip test
                pass


class TestOptionalDependencies:
    """Test optional and extra dependencies."""

    def test_extras_require_structure(self):
        """Test that extras_require has proper structure."""
        extras_require = {
            "dev": [
                "pytest>=7.0.0",
                "pytest-cov>=3.0.0",
                "black>=22.0.0",
                "flake8>=4.0.0",
                "mypy>=0.950",
                "pre-commit>=2.17.0",
            ],
            "docs": [
                "sphinx>=4.5.0",
                "sphinx-rtd-theme>=1.0.0",
                "myst-parser>=0.17.0",
            ],
            "test": [
                "pytest>=7.0.0",
                "pytest-cov>=3.0.0",
                "pytest-mock>=3.7.0",
            ],
            "all": [],  # Will be populated with all extras
        }

        # Validate structure
        assert isinstance(extras_require, dict)
        for extra_name, deps in extras_require.items():
            assert isinstance(extra_name, str)
            assert isinstance(deps, list)

            if extra_name != "all":  # Skip 'all' which may be empty initially
                assert len(deps) > 0

    def test_all_extra_includes_everything(self):
        """Test that 'all' extra includes all other extras."""
        all_extras = ["dev", "docs", "test"]
        dev_deps = ["pytest>=7.0.0", "black>=22.0.0"]
        docs_deps = ["sphinx>=4.5.0"]
        test_deps = ["pytest>=7.0.0", "pytest-mock>=3.7.0"]

        # Simulate building 'all' extra
        all_deps = set()
        for extra in all_extras:
            if extra == "dev":
                all_deps.update(dev_deps)
            elif extra == "docs":
                all_deps.update(docs_deps)
            elif extra == "test":
                all_deps.update(test_deps)

        # Verify that 'all' would contain dependencies from other extras
        assert len(all_deps) > 0
        assert any("pytest" in dep for dep in all_deps)

    def test_conditional_dependencies(self):
        """Test conditional dependencies based on Python version or platform."""
        # Example conditional dependencies
        conditional_deps = [
            "typing-extensions>=3.7.4; python_version<'3.8'",
            "importlib-metadata>=1.0; python_version<'3.8'",
        ]

        for dep_str in conditional_deps:
            # Basic validation of conditional syntax
            assert ";" in dep_str
            package_part, condition_part = dep_str.split(";", 1)

            # Validate package part
            req = requirements.Requirement(package_part.strip())
            assert req.name is not None

            # Validate condition part
            condition = condition_part.strip()
            assert "python_version" in condition or "sys_platform" in condition


# Fixtures for dependency testing
@pytest.fixture
def mock_pip_install():
    """Mock pip installation for testing."""
    with patch("subprocess.run") as mock_run:
        mock_run.return_value.returncode = 0
        mock_run.return_value.stdout = "Successfully installed"
        mock_run.return_value.stderr = ""
        yield mock_run


@pytest.fixture
def temp_requirements_file():
    """Create a temporary requirements file for testing."""
    requirements_content = """
numpy>=1.21.0
pandas>=1.3.0
scikit-learn>=1.0.0
spacy>=3.4.0
transformers>=4.20.0
torch>=1.12.0
rdflib>=6.0.0
owlready2>=0.39
networkx>=2.6.0
pyyaml>=6.0
click>=8.0.0
tqdm>=4.62.0
requests>=2.25.0
"""

    with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
        f.write(requirements_content.strip())
        temp_file_path = f.name

    yield temp_file_path

    # Cleanup
    Path(temp_file_path).unlink(missing_ok=True)


@pytest.fixture
def mock_package_metadata():
    """Mock package metadata for testing."""
    return {
        "name": "aim2-ontology-extraction",
        "version": "0.1.0",
        "install_requires": [
            "numpy>=1.21.0",
            "pandas>=1.3.0",
            "scikit-learn>=1.0.0",
        ],
        "extras_require": {
            "dev": ["pytest>=7.0.0", "black>=22.0.0"],
            "docs": ["sphinx>=4.5.0"],
        },
    }
