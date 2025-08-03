"""
Pytest configuration and shared fixtures for setup.py testing.

This module provides common fixtures, test utilities, and configuration
for all setup.py related tests.
"""

import tempfile
from pathlib import Path
from unittest.mock import Mock, patch
import pytest


# Test configuration
def pytest_configure(config):
    """Configure pytest with custom markers and settings."""
    config.addinivalue_line("markers", "integration: mark test as integration test")
    config.addinivalue_line("markers", "slow: mark test as slow running")
    config.addinivalue_line(
        "markers", "requires_network: mark test as requiring network access"
    )


def pytest_collection_modifyitems(config, items):
    """Modify test collection to add markers automatically."""
    for item in items:
        # Mark integration tests
        if "integration" in str(item.fspath):
            item.add_marker(pytest.mark.integration)

        # Mark tests that install packages as slow
        if any(
            keyword in item.name.lower() for keyword in ["install", "build", "wheel"]
        ):
            item.add_marker(pytest.mark.slow)


# Project fixtures
@pytest.fixture(scope="session")
def project_root():
    """Get the project root directory."""
    return Path(__file__).parent.parent.parent


@pytest.fixture(scope="session")
def aim2_project_dir(project_root):
    """Get the aim2_project directory."""
    return project_root / "aim2_project"


@pytest.fixture
def mock_setup_py_content():
    """Mock content for setup.py file."""
    return """
from setuptools import setup, find_packages

setup(
    name="aim2-ontology-extraction",
    version="0.1.0",
    description="AIM2 Ontology Information Extraction Framework",
    long_description="A comprehensive framework for ontology-based information extraction",
    long_description_content_type="text/markdown",
    author="AIM2 Development Team",
    author_email="team@aim2.example.com",
    url="https://github.com/aim2/ontology-extraction",
    license="MIT",
    packages=find_packages(),
    package_data={
        "aim2_project": [
            "configs/*.yaml",
            "data/ontologies/*.owl",
            "data/benchmarks/*.json",
        ]
    },
    install_requires=[
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
    ],
    extras_require={
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
        ]
    },
    entry_points={
        "console_scripts": [
            "aim2-extract=aim2_project.cli:extract_main",
            "aim2-ontology=aim2_project.cli:ontology_main",
            "aim2-benchmark=aim2_project.cli:benchmark_main",
        ]
    },
    python_requires=">=3.8",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Text Processing :: Linguistic",
    ],
)
"""


@pytest.fixture
def temp_project_structure():
    """Create a temporary project structure for testing."""
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)

        # Create project structure
        project_dir = temp_path / "test_project"
        project_dir.mkdir()

        # Create aim2_project package
        aim2_dir = project_dir / "aim2_project"
        aim2_dir.mkdir()
        (aim2_dir / "__init__.py").write_text("__version__ = '0.1.0'")

        # Create subpackages
        for subpackage in ["aim2_ontology", "aim2_extraction", "aim2_utils", "data"]:
            subpkg_dir = aim2_dir / subpackage
            subpkg_dir.mkdir()
            (subpkg_dir / "__init__.py").write_text("")

        # Create configs directory
        configs_dir = aim2_dir / "configs"
        configs_dir.mkdir()
        (configs_dir / "default_config.yaml").write_text("# Default configuration")

        # Create data subdirectories
        data_dir = aim2_dir / "data"
        for data_subdir in ["ontologies", "benchmarks"]:
            (data_dir / data_subdir).mkdir()

        # Create sample files
        (data_dir / "ontologies" / "sample.owl").write_text("<!-- Sample OWL file -->")
        (data_dir / "benchmarks" / "sample.json").write_text('{"sample": "data"}')

        # Create requirements.txt
        (project_dir / "requirements.txt").write_text(
            """
numpy>=1.21.0
pandas>=1.3.0
scikit-learn>=1.0.0
""".strip()
        )

        # Create README
        (project_dir / "README.md").write_text("# Test Project")

        # Create LICENSE
        (project_dir / "LICENSE").write_text("MIT License")

        yield project_dir


@pytest.fixture
def mock_subprocess():
    """Mock subprocess calls for testing."""
    with patch("subprocess.run") as mock_run:
        mock_run.return_value.returncode = 0
        mock_run.return_value.stdout = "Success"
        mock_run.return_value.stderr = ""
        yield mock_run


@pytest.fixture
def mock_pkg_resources():
    """Mock pkg_resources for testing."""
    mock_distributions = {
        "numpy": Mock(version="1.21.5"),
        "pandas": Mock(version="1.4.0"),
        "torch": Mock(version="1.12.1"),
        "transformers": Mock(version="4.20.1"),
        "spacy": Mock(version="3.4.1"),
    }

    def mock_get_distribution(package_name):
        if package_name in mock_distributions:
            return mock_distributions[package_name]
        raise pkg_resources.DistributionNotFound()

    with patch("pkg_resources.get_distribution", side_effect=mock_get_distribution):
        yield mock_distributions


@pytest.fixture
def mock_setuptools():
    """Mock setuptools.setup for testing."""
    with patch("setuptools.setup") as mock_setup:
        mock_setup.return_value = None
        yield mock_setup


@pytest.fixture
def sample_requirements():
    """Sample requirements for testing."""
    return [
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


@pytest.fixture
def sample_extras_require():
    """Sample extras_require for testing."""
    return {
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
    }


@pytest.fixture
def sample_entry_points():
    """Sample entry points for testing."""
    return {
        "console_scripts": [
            "aim2-extract=aim2_project.cli:extract_main",
            "aim2-ontology=aim2_project.cli:ontology_main",
            "aim2-benchmark=aim2_project.cli:benchmark_main",
        ]
    }


@pytest.fixture
def sample_classifiers():
    """Sample PyPI classifiers for testing."""
    return [
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Programming Language :: Python :: 3.13",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Text Processing :: Linguistic",
    ]


# Test utilities
class TestUtilities:
    """Utility functions for testing setup.py functionality."""

    @staticmethod
    def create_mock_setup_py(content: str, target_dir: Path) -> Path:
        """Create a mock setup.py file with given content."""
        setup_py = target_dir / "setup.py"
        setup_py.write_text(content)
        return setup_py

    @staticmethod
    def validate_package_structure(package_dir: Path) -> bool:
        """Validate that package has proper structure."""
        required_files = [
            "__init__.py",
        ]

        for file_name in required_files:
            if not (package_dir / file_name).exists():
                return False

        return True

    @staticmethod
    def parse_requirements_file(requirements_file: Path) -> list:
        """Parse a requirements.txt file."""
        if not requirements_file.exists():
            return []

        with open(requirements_file, "r") as f:
            lines = f.readlines()

        requirements = []
        for line in lines:
            line = line.strip()
            if line and not line.startswith("#"):
                requirements.append(line)

        return requirements


# Pytest plugins configuration
# pytest_plugins = []  # Removed to avoid pytest warning
