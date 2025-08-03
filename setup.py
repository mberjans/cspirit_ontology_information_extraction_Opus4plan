#!/usr/bin/env python3
"""
Setup script for AIM2 Ontology Information Extraction Project.

This setup.py configures the AIM2 project for installation as a Python package,
including all dependencies, entry points, and package metadata.
"""

from pathlib import Path
from setuptools import setup, find_packages

# Get the long description from the README file
here = Path(__file__).parent.resolve()
long_description = (here / "README.md").read_text(encoding="utf-8")


# Read requirements from requirements.txt
def read_requirements(filename):
    """Read requirements from a requirements file."""
    requirements_path = here / "aim2_project" / filename
    if requirements_path.exists():
        with open(requirements_path, "r", encoding="utf-8") as f:
            requirements = []
            for line in f:
                line = line.strip()
                # Skip empty lines and comments
                if line and not line.startswith("#"):
                    # Remove inline comments
                    if "#" in line:
                        line = line.split("#")[0].strip()
                    if line:  # Check again after removing comments
                        requirements.append(line)
            return requirements
    return []


# Core dependencies from requirements.txt
install_requires = read_requirements("requirements.txt")


# Development dependencies
def read_dev_requirements():
    """Read development requirements from requirements-dev.txt."""
    dev_requirements_path = here / "requirements-dev.txt"
    if dev_requirements_path.exists():
        with open(dev_requirements_path, "r", encoding="utf-8") as f:
            dev_requirements = []
            for line in f:
                line = line.strip()
                # Skip empty lines, comments, and section headers
                if line and not line.startswith("#") and not line.startswith("="):
                    # Remove inline comments
                    if "#" in line:
                        line = line.split("#")[0].strip()
                    if line:  # Check again after removing comments
                        # Skip problematic package names that have special characters
                        if "pdb++" in line or "html5lib" in line:
                            continue
                        dev_requirements.append(line)
            return dev_requirements
    return []


extras_require = {
    "dev": [
        "pytest>=7.4.0",
        "pytest-cov>=4.1.0",
        "black>=23.7.0",
        "isort>=5.12.0",
        "mypy>=1.5.0",
        "pre-commit>=3.3.0",
        "flake8>=6.0.0",
        "bandit>=1.7.5",
    ],
    "test": [
        "pytest>=7.4.0",
        "pytest-cov>=4.1.0",
        "pytest-mock>=3.11.0",
        "pytest-xdist>=3.3.0",
        "pytest-timeout>=2.1.0",
        "pytest-asyncio>=0.21.0",
        "coverage>=7.2.0",
    ],
    "lint": [
        "black>=23.7.0",
        "isort>=5.12.0",
        "flake8>=6.0.0",
        "mypy>=1.5.0",
        "pylint>=2.17.0",
        "bandit>=1.7.5",
    ],
    "docs": ["sphinx>=7.1.0", "sphinx-rtd-theme>=1.3.0", "myst-parser>=2.0.0"],
    "all": [
        "pytest>=7.4.0",
        "pytest-cov>=4.1.0",
        "black>=23.7.0",
        "isort>=5.12.0",
        "mypy>=1.5.0",
        "pre-commit>=3.3.0",
        "flake8>=6.0.0",
        "bandit>=1.7.5",
        "sphinx>=7.1.0",
        "sphinx-rtd-theme>=1.3.0",
    ],
}

setup(
    # Basic package information
    name="aim2-project",
    version="0.1.0",
    description="AI-First Ontology Information Extraction framework for automated scientific literature processing and knowledge graph construction",
    long_description=long_description,
    long_description_content_type="text/markdown",
    # Author and contact information
    author="AIM2 Project Contributors",
    author_email="",  # Add email if available
    maintainer="AIM2 Project Contributors",
    # URLs
    url="https://github.com/your-org/aim2-project",  # Update with actual repository URL
    download_url="https://github.com/your-org/aim2-project/archive/v0.1.0.tar.gz",
    project_urls={
        "Bug Reports": "https://github.com/your-org/aim2-project/issues",
        "Source": "https://github.com/your-org/aim2-project",
        "Documentation": "https://github.com/your-org/aim2-project/docs",
    },
    # License
    license="MIT",
    # Package discovery and structure
    packages=find_packages(include=["aim2_project", "aim2_project.*"]),
    package_dir={"": "."},
    # Include additional files
    include_package_data=True,
    package_data={
        "aim2_project": [
            "configs/*.yaml",
            "configs/*.yml",
            "data/benchmarks/*",
            "data/corpus/*",
            "data/ontologies/*",
            "data/synthetic/*",
        ],
    },
    # Dependencies
    python_requires=">=3.8",
    install_requires=install_requires,
    extras_require=extras_require,
    # Entry points for command-line tools
    entry_points={
        "console_scripts": [
            # Ontology management commands
            "aim2-ontology-manager=aim2_project.aim2_ontology.ontology_manager:main",
            "aim2-ontology-trimmer=aim2_project.aim2_ontology.ontology_trimmer:main",
            "aim2-ontology-integrator=aim2_project.aim2_ontology.ontology_integrator:main",
            # Information extraction commands
            "aim2-ner-extractor=aim2_project.aim2_extraction.ner_extractor:main",
            "aim2-corpus-builder=aim2_project.aim2_extraction.corpus_builder:main",
            "aim2-relationship-extractor=aim2_project.aim2_extraction.relationship_extractor:main",
            "aim2-evaluation-benchmarker=aim2_project.aim2_extraction.evaluation_benchmarker:main",
            # Utility commands
            "aim2-synthetic-generator=aim2_project.aim2_utils.synthetic_data_generator:main",
            "aim2-config-manager=aim2_project.aim2_utils.config_manager:main",
        ],
    },
    # PyPI classifiers
    classifiers=[
        # Development status
        "Development Status :: 3 - Alpha",
        # Intended audience
        "Intended Audience :: Science/Research",
        "Intended Audience :: Developers",
        "Intended Audience :: Education",
        # Topic classification
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Bio-Informatics",
        "Topic :: Scientific/Engineering :: Information Analysis",
        "Topic :: Text Processing :: Linguistic",
        "Topic :: Software Development :: Libraries :: Python Modules",
        # License
        "License :: OSI Approved :: MIT License",
        # Programming language
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        # Operating systems
        "Operating System :: OS Independent",
        "Operating System :: POSIX :: Linux",
        "Operating System :: MacOS",
        "Operating System :: Microsoft :: Windows",
        # Natural language
        "Natural Language :: English",
        # Environment
        "Environment :: Console",
        "Environment :: Other Environment",
    ],
    # Keywords for PyPI search
    keywords=[
        "ontology",
        "information-extraction",
        "nlp",
        "llm",
        "ai",
        "knowledge-graph",
        "scientific-literature",
        "entity-recognition",
        "relationship-extraction",
        "machine-learning",
        "transformers",
        "bioinformatics",
        "text-mining",
        "automation",
        "synthetic-data",
    ],
    # Zip safety
    zip_safe=False,
    # Additional metadata
    platforms=["any"],
)
