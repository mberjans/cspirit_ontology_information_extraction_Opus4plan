# AIM2 Project: AI-First Ontology Information Extraction

[![Python](https://img.shields.io/badge/python-3.8%2B-blue)](https://python.org)
[![License](https://img.shields.io/badge/license-MIT-green)](LICENSE)
[![Status](https://img.shields.io/badge/status-in%20development-orange)]()

## Overview

The AIM2 (AI-Integrated Management and Mining) Project is a comprehensive Python framework for ontology information extraction that takes an **AI-first approach** to automating traditionally manual tasks in scientific literature processing and knowledge graph construction. By leveraging state-of-the-art Large Language Models (LLMs) and advanced NLP techniques, AIM2 eliminates the need for extensive manual annotation and curation.

## Key Features

### 🤖 AI-First Architecture
- **Automated ontology trimming**: Reduce large ontologies (e.g., 2008 anatomical terms → 293 relevant terms) using LLM-based relevance scoring
- **Synthetic data generation**: Replace manual annotation with AI-generated training and testing datasets
- **Intelligent conflict resolution**: Automatically resolve ontology integration conflicts using LLM reasoning
- **Zero-shot entity recognition**: Extract entities without requiring pre-trained domain-specific models

### 🔧 Comprehensive Toolset
- **Multi-source ontology integration**: ChEBI, Plant Ontology, NCBI Taxonomy, Gene Ontology, and more
- **Advanced information extraction**: Ensemble of BERT-based and LLM-based Named Entity Recognition (NER)
- **Relationship extraction**: Hierarchical and context-dependent relationship identification
- **Literature corpus building**: Automated paper collection from PubMed, PMC, and arXiv with API rate limiting
- **Synthetic benchmarking**: Generate gold-standard evaluation datasets using LLMs

### 🏗️ Modular Design
- **Standalone operation**: Each module can run independently or be imported as a library
- **Flexible LLM backends**: Support for local models (Llama) and cloud APIs (GPT-4, Gemma)
- **Multiple ontology formats**: OWL, CSV, JSON-LD support with automatic format detection
- **Configurable pipelines**: YAML-based configuration for all components

## Project Structure

```
aim2_project/
├── aim2_ontology/          # Ontology Development and Management
│   ├── ontology_manager.py     # Main ontology operations coordinator
│   ├── ontology_trimmer.py     # AI-assisted ontology pruning
│   ├── ontology_integrator.py  # Multi-source ontology merger
│   ├── relationship_manager.py # Relationship graph management
│   └── ontology_exporter.py    # Format conversion and export
├── aim2_extraction/        # Information Extraction Pipeline
│   ├── corpus_builder.py       # Literature corpus construction
│   ├── text_processor.py       # Document parsing and preprocessing
│   ├── ner_extractor.py        # Multi-model entity recognition
│   ├── relationship_extractor.py # LLM-based relationship extraction
│   ├── ontology_mapper.py      # Entity-to-ontology mapping
│   └── evaluation_benchmarker.py # Performance evaluation system
├── aim2_utils/             # Core Utilities
│   ├── llm_interface.py        # Unified LLM provider interface
│   ├── synthetic_data_generator.py # AI-generated training data
│   └── config_manager.py       # Configuration management
├── data/                   # Data Storage
│   ├── ontologies/            # Ontology files (OWL, CSV, JSON-LD)
│   ├── corpus/               # Literature corpus
│   ├── synthetic/            # Generated datasets
│   └── benchmarks/           # Evaluation benchmarks
└── configs/               # Configuration Files
    └── default_config.yaml    # Default settings template
```

## Environment Setup

This section provides comprehensive instructions for setting up the AIM2 development and production environments. The project includes automated setup scripts, development tools, and code quality checks.

### Prerequisites

Before installing AIM2, ensure you have the following system requirements:

#### System Requirements
- **Python**: 3.8 to 3.12 (tested on all versions)
- **Operating System**: Linux, macOS, or Windows
- **Memory**: Minimum 8GB RAM (16GB+ recommended for ML models)
- **Storage**: At least 2GB free space for dependencies and data

#### Python Installation
```bash
# Check your Python version
python --version  # or python3 --version

# If Python is not installed or version < 3.8:
# Ubuntu/Debian: sudo apt update && sudo apt install python3 python3-pip python3-venv
# macOS: brew install python@3.11
# Windows: Download from https://python.org/downloads/
```

### Installation Methods

#### Method 1: Quick Setup (Recommended)

Use the automated setup script for the fastest installation:

```bash
# Clone the repository
git clone <repository-url>
cd cspirit_ontology_information_extraction_Opus4plan

# Run the automated setup script
./setup_env.sh

# The script will:
# - Check Python version compatibility
# - Create virtual environment
# - Install all dependencies
# - Set up pre-commit hooks
# - Configure basic settings
```

#### Method 2: Manual Setup

For more control over the installation process:

```bash
# Clone the repository
git clone <repository-url>
cd cspirit_ontology_information_extraction_Opus4plan

# Create and activate virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Upgrade pip for better dependency resolution
pip install --upgrade pip setuptools wheel

# Install production dependencies
pip install -r aim2_project/requirements.txt

# Optional: Install development dependencies
pip install -r requirements-dev.txt

# Install the package in development mode
pip install -e .
```

#### Method 3: Using Makefile (Development Environment)

For developers working on the project:

```bash
# Clone the repository
git clone <repository-url>
cd cspirit_ontology_information_extraction_Opus4plan

# View available commands
make help

# Set up complete development environment
make setup-dev

# This will:
# - Create virtual environment
# - Install all dependencies (production + development)
# - Install pre-commit hooks
# - Run initial code quality checks
```

#### Method 4: Production-Only Installation

For production deployments without development tools:

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate

# Install only production dependencies
pip install -r aim2_project/requirements.txt

# Install the package
pip install .
```

### Configuration Setup

#### 1. Basic Configuration

```bash
# Copy the default configuration template
cp aim2_project/configs/default_config.yaml config.yaml

# Edit the configuration file with your settings
# Use your preferred editor: nano, vim, code, etc.
nano config.yaml
```

#### 2. LLM API Configuration

Add your API keys to the configuration file:

```yaml
# config.yaml
llm:
  default_model: "llama-70b"
  api_keys:
    openai: "your-openai-api-key"        # For GPT models
    google: "your-google-api-key"        # For Gemini models
    anthropic: "your-anthropic-api-key"  # For Claude models
  local_models:
    llama_path: "/path/to/local/llama/model"  # Optional: for local models

# Environment variables (alternative to config file)
export OPENAI_API_KEY="your-openai-api-key"
export GOOGLE_API_KEY="your-google-api-key"
```

#### 3. Directory Structure Setup

The installation automatically creates the required directory structure:

```
aim2_project/
├── data/
│   ├── ontologies/     # Ontology files will be stored here
│   ├── corpus/         # Literature corpus storage
│   ├── synthetic/      # Generated synthetic data
│   └── benchmarks/     # Evaluation benchmarks
└── configs/
    └── default_config.yaml  # Configuration template
```

### Development Environment Setup

#### Installing Pre-commit Hooks

Pre-commit hooks ensure code quality and consistency:

```bash
# If using Makefile
make pre-commit-install

# Or manually
pip install pre-commit
pre-commit install

# Test the hooks
pre-commit run --all-files
```

#### Available Development Commands

```bash
# Code formatting and quality checks
make format          # Format code with black and isort
make lint            # Run linting checks (flake8, pylint)
make type-check      # Run type checking with mypy
make security-check  # Run security analysis with bandit

# Testing
make test           # Run all tests
make test-cov       # Run tests with coverage report
make test-fast      # Run tests in parallel

# Documentation
make docs           # Build documentation
make docs-serve     # Serve documentation locally

# Environment management
make clean          # Clean build artifacts
make clean-all      # Clean everything including venv
make update-deps    # Update all dependencies
```

### Installation Verification

#### 1. Basic Installation Check

```bash
# Verify Python environment
python --version
which python

# Check package installation
python -c "import aim2_project; print('Installation successful!')"

# Test CLI commands
aim2-ontology-manager --help
aim2-ner-extractor --help
```

#### 2. Run Basic Tests

```bash
# Quick functionality test
python -c "
from aim2_ontology import OntologyManager
from aim2_extraction import NERExtractor
print('Core modules imported successfully!')
"

# If development environment is set up
make test-fast
```

#### 3. Validate Configuration

```bash
# Check configuration loading
python -c "
from aim2_utils.config_manager import ConfigManager
config = ConfigManager('config.yaml')
print('Configuration loaded successfully!')
"
```

### Troubleshooting

#### Common Installation Issues

**Python Version Compatibility**
```bash
# Error: "Python 3.x is not supported"
# Solution: Install Python 3.8-3.12
pyenv install 3.11.0  # If using pyenv
pyenv local 3.11.0
```

**Virtual Environment Issues**
```bash
# Error: "venv module not found"
# Ubuntu/Debian: sudo apt install python3-venv
# macOS: Already included with Python 3.3+
# Windows: Included with Python from python.org

# Error: "Permission denied"
# Windows: Run command prompt as administrator
# Linux/macOS: Check directory permissions
```

**Dependency Conflicts**
```bash
# Clear pip cache and reinstall
pip cache purge
pip install --upgrade --force-reinstall -r aim2_project/requirements.txt

# For torch/CUDA issues
pip install torch --index-url https://download.pytorch.org/whl/cpu
```

**Memory Issues During Installation**
```bash
# For systems with limited memory
pip install --no-cache-dir -r aim2_project/requirements.txt

# Install dependencies one by one if needed
pip install transformers torch spacy scikit-learn
```

#### Platform-Specific Notes

**Windows**
- Use `venv\Scripts\activate` instead of `source venv/bin/activate`
- Some packages may require Microsoft Visual C++ Build Tools
- Long path support may need to be enabled for deep directory structures

**macOS**
- Install Xcode Command Line Tools: `xcode-select --install`
- Use Homebrew for Python installation: `brew install python@3.11`
- M1/M2 Macs may need specific PyTorch builds

**Linux**
- Install system dependencies: `sudo apt update && sudo apt install build-essential python3-dev`
- For GPU support: Install CUDA toolkit if needed
- Check available memory: `free -h`

#### Getting Help

If you encounter issues:

1. **Check system requirements** and ensure Python 3.8-3.12 is installed
2. **Review error messages** carefully - they often contain specific solutions
3. **Search existing issues** in the project repository
4. **Create a new issue** with:
   - Operating system and version
   - Python version (`python --version`)
   - Complete error message
   - Steps to reproduce the problem

### Hardware Recommendations

For optimal performance:

- **CPU**: Multi-core processor (4+ cores recommended)
- **RAM**: 16GB+ for processing large ontologies and ML models
- **Storage**: SSD recommended for faster I/O operations
- **GPU**: Optional but beneficial for local LLM inference (CUDA-compatible)

## Quick Start

### Standalone Module Execution

```bash
# Trim a large ontology using AI
python -m aim2_ontology.ontology_manager --config config.yaml --trim --target-size 300

# Extract entities from a document collection
python -m aim2_extraction.ner_extractor --input data/corpus/ --output results/

# Generate synthetic training data
python -m aim2_utils.synthetic_data_generator --ontology data/ontologies/chebi.owl --count 100
```

### Library Usage

```python
from aim2_ontology import OntologyManager, OntologyTrimmer
from aim2_extraction import NERExtractor, RelationshipExtractor
from aim2_utils import LLMInterface

# Initialize components
llm = LLMInterface(model='llama-70b', local=True)
ontology_manager = OntologyManager(config_path='config.yaml')
ner = NERExtractor(model_type='ensemble')

# Load and trim ontology
ontology = ontology_manager.load_ontologies()
trimmer = OntologyTrimmer(llm)
trimmed_ontology = trimmer.auto_trim_ontology(ontology, target_size=300)

# Extract information from text
text = "Quercetin is a flavonoid found in onions that exhibits antioxidant properties."
entities = ner.extract_entities_llm(text)
print(f"Extracted entities: {entities}")

# Extract relationships
rel_extractor = RelationshipExtractor(llm)
relationships = rel_extractor.extract_relationships(text, entities)
print(f"Extracted relationships: {relationships}")
```

## Module Descriptions

### 🧬 Ontology Management (`aim2_ontology/`)
- **OntologyManager**: Coordinates loading, integration, and export of multiple ontology sources
- **OntologyTrimmer**: Uses LLMs to intelligently reduce ontology size while preserving relevance
- **OntologyIntegrator**: Merges structural, source, and functional annotation ontologies
- **RelationshipManager**: Manages complex hierarchical relationships and infers missing connections

### 📄 Information Extraction (`aim2_extraction/`)
- **CorpusBuilder**: Automated literature collection with deduplication and metadata management
- **NERExtractor**: Ensemble approach combining BERT-based and LLM-based entity recognition
- **RelationshipExtractor**: Sophisticated relationship extraction using advanced LLM prompting
- **EvaluationBenchmarker**: Comprehensive evaluation system with synthetic gold standards

### 🔧 Utilities (`aim2_utils/`)
- **LLMInterface**: Unified interface supporting multiple LLM providers with rate limiting and caching
- **SyntheticDataGenerator**: Generates realistic training and testing data to replace manual annotation
- **ConfigManager**: YAML-based configuration system with environment variable support

## Advanced Features

### Multi-Model Ensemble
```python
# Use multiple models for higher accuracy
ner = NERExtractor(model_type='ensemble')
entities_bert = ner.extract_entities_bert(text, model='cybert')
entities_llm = ner.extract_entities_llm(text, model='llama-70b')
entities_combined = ner.ensemble_extraction(text)
```

### Batch Processing
```python
# Process large document collections efficiently
corpus_builder = CorpusBuilder()
documents = corpus_builder.search_and_download(['plant metabolites'], limit=1000)
batch_results = ner.batch_process(documents)
```

### Synthetic Benchmark Generation
```python
# Generate evaluation datasets automatically
benchmarker = EvaluationBenchmarker()
gold_standard = benchmarker.generate_gold_standard(n_papers=25)
results = benchmarker.benchmark_models(['llama', 'gpt4', 'gemma'], gold_standard)
```

## Configuration

The system uses YAML configuration files for all settings:

```yaml
# config.yaml
llm:
  default_model: "llama-70b"
  api_keys:
    openai: "your-openai-key"
    google: "your-google-key"
  local_models:
    llama_path: "/path/to/llama/model"

ontology:
  sources:
    - "chebi"
    - "plant_ontology"
    - "ncbi_taxonomy"
  cache_dir: "data/ontologies"

extraction:
  ner_models: ["bert", "llm"]
  confidence_threshold: 0.7
  batch_size: 32
```

## Evaluation and Benchmarking

AIM2 includes comprehensive evaluation capabilities:

- **Automated benchmarking**: Compare multiple models on standardized tasks
- **Synthetic evaluation datasets**: Generate domain-specific test sets
- **Performance metrics**: Precision, recall, F1-score, and custom domain metrics
- **Resource tracking**: Monitor computational costs and processing times

## Implementation Status

The project is currently in active development:

- ✅ **Repository Setup**: Git structure, dependencies, configuration
- ✅ **Basic Infrastructure**: Package structure, imports, configuration system
- 🚧 **Core Modules**: Ontology management and extraction pipelines (in progress)
- ⏳ **Integration Testing**: End-to-end workflow validation (planned)
- ⏳ **Documentation**: API docs and tutorials (planned)

See [docs/checklist.md](docs/checklist.md) for detailed progress tracking.

## Contributing

We welcome contributions to the AIM2 project! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines on:

- Code style and conventions
- Testing requirements
- Pull request process
- Issue reporting

## Research Applications

AIM2 is designed for researchers working on:

- **Plant biology**: Secondary metabolite research, pathway analysis
- **Pharmacology**: Drug discovery, compound-target relationships
- **Systems biology**: Multi-omics integration, pathway reconstruction
- **Knowledge graphs**: Scientific literature mining, ontology construction

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- Built on advances in transformer models and large language models
- Integrates multiple public ontology sources (ChEBI, GO, Plant Ontology, etc.)
- Designed for the scientific research community

## Support

For questions, issues, or feature requests:

1. Check the [documentation](docs/)
2. Search existing [issues](../../issues)
3. Create a new issue with detailed information
4. Join our community discussions

---

**Note**: This is a research project under active development. APIs and functionality may change between versions. Please refer to the [roadmap](docs/roadmap.md) for planned features and timeline.
