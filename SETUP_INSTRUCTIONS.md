# AIM2 Project Virtual Environment Setup Instructions

This document provides detailed instructions for setting up the AIM2 Ontology Information Extraction project development environment.

## Quick Start

### Windows
1. Double-click `setup_env.bat` or run from Command Prompt:
   ```cmd
   setup_env.bat
   ```

### macOS/Linux
1. Run the shell script:
   ```bash
   ./setup_env.sh
   ```

### Cross-Platform Python
1. Run the Python script directly:
   ```bash
   python setup_env.py
   ```

## Setup Script Features

The `setup_env.py` script provides comprehensive virtual environment setup with the following features:

- **Cross-platform compatibility** (Windows, macOS, Linux)
- **Python version validation** (requires Python 3.8+)
- **Automatic virtual environment creation**
- **Dependency installation** from requirements files
- **Development vs production setup modes**
- **Progress indicators** and detailed status messages
- **Error handling and recovery**
- **Idempotent operation** (safe to run multiple times)

## Setup Options

### Development Setup (Default)
Installs all dependencies including development tools:
```bash
python setup_env.py
# or
python setup_env.py --dev
```

### Production Setup
Installs only core dependencies:
```bash
python setup_env.py --prod
```

### Custom Virtual Environment Name
Specify a custom name for the virtual environment:
```bash
python setup_env.py --venv-name myproject
```

### Force Recreate Environment
Force recreation of an existing virtual environment:
```bash
python setup_env.py --force-recreate
```

### Specify Python Executable
Use a specific Python executable:
```bash
python setup_env.py --python /usr/bin/python3.9
```

### Verbose Output
Enable detailed output for debugging:
```bash
python setup_env.py --verbose
```

## What Gets Installed

### Core Dependencies (Production)
- **Machine Learning**: transformers, torch, scikit-learn, sentence-transformers
- **NLP**: spacy, langchain
- **Ontology Processing**: owlready2, networkx, rdflib
- **Data Processing**: pandas, numpy
- **Scientific Libraries**: biopython, pubchempy
- **Document Processing**: lxml, pypdf
- **Utilities**: pyyaml, requests

### Development Dependencies (Development Mode)
- **Testing**: pytest, pytest-cov, pytest-mock, coverage
- **Code Quality**: black, isort, flake8, pylint, bandit
- **Type Checking**: mypy with type stubs
- **Documentation**: sphinx, sphinx-rtd-theme
- **Development Tools**: ipython, jupyter, jupyterlab
- **Build Tools**: build, wheel, twine
- **Profiling**: memory-profiler, line-profiler
- **Security**: safety, pip-audit
- **And many more...**

## After Setup

### Activating the Virtual Environment

**Windows:**
```cmd
# Command Prompt
venv\Scripts\activate.bat

# PowerShell
venv\Scripts\Activate.ps1
```

**macOS/Linux:**
```bash
# Bash/Zsh
source venv/bin/activate

# Fish
source venv/bin/activate.fish

# Csh/Tcsh
source venv/bin/activate.csh
```

### Available CLI Commands
Once activated, you'll have access to AIM2 project commands:
- `aim2-ontology-manager` - Ontology management
- `aim2-ner-extractor` - Named entity recognition
- `aim2-corpus-builder` - Corpus building
- `aim2-evaluation-benchmarker` - Evaluation and benchmarking
- `aim2-synthetic-generator` - Synthetic data generation

### Development Tools (Dev Mode)
- `pytest` - Run tests
- `black .` - Format code
- `mypy aim2_project` - Type checking
- `jupyter lab` - Start JupyterLab

### Deactivating
```bash
deactivate
```

## Troubleshooting

### Python Version Issues
- Ensure Python 3.8+ is installed
- On some systems, use `python3` instead of `python`
- Check with: `python --version`

### Permission Issues (Unix)
```bash
chmod +x setup_env.py
chmod +x setup_env.sh
```

### Windows Execution Policy (PowerShell)
```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

### Virtual Environment Already Exists
Use `--force-recreate` to recreate:
```bash
python setup_env.py --force-recreate
```

### Installation Timeouts
The script includes generous timeouts, but for very slow connections:
- Use `--verbose` to see detailed progress
- Run individual pip install commands manually if needed

### Missing Dependencies
If setup fails due to missing system dependencies:
- **macOS**: Install Xcode Command Line Tools: `xcode-select --install`
- **Ubuntu/Debian**: `sudo apt-get install python3-dev build-essential`
- **CentOS/RHEL**: `sudo yum install python3-devel gcc`

## Manual Setup (If Script Fails)

If the automated setup fails, you can set up manually:

1. Create virtual environment:
   ```bash
   python -m venv venv
   ```

2. Activate virtual environment (see activation instructions above)

3. Upgrade pip:
   ```bash
   pip install --upgrade pip
   ```

4. Install dependencies:
   ```bash
   pip install -r aim2_project/requirements.txt
   pip install -r requirements-dev.txt  # For development
   ```

5. Install project in editable mode:
   ```bash
   pip install -e .  # Production
   pip install -e .[dev]  # Development
   ```

## Configuration

After setup, configure the project:
1. Copy `aim2_project/configs/default_config.yaml` to create your configuration
2. Modify settings as needed for your environment
3. Set environment variables if required

## Support

For issues with the setup script:
1. Run with `--verbose` flag for detailed output
2. Check the error messages and common issues above
3. Ensure all system requirements are met
4. Try manual setup if automated setup fails
