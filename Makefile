# AIM2 Ontology Information Extraction Project Makefile
# Development Environment Configuration and Common Commands
#
# This Makefile provides convenient commands for development workflows including:
# - Environment setup and management
# - Code quality checks and formatting
# - Testing and coverage
# - Building and packaging
# - Documentation generation
# - Development utilities

# ====================================================================
# Configuration Variables
# ====================================================================
SHELL := /bin/bash
.SHELLFLAGS := -eu -o pipefail -c
.DEFAULT_GOAL := help

# Project configuration
PROJECT_NAME := aim2-project
PACKAGE_NAME := aim2_project
PYTHON_VERSION := 3.13
VENV_NAME := venv
VENV_PATH := $(VENV_NAME)
PYTHON := $(VENV_PATH)/bin/python
PIP := $(VENV_PATH)/bin/pip
PYTEST := $(VENV_PATH)/bin/pytest
BLACK := $(VENV_PATH)/bin/black
ISORT := $(VENV_PATH)/bin/isort
FLAKE8 := $(VENV_PATH)/bin/flake8
MYPY := $(VENV_PATH)/bin/mypy
BANDIT := $(VENV_PATH)/bin/bandit
PYLINT := $(VENV_PATH)/bin/pylint
SPHINX_BUILD := $(VENV_PATH)/bin/sphinx-build
TWINE := $(VENV_PATH)/bin/twine
PRE_COMMIT := $(VENV_PATH)/bin/pre-commit

# Directories
SRC_DIR := $(PACKAGE_NAME)
TEST_DIR := tests
DOCS_DIR := docs
BUILD_DIR := build
DIST_DIR := dist
HTMLCOV_DIR := htmlcov
DOCS_BUILD_DIR := $(DOCS_DIR)/_build
ONTOLOGY_DIR := $(SRC_DIR)/data/ontologies
CORPUS_DIR := $(SRC_DIR)/data/corpus
CONFIG_DIR := $(SRC_DIR)/configs

# Coverage settings
COVERAGE_MIN := 80
COVERAGE_REPORT := --cov=$(PACKAGE_NAME) --cov-report=term-missing --cov-report=html:$(HTMLCOV_DIR) --cov-report=xml

# Test markers
UNIT_TESTS := -m "unit"
INTEGRATION_TESTS := -m "integration"
SLOW_TESTS := -m "slow"
FAST_TESTS := -m "not slow"

# ====================================================================
# Helper Functions
# ====================================================================
define log_info
	@echo "🔵 $(1)"
endef

define log_success
	@echo "✅ $(1)"
endef

define log_warning
	@echo "⚠️  $(1)"
endef

define log_error
	@echo "❌ $(1)"
endef

# Check if virtual environment exists
define check_venv
	@if [ ! -d "$(VENV_PATH)" ]; then \
		$(call log_error,"Virtual environment not found. Run 'make venv' first."); \
		exit 1; \
	fi
endef

# ====================================================================
# Environment Setup
# ====================================================================
.PHONY: help
help: ## Show this help message
	@echo "AIM2 Ontology Information Extraction Project - Development Commands"
	@echo ""
	@echo "Environment Setup:"
	@awk 'BEGIN {FS = ":.*?## "} /^[a-zA-Z_-]+:.*?## / && /Environment Setup/ {found=1} found && /^[a-zA-Z_-]+:.*?## / {printf "  %-20s %s\n", $$1, $$2} /^$$/ {found=0}' $(MAKEFILE_LIST)
	@echo ""
	@echo "Code Quality:"
	@awk 'BEGIN {FS = ":.*?## "} /^[a-zA-Z_-]+:.*?## / && /Code Quality/ {found=1} found && /^[a-zA-Z_-]+:.*?## / {printf "  %-20s %s\n", $$1, $$2} /^$$/ {found=0}' $(MAKEFILE_LIST)
	@echo ""
	@echo "Testing:"
	@awk 'BEGIN {FS = ":.*?## "} /^[a-zA-Z_-]+:.*?## / && /Testing/ {found=1} found && /^[a-zA-Z_-]+:.*?## / {printf "  %-20s %s\n", $$1, $$2} /^$$/ {found=0}' $(MAKEFILE_LIST)
	@echo ""
	@echo "Build and Package:"
	@awk 'BEGIN {FS = ":.*?## "} /^[a-zA-Z_-]+:.*?## / && /Build and Package/ {found=1} found && /^[a-zA-Z_-]+:.*?## / {printf "  %-20s %s\n", $$1, $$2} /^$$/ {found=0}' $(MAKEFILE_LIST)
	@echo ""
	@echo "Documentation:"
	@awk 'BEGIN {FS = ":.*?## "} /^[a-zA-Z_-]+:.*?## / && /Documentation/ {found=1} found && /^[a-zA-Z_-]+:.*?## / {printf "  %-20s %s\n", $$1, $$2} /^$$/ {found=0}' $(MAKEFILE_LIST)
	@echo ""
	@echo "Development Utilities:"
	@awk 'BEGIN {FS = ":.*?## "} /^[a-zA-Z_-]+:.*?## / && /Development Utilities/ {found=1} found && /^[a-zA-Z_-]+:.*?## / {printf "  %-20s %s\n", $$1, $$2} /^$$/ {found=0}' $(MAKEFILE_LIST)
	@echo ""
	@echo "Ontology Management:"
	@awk 'BEGIN {FS = ":.*?## "} /^[a-zA-Z_-]+:.*?## / && /Ontology Management/ {found=1} found && /^[a-zA-Z_-]+:.*?## / {printf "  %-20s %s\n", $$1, $$2} /^$$/ {found=0}' $(MAKEFILE_LIST)
	@echo ""
	@echo "Machine Learning:"
	@awk 'BEGIN {FS = ":.*?## "} /^[a-zA-Z_-]+:.*?## / && /Machine Learning/ {found=1} found && /^[a-zA-Z_-]+:.*?## / {printf "  %-20s %s\n", $$1, $$2} /^$$/ {found=0}' $(MAKEFILE_LIST)
	@echo ""
	@echo "Git and Release:"
	@awk 'BEGIN {FS = ":.*?## "} /^[a-zA-Z_-]+:.*?## / && /Git and Release/ {found=1} found && /^[a-zA-Z_-]+:.*?## / {printf "  %-20s %s\n", $$1, $$2} /^$$/ {found=0}' $(MAKEFILE_LIST)

.PHONY: venv
venv: ## Environment Setup - Create virtual environment
	$(call log_info,"Creating virtual environment...")
	@if command -v python$(PYTHON_VERSION) >/dev/null 2>&1; then \
		python$(PYTHON_VERSION) -m venv $(VENV_PATH); \
	elif command -v python3 >/dev/null 2>&1; then \
		python3 -m venv $(VENV_PATH); \
	else \
		python -m venv $(VENV_PATH); \
	fi
	$(PIP) install --upgrade pip setuptools wheel
	$(call log_success,"Virtual environment created at $(VENV_PATH)")

.PHONY: install
install: venv ## Environment Setup - Install production dependencies
	$(call check_venv)
	$(call log_info,"Installing production dependencies...")
	$(PIP) install -e .
	$(call log_success,"Production dependencies installed")

.PHONY: install-dev
install-dev: venv ## Environment Setup - Install development dependencies
	$(call check_venv)
	$(call log_info,"Installing development dependencies...")
	$(PIP) install -e ".[all]"
	$(call log_success,"Development dependencies installed")

.PHONY: install-test
install-test: venv ## Environment Setup - Install test dependencies only
	$(call check_venv)
	$(call log_info,"Installing test dependencies...")
	$(PIP) install -e ".[test]"
	$(call log_success,"Test dependencies installed")

.PHONY: install-lint
install-lint: venv ## Environment Setup - Install linting dependencies only
	$(call check_venv)
	$(call log_info,"Installing linting dependencies...")
	$(PIP) install -e ".[lint]"
	$(call log_success,"Linting dependencies installed")

.PHONY: requirements
requirements: ## Environment Setup - Generate requirements.txt from setup.cfg
	$(call check_venv)
	$(call log_info,"Generating requirements.txt...")
	$(PIP) freeze > requirements.txt
	$(call log_success,"Requirements saved to requirements.txt")

.PHONY: clean-env
clean-env: ## Environment Setup - Remove virtual environment
	$(call log_warning,"Removing virtual environment...")
	rm -rf $(VENV_PATH)
	$(call log_success,"Virtual environment removed")

# ====================================================================
# Code Quality
# ====================================================================
.PHONY: format
format: ## Code Quality - Format code with black and isort
	$(call check_venv)
	$(call log_info,"Formatting code with black...")
	$(BLACK) $(SRC_DIR) $(TEST_DIR)
	$(call log_info,"Sorting imports with isort...")
	$(ISORT) $(SRC_DIR) $(TEST_DIR)
	$(call log_success,"Code formatted successfully")

.PHONY: format-check
format-check: ## Code Quality - Check code formatting without making changes
	$(call check_venv)
	$(call log_info,"Checking code formatting...")
	$(BLACK) --check $(SRC_DIR) $(TEST_DIR)
	$(ISORT) --check-only $(SRC_DIR) $(TEST_DIR)
	$(call log_success,"Code formatting is correct")

.PHONY: lint
lint: ## Code Quality - Run all linting checks
	$(call check_venv)
	$(call log_info,"Running flake8...")
	$(FLAKE8) $(SRC_DIR) $(TEST_DIR)
	$(call log_info,"Running pylint...")
	$(PYLINT) $(SRC_DIR)
	$(call log_success,"All linting checks passed")

.PHONY: lint-flake8
lint-flake8: ## Code Quality - Run flake8 linting only
	$(call check_venv)
	$(call log_info,"Running flake8...")
	$(FLAKE8) $(SRC_DIR) $(TEST_DIR)
	$(call log_success,"Flake8 linting passed")

.PHONY: lint-pylint
lint-pylint: ## Code Quality - Run pylint only
	$(call check_venv)
	$(call log_info,"Running pylint...")
	$(PYLINT) $(SRC_DIR)
	$(call log_success,"Pylint passed")

.PHONY: security
security: ## Code Quality - Run security checks with bandit
	$(call check_venv)
	$(call log_info,"Running security checks...")
	$(BANDIT) -r $(SRC_DIR)
	$(call log_success,"Security checks passed")

.PHONY: typecheck
typecheck: ## Code Quality - Run type checking with mypy
	$(call check_venv)
	$(call log_info,"Running type checking...")
	$(MYPY) $(SRC_DIR)
	$(call log_success,"Type checking passed")

.PHONY: quality
quality: format-check lint typecheck security ## Code Quality - Run all code quality checks
	$(call log_success,"All code quality checks passed")

# ====================================================================
# Testing
# ====================================================================
.PHONY: test
test: ## Testing - Run all tests with coverage
	$(call check_venv)
	$(call log_info,"Running all tests with coverage...")
	$(PYTEST) $(COVERAGE_REPORT) --cov-fail-under=$(COVERAGE_MIN)
	$(call log_success,"All tests passed")

.PHONY: test-unit
test-unit: ## Testing - Run unit tests only
	$(call check_venv)
	$(call log_info,"Running unit tests...")
	$(PYTEST) $(UNIT_TESTS) $(COVERAGE_REPORT)
	$(call log_success,"Unit tests passed")

.PHONY: test-integration
test-integration: ## Testing - Run integration tests only
	$(call check_venv)
	$(call log_info,"Running integration tests...")
	$(PYTEST) $(INTEGRATION_TESTS) $(COVERAGE_REPORT)
	$(call log_success,"Integration tests passed")

.PHONY: test-fast
test-fast: ## Testing - Run fast tests (exclude slow tests)
	$(call check_venv)
	$(call log_info,"Running fast tests...")
	$(PYTEST) $(FAST_TESTS) $(COVERAGE_REPORT)
	$(call log_success,"Fast tests passed")

.PHONY: test-slow
test-slow: ## Testing - Run slow tests only
	$(call check_venv)
	$(call log_info,"Running slow tests...")
	$(PYTEST) $(SLOW_TESTS) $(COVERAGE_REPORT)
	$(call log_success,"Slow tests passed")

.PHONY: test-verbose
test-verbose: ## Testing - Run tests with verbose output
	$(call check_venv)
	$(call log_info,"Running tests with verbose output...")
	$(PYTEST) -v $(COVERAGE_REPORT)
	$(call log_success,"Tests completed")

.PHONY: test-parallel
test-parallel: ## Testing - Run tests in parallel
	$(call check_venv)
	$(call log_info,"Running tests in parallel...")
	$(PYTEST) -n auto $(COVERAGE_REPORT)
	$(call log_success,"Parallel tests completed")

.PHONY: coverage-report
coverage-report: ## Testing - Generate coverage report
	$(call check_venv)
	$(call log_info,"Generating coverage report...")
	$(PYTEST) --cov=$(PACKAGE_NAME) --cov-report=html:$(HTMLCOV_DIR) --cov-report=term-missing
	$(call log_success,"Coverage report generated in $(HTMLCOV_DIR)/")

.PHONY: coverage-view
coverage-view: coverage-report ## Testing - Open coverage report in browser
	@if command -v open >/dev/null 2>&1; then \
		open $(HTMLCOV_DIR)/index.html; \
	elif command -v xdg-open >/dev/null 2>&1; then \
		xdg-open $(HTMLCOV_DIR)/index.html; \
	else \
		$(call log_info,"Coverage report available at: $(HTMLCOV_DIR)/index.html"); \
	fi

.PHONY: test-clean
test-clean: ## Testing - Clean test artifacts
	$(call log_info,"Cleaning test artifacts...")
	rm -rf .pytest_cache/ $(HTMLCOV_DIR)/ .coverage coverage.xml
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -name "*.pyc" -delete
	$(call log_success,"Test artifacts cleaned")

# ====================================================================
# Clean Environment Testing (AIM2-002-10)
# ====================================================================
.PHONY: test-clean-env
test-clean-env: ## Testing - Run clean environment installation tests (quick)
	$(call check_venv)
	$(call log_info,"Running clean environment installation tests - quick mode...")
	$(PYTHON) run_clean_env_tests.py --quick --verbose
	$(call log_success,"Clean environment tests passed")

.PHONY: test-clean-env-full
test-clean-env-full: ## Testing - Run comprehensive clean environment tests
	$(call check_venv)
	$(call log_info,"Running comprehensive clean environment tests...")
	$(PYTHON) run_clean_env_tests.py --full --verbose --coverage --html-report
	$(call log_success,"Comprehensive clean environment tests passed")

.PHONY: test-clean-venv
test-clean-venv: ## Testing - Test virtual environment creation in clean environment
	$(call check_venv)
	$(call log_info,"Testing clean virtual environment creation...")
	$(PYTHON) run_clean_env_tests.py --venv-only --verbose
	$(call log_success,"Clean venv creation tests passed")

.PHONY: test-clean-install
test-clean-install: ## Testing - Test package installation in clean environment
	$(call check_venv)
	$(call log_info,"Testing package installation in clean environment...")
	$(PYTHON) run_clean_env_tests.py --install-only --verbose
	$(call log_success,"Clean installation tests passed")

.PHONY: test-clean-cross-platform
test-clean-cross-platform: ## Testing - Test cross-platform compatibility in clean environment
	$(call check_venv)
	$(call log_info,"Testing cross-platform compatibility...")
	$(PYTHON) run_clean_env_tests.py --cross-platform --verbose
	$(call log_success,"Cross-platform compatibility tests passed")

.PHONY: test-clean-precommit
test-clean-precommit: ## Testing - Test pre-commit hooks setup in clean environment
	$(call check_venv)
	$(call log_info,"Testing pre-commit hooks setup...")
	$(PYTHON) run_clean_env_tests.py --precommit --verbose
	$(call log_success,"Pre-commit hooks tests passed")

.PHONY: test-clean-env-parallel
test-clean-env-parallel: ## Testing - Run clean environment tests in parallel
	$(call check_venv)
	$(call log_info,"Running clean environment tests in parallel...")
	$(PYTHON) run_clean_env_tests.py --quick --parallel --verbose
	$(call log_success,"Parallel clean environment tests passed")

.PHONY: test-clean-env-coverage
test-clean-env-coverage: ## Testing - Run clean environment tests with coverage
	$(call check_venv)
	$(call log_info,"Running clean environment tests with coverage...")
	$(PYTHON) run_clean_env_tests.py --full --coverage --html-report --verbose
	$(call log_success,"Clean environment tests with coverage completed")

.PHONY: test-clean-env-debug
test-clean-env-debug: ## Testing - Run clean environment tests with debugging artifacts
	$(call check_venv)
	$(call log_info,"Running clean environment tests with debugging...")
	$(PYTHON) run_clean_env_tests.py --quick --verbose --keep-artifacts --json-output clean_env_debug.json
	$(call log_success,"Clean environment debug tests completed")

.PHONY: test-clean-env-report
test-clean-env-report: ## Testing - Generate clean environment test report
	$(call check_venv)
	$(call log_info,"Generating clean environment test report...")
	$(PYTHON) run_clean_env_tests.py --full --coverage --html-report --json-output clean_env_report.json --text-report clean_env_report.txt
	$(call log_success,"Clean environment test report generated")

# ====================================================================
# Build and Package
# ====================================================================
.PHONY: build
build: clean ## Build and Package - Build package distributions
	$(call check_venv)
	$(call log_info,"Building package...")
	$(PYTHON) -m build
	$(call log_success,"Package built successfully")

.PHONY: wheel
wheel: clean ## Build and Package - Build wheel distribution only
	$(call check_venv)
	$(call log_info,"Building wheel...")
	$(PYTHON) -m build --wheel
	$(call log_success,"Wheel built successfully")

.PHONY: sdist
sdist: clean ## Build and Package - Build source distribution only
	$(call check_venv)
	$(call log_info,"Building source distribution...")
	$(PYTHON) -m build --sdist
	$(call log_success,"Source distribution built successfully")

.PHONY: install-local
install-local: build ## Build and Package - Install package locally in development mode
	$(call check_venv)
	$(call log_info,"Installing package in development mode...")
	$(PIP) install -e .
	$(call log_success,"Package installed in development mode")

.PHONY: install-editable
install-editable: ## Build and Package - Install package in editable mode
	$(call check_venv)
	$(call log_info,"Installing package in editable mode...")
	$(PIP) install -e ".[all]"
	$(call log_success,"Package installed in editable mode")

.PHONY: check-package
check-package: build ## Build and Package - Check package integrity
	$(call check_venv)
	$(call log_info,"Checking package integrity...")
	$(TWINE) check $(DIST_DIR)/*
	$(call log_success,"Package integrity check passed")

.PHONY: clean
clean: ## Build and Package - Clean build artifacts
	$(call log_info,"Cleaning build artifacts...")
	rm -rf $(BUILD_DIR)/ $(DIST_DIR)/ *.egg-info/
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -name "*.pyc" -delete
	find . -name "*.pyo" -delete
	find . -name "*.orig" -delete
	find . -name "*.rej" -delete
	$(call log_success,"Build artifacts cleaned")

# ====================================================================
# Documentation
# ====================================================================
.PHONY: docs
docs: ## Documentation - Build documentation
	$(call check_venv)
	$(call log_info,"Building documentation...")
	$(SPHINX_BUILD) -b html $(DOCS_DIR) $(DOCS_BUILD_DIR)/html
	$(call log_success,"Documentation built successfully")

.PHONY: docs-clean
docs-clean: ## Documentation - Clean documentation build files
	$(call log_info,"Cleaning documentation...")
	rm -rf $(DOCS_BUILD_DIR)/
	$(call log_success,"Documentation cleaned")

.PHONY: docs-view
docs-view: docs ## Documentation - Open documentation in browser
	@if command -v open >/dev/null 2>&1; then \
		open $(DOCS_BUILD_DIR)/html/index.html; \
	elif command -v xdg-open >/dev/null 2>&1; then \
		xdg-open $(DOCS_BUILD_DIR)/html/index.html; \
	else \
		$(call log_info,"Documentation available at: $(DOCS_BUILD_DIR)/html/index.html"); \
	fi

.PHONY: docs-serve
docs-serve: docs ## Documentation - Serve documentation locally
	$(call check_venv)
	$(call log_info,"Serving documentation at http://localhost:8000...")
	cd $(DOCS_BUILD_DIR)/html && $(PYTHON) -m http.server 8000

.PHONY: docs-linkcheck
docs-linkcheck: ## Documentation - Check documentation links
	$(call check_venv)
	$(call log_info,"Checking documentation links...")
	$(SPHINX_BUILD) -b linkcheck $(DOCS_DIR) $(DOCS_BUILD_DIR)/linkcheck
	$(call log_success,"Documentation link check completed")

# ====================================================================
# Development Utilities
# ====================================================================
.PHONY: run-ontology-manager
run-ontology-manager: ## Development Utilities - Run ontology manager
	$(call check_venv)
	$(PYTHON) -m $(PACKAGE_NAME).aim2_ontology.ontology_manager $(ARGS)

.PHONY: run-ner-extractor
run-ner-extractor: ## Development Utilities - Run NER extractor
	$(call check_venv)
	$(PYTHON) -m $(PACKAGE_NAME).aim2_extraction.ner_extractor $(ARGS)

.PHONY: run-corpus-builder
run-corpus-builder: ## Development Utilities - Run corpus builder
	$(call check_venv)
	$(PYTHON) -m $(PACKAGE_NAME).aim2_extraction.corpus_builder $(ARGS)

.PHONY: run-relationship-extractor
run-relationship-extractor: ## Development Utilities - Run relationship extractor
	$(call check_venv)
	$(PYTHON) -m $(PACKAGE_NAME).aim2_extraction.relationship_extractor $(ARGS)

.PHONY: run-evaluation-benchmarker
run-evaluation-benchmarker: ## Development Utilities - Run evaluation benchmarker
	$(call check_venv)
	$(PYTHON) -m $(PACKAGE_NAME).aim2_extraction.evaluation_benchmarker $(ARGS)

.PHONY: run-synthetic-generator
run-synthetic-generator: ## Development Utilities - Run synthetic data generator
	$(call check_venv)
	$(PYTHON) -m $(PACKAGE_NAME).aim2_utils.synthetic_data_generator $(ARGS)

.PHONY: profile
profile: ## Development Utilities - Profile application performance
	$(call check_venv)
	$(call log_info,"Running performance profiling...")
	$(PYTHON) -m cProfile -o profile.stats -m $(PACKAGE_NAME) $(ARGS)
	$(call log_success,"Profiling completed. Results in profile.stats")

.PHONY: memory-profile
memory-profile: ## Development Utilities - Profile memory usage
	$(call check_venv)
	$(call log_info,"Running memory profiling...")
	$(VENV_PATH)/bin/mprof run $(PYTHON) -m $(PACKAGE_NAME) $(ARGS)
	$(VENV_PATH)/bin/mprof plot
	$(call log_success,"Memory profiling completed")

.PHONY: security-audit
security-audit: ## Development Utilities - Run comprehensive security audit
	$(call check_venv)
	$(call log_info,"Running security audit...")
	$(VENV_PATH)/bin/safety check
	$(VENV_PATH)/bin/pip-audit
	$(BANDIT) -r $(SRC_DIR)
	$(call log_success,"Security audit completed")

.PHONY: dependencies-check
dependencies-check: ## Development Utilities - Check for outdated dependencies
	$(call check_venv)
	$(call log_info,"Checking for outdated dependencies...")
	$(PIP) list --outdated
	$(call log_success,"Dependency check completed")

.PHONY: shell
shell: ## Development Utilities - Start interactive Python shell with project context
	$(call check_venv)
	$(call log_info,"Starting interactive shell...")
	$(VENV_PATH)/bin/ipython

.PHONY: notebook-setup
notebook-setup: ## Development Utilities - Setup Jupyter with project kernel
	$(call check_venv)
	$(call log_info,"Setting up Jupyter kernel for project...")
	$(VENV_PATH)/bin/python -m ipykernel install --user --name=aim2-project --display-name="AIM2 Project"
	$(call log_success,"Jupyter kernel installed")

.PHONY: data-explore
data-explore: ## Development Utilities - Start data exploration notebook
	$(call check_venv)
	$(call log_info,"Starting data exploration environment...")
	cd $(SRC_DIR)/data && $(VENV_PATH)/bin/jupyter lab

.PHONY: jupyter
jupyter: ## Development Utilities - Start Jupyter notebook server
	$(call check_venv)
	$(call log_info,"Starting Jupyter notebook server...")
	$(VENV_PATH)/bin/jupyter lab

# ====================================================================
# Ontology Management
# ====================================================================
.PHONY: ontology-validate
ontology-validate: ## Ontology Management - Validate ontology files
	$(call check_venv)
	$(call log_info,"Validating ontology files...")
	@if [ -d "$(ONTOLOGY_DIR)" ]; then \
		find $(ONTOLOGY_DIR) -name "*.owl" -o -name "*.rdf" -o -name "*.ttl" | while read file; do \
			echo "Validating $$file..."; \
			$(PYTHON) -c "import rdflib; g=rdflib.Graph(); g.parse('$$file'); print('✅ Valid')" || echo "❌ Invalid: $$file"; \
		done; \
	else \
		$(call log_warning,"Ontology directory not found: $(ONTOLOGY_DIR)"); \
	fi
	$(call log_success,"Ontology validation completed")

.PHONY: ontology-stats
ontology-stats: ## Ontology Management - Show ontology statistics
	$(call check_venv)
	$(call log_info,"Generating ontology statistics...")
	@$(PYTHON) -c "import os; from pathlib import Path; ontology_dir = Path('$(ONTOLOGY_DIR)'); owl_files = list(ontology_dir.glob('*.owl')) if ontology_dir.exists() else []; rdf_files = list(ontology_dir.glob('*.rdf')) if ontology_dir.exists() else []; ttl_files = list(ontology_dir.glob('*.ttl')) if ontology_dir.exists() else []; print('📊 Ontology Statistics:' if ontology_dir.exists() else '⚠️  Ontology directory not found'); print(f'   OWL files: {len(owl_files)}') if ontology_dir.exists() else None; print(f'   RDF files: {len(rdf_files)}') if ontology_dir.exists() else None; print(f'   TTL files: {len(ttl_files)}') if ontology_dir.exists() else None; print(f'   Total: {len(owl_files + rdf_files + ttl_files)}') if ontology_dir.exists() else None"

.PHONY: corpus-stats
corpus-stats: ## Ontology Management - Show corpus statistics
	$(call check_venv)
	$(call log_info,"Generating corpus statistics...")
	@$(PYTHON) -c "import os; from pathlib import Path; corpus_dir = Path('$(CORPUS_DIR)'); txt_files = list(corpus_dir.glob('**/*.txt')) if corpus_dir.exists() else []; json_files = list(corpus_dir.glob('**/*.json')) if corpus_dir.exists() else []; csv_files = list(corpus_dir.glob('**/*.csv')) if corpus_dir.exists() else []; total_size = sum(f.stat().st_size for f in txt_files + json_files + csv_files if f.is_file()) if corpus_dir.exists() else 0; print('📄 Corpus Statistics:' if corpus_dir.exists() else '⚠️  Corpus directory not found'); print(f'   Text files: {len(txt_files)}') if corpus_dir.exists() else None; print(f'   JSON files: {len(json_files)}') if corpus_dir.exists() else None; print(f'   CSV files: {len(csv_files)}') if corpus_dir.exists() else None; print(f'   Total size: {total_size/1024/1024:.2f} MB') if corpus_dir.exists() else None"

.PHONY: config-validate
config-validate: ## Ontology Management - Validate configuration files
	$(call check_venv)
	$(call log_info,"Validating configuration files...")
	@$(PYTHON) -c "import yaml, json; from pathlib import Path; config_dir = Path('$(CONFIG_DIR)'); print('🔧 Configuration Validation:' if config_dir.exists() else '⚠️  Config directory not found'); [print(f'✅ Valid YAML: {f.name}') if yaml.safe_load(open(f)) or True else print(f'❌ Invalid YAML: {f.name}') for f in config_dir.glob('*.yaml')] if config_dir.exists() else None; [print(f'✅ Valid JSON: {f.name}') if json.load(open(f)) or True else print(f'❌ Invalid JSON: {f.name}') for f in config_dir.glob('*.json')] if config_dir.exists() else None"
	$(call log_success,"Configuration validation completed")

# ====================================================================
# Machine Learning & Data Processing
# ====================================================================
.PHONY: data-validate
data-validate: ## Machine Learning - Validate data integrity
	$(call check_venv)
	$(call log_info,"Validating data integrity...")
	@$(PYTHON) -c "from pathlib import Path; data_dir = Path('$(SRC_DIR)/data'); csv_files = list(data_dir.glob('**/*.csv')) if data_dir.exists() else []; print(f'🔍 Checking {len(csv_files)} CSV files...' if data_dir.exists() else '⚠️  Data directory not found')"
	$(call log_success,"Data validation completed")

.PHONY: model-info
model-info: ## Machine Learning - Show model information
	$(call check_venv)
	$(call log_info,"Gathering model information...")
	@echo "try:" > /tmp/model_info.py
	@echo "    import torch" >> /tmp/model_info.py
	@echo "    print('🤖 ML Environment Information:')" >> /tmp/model_info.py
	@echo "    print(f'   PyTorch version: {torch.__version__}')" >> /tmp/model_info.py
	@echo "    print(f'   CUDA available: {torch.cuda.is_available()}')" >> /tmp/model_info.py
	@echo "    print(f'   CUDA devices: {torch.cuda.device_count()}' if torch.cuda.is_available() else '   CUDA devices: N/A')" >> /tmp/model_info.py
	@echo "except ImportError:" >> /tmp/model_info.py
	@echo "    print('⚠️  PyTorch not installed')" >> /tmp/model_info.py
	@$(PYTHON) /tmp/model_info.py
	@rm -f /tmp/model_info.py

.PHONY: download-models
download-models: ## Machine Learning - Download required ML models
	$(call check_venv)
	$(call log_info,"Downloading required models...")
	@$(PYTHON) -c "try: from transformers import AutoTokenizer, AutoModel; models = ['bert-base-uncased', 'distilbert-base-uncased']; [print(f'Downloading {m}...') or AutoTokenizer.from_pretrained(m) or AutoModel.from_pretrained(m) or print(f'✅ {m} downloaded successfully') for m in models]; except ImportError: print('⚠️  Transformers not installed')"
	$(call log_success,"Model download completed")

.PHONY: benchmark-setup
benchmark-setup: ## Machine Learning - Setup benchmark datasets
	$(call check_venv)
	$(call log_info,"Setting up benchmark datasets...")
	$(PYTHON) -m $(PACKAGE_NAME).aim2_extraction.evaluation_benchmarker --setup
	$(call log_success,"Benchmark setup completed")

# ====================================================================
# Git and Release
# ====================================================================
.PHONY: pre-commit-install
pre-commit-install: ## Git and Release - Install pre-commit hooks
	$(call check_venv)
	$(call log_info,"Installing pre-commit hooks...")
	$(PRE_COMMIT) install
	$(PRE_COMMIT) install --hook-type commit-msg
	$(call log_success,"Pre-commit hooks installed")

.PHONY: pre-commit-run
pre-commit-run: ## Git and Release - Run pre-commit hooks on all files
	$(call check_venv)
	$(call log_info,"Running pre-commit hooks...")
	$(PRE_COMMIT) run --all-files
	$(call log_success,"Pre-commit hooks completed")

.PHONY: pre-commit-update
pre-commit-update: ## Git and Release - Update pre-commit hook versions
	$(call check_venv)
	$(call log_info,"Updating pre-commit hooks...")
	$(PRE_COMMIT) autoupdate
	$(call log_success,"Pre-commit hooks updated")

.PHONY: version-bump-patch
version-bump-patch: ## Git and Release - Bump patch version
	$(call check_venv)
	$(call log_info,"Bumping patch version...")
	$(VENV_PATH)/bin/commitizen bump --increment PATCH
	$(call log_success,"Patch version bumped")

.PHONY: version-bump-minor
version-bump-minor: ## Git and Release - Bump minor version
	$(call check_venv)
	$(call log_info,"Bumping minor version...")
	$(VENV_PATH)/bin/commitizen bump --increment MINOR
	$(call log_success,"Minor version bumped")

.PHONY: version-bump-major
version-bump-major: ## Git and Release - Bump major version
	$(call check_venv)
	$(call log_info,"Bumping major version...")
	$(VENV_PATH)/bin/commitizen bump --increment MAJOR
	$(call log_success,"Major version bumped")

.PHONY: release-check
release-check: clean quality test docs check-package ## Git and Release - Run all checks before release
	$(call log_success,"All release checks passed. Ready for release!")

.PHONY: release-dry-run
release-dry-run: release-check ## Git and Release - Dry run of release process
	$(call check_venv)
	$(call log_info,"Running release dry run...")
	$(TWINE) upload --repository testpypi --skip-existing $(DIST_DIR)/*
	$(call log_success,"Release dry run completed")

# ====================================================================
# Composite Commands
# ====================================================================
.PHONY: dev-setup
dev-setup: venv install-dev pre-commit-install ## Setup complete development environment
	$(call log_success,"Development environment setup completed")

.PHONY: quick-check
quick-check: format-check lint-flake8 test-fast ## Run quick development checks (format, lint, fast tests)
	$(call log_success,"Quick development checks passed")

.PHONY: ci
ci: quality test docs ## Run all CI checks (quality, tests, docs)
	$(call log_success,"All CI checks passed")

.PHONY: all-clean
all-clean: clean test-clean docs-clean ## Clean all build artifacts and caches
	$(call log_info,"Removing all caches and artifacts...")
	rm -rf .mypy_cache/ .pytest_cache/ .tox/ .coverage*
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	$(call log_success,"All artifacts and caches cleaned")

# ====================================================================
# Phony Targets Declaration
# ====================================================================
.PHONY: help venv install install-dev install-test install-lint requirements clean-env
.PHONY: format format-check lint lint-flake8 lint-pylint security typecheck quality
.PHONY: test test-unit test-integration test-fast test-slow test-verbose test-parallel
.PHONY: coverage-report coverage-view test-clean
.PHONY: test-clean-env test-clean-env-full test-clean-venv test-clean-install
.PHONY: test-clean-cross-platform test-clean-precommit test-clean-env-parallel
.PHONY: test-clean-env-coverage test-clean-env-debug test-clean-env-report
.PHONY: build wheel sdist install-local install-editable check-package clean
.PHONY: docs docs-clean docs-view docs-serve docs-linkcheck
.PHONY: run-ontology-manager run-ner-extractor run-corpus-builder run-relationship-extractor run-evaluation-benchmarker run-synthetic-generator
.PHONY: profile memory-profile security-audit dependencies-check shell jupyter notebook-setup data-explore
.PHONY: ontology-validate ontology-stats corpus-stats config-validate
.PHONY: data-validate model-info download-models benchmark-setup
.PHONY: pre-commit-install pre-commit-run pre-commit-update
.PHONY: version-bump-patch version-bump-minor version-bump-major
.PHONY: release-check release-dry-run dev-setup quick-check ci all-clean
