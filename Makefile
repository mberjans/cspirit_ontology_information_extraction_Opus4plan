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
PYTHON_VERSION := 3.8
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
	@echo "Git and Release:"
	@awk 'BEGIN {FS = ":.*?## "} /^[a-zA-Z_-]+:.*?## / && /Git and Release/ {found=1} found && /^[a-zA-Z_-]+:.*?## / {printf "  %-20s %s\n", $$1, $$2} /^$$/ {found=0}' $(MAKEFILE_LIST)

.PHONY: venv
venv: ## Environment Setup - Create virtual environment
	$(call log_info,"Creating virtual environment...")
	python$(PYTHON_VERSION) -m venv $(VENV_PATH)
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

.PHONY: jupyter
jupyter: ## Development Utilities - Start Jupyter notebook server
	$(call check_venv)
	$(call log_info,"Starting Jupyter notebook server...")
	$(VENV_PATH)/bin/jupyter lab

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
.PHONY: build wheel sdist install-local install-editable check-package clean
.PHONY: docs docs-clean docs-view docs-serve docs-linkcheck
.PHONY: run-ontology-manager run-ner-extractor run-corpus-builder profile memory-profile
.PHONY: security-audit dependencies-check shell jupyter
.PHONY: pre-commit-install pre-commit-run pre-commit-update
.PHONY: version-bump-patch version-bump-minor version-bump-major
.PHONY: release-check release-dry-run dev-setup ci all-clean