# Setup.py Test Suite

This directory contains comprehensive unit and integration tests for the setup.py functionality of the AIM2 Ontology Information Extraction project.

## Overview

The test suite is designed following Test-Driven Development (TDD) principles and validates all aspects of the setup.py configuration including:

- Package metadata validation
- Dependency management and resolution
- Package structure and imports
- Installation process validation
- Python version compatibility
- Build and distribution processes

## Test Structure

```
tests/
├── __init__.py                     # Test package marker
├── README.md                       # This documentation
├── pytest.ini                     # Pytest configuration (in project root)
├── test_runner.py                  # Convenient test runner script
├── unit/                          # Unit tests
│   ├── __init__.py
│   ├── test_setup.py              # Main setup.py functionality tests
│   └── test_dependencies.py       # Dependency-specific tests
├── integration/                   # Integration tests
│   ├── __init__.py
│   └── test_setup_integration.py  # End-to-end installation tests
└── fixtures/                     # Test fixtures and utilities
    └── conftest.py               # Pytest fixtures and configuration
```

## Test Categories

### Unit Tests (`tests/unit/`)

#### `test_setup.py`
- **TestSetupMetadata**: Validates package metadata (name, version, description, author, etc.)
- **TestSetupDependencies**: Tests dependency format and structure
- **TestPackageStructure**: Validates package discovery and data inclusion
- **TestInstallationProcess**: Tests installation workflow
- **TestPythonCompatibility**: Validates Python version support
- **TestSetupConfiguration**: Tests build configuration

#### `test_dependencies.py`
- **TestDependencyValidation**: Validates dependency specifications
- **TestDependencyInstallation**: Tests installation processes
- **TestDependencyConstraints**: Tests version compatibility
- **TestDependencyResolution**: Tests conflict resolution
- **TestOptionalDependencies**: Tests extras_require functionality

### Integration Tests (`tests/integration/`)

#### `test_setup_integration.py`
- **TestSetupIntegration**: End-to-end installation and usage tests
- **TestSetupWithRealDependencies**: Tests with actual package installation
- **TestSetupCompatibility**: Cross-platform and version compatibility

## Running Tests

### Using the Test Runner (Recommended)

```bash
# Run all tests
python tests/test_runner.py

# Run only unit tests
python tests/test_runner.py --category unit

# Run only integration tests
python tests/test_runner.py --category integration

# Run specific test categories
python tests/test_runner.py --category setup
python tests/test_runner.py --category dependencies
python tests/test_runner.py --category packaging

# Run with coverage
python tests/test_runner.py --coverage

# Skip slow tests
python tests/test_runner.py --fast

# Verbose output
python tests/test_runner.py --verbose
```

### Using pytest Directly

```bash
# Run all tests
pytest

# Run unit tests only
pytest tests/unit/

# Run integration tests only
pytest tests/integration/

# Run with coverage
pytest --cov=aim2_project --cov-report=term-missing

# Run specific test file
pytest tests/unit/test_setup.py

# Run specific test class
pytest tests/unit/test_setup.py::TestSetupMetadata

# Run specific test method
pytest tests/unit/test_setup.py::TestSetupMetadata::test_package_name_is_valid

# Skip slow tests
pytest -m "not slow"

# Run only setup-related tests
pytest -m setup

# Run with verbose output
pytest -v
```

## Test Markers

The test suite uses pytest markers to categorize tests:

- `@pytest.mark.unit`: Unit tests
- `@pytest.mark.integration`: Integration tests
- `@pytest.mark.slow`: Tests that take significant time
- `@pytest.mark.requires_network`: Tests requiring internet access
- `@pytest.mark.setup`: Setup.py related tests
- `@pytest.mark.dependencies`: Dependency management tests
- `@pytest.mark.packaging`: Packaging and distribution tests

## Test Dependencies

The test suite requires the following packages:

```
pytest>=7.0.0
pytest-cov>=3.0.0
pytest-mock>=3.7.0
packaging>=21.0
```

For integration tests that install packages:
```
wheel>=0.37.0
setuptools>=60.0.0
```

## TDD Workflow

These tests are designed to support Test-Driven Development:

1. **Red Phase**: Tests initially fail because setup.py doesn't exist
2. **Green Phase**: Create setup.py to make tests pass
3. **Refactor Phase**: Improve setup.py while keeping tests green

### Expected Test Behavior Before setup.py Creation

- Most tests will be **skipped** with messages like "setup.py not yet created"
- Metadata validation tests will **pass** by validating expected values
- Dependency format tests will **pass** by validating requirement strings
- Package structure tests will **pass** if the package structure exists

### Expected Test Behavior After setup.py Creation

- All tests should **pass** if setup.py is correctly implemented
- Integration tests will perform actual installation and validation
- Build tests will create distributions and verify contents

## Test Configuration

### pytest.ini

The project includes a comprehensive pytest configuration:

- Test discovery patterns
- Coverage reporting
- Custom markers
- Warning filters
- Minimum Python version requirements

### Coverage Requirements

- Minimum coverage: 80%
- Coverage reports generated in multiple formats (terminal, HTML, XML)
- Excludes test files and non-critical paths

## Writing New Tests

When adding new tests for setup.py functionality:

1. **Add unit tests first** in `tests/unit/`
2. **Use appropriate markers** to categorize tests
3. **Mock external dependencies** when possible
4. **Add integration tests** for end-to-end validation
5. **Update this documentation** if adding new test categories

### Test Naming Conventions

- Test files: `test_*.py`
- Test classes: `Test*`
- Test methods: `test_*`
- Use descriptive names that explain what is being tested

### Example Test Structure

```python
class TestNewFeature:
    """Test new setup.py feature."""

    def test_feature_basic_functionality(self):
        """Test that feature works in basic case."""
        # Arrange
        expected_value = "expected"

        # Act
        result = function_under_test()

        # Assert
        assert result == expected_value

    def test_feature_edge_case(self):
        """Test feature handles edge case correctly."""
        # Test implementation
        pass

    @pytest.mark.slow
    def test_feature_integration(self):
        """Test feature in integration scenario."""
        # Test implementation
        pass
```

## Common Issues and Solutions

### Tests Skipped Due to Missing setup.py

This is expected behavior in TDD. Tests are designed to be skipped until setup.py is created.

### Import Errors for aim2_project

If tests fail with import errors:
1. Ensure `__init__.py` files exist in all package directories
2. Run tests from the project root directory
3. Check that PYTHONPATH includes the project root

### Dependency Installation Failures

For integration tests that install packages:
1. Ensure you have write permissions to the test environment
2. Check network connectivity for package downloads
3. Consider using `--fast` flag to skip slow dependency tests

### Coverage Below Threshold

If coverage falls below 80%:
1. Add tests for uncovered code paths
2. Update coverage exclusions in pytest.ini if appropriate
3. Consider if certain code paths should be tested

## Continuous Integration

The test suite is designed to work in CI environments:

- All external dependencies are mocked in unit tests
- Integration tests can be skipped in restricted environments
- Coverage reports are generated in CI-friendly formats
- Tests are categorized to allow selective execution

### CI Configuration Example

```yaml
# Example GitHub Actions configuration
- name: Run unit tests
  run: python tests/test_runner.py --category unit --coverage

- name: Run integration tests
  run: python tests/test_runner.py --category integration --fast
```

## Troubleshooting

### Debug Test Failures

```bash
# Run with maximum verbosity
pytest -vv -s

# Run specific failing test with debug output
pytest -vv -s tests/unit/test_setup.py::TestSetupMetadata::test_package_name_is_valid

# Run with pytest debugging
pytest --pdb

# Run with coverage debug
pytest --cov-report=term-missing --cov-report=html
```

### Environment Issues

```bash
# Check Python version
python --version

# Check pytest installation
python -m pytest --version

# Check if aim2_project can be imported
python -c "import aim2_project"

# List installed packages
pip list
```

## Contributing

When contributing to the test suite:

1. Follow the existing test structure and conventions
2. Add appropriate markers to new tests
3. Update documentation for new test categories
4. Ensure tests work in both TDD and post-implementation phases
5. Add integration tests for complex functionality
6. Consider cross-platform compatibility

For questions about the test suite, refer to the project's CONTRIBUTING.md file or contact the development team.
