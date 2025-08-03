# Setup Environment Test Suite

This directory contains a comprehensive test suite for the `setup_env.py` virtual environment setup script. The test suite validates all functionality of the setup script to ensure reliable virtual environment creation and dependency installation across different platforms and scenarios.

## Overview

The `setup_env.py` script provides automated virtual environment setup for the AIM2 Ontology Information Extraction project. This test suite ensures that the script works correctly by testing:

- Virtual environment creation and management
- Dependency installation from requirements files
- Cross-platform compatibility (Windows, macOS, Linux)
- Error handling and edge cases
- Command-line argument parsing
- Progress indicators and user feedback
- Performance and timeout handling

## Test Files

### `test_setup_env.py`
The main comprehensive test suite containing:

- **Unit Tests**: Test individual components and methods
- **Integration Tests**: Test complete setup workflows
- **Cross-Platform Tests**: Validate functionality across different operating systems
- **Error Handling Tests**: Test edge cases and error conditions
- **Performance Tests**: Validate timeout and performance characteristics
- **Mock Tests**: Test with mocked external dependencies

### `run_setup_env_tests.py`
A specialized test runner script that provides:

- Easy execution of different test categories
- Coverage reporting
- HTML test reports
- Dependency checking
- Convenient command-line interface

### `test_setup_env_demo.py`
A demonstration script that shows how to use the test suite and validates basic functionality.

## Quick Start

### 1. Install Test Dependencies

```bash
pip install pytest pytest-mock pytest-cov pytest-html pytest-timeout
```

### 2. Run Quick Tests

```bash
# Run the demo to validate everything is working
python test_setup_env_demo.py

# Run quick unit tests only
python run_setup_env_tests.py --quick

# Run all tests with coverage
python run_setup_env_tests.py --full --coverage
```

### 3. Generate Reports

```bash
# Generate HTML test report
python run_setup_env_tests.py --full --html-report

# Generate coverage report
python run_setup_env_tests.py --unit --coverage
```

## Test Categories

### Unit Tests (`--unit`)
Test individual components of the setup script:
- `TestColors`: Color output functionality
- `TestProgressIndicator`: Progress display and feedback
- `TestVirtualEnvSetupInitialization`: Setup class initialization
- `TestVirtualEnvSetupMethods`: Individual method testing
- `TestArgumentParsing`: Command-line argument handling

### Integration Tests (`--integration`)
Test complete workflows:
- `TestVirtualEnvSetupIntegration`: Full setup process testing
- End-to-end virtual environment creation
- Complete dependency installation workflows

### Cross-Platform Tests (`--cross-platform`)
Test platform-specific functionality:
- `TestCrossPlatformCompatibility`: Platform-specific path handling
- Windows vs Unix executable paths
- Platform-specific activation instructions

### Error Handling Tests
Test edge cases and error conditions:
- `TestErrorHandling`: Exception handling and recovery
- Network timeouts
- Permission errors
- Missing dependencies
- Invalid Python versions

### Performance Tests (`--timeout`)
Test performance characteristics:
- `TestPerformanceAndTimeouts`: Timeout handling
- Long-running operation handling
- Resource usage validation

## Test Execution Options

### Quick Tests (Fast)
```bash
python run_setup_env_tests.py --quick
```
Runs only fast unit tests, excludes slow integration tests.

### Full Test Suite
```bash
python run_setup_env_tests.py --full
```
Runs all tests including integration and slow tests.

### Specific Categories
```bash
# Cross-platform tests only
python run_setup_env_tests.py --cross-platform

# Integration tests only
python run_setup_env_tests.py --integration

# Unit tests only
python run_setup_env_tests.py --unit
```

### With Coverage and Reports
```bash
# Generate coverage report
python run_setup_env_tests.py --full --coverage

# Generate HTML test report
python run_setup_env_tests.py --full --html-report

# Both coverage and HTML report
python run_setup_env_tests.py --full --coverage --html-report
```

### Using pytest Directly
```bash
# Run specific test class
pytest tests/test_setup_env.py::TestColors -v

# Run with specific markers
pytest tests/test_setup_env.py -m "unit and not slow" -v

# Run with coverage
pytest tests/test_setup_env.py --cov=setup_env --cov-report=html
```

## Test Markers

The test suite uses pytest markers to categorize tests:

- `unit`: Unit tests for individual components
- `integration`: Integration tests for complete workflows
- `slow`: Tests that take longer to run
- `cross_platform`: Cross-platform compatibility tests
- `mock_external`: Tests that mock external dependencies
- `timeout`: Tests that verify timeout behavior

## Test Configuration

### Environment Variables
```bash
# Set test environment (minimal, full, ci)
export TEST_ENV=full

# Enable verbose output
export PYTEST_VERBOSE=1
```

### pytest.ini Configuration
The project's `pytest.ini` file includes specific configuration for the setup_env tests:

```ini
markers =
    setup_env: Tests for setup_env.py virtual environment setup script
    cross_platform: Tests for cross-platform compatibility
    mock_external: Tests that mock external dependencies
    timeout: Tests that verify timeout behavior
```

## Mocking and External Dependencies

The test suite extensively uses mocking to avoid side effects:

- **subprocess calls**: Mocked to avoid actual command execution
- **file system operations**: Mocked for safety and speed
- **network operations**: Mocked to avoid external dependencies
- **virtual environment creation**: Can be mocked or use temporary directories

### Example Mock Usage
```python
@patch('subprocess.run')
def test_python_version_check(mock_subprocess):
    mock_subprocess.return_value.returncode = 0
    mock_subprocess.return_value.stdout = "Python 3.9.5"

    setup = VirtualEnvSetup(mock_args)
    result = setup.check_python_version()

    assert result is True
```

## Test Data and Fixtures

### Temporary Directories
Tests use temporary directories for safe file operations:

```python
@pytest.fixture
def temp_project_dir():
    with tempfile.TemporaryDirectory() as temp_dir:
        # Create test project structure
        yield Path(temp_dir)
```

### Mock Arguments
```python
@pytest.fixture
def mock_args():
    args = Mock()
    args.venv_name = "test_venv"
    args.verbose = False
    args.prod = False
    return args
```

## Troubleshooting

### Common Issues

1. **Missing Dependencies**
   ```bash
   python run_setup_env_tests.py --check-deps
   ```

2. **Import Errors**
   - Ensure `setup_env.py` is in the project root
   - Check that Python path includes the project directory

3. **Permission Errors**
   - Tests should use temporary directories
   - Ensure test runner has appropriate permissions

4. **Timeout Issues**
   - Adjust timeout settings in test configuration
   - Use `--verbose` to see where tests are hanging

### Debugging Tests

```bash
# Run with maximum verbosity
python run_setup_env_tests.py --unit --verbose

# Run specific test with debugging
pytest tests/test_setup_env.py::TestColors::test_colors_initialization -v -s

# Run with pdb on failure
pytest tests/test_setup_env.py --pdb
```

## Continuous Integration

### CI Configuration
The test suite is designed to work in CI environments:

```yaml
# Example GitHub Actions configuration
- name: Run setup_env tests
  run: |
    python run_setup_env_tests.py --full --coverage
    python -m pytest tests/test_setup_env.py -m "not slow" --junit-xml=test-results.xml
```

### Test Reports
- **Coverage reports**: Generated in `htmlcov/` directory
- **HTML test reports**: Generated in `test_reports/` directory
- **JUnit XML**: For CI integration

## Performance Considerations

### Test Execution Time
- **Quick tests**: ~30 seconds
- **Full test suite**: ~2-5 minutes
- **Integration tests**: ~1-3 minutes

### Optimization Tips
1. Use `--quick` for development
2. Run `--full` before commits
3. Use parallel execution: `pytest -n auto`
4. Cache test dependencies when possible

## Coverage Goals

The test suite aims for high coverage of the `setup_env.py` script:

- **Target coverage**: 90%+
- **Critical paths**: 100% (error handling, core functionality)
- **Platform-specific code**: Best effort coverage

### Coverage Reports
```bash
# Generate detailed coverage report
python run_setup_env_tests.py --full --coverage

# View HTML coverage report
open htmlcov/index.html
```

## Contributing

### Adding New Tests

1. **Follow naming conventions**: `test_<functionality>`
2. **Use appropriate markers**: Add pytest markers for categorization
3. **Mock external dependencies**: Avoid side effects
4. **Add docstrings**: Document test purpose and expectations
5. **Test error conditions**: Include negative test cases

### Test Categories Guidelines

- **Unit tests**: Test individual functions/methods
- **Integration tests**: Test complete workflows
- **Cross-platform tests**: Test platform-specific behavior
- **Error tests**: Test exception handling
- **Performance tests**: Test timeouts and resource usage

### Example New Test
```python
@pytest.mark.unit
@pytest.mark.setup_env
def test_new_functionality(self, setup_instance):
    """Test new functionality in setup_env.py."""
    # Arrange
    test_input = "test_value"

    # Act
    result = setup_instance.new_method(test_input)

    # Assert
    assert result is not None
    assert result.status == "success"
```

## Related Documentation

- `../setup_env.py`: The main setup script being tested
- `../run_instructions.txt`: Project setup instructions
- `../requirements-dev.txt`: Development dependencies
- `../pytest.ini`: pytest configuration

## Support

For issues with the test suite:

1. Check this documentation
2. Run the demo script: `python test_setup_env_demo.py`
3. Check dependencies: `python run_setup_env_tests.py --check-deps`
4. Review test output with `--verbose` flag
5. Check project issues or create a new one

---

This test suite ensures that the `setup_env.py` script is reliable, cross-platform compatible, and handles edge cases properly. Regular execution of these tests helps maintain code quality and prevents regressions in the virtual environment setup functionality.
