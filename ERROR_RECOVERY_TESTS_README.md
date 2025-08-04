# Error Recovery System Tests

This directory contains comprehensive unit tests for the error recovery functionality implemented in the parser system. The error recovery system provides robust error handling with configurable recovery strategies, detailed statistics tracking, and parser-specific recovery methods.

## Overview

The error recovery system includes:

- **Error Classification**: Categorizes errors as WARNING, RECOVERABLE, or FATAL
- **Recovery Strategies**: Six different strategies (SKIP, DEFAULT, RETRY, REPLACE, ABORT, CONTINUE)
- **Error Context Tracking**: Maintains detailed context about errors and recovery attempts
- **Parser-Specific Recovery**: Specialized recovery methods for OWL, CSV, and JSON-LD parsers
- **Statistics Collection**: Comprehensive tracking of errors and recovery success rates
- **Configuration Options**: Customizable recovery behavior through configuration settings

## Test Structure

### 1. Unit Tests (`tests/unit/test_error_recovery.py`)

Comprehensive unit tests covering all aspects of the error recovery system:

#### Test Classes:
- **TestErrorSeverityClassification**: Tests error severity classification logic
- **TestRecoveryStrategySelection**: Tests recovery strategy selection algorithms
- **TestAbstractParserErrorRecovery**: Tests base parser error recovery methods
- **TestOWLParserErrorRecovery**: Tests OWL parser-specific recovery methods
- **TestCSVParserErrorRecovery**: Tests CSV parser-specific recovery methods
- **TestJSONLDParserErrorRecovery**: Tests JSON-LD parser-specific recovery methods
- **TestErrorContextManagement**: Tests error context creation and tracking
- **TestErrorStatisticsTracking**: Tests statistics collection and reporting
- **TestRecoveryConfigurationOptions**: Tests configuration-driven recovery behavior
- **TestRecoveryFallbackBehaviors**: Tests edge cases and fallback scenarios

#### Key Test Methods:
```python
test_error_classification()
test_recovery_strategy_selection()
test_owl_parser_error_recovery()
test_csv_parser_field_mismatch_recovery()
test_jsonld_parser_malformed_json_recovery()
test_error_statistics_tracking()
test_recovery_configuration_options()
```

### 2. Integration Tests (`tests/unit/test_error_recovery_integration.py`)

End-to-end integration tests demonstrating the complete error recovery workflow:

#### Test Classes:
- **TestErrorRecoveryWorkflow**: Tests complete error recovery workflows
- **TestParserErrorRecoveryIntegration**: Tests parser-specific recovery integration
- **TestErrorRecoveryConfiguration**: Tests configuration-driven recovery behavior
- **TestErrorRecoveryPerformance**: Tests recovery system performance
- **TestErrorRecoveryReporting**: Tests error reporting and statistics collection

### 3. Demonstration Script (`demo_error_recovery.py`)

Interactive demonstration showing the error recovery system in action:

#### Demonstration Sections:
- Error severity classification examples
- Recovery strategy selection examples
- OWL parser error recovery scenarios
- CSV parser error recovery scenarios
- JSON-LD parser error recovery scenarios
- Error statistics and reporting
- Configuration impact examples

## Running the Tests

### Quick Start

Run all tests and demonstrations:
```bash
python run_error_recovery_tests.py
```

### Specific Test Suites

Run only unit tests:
```bash
python run_error_recovery_tests.py --unit
```

Run only integration tests:
```bash
python run_error_recovery_tests.py --integration
```

Run only the demonstration:
```bash
python run_error_recovery_tests.py --demo
```

### Advanced Options

Run with verbose output:
```bash
python run_error_recovery_tests.py --verbose
```

Generate detailed test report:
```bash
python run_error_recovery_tests.py --report
```

Show test overview:
```bash
python run_error_recovery_tests.py --overview
```

### Using pytest Directly

Run unit tests with pytest:
```bash
pytest tests/unit/test_error_recovery.py -v
```

Run integration tests with pytest:
```bash
pytest tests/unit/test_error_recovery_integration.py -v
```

Run specific test classes:
```bash
pytest tests/unit/test_error_recovery.py::TestErrorSeverityClassification -v
```

## Error Recovery Components

### Error Severity Levels

```python
class ErrorSeverity(Enum):
    WARNING = "warning"          # Minor issues that don't prevent processing
    RECOVERABLE = "recoverable"  # Errors that can be handled with fallback strategies
    FATAL = "fatal"             # Critical errors that prevent further processing
```

### Recovery Strategies

```python
class RecoveryStrategy(Enum):
    SKIP = "skip"               # Skip the problematic data/section
    DEFAULT = "default"         # Use default/fallback values
    RETRY = "retry"             # Retry with different parameters
    REPLACE = "replace"         # Replace with corrected data
    ABORT = "abort"             # Abort processing due to unrecoverable error
    CONTINUE = "continue"       # Continue processing despite the error
```

### Error Context

```python
@dataclass
class ErrorContext:
    error: Exception
    severity: ErrorSeverity
    location: str
    recovery_strategy: Optional[RecoveryStrategy] = None
    attempted_recoveries: List[RecoveryStrategy] = field(default_factory=list)
    recovery_data: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)
```

## Test Scenarios Covered

### 1. Error Classification Tests
- SyntaxError → RECOVERABLE
- ValueError → RECOVERABLE  
- KeyError (optional field) → WARNING
- MemoryError → FATAL
- TimeoutError → RECOVERABLE
- Unknown errors → RECOVERABLE (default)

### 2. Recovery Strategy Selection Tests
- Fatal errors → ABORT
- Warning errors → CONTINUE
- SyntaxError → SKIP then DEFAULT
- KeyError → DEFAULT then SKIP
- TimeoutError → RETRY (with retry limit)

### 3. OWL Parser Recovery Tests
- Malformed RDF/XML recovery
- Missing namespace handling
- Parser fallback (rdflib ↔ owlready2)
- Content sanitization
- Minimal ontology creation

### 4. CSV Parser Recovery Tests
- Field count mismatch recovery
- Bad row skipping
- Encoding error recovery
- Dialect detection failures
- Missing data inference

### 5. JSON-LD Parser Recovery Tests
- Malformed JSON recovery
- Missing @context handling
- JSON-LD expansion failures
- Content sanitization
- Minimal document creation

### 6. Integration Workflow Tests
- Complete error-to-recovery workflows
- Multiple error handling
- Configuration impact
- Performance characteristics
- Statistics aggregation

## Expected Test Results

### Unit Tests
The unit tests should pass with 100% success rate, covering:
- 50+ individual test methods
- All error classification scenarios
- All recovery strategy combinations
- All parser-specific recovery methods
- Error context and statistics functionality
- Configuration option handling

### Integration Tests  
The integration tests demonstrate realistic scenarios:
- End-to-end error recovery workflows
- Parser-specific recovery integration
- Performance impact measurement
- Comprehensive error reporting
- Configuration-driven behavior

### Demonstration Output
The demonstration script provides interactive examples showing:
- Error classification in action
- Strategy selection decision-making
- Parser-specific recovery examples
- Statistics and reporting capabilities
- Configuration impact examples

## Configuration Options

The error recovery system supports various configuration options:

```python
config = {
    "error_recovery": {
        "max_retries": 3,                    # Maximum retry attempts
        "skip_malformed_rows": True,         # Skip malformed data
        "use_default_values": True,          # Use default values for missing data
        "abort_on_fatal": True,              # Abort on fatal errors
        "collect_statistics": True,          # Collect error statistics
        "recovery_strategies": {             # Custom strategy mappings
            "SyntaxError": "skip",
            "ValueError": "default",
            "KeyError": "continue",
            "TimeoutError": "retry"
        }
    }
}
```

## Performance Characteristics

The error recovery system is designed to be:
- **Lightweight**: Minimal overhead during normal processing
- **Efficient**: Fast error classification and strategy selection
- **Scalable**: Handles large files and multiple errors gracefully
- **Configurable**: Adjustable performance/recovery trade-offs

Expected performance metrics:
- Error classification: < 1ms per error
- Recovery strategy selection: < 1ms per error
- Recovery overhead: < 100% of normal processing time
- Memory usage: < 10MB for statistics tracking

## Troubleshooting

### Common Issues

1. **Import Errors**: Ensure the parser classes are properly imported
   ```python
   from aim2_project.aim2_ontology.parsers import (
       ErrorSeverity, RecoveryStrategy, ErrorContext
   )
   ```

2. **Test Failures**: Check that all dependencies are installed
   ```bash
   pip install pytest pytest-mock
   ```

3. **Mock Issues**: The tests use mocking for TDD approach - actual parser implementations may differ

### Test Dependencies

Required packages:
- pytest
- unittest.mock (built-in)
- tempfile (built-in)
- json (built-in)
- csv (built-in)
- datetime (built-in)

Optional packages for enhanced testing:
- pytest-cov (for coverage reports)
- pytest-html (for HTML reports)
- pytest-json-report (for JSON reports)

## Contributing

When adding new error recovery tests:

1. **Follow Naming Conventions**: Use descriptive test method names
2. **Test All Scenarios**: Cover success, failure, and edge cases
3. **Use Proper Mocking**: Mock external dependencies appropriately
4. **Document Test Purpose**: Include clear docstrings
5. **Verify Integration**: Ensure tests work with the actual parser implementations

### Adding New Test Cases

```python
def test_new_error_scenario(self, mock_parser):
    """Test description of the new error scenario."""
    # Arrange
    error = NewErrorType("Error message")
    expected_strategy = RecoveryStrategy.SKIP
    
    # Act
    result = mock_parser.handle_error(error)
    
    # Assert
    assert result.recovery_strategy == expected_strategy
    mock_parser.handle_error.assert_called_once_with(error)
```

## Future Enhancements

Potential improvements to the error recovery system:
- Machine learning-based recovery strategy selection
- Adaptive retry intervals for network errors
- Custom recovery strategy plugins
- Real-time error recovery dashboards
- Recovery strategy effectiveness analytics
- Cross-parser error pattern analysis

---

For more information about the error recovery system implementation, see the parser source code in `aim2_project/aim2_ontology/parsers/`.