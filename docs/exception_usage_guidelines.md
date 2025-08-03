# AIM2 Exception Usage Guidelines

## Overview

This document provides comprehensive guidelines for using the AIM2 exception hierarchy effectively throughout the AIM2 project. It covers best practices, patterns, examples, and troubleshooting guidance for the structured exception system.

## Essential Imports

```python
# Essential imports for AIM2 exception system
from aim2_project.exceptions import (
    # Exception classes
    AIM2Exception, OntologyException, ExtractionException,
    LLMException, ValidationException,

    # Error code enums
    AIM2ErrorCodes, BaseErrorCodes, OntologyErrorCodes,
    ExtractionErrorCodes, LLMErrorCodes, ValidationErrorCodes,

    # Registry and formatting
    ErrorCodeRegistry, MessageFormat, MessageTemplateRegistry,
    ErrorSeverity, ErrorCodeInfo
)
```

## Table of Contents

1. [Exception Hierarchy Overview](#exception-hierarchy-overview)
2. [When to Use Which Exception](#when-to-use-which-exception)
3. [Error Code Usage](#error-code-usage)
4. [Exception Creation Patterns](#exception-creation-patterns)
5. [Exception Handling Best Practices](#exception-handling-best-practices)
6. [Context Information Guidelines](#context-information-guidelines)
7. [Message Formatting](#message-formatting)
8. [Advanced Message Formatting](#advanced-message-formatting)
9. [Serialization and Logging](#serialization-and-logging)
10. [Testing Exception Handling](#testing-exception-handling)
11. [Production Deployment Considerations](#production-deployment-considerations)
12. [Migration from Standard Exceptions](#migration-from-standard-exceptions)
13. [Common Pitfalls and Solutions](#common-pitfalls-and-solutions)
14. [Quick Reference](#quick-reference)

## Exception Hierarchy Overview

The AIM2 project uses a structured exception hierarchy built around the base `AIM2Exception` class:

```
AIM2Exception (Base)
├── OntologyException (Ontology operations)
├── ExtractionException (Information extraction)
├── LLMException (LLM interface operations)
└── ValidationException (Data validation)
```

### Key Features

- **Error Code System**: Hierarchical error codes with severity levels
- **Context Information**: Structured debugging information
- **Cause Tracking**: Exception chaining for root cause analysis
- **Multiple Output Formats**: Console, API, JSON, Markdown formatting
- **Serialization Support**: Convert to/from dictionaries for persistence
- **Message Templates**: Consistent error message formatting

## When to Use Which Exception

### AIM2Exception (Base)
Use the base exception for:
- System-wide errors that don't fit other categories
- General application errors
- Configuration and initialization issues
- Cross-cutting concerns

```python
# System initialization failure
raise AIM2Exception(
    "Failed to initialize AIM2 system",
    error_code=BaseErrorCodes.SYSTEM_INIT_FAILURE,
    context={"missing_components": ["config", "database"]}
)
```

### OntologyException
Use for ontology-related operations:
- Ontology file parsing errors
- Ontology validation failures
- Ontology integration conflicts
- Ontology export/import issues

```python
# Ontology parsing error
raise OntologyException(
    "Failed to parse OWL ontology file",
    error_code=OntologyErrorCodes.OWL_PARSING_FAILURE,
    context={
        "file_path": "/path/to/ontology.owl",
        "line_number": 42,
        "syntax_error": "Invalid RDF/XML syntax"
    }
)
```

### ExtractionException
Use for information extraction operations:
- Text processing failures
- NER model errors
- Relationship extraction issues
- Corpus building problems

```python
# NER model loading failure
raise ExtractionException(
    "Failed to load NER model",
    error_code=ExtractionErrorCodes.NER_MODEL_LOADING_FAILURE,
    context={
        "model_path": "/models/bio-ner-bert",
        "model_format": "pytorch",
        "expected_files": ["model.bin", "config.json", "tokenizer.json"]
    }
)
```

### LLMException
Use for LLM interface operations:
- API request failures
- Authentication issues
- Model inference errors
- Response parsing problems

```python
# LLM API request failure
raise LLMException(
    "OpenAI API request failed",
    error_code=LLMErrorCodes.API_REQUEST_FAILED,
    context={
        "model": "gpt-3.5-turbo",
        "endpoint": "https://api.openai.com/v1/chat/completions",
        "status_code": 429,
        "retry_count": 3
    }
)
```

### ValidationException
Use for data validation operations:
- Schema validation failures
- Input parameter validation
- Business rule violations
- Data integrity issues

```python
# Configuration validation error
raise ValidationException(
    "Configuration schema validation failed",
    error_code=ValidationErrorCodes.SCHEMA_VALIDATION_FAILED,
    context={
        "config_file": "/etc/aim2/config.yaml",
        "validation_errors": [
            "Missing required field: database.host",
            "Invalid port number: -1"
        ]
    }
)
```

## Error Code Usage

### Using Enum-Based Error Codes (Recommended)

```python
from aim2_project.exceptions import OntologyException, OntologyErrorCodes

# Preferred approach using convenience classes
raise OntologyException(
    "Ontology merge conflict detected",
    error_code=OntologyErrorCodes.MERGE_CONFLICT,
    context={"conflicting_concepts": ["Protein", "Gene"]}
)

# Alternative using the main enum
raise OntologyException(
    "RDF extraction failed",
    error_code=OntologyErrorCodes.RDF_EXTRACTION_FAILURE,
    context={"namespace_issues": ["missing prefix", "invalid URI"]}
)
```

### Using String-Based Error Codes (Backward Compatible)

```python
# String-based codes still work
raise ExtractionException(
    "Text tokenization failed",
    error_code="AIM2_EXTR_NER_E_002",
    context={"text_encoding": "unknown", "text_length": 1024}
)
```

### Benefits of Enum-Based Codes

1. **Metadata Access**: Get severity, description, and resolution hints
2. **Type Safety**: IDE autocompletion and type checking
3. **Documentation**: Built-in documentation in error info
4. **Consistency**: Standardized error categorization

```python
try:
    # Some operation
    pass
except OntologyException as e:
    if e.error_info:
        print(f"Severity: {e.error_info.severity.name}")
        print(f"Description: {e.error_info.description}")
        print(f"Resolution: {e.error_info.resolution_hint}")
```

## Exception Creation Patterns

### Basic Exception Creation

```python
# Simple exception with message
raise AIM2Exception("Operation failed")

# Exception with error code
raise AIM2Exception("Database connection failed",
                   error_code="DB_CONNECTION_ERROR")
```

### Exception with Context

```python
# Rich context information for debugging
raise ExtractionException(
    "NER inference failed",
    error_code=ExtractionErrorCodes.ENTITY_RECOGNITION_TIMEOUT,
    context={
        "model_name": "bio-ner-bert",
        "input_text_length": 2048,
        "batch_size": 16,
        "timeout_seconds": 30,
        "entities_processed": 45,
        "memory_usage_mb": 512
    }
)
```

### Exception Chaining

```python
# Proper exception chaining preserves stack traces
try:
    result = external_api_call()
except requests.RequestException as e:
    raise LLMException(
        "Failed to communicate with LLM service",
        error_code=LLMErrorCodes.API_REQUEST_FAILED,
        cause=e,  # This preserves the original exception
        context={
            "service_url": "https://api.example.com",
            "request_id": "req_123456",
            "retry_attempts": 3
        }
    )
```

### Module-Specific Patterns

#### Configuration Errors
```python
try:
    config = load_config(config_path)
except FileNotFoundError as e:
    raise AIM2Exception(
        "Configuration file not found",
        error_code=BaseErrorCodes.CONFIG_FILE_NOT_FOUND,
        cause=e,
        context={"config_path": config_path, "search_paths": ["/etc", "~/.config"]}
    )
```

#### Ontology Processing Errors
```python
try:
    ontology = parse_owl_file(file_path)
except SyntaxError as e:
    raise OntologyException(
        "OWL file contains syntax errors",
        error_code=OntologyErrorCodes.OWL_PARSING_FAILURE,
        cause=e,
        context={
            "file_path": file_path,
            "file_size": os.path.getsize(file_path),
            "parser": "owlready2",
            "line_number": getattr(e, 'lineno', None)
        }
    )
```

## Exception Handling Best Practices

### Catching Specific Exceptions

```python
# Good: Catch specific exceptions
try:
    result = process_ontology(ontology_file)
except OntologyException as e:
    logger.error(f"Ontology processing failed: {e.get_detailed_message()}")
    # Handle ontology-specific error
except ValidationException as e:
    logger.error(f"Validation failed: {e.get_detailed_message()}")
    # Handle validation-specific error
except AIM2Exception as e:
    logger.error(f"General AIM2 error: {e.get_detailed_message()}")
    # Handle other AIM2 errors
```

### Avoiding Broad Exception Catching

```python
# Avoid: Too broad exception catching
try:
    result = process_data()
except Exception as e:  # Too broad!
    logger.error(f"Something went wrong: {e}")

# Better: Catch specific exceptions and let unexpected ones bubble up
try:
    result = process_data()
except (OntologyException, ExtractionException, ValidationException) as e:
    logger.error(f"Processing failed: {e.get_detailed_message()}")
    # Handle known exceptions
# Let other exceptions bubble up for investigation
```

### Re-raising Exceptions

```python
# Good: Add context when re-raising
try:
    process_batch(items)
except ExtractionException as e:
    # Add batch-specific context
    e.context.update({
        "batch_id": batch_id,
        "batch_size": len(items),
        "processed_count": processed_count
    })
    raise  # Re-raise with additional context
```

### Logging Exception Details

```python
import logging

logger = logging.getLogger(__name__)

try:
    result = risky_operation()
except AIM2Exception as e:
    # Log with different formats for different purposes
    logger.error(e.get_console_message())  # Human-readable
    logger.debug(e.get_structured_message())  # Detailed for debugging

    # Log as JSON for monitoring systems
    import json
    logger.info(json.dumps({
        "event": "exception_occurred",
        "exception_data": e.to_dict()
    }))
```

## Context Information Guidelines

### What to Include in Context

**Always Include:**
- Input parameters and their values
- File paths, URLs, and identifiers
- Current state information
- Configuration settings relevant to the error

**When Relevant:**
- Performance metrics (timing, memory usage)
- Progress indicators (items processed, remaining)
- External service information (endpoints, response codes)
- User or session identifiers

**Never Include:**
- Passwords, API keys, or other secrets
- Personal information (unless necessary for debugging)
- Large data structures (use summaries instead)

### Context Structure Examples

```python
# Good context structure
context = {
    # Operation identifiers
    "operation": "ontology_integration",
    "request_id": "req_12345",

    # Input information
    "source_ontology": "chebi.owl",
    "target_ontology": "plant_ontology.owl",
    "merge_strategy": "union",

    # State information
    "concepts_processed": 1542,
    "conflicts_found": 3,
    "processing_time_ms": 2340,

    # Error-specific details
    "conflicting_concepts": ["GO:0008150", "CHEBI:24431"],
    "conflict_type": "class_hierarchy_mismatch"
}
```

### Sensitive Information Handling

```python
# Bad: Exposing sensitive information
context = {
    "database_url": "postgresql://user:password@host:5432/db",  # Contains password!
    "api_key": "sk-1234567890abcdef"  # Exposed API key!
}

# Good: Safe context information
context = {
    "database_host": "db.example.com",
    "database_name": "aim2_db",
    "api_provider": "openai",
    "api_key_provided": True,  # Boolean instead of actual key
    "connection_encrypted": True
}
```

## Message Formatting

### Basic Message Formatting

```python
# Get formatted messages for different contexts
try:
    raise LLMException(
        "API rate limit exceeded",
        error_code=LLMErrorCodes.API_QUOTA_EXCEEDED,
        context={"requests_per_minute": 60, "limit": 50}
    )
except LLMException as e:
    # Basic detailed message (backward compatible)
    print(e.get_detailed_message())

    # String representation
    print(str(e))
```

### Custom Message Formatting

```python
try:
    raise ValidationException(
        "Schema validation failed",
        error_code=ValidationErrorCodes.SCHEMA_VALIDATION_FAILED
    )
except ValidationException as e:
    # Custom formatting with specific options
    detailed_message = e.format_message(
        MessageFormat.STRUCTURED,
        include_timestamp=True,
        include_context=True,
        include_resolution=True
    )
```

## Advanced Message Formatting

The AIM2 exception system provides sophisticated message formatting capabilities through built-in convenience methods and a flexible template system.

### Built-in Convenience Methods

```python
try:
    raise LLMException(
        "API rate limit exceeded",
        error_code=LLMErrorCodes.API_QUOTA_EXCEEDED,
        context={"requests_per_minute": 60, "limit": 50}
    )
except LLMException as e:
    # Console output with colors and symbols
    console_msg = e.get_console_message(include_colors=True)

    # Clean API response format
    api_response = e.get_api_message(include_context=False)

    # JSON for logging systems
    json_data = e.get_json_message(include_timestamp=True)

    # Structured format for debugging
    debug_msg = e.get_structured_message(include_timestamp=True)

    # Markdown for documentation
    markdown_report = e.get_markdown_message()
```

### Message Template System

```python
# Use the template registry for advanced formatting
from aim2_project.exceptions import MessageTemplateRegistry

registry = MessageTemplateRegistry()

# Set format preferences
registry.set_format_preference(MessageFormat.API, "compact")
registry.set_format_preference(MessageFormat.CONSOLE, "standard")

# Format using registry
formatted_msg = registry.format_message(
    message="Operation failed",
    error_code="AIM2_TEST_E_001",
    format_type=MessageFormat.CONSOLE,
    include_colors=True
)
```

### Output Format Examples

**Console Format (with colors):**
```
🔴 [CRITICAL] AIM2_LLM_API_E_003: API rate limit exceeded
Description: API quota exceeded
Context: requests_per_minute=60, limit=50
Resolution: Wait for quota reset or upgrade plan
```

**API Format (clean):**
```
[ERROR] API rate limit exceeded | Suggested action: Wait for quota reset or upgrade plan
```

**JSON Format:**
```json
{
  "error_code": "AIM2_LLM_API_E_003",
  "message": "API rate limit exceeded",
  "severity": "ERROR",
  "module": "LLM",
  "category": "API",
  "description": "API quota exceeded",
  "resolution_hint": "Wait for quota reset or upgrade plan",
  "context": {
    "requests_per_minute": 60,
    "limit": 50
  }
}
```

**Structured Format (for logging):**
```
Severity: ERROR
Code: AIM2_LLM_API_E_003
Module: LLM
Category: API
Message: API rate limit exceeded
Description: API quota exceeded
Context:
  requests_per_minute: 60
  limit: 50
Resolution: Wait for quota reset or upgrade plan
```

## Serialization and Logging

### Exception Serialization

```python
# Serialize exception for storage or transmission
try:
    risky_operation()
except AIM2Exception as e:
    # Convert to dictionary
    exception_data = e.to_dict()

    # Store in database, send to monitoring system, etc.
    store_exception(exception_data)

    # Later, recreate the exception
    restored_exception = AIM2Exception.from_dict(exception_data)
```

### Structured Logging

```python
import logging
import json

logger = logging.getLogger(__name__)

def log_exception(exception: AIM2Exception, operation: str):
    """Log exception with structured data for monitoring."""
    log_entry = {
        "timestamp": datetime.now().isoformat(),
        "operation": operation,
        "exception": exception.to_dict(),
        "severity": exception.error_info.severity.name if exception.error_info else "ERROR"
    }

    logger.error(json.dumps(log_entry, default=str))

try:
    process_documents()
except ExtractionException as e:
    log_exception(e, "document_processing")
```

### Integration with Monitoring Systems

```python
# Example integration with monitoring system
def report_exception_to_monitoring(exception: AIM2Exception):
    """Report exception to external monitoring system."""
    monitoring_data = {
        "error_code": exception.error_code,
        "module": exception.error_info.module if exception.error_info else "unknown",
        "severity": exception.error_info.severity.name if exception.error_info else "ERROR",
        "message": exception.message,
        "context": exception.context,
        "timestamp": time.time()
    }

    # Send to monitoring system (Sentry, DataDog, etc.)
    monitoring_client.capture_exception(monitoring_data)

try:
    critical_operation()
except AIM2Exception as e:
    report_exception_to_monitoring(e)
    raise  # Re-raise after reporting
```

## Testing Exception Handling

### Unit Testing Exceptions

```python
import pytest
from aim2_project.exceptions import OntologyException, OntologyErrorCodes

def test_ontology_parsing_error():
    """Test that ontology parsing raises appropriate exception."""
    with pytest.raises(OntologyException) as exc_info:
        parse_invalid_ontology("invalid.owl")

    # Check error code
    assert exc_info.value.error_code == OntologyErrorCodes.OWL_PARSING_FAILURE.value.code

    # Check context information
    assert "file_path" in exc_info.value.context
    assert exc_info.value.context["file_path"] == "invalid.owl"

    # Check error info is populated
    assert exc_info.value.error_info is not None
    assert exc_info.value.error_info.severity.name == "ERROR"

def test_exception_serialization():
    """Test that exceptions can be serialized and deserialized."""
    original = OntologyException(
        "Test error",
        error_code=OntologyErrorCodes.MERGE_CONFLICT,
        context={"test": "value"}
    )

    # Serialize
    data = original.to_dict()

    # Deserialize
    restored = OntologyException.from_dict(data)

    # Verify
    assert restored.message == original.message
    assert restored.error_code == original.error_code
    assert restored.context == original.context
```

### Integration Testing

```python
def test_ontology_pipeline_error_handling():
    """Test error handling in complete ontology processing pipeline."""
    with pytest.raises(OntologyException) as exc_info:
        pipeline = OntologyProcessingPipeline()
        pipeline.process_file("nonexistent.owl")

    # Verify error propagation through pipeline
    exception = exc_info.value
    assert exception.error_code == OntologyErrorCodes.OWL_PARSING_FAILURE.value.code
    assert "file_path" in exception.context

    # Test structured logging output
    log_message = exception.get_structured_message()
    assert "Timestamp:" in log_message
    assert "Module: ONTO" in log_message

def test_end_to_end_error_handling():
    """Test error handling in complete workflow."""
    with pytest.raises(ExtractionException) as exc_info:
        # Test complete extraction pipeline with known failure
        pipeline = ExtractionPipeline()
        pipeline.process_document("nonexistent.pdf")

    # Verify error propagation
    exception = exc_info.value
    assert "document_path" in exception.context
    assert exception.error_info.module == "EXTR"

    # Test error message formatting
    console_msg = exception.get_console_message()
    assert "nonexistent.pdf" in console_msg

    # Test serialization in error state
    serialized = exception.to_dict()
    assert serialized["exception_type"] == "ExtractionException"
```

## Production Deployment Considerations

### Performance Impact

The AIM2 exception system is designed for production use with minimal performance overhead:

- **Exception Creation**: Lazy loading of error info reduces initialization cost
- **Template Formatting**: Caching system for frequently used error patterns
- **Context Serialization**: Efficient JSON encoding with fallback handling
- **Memory Usage**: Minimal memory footprint through shared error code registry

```python
# Performance considerations
def process_large_dataset(items):
    """Example showing efficient exception handling in high-volume scenarios."""
    errors = []

    for item in items:
        try:
            process_item(item)
        except AIM2Exception as e:
            # Store exception data efficiently
            errors.append({
                "item_id": item.id,
                "error_code": e.error_code,
                "message": e.message,
                "context_summary": {k: v for k, v in e.context.items()
                                  if k in ["critical_field1", "critical_field2"]}
            })

            # Only log detailed info for critical errors
            if e.error_info and e.error_info.severity == ErrorSeverity.CRITICAL:
                logger.error(e.get_structured_message())
```

### Monitoring Integration

```python
# Example integration with monitoring systems
class ExceptionMonitor:
    def __init__(self, metrics_client, alert_client):
        self.metrics = metrics_client
        self.alerts = alert_client

    def report_exception(self, exc: AIM2Exception, operation: str):
        """Report exception to monitoring systems."""
        # Increment error metrics
        if exc.error_info:
            self.metrics.increment(f"error.{exc.error_info.module.lower()}")
            self.metrics.increment(f"error.{exc.error_info.category.lower()}")

            # Send alerts for critical errors
            if exc.error_info.severity == ErrorSeverity.CRITICAL:
                self.alerts.send_alert({
                    "error_code": exc.error_code,
                    "operation": operation,
                    "severity": exc.error_info.severity.name,
                    "context": exc.context
                })

# Usage in production code
monitor = ExceptionMonitor(metrics_client, alert_client)

try:
    critical_operation()
except AIM2Exception as e:
    monitor.report_exception(e, "critical_operation")
    raise  # Re-raise after monitoring
```

### Structured Logging for Production

```python
import logging
import json
from datetime import datetime

class ProductionExceptionLogger:
    def __init__(self, logger_name: str):
        self.logger = logging.getLogger(logger_name)

    def log_exception(self, exc: AIM2Exception, operation: str, user_id: str = None):
        """Log exception with production-ready structured data."""
        log_entry = {
            "timestamp": datetime.utcnow().isoformat(),
            "level": "ERROR",
            "operation": operation,
            "user_id": user_id,
            "exception": {
                "type": exc.__class__.__name__,
                "code": exc.error_code,
                "message": exc.message,
                "module": exc.error_info.module if exc.error_info else "unknown",
                "severity": exc.error_info.severity.name if exc.error_info else "ERROR",
                "context": exc.context
            }
        }

        self.logger.error(json.dumps(log_entry, default=str))

# Usage
prod_logger = ProductionExceptionLogger("aim2.production")

try:
    user_operation(user_id="user123")
except ValidationException as e:
    prod_logger.log_exception(e, "user_validation", user_id="user123")
```

### Error Code Registry Analytics

```python
# Analyze error patterns in production
registry = ErrorCodeRegistry()

def analyze_error_patterns():
    """Generate error analytics for production monitoring."""
    stats = registry.get_statistics()

    print(f"Total error codes defined: {stats['total_codes']}")
    print(f"Modules: {', '.join(stats['modules'])}")

    # Analyze by severity
    for severity, count in stats['severity_distribution'].items():
        print(f"{severity} errors: {count}")

    # Get critical errors for special monitoring
    critical_errors = registry.get_codes_by_severity(ErrorSeverity.CRITICAL)
    print(f"Critical errors to monitor: {len(critical_errors)}")

    return {
        "critical_codes": [err.code for err in critical_errors],
        "modules": stats['modules'],
        "total_codes": stats['total_codes']
    }
```

## Migration from Standard Exceptions

### Gradual Migration Strategy

1. **Phase 1**: Add AIM2 exception imports alongside existing code
2. **Phase 2**: Replace critical path exceptions with AIM2 exceptions
3. **Phase 3**: Migrate remaining exceptions module by module
4. **Phase 4**: Remove legacy exception handling code

### Migration Examples

#### Before (Standard Exceptions)
```python
def load_config(config_path):
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found: {config_path}")

    try:
        with open(config_path) as f:
            return yaml.safe_load(f)
    except yaml.YAMLError as e:
        raise ValueError(f"Invalid YAML syntax: {e}")
```

#### After (AIM2 Exceptions)
```python
from aim2_project.exceptions import AIM2Exception, BaseErrorCodes

def load_config(config_path):
    if not os.path.exists(config_path):
        raise AIM2Exception(
            "Configuration file not found",
            error_code=BaseErrorCodes.CONFIG_FILE_NOT_FOUND,
            context={
                "config_path": config_path,
                "current_directory": os.getcwd(),
                "search_paths": get_config_search_paths()
            }
        )

    try:
        with open(config_path) as f:
            return yaml.safe_load(f)
    except yaml.YAMLError as e:
        raise AIM2Exception(
            "Configuration file contains invalid YAML syntax",
            error_code=BaseErrorCodes.INVALID_CONFIG_SYNTAX,
            cause=e,
            context={
                "config_path": config_path,
                "yaml_error": str(e),
                "line_number": getattr(e, 'problem_mark', {}).get('line', None)
            }
        )
```

### Wrapper Pattern for Legacy Code

```python
def wrap_legacy_exceptions(func):
    """Decorator to wrap legacy exceptions with AIM2 exceptions."""
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except FileNotFoundError as e:
            raise AIM2Exception(
                "File not found",
                error_code=BaseErrorCodes.FILE_PERMISSION_DENIED,
                cause=e,
                context={"function": func.__name__, "args": args}
            )
        except PermissionError as e:
            raise AIM2Exception(
                "Permission denied",
                error_code=BaseErrorCodes.FILE_PERMISSION_DENIED,
                cause=e,
                context={"function": func.__name__, "args": args}
            )
    return wrapper

@wrap_legacy_exceptions
def legacy_file_operation(file_path):
    # Legacy code that raises standard exceptions
    with open(file_path, 'r') as f:
        return f.read()
```

## Common Pitfalls and Solutions

### Pitfall 1: Losing Original Stack Traces

**Problem:**
```python
# Bad: Original stack trace is lost
try:
    risky_operation()
except Exception as e:
    raise AIM2Exception("Operation failed")  # Original exception lost!
```

**Solution:**
```python
# Good: Preserve original exception
try:
    risky_operation()
except Exception as e:
    raise AIM2Exception(
        "Operation failed",
        cause=e  # Preserves original exception and stack trace
    )
```

### Pitfall 2: Including Sensitive Information in Context

**Problem:**
```python
# Bad: Exposing credentials
raise LLMException(
    "API authentication failed",
    context={
        "api_key": "sk-secret123",  # Exposed!
        "username": "user@example.com",
        "password": "secret_password"  # Exposed!
    }
)
```

**Solution:**
```python
# Good: Safe context information
raise LLMException(
    "API authentication failed",
    error_code=LLMErrorCodes.INVALID_API_KEY,
    context={
        "api_provider": "openai",
        "api_key_provided": True,
        "api_key_prefix": "sk-xxx",  # Only show prefix
        "authentication_method": "bearer_token"
    }
)
```

### Pitfall 3: Creating Exceptions with No Context

**Problem:**
```python
# Bad: No helpful debugging information
raise OntologyException("Parsing failed")
```

**Solution:**
```python
# Good: Rich context for debugging
raise OntologyException(
    "Ontology parsing failed",
    error_code=OntologyErrorCodes.OWL_PARSING_FAILURE,
    context={
        "file_path": ontology_path,
        "file_size": os.path.getsize(ontology_path),
        "parser": "owlready2",
        "parsing_stage": "class_hierarchy_construction",
        "classes_processed": 150,
        "total_classes": 200
    }
)
```

### Pitfall 4: Incorrect Exception Type Selection

**Problem:**
```python
# Bad: Using wrong exception type
raise ValidationException("Failed to load NER model")  # Wrong type!
```

**Solution:**
```python
# Good: Correct exception type
raise ExtractionException(
    "Failed to load NER model",
    error_code=ExtractionErrorCodes.NER_MODEL_LOADING_FAILURE,
    context={"model_path": model_path, "model_format": "pytorch"}
)
```

### Pitfall 5: Not Using Error Codes Consistently

**Problem:**
```python
# Inconsistent error code usage
raise AIM2Exception("Config error", error_code="CONFIG_ERR")  # Non-standard
raise AIM2Exception("Another config error", error_code="CFG_ERROR_001")  # Different format
```

**Solution:**
```python
# Consistent error code usage
raise AIM2Exception(
    "Configuration file not found",
    error_code=BaseErrorCodes.CONFIG_FILE_NOT_FOUND  # Standard enum
)

raise AIM2Exception(
    "Invalid configuration syntax",
    error_code=BaseErrorCodes.INVALID_CONFIG_SYNTAX  # Standard enum
)
```

## Best Practices Summary

1. **Use Appropriate Exception Types**: Choose the most specific exception type for your error
2. **Include Rich Context**: Provide detailed context information for debugging
3. **Use Error Codes**: Leverage the structured error code system for categorization
4. **Preserve Stack Traces**: Always use the `cause` parameter when wrapping exceptions
5. **Protect Sensitive Data**: Never include passwords, API keys, or personal data in context
6. **Test Exception Handling**: Write unit tests for exception scenarios
7. **Use Consistent Formatting**: Leverage built-in message formatters for consistency
8. **Document Custom Exceptions**: Add clear docstrings explaining when to use custom exceptions
9. **Log Appropriately**: Use structured logging with exception serialization
10. **Monitor and Alert**: Integrate with monitoring systems for production error tracking

Following these guidelines will ensure consistent, debuggable, and maintainable exception handling throughout the AIM2 project.

## Quick Reference

### Common Exception Patterns

```python
# Basic exception with error code
raise OntologyException(
    "Ontology parsing failed",
    error_code=OntologyErrorCodes.OWL_PARSING_FAILURE,
    context={"file": "ontology.owl"}
)

# Exception chaining
try:
    risky_operation()
except ValueError as e:
    raise ExtractionException(
        "Processing failed",
        error_code=ExtractionErrorCodes.NER_MODEL_LOADING_FAILURE,
        cause=e,
        context={"model": "bert-ner"}
    )

# Get formatted output
try:
    operation()
except AIM2Exception as e:
    console_msg = e.get_console_message()  # Human-readable
    json_data = e.get_json_message()       # For logging
    api_msg = e.get_api_message()          # Clean API response
```

### Error Code Usage

```python
# Enum-based (recommended)
error_code=OntologyErrorCodes.MERGE_CONFLICT

# String-based (backward compatible)
error_code="AIM2_ONTO_INTG_E_001"

# Access error info
if exception.error_info:
    severity = exception.error_info.severity.name
    resolution = exception.error_info.resolution_hint
```

### Context Guidelines

```python
# Good context
context = {
    "file_path": "/path/to/file.owl",
    "line_number": 42,
    "expected_format": "RDF/XML",
    "actual_format": "turtle"
}

# Avoid sensitive data
context = {
    "api_provider": "openai",
    "api_key_provided": True,  # Don't include actual key
    "model": "gpt-3.5-turbo"
}
```

### Testing Patterns

```python
# Test exception type and error code
with pytest.raises(OntologyException) as exc_info:
    parse_invalid_ontology()

assert exc_info.value.error_code == OntologyErrorCodes.OWL_PARSING_FAILURE.value.code
assert "file_path" in exc_info.value.context

# Test serialization
data = exception.to_dict()
restored = OntologyException.from_dict(data)
```

### Production Logging

```python
# Structured logging
logger.error(exception.get_structured_message())

# JSON logging for monitoring
logger.info(json.dumps({
    "event": "exception_occurred",
    "data": exception.to_dict()
}))

# Performance-conscious logging
if exception.error_info.severity == ErrorSeverity.CRITICAL:
    logger.error(exception.get_detailed_message())
```
