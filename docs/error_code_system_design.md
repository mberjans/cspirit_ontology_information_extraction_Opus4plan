# AIM2 Error Code System Design Document

## Executive Summary

This document presents a comprehensive design for a structured error code enumeration system for the AIM2 (Artificial Intelligence for Medical Ontology Information Extraction) project. The system builds upon the existing AIM2Exception hierarchy to provide systematic error categorization, severity classification, and maintainable error handling across all project modules.

## 1. Current System Analysis

### 1.1 Existing Exception Hierarchy
The current system includes:
- **AIM2Exception**: Base exception class with error code system
- **OntologyException**: Errors related to ontology operations
- **ExtractionException**: Errors related to information extraction
- **LLMException**: Errors related to LLM interface operations
- **ValidationException**: Errors related to data validation

### 1.2 Current Error Code Pattern
- Default codes follow pattern: `AIM2_{MODULE}_ERROR`
- Examples: `AIM2_ONTOLOGY_ERROR`, `AIM2_EXTRACTION_ERROR`
- Custom error codes supported but lack structured categorization
- No severity levels or systematic numbering scheme

### 1.3 Identified Gaps
- Lack of hierarchical error categorization
- No severity level classification
- Missing standardized numbering system
- No systematic documentation structure
- Limited extensibility for new error types

## 2. Error Code Structure Design

### 2.1 Hierarchical Code Format
```
AIM2_{MODULE}_{CATEGORY}_{SEVERITY}_{NUMBER}
```

**Components:**
- **AIM2**: Project prefix (constant)
- **MODULE**: Module identifier (4-6 characters)
- **CATEGORY**: Error category (3-4 characters)
- **SEVERITY**: Severity level (1 character)
- **NUMBER**: Sequential number (3 digits)

### 2.2 Module Identifiers
| Module | Identifier | Description |
|--------|------------|-------------|
| Base/General | BASE | System-wide, configuration, and general errors |
| Ontology | ONTO | Ontology management, parsing, validation, integration |
| Extraction | EXTR | Information extraction, NER, relationship extraction |
| LLM | LLM | Large Language Model interface and operations |
| Validation | VALID | Data validation, schema validation, constraints |
| Utils | UTILS | Utility functions, logging, performance monitoring |

### 2.3 Category Codes by Module

#### 2.3.1 Base Module Categories
| Category | Code | Description |
|----------|------|-------------|
| System | SYS | System-level errors, initialization failures |
| Configuration | CFG | Configuration loading, parsing, validation |
| Persistence | PER | Database, file I/O, storage operations |
| Network | NET | Network connectivity, API communication |
| Security | SEC | Authentication, authorization, security violations |

#### 2.3.2 Ontology Module Categories
| Category | Code | Description |
|----------|------|-------------|
| Parsing | PARS | Ontology file parsing (OWL, RDF, JSON-LD) |
| Validation | VALD | Ontology structure and semantic validation |
| Integration | INTG | Ontology merging, alignment, integration |
| Export | EXPO | Ontology export and serialization |
| Query | QURY | Ontology querying and reasoning |
| Management | MGMT | Ontology lifecycle management |

#### 2.3.3 Extraction Module Categories
| Category | Code | Description |
|----------|------|-------------|
| NER | NER | Named Entity Recognition processing |
| Relation | REL | Relationship extraction operations |
| Corpus | CORP | Corpus building and management |
| Processing | PROC | Text processing and preprocessing |
| Mapping | MAP | Entity-to-ontology mapping |
| Evaluation | EVAL | Benchmarking and evaluation |

#### 2.3.4 LLM Module Categories
| Category | Code | Description |
|----------|------|-------------|
| API | API | LLM API communication and requests |
| Authentication | AUTH | API key, token, authentication issues |
| Model | MDL | Model loading, configuration, inference |
| Timeout | TIME | Request timeouts and delays |
| Rate Limiting | RATE | Rate limiting and quota issues |
| Response | RESP | Response parsing and validation |

#### 2.3.5 Validation Module Categories
| Category | Code | Description |
|----------|------|-------------|
| Schema | SCHM | JSON/YAML schema validation |
| Data | DATA | Data format and structure validation |
| Business | BIZ | Business rule and logic validation |
| Constraint | CONS | Database and referential constraints |
| Input | INPT | User input and parameter validation |

### 2.4 Severity Levels
| Level | Code | Description | Use Cases |
|-------|------|-------------|-----------|
| Critical | C | System failure, data corruption | System crashes, data loss |
| Error | E | Operation failure, recovery possible | Failed operations, invalid inputs |
| Warning | W | Potential issues, operation continues | Deprecated features, minor issues |
| Info | I | Informational, for tracking | Status updates, progress indicators |

### 2.5 Error Code Examples
```
AIM2_BASE_SYS_C_001    # Critical system initialization failure
AIM2_ONTO_PARS_E_001   # Error parsing OWL file
AIM2_EXTR_NER_E_002    # Error in NER model inference
AIM2_LLM_API_E_003     # Error in LLM API request
AIM2_VALID_SCHM_W_001  # Warning for deprecated schema field
```

## 3. Error Code Enumeration Implementation

### 3.1 Enum Structure Design
```python
from enum import Enum
from typing import NamedTuple

class ErrorSeverity(Enum):
    CRITICAL = "C"
    ERROR = "E"
    WARNING = "W"
    INFO = "I"

class ErrorCodeInfo(NamedTuple):
    code: str
    module: str
    category: str
    severity: ErrorSeverity
    description: str
    resolution_hint: str

class AIM2ErrorCodes(Enum):
    # Base Module Errors
    BASE_SYS_C_001 = ErrorCodeInfo(
        code="AIM2_BASE_SYS_C_001",
        module="BASE",
        category="SYS",
        severity=ErrorSeverity.CRITICAL,
        description="System initialization failure",
        resolution_hint="Check system dependencies and configuration"
    )

    # Ontology Module Errors
    ONTO_PARS_E_001 = ErrorCodeInfo(
        code="AIM2_ONTO_PARS_E_001",
        module="ONTO",
        category="PARS",
        severity=ErrorSeverity.ERROR,
        description="Failed to parse OWL ontology file",
        resolution_hint="Verify file format and syntax"
    )

    # Additional error codes...
```

### 3.2 Error Registry System
```python
class ErrorCodeRegistry:
    """Central registry for all AIM2 error codes with lookup capabilities."""

    def __init__(self):
        self._codes = {}
        self._register_all_codes()

    def get_error_info(self, code: str) -> ErrorCodeInfo:
        """Get error information by code string."""
        return self._codes.get(code)

    def get_codes_by_module(self, module: str) -> List[ErrorCodeInfo]:
        """Get all error codes for a specific module."""
        return [info for info in self._codes.values() if info.module == module]

    def get_codes_by_severity(self, severity: ErrorSeverity) -> List[ErrorCodeInfo]:
        """Get all error codes of a specific severity level."""
        return [info for info in self._codes.values() if info.severity == severity]
```

## 4. Detailed Error Code Catalog

### 4.1 Base Module Error Codes

#### System Errors (BASE_SYS)
| Code | Severity | Description | Resolution Hint |
|------|----------|-------------|-----------------|
| AIM2_BASE_SYS_C_001 | Critical | System initialization failure | Check dependencies and permissions |
| AIM2_BASE_SYS_C_002 | Critical | Memory allocation failure | Increase available memory or reduce load |
| AIM2_BASE_SYS_E_003 | Error | Module import failure | Verify module installation and Python path |
| AIM2_BASE_SYS_E_004 | Error | Environment variable missing | Set required environment variables |

#### Configuration Errors (BASE_CFG)
| Code | Severity | Description | Resolution Hint |
|------|----------|-------------|-----------------|
| AIM2_BASE_CFG_E_001 | Error | Configuration file not found | Create or specify correct config file path |
| AIM2_BASE_CFG_E_002 | Error | Invalid configuration syntax | Check YAML/JSON syntax and structure |
| AIM2_BASE_CFG_E_003 | Error | Missing required configuration field | Add missing configuration parameters |
| AIM2_BASE_CFG_W_004 | Warning | Deprecated configuration field used | Update to new configuration schema |

#### Persistence Errors (BASE_PER)
| Code | Severity | Description | Resolution Hint |
|------|----------|-------------|-----------------|
| AIM2_BASE_PER_C_001 | Critical | Database connection failure | Check database server and credentials |
| AIM2_BASE_PER_E_002 | Error | File system permission denied | Verify file/directory permissions |
| AIM2_BASE_PER_E_003 | Error | Disk space insufficient | Free up disk space or change location |
| AIM2_BASE_PER_E_004 | Error | File corruption detected | Restore from backup or recreate file |

### 4.2 Ontology Module Error Codes

#### Parsing Errors (ONTO_PARS)
| Code | Severity | Description | Resolution Hint |
|------|----------|-------------|-----------------|
| AIM2_ONTO_PARS_E_001 | Error | OWL file parsing failure | Validate OWL syntax and structure |
| AIM2_ONTO_PARS_E_002 | Error | RDF triple extraction failure | Check RDF format and namespaces |
| AIM2_ONTO_PARS_E_003 | Error | JSON-LD context resolution failure | Verify JSON-LD context URLs and format |
| AIM2_ONTO_PARS_E_004 | Error | Unsupported ontology format | Use supported format (OWL, RDF, JSON-LD) |

#### Validation Errors (ONTO_VALD)
| Code | Severity | Description | Resolution Hint |
|------|----------|-------------|-----------------|
| AIM2_ONTO_VALD_E_001 | Error | Ontology consistency check failed | Resolve logical inconsistencies |
| AIM2_ONTO_VALD_E_002 | Error | Class hierarchy violation | Fix parent-child relationships |
| AIM2_ONTO_VALD_E_003 | Error | Property domain/range mismatch | Correct property definitions |
| AIM2_ONTO_VALD_W_004 | Warning | Orphaned class detected | Link to appropriate parent class |

#### Integration Errors (ONTO_INTG)
| Code | Severity | Description | Resolution Hint |
|------|----------|-------------|-----------------|
| AIM2_ONTO_INTG_E_001 | Error | Ontology merge conflict | Resolve conflicting definitions manually |
| AIM2_ONTO_INTG_E_002 | Error | Namespace collision | Use unique namespaces or prefixes |
| AIM2_ONTO_INTG_E_003 | Error | Incompatible ontology versions | Update to compatible versions |
| AIM2_ONTO_INTG_W_004 | Warning | Duplicate concept detected | Review for necessary deduplication |

### 4.3 Extraction Module Error Codes

#### NER Errors (EXTR_NER)
| Code | Severity | Description | Resolution Hint |
|------|----------|-------------|-----------------|
| AIM2_EXTR_NER_E_001 | Error | NER model loading failure | Verify model file path and format |
| AIM2_EXTR_NER_E_002 | Error | Text tokenization failure | Check text encoding and format |
| AIM2_EXTR_NER_E_003 | Error | Entity recognition timeout | Reduce text size or increase timeout |
| AIM2_EXTR_NER_W_004 | Warning | Low confidence entity detected | Review entity extraction results |

#### Relationship Extraction Errors (EXTR_REL)
| Code | Severity | Description | Resolution Hint |
|------|----------|-------------|-----------------|
| AIM2_EXTR_REL_E_001 | Error | Relationship pattern matching failed | Update extraction patterns |
| AIM2_EXTR_REL_E_002 | Error | Entity pair validation failed | Verify entity types and relationships |
| AIM2_EXTR_REL_E_003 | Error | Dependency parsing failure | Check sentence structure and grammar |
| AIM2_EXTR_REL_I_004 | Info | No relationships found in text | Normal for some text types |

### 4.4 LLM Module Error Codes

#### API Errors (LLM_API)
| Code | Severity | Description | Resolution Hint |
|------|----------|-------------|-----------------|
| AIM2_LLM_API_E_001 | Error | API request failed | Check network connectivity and API status |
| AIM2_LLM_API_E_002 | Error | Invalid API response format | Verify API endpoint and parameters |
| AIM2_LLM_API_E_003 | Error | API quota exceeded | Wait for quota reset or upgrade plan |
| AIM2_LLM_API_W_004 | Warning | API response truncated | Reduce input size or increase max tokens |

#### Authentication Errors (LLM_AUTH)
| Code | Severity | Description | Resolution Hint |
|------|----------|-------------|-----------------|
| AIM2_LLM_AUTH_E_001 | Error | API key invalid or expired | Verify and update API credentials |
| AIM2_LLM_AUTH_E_002 | Error | Insufficient API permissions | Check API key permissions and scope |
| AIM2_LLM_AUTH_E_003 | Error | Authentication service unavailable | Retry later or contact provider |

#### Timeout Errors (LLM_TIME)
| Code | Severity | Description | Resolution Hint |
|------|----------|-------------|-----------------|
| AIM2_LLM_TIME_E_001 | Error | Request timeout exceeded | Increase timeout or reduce request size |
| AIM2_LLM_TIME_E_002 | Error | Model inference timeout | Use faster model or smaller input |
| AIM2_LLM_TIME_W_003 | Warning | Slow response detected | Consider optimizing request parameters |

### 4.5 Validation Module Error Codes

#### Schema Validation Errors (VALID_SCHM)
| Code | Severity | Description | Resolution Hint |
|------|----------|-------------|-----------------|
| AIM2_VALID_SCHM_E_001 | Error | JSON schema validation failed | Fix data structure to match schema |
| AIM2_VALID_SCHM_E_002 | Error | Required field missing | Add missing required fields |
| AIM2_VALID_SCHM_E_003 | Error | Invalid field type | Correct field type to match schema |
| AIM2_VALID_SCHM_W_004 | Warning | Unknown field in data | Remove or add field to schema |

#### Data Validation Errors (VALID_DATA)
| Code | Severity | Description | Resolution Hint |
|------|----------|-------------|-----------------|
| AIM2_VALID_DATA_E_001 | Error | Data format validation failed | Correct data format and structure |
| AIM2_VALID_DATA_E_002 | Error | Value out of valid range | Adjust value to be within acceptable range |
| AIM2_VALID_DATA_E_003 | Error | Invalid data encoding | Use correct character encoding |
| AIM2_VALID_DATA_W_004 | Warning | Data quality issue detected | Review and clean data if necessary |

## 5. Documentation Structure

### 5.1 Error Code Documentation Template
```markdown
### Error Code: AIM2_{MODULE}_{CATEGORY}_{SEVERITY}_{NUMBER}

**Module:** {Module Name}
**Category:** {Category Name}
**Severity:** {Severity Level}
**Description:** {Detailed error description}

**Common Causes:**
- Cause 1
- Cause 2
- Cause 3

**Resolution Steps:**
1. Step 1
2. Step 2
3. Step 3

**Related Error Codes:**
- Related code 1
- Related code 2

**Example:**
```python
# Code example showing when this error occurs
```

**See Also:**
- Link to relevant documentation
- Link to troubleshooting guide
```

### 5.2 Error Code Reference Structure
```
docs/
├── error_codes/
│   ├── README.md                    # Error code system overview
│   ├── base_module_errors.md        # Base module error codes
│   ├── ontology_module_errors.md    # Ontology module error codes
│   ├── extraction_module_errors.md  # Extraction module error codes
│   ├── llm_module_errors.md         # LLM module error codes
│   ├── validation_module_errors.md  # Validation module error codes
│   ├── error_code_registry.md       # Complete error code listing
│   └── troubleshooting_guide.md     # Common issues and solutions
```

## 6. Integration with Existing System

### 6.1 Exception Class Modifications
```python
class AIM2Exception(Exception):
    def __init__(
        self,
        message: str,
        error_code: Optional[Union[str, AIM2ErrorCodes]] = None,
        cause: Optional[Exception] = None,
        context: Optional[Dict[str, Any]] = None,
    ):
        super().__init__(message)
        self.message = message

        # Handle both string and enum error codes
        if isinstance(error_code, AIM2ErrorCodes):
            self.error_code = error_code.value.code
            self.error_info = error_code.value
        else:
            self.error_code = error_code or self._get_default_error_code()
            self.error_info = ErrorCodeRegistry().get_error_info(self.error_code)

        self.cause = cause
        self.context = context or {}

        if cause:
            self.__cause__ = cause
```

### 6.2 Backward Compatibility
- Existing string-based error codes continue to work
- New enum-based codes provide additional information
- Migration path for gradual adoption
- No breaking changes to current API

### 6.3 Usage Examples
```python
# Using enum-based error codes
raise OntologyException(
    "Failed to parse OWL file",
    error_code=AIM2ErrorCodes.ONTO_PARS_E_001,
    context={"file_path": "/path/to/ontology.owl"}
)

# Using string-based codes (backward compatible)
raise ExtractionException(
    "NER model loading failed",
    error_code="AIM2_EXTR_NER_E_001",
    context={"model_path": "/path/to/model"}
)

# Accessing error information
try:
    # Some operation
    pass
except AIM2Exception as e:
    if e.error_info:
        print(f"Severity: {e.error_info.severity.value}")
        print(f"Resolution: {e.error_info.resolution_hint}")
```

## 7. Benefits and Implementation Roadmap

### 7.1 Benefits of the New System
1. **Structured Categorization**: Hierarchical organization of errors
2. **Severity Classification**: Clear error severity levels
3. **Improved Debugging**: Better error context and resolution hints
4. **Maintainability**: Centralized error code management
5. **Documentation**: Comprehensive error documentation
6. **Extensibility**: Easy addition of new error types
7. **Monitoring**: Better error tracking and analytics

### 7.2 Implementation Phases

#### Phase 1: Foundation (1-2 weeks)
- Implement error code enums and registry
- Create base error code structure
- Update AIM2Exception class
- Ensure backward compatibility

#### Phase 2: Module-Specific Codes (2-3 weeks)
- Define error codes for each module
- Update module-specific exception classes
- Create error code documentation
- Write unit tests for new functionality

#### Phase 3: Integration and Testing (1 week)
- Integrate with existing codebase
- Comprehensive testing
- Performance validation
- Documentation updates

#### Phase 4: Migration and Enhancement (Ongoing)
- Gradual migration of existing error codes
- Addition of new error codes as needed
- Continuous documentation updates
- Error analytics and monitoring

### 7.3 Success Metrics
- Reduction in debugging time
- Improved error resolution rate
- Better error tracking and analytics
- Enhanced developer experience
- Reduced support requests

## 8. Conclusion

This error code system design provides a comprehensive, scalable, and maintainable approach to error handling in the AIM2 project. The hierarchical structure, severity levels, and detailed documentation will significantly improve debugging, monitoring, and maintenance capabilities while maintaining backward compatibility with the existing system.

The proposed system balances structure with flexibility, allowing for systematic error categorization while supporting the diverse needs of the AIM2 project's various modules. Implementation should follow the phased approach to ensure smooth integration and adoption across the project.
