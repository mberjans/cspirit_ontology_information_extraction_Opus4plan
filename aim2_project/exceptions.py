"""
AIM2 Exception Hierarchy Module

This module provides a comprehensive exception hierarchy for the AIM2 project,
including base and specialized exception classes with error code management,
serialization support, and cause tracking capabilities.

The exception hierarchy follows a structured approach where all AIM2-specific
exceptions inherit from the base AIM2Exception class, providing consistent
error handling, logging, and debugging capabilities across the entire project.

Error Code System Components:
    ErrorSeverity: Enumeration of error severity levels (CRITICAL, ERROR, WARNING, INFO)
    ErrorCodeInfo: NamedTuple containing error code metadata
    AIM2ErrorCodes: Comprehensive enumeration of all project error codes
    ErrorCodeRegistry: Central registry for error code management and lookup
    BaseErrorCodes, OntologyErrorCodes, etc.: Module-specific convenience classes

Exception Classes:
    AIM2Exception: Base exception class with comprehensive error management
    OntologyException: Exceptions related to ontology operations
    ExtractionException: Exceptions related to information extraction
    LLMException: Exceptions related to LLM interface operations
    ValidationException: Exceptions related to data validation

Features:
    - Hierarchical error code enumeration system with structured metadata
    - Support for both string and enum error codes with backward compatibility
    - Exception chaining and cause tracking
    - Context information support for debugging
    - Enhanced error message formatting with severity levels and resolution hints
    - Serialization and deserialization support (to_dict/from_dict)
    - Central error code registry for management and analytics
    - Module-specific error code groupings for easy access
    - Comprehensive docstrings and type hints

Dependencies:
    - traceback: For traceback information in serialization
    - typing: For type hints and annotations

Usage:
    # Basic usage
    raise AIM2Exception("Something went wrong")

    # Using enum-based error codes (recommended)
    raise OntologyException(
        "Failed to parse OWL file",
        error_code=AIM2ErrorCodes.ONTO_PARS_E_001,
        context={"file": "ontology.owl", "line": 42}
    )

    # Using convenience classes for easier access
    raise ExtractionException(
        "NER model loading failed",
        error_code=ExtractionErrorCodes.NER_MODEL_LOADING_FAILURE,
        context={"model_path": "/path/to/model"}
    )

    # Using string-based error codes (backward compatible)
    raise ValidationException(
        "Schema validation failed",
        error_code="AIM2_VALID_SCHM_E_001",
        context={"schema_file": "config.schema.json"}
    )

    # With cause chaining and enhanced error info
    try:
        # Some operation that might fail
        pass
    except FileNotFoundError as e:
        raise ExtractionException(
            "Could not load extraction model",
            error_code=ExtractionErrorCodes.NER_MODEL_LOADING_FAILURE,
            cause=e,
            context={"model_path": "/path/to/model"}
        )

    # Accessing error information and detailed messages
    try:
        raise LLMException(
            "API request failed",
            error_code=LLMErrorCodes.API_REQUEST_FAILED
        )
    except LLMException as e:
        print(e.get_detailed_message())  # Formatted with severity and resolution
        if e.error_info:
            print(f"Severity: {e.error_info.severity.name}")
            print(f"Module: {e.error_info.module}")

    # Using the error code registry
    registry = ErrorCodeRegistry()
    critical_errors = registry.get_codes_by_severity(ErrorSeverity.CRITICAL)
    stats = registry.get_statistics()

    # Serialization with enhanced metadata
    exception = ValidationException(
        "Invalid config",
        error_code=ValidationErrorCodes.SCHEMA_VALIDATION_FAILED
    )
    exception_data = exception.to_dict()  # Includes error_info metadata
    restored_exception = ValidationException.from_dict(exception_data)

Authors:
    AIM2 Development Team

Version:
    1.0.0

Created:
    2025-08-03
"""

import traceback
from enum import Enum
from typing import Dict, Any, Optional, Union, List, NamedTuple, Protocol
from abc import ABC, abstractmethod
import string
import json
from datetime import datetime


# Error Code System Components


class ErrorSeverity(Enum):
    """
    Enumeration of error severity levels for AIM2 error codes.

    This enum defines the standard severity levels used throughout the AIM2
    error code system to classify the impact and urgency of different errors.

    Attributes:
        CRITICAL (str): System failure, data corruption, requires immediate attention
        ERROR (str): Operation failure, recovery possible, needs attention
        WARNING (str): Potential issues, operation continues, should be monitored
        INFO (str): Informational, for tracking and debugging purposes

    Examples:
        >>> severity = ErrorSeverity.CRITICAL
        >>> print(severity.value)  # Output: 'C'
        >>> print(severity.name)   # Output: 'CRITICAL'
    """

    CRITICAL = "C"
    ERROR = "E"
    WARNING = "W"
    INFO = "I"


class ErrorCodeInfo(NamedTuple):
    """
    Information container for error code metadata.

    This NamedTuple stores comprehensive metadata about each error code,
    including descriptive information and resolution guidance to aid in
    debugging and error handling.

    Attributes:
        code (str): Full error code string (e.g., "AIM2_BASE_SYS_C_001")
        module (str): Module identifier (e.g., "BASE", "ONTO", "EXTR")
        category (str): Error category (e.g., "SYS", "PARS", "API")
        severity (ErrorSeverity): Severity level of the error
        description (str): Human-readable description of the error
        resolution_hint (str): Guidance for resolving the error

    Examples:
        >>> info = ErrorCodeInfo(
        ...     code="AIM2_BASE_SYS_C_001",
        ...     module="BASE",
        ...     category="SYS",
        ...     severity=ErrorSeverity.CRITICAL,
        ...     description="System initialization failure",
        ...     resolution_hint="Check system dependencies and configuration"
        ... )
        >>> print(info.code)  # Output: 'AIM2_BASE_SYS_C_001'
    """

    code: str
    module: str
    category: str
    severity: ErrorSeverity
    description: str
    resolution_hint: str


class AIM2ErrorCodes(Enum):
    """
    Comprehensive enumeration of all AIM2 error codes.

    This enum provides a centralized definition of all error codes used
    throughout the AIM2 project, organized by module and category with
    detailed metadata for each error code.

    The error codes follow the hierarchical format:
    AIM2_{MODULE}_{CATEGORY}_{SEVERITY}_{NUMBER}

    Examples:
        >>> error_code = AIM2ErrorCodes.BASE_SYS_C_001
        >>> print(error_code.value.code)  # Output: 'AIM2_BASE_SYS_C_001'
        >>> print(error_code.value.description)  # Output: 'System initialization failure'
    """

    # Base Module Errors - System (BASE_SYS)
    BASE_SYS_C_001 = ErrorCodeInfo(
        code="AIM2_BASE_SYS_C_001",
        module="BASE",
        category="SYS",
        severity=ErrorSeverity.CRITICAL,
        description="System initialization failure",
        resolution_hint="Check dependencies and permissions",
    )
    BASE_SYS_C_002 = ErrorCodeInfo(
        code="AIM2_BASE_SYS_C_002",
        module="BASE",
        category="SYS",
        severity=ErrorSeverity.CRITICAL,
        description="Memory allocation failure",
        resolution_hint="Increase available memory or reduce load",
    )
    BASE_SYS_E_003 = ErrorCodeInfo(
        code="AIM2_BASE_SYS_E_003",
        module="BASE",
        category="SYS",
        severity=ErrorSeverity.ERROR,
        description="Module import failure",
        resolution_hint="Verify module installation and Python path",
    )
    BASE_SYS_E_004 = ErrorCodeInfo(
        code="AIM2_BASE_SYS_E_004",
        module="BASE",
        category="SYS",
        severity=ErrorSeverity.ERROR,
        description="Environment variable missing",
        resolution_hint="Set required environment variables",
    )

    # Base Module Errors - Configuration (BASE_CFG)
    BASE_CFG_E_001 = ErrorCodeInfo(
        code="AIM2_BASE_CFG_E_001",
        module="BASE",
        category="CFG",
        severity=ErrorSeverity.ERROR,
        description="Configuration file not found",
        resolution_hint="Create or specify correct config file path",
    )
    BASE_CFG_E_002 = ErrorCodeInfo(
        code="AIM2_BASE_CFG_E_002",
        module="BASE",
        category="CFG",
        severity=ErrorSeverity.ERROR,
        description="Invalid configuration syntax",
        resolution_hint="Check YAML/JSON syntax and structure",
    )
    BASE_CFG_E_003 = ErrorCodeInfo(
        code="AIM2_BASE_CFG_E_003",
        module="BASE",
        category="CFG",
        severity=ErrorSeverity.ERROR,
        description="Missing required configuration field",
        resolution_hint="Add missing configuration parameters",
    )
    BASE_CFG_W_004 = ErrorCodeInfo(
        code="AIM2_BASE_CFG_W_004",
        module="BASE",
        category="CFG",
        severity=ErrorSeverity.WARNING,
        description="Deprecated configuration field used",
        resolution_hint="Update to new configuration schema",
    )

    # Base Module Errors - Persistence (BASE_PER)
    BASE_PER_C_001 = ErrorCodeInfo(
        code="AIM2_BASE_PER_C_001",
        module="BASE",
        category="PER",
        severity=ErrorSeverity.CRITICAL,
        description="Database connection failure",
        resolution_hint="Check database server and credentials",
    )
    BASE_PER_E_002 = ErrorCodeInfo(
        code="AIM2_BASE_PER_E_002",
        module="BASE",
        category="PER",
        severity=ErrorSeverity.ERROR,
        description="File system permission denied",
        resolution_hint="Verify file/directory permissions",
    )
    BASE_PER_E_003 = ErrorCodeInfo(
        code="AIM2_BASE_PER_E_003",
        module="BASE",
        category="PER",
        severity=ErrorSeverity.ERROR,
        description="Disk space insufficient",
        resolution_hint="Free up disk space or change location",
    )
    BASE_PER_E_004 = ErrorCodeInfo(
        code="AIM2_BASE_PER_E_004",
        module="BASE",
        category="PER",
        severity=ErrorSeverity.ERROR,
        description="File corruption detected",
        resolution_hint="Restore from backup or recreate file",
    )

    # Ontology Module Errors - Parsing (ONTO_PARS)
    ONTO_PARS_E_001 = ErrorCodeInfo(
        code="AIM2_ONTO_PARS_E_001",
        module="ONTO",
        category="PARS",
        severity=ErrorSeverity.ERROR,
        description="OWL file parsing failure",
        resolution_hint="Validate OWL syntax and structure",
    )
    ONTO_PARS_E_002 = ErrorCodeInfo(
        code="AIM2_ONTO_PARS_E_002",
        module="ONTO",
        category="PARS",
        severity=ErrorSeverity.ERROR,
        description="RDF triple extraction failure",
        resolution_hint="Check RDF format and namespaces",
    )
    ONTO_PARS_E_003 = ErrorCodeInfo(
        code="AIM2_ONTO_PARS_E_003",
        module="ONTO",
        category="PARS",
        severity=ErrorSeverity.ERROR,
        description="JSON-LD context resolution failure",
        resolution_hint="Verify JSON-LD context URLs and format",
    )
    ONTO_PARS_E_004 = ErrorCodeInfo(
        code="AIM2_ONTO_PARS_E_004",
        module="ONTO",
        category="PARS",
        severity=ErrorSeverity.ERROR,
        description="Unsupported ontology format",
        resolution_hint="Use supported format (OWL, RDF, JSON-LD)",
    )

    # Ontology Module Errors - Validation (ONTO_VALD)
    ONTO_VALD_E_001 = ErrorCodeInfo(
        code="AIM2_ONTO_VALD_E_001",
        module="ONTO",
        category="VALD",
        severity=ErrorSeverity.ERROR,
        description="Ontology consistency check failed",
        resolution_hint="Resolve logical inconsistencies",
    )
    ONTO_VALD_E_002 = ErrorCodeInfo(
        code="AIM2_ONTO_VALD_E_002",
        module="ONTO",
        category="VALD",
        severity=ErrorSeverity.ERROR,
        description="Class hierarchy violation",
        resolution_hint="Fix parent-child relationships",
    )
    ONTO_VALD_E_003 = ErrorCodeInfo(
        code="AIM2_ONTO_VALD_E_003",
        module="ONTO",
        category="VALD",
        severity=ErrorSeverity.ERROR,
        description="Property domain/range mismatch",
        resolution_hint="Correct property definitions",
    )
    ONTO_VALD_W_004 = ErrorCodeInfo(
        code="AIM2_ONTO_VALD_W_004",
        module="ONTO",
        category="VALD",
        severity=ErrorSeverity.WARNING,
        description="Orphaned class detected",
        resolution_hint="Link to appropriate parent class",
    )

    # Ontology Module Errors - Integration (ONTO_INTG)
    ONTO_INTG_E_001 = ErrorCodeInfo(
        code="AIM2_ONTO_INTG_E_001",
        module="ONTO",
        category="INTG",
        severity=ErrorSeverity.ERROR,
        description="Ontology merge conflict",
        resolution_hint="Resolve conflicting definitions manually",
    )
    ONTO_INTG_E_002 = ErrorCodeInfo(
        code="AIM2_ONTO_INTG_E_002",
        module="ONTO",
        category="INTG",
        severity=ErrorSeverity.ERROR,
        description="Namespace collision",
        resolution_hint="Use unique namespaces or prefixes",
    )
    ONTO_INTG_E_003 = ErrorCodeInfo(
        code="AIM2_ONTO_INTG_E_003",
        module="ONTO",
        category="INTG",
        severity=ErrorSeverity.ERROR,
        description="Incompatible ontology versions",
        resolution_hint="Update to compatible versions",
    )
    ONTO_INTG_W_004 = ErrorCodeInfo(
        code="AIM2_ONTO_INTG_W_004",
        module="ONTO",
        category="INTG",
        severity=ErrorSeverity.WARNING,
        description="Duplicate concept detected",
        resolution_hint="Review for necessary deduplication",
    )

    # Extraction Module Errors - NER (EXTR_NER)
    EXTR_NER_E_001 = ErrorCodeInfo(
        code="AIM2_EXTR_NER_E_001",
        module="EXTR",
        category="NER",
        severity=ErrorSeverity.ERROR,
        description="NER model loading failure",
        resolution_hint="Verify model file path and format",
    )
    EXTR_NER_E_002 = ErrorCodeInfo(
        code="AIM2_EXTR_NER_E_002",
        module="EXTR",
        category="NER",
        severity=ErrorSeverity.ERROR,
        description="Text tokenization failure",
        resolution_hint="Check text encoding and format",
    )
    EXTR_NER_E_003 = ErrorCodeInfo(
        code="AIM2_EXTR_NER_E_003",
        module="EXTR",
        category="NER",
        severity=ErrorSeverity.ERROR,
        description="Entity recognition timeout",
        resolution_hint="Reduce text size or increase timeout",
    )
    EXTR_NER_W_004 = ErrorCodeInfo(
        code="AIM2_EXTR_NER_W_004",
        module="EXTR",
        category="NER",
        severity=ErrorSeverity.WARNING,
        description="Low confidence entity detected",
        resolution_hint="Review entity extraction results",
    )

    # Extraction Module Errors - Relationship (EXTR_REL)
    EXTR_REL_E_001 = ErrorCodeInfo(
        code="AIM2_EXTR_REL_E_001",
        module="EXTR",
        category="REL",
        severity=ErrorSeverity.ERROR,
        description="Relationship pattern matching failed",
        resolution_hint="Update extraction patterns",
    )
    EXTR_REL_E_002 = ErrorCodeInfo(
        code="AIM2_EXTR_REL_E_002",
        module="EXTR",
        category="REL",
        severity=ErrorSeverity.ERROR,
        description="Entity pair validation failed",
        resolution_hint="Verify entity types and relationships",
    )
    EXTR_REL_E_003 = ErrorCodeInfo(
        code="AIM2_EXTR_REL_E_003",
        module="EXTR",
        category="REL",
        severity=ErrorSeverity.ERROR,
        description="Dependency parsing failure",
        resolution_hint="Check sentence structure and grammar",
    )
    EXTR_REL_I_004 = ErrorCodeInfo(
        code="AIM2_EXTR_REL_I_004",
        module="EXTR",
        category="REL",
        severity=ErrorSeverity.INFO,
        description="No relationships found in text",
        resolution_hint="Normal for some text types",
    )

    # LLM Module Errors - API (LLM_API)
    LLM_API_E_001 = ErrorCodeInfo(
        code="AIM2_LLM_API_E_001",
        module="LLM",
        category="API",
        severity=ErrorSeverity.ERROR,
        description="API request failed",
        resolution_hint="Check network connectivity and API status",
    )
    LLM_API_E_002 = ErrorCodeInfo(
        code="AIM2_LLM_API_E_002",
        module="LLM",
        category="API",
        severity=ErrorSeverity.ERROR,
        description="Invalid API response format",
        resolution_hint="Verify API endpoint and parameters",
    )
    LLM_API_E_003 = ErrorCodeInfo(
        code="AIM2_LLM_API_E_003",
        module="LLM",
        category="API",
        severity=ErrorSeverity.ERROR,
        description="API quota exceeded",
        resolution_hint="Wait for quota reset or upgrade plan",
    )
    LLM_API_W_004 = ErrorCodeInfo(
        code="AIM2_LLM_API_W_004",
        module="LLM",
        category="API",
        severity=ErrorSeverity.WARNING,
        description="API response truncated",
        resolution_hint="Reduce input size or increase max tokens",
    )

    # LLM Module Errors - Authentication (LLM_AUTH)
    LLM_AUTH_E_001 = ErrorCodeInfo(
        code="AIM2_LLM_AUTH_E_001",
        module="LLM",
        category="AUTH",
        severity=ErrorSeverity.ERROR,
        description="API key invalid or expired",
        resolution_hint="Verify and update API credentials",
    )
    LLM_AUTH_E_002 = ErrorCodeInfo(
        code="AIM2_LLM_AUTH_E_002",
        module="LLM",
        category="AUTH",
        severity=ErrorSeverity.ERROR,
        description="Insufficient API permissions",
        resolution_hint="Check API key permissions and scope",
    )
    LLM_AUTH_E_003 = ErrorCodeInfo(
        code="AIM2_LLM_AUTH_E_003",
        module="LLM",
        category="AUTH",
        severity=ErrorSeverity.ERROR,
        description="Authentication service unavailable",
        resolution_hint="Retry later or contact provider",
    )

    # LLM Module Errors - Timeout (LLM_TIME)
    LLM_TIME_E_001 = ErrorCodeInfo(
        code="AIM2_LLM_TIME_E_001",
        module="LLM",
        category="TIME",
        severity=ErrorSeverity.ERROR,
        description="Request timeout exceeded",
        resolution_hint="Increase timeout or reduce request size",
    )
    LLM_TIME_E_002 = ErrorCodeInfo(
        code="AIM2_LLM_TIME_E_002",
        module="LLM",
        category="TIME",
        severity=ErrorSeverity.ERROR,
        description="Model inference timeout",
        resolution_hint="Use faster model or smaller input",
    )
    LLM_TIME_W_003 = ErrorCodeInfo(
        code="AIM2_LLM_TIME_W_003",
        module="LLM",
        category="TIME",
        severity=ErrorSeverity.WARNING,
        description="Slow response detected",
        resolution_hint="Consider optimizing request parameters",
    )

    # Validation Module Errors - Schema (VALID_SCHM)
    VALID_SCHM_E_001 = ErrorCodeInfo(
        code="AIM2_VALID_SCHM_E_001",
        module="VALID",
        category="SCHM",
        severity=ErrorSeverity.ERROR,
        description="JSON schema validation failed",
        resolution_hint="Fix data structure to match schema",
    )
    VALID_SCHM_E_002 = ErrorCodeInfo(
        code="AIM2_VALID_SCHM_E_002",
        module="VALID",
        category="SCHM",
        severity=ErrorSeverity.ERROR,
        description="Required field missing",
        resolution_hint="Add missing required fields",
    )
    VALID_SCHM_E_003 = ErrorCodeInfo(
        code="AIM2_VALID_SCHM_E_003",
        module="VALID",
        category="SCHM",
        severity=ErrorSeverity.ERROR,
        description="Invalid field type",
        resolution_hint="Correct field type to match schema",
    )
    VALID_SCHM_W_004 = ErrorCodeInfo(
        code="AIM2_VALID_SCHM_W_004",
        module="VALID",
        category="SCHM",
        severity=ErrorSeverity.WARNING,
        description="Unknown field in data",
        resolution_hint="Remove or add field to schema",
    )

    # Validation Module Errors - Data (VALID_DATA)
    VALID_DATA_E_001 = ErrorCodeInfo(
        code="AIM2_VALID_DATA_E_001",
        module="VALID",
        category="DATA",
        severity=ErrorSeverity.ERROR,
        description="Data format validation failed",
        resolution_hint="Correct data format and structure",
    )
    VALID_DATA_E_002 = ErrorCodeInfo(
        code="AIM2_VALID_DATA_E_002",
        module="VALID",
        category="DATA",
        severity=ErrorSeverity.ERROR,
        description="Value out of valid range",
        resolution_hint="Adjust value to be within acceptable range",
    )
    VALID_DATA_E_003 = ErrorCodeInfo(
        code="AIM2_VALID_DATA_E_003",
        module="VALID",
        category="DATA",
        severity=ErrorSeverity.ERROR,
        description="Invalid data encoding",
        resolution_hint="Use correct character encoding",
    )
    VALID_DATA_W_004 = ErrorCodeInfo(
        code="AIM2_VALID_DATA_W_004",
        module="VALID",
        category="DATA",
        severity=ErrorSeverity.WARNING,
        description="Data quality issue detected",
        resolution_hint="Review and clean data if necessary",
    )


class ErrorCodeRegistry:
    """
    Central registry for all AIM2 error codes with lookup and management capabilities.

    This class provides a centralized system for managing, looking up, and analyzing
    error codes throughout the AIM2 project. It offers various methods to retrieve
    error information by different criteria and supports analytics and reporting.

    The registry is automatically populated with all error codes from the
    AIM2ErrorCodes enum during initialization.

    Attributes:
        _codes (Dict[str, ErrorCodeInfo]): Internal storage for error code mappings
        _instance (ErrorCodeRegistry): Singleton instance

    Examples:
        >>> registry = ErrorCodeRegistry()
        >>> info = registry.get_error_info("AIM2_BASE_SYS_C_001")
        >>> print(info.description)  # Output: 'System initialization failure'

        >>> base_errors = registry.get_codes_by_module("BASE")
        >>> print(len(base_errors))  # Output: number of BASE module errors

        >>> critical_errors = registry.get_codes_by_severity(ErrorSeverity.CRITICAL)
        >>> print(len(critical_errors))  # Output: number of critical errors
    """

    _instance = None

    def __new__(cls):
        """Implement singleton pattern for the error code registry."""
        if cls._instance is None:
            cls._instance = super(ErrorCodeRegistry, cls).__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        """Initialize the error code registry with all defined error codes."""
        if self._initialized:
            return

        self._codes: Dict[str, ErrorCodeInfo] = {}
        self._register_all_codes()
        self._initialized = True

    def _register_all_codes(self) -> None:
        """
        Register all error codes from the AIM2ErrorCodes enum.

        This method automatically discovers and registers all error codes
        defined in the AIM2ErrorCodes enum, building the internal lookup
        dictionary for fast access.
        """
        for error_code in AIM2ErrorCodes:
            self._codes[error_code.value.code] = error_code.value

    def get_error_info(self, code: str) -> Optional[ErrorCodeInfo]:
        """
        Get error information by error code string.

        Args:
            code (str): The error code string to look up

        Returns:
            Optional[ErrorCodeInfo]: Error information if found, None otherwise

        Examples:
            >>> registry = ErrorCodeRegistry()
            >>> info = registry.get_error_info("AIM2_BASE_SYS_C_001")
            >>> if info:
            ...     print(f"Description: {info.description}")
            ...     print(f"Severity: {info.severity.name}")
        """
        return self._codes.get(code)

    def get_codes_by_module(self, module: str) -> List[ErrorCodeInfo]:
        """
        Get all error codes for a specific module.

        Args:
            module (str): Module identifier (e.g., "BASE", "ONTO", "EXTR")

        Returns:
            List[ErrorCodeInfo]: List of error codes for the specified module

        Examples:
            >>> registry = ErrorCodeRegistry()
            >>> base_errors = registry.get_codes_by_module("BASE")
            >>> for error in base_errors:
            ...     print(f"{error.code}: {error.description}")
        """
        return [info for info in self._codes.values() if info.module == module]

    def get_codes_by_category(self, category: str) -> List[ErrorCodeInfo]:
        """
        Get all error codes for a specific category.

        Args:
            category (str): Category identifier (e.g., "SYS", "PARS", "API")

        Returns:
            List[ErrorCodeInfo]: List of error codes for the specified category

        Examples:
            >>> registry = ErrorCodeRegistry()
            >>> sys_errors = registry.get_codes_by_category("SYS")
            >>> for error in sys_errors:
            ...     print(f"{error.code}: {error.description}")
        """
        return [info for info in self._codes.values() if info.category == category]

    def get_codes_by_severity(self, severity: ErrorSeverity) -> List[ErrorCodeInfo]:
        """
        Get all error codes of a specific severity level.

        Args:
            severity (ErrorSeverity): Severity level to filter by

        Returns:
            List[ErrorCodeInfo]: List of error codes with the specified severity

        Examples:
            >>> registry = ErrorCodeRegistry()
            >>> critical_errors = registry.get_codes_by_severity(ErrorSeverity.CRITICAL)
            >>> for error in critical_errors:
            ...     print(f"{error.code}: {error.description}")
        """
        return [info for info in self._codes.values() if info.severity == severity]

    def get_all_codes(self) -> List[ErrorCodeInfo]:
        """
        Get all registered error codes.

        Returns:
            List[ErrorCodeInfo]: List of all registered error codes

        Examples:
            >>> registry = ErrorCodeRegistry()
            >>> all_errors = registry.get_all_codes()
            >>> print(f"Total error codes: {len(all_errors)}")
        """
        return list(self._codes.values())

    def validate_error_code(self, code: str) -> bool:
        """
        Validate if an error code is registered.

        Args:
            code (str): Error code string to validate

        Returns:
            bool: True if error code is registered, False otherwise

        Examples:
            >>> registry = ErrorCodeRegistry()
            >>> is_valid = registry.validate_error_code("AIM2_BASE_SYS_C_001")
            >>> print(is_valid)  # Output: True
        """
        return code in self._codes

    def get_modules(self) -> List[str]:
        """
        Get list of all registered modules.

        Returns:
            List[str]: List of unique module identifiers

        Examples:
            >>> registry = ErrorCodeRegistry()
            >>> modules = registry.get_modules()
            >>> print(modules)  # Output: ['BASE', 'ONTO', 'EXTR', 'LLM', 'VALID']
        """
        return list(set(info.module for info in self._codes.values()))

    def get_categories(self) -> List[str]:
        """
        Get list of all registered categories.

        Returns:
            List[str]: List of unique category identifiers

        Examples:
            >>> registry = ErrorCodeRegistry()
            >>> categories = registry.get_categories()
            >>> print(sorted(categories))  # Output: sorted list of categories
        """
        return list(set(info.category for info in self._codes.values()))

    def get_statistics(self) -> Dict[str, Any]:
        """
        Get statistical information about registered error codes.

        Returns:
            Dict[str, Any]: Dictionary containing statistics about error codes

        Examples:
            >>> registry = ErrorCodeRegistry()
            >>> stats = registry.get_statistics()
            >>> print(f"Total codes: {stats['total_codes']}")
            >>> print(f"Modules: {stats['modules']}")
        """
        all_codes = self.get_all_codes()

        severity_counts = {}
        for severity in ErrorSeverity:
            severity_counts[severity.name] = len(
                [code for code in all_codes if code.severity == severity]
            )

        module_counts = {}
        for module in self.get_modules():
            module_counts[module] = len(self.get_codes_by_module(module))

        return {
            "total_codes": len(all_codes),
            "modules": self.get_modules(),
            "categories": self.get_categories(),
            "severity_distribution": severity_counts,
            "module_distribution": module_counts,
        }


# Error Message Template System Components


class MessageFormat(Enum):
    """
    Enumeration of available message output formats.

    This enum defines the different output formats supported by the error
    message template system, allowing for context-appropriate formatting.

    Attributes:
        CONSOLE (str): Human-readable console output format
        STRUCTURED (str): Structured format for logging and debugging
        API (str): Clean format for API responses
        JSON (str): JSON-formatted output for programmatic use
        MARKDOWN (str): Markdown formatted output for documentation
    """

    CONSOLE = "console"
    STRUCTURED = "structured"
    API = "api"
    JSON = "json"
    MARKDOWN = "markdown"


class MessageTemplate(Protocol):
    """
    Protocol defining the interface for error message templates.

    This protocol ensures that all message template implementations
    provide consistent formatting capabilities while allowing for
    different template strategies and output formats.
    """

    def format_message(
        self,
        message: str,
        error_code: str,
        error_info: Optional[ErrorCodeInfo] = None,
        context: Optional[Dict[str, Any]] = None,
        format_type: MessageFormat = MessageFormat.CONSOLE,
        **kwargs,
    ) -> str:
        """
        Format an error message using the template.

        Args:
            message (str): The base error message
            error_code (str): The error code
            error_info (ErrorCodeInfo, optional): Error metadata information
            context (Dict[str, Any], optional): Additional context data
            format_type (MessageFormat): The desired output format
            **kwargs: Additional formatting parameters

        Returns:
            str: The formatted error message
        """
        ...


class BaseMessageTemplate(ABC):
    """
    Abstract base class for error message templates.

    This class provides common functionality for all message templates
    including variable substitution, context processing, and basic
    formatting utilities.

    Attributes:
        template_name (str): Name identifier for the template
        supported_formats (List[MessageFormat]): List of supported output formats
    """

    def __init__(self, template_name: str):
        """
        Initialize the base message template.

        Args:
            template_name (str): Name identifier for the template
        """
        self.template_name = template_name
        self.supported_formats = [MessageFormat.CONSOLE]

    def substitute_variables(self, template: str, variables: Dict[str, Any]) -> str:
        """
        Substitute variables in a template string.

        Uses Python's string.Template for safe variable substitution,
        handling missing variables gracefully.

        Args:
            template (str): Template string with $variable placeholders
            variables (Dict[str, Any]): Variables to substitute

        Returns:
            str: Template with variables substituted
        """
        try:
            template_obj = string.Template(template)
            return template_obj.safe_substitute(variables)
        except (KeyError, ValueError):
            # Return original template if substitution fails
            return template

    def format_context(
        self, context: Optional[Dict[str, Any]], format_type: MessageFormat
    ) -> str:
        """
        Format context information for display.

        Args:
            context (Dict[str, Any], optional): Context information
            format_type (MessageFormat): Output format type

        Returns:
            str: Formatted context string
        """
        if not context:
            return ""

        if format_type == MessageFormat.JSON:
            try:
                return json.dumps(context, indent=2, default=str)
            except (TypeError, ValueError):
                return str(context)
        elif format_type == MessageFormat.STRUCTURED:
            items = []
            for key, value in context.items():
                items.append(f"  {key}: {value}")
            return "\n".join(items)
        elif format_type == MessageFormat.CONSOLE:
            items = []
            for key, value in context.items():
                items.append(f"{key}={value}")
            return ", ".join(items)
        else:
            return str(context)

    def get_severity_symbol(self, severity: Optional[ErrorSeverity]) -> str:
        """
        Get a visual symbol for the error severity.

        Args:
            severity (ErrorSeverity, optional): Error severity level

        Returns:
            str: Symbol representing the severity
        """
        if not severity:
            return "⚠"

        symbols = {
            ErrorSeverity.CRITICAL: "🔴",
            ErrorSeverity.ERROR: "❌",
            ErrorSeverity.WARNING: "⚠️",
            ErrorSeverity.INFO: "ℹ️",
        }
        return symbols.get(severity, "⚠")

    def get_severity_color_code(self, severity: Optional[ErrorSeverity]) -> str:
        """
        Get ANSI color code for severity level.

        Args:
            severity (ErrorSeverity, optional): Error severity level

        Returns:
            str: ANSI color code
        """
        if not severity:
            return "\033[33m"  # Yellow

        colors = {
            ErrorSeverity.CRITICAL: "\033[41m\033[97m",  # Red background, white text
            ErrorSeverity.ERROR: "\033[91m",  # Bright red
            ErrorSeverity.WARNING: "\033[93m",  # Bright yellow
            ErrorSeverity.INFO: "\033[94m",  # Bright blue
        }
        return colors.get(severity, "\033[33m")

    @abstractmethod
    def format_message(
        self,
        message: str,
        error_code: str,
        error_info: Optional[ErrorCodeInfo] = None,
        context: Optional[Dict[str, Any]] = None,
        format_type: MessageFormat = MessageFormat.CONSOLE,
        **kwargs,
    ) -> str:
        """
        Format an error message using the template.

        Args:
            message (str): The base error message
            error_code (str): The error code
            error_info (ErrorCodeInfo, optional): Error metadata information
            context (Dict[str, Any], optional): Additional context data
            format_type (MessageFormat): The desired output format
            **kwargs: Additional formatting parameters

        Returns:
            str: The formatted error message
        """


class StandardMessageTemplate(BaseMessageTemplate):
    """
    Standard error message template with comprehensive formatting options.

    This template provides professional, consistent error message formatting
    across all severity levels and output formats. It includes severity
    indicators, context information, and resolution guidance.

    Features:
        - Severity-based visual indicators and color coding
        - Context information formatting
        - Resolution hints and suggestions
        - Multiple output format support
        - Timestamp inclusion for structured formats
        - Clean API response formatting
    """

    def __init__(self):
        """Initialize the standard message template."""
        super().__init__("standard")
        self.supported_formats = [
            MessageFormat.CONSOLE,
            MessageFormat.STRUCTURED,
            MessageFormat.API,
            MessageFormat.JSON,
            MessageFormat.MARKDOWN,
        ]

    def format_message(
        self,
        message: str,
        error_code: str,
        error_info: Optional[ErrorCodeInfo] = None,
        context: Optional[Dict[str, Any]] = None,
        format_type: MessageFormat = MessageFormat.CONSOLE,
        include_timestamp: bool = False,
        include_colors: bool = True,
        include_context: bool = True,
        include_resolution: bool = True,
        **kwargs,
    ) -> str:
        """
        Format an error message using the standard template.

        Args:
            message (str): The base error message
            error_code (str): The error code
            error_info (ErrorCodeInfo, optional): Error metadata information
            context (Dict[str, Any], optional): Additional context data
            format_type (MessageFormat): The desired output format
            include_timestamp (bool): Whether to include timestamp
            include_colors (bool): Whether to include ANSI color codes
            include_context (bool): Whether to include context information
            include_resolution (bool): Whether to include resolution hints
            **kwargs: Additional formatting parameters

        Returns:
            str: The formatted error message
        """
        error_info.severity if error_info else None

        if format_type == MessageFormat.CONSOLE:
            return self._format_console(
                message,
                error_code,
                error_info,
                context,
                include_colors,
                include_context,
                include_resolution,
            )
        elif format_type == MessageFormat.STRUCTURED:
            return self._format_structured(
                message,
                error_code,
                error_info,
                context,
                include_timestamp,
                include_context,
                include_resolution,
            )
        elif format_type == MessageFormat.API:
            return self._format_api(
                message, error_code, error_info, context, include_context
            )
        elif format_type == MessageFormat.JSON:
            return self._format_json(
                message,
                error_code,
                error_info,
                context,
                include_timestamp,
                include_context,
                include_resolution,
            )
        elif format_type == MessageFormat.MARKDOWN:
            return self._format_markdown(
                message,
                error_code,
                error_info,
                context,
                include_context,
                include_resolution,
            )
        else:
            # Fallback to console format
            return self._format_console(
                message,
                error_code,
                error_info,
                context,
                False,
                include_context,
                include_resolution,
            )

    def _format_console(
        self,
        message: str,
        error_code: str,
        error_info: Optional[ErrorCodeInfo],
        context: Optional[Dict[str, Any]],
        include_colors: bool,
        include_context: bool,
        include_resolution: bool,
    ) -> str:
        """Format message for console output."""
        parts = []
        severity = error_info.severity if error_info else None

        # Build severity indicator
        if include_colors and severity:
            color_code = self.get_severity_color_code(severity)
            reset_code = "\033[0m"
            severity_label = f"{color_code}[{severity.name}]{reset_code}"
        else:
            severity_symbol = self.get_severity_symbol(severity)
            severity_name = severity.name if severity else "ERROR"
            severity_label = f"{severity_symbol} [{severity_name}]"

        # Main error line
        parts.append(f"{severity_label} {error_code}: {message}")

        # Add description if available
        if error_info and error_info.description:
            parts.append(f"Description: {error_info.description}")

        # Add context if requested and available
        if include_context and context:
            context_str = self.format_context(context, MessageFormat.CONSOLE)
            if context_str:
                parts.append(f"Context: {context_str}")

        # Add resolution hint if requested and available
        if include_resolution and error_info and error_info.resolution_hint:
            parts.append(f"Resolution: {error_info.resolution_hint}")

        return "\n".join(parts)

    def _format_structured(
        self,
        message: str,
        error_code: str,
        error_info: Optional[ErrorCodeInfo],
        context: Optional[Dict[str, Any]],
        include_timestamp: bool,
        include_context: bool,
        include_resolution: bool,
    ) -> str:
        """Format message for structured logging."""
        parts = []

        # Timestamp
        if include_timestamp:
            timestamp = datetime.now().isoformat()
            parts.append(f"Timestamp: {timestamp}")

        # Severity and basic info
        severity_name = error_info.severity.name if error_info else "ERROR"
        parts.append(f"Severity: {severity_name}")
        parts.append(f"Code: {error_code}")

        # Module and category if available
        if error_info:
            parts.append(f"Module: {error_info.module}")
            parts.append(f"Category: {error_info.category}")

        # Message
        parts.append(f"Message: {message}")

        # Description
        if error_info and error_info.description:
            parts.append(f"Description: {error_info.description}")

        # Context
        if include_context and context:
            parts.append("Context:")
            context_str = self.format_context(context, MessageFormat.STRUCTURED)
            parts.append(context_str)

        # Resolution
        if include_resolution and error_info and error_info.resolution_hint:
            parts.append(f"Resolution: {error_info.resolution_hint}")

        return "\n".join(parts)

    def _format_api(
        self,
        message: str,
        error_code: str,
        error_info: Optional[ErrorCodeInfo],
        context: Optional[Dict[str, Any]],
        include_context: bool,
    ) -> str:
        """Format message for API responses (clean, professional)."""
        # For API responses, keep it clean and professional
        severity_name = error_info.severity.name if error_info else "ERROR"

        parts = [f"[{severity_name}] {message}"]

        # Add resolution hint for API users
        if error_info and error_info.resolution_hint:
            parts.append(f"Suggested action: {error_info.resolution_hint}")

        # Include minimal context if requested
        if include_context and context:
            # Only include non-sensitive context items
            safe_context = {
                k: v
                for k, v in context.items()
                if not any(
                    sensitive in k.lower()
                    for sensitive in ["password", "key", "token", "secret"]
                )
            }
            if safe_context:
                context_str = self.format_context(safe_context, MessageFormat.CONSOLE)
                parts.append(f"Details: {context_str}")

        return " | ".join(parts)

    def _format_json(
        self,
        message: str,
        error_code: str,
        error_info: Optional[ErrorCodeInfo],
        context: Optional[Dict[str, Any]],
        include_timestamp: bool,
        include_context: bool,
        include_resolution: bool,
    ) -> str:
        """Format message as JSON structure."""
        result = {
            "error_code": error_code,
            "message": message,
            "severity": error_info.severity.name if error_info else "ERROR",
        }

        if include_timestamp:
            result["timestamp"] = datetime.now().isoformat()

        if error_info:
            result["module"] = error_info.module
            result["category"] = error_info.category
            result["description"] = error_info.description

            if include_resolution:
                result["resolution_hint"] = error_info.resolution_hint

        if include_context and context:
            result["context"] = context

        try:
            return json.dumps(result, indent=2, default=str)
        except (TypeError, ValueError):
            # Fallback if JSON serialization fails
            return json.dumps(
                {
                    "error_code": error_code,
                    "message": message,
                    "severity": result["severity"],
                    "serialization_error": "Failed to serialize complete error data",
                },
                indent=2,
            )

    def _format_markdown(
        self,
        message: str,
        error_code: str,
        error_info: Optional[ErrorCodeInfo],
        context: Optional[Dict[str, Any]],
        include_context: bool,
        include_resolution: bool,
    ) -> str:
        """Format message as Markdown."""
        parts = []

        # Title with severity emoji
        severity = error_info.severity if error_info else None
        emoji = self.get_severity_symbol(severity)
        severity_name = severity.name if severity else "ERROR"

        parts.append(f"## {emoji} {severity_name}: {error_code}")
        parts.append("")

        # Message
        parts.append(f"**Message:** {message}")
        parts.append("")

        # Description
        if error_info and error_info.description:
            parts.append(f"**Description:** {error_info.description}")
            parts.append("")

        # Module and category
        if error_info:
            parts.append(f"**Module:** {error_info.module}")
            parts.append(f"**Category:** {error_info.category}")
            parts.append("")

        # Context
        if include_context and context:
            parts.append("**Context:**")
            parts.append("```json")
            parts.append(json.dumps(context, indent=2, default=str))
            parts.append("```")
            parts.append("")

        # Resolution
        if include_resolution and error_info and error_info.resolution_hint:
            parts.append(f"**Resolution:** {error_info.resolution_hint}")
            parts.append("")

        return "\n".join(parts)


class CompactMessageTemplate(BaseMessageTemplate):
    """
    Compact error message template for minimal output.

    This template provides concise error messages suitable for situations
    where space is limited or verbose output is not desired.
    """

    def __init__(self):
        """Initialize the compact message template."""
        super().__init__("compact")
        self.supported_formats = [
            MessageFormat.CONSOLE,
            MessageFormat.API,
            MessageFormat.JSON,
        ]

    def format_message(
        self,
        message: str,
        error_code: str,
        error_info: Optional[ErrorCodeInfo] = None,
        context: Optional[Dict[str, Any]] = None,
        format_type: MessageFormat = MessageFormat.CONSOLE,
        **kwargs,
    ) -> str:
        """Format a compact error message."""
        severity = error_info.severity if error_info else None

        if format_type == MessageFormat.JSON:
            return json.dumps(
                {
                    "code": error_code,
                    "message": message,
                    "severity": severity.name if severity else "ERROR",
                }
            )
        elif format_type == MessageFormat.API:
            severity_name = severity.name if severity else "ERROR"
            return f"[{severity_name}] {error_code}: {message}"
        else:  # Console format
            symbol = self.get_severity_symbol(severity)
            return f"{symbol} {error_code}: {message}"


class MessageTemplateRegistry:
    """
    Registry for managing error message templates.

    This registry provides centralized management of message templates,
    allowing for template selection, configuration, and customization
    across the entire error handling system.

    Features:
        - Template registration and lookup
        - Default template management
        - Template configuration
        - Format-specific template selection
    """

    _instance = None

    def __new__(cls):
        """Implement singleton pattern for the template registry."""
        if cls._instance is None:
            cls._instance = super(MessageTemplateRegistry, cls).__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        """Initialize the message template registry."""
        if self._initialized:
            return

        self._templates: Dict[str, BaseMessageTemplate] = {}
        self._default_template_name = "standard"
        self._format_preferences: Dict[MessageFormat, str] = {}

        # Register built-in templates
        self._register_builtin_templates()
        self._initialized = True

    def _register_builtin_templates(self) -> None:
        """Register built-in message templates."""
        self.register_template(StandardMessageTemplate())
        self.register_template(CompactMessageTemplate())

        # Set format preferences
        self._format_preferences = {
            MessageFormat.CONSOLE: "standard",
            MessageFormat.STRUCTURED: "standard",
            MessageFormat.API: "compact",
            MessageFormat.JSON: "standard",
            MessageFormat.MARKDOWN: "standard",
        }

    def register_template(self, template: BaseMessageTemplate) -> None:
        """
        Register a message template.

        Args:
            template (BaseMessageTemplate): Template to register
        """
        self._templates[template.template_name] = template

    def get_template(self, template_name: Optional[str] = None) -> BaseMessageTemplate:
        """
        Get a message template by name.

        Args:
            template_name (str, optional): Name of template to retrieve.
                If None, returns default template.

        Returns:
            BaseMessageTemplate: The requested template

        Raises:
            KeyError: If template name is not found
        """
        if template_name is None:
            template_name = self._default_template_name

        if template_name not in self._templates:
            # Fallback to default template
            template_name = self._default_template_name

        return self._templates[template_name]

    def get_preferred_template(self, format_type: MessageFormat) -> BaseMessageTemplate:
        """
        Get the preferred template for a specific format.

        Args:
            format_type (MessageFormat): The message format type

        Returns:
            BaseMessageTemplate: The preferred template for the format
        """
        preferred_name = self._format_preferences.get(
            format_type, self._default_template_name
        )
        return self.get_template(preferred_name)

    def set_default_template(self, template_name: str) -> None:
        """
        Set the default template.

        Args:
            template_name (str): Name of template to set as default

        Raises:
            KeyError: If template name is not registered
        """
        if template_name not in self._templates:
            raise KeyError(f"Template '{template_name}' is not registered")
        self._default_template_name = template_name

    def set_format_preference(
        self, format_type: MessageFormat, template_name: str
    ) -> None:
        """
        Set the preferred template for a format.

        Args:
            format_type (MessageFormat): The message format type
            template_name (str): Name of preferred template

        Raises:
            KeyError: If template name is not registered
        """
        if template_name not in self._templates:
            raise KeyError(f"Template '{template_name}' is not registered")
        self._format_preferences[format_type] = template_name

    def list_templates(self) -> List[str]:
        """
        Get list of registered template names.

        Returns:
            List[str]: List of template names
        """
        return list(self._templates.keys())

    def format_message(
        self,
        message: str,
        error_code: str,
        error_info: Optional[ErrorCodeInfo] = None,
        context: Optional[Dict[str, Any]] = None,
        format_type: MessageFormat = MessageFormat.CONSOLE,
        template_name: Optional[str] = None,
        **kwargs,
    ) -> str:
        """
        Format a message using the registry.

        Args:
            message (str): The base error message
            error_code (str): The error code
            error_info (ErrorCodeInfo, optional): Error metadata information
            context (Dict[str, Any], optional): Additional context data
            format_type (MessageFormat): The desired output format
            template_name (str, optional): Specific template to use
            **kwargs: Additional formatting parameters

        Returns:
            str: The formatted error message
        """
        if template_name:
            template = self.get_template(template_name)
        else:
            template = self.get_preferred_template(format_type)

        return template.format_message(
            message, error_code, error_info, context, format_type, **kwargs
        )


# Module-specific constant groups for easy access
class BaseErrorCodes:
    """
    Convenience class providing easy access to Base module error codes.

    This class groups all Base module error codes into logical categories
    for easier access and discovery in IDE autocomplete.

    Examples:
        >>> raise AIM2Exception(
        ...     "System initialization failed",
        ...     error_code=BaseErrorCodes.SYSTEM_INIT_FAILURE
        ... )
    """

    # System errors
    SYSTEM_INIT_FAILURE = AIM2ErrorCodes.BASE_SYS_C_001
    MEMORY_ALLOCATION_FAILURE = AIM2ErrorCodes.BASE_SYS_C_002
    MODULE_IMPORT_FAILURE = AIM2ErrorCodes.BASE_SYS_E_003
    ENVIRONMENT_VARIABLE_MISSING = AIM2ErrorCodes.BASE_SYS_E_004

    # Configuration errors
    CONFIG_FILE_NOT_FOUND = AIM2ErrorCodes.BASE_CFG_E_001
    INVALID_CONFIG_SYNTAX = AIM2ErrorCodes.BASE_CFG_E_002
    MISSING_CONFIG_FIELD = AIM2ErrorCodes.BASE_CFG_E_003
    DEPRECATED_CONFIG_FIELD = AIM2ErrorCodes.BASE_CFG_W_004

    # Persistence errors
    DATABASE_CONNECTION_FAILURE = AIM2ErrorCodes.BASE_PER_C_001
    FILE_PERMISSION_DENIED = AIM2ErrorCodes.BASE_PER_E_002
    DISK_SPACE_INSUFFICIENT = AIM2ErrorCodes.BASE_PER_E_003
    FILE_CORRUPTION_DETECTED = AIM2ErrorCodes.BASE_PER_E_004


class OntologyErrorCodes:
    """
    Convenience class providing easy access to Ontology module error codes.

    Examples:
        >>> raise OntologyException(
        ...     "Failed to parse OWL file",
        ...     error_code=OntologyErrorCodes.OWL_PARSING_FAILURE
        ... )
    """

    # Parsing errors
    OWL_PARSING_FAILURE = AIM2ErrorCodes.ONTO_PARS_E_001
    RDF_EXTRACTION_FAILURE = AIM2ErrorCodes.ONTO_PARS_E_002
    JSONLD_CONTEXT_FAILURE = AIM2ErrorCodes.ONTO_PARS_E_003
    UNSUPPORTED_FORMAT = AIM2ErrorCodes.ONTO_PARS_E_004

    # Validation errors
    CONSISTENCY_CHECK_FAILED = AIM2ErrorCodes.ONTO_VALD_E_001
    HIERARCHY_VIOLATION = AIM2ErrorCodes.ONTO_VALD_E_002
    PROPERTY_MISMATCH = AIM2ErrorCodes.ONTO_VALD_E_003
    ORPHANED_CLASS = AIM2ErrorCodes.ONTO_VALD_W_004

    # Integration errors
    MERGE_CONFLICT = AIM2ErrorCodes.ONTO_INTG_E_001
    NAMESPACE_COLLISION = AIM2ErrorCodes.ONTO_INTG_E_002
    INCOMPATIBLE_VERSIONS = AIM2ErrorCodes.ONTO_INTG_E_003
    DUPLICATE_CONCEPT = AIM2ErrorCodes.ONTO_INTG_W_004


class ExtractionErrorCodes:
    """
    Convenience class providing easy access to Extraction module error codes.

    Examples:
        >>> raise ExtractionException(
        ...     "NER model loading failed",
        ...     error_code=ExtractionErrorCodes.NER_MODEL_LOADING_FAILURE
        ... )
    """

    # NER errors
    NER_MODEL_LOADING_FAILURE = AIM2ErrorCodes.EXTR_NER_E_001
    TEXT_TOKENIZATION_FAILURE = AIM2ErrorCodes.EXTR_NER_E_002
    ENTITY_RECOGNITION_TIMEOUT = AIM2ErrorCodes.EXTR_NER_E_003
    LOW_CONFIDENCE_ENTITY = AIM2ErrorCodes.EXTR_NER_W_004

    # Relationship extraction errors
    PATTERN_MATCHING_FAILED = AIM2ErrorCodes.EXTR_REL_E_001
    ENTITY_PAIR_VALIDATION_FAILED = AIM2ErrorCodes.EXTR_REL_E_002
    DEPENDENCY_PARSING_FAILURE = AIM2ErrorCodes.EXTR_REL_E_003
    NO_RELATIONSHIPS_FOUND = AIM2ErrorCodes.EXTR_REL_I_004


class LLMErrorCodes:
    """
    Convenience class providing easy access to LLM module error codes.

    Examples:
        >>> raise LLMException(
        ...     "API request failed",
        ...     error_code=LLMErrorCodes.API_REQUEST_FAILED
        ... )
    """

    # API errors
    API_REQUEST_FAILED = AIM2ErrorCodes.LLM_API_E_001
    INVALID_API_RESPONSE = AIM2ErrorCodes.LLM_API_E_002
    API_QUOTA_EXCEEDED = AIM2ErrorCodes.LLM_API_E_003
    API_RESPONSE_TRUNCATED = AIM2ErrorCodes.LLM_API_W_004

    # Authentication errors
    INVALID_API_KEY = AIM2ErrorCodes.LLM_AUTH_E_001
    INSUFFICIENT_PERMISSIONS = AIM2ErrorCodes.LLM_AUTH_E_002
    AUTH_SERVICE_UNAVAILABLE = AIM2ErrorCodes.LLM_AUTH_E_003

    # Timeout errors
    REQUEST_TIMEOUT = AIM2ErrorCodes.LLM_TIME_E_001
    MODEL_INFERENCE_TIMEOUT = AIM2ErrorCodes.LLM_TIME_E_002
    SLOW_RESPONSE_DETECTED = AIM2ErrorCodes.LLM_TIME_W_003


class ValidationErrorCodes:
    """
    Convenience class providing easy access to Validation module error codes.

    Examples:
        >>> raise ValidationException(
        ...     "Schema validation failed",
        ...     error_code=ValidationErrorCodes.SCHEMA_VALIDATION_FAILED
        ... )
    """

    # Schema validation errors
    SCHEMA_VALIDATION_FAILED = AIM2ErrorCodes.VALID_SCHM_E_001
    REQUIRED_FIELD_MISSING = AIM2ErrorCodes.VALID_SCHM_E_002
    INVALID_FIELD_TYPE = AIM2ErrorCodes.VALID_SCHM_E_003
    UNKNOWN_FIELD = AIM2ErrorCodes.VALID_SCHM_W_004

    # Data validation errors
    DATA_FORMAT_INVALID = AIM2ErrorCodes.VALID_DATA_E_001
    VALUE_OUT_OF_RANGE = AIM2ErrorCodes.VALID_DATA_E_002
    INVALID_DATA_ENCODING = AIM2ErrorCodes.VALID_DATA_E_003
    DATA_QUALITY_ISSUE = AIM2ErrorCodes.VALID_DATA_W_004


class AIM2Exception(Exception):
    """
    Base exception class for AIM2 project with comprehensive error management.

    This is the foundation exception class that all other AIM2 exceptions inherit from.
    It provides a rich set of features including error code management, cause tracking,
    context information, and serialization capabilities for robust error handling
    and debugging support.

    The class implements a hierarchical error code system where each exception type
    has a default error code based on its class name, but custom error codes can
    be provided for more specific error categorization.

    Attributes:
        message (str): Human-readable error message describing the issue
        error_code (str): Categorization code for the error (auto-generated if not provided)
        error_info (ErrorCodeInfo, optional): Detailed error information if available
        cause (Exception, optional): Original exception that caused this error
        context (Dict[str, Any]): Additional context information for debugging

    Args:
        message (str): Error message describing the issue
        error_code (str, optional): Specific error code for categorization.
            If not provided, defaults to "AIM2_{CLASS_NAME}"
        cause (Exception, optional): Original exception that caused this error.
            Automatically sets __cause__ for proper exception chaining
        context (Dict[str, Any], optional): Additional context information.
            Defaults to empty dict if not provided

    Examples:
        Basic usage:
            >>> raise AIM2Exception("Operation failed")

        With error code:
            >>> raise AIM2Exception("Database error", error_code="DB_CONNECTION_FAILED")

        With cause and context:
            >>> try:
            ...     risky_operation()
            ... except ValueError as e:
            ...     raise AIM2Exception(
            ...         "Processing failed",
            ...         cause=e,
            ...         context={"input_data": data, "step": "validation"}
            ...     )

        Serialization:
            >>> exc = AIM2Exception("Test error", context={"key": "value"})
            >>> data = exc.to_dict()
            >>> restored = AIM2Exception.from_dict(data)
    """

    def __init__(
        self,
        message: str,
        error_code: Optional[Union[str, AIM2ErrorCodes]] = None,
        cause: Optional[Exception] = None,
        context: Optional[Dict[str, Any]] = None,
    ):
        """
        Initialize AIM2Exception with message and optional parameters.

        Args:
            message (str): Error message describing the issue
            error_code (Union[str, AIM2ErrorCodes], optional): Specific error code for categorization.
                Can be either a string or an AIM2ErrorCodes enum value.
                If not provided, defaults to "AIM2_{CLASS_NAME}"
            cause (Exception, optional): Original exception that caused this error
            context (Dict[str, Any], optional): Additional context information

        Note:
            If error_code is None or empty string, the default error code will be
            generated using the pattern "AIM2_{CLASS_NAME}".
        """
        super().__init__(message)
        self.message = message

        # Handle both string and enum error codes with metadata
        if isinstance(error_code, AIM2ErrorCodes):
            self.error_code = error_code.value.code
            self.error_info = error_code.value
        elif isinstance(error_code, str) and error_code.strip():
            self.error_code = error_code
            # Try to get error info from registry for string codes
            registry = ErrorCodeRegistry()
            self.error_info = registry.get_error_info(error_code)
        else:
            # Default case: no error code provided or empty string
            self.error_code = self._get_default_error_code()
            registry = ErrorCodeRegistry()
            self.error_info = registry.get_error_info(self.error_code)

        self.cause = cause
        self.context = context or {}

        # Set the __cause__ attribute for proper exception chaining
        if cause:
            self.__cause__ = cause

    def _get_default_error_code(self) -> str:
        """
        Generate default error code based on exception class name.

        The default error code follows the pattern "AIM2_{CLASS_NAME}" where
        CLASS_NAME is the uppercase version of the exception class name.

        Returns:
            str: Default error code in format "AIM2_{CLASS_NAME}"

        Examples:
            >>> exc = AIM2Exception("test")
            >>> exc._get_default_error_code()
            'AIM2_AIM2EXCEPTION'

            >>> exc = OntologyException("test")
            >>> exc._get_default_error_code()
            'AIM2_ONTOLOGYEXCEPTION'
        """
        return f"AIM2_{self.__class__.__name__.upper()}"

    def get_detailed_message(
        self,
        format_type: MessageFormat = MessageFormat.CONSOLE,
        template_name: Optional[str] = None,
        **kwargs,
    ) -> str:
        """
        Get a detailed error message using the template system.

        Returns a comprehensive error message formatted according to the specified
        format type and template. This method enhances the original functionality
        with multiple output formats, customizable templates, and rich formatting.

        Args:
            format_type (MessageFormat): The desired output format.
                Defaults to MessageFormat.CONSOLE for backward compatibility.
            template_name (str, optional): Specific template to use.
                If None, uses the preferred template for the format type.
            **kwargs: Additional formatting parameters passed to the template

        Returns:
            str: Detailed error message formatted according to the template

        Examples:
            >>> exc = AIM2Exception(
            ...     "Test error",
            ...     error_code=AIM2ErrorCodes.BASE_SYS_C_001
            ... )
            >>> print(exc.get_detailed_message())
            🔴 [CRITICAL] AIM2_BASE_SYS_C_001: Test error
            Description: System initialization failure
            Resolution: Check dependencies and permissions

            >>> print(exc.get_detailed_message(MessageFormat.API))
            [CRITICAL] Test error | Suggested action: Check dependencies and permissions

            >>> print(exc.get_detailed_message(MessageFormat.JSON))
            {
              "error_code": "AIM2_BASE_SYS_C_001",
              "message": "Test error",
              "severity": "CRITICAL",
              "module": "BASE",
              "category": "SYS",
              "description": "System initialization failure",
              "resolution_hint": "Check dependencies and permissions"
            }
        """
        try:
            registry = MessageTemplateRegistry()
            return registry.format_message(
                message=self.message,
                error_code=self.error_code,
                error_info=getattr(self, "error_info", None),
                context=self.context,
                format_type=format_type,
                template_name=template_name,
                **kwargs,
            )
        except Exception:
            # Fallback to original implementation if template system fails
            return self._get_legacy_detailed_message()

    def _get_legacy_detailed_message(self) -> str:
        """
        Legacy implementation of get_detailed_message for backward compatibility.

        This method preserves the original behavior in case the template
        system encounters errors or is unavailable.

        Returns:
            str: Legacy formatted detailed error message
        """
        if hasattr(self, "error_info") and self.error_info:
            severity_label = f"[{self.error_info.severity.name}]"
            detailed_msg = f"{severity_label} {self.error_code}: {self.message}\n"
            detailed_msg += f"Description: {self.error_info.description}\n"
            detailed_msg += f"Resolution: {self.error_info.resolution_hint}"
            return detailed_msg
        else:
            return f"[ERROR] {self.error_code}: {self.message}"

    def format_message(
        self, format_type: MessageFormat, template_name: Optional[str] = None, **kwargs
    ) -> str:
        """
        Format the exception message using a specific format and template.

        This method provides direct access to the template formatting system,
        allowing for fine-grained control over message formatting without
        changing the default detailed message behavior.

        Args:
            format_type (MessageFormat): The desired output format
            template_name (str, optional): Specific template to use
            **kwargs: Additional formatting parameters

        Returns:
            str: Formatted error message

        Examples:
            >>> exc = AIM2Exception("Test error", context={"key": "value"})
            >>> console_msg = exc.format_message(MessageFormat.CONSOLE)
            >>> api_msg = exc.format_message(MessageFormat.API)
            >>> json_msg = exc.format_message(MessageFormat.JSON)
            >>> structured_msg = exc.format_message(
            ...     MessageFormat.STRUCTURED,
            ...     include_timestamp=True
            ... )
        """
        try:
            registry = MessageTemplateRegistry()
            return registry.format_message(
                message=self.message,
                error_code=self.error_code,
                error_info=getattr(self, "error_info", None),
                context=self.context,
                format_type=format_type,
                template_name=template_name,
                **kwargs,
            )
        except Exception:
            # Fallback to basic formatting
            return f"{self.error_code}: {self.message}"

    def get_console_message(self, include_colors: bool = True, **kwargs) -> str:
        """
        Get a console-formatted error message.

        Convenience method for getting console output with optional color coding.

        Args:
            include_colors (bool): Whether to include ANSI color codes
            **kwargs: Additional formatting parameters

        Returns:
            str: Console-formatted error message
        """
        return self.format_message(
            MessageFormat.CONSOLE, include_colors=include_colors, **kwargs
        )

    def get_api_message(self, include_context: bool = False, **kwargs) -> str:
        """
        Get an API-formatted error message.

        Convenience method for getting clean API response messages.

        Args:
            include_context (bool): Whether to include context information
            **kwargs: Additional formatting parameters

        Returns:
            str: API-formatted error message
        """
        return self.format_message(
            MessageFormat.API, include_context=include_context, **kwargs
        )

    def get_json_message(
        self, include_timestamp: bool = False, include_context: bool = True, **kwargs
    ) -> str:
        """
        Get a JSON-formatted error message.

        Convenience method for getting structured JSON error data.

        Args:
            include_timestamp (bool): Whether to include timestamp
            include_context (bool): Whether to include context information
            **kwargs: Additional formatting parameters

        Returns:
            str: JSON-formatted error message
        """
        return self.format_message(
            MessageFormat.JSON,
            include_timestamp=include_timestamp,
            include_context=include_context,
            **kwargs,
        )

    def get_structured_message(
        self, include_timestamp: bool = True, include_context: bool = True, **kwargs
    ) -> str:
        """
        Get a structured error message for logging.

        Convenience method for getting detailed structured output suitable
        for logging systems and debugging.

        Args:
            include_timestamp (bool): Whether to include timestamp
            include_context (bool): Whether to include context information
            **kwargs: Additional formatting parameters

        Returns:
            str: Structured error message
        """
        return self.format_message(
            MessageFormat.STRUCTURED,
            include_timestamp=include_timestamp,
            include_context=include_context,
            **kwargs,
        )

    def get_markdown_message(
        self, include_context: bool = True, include_resolution: bool = True, **kwargs
    ) -> str:
        """
        Get a Markdown-formatted error message.

        Convenience method for getting Markdown output suitable for
        documentation and reports.

        Args:
            include_context (bool): Whether to include context information
            include_resolution (bool): Whether to include resolution hints
            **kwargs: Additional formatting parameters

        Returns:
            str: Markdown-formatted error message
        """
        return self.format_message(
            MessageFormat.MARKDOWN,
            include_context=include_context,
            include_resolution=include_resolution,
            **kwargs,
        )

    def __str__(self) -> str:
        """
        Return string representation of the exception.

        Uses the detailed message format if error info is available,
        otherwise falls back to the standard message format.

        Returns:
            str: String representation of the exception
        """
        if hasattr(self, "error_info") and self.error_info:
            return self.get_detailed_message()
        else:
            return f"{self.error_code}: {self.message}"

    def to_dict(self) -> Dict[str, Any]:
        """
        Serialize exception to dictionary format for storage or transmission.

        Creates a comprehensive dictionary representation of the exception
        including all relevant information for debugging and error tracking.
        The serialized format can be used for logging, API responses, or
        persistent storage.

        Returns:
            Dict[str, Any]: Dictionary containing exception information with keys:
                - exception_type (str): Name of the exception class
                - message (str): Error message
                - error_code (str): Error categorization code
                - context (Dict[str, Any]): Additional context information
                - cause_type (str, optional): Class name of the causing exception
                - cause_message (str, optional): Message from the causing exception
                - traceback (str, optional): Stack trace if available

        Examples:
            >>> exc = AIM2Exception("Test error", error_code="TEST_001",
            ...                     context={"key": "value"})
            >>> data = exc.to_dict()
            >>> print(data)
            {
                'exception_type': 'AIM2Exception',
                'message': 'Test error',
                'error_code': 'TEST_001',
                'context': {'key': 'value'},
                'cause_type': None,
                'cause_message': None,
                'traceback': None
            }
        """
        result = {
            "exception_type": self.__class__.__name__,
            "message": self.message,
            "error_code": self.error_code,
            "context": self.context,
            "cause_type": self.cause.__class__.__name__ if self.cause else None,
            "cause_message": str(self.cause) if self.cause else None,
            "traceback": traceback.format_exc()
            if hasattr(self, "__traceback__") and self.__traceback__
            else None,
        }

        # Include error_info if available
        if hasattr(self, "error_info") and self.error_info:
            result["error_info"] = {
                "module": self.error_info.module,
                "category": self.error_info.category,
                "severity": self.error_info.severity.name,
                "description": self.error_info.description,
                "resolution_hint": self.error_info.resolution_hint,
            }

        return result

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "AIM2Exception":
        """
        Deserialize exception from dictionary format.

        Reconstructs an exception instance from a dictionary representation,
        typically created by the to_dict() method. This enables exception
        persistence, transmission, and reconstruction across different
        contexts and processes.

        Args:
            data (Dict[str, Any]): Dictionary containing exception information
                Required keys:
                    - message (str): Error message
                Optional keys:
                    - error_code (str): Error categorization code
                    - context (Dict[str, Any]): Additional context
                    - cause_type (str): Class name of causing exception
                    - cause_message (str): Message from causing exception
                    - error_info (Dict[str, Any]): Error metadata information

        Returns:
            AIM2Exception: Reconstructed exception instance

        Note:
            If cause information is present in the data, a generic Exception
            will be created to represent the original cause. The original
            exception type information is preserved in the cause_type field
            but the actual exception object will be a generic Exception.

            The error_info metadata will be reconstructed either from the
            serialized error_info data or by looking up the error_code in
            the error code registry.

        Examples:
            >>> data = {
            ...     'message': 'Test error',
            ...     'error_code': 'TEST_001',
            ...     'context': {'key': 'value'},
            ...     'cause_type': 'ValueError',
            ...     'cause_message': 'Invalid input'
            ... }
            >>> exc = AIM2Exception.from_dict(data)
            >>> print(exc.message)
            'Test error'
            >>> print(exc.error_code)
            'TEST_001'
            >>> print(str(exc.cause))
            'Invalid input'
        """
        cause = None
        if data.get("cause_type") and data.get("cause_message"):
            # Create a generic exception for the cause since we can't reconstruct
            # the exact original exception type without importing it
            cause = Exception(data["cause_message"])

        # Create the exception instance
        instance = cls(
            message=data["message"],
            error_code=data.get("error_code"),
            cause=cause,
            context=data.get("context", {}),
        )

        # If error_info was serialized but not automatically restored during
        # initialization, attempt to restore it from the serialized data
        if "error_info" in data and data["error_info"] and not instance.error_info:
            try:
                error_info_data = data["error_info"]
                # Reconstruct ErrorCodeInfo from serialized data
                severity = ErrorSeverity[error_info_data["severity"]]
                instance.error_info = ErrorCodeInfo(
                    code=instance.error_code,
                    module=error_info_data["module"],
                    category=error_info_data["category"],
                    severity=severity,
                    description=error_info_data["description"],
                    resolution_hint=error_info_data["resolution_hint"],
                )
            except (KeyError, ValueError, TypeError):
                # If reconstruction fails, error_info will remain None
                # This maintains backward compatibility with old serialization formats
                pass

        return instance


class OntologyException(AIM2Exception):
    """
    Exception for ontology-related operations and errors.

    This exception is raised when operations involving ontology management,
    loading, parsing, validation, integration, or manipulation encounter errors.
    It provides specialized error handling for the ontology management subsystem
    of the AIM2 project.

    Common use cases include:
        - Ontology file parsing errors
        - Ontology validation failures
        - Ontology integration conflicts
        - Ontology schema violations
        - Ontology query or reasoning errors
        - Ontology export/import failures

    Inherits all functionality from AIM2Exception including error codes,
    cause tracking, context information, and serialization support.

    Examples:
        File loading error:
            >>> raise OntologyException(
            ...     "Failed to load ontology file",
            ...     context={"file_path": "/path/to/ontology.owl", "format": "RDF/XML"}
            ... )

        Validation error with cause:
            >>> try:
            ...     validate_ontology_schema(ontology)
            ... except SchemaError as e:
            ...     raise OntologyException(
            ...         "Ontology schema validation failed",
            ...         cause=e,
            ...         context={"ontology_id": "bio_ontology_v1.2"}
            ...     )

        Integration conflict:
            >>> raise OntologyException(
            ...     "Ontology merge conflict detected",
            ...     error_code="ONTOLOGY_MERGE_CONFLICT",
            ...     context={
            ...         "source_ontology": "onto_a.owl",
            ...         "target_ontology": "onto_b.owl",
            ...         "conflicting_concepts": ["Protein", "Gene"]
            ...     }
            ... )
    """

    def _get_default_error_code(self) -> str:
        """
        Get default error code for ontology exceptions.

        Returns:
            str: Default error code "AIM2_ONTOLOGY_ERROR"
        """
        return "AIM2_ONTOLOGY_ERROR"

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "OntologyException":
        """
        Deserialize OntologyException from dictionary format.

        This method ensures that deserialized exceptions are reconstructed
        as the correct OntologyException type rather than the base AIM2Exception.

        Args:
            data (Dict[str, Any]): Dictionary containing exception information

        Returns:
            OntologyException: Reconstructed OntologyException instance

        Examples:
            >>> data = {
            ...     'exception_type': 'OntologyException',
            ...     'message': 'OWL parsing failed',
            ...     'error_code': 'AIM2_ONTO_PARS_E_001',
            ...     'context': {'file': 'ontology.owl'}
            ... }
            >>> exc = OntologyException.from_dict(data)
            >>> isinstance(exc, OntologyException)
            True
        """
        return super().from_dict(data)


class ExtractionException(AIM2Exception):
    """
    Exception for information extraction operations and errors.

    This exception is raised when operations involving text processing,
    named entity recognition (NER), relationship extraction, corpus building,
    or other information extraction processes encounter errors. It provides
    specialized error handling for the extraction subsystem of the AIM2 project.

    Common use cases include:
        - Text preprocessing and tokenization errors
        - NER model loading or inference failures
        - Relationship extraction pipeline errors
        - Entity linking and disambiguation failures
        - Corpus building and processing errors
        - Feature extraction and vectorization issues
        - Model training or evaluation failures

    Inherits all functionality from AIM2Exception including error codes,
    cause tracking, context information, and serialization support.

    Examples:
        NER processing error:
            >>> raise ExtractionException(
            ...     "NER model inference failed",
            ...     context={
            ...         "model_name": "bio-ner-bert",
            ...         "input_text_length": 1024,
            ...         "batch_size": 32
            ...     }
            ... )

        Relationship extraction error:
            >>> raise ExtractionException(
            ...     "Failed to extract protein-protein interactions",
            ...     error_code="RELATION_EXTRACTION_FAILED",
            ...     context={
            ...         "document_id": "PMC123456",
            ...         "sentence_count": 45,
            ...         "entity_pairs": [("EGFR", "TP53"), ("BRCA1", "BRCA2")]
            ...     }
            ... )

        Text processing pipeline error:
            >>> try:
            ...     process_document(doc)
            ... except EncodingError as e:
            ...     raise ExtractionException(
            ...         "Document encoding error in processing pipeline",
            ...         cause=e,
            ...         context={
            ...             "document_path": "/data/docs/paper.pdf",
            ...             "detected_encoding": "unknown",
            ...             "pipeline_stage": "text_extraction"
            ...         }
            ...     )
    """

    def _get_default_error_code(self) -> str:
        """
        Get default error code for extraction exceptions.

        Returns:
            str: Default error code "AIM2_EXTRACTION_ERROR"
        """
        return "AIM2_EXTRACTION_ERROR"

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ExtractionException":
        """
        Deserialize ExtractionException from dictionary format.

        This method ensures that deserialized exceptions are reconstructed
        as the correct ExtractionException type rather than the base AIM2Exception.

        Args:
            data (Dict[str, Any]): Dictionary containing exception information

        Returns:
            ExtractionException: Reconstructed ExtractionException instance

        Examples:
            >>> data = {
            ...     'exception_type': 'ExtractionException',
            ...     'message': 'NER model loading failed',
            ...     'error_code': 'AIM2_EXTR_NER_E_001',
            ...     'context': {'model_path': '/path/to/model'}
            ... }
            >>> exc = ExtractionException.from_dict(data)
            >>> isinstance(exc, ExtractionException)
            True
        """
        return super().from_dict(data)


class LLMException(AIM2Exception):
    """
    Exception for Large Language Model (LLM) interface operations and errors.

    This exception is raised when operations involving LLM API calls,
    model interactions, prompt processing, response parsing, or other
    LLM-related operations encounter errors. It provides specialized
    error handling for the LLM integration subsystem of the AIM2 project.

    Common use cases include:
        - LLM API connection and authentication errors
        - Request timeout and rate limiting issues
        - Model inference and generation failures
        - Prompt formatting and validation errors
        - Response parsing and validation issues
        - Model configuration and parameter errors
        - Token limit and quota exceeded errors

    Inherits all functionality from AIM2Exception including error codes,
    cause tracking, context information, and serialization support.

    Examples:
        API timeout error:
            >>> raise LLMException(
            ...     "OpenAI API request timeout",
            ...     error_code="LLM_API_TIMEOUT",
            ...     context={
            ...         "model": "gpt-3.5-turbo",
            ...         "timeout_duration": 30,
            ...         "prompt_length": 2048
            ...     }
            ... )

        Model inference error:
            >>> raise LLMException(
            ...     "Model inference failed for entity extraction",
            ...     context={
            ...         "model_name": "bio-llm-7b",
            ...         "input_tokens": 1500,
            ...         "max_tokens": 512,
            ...         "temperature": 0.3,
            ...         "error_type": "cuda_out_of_memory"
            ...     }
            ... )

        Authentication error with cause:
            >>> try:
            ...     authenticate_llm_service()
            ... except AuthenticationError as e:
            ...     raise LLMException(
            ...         "Failed to authenticate with LLM service",
            ...         cause=e,
            ...         context={
            ...             "service": "huggingface_hub",
            ...             "api_key_provided": True,
            ...             "endpoint": "https://api.huggingface.co"
            ...         }
            ...     )
    """

    def _get_default_error_code(self) -> str:
        """
        Get default error code for LLM exceptions.

        Returns:
            str: Default error code "AIM2_LLM_ERROR"
        """
        return "AIM2_LLM_ERROR"

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "LLMException":
        """
        Deserialize LLMException from dictionary format.

        This method ensures that deserialized exceptions are reconstructed
        as the correct LLMException type rather than the base AIM2Exception.

        Args:
            data (Dict[str, Any]): Dictionary containing exception information

        Returns:
            LLMException: Reconstructed LLMException instance

        Examples:
            >>> data = {
            ...     'exception_type': 'LLMException',
            ...     'message': 'API request failed',
            ...     'error_code': 'AIM2_LLM_API_E_001',
            ...     'context': {'model': 'gpt-3.5-turbo'}
            ... }
            >>> exc = LLMException.from_dict(data)
            >>> isinstance(exc, LLMException)
            True
        """
        return super().from_dict(data)


class ValidationException(AIM2Exception):
    """
    Exception for data validation operations and errors.

    This exception is raised when operations involving data validation,
    schema validation, configuration validation, input sanitization,
    or other validation processes encounter errors. It provides specialized
    error handling for validation operations across the AIM2 project.

    Common use cases include:
        - JSON/YAML schema validation failures
        - Configuration file validation errors
        - Input data format validation issues
        - Database constraint validation failures
        - API parameter validation errors
        - File format and structure validation issues
        - Business rule and logic validation failures

    Inherits all functionality from AIM2Exception including error codes,
    cause tracking, context information, and serialization support.

    Examples:
        Schema validation error:
            >>> raise ValidationException(
            ...     "Configuration schema validation failed",
            ...     error_code="CONFIG_SCHEMA_INVALID",
            ...     context={
            ...         "config_file": "/etc/aim2/config.yaml",
            ...         "schema_version": "1.0",
            ...         "validation_errors": [
            ...             "Missing required field: database.host",
            ...             "Invalid port number: -1"
            ...         ]
            ...     }
            ... )

        Input validation error:
            >>> raise ValidationException(
            ...     "Invalid input parameters for extraction pipeline",
            ...     context={
            ...         "invalid_fields": ["model_path", "batch_size"],
            ...         "model_path": None,
            ...         "batch_size": -5,
            ...         "expected_batch_size_range": "1-1000"
            ...     }
            ... )

        Database constraint violation:
            >>> try:
            ...     insert_ontology_record(record)
            ... except IntegrityError as e:
            ...     raise ValidationException(
            ...         "Database constraint violation",
            ...         cause=e,
            ...         context={
            ...             "table": "ontologies",
            ...             "constraint": "unique_ontology_id",
            ...             "attempted_id": "bio_onto_v1.0",
            ...             "operation": "insert"
            ...         }
            ...     )
    """

    def _get_default_error_code(self) -> str:
        """
        Get default error code for validation exceptions.

        Returns:
            str: Default error code "AIM2_VALIDATION_ERROR"
        """
        return "AIM2_VALIDATION_ERROR"

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ValidationException":
        """
        Deserialize ValidationException from dictionary format.

        This method ensures that deserialized exceptions are reconstructed
        as the correct ValidationException type rather than the base AIM2Exception.

        Args:
            data (Dict[str, Any]): Dictionary containing exception information

        Returns:
            ValidationException: Reconstructed ValidationException instance

        Examples:
            >>> data = {
            ...     'exception_type': 'ValidationException',
            ...     'message': 'Schema validation failed',
            ...     'error_code': 'AIM2_VALID_SCHM_E_001',
            ...     'context': {'schema_file': 'config.schema.json'}
            ... }
            >>> exc = ValidationException.from_dict(data)
            >>> isinstance(exc, ValidationException)
            True
        """
        return super().from_dict(data)
