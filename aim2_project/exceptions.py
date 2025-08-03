"""
AIM2 Exception Hierarchy Module

This module provides a comprehensive exception hierarchy for the AIM2 project,
including base and specialized exception classes with error code management,
serialization support, and cause tracking capabilities.

The exception hierarchy follows a structured approach where all AIM2-specific
exceptions inherit from the base AIM2Exception class, providing consistent
error handling, logging, and debugging capabilities across the entire project.

Exception Classes:
    AIM2Exception: Base exception class with error code system and serialization
    OntologyException: Exceptions related to ontology operations
    ExtractionException: Exceptions related to information extraction
    LLMException: Exceptions related to LLM interface operations
    ValidationException: Exceptions related to data validation

Features:
    - Hierarchical error code system with automatic default codes
    - Exception chaining and cause tracking
    - Context information support for debugging
    - Serialization and deserialization support (to_dict/from_dict)
    - Rich metadata support for comprehensive error reporting
    - Comprehensive docstrings and type hints

Dependencies:
    - traceback: For traceback information in serialization
    - typing: For type hints and annotations

Usage:
    # Basic usage
    raise AIM2Exception("Something went wrong")

    # With error code and context
    raise OntologyException(
        "Failed to parse ontology file",
        error_code="ONTOLOGY_PARSE_ERROR",
        context={"file": "ontology.owl", "line": 42}
    )

    # With cause chaining
    try:
        # Some operation that might fail
        pass
    except FileNotFoundError as e:
        raise ExtractionException(
            "Could not load extraction model",
            cause=e,
            context={"model_path": "/path/to/model"}
        )

    # Serialization
    exception = ValidationException("Invalid config")
    exception_data = exception.to_dict()
    restored_exception = ValidationException.from_dict(exception_data)

Authors:
    AIM2 Development Team

Version:
    1.0.0

Created:
    2025-08-03
"""

import traceback
from typing import Dict, Any, Optional


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
        error_code: Optional[str] = None,
        cause: Optional[Exception] = None,
        context: Optional[Dict[str, Any]] = None,
    ):
        """
        Initialize AIM2Exception with message and optional parameters.

        Args:
            message (str): Error message describing the issue
            error_code (str, optional): Specific error code for categorization
            cause (Exception, optional): Original exception that caused this error
            context (Dict[str, Any], optional): Additional context information

        Note:
            If error_code is None or empty string, the default error code will be
            generated using the pattern "AIM2_{CLASS_NAME}".
        """
        super().__init__(message)
        self.message = message
        self.error_code = error_code or self._get_default_error_code()
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
        return {
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

        Returns:
            AIM2Exception: Reconstructed exception instance

        Note:
            If cause information is present in the data, a generic Exception
            will be created to represent the original cause. The original
            exception type information is preserved in the cause_type field
            but the actual exception object will be a generic Exception.

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

        return cls(
            message=data["message"],
            error_code=data.get("error_code"),
            cause=cause,
            context=data.get("context", {}),
        )


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
