"""
Comprehensive Unit Tests for AIM2 Exception Hierarchy

This module provides comprehensive unit tests for the AIM2 project's exception hierarchy,
testing all custom exception classes including base exception functionality, error codes,
serialization, inheritance, and exception chaining.

Test Classes:
    TestAIM2Exception: Tests for the base exception class with error code system
    TestOntologyException: Tests for ontology-related errors
    TestExtractionException: Tests for information extraction errors
    TestLLMException: Tests for LLM interface errors
    TestValidationException: Tests for validation errors
    TestExceptionHierarchy: Tests for inheritance relationships
    TestExceptionSerialization: Tests for exception serialization methods
    TestExceptionChaining: Tests for exception chaining and cause tracking
    TestExceptionEdgeCases: Tests for edge cases and error conditions

Dependencies:
    - pytest: For test framework
    - unittest.mock: For mocking functionality
    - json: For serialization testing
    - traceback: For traceback testing
    - typing: For type hints
"""


from aim2_project.exceptions import (
    AIM2Exception,
    OntologyException,
    ExtractionException,
    LLMException,
    ValidationException,
    ErrorSeverity,
    ErrorCodeInfo,
    AIM2ErrorCodes,
    ErrorCodeRegistry,
    BaseErrorCodes,
    OntologyErrorCodes,
    ExtractionErrorCodes,
    LLMErrorCodes,
    ValidationErrorCodes,
    MessageFormat,
    BaseMessageTemplate,
    StandardMessageTemplate,
    CompactMessageTemplate,
    MessageTemplateRegistry,
)


class TestAIM2Exception:
    """Test the base AIM2Exception class functionality."""

    def test_basic_exception_creation(self):
        """Test basic exception creation with message only."""
        message = "Test error message"
        exception = AIM2Exception(message)

        assert str(exception) == "AIM2_AIM2EXCEPTION: Test error message"
        assert exception.message == message
        assert exception.error_code == "AIM2_AIM2EXCEPTION"
        assert exception.cause is None
        assert exception.context == {}

    def test_exception_with_error_code(self):
        """Test exception creation with custom error code."""
        message = "Test error message"
        error_code = "CUSTOM_ERROR_001"
        exception = AIM2Exception(message, error_code=error_code)

        assert exception.message == message
        assert exception.error_code == error_code
        assert exception.cause is None
        assert exception.context == {}

    def test_exception_with_cause(self):
        """Test exception creation with cause exception."""
        message = "Test error message"
        cause = ValueError("Original error")
        exception = AIM2Exception(message, cause=cause)

        assert exception.message == message
        assert exception.cause == cause
        assert exception.__cause__ == cause
        assert exception.error_code == "AIM2_AIM2EXCEPTION"

    def test_exception_with_context(self):
        """Test exception creation with context information."""
        message = "Test error message"
        context = {"operation": "test_op", "file": "test.txt", "line": 42}
        exception = AIM2Exception(message, context=context)

        assert exception.message == message
        assert exception.context == context
        assert exception.error_code == "AIM2_AIM2EXCEPTION"

    def test_exception_with_all_parameters(self):
        """Test exception creation with all parameters."""
        message = "Test error message"
        error_code = "CUSTOM_ERROR_002"
        cause = RuntimeError("Root cause")
        context = {"module": "test_module", "function": "test_func"}

        exception = AIM2Exception(
            message=message, error_code=error_code, cause=cause, context=context
        )

        assert exception.message == message
        assert exception.error_code == error_code
        assert exception.cause == cause
        assert exception.__cause__ == cause
        assert exception.context == context

    def test_exception_inheritance(self):
        """Test that AIM2Exception inherits from Exception."""
        exception = AIM2Exception("test")
        assert isinstance(exception, Exception)
        assert isinstance(exception, AIM2Exception)

    def test_exception_string_representation(self):
        """Test string representation of exception."""
        message = "Test error message"
        exception = AIM2Exception(message)
        assert str(exception) == "AIM2_AIM2EXCEPTION: Test error message"

    def test_exception_repr(self):
        """Test repr representation of exception."""
        message = "Test error message"
        error_code = "TEST_ERROR"
        exception = AIM2Exception(message, error_code=error_code)
        repr_str = repr(exception)
        assert "AIM2Exception" in repr_str
        assert message in repr_str


class TestOntologyException:
    """Test the OntologyException class functionality."""

    def test_basic_ontology_exception_creation(self):
        """Test basic ontology exception creation."""
        message = "Ontology parsing failed"
        exception = OntologyException(message)

        assert str(exception) == "AIM2_ONTOLOGY_ERROR: Ontology parsing failed"
        assert exception.message == message
        assert exception.error_code == "AIM2_ONTOLOGY_ERROR"
        assert isinstance(exception, AIM2Exception)
        assert isinstance(exception, OntologyException)

    def test_ontology_exception_with_cause(self):
        """Test ontology exception with cause."""
        message = "Failed to load ontology file"
        cause = FileNotFoundError("ontology.owl not found")
        exception = OntologyException(message, cause=cause)

        assert exception.message == message
        assert exception.cause == cause
        assert exception.__cause__ == cause
        assert exception.error_code == "AIM2_ONTOLOGY_ERROR"

    def test_ontology_exception_with_context(self):
        """Test ontology exception with context."""
        message = "Ontology validation failed"
        context = {
            "ontology_file": "/path/to/ontology.owl",
            "validation_rule": "class_consistency",
            "line_number": 150,
        }
        exception = OntologyException(message, context=context)

        assert exception.message == message
        assert exception.context == context
        assert exception.error_code == "AIM2_ONTOLOGY_ERROR"

    def test_ontology_exception_custom_error_code(self):
        """Test ontology exception with custom error code."""
        message = "Ontology integration conflict"
        error_code = "ONTOLOGY_MERGE_CONFLICT"
        exception = OntologyException(message, error_code=error_code)

        assert exception.message == message
        assert exception.error_code == error_code


class TestExtractionException:
    """Test the ExtractionException class functionality."""

    def test_basic_extraction_exception_creation(self):
        """Test basic extraction exception creation."""
        message = "NER extraction failed"
        exception = ExtractionException(message)

        assert str(exception) == "AIM2_EXTRACTION_ERROR: NER extraction failed"
        assert exception.message == message
        assert exception.error_code == "AIM2_EXTRACTION_ERROR"
        assert isinstance(exception, AIM2Exception)
        assert isinstance(exception, ExtractionException)

    def test_extraction_exception_with_cause(self):
        """Test extraction exception with cause."""
        message = "Text processing pipeline failed"
        cause = ValueError("Invalid text encoding")
        exception = ExtractionException(message, cause=cause)

        assert exception.message == message
        assert exception.cause == cause
        assert exception.__cause__ == cause
        assert exception.error_code == "AIM2_EXTRACTION_ERROR"

    def test_extraction_exception_with_context(self):
        """Test extraction exception with context."""
        message = "Relationship extraction failed"
        context = {
            "document_id": "doc_123",
            "sentence_index": 45,
            "entity_pair": ("protein_A", "protein_B"),
            "extraction_model": "bert-base-cased",
        }
        exception = ExtractionException(message, context=context)

        assert exception.message == message
        assert exception.context == context
        assert exception.error_code == "AIM2_EXTRACTION_ERROR"


class TestLLMException:
    """Test the LLMException class functionality."""

    def test_basic_llm_exception_creation(self):
        """Test basic LLM exception creation."""
        message = "LLM API request failed"
        exception = LLMException(message)

        assert str(exception) == "AIM2_LLM_ERROR: LLM API request failed"
        assert exception.message == message
        assert exception.error_code == "AIM2_LLM_ERROR"
        assert isinstance(exception, AIM2Exception)
        assert isinstance(exception, LLMException)

    def test_llm_exception_with_cause(self):
        """Test LLM exception with cause."""
        message = "OpenAI API timeout"
        cause = TimeoutError("Request timed out after 30 seconds")
        exception = LLMException(message, cause=cause)

        assert exception.message == message
        assert exception.cause == cause
        assert exception.__cause__ == cause
        assert exception.error_code == "AIM2_LLM_ERROR"

    def test_llm_exception_with_context(self):
        """Test LLM exception with context."""
        message = "Model inference failed"
        context = {
            "model_name": "gpt-3.5-turbo",
            "prompt_length": 2048,
            "max_tokens": 1000,
            "temperature": 0.7,
            "request_id": "req_abc123",
        }
        exception = LLMException(message, context=context)

        assert exception.message == message
        assert exception.context == context
        assert exception.error_code == "AIM2_LLM_ERROR"


class TestValidationException:
    """Test the ValidationException class functionality."""

    def test_basic_validation_exception_creation(self):
        """Test basic validation exception creation."""
        message = "Schema validation failed"
        exception = ValidationException(message)

        assert str(exception) == "AIM2_VALIDATION_ERROR: Schema validation failed"
        assert exception.message == message
        assert exception.error_code == "AIM2_VALIDATION_ERROR"
        assert isinstance(exception, AIM2Exception)
        assert isinstance(exception, ValidationException)

    def test_validation_exception_with_cause(self):
        """Test validation exception with cause."""
        message = "JSON schema validation failed"
        cause = ValueError("Required field 'name' is missing")
        exception = ValidationException(message, cause=cause)

        assert exception.message == message
        assert exception.cause == cause
        assert exception.__cause__ == cause
        assert exception.error_code == "AIM2_VALIDATION_ERROR"

    def test_validation_exception_with_context(self):
        """Test validation exception with context."""
        message = "Configuration validation failed"
        context = {
            "config_file": "/path/to/config.yaml",
            "schema_version": "1.0",
            "validation_errors": [
                "Invalid port number: -1",
                "Missing required field: database.host",
            ],
        }
        exception = ValidationException(message, context=context)

        assert exception.message == message
        assert exception.context == context
        assert exception.error_code == "AIM2_VALIDATION_ERROR"


class TestExceptionHierarchy:
    """Test the exception inheritance hierarchy."""

    def test_exception_inheritance_chain(self):
        """Test that all custom exceptions inherit from AIM2Exception."""
        ontology_exc = OntologyException("test")
        extraction_exc = ExtractionException("test")
        llm_exc = LLMException("test")
        validation_exc = ValidationException("test")

        # Test inheritance from AIM2Exception
        assert isinstance(ontology_exc, AIM2Exception)
        assert isinstance(extraction_exc, AIM2Exception)
        assert isinstance(llm_exc, AIM2Exception)
        assert isinstance(validation_exc, AIM2Exception)

        # Test inheritance from Exception
        assert isinstance(ontology_exc, Exception)
        assert isinstance(extraction_exc, Exception)
        assert isinstance(llm_exc, Exception)
        assert isinstance(validation_exc, Exception)

    def test_exception_type_checking(self):
        """Test type checking for specific exception types."""
        ontology_exc = OntologyException("test")
        extraction_exc = ExtractionException("test")

        # Positive type checks
        assert isinstance(ontology_exc, OntologyException)
        assert isinstance(extraction_exc, ExtractionException)

        # Negative type checks
        assert not isinstance(ontology_exc, ExtractionException)
        assert not isinstance(extraction_exc, OntologyException)

    def test_exception_mro(self):
        """Test method resolution order (MRO) for exceptions."""
        ontology_exc = OntologyException("test")
        mro = type(ontology_exc).__mro__

        assert OntologyException in mro
        assert AIM2Exception in mro
        assert Exception in mro
        assert object in mro

    def test_exception_class_names(self):
        """Test exception class names are correct."""
        assert AIM2Exception.__name__ == "AIM2Exception"
        assert OntologyException.__name__ == "OntologyException"
        assert ExtractionException.__name__ == "ExtractionException"
        assert LLMException.__name__ == "LLMException"
        assert ValidationException.__name__ == "ValidationException"


class TestExceptionSerialization:
    """Test exception serialization and deserialization."""

    def test_basic_exception_to_dict(self):
        """Test basic exception serialization to dictionary."""
        message = "Test error"
        error_code = "TEST_001"
        context = {"key": "value"}

        exception = AIM2Exception(message, error_code=error_code, context=context)
        result = exception.to_dict()

        assert result["exception_type"] == "AIM2Exception"
        assert result["message"] == message
        assert result["error_code"] == error_code
        assert result["context"] == context
        assert result["cause_type"] is None
        assert result["cause_message"] is None

    def test_exception_with_cause_to_dict(self):
        """Test exception with cause serialization."""
        message = "Wrapper error"
        cause = ValueError("Original error")

        exception = AIM2Exception(message, cause=cause)
        result = exception.to_dict()

        assert result["exception_type"] == "AIM2Exception"
        assert result["message"] == message
        assert result["cause_type"] == "ValueError"
        assert result["cause_message"] == "Original error"

    def test_exception_from_dict(self):
        """Test exception deserialization from dictionary."""
        data = {
            "exception_type": "AIM2Exception",
            "message": "Test error",
            "error_code": "TEST_002",
            "context": {"operation": "test"},
            "cause_type": "RuntimeError",
            "cause_message": "Runtime issue",
        }

        exception = AIM2Exception.from_dict(data)

        assert exception.message == "Test error"
        assert exception.error_code == "TEST_002"
        assert exception.context == {"operation": "test"}
        assert exception.cause is not None
        assert str(exception.cause) == "Runtime issue"

    def test_round_trip_serialization(self):
        """Test round-trip serialization (to_dict then from_dict)."""
        original = AIM2Exception(
            message="Round trip test",
            error_code="ROUND_TRIP_001",
            context={"test": True, "number": 42},
        )

        # Serialize to dict
        data = original.to_dict()

        # Deserialize from dict
        restored = AIM2Exception.from_dict(data)

        assert restored.message == original.message
        assert restored.error_code == original.error_code
        assert restored.context == original.context

    def test_specialized_exception_serialization(self):
        """Test serialization of specialized exception classes."""
        ontology_exc = OntologyException("Ontology error")
        extraction_exc = ExtractionException("Extraction error")

        ontology_dict = ontology_exc.to_dict()
        extraction_dict = extraction_exc.to_dict()

        assert ontology_dict["exception_type"] == "OntologyException"
        assert ontology_dict["error_code"] == "AIM2_ONTOLOGY_ERROR"

        assert extraction_dict["exception_type"] == "ExtractionException"
        assert extraction_dict["error_code"] == "AIM2_EXTRACTION_ERROR"


class TestExceptionChaining:
    """Test exception chaining and cause tracking."""

    def test_simple_exception_chaining(self):
        """Test simple exception chaining with cause."""
        root_cause = ValueError("Root cause error")
        wrapper = AIM2Exception("Wrapper error", cause=root_cause)

        assert wrapper.cause == root_cause
        assert wrapper.__cause__ == root_cause

    def test_nested_exception_chaining(self):
        """Test nested exception chaining."""
        root_error = IOError("File not found")
        middle_error = OntologyException("Failed to load ontology", cause=root_error)
        top_error = ExtractionException(
            "Extraction pipeline failed", cause=middle_error
        )

        assert top_error.cause == middle_error
        assert top_error.__cause__ == middle_error
        assert middle_error.cause == root_error
        assert middle_error.__cause__ == root_error

    def test_cause_preservation_in_serialization(self):
        """Test that cause information is preserved during serialization."""
        root_cause = RuntimeError("Database connection failed")
        wrapper = ValidationException("Config validation failed", cause=root_cause)

        serialized = wrapper.to_dict()

        assert serialized["cause_type"] == "RuntimeError"
        assert serialized["cause_message"] == "Database connection failed"

    def test_cause_restoration_from_serialization(self):
        """Test that cause can be restored from serialization data."""
        data = {
            "exception_type": "LLMException",
            "message": "LLM request failed",
            "error_code": "LLM_TIMEOUT",
            "context": {},
            "cause_type": "TimeoutError",
            "cause_message": "Request timeout after 30s",
        }

        exception = LLMException.from_dict(data)

        assert exception.cause is not None
        assert str(exception.cause) == "Request timeout after 30s"


class TestExceptionEdgeCases:
    """Test edge cases and error conditions for exceptions."""

    def test_empty_message(self):
        """Test exception with empty message."""
        exception = AIM2Exception("")
        assert exception.message == ""
        assert str(exception) == "AIM2_AIM2EXCEPTION: "

    def test_none_message_handling(self):
        """Test handling of None message."""
        # Python allows None as message, it gets converted to "None"
        exception = AIM2Exception(None)
        assert exception.message is None
        assert str(exception) == "AIM2_AIM2EXCEPTION: None"

    def test_non_string_message(self):
        """Test handling of non-string message."""
        # This should work due to Python's automatic string conversion
        exception = AIM2Exception(123)
        assert str(exception) == "AIM2_AIM2EXCEPTION: 123"

    def test_none_context(self):
        """Test handling of None context."""
        exception = AIM2Exception("test", context=None)
        assert exception.context == {}

    def test_invalid_context_type(self):
        """Test handling of invalid context type."""
        # Should accept any object that can be assigned
        exception = AIM2Exception("test", context="invalid")
        assert exception.context == "invalid"

    def test_none_error_code(self):
        """Test handling of None error code."""
        exception = AIM2Exception("test", error_code=None)
        assert exception.error_code == "AIM2_AIM2EXCEPTION"

    def test_empty_error_code(self):
        """Test handling of empty error code."""
        exception = AIM2Exception("test", error_code="")
        assert exception.error_code == "AIM2_AIM2EXCEPTION"

    def test_very_long_message(self):
        """Test handling of very long error messages."""
        long_message = "x" * 10000
        exception = AIM2Exception(long_message)
        assert exception.message == long_message
        # String representation includes error code prefix: "AIM2_AIM2EXCEPTION: " + message
        expected_length = len("AIM2_AIM2EXCEPTION: ") + 10000
        assert len(str(exception)) == expected_length

    def test_unicode_message(self):
        """Test handling of Unicode characters in messages."""
        unicode_message = "Error: 测试 🚫 áéíóú"
        exception = AIM2Exception(unicode_message)
        assert exception.message == unicode_message
        assert str(exception) == f"AIM2_AIM2EXCEPTION: {unicode_message}"

    def test_complex_context_data(self):
        """Test handling of complex context data structures."""
        complex_context = {
            "nested": {"level1": {"level2": "deep_value"}},
            "list": [1, 2, {"inner": "value"}],
            "unicode": "测试数据",
            "numbers": [1, 2.5, -3],
            "boolean": True,
            "null": None,
        }

        exception = AIM2Exception("test", context=complex_context)
        assert exception.context == complex_context

    def test_serialization_with_missing_fields(self):
        """Test deserialization with missing fields."""
        minimal_data = {"message": "Minimal error", "exception_type": "AIM2Exception"}

        exception = AIM2Exception.from_dict(minimal_data)
        assert exception.message == "Minimal error"
        assert exception.error_code is not None  # Should use default
        assert exception.context == {}
        assert exception.cause is None

    def test_serialization_with_extra_fields(self):
        """Test deserialization with extra fields (should be ignored)."""
        data_with_extras = {
            "message": "Test error",
            "exception_type": "AIM2Exception",
            "error_code": "TEST_001",
            "context": {"key": "value"},
            "extra_field": "should_be_ignored",
            "another_extra": 123,
        }

        exception = AIM2Exception.from_dict(data_with_extras)
        assert exception.message == "Test error"
        assert exception.error_code == "TEST_001"
        assert exception.context == {"key": "value"}
        # Extra fields should not cause errors

    def test_multiple_exception_instances(self):
        """Test creating multiple exception instances with different parameters."""
        exceptions = []

        for i in range(100):
            exc = AIM2Exception(
                message=f"Error {i}",
                error_code=f"ERR_{i:03d}",
                context={"iteration": i},
            )
            exceptions.append(exc)

        # Verify all exceptions are independent
        for i, exc in enumerate(exceptions):
            assert exc.message == f"Error {i}"
            assert exc.error_code == f"ERR_{i:03d}"
            assert exc.context["iteration"] == i

    def test_exception_equality(self):
        """Test exception equality (exceptions are not equal even with same params)."""
        exc1 = AIM2Exception("same message", error_code="SAME_CODE")
        exc2 = AIM2Exception("same message", error_code="SAME_CODE")

        # Exception instances are not equal even with same parameters
        assert exc1 is not exc2
        assert exc1 != exc2

    def test_exception_hashing(self):
        """Test that exceptions can be used as dictionary keys."""
        exc1 = AIM2Exception("test1")
        exc2 = AIM2Exception("test2")

        exception_dict = {exc1: "value1", exc2: "value2"}

        assert exception_dict[exc1] == "value1"
        assert exception_dict[exc2] == "value2"
        assert len(exception_dict) == 2

    def test_exception_with_enum_error_code(self):
        """Test exception creation with enum error code."""
        message = "System initialization failed"
        error_code = AIM2ErrorCodes.BASE_SYS_C_001
        exception = AIM2Exception(message, error_code=error_code)

        assert exception.message == message
        assert exception.error_code == "AIM2_BASE_SYS_C_001"
        assert exception.error_info is not None
        assert isinstance(exception.error_info, ErrorCodeInfo)
        assert exception.error_info.code == "AIM2_BASE_SYS_C_001"
        assert exception.error_info.module == "BASE"
        assert exception.error_info.severity == ErrorSeverity.CRITICAL

    def test_exception_error_info_population(self):
        """Test that error_info is properly populated for enum error codes."""
        error_code = AIM2ErrorCodes.ONTO_PARS_E_001
        exception = AIM2Exception("Test", error_code=error_code)

        # Test error_info attributes
        assert exception.error_info.code == "AIM2_ONTO_PARS_E_001"
        assert exception.error_info.module == "ONTO"
        assert exception.error_info.category == "PARS"
        assert exception.error_info.severity == ErrorSeverity.ERROR
        assert exception.error_info.description == "OWL file parsing failure"
        assert (
            exception.error_info.resolution_hint == "Validate OWL syntax and structure"
        )

    def test_exception_error_info_string_codes(self):
        """Test error_info population for string-based error codes."""
        # Test with valid string code that exists in registry
        exception1 = AIM2Exception("Test", error_code="AIM2_BASE_SYS_C_001")
        assert exception1.error_info is not None
        assert exception1.error_info.code == "AIM2_BASE_SYS_C_001"

        # Test with invalid string code
        exception2 = AIM2Exception("Test", error_code="INVALID_CODE")
        assert exception2.error_info is None
        assert exception2.error_code == "INVALID_CODE"

    def test_exception_backward_compatibility(self):
        """Test backward compatibility with string-based error codes."""
        # Old way with string codes should still work
        exception = AIM2Exception("Test", error_code="CUSTOM_ERROR_001")
        assert exception.error_code == "CUSTOM_ERROR_001"
        assert exception.message == "Test"

        # Should not have error_info for unknown codes
        assert exception.error_info is None

    def test_exception_get_detailed_message_with_error_info(self):
        """Test get_detailed_message with error_info."""
        error_code = AIM2ErrorCodes.LLM_API_E_001
        exception = AIM2Exception("API request failed", error_code=error_code)

        detailed_message = exception.get_detailed_message()

        # Check that detailed message contains expected elements
        assert "[ERROR]" in detailed_message
        assert "AIM2_LLM_API_E_001" in detailed_message
        assert "API request failed" in detailed_message
        assert "Description:" in detailed_message
        assert "Resolution:" in detailed_message
        assert error_code.value.description in detailed_message
        assert error_code.value.resolution_hint in detailed_message

    def test_exception_get_detailed_message_without_error_info(self):
        """Test get_detailed_message without error_info."""
        exception = AIM2Exception("Test", error_code="CUSTOM_CODE")
        detailed_message = exception.get_detailed_message()

        # Should use template format with warning symbol for unknown severity
        assert "⚠ [ERROR]" in detailed_message or "[ERROR]" in detailed_message
        assert "CUSTOM_CODE" in detailed_message
        assert "Test" in detailed_message

    def test_exception_str_representation_with_error_info(self):
        """Test string representation with error_info."""
        error_code = AIM2ErrorCodes.VALID_SCHM_E_001
        exception = AIM2Exception("Schema validation failed", error_code=error_code)

        str_repr = str(exception)

        # Should use detailed message format
        assert "[ERROR]" in str_repr
        assert "AIM2_VALID_SCHM_E_001" in str_repr
        assert "Schema validation failed" in str_repr
        assert "Description:" in str_repr

    def test_exception_str_representation_without_error_info(self):
        """Test string representation without error_info."""
        exception = AIM2Exception("Test", error_code="CUSTOM_CODE")
        str_repr = str(exception)

        # Should use simple format when no error_info is available (backward compatibility)
        assert str_repr == "CUSTOM_CODE: Test"

    def test_exception_serialization_with_error_info(self):
        """Test serialization includes error_info metadata."""
        error_code = AIM2ErrorCodes.EXTR_NER_E_001
        exception = AIM2Exception("NER model loading failed", error_code=error_code)

        serialized = exception.to_dict()

        # Test that error_info is included
        assert "error_info" in serialized
        error_info = serialized["error_info"]

        assert error_info["module"] == "EXTR"
        assert error_info["category"] == "NER"
        assert error_info["severity"] == "ERROR"
        assert error_info["description"] == "NER model loading failure"
        assert error_info["resolution_hint"] == "Verify model file path and format"

    def test_exception_serialization_without_error_info(self):
        """Test serialization without error_info."""
        exception = AIM2Exception("Test", error_code="CUSTOM_CODE")
        serialized = exception.to_dict()

        # Should not include error_info for unknown codes
        assert "error_info" not in serialized or serialized.get("error_info") is None

    def test_exception_deserialization_compatibility(self):
        """Test deserialization maintains compatibility."""
        # Test deserializing data without error_info (old format)
        old_format_data = {
            "exception_type": "AIM2Exception",
            "message": "Test error",
            "error_code": "TEST_001",
            "context": {"key": "value"},
            "cause_type": None,
            "cause_message": None,
        }

        exception = AIM2Exception.from_dict(old_format_data)
        assert exception.message == "Test error"
        assert exception.error_code == "TEST_001"
        assert exception.context == {"key": "value"}

    def test_exception_mixed_usage_patterns(self):
        """Test mixed usage of enum and string error codes."""
        # Create exceptions with different error code types
        exc1 = AIM2Exception("Test 1", error_code=AIM2ErrorCodes.BASE_SYS_C_001)
        exc2 = AIM2Exception("Test 2", error_code="CUSTOM_ERROR_001")
        exc3 = AIM2Exception("Test 3")  # Default error code

        # All should work correctly
        assert exc1.error_code == "AIM2_BASE_SYS_C_001"
        assert exc1.error_info is not None

        assert exc2.error_code == "CUSTOM_ERROR_001"
        assert exc2.error_info is None

        assert exc3.error_code == "AIM2_AIM2EXCEPTION"
        # error_info might be None if default code is not in registry

    def test_exception_convenience_class_integration(self):
        """Test integration with convenience classes."""
        # Test using convenience classes directly
        exc1 = AIM2Exception(
            "System error", error_code=BaseErrorCodes.SYSTEM_INIT_FAILURE
        )
        assert exc1.error_code == "AIM2_BASE_SYS_C_001"
        assert exc1.error_info.severity == ErrorSeverity.CRITICAL

        exc2 = AIM2Exception(
            "Ontology error", error_code=OntologyErrorCodes.OWL_PARSING_FAILURE
        )
        assert exc2.error_code == "AIM2_ONTO_PARS_E_001"
        assert exc2.error_info.module == "ONTO"

    def test_exception_error_code_validation_integration(self):
        """Test integration with error code validation."""
        registry = ErrorCodeRegistry()

        # Test with valid enum error code
        exc1 = AIM2Exception("Test", error_code=AIM2ErrorCodes.LLM_API_E_001)
        assert registry.validate_error_code(exc1.error_code)

        # Test with invalid string error code
        exc2 = AIM2Exception("Test", error_code="INVALID_CODE")
        assert not registry.validate_error_code(exc2.error_code)

    def test_exception_error_info_immutability(self):
        """Test that error_info is properly immutable (NamedTuple)."""
        exc = AIM2Exception("Test", error_code=AIM2ErrorCodes.BASE_SYS_C_001)
        error_info = exc.error_info

        # Should be able to access all fields
        assert error_info.code == "AIM2_BASE_SYS_C_001"
        assert error_info.module == "BASE"
        assert error_info.category == "SYS"
        assert error_info.severity == ErrorSeverity.CRITICAL
        assert error_info.description == "System initialization failure"
        assert error_info.resolution_hint == "Check dependencies and permissions"

        # Should be immutable (NamedTuple)
        try:
            error_info.code = "MODIFIED"
            assert False, "Should not be able to modify NamedTuple"
        except AttributeError:
            pass  # Expected behavior


class TestErrorSeverity:
    """Test the ErrorSeverity enum functionality."""

    def test_all_severity_levels_exist(self):
        """Test that all expected severity levels are defined."""
        expected_severities = ["CRITICAL", "ERROR", "WARNING", "INFO"]

        for severity_name in expected_severities:
            assert hasattr(ErrorSeverity, severity_name)
            severity = getattr(ErrorSeverity, severity_name)
            assert isinstance(severity, ErrorSeverity)

    def test_severity_values_correct(self):
        """Test that severity levels have correct string values."""
        assert ErrorSeverity.CRITICAL.value == "C"
        assert ErrorSeverity.ERROR.value == "E"
        assert ErrorSeverity.WARNING.value == "W"
        assert ErrorSeverity.INFO.value == "I"

    def test_severity_names_correct(self):
        """Test that severity levels have correct names."""
        assert ErrorSeverity.CRITICAL.name == "CRITICAL"
        assert ErrorSeverity.ERROR.name == "ERROR"
        assert ErrorSeverity.WARNING.name == "WARNING"
        assert ErrorSeverity.INFO.name == "INFO"

    def test_severity_enum_membership(self):
        """Test membership and iteration over severity enum."""
        all_severities = list(ErrorSeverity)
        assert len(all_severities) == 4

        expected_severities = [
            ErrorSeverity.CRITICAL,
            ErrorSeverity.ERROR,
            ErrorSeverity.WARNING,
            ErrorSeverity.INFO,
        ]

        for severity in expected_severities:
            assert severity in all_severities

    def test_severity_comparison_and_ordering(self):
        """Test that severity levels can be compared and ordered."""
        # Test equality
        assert ErrorSeverity.CRITICAL == ErrorSeverity.CRITICAL
        assert ErrorSeverity.ERROR == ErrorSeverity.ERROR

        # Test inequality
        assert ErrorSeverity.CRITICAL != ErrorSeverity.ERROR
        assert ErrorSeverity.WARNING != ErrorSeverity.INFO

    def test_severity_string_representation(self):
        """Test string representation of severity levels."""
        assert str(ErrorSeverity.CRITICAL) == "ErrorSeverity.CRITICAL"
        assert str(ErrorSeverity.ERROR) == "ErrorSeverity.ERROR"
        assert str(ErrorSeverity.WARNING) == "ErrorSeverity.WARNING"
        assert str(ErrorSeverity.INFO) == "ErrorSeverity.INFO"

    def test_severity_repr_representation(self):
        """Test repr representation of severity levels."""
        assert "ErrorSeverity.CRITICAL" in repr(ErrorSeverity.CRITICAL)
        assert "ErrorSeverity.ERROR" in repr(ErrorSeverity.ERROR)

    def test_severity_value_access(self):
        """Test accessing severity values directly."""
        critical = ErrorSeverity.CRITICAL
        assert critical.value == "C"
        assert critical.name == "CRITICAL"


class TestAIM2ErrorCodes:
    """Test the AIM2ErrorCodes enum functionality."""

    def test_all_error_codes_defined(self):
        """Test that all 50 error codes are defined."""
        all_codes = list(AIM2ErrorCodes)
        assert len(all_codes) == 50, f"Expected 50 error codes, found {len(all_codes)}"

    def test_error_code_format_validation(self):
        """Test that all error codes follow the expected format."""
        expected_pattern = r"^AIM2_[A-Z]+_[A-Z]+_[CEWI]_\d{3}$"
        import re

        for error_code in AIM2ErrorCodes:
            code_str = error_code.value.code
            assert re.match(
                expected_pattern, code_str
            ), f"Error code {code_str} doesn't match expected pattern"

    def test_error_code_metadata_structure(self):
        """Test that each error code has proper metadata structure."""
        for error_code in AIM2ErrorCodes:
            error_info = error_code.value

            # Test that error_info is an ErrorCodeInfo instance
            assert isinstance(error_info, ErrorCodeInfo)

            # Test required fields are present and non-empty
            assert error_info.code, f"Error code {error_code.name} has empty code"
            assert error_info.module, f"Error code {error_code.name} has empty module"
            assert (
                error_info.category
            ), f"Error code {error_code.name} has empty category"
            assert isinstance(
                error_info.severity, ErrorSeverity
            ), f"Error code {error_code.name} has invalid severity type"
            assert (
                error_info.description
            ), f"Error code {error_code.name} has empty description"
            assert (
                error_info.resolution_hint
            ), f"Error code {error_code.name} has empty resolution_hint"

    def test_error_code_severity_consistency(self):
        """Test that error code severity matches the code format."""
        for error_code in AIM2ErrorCodes:
            error_info = error_code.value
            code_str = error_info.code

            # Extract severity from code (last letter before the number)
            severity_char = code_str.split("_")[-2]
            expected_severity_value = error_info.severity.value

            assert (
                severity_char == expected_severity_value
            ), f"Error code {code_str} has severity mismatch: {severity_char} vs {expected_severity_value}"

    def test_module_distribution(self):
        """Test error codes are distributed across expected modules."""
        expected_modules = ["BASE", "ONTO", "EXTR", "LLM", "VALID"]
        actual_modules = set()
        module_counts = {}

        for error_code in AIM2ErrorCodes:
            module = error_code.value.module
            actual_modules.add(module)
            module_counts[module] = module_counts.get(module, 0) + 1

        # Test all expected modules are present
        for expected_module in expected_modules:
            assert (
                expected_module in actual_modules
            ), f"Module {expected_module} not found in error codes"

        # Test each module has reasonable number of error codes
        for module, count in module_counts.items():
            assert count > 0, f"Module {module} has no error codes"
            assert count <= 15, f"Module {module} has too many error codes ({count})"

    def test_category_distribution(self):
        """Test error codes have appropriate category distribution."""
        categories = set()
        category_counts = {}

        for error_code in AIM2ErrorCodes:
            category = error_code.value.category
            categories.add(category)
            category_counts[category] = category_counts.get(category, 0) + 1

        # Test we have multiple categories
        assert (
            len(categories) >= 5
        ), f"Expected at least 5 categories, found {len(categories)}"

        # Test each category has reasonable number of error codes
        for category, count in category_counts.items():
            assert count > 0, f"Category {category} has no error codes"
            assert (
                count <= 20
            ), f"Category {category} has too many error codes ({count})"

    def test_severity_distribution(self):
        """Test error codes have appropriate severity distribution."""
        severity_counts = {}

        for error_code in AIM2ErrorCodes:
            severity = error_code.value.severity
            severity_counts[severity] = severity_counts.get(severity, 0) + 1

        # Test all severity levels are used
        for severity in ErrorSeverity:
            assert (
                severity in severity_counts
            ), f"Severity {severity.name} not used in any error code"

        # Test reasonable distribution (ERROR should be most common)
        assert severity_counts[ErrorSeverity.ERROR] > 0, "No ERROR severity codes found"

        # Test CRITICAL is least common (should be used sparingly)
        if ErrorSeverity.CRITICAL in severity_counts:
            critical_count = severity_counts[ErrorSeverity.CRITICAL]
            total_count = sum(severity_counts.values())
            critical_percentage = critical_count / total_count
            assert (
                critical_percentage <= 0.2
            ), f"Too many CRITICAL errors ({critical_percentage:.1%})"

    def test_specific_error_codes_exist(self):
        """Test that specific important error codes exist."""
        # Test some specific critical error codes
        critical_codes = [
            "AIM2_BASE_SYS_C_001",  # System initialization failure
            "AIM2_BASE_SYS_C_002",  # Memory allocation failure
            "AIM2_BASE_PER_C_001",  # Database connection failure
        ]

        all_code_strings = [error_code.value.code for error_code in AIM2ErrorCodes]

        for critical_code in critical_codes:
            assert (
                critical_code in all_code_strings
            ), f"Critical error code {critical_code} not found"

    def test_error_code_uniqueness(self):
        """Test that all error codes are unique."""
        all_codes = [error_code.value.code for error_code in AIM2ErrorCodes]
        unique_codes = set(all_codes)

        assert len(all_codes) == len(unique_codes), "Duplicate error codes found"

    def test_error_code_descriptions_unique(self):
        """Test that error code descriptions are unique."""
        all_descriptions = [
            error_code.value.description for error_code in AIM2ErrorCodes
        ]
        unique_descriptions = set(all_descriptions)

        # Allow for some similarity but not exact duplicates
        assert (
            len(unique_descriptions) >= len(all_descriptions) * 0.9
        ), "Too many duplicate descriptions found"

    def test_error_code_enum_access(self):
        """Test accessing error codes by enum name."""
        # Test a few specific error codes can be accessed
        assert hasattr(AIM2ErrorCodes, "BASE_SYS_C_001")
        assert hasattr(AIM2ErrorCodes, "ONTO_PARS_E_001")
        assert hasattr(AIM2ErrorCodes, "EXTR_NER_E_001")
        assert hasattr(AIM2ErrorCodes, "LLM_API_E_001")
        assert hasattr(AIM2ErrorCodes, "VALID_SCHM_E_001")

        # Test accessing the values
        base_error = AIM2ErrorCodes.BASE_SYS_C_001
        assert isinstance(base_error.value, ErrorCodeInfo)
        assert base_error.value.code == "AIM2_BASE_SYS_C_001"

    def test_error_code_resolution_hints_quality(self):
        """Test that resolution hints are meaningful."""
        for error_code in AIM2ErrorCodes:
            resolution_hint = error_code.value.resolution_hint

            # Test resolution hints are not just empty or too short
            assert (
                len(resolution_hint) >= 10
            ), f"Resolution hint for {error_code.value.code} is too short: '{resolution_hint}'"

            # Test resolution hints contain action words
            action_words = [
                "check",
                "verify",
                "update",
                "fix",
                "correct",
                "add",
                "remove",
                "increase",
                "reduce",
                "retry",
                "contact",
                "review",
                "restore",
                "set",
                "free",
                "validate",
                "use",
                "resolve",
                "link",
                "normal",
                "consider",
                "adjust",
            ]
            has_action_word = any(
                word in resolution_hint.lower() for word in action_words
            )
            assert (
                has_action_word
            ), f"Resolution hint for {error_code.value.code} lacks action words: '{resolution_hint}'"


class TestErrorCodeRegistry:
    """Test the ErrorCodeRegistry functionality."""

    def test_singleton_pattern(self):
        """Test that ErrorCodeRegistry implements singleton pattern."""
        registry1 = ErrorCodeRegistry()
        registry2 = ErrorCodeRegistry()

        # Should be the same instance
        assert registry1 is registry2
        assert id(registry1) == id(registry2)

    def test_registry_initialization(self):
        """Test that registry is properly initialized with all error codes."""
        registry = ErrorCodeRegistry()

        # Test that internal codes dictionary is populated
        assert hasattr(registry, "_codes")
        assert len(registry._codes) == 50  # All 50 error codes should be registered

    def test_get_error_info_valid_codes(self):
        """Test getting error info for valid error codes."""
        registry = ErrorCodeRegistry()

        # Test getting info for specific codes
        test_codes = [
            "AIM2_BASE_SYS_C_001",
            "AIM2_ONTO_PARS_E_001",
            "AIM2_EXTR_NER_E_001",
            "AIM2_LLM_API_E_001",
            "AIM2_VALID_SCHM_E_001",
        ]

        for code in test_codes:
            error_info = registry.get_error_info(code)
            assert error_info is not None, f"Error info for {code} should not be None"
            assert isinstance(error_info, ErrorCodeInfo)
            assert error_info.code == code

    def test_get_error_info_invalid_codes(self):
        """Test getting error info for invalid error codes."""
        registry = ErrorCodeRegistry()

        # Test with invalid codes
        invalid_codes = ["INVALID_CODE", "AIM2_NONEXISTENT_E_001", "", None]

        for code in invalid_codes:
            error_info = registry.get_error_info(code)
            assert (
                error_info is None
            ), f"Error info for invalid code {code} should be None"

    def test_get_codes_by_module(self):
        """Test getting error codes by module."""
        registry = ErrorCodeRegistry()

        # Test each expected module
        modules = ["BASE", "ONTO", "EXTR", "LLM", "VALID"]

        for module in modules:
            codes = registry.get_codes_by_module(module)
            assert isinstance(codes, list)
            assert len(codes) > 0, f"Module {module} should have error codes"

            # Test that all returned codes belong to the requested module
            for code in codes:
                assert isinstance(code, ErrorCodeInfo)
                assert code.module == module

    def test_get_codes_by_module_invalid(self):
        """Test getting error codes by invalid module."""
        registry = ErrorCodeRegistry()

        # Test with invalid module
        codes = registry.get_codes_by_module("NONEXISTENT")
        assert isinstance(codes, list)
        assert len(codes) == 0

    def test_get_codes_by_category(self):
        """Test getting error codes by category."""
        registry = ErrorCodeRegistry()

        # Get all categories first
        all_categories = registry.get_categories()
        assert len(all_categories) > 0

        for category in all_categories:
            codes = registry.get_codes_by_category(category)
            assert isinstance(codes, list)
            assert len(codes) > 0, f"Category {category} should have error codes"

            # Test that all returned codes belong to the requested category
            for code in codes:
                assert isinstance(code, ErrorCodeInfo)
                assert code.category == category

    def test_get_codes_by_severity(self):
        """Test getting error codes by severity."""
        registry = ErrorCodeRegistry()

        # Test each severity level
        for severity in ErrorSeverity:
            codes = registry.get_codes_by_severity(severity)
            assert isinstance(codes, list)

            # Test that all returned codes have the requested severity
            for code in codes:
                assert isinstance(code, ErrorCodeInfo)
                assert code.severity == severity

    def test_get_all_codes(self):
        """Test getting all registered error codes."""
        registry = ErrorCodeRegistry()
        all_codes = registry.get_all_codes()

        assert isinstance(all_codes, list)
        assert len(all_codes) == 50

        # Test that all are ErrorCodeInfo instances
        for code in all_codes:
            assert isinstance(code, ErrorCodeInfo)

    def test_validate_error_code(self):
        """Test error code validation."""
        registry = ErrorCodeRegistry()

        # Test valid codes
        valid_codes = [
            "AIM2_BASE_SYS_C_001",
            "AIM2_ONTO_PARS_E_001",
            "AIM2_EXTR_NER_E_001",
        ]

        for code in valid_codes:
            assert registry.validate_error_code(code), f"Code {code} should be valid"

        # Test invalid codes
        invalid_codes = [
            "INVALID_CODE",
            "AIM2_NONEXISTENT_E_001",
            "",
            "AIM2_BASE_SYS_C_999",
        ]

        for code in invalid_codes:
            assert not registry.validate_error_code(
                code
            ), f"Code {code} should be invalid"

    def test_get_modules(self):
        """Test getting list of all modules."""
        registry = ErrorCodeRegistry()
        modules = registry.get_modules()

        assert isinstance(modules, list)
        expected_modules = ["BASE", "ONTO", "EXTR", "LLM", "VALID"]

        for expected_module in expected_modules:
            assert (
                expected_module in modules
            ), f"Module {expected_module} should be in the list"

        # Test that all modules are strings
        for module in modules:
            assert isinstance(module, str)
            assert len(module) > 0

    def test_get_categories(self):
        """Test getting list of all categories."""
        registry = ErrorCodeRegistry()
        categories = registry.get_categories()

        assert isinstance(categories, list)
        assert len(categories) >= 5  # Should have multiple categories

        # Test that all categories are strings
        for category in categories:
            assert isinstance(category, str)
            assert len(category) > 0

    def test_get_statistics(self):
        """Test getting registry statistics."""
        registry = ErrorCodeRegistry()
        stats = registry.get_statistics()

        assert isinstance(stats, dict)

        # Test required keys are present
        required_keys = [
            "total_codes",
            "modules",
            "categories",
            "severity_distribution",
            "module_distribution",
        ]
        for key in required_keys:
            assert key in stats, f"Statistics should contain {key}"

        # Test total_codes
        assert stats["total_codes"] == 50

        # Test modules list
        assert isinstance(stats["modules"], list)
        assert len(stats["modules"]) == 5  # BASE, ONTO, EXTR, LLM, VALID

        # Test categories list
        assert isinstance(stats["categories"], list)
        assert len(stats["categories"]) >= 5

        # Test severity_distribution
        severity_dist = stats["severity_distribution"]
        assert isinstance(severity_dist, dict)
        for severity in ErrorSeverity:
            assert severity.name in severity_dist
            assert isinstance(severity_dist[severity.name], int)
            assert severity_dist[severity.name] >= 0

        # Test module_distribution
        module_dist = stats["module_distribution"]
        assert isinstance(module_dist, dict)
        for module in stats["modules"]:
            assert module in module_dist
            assert isinstance(module_dist[module], int)
            assert module_dist[module] > 0

        # Test that distribution totals match
        total_from_severity = sum(severity_dist.values())
        total_from_modules = sum(module_dist.values())
        assert total_from_severity == stats["total_codes"]
        assert total_from_modules == stats["total_codes"]

    def test_registry_consistency(self):
        """Test internal consistency of the registry."""
        registry = ErrorCodeRegistry()

        # Test that the number of codes returned by different methods is consistent
        all_codes = registry.get_all_codes()
        total_codes = len(all_codes)

        # Test module totals
        modules = registry.get_modules()
        module_total = sum(
            len(registry.get_codes_by_module(module)) for module in modules
        )
        assert module_total == total_codes

        # Test category totals
        categories = registry.get_categories()
        category_total = sum(
            len(registry.get_codes_by_category(category)) for category in categories
        )
        assert category_total == total_codes

        # Test severity totals
        severity_total = sum(
            len(registry.get_codes_by_severity(severity)) for severity in ErrorSeverity
        )
        assert severity_total == total_codes

    def test_registry_thread_safety_simulation(self):
        """Test that multiple registry instances work correctly (simulating thread safety)."""
        registries = [ErrorCodeRegistry() for _ in range(10)]

        # All should be the same instance (singleton)
        for i in range(1, len(registries)):
            assert registries[i] is registries[0]

        # All should return the same data
        base_stats = registries[0].get_statistics()
        for registry in registries[1:]:
            stats = registry.get_statistics()
            assert stats == base_stats


class TestConvenienceClasses:
    """Test the module-specific convenience classes."""

    def test_base_error_codes_attributes(self):
        """Test BaseErrorCodes convenience class attributes."""
        # Test system errors
        assert hasattr(BaseErrorCodes, "SYSTEM_INIT_FAILURE")
        assert hasattr(BaseErrorCodes, "MEMORY_ALLOCATION_FAILURE")
        assert hasattr(BaseErrorCodes, "MODULE_IMPORT_FAILURE")
        assert hasattr(BaseErrorCodes, "ENVIRONMENT_VARIABLE_MISSING")

        # Test configuration errors
        assert hasattr(BaseErrorCodes, "CONFIG_FILE_NOT_FOUND")
        assert hasattr(BaseErrorCodes, "INVALID_CONFIG_SYNTAX")
        assert hasattr(BaseErrorCodes, "MISSING_CONFIG_FIELD")
        assert hasattr(BaseErrorCodes, "DEPRECATED_CONFIG_FIELD")

        # Test persistence errors
        assert hasattr(BaseErrorCodes, "DATABASE_CONNECTION_FAILURE")
        assert hasattr(BaseErrorCodes, "FILE_PERMISSION_DENIED")
        assert hasattr(BaseErrorCodes, "DISK_SPACE_INSUFFICIENT")
        assert hasattr(BaseErrorCodes, "FILE_CORRUPTION_DETECTED")

    def test_base_error_codes_values(self):
        """Test BaseErrorCodes values are correct AIM2ErrorCodes enum members."""
        # Test a few specific mappings
        assert BaseErrorCodes.SYSTEM_INIT_FAILURE == AIM2ErrorCodes.BASE_SYS_C_001
        assert BaseErrorCodes.CONFIG_FILE_NOT_FOUND == AIM2ErrorCodes.BASE_CFG_E_001
        assert (
            BaseErrorCodes.DATABASE_CONNECTION_FAILURE == AIM2ErrorCodes.BASE_PER_C_001
        )

        # Test that all attributes are AIM2ErrorCodes enum members
        for attr_name in dir(BaseErrorCodes):
            if not attr_name.startswith("_"):
                attr_value = getattr(BaseErrorCodes, attr_name)
                assert isinstance(attr_value, AIM2ErrorCodes)
                # Should be BASE module codes
                assert attr_value.value.module == "BASE"

    def test_ontology_error_codes_attributes(self):
        """Test OntologyErrorCodes convenience class attributes."""
        # Test parsing errors
        assert hasattr(OntologyErrorCodes, "OWL_PARSING_FAILURE")
        assert hasattr(OntologyErrorCodes, "RDF_EXTRACTION_FAILURE")
        assert hasattr(OntologyErrorCodes, "JSONLD_CONTEXT_FAILURE")
        assert hasattr(OntologyErrorCodes, "UNSUPPORTED_FORMAT")

        # Test validation errors
        assert hasattr(OntologyErrorCodes, "CONSISTENCY_CHECK_FAILED")
        assert hasattr(OntologyErrorCodes, "HIERARCHY_VIOLATION")
        assert hasattr(OntologyErrorCodes, "PROPERTY_MISMATCH")
        assert hasattr(OntologyErrorCodes, "ORPHANED_CLASS")

        # Test integration errors
        assert hasattr(OntologyErrorCodes, "MERGE_CONFLICT")
        assert hasattr(OntologyErrorCodes, "NAMESPACE_COLLISION")
        assert hasattr(OntologyErrorCodes, "INCOMPATIBLE_VERSIONS")
        assert hasattr(OntologyErrorCodes, "DUPLICATE_CONCEPT")

    def test_ontology_error_codes_values(self):
        """Test OntologyErrorCodes values are correct."""
        assert OntologyErrorCodes.OWL_PARSING_FAILURE == AIM2ErrorCodes.ONTO_PARS_E_001
        assert (
            OntologyErrorCodes.CONSISTENCY_CHECK_FAILED
            == AIM2ErrorCodes.ONTO_VALD_E_001
        )
        assert OntologyErrorCodes.MERGE_CONFLICT == AIM2ErrorCodes.ONTO_INTG_E_001

        # Test that all attributes are ONTO module codes
        for attr_name in dir(OntologyErrorCodes):
            if not attr_name.startswith("_"):
                attr_value = getattr(OntologyErrorCodes, attr_name)
                assert isinstance(attr_value, AIM2ErrorCodes)
                assert attr_value.value.module == "ONTO"

    def test_extraction_error_codes_attributes(self):
        """Test ExtractionErrorCodes convenience class attributes."""
        # Test NER errors
        assert hasattr(ExtractionErrorCodes, "NER_MODEL_LOADING_FAILURE")
        assert hasattr(ExtractionErrorCodes, "TEXT_TOKENIZATION_FAILURE")
        assert hasattr(ExtractionErrorCodes, "ENTITY_RECOGNITION_TIMEOUT")
        assert hasattr(ExtractionErrorCodes, "LOW_CONFIDENCE_ENTITY")

        # Test relationship extraction errors
        assert hasattr(ExtractionErrorCodes, "PATTERN_MATCHING_FAILED")
        assert hasattr(ExtractionErrorCodes, "ENTITY_PAIR_VALIDATION_FAILED")
        assert hasattr(ExtractionErrorCodes, "DEPENDENCY_PARSING_FAILURE")
        assert hasattr(ExtractionErrorCodes, "NO_RELATIONSHIPS_FOUND")

    def test_extraction_error_codes_values(self):
        """Test ExtractionErrorCodes values are correct."""
        assert (
            ExtractionErrorCodes.NER_MODEL_LOADING_FAILURE
            == AIM2ErrorCodes.EXTR_NER_E_001
        )
        assert (
            ExtractionErrorCodes.PATTERN_MATCHING_FAILED
            == AIM2ErrorCodes.EXTR_REL_E_001
        )

        # Test that all attributes are EXTR module codes
        for attr_name in dir(ExtractionErrorCodes):
            if not attr_name.startswith("_"):
                attr_value = getattr(ExtractionErrorCodes, attr_name)
                assert isinstance(attr_value, AIM2ErrorCodes)
                assert attr_value.value.module == "EXTR"

    def test_llm_error_codes_attributes(self):
        """Test LLMErrorCodes convenience class attributes."""
        # Test API errors
        assert hasattr(LLMErrorCodes, "API_REQUEST_FAILED")
        assert hasattr(LLMErrorCodes, "INVALID_API_RESPONSE")
        assert hasattr(LLMErrorCodes, "API_QUOTA_EXCEEDED")
        assert hasattr(LLMErrorCodes, "API_RESPONSE_TRUNCATED")

        # Test authentication errors
        assert hasattr(LLMErrorCodes, "INVALID_API_KEY")
        assert hasattr(LLMErrorCodes, "INSUFFICIENT_PERMISSIONS")
        assert hasattr(LLMErrorCodes, "AUTH_SERVICE_UNAVAILABLE")

        # Test timeout errors
        assert hasattr(LLMErrorCodes, "REQUEST_TIMEOUT")
        assert hasattr(LLMErrorCodes, "MODEL_INFERENCE_TIMEOUT")
        assert hasattr(LLMErrorCodes, "SLOW_RESPONSE_DETECTED")

    def test_llm_error_codes_values(self):
        """Test LLMErrorCodes values are correct."""
        assert LLMErrorCodes.API_REQUEST_FAILED == AIM2ErrorCodes.LLM_API_E_001
        assert LLMErrorCodes.INVALID_API_KEY == AIM2ErrorCodes.LLM_AUTH_E_001
        assert LLMErrorCodes.REQUEST_TIMEOUT == AIM2ErrorCodes.LLM_TIME_E_001

        # Test that all attributes are LLM module codes
        for attr_name in dir(LLMErrorCodes):
            if not attr_name.startswith("_"):
                attr_value = getattr(LLMErrorCodes, attr_name)
                assert isinstance(attr_value, AIM2ErrorCodes)
                assert attr_value.value.module == "LLM"

    def test_validation_error_codes_attributes(self):
        """Test ValidationErrorCodes convenience class attributes."""
        # Test schema validation errors
        assert hasattr(ValidationErrorCodes, "SCHEMA_VALIDATION_FAILED")
        assert hasattr(ValidationErrorCodes, "REQUIRED_FIELD_MISSING")
        assert hasattr(ValidationErrorCodes, "INVALID_FIELD_TYPE")
        assert hasattr(ValidationErrorCodes, "UNKNOWN_FIELD")

        # Test data validation errors
        assert hasattr(ValidationErrorCodes, "DATA_FORMAT_INVALID")
        assert hasattr(ValidationErrorCodes, "VALUE_OUT_OF_RANGE")
        assert hasattr(ValidationErrorCodes, "INVALID_DATA_ENCODING")
        assert hasattr(ValidationErrorCodes, "DATA_QUALITY_ISSUE")

    def test_validation_error_codes_values(self):
        """Test ValidationErrorCodes values are correct."""
        assert (
            ValidationErrorCodes.SCHEMA_VALIDATION_FAILED
            == AIM2ErrorCodes.VALID_SCHM_E_001
        )
        assert (
            ValidationErrorCodes.DATA_FORMAT_INVALID == AIM2ErrorCodes.VALID_DATA_E_001
        )

        # Test that all attributes are VALID module codes
        for attr_name in dir(ValidationErrorCodes):
            if not attr_name.startswith("_"):
                attr_value = getattr(ValidationErrorCodes, attr_name)
                assert isinstance(attr_value, AIM2ErrorCodes)
                assert attr_value.value.module == "VALID"

    def test_convenience_class_completeness(self):
        """Test that convenience classes cover all error codes for their modules."""
        # Get all error codes by module from the registry
        registry = ErrorCodeRegistry()

        # Test BASE module coverage
        base_codes_from_registry = set(
            code.code for code in registry.get_codes_by_module("BASE")
        )
        base_codes_from_convenience = set()
        for attr_name in dir(BaseErrorCodes):
            if not attr_name.startswith("_"):
                attr_value = getattr(BaseErrorCodes, attr_name)
                base_codes_from_convenience.add(attr_value.value.code)
        assert (
            base_codes_from_registry == base_codes_from_convenience
        ), "BaseErrorCodes doesn't cover all BASE module error codes"

        # Test ONTO module coverage
        onto_codes_from_registry = set(
            code.code for code in registry.get_codes_by_module("ONTO")
        )
        onto_codes_from_convenience = set()
        for attr_name in dir(OntologyErrorCodes):
            if not attr_name.startswith("_"):
                attr_value = getattr(OntologyErrorCodes, attr_name)
                onto_codes_from_convenience.add(attr_value.value.code)
        assert (
            onto_codes_from_registry == onto_codes_from_convenience
        ), "OntologyErrorCodes doesn't cover all ONTO module error codes"

        # Test EXTR module coverage
        extr_codes_from_registry = set(
            code.code for code in registry.get_codes_by_module("EXTR")
        )
        extr_codes_from_convenience = set()
        for attr_name in dir(ExtractionErrorCodes):
            if not attr_name.startswith("_"):
                attr_value = getattr(ExtractionErrorCodes, attr_name)
                extr_codes_from_convenience.add(attr_value.value.code)
        assert (
            extr_codes_from_registry == extr_codes_from_convenience
        ), "ExtractionErrorCodes doesn't cover all EXTR module error codes"

        # Test LLM module coverage
        llm_codes_from_registry = set(
            code.code for code in registry.get_codes_by_module("LLM")
        )
        llm_codes_from_convenience = set()
        for attr_name in dir(LLMErrorCodes):
            if not attr_name.startswith("_"):
                attr_value = getattr(LLMErrorCodes, attr_name)
                llm_codes_from_convenience.add(attr_value.value.code)
        assert (
            llm_codes_from_registry == llm_codes_from_convenience
        ), "LLMErrorCodes doesn't cover all LLM module error codes"

        # Test VALID module coverage
        valid_codes_from_registry = set(
            code.code for code in registry.get_codes_by_module("VALID")
        )
        valid_codes_from_convenience = set()
        for attr_name in dir(ValidationErrorCodes):
            if not attr_name.startswith("_"):
                attr_value = getattr(ValidationErrorCodes, attr_name)
                valid_codes_from_convenience.add(attr_value.value.code)
        assert (
            valid_codes_from_registry == valid_codes_from_convenience
        ), "ValidationErrorCodes doesn't cover all VALID module error codes"

    def test_convenience_class_usage_examples(self):
        """Test that convenience classes can be used in exception creation."""
        # Test that each convenience class can be used with exceptions

        # BaseErrorCodes usage
        exc1 = AIM2Exception("Test", error_code=BaseErrorCodes.SYSTEM_INIT_FAILURE)
        assert exc1.error_code == "AIM2_BASE_SYS_C_001"
        assert exc1.error_info is not None
        assert exc1.error_info.module == "BASE"

        # OntologyErrorCodes usage
        exc2 = OntologyException(
            "Test", error_code=OntologyErrorCodes.OWL_PARSING_FAILURE
        )
        assert exc2.error_code == "AIM2_ONTO_PARS_E_001"
        assert exc2.error_info.module == "ONTO"

        # ExtractionErrorCodes usage
        exc3 = ExtractionException(
            "Test", error_code=ExtractionErrorCodes.NER_MODEL_LOADING_FAILURE
        )
        assert exc3.error_code == "AIM2_EXTR_NER_E_001"
        assert exc3.error_info.module == "EXTR"

        # LLMErrorCodes usage
        exc4 = LLMException("Test", error_code=LLMErrorCodes.API_REQUEST_FAILED)
        assert exc4.error_code == "AIM2_LLM_API_E_001"
        assert exc4.error_info.module == "LLM"

        # ValidationErrorCodes usage
        exc5 = ValidationException(
            "Test", error_code=ValidationErrorCodes.SCHEMA_VALIDATION_FAILED
        )
        assert exc5.error_code == "AIM2_VALID_SCHM_E_001"
        assert exc5.error_info.module == "VALID"


class TestErrorCodeSystemIntegration:
    """Test integration scenarios for the error code system."""

    def test_all_exception_subclasses_with_enum_codes(self):
        """Test that all exception subclasses work with enum error codes."""
        # Test OntologyException with ontology error codes
        onto_exc = OntologyException(
            "OWL parsing failed", error_code=AIM2ErrorCodes.ONTO_PARS_E_001
        )
        assert onto_exc.error_code == "AIM2_ONTO_PARS_E_001"
        assert onto_exc.error_info is not None
        assert onto_exc.error_info.module == "ONTO"

        # Test ExtractionException with extraction error codes
        extr_exc = ExtractionException(
            "NER model loading failed", error_code=AIM2ErrorCodes.EXTR_NER_E_001
        )
        assert extr_exc.error_code == "AIM2_EXTR_NER_E_001"
        assert extr_exc.error_info.module == "EXTR"

        # Test LLMException with LLM error codes
        llm_exc = LLMException(
            "API request failed", error_code=AIM2ErrorCodes.LLM_API_E_001
        )
        assert llm_exc.error_code == "AIM2_LLM_API_E_001"
        assert llm_exc.error_info.module == "LLM"

        # Test ValidationException with validation error codes
        valid_exc = ValidationException(
            "Schema validation failed", error_code=AIM2ErrorCodes.VALID_SCHM_E_001
        )
        assert valid_exc.error_code == "AIM2_VALID_SCHM_E_001"
        assert valid_exc.error_info.module == "VALID"

    def test_exception_subclasses_with_convenience_classes(self):
        """Test exception subclasses with convenience classes."""
        # OntologyException with OntologyErrorCodes
        onto_exc = OntologyException(
            "Consistency check failed",
            error_code=OntologyErrorCodes.CONSISTENCY_CHECK_FAILED,
        )
        assert onto_exc.error_code == "AIM2_ONTO_VALD_E_001"

        # ExtractionException with ExtractionErrorCodes
        extr_exc = ExtractionException(
            "Entity recognition timeout",
            error_code=ExtractionErrorCodes.ENTITY_RECOGNITION_TIMEOUT,
        )
        assert extr_exc.error_code == "AIM2_EXTR_NER_E_003"

        # LLMException with LLMErrorCodes
        llm_exc = LLMException(
            "Invalid API key", error_code=LLMErrorCodes.INVALID_API_KEY
        )
        assert llm_exc.error_code == "AIM2_LLM_AUTH_E_001"

        # ValidationException with ValidationErrorCodes
        valid_exc = ValidationException(
            "Required field missing",
            error_code=ValidationErrorCodes.REQUIRED_FIELD_MISSING,
        )
        assert valid_exc.error_code == "AIM2_VALID_SCHM_E_002"

    def test_mixed_error_code_usage_patterns(self):
        """Test mixed usage patterns of enum and string error codes."""
        exceptions = []

        # Mix of enum codes, string codes, and default codes
        exceptions.append(
            AIM2Exception("Test 1", error_code=AIM2ErrorCodes.BASE_SYS_C_001)
        )
        exceptions.append(OntologyException("Test 2", error_code="CUSTOM_ONTO_ERROR"))
        exceptions.append(ExtractionException("Test 3"))  # Default code
        exceptions.append(
            LLMException("Test 4", error_code=LLMErrorCodes.API_REQUEST_FAILED)
        )
        exceptions.append(
            ValidationException("Test 5", error_code="LEGACY_VALIDATION_ERROR")
        )

        # Verify all work correctly
        assert exceptions[0].error_info is not None  # Enum code should have error_info
        assert (
            exceptions[1].error_info is None
        )  # Custom string code should not have error_info
        assert (
            exceptions[2].error_info is None
        )  # Default code might not have error_info
        assert exceptions[3].error_info is not None  # Enum code should have error_info
        assert (
            exceptions[4].error_info is None
        )  # Custom string code should not have error_info

    def test_error_registry_integration_with_exceptions(self):
        """Test error registry integration with exception instances."""
        registry = ErrorCodeRegistry()

        # Create exceptions with various error code types
        exceptions = [
            AIM2Exception("Test", error_code=AIM2ErrorCodes.BASE_SYS_C_001),
            OntologyException(
                "Test", error_code=OntologyErrorCodes.OWL_PARSING_FAILURE
            ),
            ExtractionException("Test", error_code="CUSTOM_ERROR"),
            LLMException("Test", error_code=LLMErrorCodes.REQUEST_TIMEOUT),
            ValidationException("Test"),  # Default
        ]

        # Test registry validation against exception error codes
        for exc in exceptions:
            is_valid = registry.validate_error_code(exc.error_code)
            has_error_info = exc.error_info is not None

            # Valid codes should have error_info, invalid ones should not
            if is_valid:
                assert (
                    has_error_info
                ), f"Valid code {exc.error_code} should have error_info"
            # Note: Invalid codes will not have error_info

    def test_exception_hierarchy_with_error_codes(self):
        """Test exception hierarchy still works with new error code system."""
        # Create exceptions with error codes
        base_exc = AIM2Exception("Base error", error_code=AIM2ErrorCodes.BASE_SYS_C_001)
        onto_exc = OntologyException(
            "Onto error", error_code=OntologyErrorCodes.OWL_PARSING_FAILURE
        )

        # Test inheritance
        assert isinstance(onto_exc, AIM2Exception)
        assert isinstance(onto_exc, Exception)

        # Test that both have error_info
        assert base_exc.error_info is not None
        assert onto_exc.error_info is not None

        # Test different modules
        assert base_exc.error_info.module == "BASE"
        assert onto_exc.error_info.module == "ONTO"

    def test_exception_chaining_with_error_codes(self):
        """Test exception chaining works with error codes."""
        # Create a chain of exceptions with error codes
        root_cause = ValueError("Invalid input")

        middle_exc = ExtractionException(
            "Text processing failed",
            error_code=ExtractionErrorCodes.TEXT_TOKENIZATION_FAILURE,
            cause=root_cause,
        )

        top_exc = AIM2Exception(
            "Pipeline failed",
            error_code=AIM2ErrorCodes.BASE_SYS_E_003,
            cause=middle_exc,
        )

        # Test chaining
        assert top_exc.cause == middle_exc
        assert middle_exc.cause == root_cause

        # Test error_info preservation
        assert top_exc.error_info is not None
        assert middle_exc.error_info is not None
        assert top_exc.error_info.module == "BASE"
        assert middle_exc.error_info.module == "EXTR"

    def test_exception_serialization_deserialization_integration(self):
        """Test serialization/deserialization with error code system."""
        # Create exception with enum error code
        original = OntologyException(
            "Ontology parsing failed",
            error_code=OntologyErrorCodes.OWL_PARSING_FAILURE,
            context={"file": "test.owl", "line": 42},
        )

        # Serialize
        serialized = original.to_dict()

        # Verify error_info is included
        assert "error_info" in serialized
        assert serialized["error_info"]["module"] == "ONTO"
        assert serialized["error_info"]["severity"] == "ERROR"

        # Deserialize
        restored = OntologyException.from_dict(serialized)

        # Verify restoration
        assert restored.message == original.message
        assert restored.error_code == original.error_code
        assert restored.context == original.context
        # Note: error_info will be repopulated from registry during reconstruction

    def test_performance_considerations(self):
        """Test performance considerations of error code system."""
        import time

        # Test registry singleton performance
        start_time = time.time()
        registries = [ErrorCodeRegistry() for _ in range(100)]
        registry_time = time.time() - start_time

        # Should be fast (all same instance)
        assert registry_time < 0.1  # Should complete in less than 100ms
        assert all(reg is registries[0] for reg in registries)

        # Test exception creation performance
        start_time = time.time()
        exceptions = []
        for i in range(100):
            if i % 2 == 0:
                exc = AIM2Exception(
                    f"Test {i}", error_code=AIM2ErrorCodes.BASE_SYS_C_001
                )
            else:
                exc = OntologyException(
                    f"Test {i}", error_code=OntologyErrorCodes.OWL_PARSING_FAILURE
                )
            exceptions.append(exc)
        exception_time = time.time() - start_time

        # Should be reasonably fast
        assert exception_time < 1.0  # Should complete in less than 1 second
        assert len(exceptions) == 100

    def test_comprehensive_error_code_coverage(self):
        """Test comprehensive coverage of error code system features."""
        registry = ErrorCodeRegistry()

        # Test that we can create exceptions for all error codes
        successful_creations = 0

        for error_code in AIM2ErrorCodes:
            try:
                # Create exception with enum
                exc1 = AIM2Exception("Test", error_code=error_code)
                assert exc1.error_info is not None
                assert exc1.error_info.code == error_code.value.code

                # Verify registry validation
                assert registry.validate_error_code(exc1.error_code)

                # Test detailed message
                detailed_msg = exc1.get_detailed_message()
                assert len(detailed_msg) > 0
                assert error_code.value.severity.name in detailed_msg

                successful_creations += 1

            except Exception as e:
                assert False, f"Failed to create exception for {error_code}: {e}"

        # Should have successfully created exceptions for all error codes
        assert successful_creations == 50

    def test_cross_module_error_usage(self):
        """Test using error codes across different modules (anti-pattern detection)."""
        # While technically possible, using error codes from wrong modules
        # should still work but might indicate design issues

        # Using ONTO error code with ExtractionException (unusual but valid)
        exc1 = ExtractionException(
            "Unusual case", error_code=AIM2ErrorCodes.ONTO_PARS_E_001
        )
        assert exc1.error_code == "AIM2_ONTO_PARS_E_001"
        assert (
            exc1.error_info.module == "ONTO"
        )  # Module from error code, not exception type

        # Using LLM error code with ValidationException (unusual but valid)
        exc2 = ValidationException(
            "Cross-module usage", error_code=LLMErrorCodes.API_REQUEST_FAILED
        )
        assert exc2.error_code == "AIM2_LLM_API_E_001"
        assert exc2.error_info.module == "LLM"

        # Both should work correctly despite the unusual usage pattern


class TestEnhancedSerialization:
    """Test enhanced serialization methods for exception hierarchy."""

    def test_error_info_serialization_round_trip(self):
        """Test that error_info metadata is preserved in serialization round-trip."""
        # Create exception with enum error code (has error_info)
        original = AIM2Exception(
            "System initialization failed",
            error_code=AIM2ErrorCodes.BASE_SYS_C_001,
            context={"system": "test", "component": "init"},
        )

        # Verify original has error_info
        assert original.error_info is not None
        assert original.error_info.module == "BASE"
        assert original.error_info.severity == ErrorSeverity.CRITICAL

        # Serialize
        serialized = original.to_dict()

        # Verify error_info is in serialized data
        assert "error_info" in serialized
        error_info = serialized["error_info"]
        assert error_info["module"] == "BASE"
        assert error_info["category"] == "SYS"
        assert error_info["severity"] == "CRITICAL"
        assert error_info["description"] == "System initialization failure"
        assert error_info["resolution_hint"] == "Check dependencies and permissions"

        # Deserialize
        restored = AIM2Exception.from_dict(serialized)

        # Verify error_info is properly restored
        assert restored.error_info is not None
        assert restored.error_info.code == original.error_code
        assert restored.error_info.module == original.error_info.module
        assert restored.error_info.category == original.error_info.category
        assert restored.error_info.severity == original.error_info.severity
        assert restored.error_info.description == original.error_info.description
        assert (
            restored.error_info.resolution_hint == original.error_info.resolution_hint
        )

    def test_specialized_exception_serialization_round_trip(self):
        """Test serialization round-trip for specialized exception classes."""
        # Test OntologyException
        onto_original = OntologyException(
            "OWL parsing failed",
            error_code=OntologyErrorCodes.OWL_PARSING_FAILURE,
            context={"file": "test.owl", "line": 42},
        )

        onto_serialized = onto_original.to_dict()
        onto_restored = OntologyException.from_dict(onto_serialized)

        assert isinstance(onto_restored, OntologyException)
        assert onto_restored.message == onto_original.message
        assert onto_restored.error_code == onto_original.error_code
        assert onto_restored.context == onto_original.context
        assert onto_restored.error_info is not None
        assert onto_restored.error_info.module == "ONTO"

        # Test ExtractionException
        extr_original = ExtractionException(
            "NER model loading failed",
            error_code=ExtractionErrorCodes.NER_MODEL_LOADING_FAILURE,
            context={"model_path": "/path/to/model"},
        )

        extr_serialized = extr_original.to_dict()
        extr_restored = ExtractionException.from_dict(extr_serialized)

        assert isinstance(extr_restored, ExtractionException)
        assert extr_restored.message == extr_original.message
        assert extr_restored.error_code == extr_original.error_code
        assert extr_restored.context == extr_original.context
        assert extr_restored.error_info is not None
        assert extr_restored.error_info.module == "EXTR"

    def test_exception_chaining_serialization_round_trip(self):
        """Test that exception chaining is preserved in serialization."""
        # Create exception chain
        root_cause = ValueError("Invalid input data")

        middle_exc = ExtractionException(
            "Text processing failed",
            error_code=ExtractionErrorCodes.TEXT_TOKENIZATION_FAILURE,
            cause=root_cause,
            context={"stage": "tokenization", "input_length": 1024},
        )

        top_exc = AIM2Exception(
            "Pipeline execution failed",
            error_code=AIM2ErrorCodes.BASE_SYS_E_003,
            cause=middle_exc,
            context={"pipeline": "extraction", "stage": "processing"},
        )

        # Serialize top exception
        serialized = top_exc.to_dict()

        # Verify cause information is captured
        assert serialized["cause_type"] == "ExtractionException"
        assert serialized["cause_message"] == str(middle_exc)

        # Deserialize
        restored = AIM2Exception.from_dict(serialized)

        # Verify cause is restored (as generic Exception)
        assert restored.cause is not None
        assert isinstance(restored.cause, Exception)
        assert str(restored.cause) == str(middle_exc)
        assert restored.__cause__ == restored.cause

    def test_nested_exception_chaining_serialization(self):
        """Test serialization of deeply nested exception chains."""
        # Create nested chain: IOError -> OntologyException -> ValidationException
        io_error = IOError("File not found: ontology.owl")

        onto_exc = OntologyException(
            "Failed to load ontology",
            error_code=OntologyErrorCodes.OWL_PARSING_FAILURE,
            cause=io_error,
            context={"file_path": "/path/to/ontology.owl"},
        )

        valid_exc = ValidationException(
            "Ontology validation failed",
            error_code=ValidationErrorCodes.SCHEMA_VALIDATION_FAILED,
            cause=onto_exc,
            context={"validation_stage": "ontology_check"},
        )

        # Serialize the top-level exception
        serialized = valid_exc.to_dict()

        # Verify serialization captures the immediate cause
        assert serialized["cause_type"] == "OntologyException"
        assert "Failed to load ontology" in serialized["cause_message"]

        # Deserialize
        restored = ValidationException.from_dict(serialized)

        # Verify restoration
        assert isinstance(restored, ValidationException)
        assert restored.cause is not None
        assert isinstance(restored.cause, Exception)
        # Note: Only immediate cause is preserved in serialization
        # Deep nesting requires recursive serialization if needed

    def test_mixed_error_code_types_serialization(self):
        """Test serialization with mixed error code types (enum vs string)."""
        # Exception with enum error code
        enum_exc = LLMException(
            "API request failed",
            error_code=LLMErrorCodes.API_REQUEST_FAILED,
            context={"endpoint": "/api/v1/chat", "status_code": 500},
        )

        # Exception with string error code
        string_exc = ValidationException(
            "Custom validation failed",
            error_code="CUSTOM_VALIDATION_ERROR",
            context={"validator": "custom", "field": "username"},
        )

        # Serialize both
        enum_serialized = enum_exc.to_dict()
        string_serialized = string_exc.to_dict()

        # Verify enum error code includes error_info
        assert "error_info" in enum_serialized
        assert enum_serialized["error_info"]["module"] == "LLM"

        # Verify string error code does not include error_info
        assert (
            "error_info" not in string_serialized
            or string_serialized["error_info"] is None
        )

        # Deserialize both
        enum_restored = LLMException.from_dict(enum_serialized)
        string_restored = ValidationException.from_dict(string_serialized)

        # Verify enum error code restoration
        assert enum_restored.error_info is not None
        assert enum_restored.error_info.module == "LLM"

        # Verify string error code restoration
        assert string_restored.error_info is None
        assert string_restored.error_code == "CUSTOM_VALIDATION_ERROR"

    def test_backward_compatibility_serialization(self):
        """Test backward compatibility with old serialization format."""
        # Old format data (without error_info)
        old_format_data = {
            "exception_type": "AIM2Exception",
            "message": "Legacy error",
            "error_code": "AIM2_BASE_SYS_C_001",  # Valid error code
            "context": {"legacy": True},
            "cause_type": None,
            "cause_message": None,
            "traceback": None
            # Note: no "error_info" field
        }

        # Should deserialize successfully
        restored = AIM2Exception.from_dict(old_format_data)

        assert restored.message == "Legacy error"
        assert restored.error_code == "AIM2_BASE_SYS_C_001"
        assert restored.context == {"legacy": True}
        # error_info should be populated from registry lookup during init
        assert restored.error_info is not None
        assert restored.error_info.module == "BASE"

    def test_serialization_error_handling(self):
        """Test error handling in serialization/deserialization."""
        # Test with corrupted error_info data
        corrupted_data = {
            "message": "Test error",
            "error_code": "AIM2_BASE_SYS_C_001",
            "error_info": {
                "module": "BASE",
                # Missing required fields
                "invalid_severity": "INVALID",
            },
        }

        # Should handle gracefully
        restored = AIM2Exception.from_dict(corrupted_data)
        assert restored.message == "Test error"
        assert restored.error_code == "AIM2_BASE_SYS_C_001"
        # error_info should be populated from registry lookup since corrupted data failed
        assert restored.error_info is not None

    def test_comprehensive_serialization_metadata(self):
        """Test that all metadata is preserved in serialization."""
        # Create exception with all possible metadata
        original = ValidationException(
            "Comprehensive test error",
            error_code=ValidationErrorCodes.SCHEMA_VALIDATION_FAILED,
            cause=RuntimeError("Root cause"),
            context={
                "nested": {"level1": {"level2": "deep_value"}},
                "list": [1, 2, {"inner": "value"}],
                "unicode": "测试数据",
                "numbers": [1, 2.5, -3],
                "boolean": True,
                "null": None,
            },
        )

        # Serialize
        serialized = original.to_dict()

        # Verify all fields are present
        required_fields = [
            "exception_type",
            "message",
            "error_code",
            "context",
            "cause_type",
            "cause_message",
            "error_info",
        ]
        for field in required_fields:
            assert field in serialized

        # Verify complex context is preserved
        assert serialized["context"]["nested"]["level1"]["level2"] == "deep_value"
        assert serialized["context"]["unicode"] == "测试数据"
        assert serialized["context"]["boolean"] is True
        assert serialized["context"]["null"] is None

        # Verify error_info metadata
        error_info = serialized["error_info"]
        assert error_info["module"] == "VALID"
        assert error_info["category"] == "SCHM"
        assert error_info["severity"] == "ERROR"

        # Deserialize and verify
        restored = ValidationException.from_dict(serialized)

        assert isinstance(restored, ValidationException)
        assert restored.message == original.message
        assert restored.error_code == original.error_code
        assert restored.context == original.context
        assert restored.cause is not None
        assert restored.error_info is not None
        assert restored.error_info.module == original.error_info.module

    def test_exception_type_preservation_in_serialization(self):
        """Test that exception types are correctly preserved and restored."""
        exceptions = [
            AIM2Exception("Base error", error_code=AIM2ErrorCodes.BASE_SYS_C_001),
            OntologyException(
                "Onto error", error_code=OntologyErrorCodes.OWL_PARSING_FAILURE
            ),
            ExtractionException(
                "Extr error", error_code=ExtractionErrorCodes.NER_MODEL_LOADING_FAILURE
            ),
            LLMException("LLM error", error_code=LLMErrorCodes.API_REQUEST_FAILED),
            ValidationException(
                "Valid error", error_code=ValidationErrorCodes.SCHEMA_VALIDATION_FAILED
            ),
        ]

        for original in exceptions:
            # Serialize
            serialized = original.to_dict()

            # Verify exception_type is correct
            assert serialized["exception_type"] == original.__class__.__name__

            # Deserialize using the specific class
            restored = original.__class__.from_dict(serialized)

            # Verify type is preserved
            assert isinstance(restored, original.__class__)
            assert type(restored) == type(original)
            assert restored.message == original.message
            assert restored.error_code == original.error_code


class TestMessageFormat:
    """Test the MessageFormat enum functionality."""

    def test_all_message_formats_exist(self):
        """Test that all expected message formats are defined."""
        expected_formats = ["CONSOLE", "STRUCTURED", "API", "JSON", "MARKDOWN"]

        for format_name in expected_formats:
            assert hasattr(MessageFormat, format_name)
            format_obj = getattr(MessageFormat, format_name)
            assert isinstance(format_obj, MessageFormat)

    def test_message_format_values(self):
        """Test that message formats have correct string values."""
        assert MessageFormat.CONSOLE.value == "console"
        assert MessageFormat.STRUCTURED.value == "structured"
        assert MessageFormat.API.value == "api"
        assert MessageFormat.JSON.value == "json"
        assert MessageFormat.MARKDOWN.value == "markdown"

    def test_message_format_enumeration(self):
        """Test enumeration and membership of message formats."""
        all_formats = list(MessageFormat)
        assert len(all_formats) == 5

        for format_obj in [
            MessageFormat.CONSOLE,
            MessageFormat.STRUCTURED,
            MessageFormat.API,
            MessageFormat.JSON,
            MessageFormat.MARKDOWN,
        ]:
            assert format_obj in all_formats


class TestStandardMessageTemplate:
    """Test the StandardMessageTemplate functionality."""

    def setup_method(self):
        """Set up test fixtures."""
        self.template = StandardMessageTemplate()
        self.error_info = AIM2ErrorCodes.BASE_SYS_C_001.value
        self.message = "Test error message"
        self.error_code = "AIM2_BASE_SYS_C_001"
        self.context = {"file": "test.py", "line": 42, "function": "test_func"}

    def test_template_initialization(self):
        """Test that template is properly initialized."""
        assert self.template.template_name == "standard"
        assert MessageFormat.CONSOLE in self.template.supported_formats
        assert MessageFormat.STRUCTURED in self.template.supported_formats
        assert MessageFormat.API in self.template.supported_formats
        assert MessageFormat.JSON in self.template.supported_formats
        assert MessageFormat.MARKDOWN in self.template.supported_formats

    def test_console_format(self):
        """Test console format output."""
        result = self.template.format_message(
            self.message,
            self.error_code,
            self.error_info,
            self.context,
            MessageFormat.CONSOLE,
            include_colors=False,
        )

        assert "[CRITICAL]" in result
        assert self.error_code in result
        assert self.message in result
        assert "Description:" in result
        assert self.error_info.description in result
        assert "Resolution:" in result
        assert self.error_info.resolution_hint in result
        assert "Context:" in result

    def test_console_format_with_colors(self):
        """Test console format with color codes."""
        result = self.template.format_message(
            self.message,
            self.error_code,
            self.error_info,
            self.context,
            MessageFormat.CONSOLE,
            include_colors=True,
        )

        # Should contain ANSI color codes for CRITICAL severity
        assert "\033[41m\033[97m" in result or "🔴" in result
        assert self.message in result

    def test_structured_format(self):
        """Test structured format output."""
        result = self.template.format_message(
            self.message,
            self.error_code,
            self.error_info,
            self.context,
            MessageFormat.STRUCTURED,
            include_timestamp=True,
        )

        assert "Timestamp:" in result
        assert "Severity: CRITICAL" in result
        assert f"Code: {self.error_code}" in result
        assert "Module: BASE" in result
        assert "Category: SYS" in result
        assert f"Message: {self.message}" in result
        assert "Context:" in result

    def test_api_format(self):
        """Test API format output."""
        result = self.template.format_message(
            self.message,
            self.error_code,
            self.error_info,
            self.context,
            MessageFormat.API,
            include_context=True,
        )

        assert "[CRITICAL]" in result
        assert self.message in result
        assert "Suggested action:" in result
        assert self.error_info.resolution_hint in result

    def test_json_format(self):
        """Test JSON format output."""
        import json as json_module

        result = self.template.format_message(
            self.message,
            self.error_code,
            self.error_info,
            self.context,
            MessageFormat.JSON,
            include_timestamp=True,
            include_context=True,
        )

        # Should be valid JSON
        parsed = json_module.loads(result)
        assert parsed["error_code"] == self.error_code
        assert parsed["message"] == self.message
        assert parsed["severity"] == "CRITICAL"
        assert parsed["module"] == "BASE"
        assert parsed["category"] == "SYS"
        assert "timestamp" in parsed
        assert "context" in parsed

    def test_markdown_format(self):
        """Test Markdown format output."""
        result = self.template.format_message(
            self.message,
            self.error_code,
            self.error_info,
            self.context,
            MessageFormat.MARKDOWN,
            include_context=True,
        )

        assert "## 🔴 CRITICAL:" in result
        assert f"**Message:** {self.message}" in result
        assert f"**Module:** BASE" in result
        assert "**Context:**" in result
        assert "```json" in result

    def test_format_without_error_info(self):
        """Test formatting without error info."""
        result = self.template.format_message(
            self.message, self.error_code, None, self.context, MessageFormat.CONSOLE
        )

        assert "⚠ [ERROR]" in result
        assert self.message in result
        assert self.error_code in result

    def test_format_without_context(self):
        """Test formatting without context."""
        result = self.template.format_message(
            self.message,
            self.error_code,
            self.error_info,
            None,
            MessageFormat.CONSOLE,
            include_context=True,
        )

        assert self.message in result
        # Should not contain context section
        assert "Context:" not in result

    def test_format_options(self):
        """Test various formatting options."""
        # Test without resolution
        result = self.template.format_message(
            self.message,
            self.error_code,
            self.error_info,
            self.context,
            MessageFormat.CONSOLE,
            include_resolution=False,
        )
        assert "Resolution:" not in result

        # Test without context
        result = self.template.format_message(
            self.message,
            self.error_code,
            self.error_info,
            self.context,
            MessageFormat.CONSOLE,
            include_context=False,
        )
        assert "Context:" not in result

    def test_api_format_security(self):
        """Test that API format filters sensitive context."""
        sensitive_context = {
            "username": "test_user",
            "password": "secret123",
            "api_key": "key_123",
            "token": "token_abc",
            "file": "test.py",
        }

        result = self.template.format_message(
            self.message,
            self.error_code,
            self.error_info,
            sensitive_context,
            MessageFormat.API,
            include_context=True,
        )

        # Should include non-sensitive data
        assert "file=test.py" in result
        assert "username=test_user" in result

        # Should exclude sensitive data
        assert "password" not in result
        assert "secret123" not in result
        assert "key_123" not in result
        assert "token_abc" not in result


class TestCompactMessageTemplate:
    """Test the CompactMessageTemplate functionality."""

    def setup_method(self):
        """Set up test fixtures."""
        self.template = CompactMessageTemplate()
        self.error_info = AIM2ErrorCodes.LLM_API_E_001.value
        self.message = "API request failed"
        self.error_code = "AIM2_LLM_API_E_001"

    def test_template_initialization(self):
        """Test that template is properly initialized."""
        assert self.template.template_name == "compact"
        assert MessageFormat.CONSOLE in self.template.supported_formats
        assert MessageFormat.API in self.template.supported_formats
        assert MessageFormat.JSON in self.template.supported_formats

    def test_console_format(self):
        """Test compact console format."""
        result = self.template.format_message(
            self.message, self.error_code, self.error_info, None, MessageFormat.CONSOLE
        )

        # Should be concise
        assert "❌" in result
        assert self.error_code in result
        assert self.message in result
        # Should not contain description or resolution
        assert "Description:" not in result
        assert "Resolution:" not in result

    def test_api_format(self):
        """Test compact API format."""
        result = self.template.format_message(
            self.message, self.error_code, self.error_info, None, MessageFormat.API
        )

        assert "[ERROR]" in result
        assert self.error_code in result
        assert self.message in result

    def test_json_format(self):
        """Test compact JSON format."""
        import json as json_module

        result = self.template.format_message(
            self.message, self.error_code, self.error_info, None, MessageFormat.JSON
        )

        parsed = json_module.loads(result)
        assert parsed["code"] == self.error_code
        assert parsed["message"] == self.message
        assert parsed["severity"] == "ERROR"
        # Should not contain extra metadata
        assert "description" not in parsed
        assert "resolution_hint" not in parsed


class TestMessageTemplateRegistry:
    """Test the MessageTemplateRegistry functionality."""

    def setup_method(self):
        """Set up test fixtures."""
        # Get fresh registry instance
        MessageTemplateRegistry._instance = None
        self.registry = MessageTemplateRegistry()

    def test_singleton_pattern(self):
        """Test that registry implements singleton pattern."""
        registry1 = MessageTemplateRegistry()
        registry2 = MessageTemplateRegistry()

        assert registry1 is registry2
        assert id(registry1) == id(registry2)

    def test_builtin_templates_registered(self):
        """Test that built-in templates are registered."""
        templates = self.registry.list_templates()
        assert "standard" in templates
        assert "compact" in templates

    def test_get_template(self):
        """Test getting templates by name."""
        standard = self.registry.get_template("standard")
        assert isinstance(standard, StandardMessageTemplate)

        compact = self.registry.get_template("compact")
        assert isinstance(compact, CompactMessageTemplate)

        # Test default template
        default = self.registry.get_template()
        assert isinstance(default, StandardMessageTemplate)

    def test_get_template_invalid_name(self):
        """Test getting template with invalid name falls back to default."""
        template = self.registry.get_template("nonexistent")
        assert isinstance(template, StandardMessageTemplate)

    def test_preferred_templates(self):
        """Test getting preferred templates for formats."""
        console_template = self.registry.get_preferred_template(MessageFormat.CONSOLE)
        assert isinstance(console_template, StandardMessageTemplate)

        api_template = self.registry.get_preferred_template(MessageFormat.API)
        assert isinstance(api_template, CompactMessageTemplate)

    def test_set_default_template(self):
        """Test setting default template."""
        self.registry.set_default_template("compact")
        default = self.registry.get_template()
        assert isinstance(default, CompactMessageTemplate)

    def test_set_default_template_invalid(self):
        """Test setting invalid default template raises error."""
        try:
            self.registry.set_default_template("nonexistent")
            assert False, "Should raise KeyError"
        except KeyError:
            pass

    def test_set_format_preference(self):
        """Test setting format preferences."""
        self.registry.set_format_preference(MessageFormat.CONSOLE, "compact")
        template = self.registry.get_preferred_template(MessageFormat.CONSOLE)
        assert isinstance(template, CompactMessageTemplate)

    def test_set_format_preference_invalid(self):
        """Test setting invalid format preference raises error."""
        try:
            self.registry.set_format_preference(MessageFormat.CONSOLE, "nonexistent")
            assert False, "Should raise KeyError"
        except KeyError:
            pass

    def test_format_message_via_registry(self):
        """Test formatting messages through the registry."""
        message = "Test error"
        error_code = "TEST_001"

        result = self.registry.format_message(
            message, error_code, format_type=MessageFormat.CONSOLE
        )

        assert message in result
        assert error_code in result

    def test_format_message_with_specific_template(self):
        """Test formatting with specific template name."""
        message = "Test error"
        error_code = "TEST_001"

        result = self.registry.format_message(
            message,
            error_code,
            format_type=MessageFormat.CONSOLE,
            template_name="compact",
        )

        # Should use compact format even for console
        assert message in result
        assert error_code in result

    def test_register_custom_template(self):
        """Test registering custom template."""

        class CustomTemplate(BaseMessageTemplate):
            def __init__(self):
                super().__init__("custom")

            def format_message(
                self,
                message,
                error_code,
                error_info=None,
                context=None,
                format_type=MessageFormat.CONSOLE,
                **kwargs,
            ):
                return f"CUSTOM: {error_code} - {message}"

        custom = CustomTemplate()
        self.registry.register_template(custom)

        templates = self.registry.list_templates()
        assert "custom" in templates

        template = self.registry.get_template("custom")
        assert isinstance(template, CustomTemplate)

        result = self.registry.format_message(
            "test", "TEST_001", template_name="custom"
        )
        assert result == "CUSTOM: TEST_001 - test"


class TestTemplateSystemIntegration:
    """Test integration of template system with exception classes."""

    def setup_method(self):
        """Set up test fixtures."""
        # Reset registry for clean tests
        MessageTemplateRegistry._instance = None

    def test_get_detailed_message_default_behavior(self):
        """Test that get_detailed_message works with default settings."""
        exc = AIM2Exception(
            "System error",
            error_code=AIM2ErrorCodes.BASE_SYS_C_001,
            context={"component": "init"},
        )

        result = exc.get_detailed_message()

        assert "🔴" in result or "[CRITICAL]" in result
        assert "AIM2_BASE_SYS_C_001" in result
        assert "System error" in result
        assert "Description:" in result
        assert "Resolution:" in result

    def test_get_detailed_message_different_formats(self):
        """Test get_detailed_message with different formats."""
        exc = LLMException(
            "API timeout",
            error_code=LLMErrorCodes.REQUEST_TIMEOUT,
            context={"timeout": 30},
        )

        # Console format
        console = exc.get_detailed_message(MessageFormat.CONSOLE)
        assert "[ERROR]" in console or "❌" in console

        # API format
        api = exc.get_detailed_message(MessageFormat.API)
        assert "[ERROR]" in api
        # API format might use compact template which doesn't include resolution hints
        assert "API timeout" in api

        # JSON format
        json_result = exc.get_detailed_message(MessageFormat.JSON)
        assert '"error_code"' in json_result
        assert '"severity": "ERROR"' in json_result

    def test_format_message_method(self):
        """Test the format_message method on exceptions."""
        exc = OntologyException(
            "Parse failed",
            error_code=OntologyErrorCodes.OWL_PARSING_FAILURE,
            context={"file": "test.owl"},
        )

        console_msg = exc.format_message(MessageFormat.CONSOLE)
        api_msg = exc.format_message(MessageFormat.API)

        assert console_msg != api_msg
        assert "Parse failed" in console_msg
        assert "Parse failed" in api_msg

    def test_convenience_methods(self):
        """Test convenience formatting methods."""
        exc = ValidationException(
            "Schema invalid",
            error_code=ValidationErrorCodes.SCHEMA_VALIDATION_FAILED,
            context={"schema": "config.json"},
        )

        # Test all convenience methods
        console = exc.get_console_message()
        api = exc.get_api_message()
        json_msg = exc.get_json_message()
        structured = exc.get_structured_message()
        markdown = exc.get_markdown_message()

        # All should contain the basic message
        for formatted in [console, api, json_msg, structured, markdown]:
            assert "Schema invalid" in formatted

        # JSON should be parseable
        import json as json_module

        parsed = json_module.loads(json_msg)
        assert parsed["message"] == "Schema invalid"

    def test_str_method_uses_templates(self):
        """Test that __str__ method uses the template system."""
        exc = ExtractionException(
            "NER failed", error_code=ExtractionErrorCodes.NER_MODEL_LOADING_FAILURE
        )

        str_result = str(exc)

        # Should use enhanced formatting
        assert "NER failed" in str_result
        assert "AIM2_EXTR_NER_E_001" in str_result
        # Should include additional template formatting
        assert "[ERROR]" in str_result or "❌" in str_result

    def test_backward_compatibility(self):
        """Test that existing code continues to work."""
        # Old usage pattern should still work
        exc = AIM2Exception("Old style error")
        detailed = exc.get_detailed_message()

        assert "Old style error" in detailed
        assert "AIM2_AIM2EXCEPTION" in detailed

    def test_template_system_fallback(self):
        """Test that template system gracefully falls back on errors."""
        # This is hard to test directly, but we can verify basic functionality
        exc = AIM2Exception("Test error", error_code="CUSTOM_CODE")

        # Should not raise exceptions even with custom error codes
        console = exc.get_console_message()
        api = exc.get_api_message()

        assert "Test error" in console
        assert "Test error" in api

    def test_template_customization(self):
        """Test template system customization."""
        exc = AIM2Exception("Custom test", error_code=AIM2ErrorCodes.BASE_SYS_E_003)

        # Test with custom formatting options
        no_colors = exc.get_console_message(include_colors=False)
        with_timestamp = exc.get_structured_message(include_timestamp=True)
        no_context = exc.get_api_message(include_context=False)

        assert "Custom test" in no_colors
        assert "Timestamp:" in with_timestamp
        assert "Custom test" in no_context

    def test_exception_serialization_with_templates(self):
        """Test that serialization works with template-enhanced exceptions."""
        exc = LLMException(
            "Template test",
            error_code=LLMErrorCodes.API_REQUEST_FAILED,
            context={"endpoint": "/api/chat"},
        )

        # Test that serialization still works
        serialized = exc.to_dict()
        restored = LLMException.from_dict(serialized)

        # Both should support template formatting
        original_console = exc.get_console_message()
        restored_console = restored.get_console_message()

        # Should produce similar output (timestamps might differ)
        assert "Template test" in original_console
        assert "Template test" in restored_console

    def test_complex_formatting_scenario(self):
        """Test complex formatting scenario with all features."""
        exc = ValidationException(
            "Complex validation error with unicode: 测试",
            error_code=ValidationErrorCodes.DATA_FORMAT_INVALID,
            context={
                "field": "user_input",
                "expected": "string",
                "actual": "int",
                "value": 123,
                "nested": {"level1": {"level2": "deep"}},
            },
        )

        # Test all formats
        formats = [
            MessageFormat.CONSOLE,
            MessageFormat.STRUCTURED,
            MessageFormat.API,
            MessageFormat.JSON,
            MessageFormat.MARKDOWN,
        ]

        for format_type in formats:
            result = exc.format_message(format_type)

            # All should contain the basic message
            assert "Complex validation error" in result

            # Format-specific checks
            if format_type == MessageFormat.JSON:
                import json as json_module

                parsed = json_module.loads(result)
                # Check the parsed JSON contains the unicode characters
                assert "测试" in parsed["message"]
            elif format_type == MessageFormat.MARKDOWN:
                assert "**Message:**" in result
                # For markdown, check for direct unicode presence
                assert "测试" in result
            else:
                # For other formats, check for direct unicode presence
                assert "测试" in result
