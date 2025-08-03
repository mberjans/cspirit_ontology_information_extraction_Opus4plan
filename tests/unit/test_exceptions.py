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
)


class TestAIM2Exception:
    """Test the base AIM2Exception class functionality."""

    def test_basic_exception_creation(self):
        """Test basic exception creation with message only."""
        message = "Test error message"
        exception = AIM2Exception(message)

        assert str(exception) == message
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
        assert str(exception) == message

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

        assert str(exception) == message
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

        assert str(exception) == message
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

        assert str(exception) == message
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

        assert str(exception) == message
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
        assert str(exception) == ""

    def test_none_message_handling(self):
        """Test handling of None message."""
        # Python allows None as message, it gets converted to "None"
        exception = AIM2Exception(None)
        assert exception.message is None
        assert str(exception) == "None"

    def test_non_string_message(self):
        """Test handling of non-string message."""
        # This should work due to Python's automatic string conversion
        exception = AIM2Exception(123)
        assert str(exception) == "123"

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
        assert len(str(exception)) == 10000

    def test_unicode_message(self):
        """Test handling of Unicode characters in messages."""
        unicode_message = "Error: 测试 🚫 áéíóú"
        exception = AIM2Exception(unicode_message)
        assert exception.message == unicode_message
        assert str(exception) == unicode_message

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
