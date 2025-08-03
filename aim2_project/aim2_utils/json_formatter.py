"""
JSON Formatter Module for AIM2 Project

This module provides a comprehensive JSON formatter for structured log output that:
- Extends logging.Formatter for seamless integration
- Supports configurable fields and output formats
- Provides both pretty-printed and compact JSON output
- Handles edge cases (None values, special characters, etc.)
- Ensures thread-safe and efficient operation
- Follows existing code patterns from the AIM2 logging framework

Classes:
    JSONFormatterError: Custom exception for JSON formatter-related errors
    JSONFormatter: Main JSON formatter class for structured logging

Dependencies:
    - json: For JSON serialization
    - logging: For base formatter functionality
    - datetime: For timestamp operations
    - traceback: For exception formatting
    - threading: For thread safety
    - typing: For type hints
"""

import json
import logging
import traceback
import threading
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Union


class JSONFormatterError(Exception):
    """
    Custom exception for JSON formatter-related errors.

    This exception is raised when JSON formatter operations encounter errors
    such as serialization failures, configuration issues, or field processing problems.

    Args:
        message (str): Error message describing the issue
        cause (Exception, optional): Original exception that caused this error
    """

    def __init__(self, message: str, cause: Optional[Exception] = None):
        super().__init__(message)
        self.message = message
        self.cause = cause
        if cause:
            self.__cause__ = cause


class JSONFormatter(logging.Formatter):
    """
    JSON formatter for structured log output.

    This formatter extends the standard logging.Formatter to provide structured
    JSON output with configurable fields. It supports both pretty-printed and
    compact JSON formats, handles edge cases gracefully, and is thread-safe.

    Attributes:
        fields (List[str]): List of fields to include in JSON output
        pretty_print (bool): Whether to pretty-print JSON output
        custom_fields (Dict[str, Any]): Static custom fields to include
        timestamp_format (str): Format for timestamp field
        use_utc (bool): Whether to use UTC for timestamps
        include_exception_traceback (bool): Whether to include full exception traceback
        max_message_length (Optional[int]): Maximum length for log messages
        field_mapping (Dict[str, str]): Mapping of standard fields to custom names
    """

    # Default fields to include in JSON output
    DEFAULT_FIELDS = [
        "timestamp",
        "level",
        "logger_name",
        "module",
        "function",
        "line_number",
        "message",
    ]

    # Valid field names and their corresponding LogRecord attributes
    FIELD_MAPPING = {
        "timestamp": "created",
        "level": "levelname",
        "level_number": "levelno",
        "logger_name": "name",
        "module": "module",
        "function": "funcName",
        "line_number": "lineno",
        "message": "getMessage",
        "pathname": "pathname",
        "filename": "filename",
        "thread": "thread",
        "thread_name": "threadName",
        "process": "process",
        "process_name": "processName",
        "exception": "exc_info",
        "stack_info": "stack_info",
        "extra": None,  # Special handling for extra fields
    }

    def __init__(
        self,
        fields: Optional[List[str]] = None,
        pretty_print: bool = False,
        custom_fields: Optional[Dict[str, Any]] = None,
        timestamp_format: str = "iso",
        use_utc: bool = False,
        include_exception_traceback: bool = True,
        max_message_length: Optional[int] = None,
        field_mapping: Optional[Dict[str, str]] = None,
        ensure_ascii: bool = False,
    ):
        """
        Initialize the JSONFormatter.

        Args:
            fields: List of fields to include in JSON output (defaults to DEFAULT_FIELDS)
            pretty_print: Whether to pretty-print JSON output
            custom_fields: Static custom fields to include in every log entry
            timestamp_format: Format for timestamp ("iso", "epoch", or strftime format)
            use_utc: Whether to use UTC for timestamps
            include_exception_traceback: Whether to include full exception traceback
            max_message_length: Maximum length for log messages (None for no limit)
            field_mapping: Custom mapping of field names to output names
            ensure_ascii: Whether to ensure ASCII-only output

        Raises:
            JSONFormatterError: If configuration is invalid
        """
        super().__init__()

        # Validate and set fields
        self.fields = fields or self.DEFAULT_FIELDS.copy()
        self._validate_fields()

        self.pretty_print = pretty_print
        self.custom_fields = custom_fields or {}
        self.timestamp_format = timestamp_format
        self.use_utc = use_utc
        self.include_exception_traceback = include_exception_traceback
        self.max_message_length = max_message_length
        self.field_mapping = field_mapping or {}
        self.ensure_ascii = ensure_ascii

        # Thread safety
        self._lock = threading.RLock()

        # Validate configuration
        self._validate_configuration()

    def _validate_fields(self) -> None:
        """
        Validate the fields configuration.

        Raises:
            JSONFormatterError: If fields configuration is invalid
        """
        if not isinstance(self.fields, list):
            raise JSONFormatterError("Fields must be a list")

        invalid_fields = []
        for field in self.fields:
            if not isinstance(field, str):
                invalid_fields.append(f"Non-string field: {field}")
            elif field not in self.FIELD_MAPPING and field != "custom_fields":
                invalid_fields.append(f"Unknown field: {field}")

        if invalid_fields:
            raise JSONFormatterError(
                f"Invalid fields: {', '.join(invalid_fields)}. "
                f"Valid fields: {', '.join(self.FIELD_MAPPING.keys())}, custom_fields"
            )

    def _validate_configuration(self) -> None:
        """
        Validate the formatter configuration.

        Raises:
            JSONFormatterError: If configuration is invalid
        """
        # Validate timestamp format
        if not isinstance(self.timestamp_format, str):
            raise JSONFormatterError("timestamp_format must be a string")

        if self.timestamp_format not in ["iso", "epoch"] and not self.timestamp_format.startswith("%"):
            raise JSONFormatterError(
                "timestamp_format must be 'iso', 'epoch', or a valid strftime format"
            )

        # Validate custom fields
        if not isinstance(self.custom_fields, dict):
            raise JSONFormatterError("custom_fields must be a dictionary")

        # Validate max_message_length
        if self.max_message_length is not None:
            if not isinstance(self.max_message_length, int) or self.max_message_length < 1:
                raise JSONFormatterError("max_message_length must be a positive integer or None")

        # Validate field_mapping
        if not isinstance(self.field_mapping, dict):
            raise JSONFormatterError("field_mapping must be a dictionary")

    def format(self, record: logging.LogRecord) -> str:
        """
        Format the log record as a JSON string.

        Args:
            record: The log record to format

        Returns:
            str: JSON-formatted log entry

        Raises:
            JSONFormatterError: If formatting fails
        """
        with self._lock:
            try:
                # Build the JSON data structure
                json_data = self._build_json_data(record)

                # Serialize to JSON
                return self._serialize_json(json_data)

            except Exception as e:
                # If JSON formatting fails, fall back to a minimal safe format
                fallback_data = {
                    "timestamp": self._format_timestamp(record.created),
                    "level": "ERROR",
                    "logger_name": "json_formatter",
                    "message": f"JSON formatting failed: {str(e)}",
                    "original_message": getattr(record, "getMessage", lambda: str(record.msg))(),
                    "error": str(e),
                }

                try:
                    return self._serialize_json(fallback_data)
                except Exception:
                    # Ultimate fallback - return a simple string
                    return json.dumps({
                        "error": "JSON formatting completely failed",
                        "message": str(record.msg) if hasattr(record, "msg") else "Unknown message",
                        "timestamp": datetime.now().isoformat(),
                    })

    def _build_json_data(self, record: logging.LogRecord) -> Dict[str, Any]:
        """
        Build the JSON data structure from the log record.

        Args:
            record: The log record to process

        Returns:
            Dict[str, Any]: JSON data structure
        """
        json_data = {}

        # Process each configured field
        for field in self.fields:
            try:
                value = self._extract_field_value(field, record)
                if value is not None or field in ["message"]:  # Always include message even if None
                    output_field_name = self.field_mapping.get(field, field)
                    json_data[output_field_name] = value
            except Exception as e:
                # Log field extraction error but continue with other fields
                json_data[f"{field}_error"] = f"Field extraction failed: {str(e)}"

        # Add custom fields if requested
        if "custom_fields" in self.fields and self.custom_fields:
            custom_output_name = self.field_mapping.get("custom_fields", "custom_fields")
            json_data[custom_output_name] = self._safe_serialize_value(self.custom_fields)

        # Add any extra fields from the record
        if hasattr(record, "__dict__"):
            extra_fields = self._extract_extra_fields(record)
            if extra_fields:
                extra_output_name = self.field_mapping.get("extra", "extra")
                json_data[extra_output_name] = extra_fields

        return json_data

    def _extract_field_value(self, field: str, record: logging.LogRecord) -> Any:
        """
        Extract a field value from the log record.

        Args:
            field: Field name to extract
            record: Log record to extract from

        Returns:
            Any: Extracted field value
        """
        if field == "timestamp":
            return self._format_timestamp(record.created)

        elif field == "message":
            return self._format_message(record)

        elif field == "exception":
            return self._format_exception(record)

        elif field == "custom_fields":
            # Handled separately in _build_json_data
            return None

        elif field in self.FIELD_MAPPING:
            attr_name = self.FIELD_MAPPING[field]
            if attr_name == "getMessage":
                return self._safe_get_message(record)
            else:
                return self._safe_get_attribute(record, attr_name)

        else:
            # Unknown field - try to get it as an attribute
            return self._safe_get_attribute(record, field)

    def _format_timestamp(self, created: float) -> str:
        """
        Format timestamp according to configuration.

        Args:
            created: Timestamp from log record

        Returns:
            str: Formatted timestamp
        """
        try:
            if self.timestamp_format == "epoch":
                return str(created)

            # Convert to datetime
            if self.use_utc:
                dt = datetime.fromtimestamp(created, tz=timezone.utc)
            else:
                dt = datetime.fromtimestamp(created)

            if self.timestamp_format == "iso":
                return dt.isoformat()
            else:
                # Custom strftime format
                return dt.strftime(self.timestamp_format)

        except Exception:
            # Fallback to ISO format with current time
            return datetime.now().isoformat()

    def _format_message(self, record: logging.LogRecord) -> str:
        """
        Format the log message with length limits.

        Args:
            record: Log record to extract message from

        Returns:
            str: Formatted message
        """
        try:
            message = self._safe_get_message(record)
            if message is None:
                return ""

            message = str(message)

            # Apply length limit if configured
            if self.max_message_length and len(message) > self.max_message_length:
                message = message[: self.max_message_length - 3] + "..."

            return message

        except Exception:
            return "Message formatting failed"

    def _format_exception(self, record: logging.LogRecord) -> Optional[Dict[str, Any]]:
        """
        Format exception information from the log record.

        Args:
            record: Log record to extract exception from

        Returns:
            Optional[Dict[str, Any]]: Exception information or None
        """
        if not record.exc_info:
            return None

        try:
            exc_type, exc_value, exc_traceback = record.exc_info
            exception_data = {
                "type": exc_type.__name__ if exc_type else None,
                "message": str(exc_value) if exc_value else None,
            }

            if self.include_exception_traceback and exc_traceback:
                exception_data["traceback"] = traceback.format_exception(
                    exc_type, exc_value, exc_traceback
                )

            return exception_data

        except Exception:
            return {"error": "Exception formatting failed"}

    def _extract_extra_fields(self, record: logging.LogRecord) -> Dict[str, Any]:
        """
        Extract extra fields from the log record.

        Args:
            record: Log record to extract from

        Returns:
            Dict[str, Any]: Extra fields
        """
        extra_fields = {}

        # Standard LogRecord attributes to exclude
        standard_attrs = {
            "name", "msg", "args", "levelname", "levelno", "pathname", "filename",
            "module", "lineno", "funcName", "created", "msecs", "relativeCreated",
            "thread", "threadName", "processName", "process", "getMessage",
            "exc_info", "exc_text", "stack_info", "message"
        }

        try:
            for key, value in record.__dict__.items():
                if key not in standard_attrs and not key.startswith("_"):
                    extra_fields[key] = self._safe_serialize_value(value)

        except Exception:
            extra_fields["extraction_error"] = "Failed to extract extra fields"

        return extra_fields

    def _safe_get_message(self, record: logging.LogRecord) -> Optional[str]:
        """
        Safely get the formatted message from a log record.

        Args:
            record: Log record to get message from

        Returns:
            Optional[str]: Formatted message or None
        """
        try:
            if hasattr(record, "getMessage"):
                return record.getMessage()
            elif hasattr(record, "msg"):
                if record.args:
                    return str(record.msg) % record.args
                else:
                    return str(record.msg)
            else:
                return None
        except Exception:
            # If message formatting fails, return the raw message
            return str(getattr(record, "msg", ""))

    def _safe_get_attribute(self, record: logging.LogRecord, attr_name: str) -> Any:
        """
        Safely get an attribute from the log record.

        Args:
            record: Log record to get attribute from
            attr_name: Name of the attribute

        Returns:
            Any: Attribute value or None if not found
        """
        try:
            value = getattr(record, attr_name, None)
            return self._safe_serialize_value(value)
        except Exception:
            return None

    def _safe_serialize_value(self, value: Any) -> Any:
        """
        Safely serialize a value for JSON output.

        Args:
            value: Value to serialize

        Returns:
            Any: JSON-serializable value
        """
        try:
            # Test if value is JSON serializable
            json.dumps(value)
            return value
        except (TypeError, ValueError):
            # Convert non-serializable values to strings
            try:
                return str(value)
            except Exception:
                return "<non-serializable value>"

    def _serialize_json(self, data: Dict[str, Any]) -> str:
        """
        Serialize data to JSON string.

        Args:
            data: Data to serialize

        Returns:
            str: JSON string

        Raises:
            JSONFormatterError: If serialization fails
        """
        try:
            if self.pretty_print:
                return json.dumps(
                    data,
                    indent=2,
                    ensure_ascii=self.ensure_ascii,
                    sort_keys=True,
                    default=str,
                )
            else:
                return json.dumps(
                    data,
                    ensure_ascii=self.ensure_ascii,
                    separators=(",", ":"),
                    default=str,
                )

        except Exception as e:
            raise JSONFormatterError(f"Failed to serialize JSON: {str(e)}", e)

    def set_fields(self, fields: List[str]) -> None:
        """
        Update the fields to include in JSON output.

        Args:
            fields: New list of fields to include

        Raises:
            JSONFormatterError: If fields are invalid
        """
        with self._lock:
            old_fields = self.fields
            self.fields = fields
            try:
                self._validate_fields()
            except JSONFormatterError:
                # Restore old fields if validation fails
                self.fields = old_fields
                raise

    def add_field(self, field: str) -> None:
        """
        Add a field to the JSON output.

        Args:
            field: Field name to add

        Raises:
            JSONFormatterError: If field is invalid
        """
        if field not in self.fields:
            with self._lock:
                new_fields = self.fields + [field]
                self.set_fields(new_fields)

    def remove_field(self, field: str) -> None:
        """
        Remove a field from the JSON output.

        Args:
            field: Field name to remove
        """
        with self._lock:
            if field in self.fields:
                self.fields = [f for f in self.fields if f != field]

    def set_custom_fields(self, custom_fields: Dict[str, Any]) -> None:
        """
        Update the custom fields to include in JSON output.

        Args:
            custom_fields: Dictionary of custom fields

        Raises:
            JSONFormatterError: If custom_fields is not a dictionary
        """
        if not isinstance(custom_fields, dict):
            raise JSONFormatterError("custom_fields must be a dictionary")

        with self._lock:
            self.custom_fields = custom_fields.copy()

    def set_pretty_print(self, pretty_print: bool) -> None:
        """
        Enable or disable pretty-printing of JSON output.

        Args:
            pretty_print: Whether to pretty-print JSON
        """
        with self._lock:
            self.pretty_print = bool(pretty_print)

    def set_timestamp_format(self, timestamp_format: str) -> None:
        """
        Set the timestamp format.

        Args:
            timestamp_format: New timestamp format

        Raises:
            JSONFormatterError: If format is invalid
        """
        if not isinstance(timestamp_format, str):
            raise JSONFormatterError("timestamp_format must be a string")

        if timestamp_format not in ["iso", "epoch"] and not timestamp_format.startswith("%"):
            raise JSONFormatterError(
                "timestamp_format must be 'iso', 'epoch', or a valid strftime format"
            )

        with self._lock:
            self.timestamp_format = timestamp_format

    def get_configuration(self) -> Dict[str, Any]:
        """
        Get the current formatter configuration.

        Returns:
            Dict[str, Any]: Current configuration
        """
        with self._lock:
            return {
                "fields": self.fields.copy(),
                "pretty_print": self.pretty_print,
                "custom_fields": self.custom_fields.copy(),
                "timestamp_format": self.timestamp_format,
                "use_utc": self.use_utc,
                "include_exception_traceback": self.include_exception_traceback,
                "max_message_length": self.max_message_length,
                "field_mapping": self.field_mapping.copy(),
                "ensure_ascii": self.ensure_ascii,
            }

    def __repr__(self) -> str:
        """
        String representation of the formatter.

        Returns:
            str: String representation
        """
        return (
            f"JSONFormatter("
            f"fields={len(self.fields)}, "
            f"pretty_print={self.pretty_print}, "
            f"timestamp_format='{self.timestamp_format}', "
            f"custom_fields={len(self.custom_fields)}"
            f")"
        )


# Factory function for easier integration
def create_json_formatter(
    fields: Optional[List[str]] = None,
    pretty_print: bool = False,
    custom_fields: Optional[Dict[str, Any]] = None,
    timestamp_format: str = "iso",
    use_utc: bool = False,
    include_exception_traceback: bool = True,
    max_message_length: Optional[int] = None,
    field_mapping: Optional[Dict[str, str]] = None,
    ensure_ascii: bool = False,
) -> JSONFormatter:
    """
    Factory function to create a JSONFormatter instance.

    Args:
        fields: List of fields to include in JSON output
        pretty_print: Whether to pretty-print JSON output
        custom_fields: Static custom fields to include
        timestamp_format: Format for timestamp field
        use_utc: Whether to use UTC for timestamps
        include_exception_traceback: Whether to include full exception traceback
        max_message_length: Maximum length for log messages
        field_mapping: Custom mapping of field names
        ensure_ascii: Whether to ensure ASCII-only output

    Returns:
        JSONFormatter: Configured formatter instance

    Raises:
        JSONFormatterError: If formatter creation fails

    Examples:
        # Basic usage
        formatter = create_json_formatter()

        # Pretty-printed with custom fields
        formatter = create_json_formatter(
            pretty_print=True,
            custom_fields={"service": "web-api", "version": "1.0.0"}
        )

        # Compact with specific fields
        formatter = create_json_formatter(
            fields=["timestamp", "level", "message"],
            timestamp_format="epoch"
        )
    """
    try:
        return JSONFormatter(
            fields=fields,
            pretty_print=pretty_print,
            custom_fields=custom_fields,
            timestamp_format=timestamp_format,
            use_utc=use_utc,
            include_exception_traceback=include_exception_traceback,
            max_message_length=max_message_length,
            field_mapping=field_mapping,
            ensure_ascii=ensure_ascii,
        )
    except Exception as e:
        raise JSONFormatterError(f"Failed to create JSON formatter: {str(e)}", e)