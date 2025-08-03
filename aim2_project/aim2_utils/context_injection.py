"""
Context Injection Module for AIM2 Project

This module provides context injection functionality for automatic inclusion of
request IDs and other contextual information into log records. It extends the
standard logging system to seamlessly integrate with the RequestContextManager
without requiring changes to existing logging calls.

Classes:
    ContextInjectingLogRecord: Enhanced LogRecord with automatic context injection
    ContextInjectingLogger: Logger class that uses context-injecting records
    ContextInjectionFilter: Logging filter that adds context to log records

Dependencies:
    - logging: For base logging functionality
    - typing: For type hints
    - .request_context: For context management
"""

import logging
from typing import Any, Dict, Optional, Callable
from .request_context import RequestContextManager, get_global_context_manager


class ContextInjectingLogRecord(logging.LogRecord):
    """
    Enhanced LogRecord that automatically injects context information.

    Extends the standard LogRecord to include context information from the
    RequestContextManager. Context is injected as extra attributes that can
    be used by formatters without modifying existing logging calls.

    Attributes:
        context_manager (RequestContextManager): Context manager instance
        _context_injected (bool): Flag to prevent duplicate injection
    """

    def __init__(self, *args, **kwargs):
        """
        Initialize the ContextInjectingLogRecord.

        Calls parent constructor and then injects context information
        from the current thread's RequestContextManager.
        """
        # Extract context manager if provided in kwargs
        self.context_manager = (
            kwargs.pop("context_manager", None) or get_global_context_manager()
        )

        # Call parent constructor
        super().__init__(*args, **kwargs)

        # Inject context
        self._context_injected = False
        self._inject_context()

    def _inject_context(self) -> None:
        """
        Inject context information into the log record.

        Adds context fields as attributes to the log record. Uses a flag
        to prevent duplicate injection.
        """
        if self._context_injected:
            return

        try:
            # Get context for logging
            context = self.context_manager.get_context_for_logging()

            # Add context fields as record attributes
            for key, value in context.items():
                # Avoid overwriting existing attributes
                if not hasattr(self, key):
                    setattr(self, key, value)

            # Store full context in a special attribute for formatters
            self.logging_context = context.copy()

            self._context_injected = True

        except Exception:
            # If context injection fails, continue without context
            # This ensures logging always works even if context has issues
            self.logging_context = {}
            self._context_injected = True


class ContextInjectingLogger(logging.Logger):
    """
    Logger class that creates context-injecting log records.

    Extends the standard Logger to use ContextInjectingLogRecord for
    automatic context injection without requiring changes to logging calls.

    Attributes:
        context_manager (RequestContextManager): Context manager instance
    """

    def __init__(
        self,
        name: str,
        level: int = logging.NOTSET,
        context_manager: Optional[RequestContextManager] = None,
    ):
        """
        Initialize the ContextInjectingLogger.

        Args:
            name: Logger name
            level: Logging level
            context_manager: Optional context manager instance
        """
        super().__init__(name, level)
        self.context_manager = context_manager or get_global_context_manager()

    def makeRecord(
        self,
        name: str,
        level: int,
        fn: str,
        lno: int,
        msg: Any,
        args: tuple,
        exc_info: Optional[Any] = None,
        func: Optional[str] = None,
        extra: Optional[Dict[str, Any]] = None,
        sinfo: Optional[str] = None,
    ) -> ContextInjectingLogRecord:
        """
        Create a ContextInjectingLogRecord instead of standard LogRecord.

        This method is called by the logging framework to create log records.
        By overriding it, we ensure all log records automatically include context.

        Args:
            name: Logger name
            level: Log level
            fn: Filename
            lno: Line number
            msg: Log message
            args: Message arguments
            exc_info: Exception information
            func: Function name
            extra: Extra fields
            sinfo: Stack information

        Returns:
            ContextInjectingLogRecord: Enhanced log record with context
        """
        # Create enhanced log record with context injection
        record = ContextInjectingLogRecord(
            name,
            level,
            fn,
            lno,
            msg,
            args,
            exc_info,
            func,
            sinfo,
            context_manager=self.context_manager,
        )

        # Add any extra fields
        if extra:
            for key, value in extra.items():
                if key in ("message", "asctime") or key in record.__dict__:
                    raise KeyError(f"Attempt to overwrite '{key}' in LogRecord")
                record.__dict__[key] = value

        return record


class ContextInjectionFilter(logging.Filter):
    """
    Logging filter that adds context information to log records.

    This filter can be used with existing loggers to add context injection
    without replacing the logger class. It modifies log records in-place
    to include context information.

    Attributes:
        context_manager (RequestContextManager): Context manager instance
        context_prefix (str): Prefix for context field names
        include_full_context (bool): Whether to include full context dict
    """

    def __init__(
        self,
        context_manager: Optional[RequestContextManager] = None,
        context_prefix: str = "",
        include_full_context: bool = True,
        name: str = "",
    ):
        """
        Initialize the ContextInjectionFilter.

        Args:
            context_manager: Context manager instance
            context_prefix: Prefix to add to context field names
            include_full_context: Whether to include full context as 'logging_context'
            name: Filter name
        """
        super().__init__(name)
        self.context_manager = context_manager or get_global_context_manager()
        self.context_prefix = context_prefix
        self.include_full_context = include_full_context

    def filter(self, record: logging.LogRecord) -> bool:
        """
        Add context information to the log record.

        Args:
            record: Log record to modify

        Returns:
            bool: Always True (don't filter out records)
        """
        try:
            # Skip if already processed by ContextInjectingLogRecord
            if hasattr(record, "_context_injected") and record._context_injected:
                return True

            # Get context for logging
            context = self.context_manager.get_context_for_logging()

            # Add context fields with optional prefix
            for key, value in context.items():
                attr_name = (
                    f"{self.context_prefix}{key}" if self.context_prefix else key
                )

                # Avoid overwriting existing attributes
                if not hasattr(record, attr_name):
                    setattr(record, attr_name, value)

            # Add full context if requested
            if self.include_full_context:
                record.logging_context = context.copy()

            # Mark as processed
            record._context_injected = True

        except Exception:
            # If context injection fails, don't filter out the record
            # This ensures logging always works
            pass

        return True


class ContextAwareFormatter(logging.Formatter):
    """
    Base formatter class that's aware of injected context.

    Provides utilities for formatters to work with injected context information.
    Can be subclassed to create custom formatters that use context data.

    Attributes:
        context_fields (list): List of context fields to include in output
        context_format (str): Format string for context fields
        include_context_in_message (bool): Whether to append context to message
    """

    def __init__(
        self,
        fmt: Optional[str] = None,
        datefmt: Optional[str] = None,
        style: str = "%",
        context_fields: Optional[list] = None,
        context_format: str = "[{key}={value}]",
        include_context_in_message: bool = False,
    ):
        """
        Initialize the ContextAwareFormatter.

        Args:
            fmt: Format string
            datefmt: Date format string
            style: Format style ('%', '{', or '$')
            context_fields: List of context fields to include
            context_format: Format string for context fields
            include_context_in_message: Whether to append context to message
        """
        super().__init__(fmt, datefmt, style)
        self.context_fields = context_fields or []
        self.context_format = context_format
        self.include_context_in_message = include_context_in_message

    def get_context_from_record(self, record: logging.LogRecord) -> Dict[str, Any]:
        """
        Extract context information from a log record.

        Args:
            record: Log record to extract context from

        Returns:
            Dict[str, Any]: Context information
        """
        # Check for full context first
        if hasattr(record, "logging_context"):
            return record.logging_context

        # Fallback to extracting individual context fields
        context = {}
        context_manager = get_global_context_manager()

        for field in context_manager.get_allowed_fields():
            if hasattr(record, field):
                context[field] = getattr(record, field)

        return context

    def format_context(self, context: Dict[str, Any]) -> str:
        """
        Format context information for display.

        Args:
            context: Context dictionary

        Returns:
            str: Formatted context string
        """
        if not context:
            return ""

        # Filter context fields if specified
        if self.context_fields:
            filtered_context = {
                k: v for k, v in context.items() if k in self.context_fields
            }
        else:
            filtered_context = context

        # Format each context field
        formatted_parts = []
        for key, value in filtered_context.items():
            if value is not None:
                formatted_part = self.context_format.format(key=key, value=value)
                formatted_parts.append(formatted_part)

        return " ".join(formatted_parts)

    def format(self, record: logging.LogRecord) -> str:
        """
        Format the log record with context awareness.

        Args:
            record: Log record to format

        Returns:
            str: Formatted log message
        """
        # Get base formatted message
        base_message = super().format(record)

        # Add context to message if requested
        if self.include_context_in_message:
            context = self.get_context_from_record(record)
            context_str = self.format_context(context)

            if context_str:
                base_message = f"{base_message} {context_str}"

        return base_message


# Utility functions for easy integration
def install_context_injection(
    logger: Optional[logging.Logger] = None,
    context_manager: Optional[RequestContextManager] = None,
    method: str = "filter",
) -> None:
    """
    Install context injection on a logger.

    Args:
        logger: Logger to install on. If None, uses root logger.
        context_manager: Context manager to use
        method: Installation method ("filter" or "replace")

    Examples:
        >>> logger = logging.getLogger("my_app")
        >>> install_context_injection(logger)
        >>> # Now all logs from logger will include context
    """
    if logger is None:
        logger = logging.getLogger()

    context_manager = context_manager or get_global_context_manager()

    if method == "filter":
        # Add context injection filter
        context_filter = ContextInjectionFilter(context_manager)
        logger.addFilter(context_filter)

    elif method == "replace":
        # Replace logger class (more invasive but comprehensive)
        logger.__class__ = ContextInjectingLogger
        logger.context_manager = context_manager

    else:
        raise ValueError(f"Unknown installation method: {method}")


def create_context_logger(
    name: str,
    level: int = logging.NOTSET,
    context_manager: Optional[RequestContextManager] = None,
) -> ContextInjectingLogger:
    """
    Create a context-injecting logger.

    Args:
        name: Logger name
        level: Logging level
        context_manager: Context manager instance

    Returns:
        ContextInjectingLogger: Logger with automatic context injection

    Examples:
        >>> logger = create_context_logger("my_app")
        >>> logger.info("This will include context automatically")
    """
    return ContextInjectingLogger(name, level, context_manager)


def add_context_to_all_loggers(
    context_manager: Optional[RequestContextManager] = None,
) -> None:
    """
    Add context injection to all existing loggers.

    This is a global operation that affects all loggers in the application.
    Use with caution.

    Args:
        context_manager: Context manager to use

    Examples:
        >>> add_context_to_all_loggers()
        >>> # All existing loggers now inject context
    """
    context_manager = context_manager or get_global_context_manager()
    context_filter = ContextInjectionFilter(context_manager)

    # Add to root logger (affects all loggers by default)
    root_logger = logging.getLogger()
    root_logger.addFilter(context_filter)


# Context injection decorators
def with_request_context(
    request_id: Optional[str] = None, **context_kwargs: Any
) -> Callable:
    """
    Decorator to wrap function execution with request context.

    Args:
        request_id: Optional request ID
        **context_kwargs: Additional context values

    Returns:
        Callable: Decorated function

    Examples:
        >>> @with_request_context(operation="data_processing")
        ... def process_data():
        ...     logger.info("Processing data")  # Will include context
    """

    def decorator(func: Callable) -> Callable:
        def wrapper(*args, **kwargs):
            context_manager = get_global_context_manager()
            with context_manager.request_context(request_id, **context_kwargs):
                return func(*args, **kwargs)

        return wrapper

    return decorator


def inject_request_id(func: Callable) -> Callable:
    """
    Decorator to automatically inject a request ID for function execution.

    Args:
        func: Function to decorate

    Returns:
        Callable: Decorated function

    Examples:
        >>> @inject_request_id
        ... def handle_request():
        ...     logger.info("Handling request")  # Will include auto-generated request ID
    """

    def wrapper(*args, **kwargs):
        context_manager = get_global_context_manager()
        with context_manager.request_context():
            return func(*args, **kwargs)

    return wrapper
