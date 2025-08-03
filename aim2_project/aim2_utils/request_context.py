"""
Request Context Manager Module for AIM2 Project

This module provides request-scoped context management for logging, allowing
automatic injection of request IDs and other contextual information into log messages.
The implementation uses thread-local storage to maintain context isolation between
concurrent requests and provides a clean API for context management.

Classes:
    RequestContextError: Custom exception for context-related errors
    RequestContextManager: Main context management class for thread-safe context handling

Dependencies:
    - threading: For thread-local storage and synchronization
    - uuid: For generating unique request IDs
    - typing: For type hints
    - contextlib: For context manager support
"""

import threading
import uuid
from typing import Any, Dict, Optional, Union, ContextManager
from contextlib import contextmanager


class RequestContextError(Exception):
    """
    Custom exception for request context-related errors.

    This exception is raised when context operations encounter errors
    such as missing context, invalid operations, or initialization failures.

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


class RequestContextManager:
    """
    Thread-safe request context manager for logging.

    Manages request-scoped context using thread-local storage, allowing automatic
    injection of request IDs and other contextual information into log messages.
    Provides context isolation between concurrent requests and supports nested
    context operations.

    Attributes:
        _local (threading.local): Thread-local storage for context data
        _lock (threading.RLock): Thread lock for safe operations
        _default_context (Dict[str, Any]): Default context values
        _auto_generate_request_id (bool): Whether to auto-generate request IDs
        _request_id_field (str): Field name for request ID in context
        _context_fields (set): Set of allowed context field names
    """

    # Default configuration
    DEFAULT_REQUEST_ID_FIELD = "request_id"
    DEFAULT_CONTEXT_FIELDS = {
        "request_id",
        "user_id",
        "session_id",
        "correlation_id",
        "trace_id",
        "operation",
        "component",
        "service",
        "version",
        "environment",
        # Performance context fields
        "perf_operation",
        "perf_duration",
        "perf_success",
        "perf_exceeded_warning",
        "perf_exceeded_critical",
        "perf_error_type",
        "perf_memory_delta_mb",
        "perf_start_time",
    }

    def __init__(
        self,
        auto_generate_request_id: bool = True,
        request_id_field: str = DEFAULT_REQUEST_ID_FIELD,
        default_context: Optional[Dict[str, Any]] = None,
        allowed_context_fields: Optional[set] = None,
    ):
        """
        Initialize the RequestContextManager.

        Args:
            auto_generate_request_id: Whether to automatically generate request IDs
            request_id_field: Field name for request ID in context
            default_context: Default context values to include
            allowed_context_fields: Set of allowed context field names
        """
        self._local = threading.local()
        self._lock = threading.RLock()
        self._auto_generate_request_id = auto_generate_request_id
        self._request_id_field = request_id_field
        self._default_context = default_context or {}
        self._context_fields = (
            allowed_context_fields or self.DEFAULT_CONTEXT_FIELDS.copy()
        )

        # Ensure request_id_field is in allowed fields
        self._context_fields.add(self._request_id_field)

    def _get_context_dict(self) -> Dict[str, Any]:
        """
        Get the current thread's context dictionary.

        Returns:
            Dict[str, Any]: Current context dictionary
        """
        if not hasattr(self._local, "context"):
            self._local.context = {}
        return self._local.context

    def _get_context_stack(self) -> list:
        """
        Get the current thread's context stack for nested contexts.

        Returns:
            list: Context stack for nested operations
        """
        if not hasattr(self._local, "context_stack"):
            self._local.context_stack = []
        return self._local.context_stack

    def generate_request_id(self) -> str:
        """
        Generate a unique request ID.

        Returns:
            str: Unique request ID string

        Examples:
            >>> manager = RequestContextManager()
            >>> req_id = manager.generate_request_id()
            >>> len(req_id) > 0
            True
        """
        return str(uuid.uuid4())

    def set_context(self, **kwargs: Any) -> None:
        """
        Set context values for the current thread.

        Args:
            **kwargs: Context key-value pairs to set

        Raises:
            RequestContextError: If context field is not allowed

        Examples:
            >>> manager = RequestContextManager()
            >>> manager.set_context(request_id="123", user_id="user456")
            >>> manager.get_context("request_id")
            '123'
        """
        with self._lock:
            context = self._get_context_dict()

            # Validate context fields
            for key in kwargs:
                if key not in self._context_fields:
                    raise RequestContextError(
                        f"Context field '{key}' is not allowed. "
                        f"Allowed fields: {sorted(self._context_fields)}"
                    )

            # Update context
            context.update(kwargs)

            # Auto-generate request ID if enabled and not provided
            if self._auto_generate_request_id and self._request_id_field not in context:
                context[self._request_id_field] = self.generate_request_id()

    def get_context(self, key: Optional[str] = None) -> Union[Any, Dict[str, Any]]:
        """
        Get context value(s) for the current thread.

        Args:
            key: Specific context key to retrieve. If None, returns entire context.

        Returns:
            Union[Any, Dict[str, Any]]: Context value or entire context dictionary

        Examples:
            >>> manager = RequestContextManager()
            >>> manager.set_context(request_id="123")
            >>> manager.get_context("request_id")
            '123'
            >>> isinstance(manager.get_context(), dict)
            True
        """
        with self._lock:
            context = self._get_context_dict()

            # Include default context values
            full_context = self._default_context.copy()
            full_context.update(context)

            if key is None:
                return full_context
            return full_context.get(key)

    def clear_context(self) -> None:
        """
        Clear all context for the current thread.

        Examples:
            >>> manager = RequestContextManager()
            >>> manager.set_context(request_id="123")
            >>> manager.clear_context()
            >>> manager.get_context()
            {}
        """
        with self._lock:
            context = self._get_context_dict()
            context.clear()

    def update_context(self, updates: Dict[str, Any]) -> None:
        """
        Update context with multiple values.

        Args:
            updates: Dictionary of context updates

        Raises:
            RequestContextError: If any context field is not allowed

        Examples:
            >>> manager = RequestContextManager()
            >>> manager.update_context({"request_id": "123", "user_id": "user456"})
            >>> manager.get_context("user_id")
            'user456'
        """
        if not isinstance(updates, dict):
            raise RequestContextError("Context updates must be a dictionary")

        self.set_context(**updates)

    def remove_context(self, *keys: str) -> None:
        """
        Remove specific context keys.

        Args:
            *keys: Context keys to remove

        Examples:
            >>> manager = RequestContextManager()
            >>> manager.set_context(request_id="123", user_id="user456")
            >>> manager.remove_context("user_id")
            >>> manager.get_context("user_id") is None
            True
        """
        with self._lock:
            context = self._get_context_dict()
            for key in keys:
                context.pop(key, None)

    def has_context(self, key: str) -> bool:
        """
        Check if a context key exists.

        Args:
            key: Context key to check

        Returns:
            bool: True if key exists in context

        Examples:
            >>> manager = RequestContextManager()
            >>> manager.set_context(request_id="123")
            >>> manager.has_context("request_id")
            True
            >>> manager.has_context("nonexistent")
            False
        """
        context = self.get_context()
        return key in context

    def get_request_id(self) -> Optional[str]:
        """
        Get the current request ID.

        Returns:
            Optional[str]: Current request ID or None if not set

        Examples:
            >>> manager = RequestContextManager()
            >>> manager.set_context(request_id="123")
            >>> manager.get_request_id()
            '123'
        """
        return self.get_context(self._request_id_field)

    def set_request_id(self, request_id: str) -> None:
        """
        Set the request ID for the current thread.

        Args:
            request_id: Request ID to set

        Examples:
            >>> manager = RequestContextManager()
            >>> manager.set_request_id("custom-123")
            >>> manager.get_request_id()
            'custom-123'
        """
        self.set_context(**{self._request_id_field: request_id})

    @contextmanager
    def request_context(
        self, request_id: Optional[str] = None, **context_kwargs: Any
    ) -> ContextManager[str]:
        """
        Context manager for request-scoped context.

        Automatically manages context lifecycle and provides isolation
        for nested operations. Restores previous context on exit.

        Args:
            request_id: Optional request ID. If None, auto-generates if enabled.
            **context_kwargs: Additional context values

        Yields:
            str: The request ID for this context

        Examples:
            >>> manager = RequestContextManager()
            >>> with manager.request_context(user_id="user123") as req_id:
            ...     print(f"Request ID: {req_id}")
            ...     print(f"User ID: {manager.get_context('user_id')}")
        """
        with self._lock:
            # Save current context
            current_context = self._get_context_dict().copy()
            context_stack = self._get_context_stack()
            context_stack.append(current_context)

            try:
                # Set new context
                new_context = context_kwargs.copy()

                if request_id is not None:
                    new_context[self._request_id_field] = request_id
                elif self._auto_generate_request_id:
                    new_context[self._request_id_field] = self.generate_request_id()

                self.set_context(**new_context)

                # Yield the request ID
                yield self.get_request_id()

            finally:
                # Restore previous context
                self._get_context_dict().clear()
                if context_stack:
                    restored_context = context_stack.pop()
                    self._get_context_dict().update(restored_context)

    def get_context_for_logging(self) -> Dict[str, Any]:
        """
        Get context formatted for logging injection.

        Returns a copy of the current context that's safe to inject into
        log records without affecting the original context.

        Returns:
            Dict[str, Any]: Context dictionary suitable for logging

        Examples:
            >>> manager = RequestContextManager()
            >>> manager.set_context(request_id="123", user_id="user456")
            >>> context = manager.get_context_for_logging()
            >>> "request_id" in context
            True
        """
        context = self.get_context()
        if not isinstance(context, dict):
            return {}

        # Filter out None values and ensure all values are serializable
        logging_context = {}
        for key, value in context.items():
            if value is not None:
                try:
                    # Test serializability (for JSON logging)
                    import json

                    json.dumps(value)
                    logging_context[key] = value
                except (TypeError, ValueError):
                    # Convert non-serializable values to strings
                    logging_context[key] = str(value)

        return logging_context

    def add_context_field(self, field_name: str) -> None:
        """
        Add a new allowed context field.

        Args:
            field_name: Name of the context field to allow

        Examples:
            >>> manager = RequestContextManager()
            >>> manager.add_context_field("custom_field")
            >>> "custom_field" in manager._context_fields
            True
        """
        with self._lock:
            self._context_fields.add(field_name)

    def remove_context_field(self, field_name: str) -> None:
        """
        Remove an allowed context field.

        Args:
            field_name: Name of the context field to disallow

        Raises:
            RequestContextError: If trying to remove required fields

        Examples:
            >>> manager = RequestContextManager()
            >>> manager.add_context_field("custom_field")
            >>> manager.remove_context_field("custom_field")
            >>> "custom_field" not in manager._context_fields
            True
        """
        if field_name == self._request_id_field:
            raise RequestContextError(
                f"Cannot remove required request ID field '{field_name}'"
            )

        with self._lock:
            self._context_fields.discard(field_name)

    def get_allowed_fields(self) -> set:
        """
        Get the set of allowed context fields.

        Returns:
            set: Set of allowed context field names

        Examples:
            >>> manager = RequestContextManager()
            >>> fields = manager.get_allowed_fields()
            >>> "request_id" in fields
            True
        """
        return self._context_fields.copy()

    def get_context_info(self) -> Dict[str, Any]:
        """
        Get information about the current context manager state.

        Returns:
            Dict[str, Any]: Context manager information

        Examples:
            >>> manager = RequestContextManager()
            >>> info = manager.get_context_info()
            >>> "auto_generate_request_id" in info
            True
        """
        with self._lock:
            context = self._get_context_dict()
            context_stack = self._get_context_stack()

            return {
                "auto_generate_request_id": self._auto_generate_request_id,
                "request_id_field": self._request_id_field,
                "current_context": context.copy(),
                "context_stack_depth": len(context_stack),
                "allowed_fields": sorted(self._context_fields),
                "default_context": self._default_context.copy(),
                "has_request_id": self._request_id_field in context,
                "current_request_id": context.get(self._request_id_field),
            }

    def reset_context_manager(self) -> None:
        """
        Reset the context manager to initial state.

        Clears all context and context stack for the current thread.
        Useful for testing or cleanup scenarios.

        Examples:
            >>> manager = RequestContextManager()
            >>> manager.set_context(request_id="123")
            >>> manager.reset_context_manager()
            >>> manager.get_context()
            {}
        """
        with self._lock:
            if hasattr(self._local, "context"):
                self._local.context.clear()
            if hasattr(self._local, "context_stack"):
                self._local.context_stack.clear()


# Global singleton instance for convenience
_global_context_manager: Optional[RequestContextManager] = None
_global_manager_lock = threading.RLock()


def get_global_context_manager() -> RequestContextManager:
    """
    Get the global RequestContextManager instance.

    Creates a singleton instance on first call. This provides a convenient
    way to access context management without explicit instantiation.

    Returns:
        RequestContextManager: Global context manager instance

    Examples:
        >>> manager = get_global_context_manager()
        >>> isinstance(manager, RequestContextManager)
        True
    """
    global _global_context_manager

    if _global_context_manager is None:
        with _global_manager_lock:
            if _global_context_manager is None:
                _global_context_manager = RequestContextManager()

    return _global_context_manager


def set_global_context_manager(manager: RequestContextManager) -> None:
    """
    Set a custom global RequestContextManager instance.

    Args:
        manager: RequestContextManager instance to use globally

    Raises:
        RequestContextError: If manager is not a RequestContextManager instance

    Examples:
        >>> custom_manager = RequestContextManager(auto_generate_request_id=False)
        >>> set_global_context_manager(custom_manager)
        >>> get_global_context_manager() is custom_manager
        True
    """
    global _global_context_manager

    if not isinstance(manager, RequestContextManager):
        raise RequestContextError(
            "Global context manager must be a RequestContextManager instance"
        )

    with _global_manager_lock:
        _global_context_manager = manager


def reset_global_context_manager() -> None:
    """
    Reset the global context manager to None.

    Forces creation of a new instance on next access. Useful for testing.

    Examples:
        >>> reset_global_context_manager()
        >>> # Next call to get_global_context_manager() will create new instance
    """
    global _global_context_manager

    with _global_manager_lock:
        _global_context_manager = None


# Convenience functions for common operations
def set_request_context(**kwargs: Any) -> None:
    """
    Set context using the global context manager.

    Args:
        **kwargs: Context key-value pairs

    Examples:
        >>> set_request_context(request_id="123", user_id="user456")
    """
    get_global_context_manager().set_context(**kwargs)


def get_request_context(key: Optional[str] = None) -> Union[Any, Dict[str, Any]]:
    """
    Get context using the global context manager.

    Args:
        key: Context key to retrieve

    Returns:
        Union[Any, Dict[str, Any]]: Context value or entire context

    Examples:
        >>> set_request_context(request_id="123")
        >>> get_request_context("request_id")
        '123'
    """
    return get_global_context_manager().get_context(key)


def clear_request_context() -> None:
    """
    Clear context using the global context manager.

    Examples:
        >>> clear_request_context()
    """
    get_global_context_manager().clear_context()


def get_current_request_id() -> Optional[str]:
    """
    Get the current request ID using the global context manager.

    Returns:
        Optional[str]: Current request ID

    Examples:
        >>> set_request_context(request_id="123")
        >>> get_current_request_id()
        '123'
    """
    return get_global_context_manager().get_request_id()


def request_context(
    request_id: Optional[str] = None, **context_kwargs: Any
) -> ContextManager[str]:
    """
    Context manager for request-scoped context using global manager.

    Args:
        request_id: Optional request ID
        **context_kwargs: Additional context values

    Returns:
        ContextManager[str]: Context manager yielding request ID

    Examples:
        >>> with request_context(user_id="user123") as req_id:
        ...     print(f"Request: {req_id}")
    """
    return get_global_context_manager().request_context(request_id, **context_kwargs)
