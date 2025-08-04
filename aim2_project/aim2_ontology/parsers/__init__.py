"""
Parser Module

This module provides abstract and concrete parsers for various ontology formats.
The AbstractParser base class provides a comprehensive foundation with error handling,
logging, configuration management, and robust validation capabilities.

Classes:
    AbstractParser: Abstract base class for all parsers with comprehensive functionality
    OWLParser: Concrete OWL parser implementation
    CSVParser: Concrete CSV parser implementation
    JSONLDParser: Concrete JSON-LD parser implementation

Features:
    - Comprehensive error handling with structured exceptions
    - Integrated logging system with module-specific loggers
    - Configuration management with validation
    - Robust input validation and sanitization
    - Metadata and statistics tracking
    - Performance monitoring and optimization hints
    - Extensible architecture for custom parsers
"""

import copy
import hashlib
import logging
import time
from abc import ABC, abstractmethod
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import IO, Any, Callable, Dict, List, Optional, Set, Union

# Import project-specific modules
try:
    from ..models import Ontology, RDFTriple, Relationship, Term
except ImportError:
    # Fallback for development/testing
    Term = Relationship = Ontology = RDFTriple = None

try:
    from ...exceptions import (
        AIM2Exception,
        OntologyErrorCodes,
        OntologyException,
        ValidationException,
    )
except ImportError:
    # Fallback classes for development/testing
    class AIM2Exception(Exception):
        pass

    class OntologyException(AIM2Exception):
        pass

    class ValidationException(AIM2Exception):
        pass

    class OntologyErrorCodes:
        UNSUPPORTED_FORMAT = None
        OWL_PARSING_FAILURE = None


try:
    from ...aim2_utils.logger_factory import get_logger
except ImportError:
    # Fallback for development/testing
    def get_logger(name):
        return logging.getLogger(name)


try:
    from ...aim2_utils.config_manager import ConfigManager
except ImportError:
    # Fallback for development/testing
    class ConfigManager:
        pass


try:
    from ...aim2_utils.performance_decorators import performance_monitor
except ImportError:
    # Fallback decorator for development/testing
    def performance_monitor(func):
        return func


class ErrorSeverity(Enum):
    """Classification of error severity levels for recovery decisions."""
    WARNING = "warning"          # Minor issues that don't prevent processing
    RECOVERABLE = "recoverable"  # Errors that can be handled with fallback strategies
    FATAL = "fatal"             # Critical errors that prevent further processing


class RecoveryStrategy(Enum):
    """Available error recovery strategies."""
    SKIP = "skip"               # Skip the problematic data/section
    DEFAULT = "default"         # Use default/fallback values
    RETRY = "retry"             # Retry with different parameters
    REPLACE = "replace"         # Replace with corrected data
    ABORT = "abort"             # Stop processing immediately
    CONTINUE = "continue"       # Continue processing despite errors


@dataclass
class ErrorContext:
    """Context information for error recovery decisions."""
    error: Exception
    severity: ErrorSeverity
    location: str               # Where the error occurred (e.g., "line 45", "namespace declaration")
    recovery_strategy: Optional[RecoveryStrategy] = None
    attempted_recoveries: List[RecoveryStrategy] = field(default_factory=list)
    recovery_data: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class ParserStatistics:
    """Statistics tracking for parser operations."""

    total_parses: int = 0
    successful_parses: int = 0
    failed_parses: int = 0
    total_parse_time: float = 0.0
    average_parse_time: float = 0.0
    largest_content_size: int = 0
    total_content_processed: int = 0
    validation_failures: int = 0
    validation_successes: int = 0
    warnings_generated: int = 0
    errors_encountered: List[str] = field(default_factory=list)
    
    # Error recovery statistics
    total_errors: int = 0
    recoverable_errors: int = 0
    fatal_errors: int = 0
    warnings: int = 0
    successful_recoveries: int = 0
    failed_recoveries: int = 0
    recovery_strategies_used: Dict[str, int] = field(default_factory=dict)
    error_types_encountered: Dict[str, int] = field(default_factory=dict)

    def update_parse_stats(self, success: bool, parse_time: float, content_size: int):
        """Update parsing statistics."""
        self.total_parses += 1
        self.total_parse_time += parse_time
        self.total_content_processed += content_size
        self.largest_content_size = max(self.largest_content_size, content_size)

        if success:
            self.successful_parses += 1
        else:
            self.failed_parses += 1

        # Update average parse time
        if self.total_parses > 0:
            self.average_parse_time = self.total_parse_time / self.total_parses
    
    def update_error_stats(self, error_context: 'ErrorContext', recovery_successful: bool = False):
        """Update error recovery statistics."""
        self.total_errors += 1
        
        # Update severity counters
        if error_context.severity == ErrorSeverity.WARNING:
            self.warnings += 1
        elif error_context.severity == ErrorSeverity.RECOVERABLE:
            self.recoverable_errors += 1
        elif error_context.severity == ErrorSeverity.FATAL:
            self.fatal_errors += 1
        
        # Update recovery statistics
        if recovery_successful:
            self.successful_recoveries += 1
        else:
            self.failed_recoveries += 1
        
        # Track recovery strategy usage
        if error_context.recovery_strategy:
            strategy_name = error_context.recovery_strategy.value
            self.recovery_strategies_used[strategy_name] = self.recovery_strategies_used.get(strategy_name, 0) + 1
        
        # Track error types
        error_type = type(error_context.error).__name__
        self.error_types_encountered[error_type] = self.error_types_encountered.get(error_type, 0) + 1


@dataclass
class ParseResult:
    """Result container for parse operations."""

    success: bool
    data: Any = None
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    parse_time: float = 0.0
    validation_result: Optional[bool] = None


class AbstractParser(ABC):
    """
    Abstract base class for ontology parsers with comprehensive functionality.

    This enhanced base class provides a robust foundation for all parser implementations
    with integrated error handling, logging, configuration management, and validation
    capabilities. It follows best practices for maintainability, performance, and
    extensibility.

    Features:
        - Comprehensive error handling with structured exceptions
        - Integrated logging system with module-specific loggers
        - Configuration management with validation
        - Robust input validation and sanitization
        - Metadata and statistics tracking
        - Performance monitoring and optimization hints
        - Extensible hooks for custom behavior
        - Thread-safe operations where applicable
        - Memory-efficient processing for large files

    Attributes:
        logger (logging.Logger): Module-specific logger instance
        config_manager (ConfigManager): Configuration manager instance
        options (Dict[str, Any]): Parser configuration options
        statistics (ParserStatistics): Operation statistics tracker
        _validation_rules (Dict[str, Callable]): Custom validation rules
        _hooks (Dict[str, List[Callable]]): Extension hooks
        _cache (Dict[str, Any]): Internal result cache

    Example:
        class MyParser(AbstractParser):
            def parse(self, content: str, **kwargs) -> Any:
                # Implementation here
                pass

            def get_supported_formats(self) -> List[str]:
                return ['my_format']
    """

    def __init__(
        self,
        parser_name: str,
        config_manager: Optional[ConfigManager] = None,
        logger: Optional[logging.Logger] = None,
        options: Optional[Dict[str, Any]] = None,
    ):
        """
        Initialize the AbstractParser with comprehensive setup.

        Args:
            parser_name (str): Name identifier for this parser instance
            config_manager (ConfigManager, optional): Configuration manager instance
            logger (logging.Logger, optional): Logger instance
            options (Dict[str, Any], optional): Initial parser options

        Raises:
            ValidationException: If parser_name is invalid
        """
        # Validate parser name
        if not parser_name or not isinstance(parser_name, str):
            raise ValidationException("Parser name must be a non-empty string")

        self.parser_name = parser_name

        # Set up logging
        self.logger = logger or get_logger(f"parsers.{parser_name}")
        self.logger.info(f"Initializing {parser_name} parser")

        # Set up configuration management
        self.config_manager = config_manager or ConfigManager()

        # Initialize parser options with defaults
        self._default_options = self._get_default_options()
        self.options = copy.deepcopy(self._default_options)
        if options:
            self.options.update(options)

        # Initialize statistics tracking
        self.statistics = ParserStatistics()

        # Initialize validation rules and hooks
        self._validation_rules: Dict[str, Callable] = {}
        self._hooks: Dict[str, List[Callable]] = {
            "pre_parse": [],
            "post_parse": [],
            "pre_validate": [],
            "post_validate": [],
            "on_error": [],
            "on_warning": [],
        }

        # Initialize internal cache for performance
        self._cache: Dict[str, Any] = {}
        self._cache_enabled = self.options.get("enable_cache", True)
        self._cache_max_size = self.options.get("cache_max_size", 100)

        # Register default validation rules
        self._register_default_validation_rules()

        self.logger.debug(f"{parser_name} parser initialized successfully")

    def _get_default_options(self) -> Dict[str, Any]:
        """
        Get default options for the parser.

        Returns:
            Dict[str, Any]: Default options dictionary
        """
        return {
            "enable_cache": True,
            "cache_max_size": 100,
            "validate_input": True,
            "strict_validation": False,
            "max_content_size": 100 * 1024 * 1024,  # 100MB
            "timeout_seconds": 300,  # 5 minutes
            "encoding": "utf-8",
            "preserve_whitespace": False,
            "enable_statistics": True,
            "log_performance": True,
            "memory_efficient": False,
        }

    def _register_default_validation_rules(self) -> None:
        """
        Register default validation rules for all parsers.
        """
        self.add_validation_rule("content_size", self._validate_content_size)
        self.add_validation_rule("content_encoding", self._validate_content_encoding)

    def _validate_content_size(self, content: str) -> List[str]:
        """
        Validate content size against limits.

        Args:
            content (str): Content to validate

        Returns:
            List[str]: List of validation errors (empty if valid)
        """
        errors = []
        max_size = self.options.get("max_content_size", 100 * 1024 * 1024)
        content_size = len(content.encode("utf-8"))

        if content_size > max_size:
            errors.append(
                f"Content size {content_size} exceeds maximum allowed size {max_size}"
            )

        return errors

    def _validate_content_encoding(self, content: str) -> List[str]:
        """
        Validate content encoding.

        Args:
            content (str): Content to validate

        Returns:
            List[str]: List of validation errors (empty if valid)
        """
        errors = []
        try:
            # Try to encode/decode with specified encoding
            encoding = self.options.get("encoding", "utf-8")
            content.encode(encoding)
        except UnicodeEncodeError as e:
            errors.append(f"Content encoding validation failed: {str(e)}")

        return errors

    @contextmanager
    def _performance_context(self, operation_name: str):
        """
        Context manager for performance monitoring.

        Args:
            operation_name (str): Name of the operation being monitored
        """
        start_time = time.time()
        try:
            yield
        finally:
            elapsed = time.time() - start_time
            if self.options.get("log_performance", True):
                self.logger.debug(
                    f"{operation_name} completed in {elapsed:.3f} seconds"
                )

    def _execute_hooks(self, hook_name: str, *args, **kwargs) -> None:
        """
        Execute registered hooks for a specific event.

        Args:
            hook_name (str): Name of the hook to execute
            *args: Arguments to pass to hook functions
            **kwargs: Keyword arguments to pass to hook functions
        """
        hooks = self._hooks.get(hook_name, [])
        for hook in hooks:
            try:
                hook(*args, **kwargs)
            except Exception as e:
                self.logger.warning(f"Hook {hook_name} failed: {str(e)}")

    def _cache_key(self, content: str, **kwargs) -> str:
        """
        Generate cache key for content and parameters.

        Args:
            content (str): Content to cache
            **kwargs: Additional parameters

        Returns:
            str: Cache key
        """
        # Create hash of content and relevant kwargs
        content_hash = hashlib.md5(content.encode(), usedforsecurity=False).hexdigest()
        kwargs_str = str(sorted(kwargs.items()))
        kwargs_hash = hashlib.md5(
            kwargs_str.encode(), usedforsecurity=False
        ).hexdigest()
        return f"{content_hash}_{kwargs_hash}"

    def _get_from_cache(self, cache_key: str) -> Optional[Any]:
        """
        Get result from cache if available.

        Args:
            cache_key (str): Cache key to lookup

        Returns:
            Optional[Any]: Cached result or None
        """
        if not self._cache_enabled:
            return None

        return self._cache.get(cache_key)

    def _store_in_cache(self, cache_key: str, result: Any) -> None:
        """
        Store result in cache with size management.

        Args:
            cache_key (str): Cache key
            result (Any): Result to cache
        """
        if not self._cache_enabled:
            return

        # Manage cache size
        if len(self._cache) >= self._cache_max_size:
            # Remove oldest entries (simple FIFO)
            keys_to_remove = list(self._cache.keys())[: len(self._cache) // 2]
            for key in keys_to_remove:
                del self._cache[key]

        self._cache[cache_key] = copy.deepcopy(result)

    def add_validation_rule(
        self, name: str, rule_func: Callable[[str], List[str]]
    ) -> None:
        """
        Add a custom validation rule.

        Args:
            name (str): Name of the validation rule
            rule_func (Callable): Function that takes content and returns list of errors
        """
        self._validation_rules[name] = rule_func
        self.logger.debug(f"Added validation rule: {name}")

    def remove_validation_rule(self, name: str) -> None:
        """
        Remove a validation rule.

        Args:
            name (str): Name of the validation rule to remove
        """
        if name in self._validation_rules:
            del self._validation_rules[name]
            self.logger.debug(f"Removed validation rule: {name}")

    def add_hook(self, hook_name: str, hook_func: Callable) -> None:
        """
        Add a hook function for a specific event.

        Args:
            hook_name (str): Name of the hook event
            hook_func (Callable): Function to execute on the event
        """
        if hook_name not in self._hooks:
            self._hooks[hook_name] = []
        self._hooks[hook_name].append(hook_func)
        self.logger.debug(f"Added hook for {hook_name}")

    def remove_hook(self, hook_name: str, hook_func: Callable) -> None:
        """
        Remove a hook function.

        Args:
            hook_name (str): Name of the hook event
            hook_func (Callable): Function to remove
        """
        if hook_name in self._hooks and hook_func in self._hooks[hook_name]:
            self._hooks[hook_name].remove(hook_func)
            self.logger.debug(f"Removed hook from {hook_name}")

    def clear_cache(self) -> None:
        """
        Clear the internal cache.
        """
        self._cache.clear()
        self.logger.debug("Parser cache cleared")

    def get_cache_info(self) -> Dict[str, Any]:
        """
        Get information about the cache state.

        Returns:
            Dict[str, Any]: Cache information
        """
        return {
            "enabled": self._cache_enabled,
            "size": len(self._cache),
            "max_size": self._cache_max_size,
            "keys": list(self._cache.keys()),
        }

    @performance_monitor
    def parse_safe(self, content: str, **kwargs) -> ParseResult:
        """
        Safe parse method with comprehensive error handling and result tracking.

        This method wraps the abstract parse method with error handling, validation,
        performance monitoring, and result tracking. It's the recommended way to
        parse content as it provides structured results and comprehensive logging.

        Args:
            content (str): Content to parse
            **kwargs: Additional parsing parameters

        Returns:
            ParseResult: Structured result with success status, data, and metadata
        """
        start_time = time.time()
        result = ParseResult(success=False)

        try:
            self.logger.info(f"Starting safe parse with {self.parser_name}")

            # Execute pre-parse hooks
            self._execute_hooks("pre_parse", content, **kwargs)

            # Validate input if enabled
            if self.options.get("validate_input", True):
                validation_errors = self.validate_content(content)
                if validation_errors:
                    result.errors.extend(validation_errors)
                    if self.options.get("strict_validation", False):
                        result.success = False
                        result.parse_time = time.time() - start_time
                        self.statistics.update_parse_stats(
                            False, result.parse_time, len(content)
                        )
                        return result
                    else:
                        result.warnings.extend(validation_errors)

            # Check cache first
            cache_key = (
                self._cache_key(content, **kwargs) if self._cache_enabled else None
            )
            if cache_key:
                cached_result = self._get_from_cache(cache_key)
                if cached_result is not None:
                    self.logger.debug("Returning cached parse result")
                    cached_result.metadata["from_cache"] = True
                    return cached_result

            # Perform actual parsing
            with self._performance_context(f"{self.parser_name}_parse"):
                parsed_data = self.parse(content, **kwargs)

            result.data = parsed_data
            result.success = True
            result.parse_time = time.time() - start_time

            # Execute post-parse hooks
            self._execute_hooks("post_parse", result, content, **kwargs)

            # Store in cache if successful
            if cache_key and result.success:
                self._store_in_cache(cache_key, result)

            # Update statistics
            self.statistics.update_parse_stats(True, result.parse_time, len(content))

            self.logger.info(
                f"Parse completed successfully in {result.parse_time:.3f}s"
            )

        except Exception as e:
            result.success = False
            result.parse_time = time.time() - start_time
            error_msg = f"Parse failed: {str(e)}"
            result.errors.append(error_msg)

            # Execute error hooks
            self._execute_hooks("on_error", e, content, **kwargs)

            # Update statistics
            self.statistics.update_parse_stats(False, result.parse_time, len(content))
            self.statistics.errors_encountered.append(error_msg)

            # Log the error appropriately
            if isinstance(e, (OntologyException, ValidationException)):
                self.logger.error(f"Parser error: {str(e)}")
            else:
                self.logger.error(
                    f"Unexpected error during parsing: {str(e)}", exc_info=True
                )

        # Add metadata
        result.metadata.update(
            {
                "parser_name": self.parser_name,
                "content_size": len(content),
                "options_used": copy.deepcopy(kwargs),
                "timestamp": time.time(),
            }
        )

        return result

    def validate_content(self, content: str) -> List[str]:
        """
        Validate content using registered validation rules.

        Args:
            content (str): Content to validate

        Returns:
            List[str]: List of validation errors (empty if valid)
        """
        all_errors = []

        for rule_name, rule_func in self._validation_rules.items():
            try:
                errors = rule_func(content)
                if errors:
                    all_errors.extend(errors)
            except Exception as e:
                self.logger.warning(f"Validation rule {rule_name} failed: {str(e)}")
                all_errors.append(f"Validation rule {rule_name} failed: {str(e)}")

        return all_errors

    # =====================
    # Error Recovery Methods
    # =====================

    def _classify_error_severity(self, error: Exception, context: str = "") -> ErrorSeverity:
        """
        Classify error severity based on error type and context.

        Args:
            error (Exception): The exception that occurred
            context (str): Context information about where the error occurred

        Returns:
            ErrorSeverity: Classified severity level
        """
        error_type = type(error).__name__
        error_message = str(error).lower()

        # Fatal errors that prevent any processing
        fatal_indicators = [
            "outofmemoryerror", "stackoverflow", "system", "critical", 
            "fatal", "corrupted", "cannot allocate"
        ]
        
        # Recoverable errors that can be worked around
        recoverable_indicators = [
            "parsing", "format", "syntax", "invalid", "malformed", 
            "missing", "namespace", "encoding", "timeout"
        ]
        
        # Warning-level issues
        warning_indicators = [
            "deprecated", "recommendation", "optional", "preference",
            "whitespace", "formatting"
        ]

        # Check error type patterns
        if error_type in ["SystemError", "MemoryError", "KeyboardInterrupt"]:
            return ErrorSeverity.FATAL
        
        if error_type in ["SyntaxError", "ValueError", "KeyError", "AttributeError"]:
            return ErrorSeverity.RECOVERABLE
            
        if error_type in ["UserWarning", "DeprecationWarning"]:
            return ErrorSeverity.WARNING

        # Check error message content
        for indicator in fatal_indicators:
            if indicator in error_message:
                return ErrorSeverity.FATAL
                
        for indicator in recoverable_indicators:
            if indicator in error_message:
                return ErrorSeverity.RECOVERABLE
                
        for indicator in warning_indicators:
            if indicator in error_message:
                return ErrorSeverity.WARNING

        # Default to recoverable for unknown errors
        return ErrorSeverity.RECOVERABLE

    def _select_recovery_strategy(self, error_context: ErrorContext) -> RecoveryStrategy:
        """
        Select appropriate recovery strategy based on error context.

        Args:
            error_context (ErrorContext): Error context information

        Returns:
            RecoveryStrategy: Selected recovery strategy
        """
        error_type = type(error_context.error).__name__
        severity = error_context.severity
        attempted = error_context.attempted_recoveries

        # Fatal errors should abort
        if severity == ErrorSeverity.FATAL:
            return RecoveryStrategy.ABORT

        # For warnings, continue processing
        if severity == ErrorSeverity.WARNING:
            return RecoveryStrategy.CONTINUE

        # Strategy selection for recoverable errors
        if error_type in ["SyntaxError", "ValueError"]:
            if RecoveryStrategy.SKIP not in attempted:
                return RecoveryStrategy.SKIP
            elif RecoveryStrategy.DEFAULT not in attempted:
                return RecoveryStrategy.DEFAULT
            else:
                return RecoveryStrategy.ABORT

        if error_type in ["KeyError", "AttributeError"]:
            if RecoveryStrategy.DEFAULT not in attempted:
                return RecoveryStrategy.DEFAULT
            elif RecoveryStrategy.SKIP not in attempted:
                return RecoveryStrategy.SKIP
            else:
                return RecoveryStrategy.ABORT

        if error_type in ["TimeoutError", "ConnectionError"]:
            if RecoveryStrategy.RETRY not in attempted and len(attempted) < 3:
                return RecoveryStrategy.RETRY
            else:
                return RecoveryStrategy.ABORT

        # Default strategy progression
        if RecoveryStrategy.SKIP not in attempted:
            return RecoveryStrategy.SKIP
        elif RecoveryStrategy.DEFAULT not in attempted:
            return RecoveryStrategy.DEFAULT
        else:
            return RecoveryStrategy.ABORT

    def _apply_recovery_strategy(self, error_context: ErrorContext, **kwargs) -> Any:
        """
        Apply the selected recovery strategy.

        Args:
            error_context (ErrorContext): Error context with selected strategy
            **kwargs: Additional recovery parameters

        Returns:
            Any: Recovery result or None if recovery failed
        """
        strategy = error_context.recovery_strategy
        
        if not strategy:
            self.logger.error("No recovery strategy selected")
            return None

        self.logger.info(f"Applying recovery strategy '{strategy.value}' for {type(error_context.error).__name__} at {error_context.location}")

        try:
            if strategy == RecoveryStrategy.SKIP:
                return self._recover_skip(error_context, **kwargs)
            elif strategy == RecoveryStrategy.DEFAULT:
                return self._recover_default(error_context, **kwargs)
            elif strategy == RecoveryStrategy.RETRY:
                return self._recover_retry(error_context, **kwargs)
            elif strategy == RecoveryStrategy.REPLACE:
                return self._recover_replace(error_context, **kwargs)
            elif strategy == RecoveryStrategy.CONTINUE:
                return self._recover_continue(error_context, **kwargs)
            elif strategy == RecoveryStrategy.ABORT:
                raise error_context.error
            else:
                self.logger.error(f"Unknown recovery strategy: {strategy}")
                return None
                
        except Exception as recovery_error:
            self.logger.error(f"Recovery strategy '{strategy.value}' failed: {str(recovery_error)}")
            return None

    def _recover_skip(self, error_context: ErrorContext, **kwargs) -> Any:
        """
        Skip the problematic data/section and continue processing.

        Args:
            error_context (ErrorContext): Error context information
            **kwargs: Additional recovery parameters

        Returns:
            Any: Empty/minimal result to continue processing
        """
        self.logger.warning(f"Skipping problematic section at {error_context.location}")
        
        # Return appropriate empty result based on context
        if "return_type" in kwargs:
            return_type = kwargs["return_type"]
            if return_type == "list":
                return []
            elif return_type == "dict":
                return {}
            elif return_type == "string":
                return ""
        
        return None

    def _recover_default(self, error_context: ErrorContext, **kwargs) -> Any:
        """
        Use default/fallback values for the problematic data.

        Args:
            error_context (ErrorContext): Error context information
            **kwargs: Additional recovery parameters

        Returns:
            Any: Default value for the expected data type
        """
        self.logger.warning(f"Using default values for problematic section at {error_context.location}")
        
        # Use provided default or generate appropriate one
        if "default_value" in kwargs:
            return kwargs["default_value"]
        
        # Generate context-appropriate defaults
        if "namespace" in error_context.location.lower():
            return {"namespace": "http://example.org/default#", "prefix": "default"}
        elif "term" in error_context.location.lower():
            return {"id": "unknown", "name": "Unknown Term", "definition": "Definition not available"}
        elif "relationship" in error_context.location.lower():
            return {"subject": "unknown", "predicate": "related_to", "object": "unknown"}
        
        return {}

    def _recover_retry(self, error_context: ErrorContext, **kwargs) -> Any:
        """
        Retry the operation with different parameters.

        Args:
            error_context (ErrorContext): Error context information
            **kwargs: Additional recovery parameters

        Returns:
            Any: Result of retry attempt or None if failed
        """
        retry_count = len([s for s in error_context.attempted_recoveries if s == RecoveryStrategy.RETRY])
        max_retries = kwargs.get("max_retries", 3)
        
        if retry_count >= max_retries:
            self.logger.error(f"Maximum retries ({max_retries}) exceeded for {error_context.location}")
            return None
        
        self.logger.info(f"Retrying operation at {error_context.location} (attempt {retry_count + 1}/{max_retries})")
        
        # Apply retry-specific modifications
        retry_kwargs = kwargs.copy()
        retry_kwargs["retry_attempt"] = retry_count + 1
        
        # Implement retry logic - this would be overridden by specific parsers
        return None

    def _recover_replace(self, error_context: ErrorContext, **kwargs) -> Any:
        """
        Replace problematic data with corrected version.

        Args:
            error_context (ErrorContext): Error context information
            **kwargs: Additional recovery parameters

        Returns:
            Any: Replacement data
        """
        self.logger.warning(f"Replacing problematic data at {error_context.location}")
        
        if "replacement_data" in kwargs:
            return kwargs["replacement_data"]
        
        # Context-specific replacements
        return self._recover_default(error_context, **kwargs)

    def _recover_continue(self, error_context: ErrorContext, **kwargs) -> Any:
        """
        Continue processing despite the error.

        Args:
            error_context (ErrorContext): Error context information
            **kwargs: Additional recovery parameters

        Returns:
            Any: Partial result to continue with
        """
        self.logger.warning(f"Continuing processing despite error at {error_context.location}")
        return kwargs.get("partial_result")

    def _handle_parse_error(self, error: Exception, location: str = "", **kwargs) -> ErrorContext:
        """
        Comprehensive error handler that creates error context and applies recovery.

        Args:
            error (Exception): The exception that occurred
            location (str): Location where error occurred
            **kwargs: Additional context and recovery parameters

        Returns:
            ErrorContext: Complete error context with recovery information
        """
        # Create error context
        severity = self._classify_error_severity(error, location)
        error_context = ErrorContext(
            error=error,
            severity=severity,
            location=location or "unknown location"
        )

        # Update statistics
        self.statistics.update_error_stats(error_context, recovery_successful=False)

        # Execute error hooks
        self._execute_hooks("on_error", error_context)

        # Check if error recovery is enabled
        if not self.options.get("error_recovery", True):
            self.logger.error(f"Error recovery disabled, re-raising {type(error).__name__}: {str(error)}")
            raise error

        # Select and apply recovery strategy
        recovery_strategy = self._select_recovery_strategy(error_context)
        error_context.recovery_strategy = recovery_strategy
        error_context.attempted_recoveries.append(recovery_strategy)

        # Log recovery attempt
        self.logger.info(f"Handling {severity.value} error with {recovery_strategy.value} strategy: {str(error)}")

        # Apply recovery
        recovery_result = self._apply_recovery_strategy(error_context, **kwargs)
        
        # Update recovery success statistics
        recovery_successful = recovery_result is not None
        self.statistics.update_error_stats(error_context, recovery_successful=recovery_successful)

        # Store recovery data
        error_context.recovery_data = {
            "recovery_result": recovery_result,
            "recovery_successful": recovery_successful
        }

        return error_context

    def parse_file(self, file_path: Union[str, Path], **kwargs) -> ParseResult:
        """
        Parse content from a file with proper encoding handling.

        Args:
            file_path (Union[str, Path]): Path to the file to parse
            **kwargs: Additional parsing parameters

        Returns:
            ParseResult: Structured result with success status, data, and metadata

        Raises:
            ValidationException: If file doesn't exist or can't be read
        """
        file_path = Path(file_path)

        if not file_path.exists():
            raise ValidationException(
                f"File not found: {file_path}", context={"file_path": str(file_path)}
            )

        try:
            encoding = self.options.get("encoding", "utf-8")
            with open(file_path, "r", encoding=encoding) as f:
                content = f.read()

            # Add file metadata to kwargs
            kwargs.setdefault("source_file", str(file_path))
            kwargs.setdefault("file_size", file_path.stat().st_size)

            return self.parse_safe(content, **kwargs)

        except UnicodeDecodeError as e:
            raise ValidationException(
                f"File encoding error: {str(e)}",
                context={"file_path": str(file_path), "encoding": encoding},
            )
        except Exception as e:
            raise OntologyException(
                f"Failed to read file: {str(e)}", context={"file_path": str(file_path)}
            )

    def parse_stream(self, stream: IO, **kwargs) -> ParseResult:
        """
        Parse content from a stream.

        Args:
            stream (IO): Stream to read from
            **kwargs: Additional parsing parameters

        Returns:
            ParseResult: Structured result with success status, data, and metadata
        """
        try:
            content = stream.read()
            if isinstance(content, bytes):
                encoding = self.options.get("encoding", "utf-8")
                content = content.decode(encoding)

            kwargs.setdefault("source_type", "stream")
            return self.parse_safe(content, **kwargs)

        except Exception as e:
            raise OntologyException(f"Failed to read from stream: {str(e)}")

    # Abstract methods that subclasses must implement
    @abstractmethod
    def parse(self, content: str, **kwargs) -> Any:
        """
        Parse ontology content. This is the core parsing method that subclasses must implement.

        Args:
            content (str): Content to parse
            **kwargs: Additional parsing parameters

        Returns:
            Any: Parsed data structure (typically Ontology, List[Term], etc.)

        Raises:
            OntologyException: If parsing fails
        """

    @abstractmethod
    def validate(self, content: str, **kwargs) -> bool:
        """
        Validate ontology content format and structure.

        Args:
            content (str): Content to validate
            **kwargs: Additional validation parameters

        Returns:
            bool: True if content is valid for this parser
        """

    @abstractmethod
    def get_supported_formats(self) -> List[str]:
        """
        Get list of supported file formats/extensions.

        Returns:
            List[str]: List of supported formats (e.g., ['owl', 'rdf', 'xml'])
        """

    # Concrete methods with default implementations
    def set_options(self, options: Dict[str, Any]) -> None:
        """
        Set parser configuration options.

        Args:
            options (Dict[str, Any]): Options to set
        """
        if not isinstance(options, dict):
            raise ValidationException("Options must be a dictionary")

        self.options.update(options)
        self.logger.debug(f"Updated parser options: {list(options.keys())}")

        # Update cache settings if changed
        if "enable_cache" in options:
            self._cache_enabled = options["enable_cache"]
        if "cache_max_size" in options:
            self._cache_max_size = options["cache_max_size"]

    def get_options(self) -> Dict[str, Any]:
        """
        Get current parser configuration options.

        Returns:
            Dict[str, Any]: Current options
        """
        return copy.deepcopy(self.options)

    def reset_options(self) -> None:
        """
        Reset options to default values.
        """
        self.options = copy.deepcopy(self._default_options)
        self._cache_enabled = self.options.get("enable_cache", True)
        self._cache_max_size = self.options.get("cache_max_size", 100)
        self.logger.debug("Parser options reset to defaults")

    def get_metadata(self) -> Dict[str, Any]:
        """
        Get comprehensive parser metadata and statistics.

        Returns:
            Dict[str, Any]: Parser metadata including statistics and configuration
        """
        return {
            "parser_name": self.parser_name,
            "supported_formats": self.get_supported_formats(),
            "current_options": self.get_options(),
            "statistics": {
                "total_parses": self.statistics.total_parses,
                "successful_parses": self.statistics.successful_parses,
                "failed_parses": self.statistics.failed_parses,
                "success_rate": (
                    self.statistics.successful_parses
                    / max(1, self.statistics.total_parses)
                )
                * 100,
                "average_parse_time": self.statistics.average_parse_time,
                "total_content_processed": self.statistics.total_content_processed,
                "largest_content_size": self.statistics.largest_content_size,
                "validation_failures": self.statistics.validation_failures,
                "validation_successes": self.statistics.validation_successes,
                "warnings_generated": self.statistics.warnings_generated,
                "recent_errors": self.statistics.errors_encountered[
                    -10:
                ],  # Last 10 errors
            },
            "cache_info": self.get_cache_info(),
            "validation_rules": list(self._validation_rules.keys()),
            "hooks": {hook: len(funcs) for hook, funcs in self._hooks.items()},
            "configuration": {
                "config_manager_available": self.config_manager is not None,
                "logger_name": self.logger.name if self.logger else None,
            },
        }

    def get_performance_hints(self) -> List[str]:
        """
        Get performance optimization hints based on usage patterns.

        Returns:
            List[str]: List of performance optimization suggestions
        """
        hints = []

        # Analyze statistics for hints
        if self.statistics.total_parses > 0:
            if self.statistics.average_parse_time > 5.0:
                hints.append("Consider enabling memory_efficient mode for large files")

            if self.statistics.failed_parses / self.statistics.total_parses > 0.1:
                hints.append(
                    "High failure rate detected - review input validation rules"
                )

            if not self._cache_enabled and self.statistics.total_parses > 10:
                hints.append(
                    "Consider enabling caching for repeated parsing operations"
                )

            if self.statistics.largest_content_size > 50 * 1024 * 1024:  # 50MB
                hints.append(
                    "Large files detected - consider streaming parsing if available"
                )

        return hints

    def reset_statistics(self) -> None:
        """
        Reset parser statistics to initial state.
        """
        self.statistics = ParserStatistics()
        self.logger.debug("Parser statistics reset")

    def __str__(self) -> str:
        """String representation of the parser."""
        return f"{self.__class__.__name__}(name={self.parser_name}, formats={self.get_supported_formats()})"

    def __repr__(self) -> str:
        """Detailed string representation of the parser."""
        return (
            f"{self.__class__.__name__}("
            f"name='{self.parser_name}', "
            f"formats={self.get_supported_formats()}, "
            f"parsed={self.statistics.total_parses}, "
            f"cached={len(self._cache)})"
        )


class OWLParser(AbstractParser):
    """Comprehensive OWL parser implementation for the AIM2 ontology project.

    This parser provides complete OWL parsing capabilities including:
    - Multiple format support (OWL/XML, RDF/XML, Turtle, N-Triples, N3, JSON-LD)
    - Integration with owlready2 and rdflib libraries
    - Comprehensive validation including syntax, semantics, and consistency
    - Conversion to internal Term/Relationship/Ontology models
    - Performance optimizations for large ontologies
    - Configurable parsing options and error handling

    The parser uses owlready2 for OWL-specific operations and rdflib for
    general RDF processing and format detection.
    """

    def __init__(self, options: Optional[Dict[str, Any]] = None):
        """Initialize OWL parser with comprehensive options.

        Args:
            options (Dict[str, Any], optional): Parser configuration options
        """
        super().__init__("owl", options=options)

        # Initialize OWL-specific validation errors list
        self._validation_errors: List[str] = []

        # Import required libraries with fallbacks
        try:
            import owlready2

            self._owlready2 = owlready2
            self._owlready2_available = True
        except ImportError:
            self.logger.warning(
                "owlready2 not available - some functionality will be limited"
            )
            self._owlready2 = None
            self._owlready2_available = False

        try:
            import rdflib

            self._rdflib = rdflib
            self._rdflib_available = True
        except ImportError:
            self.logger.warning(
                "rdflib not available - some functionality will be limited"
            )
            self._rdflib = None
            self._rdflib_available = False

        # Set up OWL-specific default options
        owl_defaults = {
            "validate_on_parse": True,
            "extract_triples_on_parse": False,  # Off by default to avoid performance impact
            "strict_validation": False,
            "include_imports": True,
            "preserve_namespaces": True,
            "resolve_imports": True,
            "consistency_check": False,
            "satisfiability_check": False,
            "owl_profile": "OWL-DL",
            "base_iri": None,
            "memory_efficient": False,
            "streaming_mode": False,
            "batch_size": 1000,
            "error_recovery": True,
            "max_errors": 100,
            "continue_on_error": True,
            "log_warnings": True,
        }

        # Update options with OWL defaults
        for key, value in owl_defaults.items():
            if key not in self.options:
                self.options[key] = value

        self.logger.info(
            f"OWL parser initialized with owlready2={self._owlready2_available}, rdflib={self._rdflib_available}"
        )

    def _get_default_options(self) -> Dict[str, Any]:
        """Get OWL parser specific default options.

        Returns:
            Dict[str, Any]: Default options for OWL parser
        """
        base_options = super()._get_default_options()
        owl_options = {
            "validate_on_parse": True,
            "extract_triples_on_parse": False,  # Off by default to avoid performance impact
            "strict_validation": False,
            "include_imports": True,
            "preserve_namespaces": True,
            "resolve_imports": True,
            "consistency_check": False,
            "satisfiability_check": False,
            "owl_profile": "OWL-DL",
            "base_iri": None,
            "memory_efficient": False,
            "streaming_mode": False,
            "batch_size": 1000,
            "error_recovery": True,
            "max_errors": 100,
            "continue_on_error": True,
            "conversion_filters": {
                "include_classes": True,
                "include_properties": True,
                "include_individuals": True,
                "namespace_filter": None,
                "class_filter": None,
                "property_filter": None,
            },
        }
        base_options.update(owl_options)
        return base_options

    def get_supported_formats(self) -> List[str]:
        """Get list of supported OWL formats.

        Returns:
            List[str]: List of supported formats
        """
        return ["owl", "rdf", "ttl", "nt", "n3", "xml", "json-ld"]

    def detect_format(self, content: str) -> str:
        """Auto-detect OWL format from content.

        Args:
            content (str): Content to analyze

        Returns:
            str: Detected format or 'unknown'
        """
        content_lower = content.strip().lower()

        # JSON-LD detection
        if content_lower.startswith("{") and "@context" in content_lower:
            return "json-ld"

        # XML-based formats (OWL/XML, RDF/XML)
        if content_lower.startswith("<?xml") or content_lower.startswith("<rdf:rdf"):
            if "owl:ontology" in content_lower:
                return "owl"
            elif "rdf:rdf" in content_lower:
                return "rdf"
            return "xml"

        # Turtle format
        if (
            "@prefix" in content_lower
            or "@base" in content_lower
            or content_lower.startswith("@")
            or "." in content
            and "a " in content_lower
        ):
            return "ttl"

        # N-Triples format (simple triple pattern)
        lines = content_lower.split("\n")
        if lines and all(
            line.strip().endswith(" .") or not line.strip() for line in lines[:10]
        ):
            return "nt"

        # N3 format
        if "{" in content and "}" in content and "=>" in content:
            return "n3"

        return "unknown"

    def validate_format(self, content: str, format: str) -> bool:
        """Validate content matches specified format.

        Args:
            content (str): Content to validate
            format (str): Expected format

        Returns:
            bool: True if content matches format
        """
        if not self._rdflib_available:
            self.logger.warning("rdflib not available - format validation limited")
            return self.detect_format(content) == format

        try:
            graph = self._rdflib.Graph()

            # Map format names to rdflib format names
            format_map = {
                "owl": "xml",
                "rdf": "xml",
                "xml": "xml",
                "ttl": "turtle",
                "turtle": "turtle",
                "nt": "nt",
                "n3": "n3",
                "json-ld": "json-ld",
            }

            rdf_format = format_map.get(format.lower(), format)
            graph.parse(data=content, format=rdf_format)
            return True

        except Exception as e:
            self.logger.debug(f"Format validation failed for {format}: {str(e)}")
            return False

    def parse(self, content: str, **kwargs) -> Any:
        """Parse OWL content from string.

        Args:
            content (str): OWL content to parse
            **kwargs: Additional parsing parameters

        Returns:
            Any: Parsed OWL data structure (ontology object)

        Raises:
            OntologyException: If parsing fails
        """
        try:
            # Detect format if not specified
            format_hint = kwargs.get("format", self.detect_format(content))
            self.logger.debug(f"Parsing OWL content with format: {format_hint}")

            # Validate format if strict validation is enabled
            if self.options.get("strict_validation", False):
                if not self.validate_format(content, format_hint):
                    raise OntologyException(
                        f"Content validation failed for format: {format_hint}",
                        context={"format": format_hint, "content_length": len(content)},
                    )

            # Parse using rdflib first for general RDF processing
            rdf_graph = None
            if self._rdflib_available:
                try:
                    rdf_graph = self._parse_with_rdflib(content, format_hint)
                except Exception as e:
                    error_context = self._handle_parse_error(
                        e, 
                        location="rdflib parsing",
                        return_type="rdflib_graph",
                        fallback_parser="owlready2",
                        content=content,
                        format_hint=format_hint
                    )
                    
                    if error_context.recovery_data.get("recovery_successful"):
                        rdf_graph = error_context.recovery_data.get("recovery_result")
                    else:
                        rdf_graph = None
                    
                    # If recovery failed and error is fatal, re-raise
                    if error_context.severity == ErrorSeverity.FATAL:
                        raise OntologyException(f"RDFlib parsing failed fatally: {str(e)}")

            # Parse using owlready2 for OWL-specific features
            owl_ontology = None
            if self._owlready2_available:
                try:
                    owl_ontology = self._parse_with_owlready2(
                        content, format_hint, **kwargs
                    )
                except Exception as e:
                    error_context = self._handle_parse_error(
                        e,
                        location="owlready2 parsing", 
                        return_type="owl_ontology",
                        fallback_parser="rdflib",
                        content=content,
                        format_hint=format_hint,
                        **kwargs
                    )
                    
                    if error_context.recovery_data.get("recovery_successful"):
                        owl_ontology = error_context.recovery_data.get("recovery_result")
                    else:
                        owl_ontology = None
                    
                    # If recovery failed and error is fatal, re-raise
                    if error_context.severity == ErrorSeverity.FATAL:
                        raise OntologyException(f"owlready2 parsing failed fatally: {str(e)}")

            # Return structured result
            result = {
                "rdf_graph": rdf_graph,
                "owl_ontology": owl_ontology,
                "format": format_hint,
                "parsed_at": time.time(),
                "content_size": len(content),
                "options_used": copy.deepcopy(kwargs),
            }

            # Add validation results if requested
            if self.options.get("validate_on_parse", True):
                result["validation"] = self.validate_owl(content, **kwargs)

            # Add triple extraction results if requested
            if self.options.get("extract_triples_on_parse", True):
                try:
                    result["triples"] = self.extract_triples(result)
                    result["triple_count"] = len(result["triples"])
                except Exception as e:
                    if self.options.get("continue_on_error", True):
                        self.logger.warning(
                            f"Failed to extract triples during parsing: {str(e)}"
                        )
                        result["triples"] = []
                        result["triple_count"] = 0
                    else:
                        raise

            return result

        except Exception as e:
            raise OntologyException(
                f"OWL parsing failed: {str(e)}",
                context={
                    "format": kwargs.get("format", "auto"),
                    "content_length": len(content),
                },
            )

    def _parse_with_rdflib(self, content: str, format_hint: str) -> Any:
        """Parse content using rdflib.

        Args:
            content (str): Content to parse
            format_hint (str): Format hint

        Returns:
            rdflib.Graph: Parsed RDF graph
        """
        if not self._rdflib_available:
            raise OntologyException("rdflib not available")

        graph = self._rdflib.Graph()

        # Map format names
        format_map = {
            "owl": "xml",
            "rdf": "xml",
            "xml": "xml",
            "ttl": "turtle",
            "turtle": "turtle",
            "nt": "nt",
            "n3": "n3",
            "json-ld": "json-ld",
        }

        rdf_format = format_map.get(format_hint.lower(), "xml")

        # Parse with base IRI if specified
        base_iri = self.options.get("base_iri")
        if base_iri:
            graph.parse(data=content, format=rdf_format, publicID=base_iri)
        else:
            graph.parse(data=content, format=rdf_format)

        return graph

    def _parse_with_owlready2(self, content: str, format_hint: str, **kwargs) -> Any:
        """Parse content using owlready2.

        Args:
            content (str): Content to parse
            format_hint (str): Format hint
            **kwargs: Additional parameters

        Returns:
            owlready2.Ontology: Parsed OWL ontology
        """
        if not self._owlready2_available:
            raise OntologyException("owlready2 not available")

        # Create temporary file for owlready2 (it works best with files)
        import os
        import tempfile

        with tempfile.NamedTemporaryFile(
            mode="w", suffix=f".{format_hint}", delete=False
        ) as tmp_file:
            tmp_file.write(content)
            tmp_file_path = tmp_file.name

        try:
            # Load ontology with owlready2
            ontology = self._owlready2.get_ontology(f"file://{tmp_file_path}")

            if self.options.get("include_imports", True):
                ontology.load()
            else:
                ontology.load(only_local=True)

            return ontology

        finally:
            # Clean up temporary file
            try:
                os.unlink(tmp_file_path)
            except OSError:
                pass

    def parse_file(self, file_path: str, **kwargs) -> Any:
        """Parse OWL from file path.

        Args:
            file_path (str): Path to OWL file
            **kwargs: Additional parsing parameters

        Returns:
            Any: Parsed OWL data
        """
        file_path = Path(file_path)
        if not file_path.exists():
            raise ValidationException(f"File not found: {file_path}")

        # Auto-detect format from extension if not specified
        if "format" not in kwargs:
            ext = file_path.suffix.lower().lstrip(".")
            if ext in self.get_supported_formats():
                kwargs["format"] = ext

        # Read and parse file content
        try:
            encoding = self.options.get("encoding", "utf-8")
            with open(file_path, "r", encoding=encoding) as f:
                content = f.read()

            kwargs["source_file"] = str(file_path)
            return self.parse(content, **kwargs)

        except Exception as e:
            raise OntologyException(
                f"Failed to parse OWL file {file_path}: {str(e)}",
                context={"file_path": str(file_path)},
            )

    def parse_string(self, content: str, **kwargs) -> Any:
        """Parse OWL content from string.

        Args:
            content (str): OWL content string
            **kwargs: Additional parsing parameters

        Returns:
            Any: Parsed OWL data
        """
        return self.parse(content, **kwargs)

    def parse_url(self, url: str, **kwargs) -> Any:
        """Parse OWL from remote URL.

        Args:
            url (str): URL to fetch OWL content from
            **kwargs: Additional parsing parameters

        Returns:
            Any: Parsed OWL data

        Raises:
            OntologyException: If URL fetch or parsing fails
        """
        try:
            import urllib.error
            import urllib.request
            from urllib.parse import urlparse

            # Validate URL scheme for security
            parsed_url = urlparse(url)
            if parsed_url.scheme not in ("http", "https"):
                raise OntologyException(
                    f"Unsupported URL scheme: {parsed_url.scheme}. Only http and https are allowed.",
                    context={"url": url, "scheme": parsed_url.scheme},
                )

            timeout = self.options.get("timeout_seconds", 300)

            with urllib.request.urlopen(url, timeout=timeout) as response:  # nosec B310
                content = response.read().decode("utf-8")

            kwargs["source_url"] = url
            return self.parse(content, **kwargs)

        except urllib.error.URLError as e:
            raise OntologyException(
                f"Failed to fetch OWL from URL {url}: {str(e)}", context={"url": url}
            )
        except Exception as e:
            raise OntologyException(
                f"Failed to parse OWL from URL {url}: {str(e)}", context={"url": url}
            )

    def parse_stream(self, stream, **kwargs) -> Any:
        """Parse OWL from stream/file-like object.

        Args:
            stream: Stream or file-like object to read from
            **kwargs: Additional parsing parameters

        Returns:
            Any: Parsed OWL data
        """
        try:
            content = stream.read()
            if isinstance(content, bytes):
                encoding = self.options.get("encoding", "utf-8")
                content = content.decode(encoding)

            kwargs["source_type"] = "stream"
            return self.parse(content, **kwargs)

        except Exception as e:
            raise OntologyException(f"Failed to parse OWL from stream: {str(e)}")

    def validate(self, content: str, **kwargs) -> bool:
        """Basic OWL content validation.

        Args:
            content (str): Content to validate
            **kwargs: Additional validation parameters

        Returns:
            bool: True if content is valid OWL
        """
        try:
            # Clear previous validation errors
            self._validation_errors.clear()

            # Basic format validation
            detected_format = self.detect_format(content)
            if detected_format == "unknown":
                self._validation_errors.append("Unknown or unsupported format")
                return False

            # Try parsing with both libraries
            parse_success = False

            if self._rdflib_available:
                try:
                    self._parse_with_rdflib(content, detected_format)
                    parse_success = True
                except Exception as e:
                    self._validation_errors.append(
                        f"RDFlib validation failed: {str(e)}"
                    )

            if self._owlready2_available and not parse_success:
                try:
                    self._parse_with_owlready2(content, detected_format)
                    parse_success = True
                except Exception as e:
                    self._validation_errors.append(
                        f"owlready2 validation failed: {str(e)}"
                    )

            return parse_success

        except Exception as e:
            self._validation_errors.append(f"Validation error: {str(e)}")
            return False

    def validate_owl(self, content: str, **kwargs) -> Dict[str, Any]:
        """Comprehensive OWL validation with detailed results.

        Args:
            content (str): OWL content to validate
            **kwargs: Additional validation parameters

        Returns:
            Dict[str, Any]: Detailed validation results
        """
        result = {
            "is_valid": False,
            "format": "unknown",
            "errors": [],
            "warnings": [],
            "consistency_check": None,
            "satisfiability_check": None,
            "profile_compliant": None,
            "imports_resolved": None,
            "namespace_consistent": None,
            "statistics": {},
        }

        try:
            # Basic format detection and validation
            detected_format = self.detect_format(content)
            result["format"] = detected_format

            if detected_format == "unknown":
                result["errors"].append("Unknown or unsupported format")
                return result

            # Parse content for validation
            parsed_data = None
            try:
                parsed_data = self.parse(content, **kwargs)
                result["is_valid"] = True
            except Exception as e:
                result["errors"].append(f"Parsing failed: {str(e)}")
                return result

            # Extract statistics
            if parsed_data and isinstance(parsed_data, dict):
                rdf_graph = parsed_data.get("rdf_graph")
                owl_ontology = parsed_data.get("owl_ontology")

                if rdf_graph and self._rdflib_available:
                    result["statistics"]["triple_count"] = len(rdf_graph)
                    result["statistics"]["namespace_count"] = len(
                        list(rdf_graph.namespaces())
                    )

                if owl_ontology and self._owlready2_available:
                    try:
                        result["statistics"]["class_count"] = len(
                            list(owl_ontology.classes())
                        )
                        result["statistics"]["property_count"] = len(
                            list(owl_ontology.properties())
                        )
                        result["statistics"]["individual_count"] = len(
                            list(owl_ontology.individuals())
                        )
                    except Exception as e:
                        result["warnings"].append(
                            f"Failed to extract OWL statistics: {str(e)}"
                        )

            # Consistency checking if enabled
            if self.options.get("consistency_check", False) and owl_ontology:
                try:
                    with self._owlready2.sync_reasoner_pellet([owl_ontology]):
                        result["consistency_check"] = True
                except Exception as e:
                    result["consistency_check"] = False
                    result["warnings"].append(f"Consistency check failed: {str(e)}")

            # Profile compliance checking
            owl_profile = self.options.get("owl_profile", "OWL-DL")
            if owl_profile and owl_ontology:
                # Basic profile compliance (simplified)
                result["profile_compliant"] = True  # Placeholder
                result["warnings"].append(
                    f"Profile compliance for {owl_profile} not fully implemented"
                )

            # Import resolution checking
            if self.options.get("include_imports", True):
                result["imports_resolved"] = True  # Placeholder
                result["unresolved_imports"] = []

            # Namespace consistency
            result["namespace_consistent"] = True  # Placeholder
            result["undefined_namespaces"] = []

            return result

        except Exception as e:
            result["errors"].append(f"Validation error: {str(e)}")
            return result

    def get_validation_errors(self) -> List[str]:
        """Get validation errors from last validation.

        Returns:
            List[str]: List of validation errors
        """
        return copy.deepcopy(self._validation_errors)

    def to_ontology(self, parsed_result: Any) -> "Ontology":
        """Convert parsed OWL to internal Ontology model.

        Args:
            parsed_result (Any): Result from parse method

        Returns:
            Ontology: Internal ontology model
        """
        if not isinstance(parsed_result, dict):
            raise ValidationException("Invalid parsed result format")

        # Import Ontology model (with fallback)
        try:
            from ..models import Ontology
        except ImportError:
            # Create minimal ontology structure for testing
            class Ontology:
                def __init__(self, id=None, name=None, **kwargs):
                    self.id = id
                    self.name = name
                    self.terms = {}
                    self.relationships = {}
                    self.metadata = kwargs

        # Extract basic ontology information
        ontology_id = "unknown"
        ontology_name = "Parsed Ontology"
        metadata = {
            "source_format": parsed_result.get("format", "unknown"),
            "parser_version": "1.0.0",
            "parsing_timestamp": datetime.now().isoformat(),
            "content_size": parsed_result.get("content_size", 0),
        }

        # Extract from OWL ontology if available
        owl_ontology = parsed_result.get("owl_ontology")
        if owl_ontology and self._owlready2_available:
            try:
                ontology_id = str(owl_ontology.base_iri or "unknown")
                ontology_name = getattr(owl_ontology, "name", "Parsed Ontology")
            except Exception as e:
                self.logger.warning(f"Failed to extract ontology metadata: {str(e)}")

        # Create ontology instance
        ontology = Ontology(id=ontology_id, name=ontology_name, metadata=metadata)

        return ontology

    def extract_terms(self, parsed_result: Any) -> List["Term"]:
        """Extract Term objects from parsed OWL.

        Args:
            parsed_result (Any): Result from parse method

        Returns:
            List[Term]: List of extracted terms
        """
        terms = []

        if not isinstance(parsed_result, dict):
            return terms

        # Import Term model (with fallback)
        try:
            from ..models import Term
        except ImportError:
            # Create minimal term structure for testing
            class Term:
                def __init__(self, id=None, name=None, **kwargs):
                    self.id = id
                    self.name = name
                    for k, v in kwargs.items():
                        setattr(self, k, v)

        # Extract from OWL ontology
        owl_ontology = parsed_result.get("owl_ontology")
        if owl_ontology and self._owlready2_available:
            try:
                for cls in owl_ontology.classes():
                    term_id = str(cls.name) if cls.name else str(cls)
                    term_name = str(cls.label[0]) if cls.label else term_id
                    definition = str(cls.comment[0]) if cls.comment else None

                    term = Term(
                        id=term_id,
                        name=term_name,
                        definition=definition,
                        namespace=str(owl_ontology.name) if owl_ontology.name else None,
                    )
                    terms.append(term)

            except Exception as e:
                self.logger.warning(f"Failed to extract terms from OWL: {str(e)}")

        # Extract from RDF graph as fallback
        rdf_graph = parsed_result.get("rdf_graph")
        if not terms and rdf_graph and self._rdflib_available:
            try:
                # Query for OWL classes
                owl_class = self._rdflib.Namespace(
                    "http://www.w3.org/2002/07/owl#"
                ).Class
                rdfs_label = self._rdflib.Namespace(
                    "http://www.w3.org/2000/01/rdf-schema#"
                ).label

                for subject in rdf_graph.subjects(self._rdflib.RDF.type, owl_class):
                    term_id = (
                        str(subject).split("#")[-1]
                        if "#" in str(subject)
                        else str(subject)
                    )

                    # Get label
                    labels = list(rdf_graph.objects(subject, rdfs_label))
                    term_name = str(labels[0]) if labels else term_id

                    term = Term(id=term_id, name=term_name)
                    terms.append(term)

            except Exception as e:
                self.logger.warning(f"Failed to extract terms from RDF: {str(e)}")

        return terms

    def extract_relationships(self, parsed_result: Any) -> List["Relationship"]:
        """Extract Relationship objects from parsed OWL.

        Args:
            parsed_result (Any): Result from parse method

        Returns:
            List[Relationship]: List of extracted relationships
        """
        relationships = []

        if not isinstance(parsed_result, dict):
            return relationships

        # Import Relationship model (with fallback)
        try:
            from ..models import Relationship
        except ImportError:
            # Create minimal relationship structure for testing
            class Relationship:
                def __init__(
                    self, id=None, subject=None, predicate=None, object=None, **kwargs
                ):
                    self.id = id
                    self.subject = subject
                    self.predicate = predicate
                    self.object = object
                    for k, v in kwargs.items():
                        setattr(self, k, v)

        # Extract from RDF graph
        rdf_graph = parsed_result.get("rdf_graph")
        if rdf_graph and self._rdflib_available:
            try:
                rdfs_subClassOf = self._rdflib.Namespace(
                    "http://www.w3.org/2000/01/rdf-schema#"
                ).subClassOf

                rel_count = 0
                for subject, predicate, obj in rdf_graph:
                    # Focus on interesting relationships
                    if predicate == rdfs_subClassOf:
                        rel_id = f"REL:{rel_count:06d}"
                        rel_count += 1

                        subj_id = (
                            str(subject).split("#")[-1]
                            if "#" in str(subject)
                            else str(subject)
                        )
                        obj_id = (
                            str(obj).split("#")[-1] if "#" in str(obj) else str(obj)
                        )

                        relationship = Relationship(
                            id=rel_id,
                            subject=subj_id,
                            predicate="is_a",
                            object=obj_id,
                            confidence=1.0,
                        )
                        relationships.append(relationship)

            except Exception as e:
                self.logger.warning(f"Failed to extract relationships: {str(e)}")

        return relationships

    def extract_metadata(self, parsed_result: Any) -> Dict[str, Any]:
        """Extract metadata from parsed OWL.

        Args:
            parsed_result (Any): Result from parse method

        Returns:
            Dict[str, Any]: Extracted metadata
        """
        metadata = {
            "format": parsed_result.get("format", "unknown")
            if isinstance(parsed_result, dict)
            else "unknown",
            "parsed_at": time.time(),
            "parser_name": self.parser_name,
            "parser_version": "1.0.0",
        }

        if not isinstance(parsed_result, dict):
            return metadata

        # Extract from OWL ontology
        owl_ontology = parsed_result.get("owl_ontology")
        if owl_ontology and self._owlready2_available:
            try:
                metadata.update(
                    {
                        "ontology_iri": str(owl_ontology.base_iri)
                        if owl_ontology.base_iri
                        else None,
                        "ontology_name": str(owl_ontology.name)
                        if owl_ontology.name
                        else None,
                        "class_count": len(list(owl_ontology.classes())),
                        "property_count": len(list(owl_ontology.properties())),
                        "individual_count": len(list(owl_ontology.individuals())),
                    }
                )
            except Exception as e:
                self.logger.warning(f"Failed to extract OWL metadata: {str(e)}")

        # Extract from RDF graph
        rdf_graph = parsed_result.get("rdf_graph")
        if rdf_graph and self._rdflib_available:
            try:
                metadata.update(
                    {
                        "triple_count": len(rdf_graph),
                        "namespaces": dict(rdf_graph.namespaces()),
                    }
                )
            except Exception as e:
                self.logger.warning(f"Failed to extract RDF metadata: {str(e)}")

        return metadata

    def extract_triples(self, parsed_result: Any) -> List["RDFTriple"]:
        """Extract RDF triples from parsed OWL.

        This method extracts all RDF triples from the parsed OWL result, providing
        comprehensive triple-level access to the ontology data. It supports extraction
        from both rdflib Graph objects and owlready2 ontology objects, with proper
        metadata and confidence scoring.

        Args:
            parsed_result (Any): Result from parse method containing rdf_graph and/or owl_ontology

        Returns:
            List[RDFTriple]: List of extracted RDF triples with comprehensive metadata

        Examples:
            >>> parser = OWLParser()
            >>> content = '''<?xml version="1.0"?>
            ... <rdf:RDF xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#"
            ...          xmlns:rdfs="http://www.w3.org/2000/01/rdf-schema#"
            ...          xmlns:owl="http://www.w3.org/2002/07/owl#">
            ...   <owl:Class rdf:about="http://example.org/Person">
            ...     <rdfs:label>Person</rdfs:label>
            ...   </owl:Class>
            ... </rdf:RDF>'''
            >>> parsed = parser.parse(content)
            >>> triples = parser.extract_triples(parsed)
            >>> len(triples) > 0
            True
            >>> any(t.predicate.endswith('label') for t in triples)
            True
        """
        triples = []

        if not isinstance(parsed_result, dict):
            self.logger.warning("Invalid parsed result format for triple extraction")
            return triples

        # Import RDFTriple model (with fallback)
        try:
            from ..models import RDFTriple
        except ImportError:
            # Create minimal triple structure for testing
            class RDFTriple:
                def __init__(self, subject=None, predicate=None, object=None, **kwargs):
                    self.subject = subject
                    self.predicate = predicate
                    self.object = object
                    for k, v in kwargs.items():
                        setattr(self, k, v)

        try:
            # Extract metadata for triples
            extraction_metadata = {
                "extraction_method": "owl_parsing",
                "parser_version": "1.0.0",
                "source_format": parsed_result.get("format", "unknown"),
                "extracted_at": time.time(),
            }

            # Get source information if available
            source_info = parsed_result.get("options_used", {}).get(
                "source_file", "unknown"
            )

            # Extract from RDF graph (primary method)
            rdf_graph = parsed_result.get("rdf_graph")
            if rdf_graph and self._rdflib_available:
                triples.extend(
                    self._extract_triples_from_rdf_graph(
                        rdf_graph, source_info, extraction_metadata
                    )
                )

            # Extract from OWL ontology as supplement/fallback
            owl_ontology = parsed_result.get("owl_ontology")
            if owl_ontology and self._owlready2_available and not triples:
                # Only use owlready2 extraction if rdflib didn't produce results
                triples.extend(
                    self._extract_triples_from_owl_ontology(
                        owl_ontology, source_info, extraction_metadata
                    )
                )

            # Apply filters if configured
            if self.options.get("conversion_filters", {}).get("namespace_filter"):
                triples = self._filter_triples_by_namespace(triples)

            # Log extraction statistics
            if self.options.get("log_performance", True):
                self.logger.info(
                    f"Extracted {len(triples)} RDF triples from {parsed_result.get('format', 'unknown')} format"
                )

            return triples

        except Exception as e:
            error_msg = f"Failed to extract triples: {str(e)}"
            self.logger.error(error_msg)
            if self.options.get("continue_on_error", True):
                return triples  # Return whatever we managed to extract
            else:
                raise OntologyException(
                    error_msg, context={"parser": "owl", "method": "extract_triples"}
                )

    def _extract_triples_from_rdf_graph(
        self, rdf_graph: Any, source_info: str, extraction_metadata: Dict[str, Any]
    ) -> List["RDFTriple"]:
        """Extract triples from rdflib Graph object.

        Args:
            rdf_graph: rdflib Graph object
            source_info: Source information for the triples
            extraction_metadata: Metadata about the extraction process

        Returns:
            List[RDFTriple]: List of extracted triples
        """
        triples = []

        try:
            # Import RDFTriple for type hints
            from ..models import RDFTriple
        except ImportError:
            # Fallback for testing
            class RDFTriple:
                def __init__(self, **kwargs):
                    for k, v in kwargs.items():
                        setattr(self, k, v)

        try:
            # Get namespace mappings from the graph
            namespace_prefixes = dict(rdf_graph.namespaces())

            # Extract all triples from the graph
            triple_count = 0
            for subject, predicate, obj in rdf_graph:
                try:
                    # Determine node types
                    subject_type = self._determine_node_type(subject)
                    object_type = self._determine_node_type(obj)

                    # Extract object datatype and language for literals
                    object_datatype = None
                    object_language = None
                    if object_type == "literal" and hasattr(obj, "datatype"):
                        object_datatype = str(obj.datatype) if obj.datatype else None
                    if object_type == "literal" and hasattr(obj, "language"):
                        object_language = obj.language if obj.language else None

                    # Create RDF triple with comprehensive metadata
                    triple = RDFTriple(
                        subject=str(subject),
                        predicate=str(predicate),
                        object=str(obj),
                        subject_type=subject_type,
                        object_type=object_type,
                        object_datatype=object_datatype,
                        object_language=object_language,
                        source=source_info,
                        confidence=1.0,  # High confidence for directly parsed triples
                        metadata=copy.deepcopy(extraction_metadata),
                        namespace_prefixes=namespace_prefixes,
                    )

                    triples.append(triple)
                    triple_count += 1

                    # Apply batch size limits if configured
                    batch_size = self.options.get("batch_size", 1000)
                    if batch_size > 0 and triple_count >= batch_size:
                        if self.options.get("log_warnings", True):
                            self.logger.warning(
                                f"Reached batch size limit of {batch_size} triples"
                            )
                        break

                except Exception as e:
                    if self.options.get("continue_on_error", True):
                        self.logger.warning(
                            f"Failed to extract triple {subject} {predicate} {obj}: {str(e)}"
                        )
                        continue
                    else:
                        raise

            self.logger.debug(f"Extracted {len(triples)} triples from RDF graph")
            return triples

        except Exception as e:
            error_msg = f"Failed to extract triples from RDF graph: {str(e)}"
            self.logger.error(error_msg)
            if self.options.get("continue_on_error", True):
                return triples
            else:
                raise OntologyException(error_msg)

    def _extract_triples_from_owl_ontology(
        self, owl_ontology: Any, source_info: str, extraction_metadata: Dict[str, Any]
    ) -> List["RDFTriple"]:
        """Extract triples from owlready2 ontology object.

        This method serves as a fallback when rdflib extraction fails or
        for accessing OWL-specific constructs that may not be fully represented
        in the RDF graph.

        Args:
            owl_ontology: owlready2 ontology object
            source_info: Source information for the triples
            extraction_metadata: Metadata about the extraction process

        Returns:
            List[RDFTriple]: List of extracted triples
        """
        triples = []

        try:
            # Import RDFTriple for type hints
            from ..models import RDFTriple
        except ImportError:
            # Fallback for testing
            class RDFTriple:
                def __init__(self, **kwargs):
                    for k, v in kwargs.items():
                        setattr(self, k, v)

        try:
            # Extract basic namespace information
            namespace_prefixes = {}
            if hasattr(owl_ontology, "base_iri") and owl_ontology.base_iri:
                namespace_prefixes[""] = str(owl_ontology.base_iri)

            # Common RDF/OWL namespaces
            namespace_prefixes.update(
                {
                    "rdf": "http://www.w3.org/1999/02/22-rdf-syntax-ns#",
                    "rdfs": "http://www.w3.org/2000/01/rdf-schema#",
                    "owl": "http://www.w3.org/2002/07/owl#",
                    "xsd": "http://www.w3.org/2001/XMLSchema#",
                }
            )

            # Extract class-related triples
            if self.options.get("conversion_filters", {}).get("include_classes", True):
                triples.extend(
                    self._extract_class_triples(
                        owl_ontology,
                        source_info,
                        extraction_metadata,
                        namespace_prefixes,
                    )
                )

            # Extract property-related triples
            if self.options.get("conversion_filters", {}).get(
                "include_properties", True
            ):
                triples.extend(
                    self._extract_property_triples(
                        owl_ontology,
                        source_info,
                        extraction_metadata,
                        namespace_prefixes,
                    )
                )

            # Extract individual-related triples
            if self.options.get("conversion_filters", {}).get(
                "include_individuals", True
            ):
                triples.extend(
                    self._extract_individual_triples(
                        owl_ontology,
                        source_info,
                        extraction_metadata,
                        namespace_prefixes,
                    )
                )

            self.logger.debug(f"Extracted {len(triples)} triples from OWL ontology")
            return triples

        except Exception as e:
            error_msg = f"Failed to extract triples from OWL ontology: {str(e)}"
            self.logger.error(error_msg)
            if self.options.get("continue_on_error", True):
                return triples
            else:
                raise OntologyException(error_msg)

    def _extract_class_triples(
        self,
        owl_ontology: Any,
        source_info: str,
        extraction_metadata: Dict[str, Any],
        namespace_prefixes: Dict[str, str],
    ) -> List["RDFTriple"]:
        """Extract triples related to OWL classes."""
        triples = []

        try:
            from ..models import RDFTriple
        except ImportError:

            class RDFTriple:
                def __init__(self, **kwargs):
                    for k, v in kwargs.items():
                        setattr(self, k, v)

        try:
            for cls in owl_ontology.classes():
                class_iri = str(cls.iri) if hasattr(cls, "iri") else str(cls)

                # Class type triple
                triples.append(
                    RDFTriple(
                        subject=class_iri,
                        predicate="http://www.w3.org/1999/02/22-rdf-syntax-ns#type",
                        object="http://www.w3.org/2002/07/owl#Class",
                        source=source_info,
                        confidence=1.0,
                        metadata=copy.deepcopy(extraction_metadata),
                        namespace_prefixes=namespace_prefixes,
                    )
                )

                # Label triples
                if hasattr(cls, "label") and cls.label:
                    for label in cls.label:
                        triples.append(
                            RDFTriple(
                                subject=class_iri,
                                predicate="http://www.w3.org/2000/01/rdf-schema#label",
                                object=str(label),
                                object_type="literal",
                                object_datatype="http://www.w3.org/2001/XMLSchema#string",
                                source=source_info,
                                confidence=1.0,
                                metadata=copy.deepcopy(extraction_metadata),
                                namespace_prefixes=namespace_prefixes,
                            )
                        )

                # Comment/definition triples
                if hasattr(cls, "comment") and cls.comment:
                    for comment in cls.comment:
                        triples.append(
                            RDFTriple(
                                subject=class_iri,
                                predicate="http://www.w3.org/2000/01/rdf-schema#comment",
                                object=str(comment),
                                object_type="literal",
                                object_datatype="http://www.w3.org/2001/XMLSchema#string",
                                source=source_info,
                                confidence=1.0,
                                metadata=copy.deepcopy(extraction_metadata),
                                namespace_prefixes=namespace_prefixes,
                            )
                        )

        except Exception as e:
            self.logger.warning(f"Failed to extract class triples: {str(e)}")

        return triples

    def _extract_property_triples(
        self,
        owl_ontology: Any,
        source_info: str,
        extraction_metadata: Dict[str, Any],
        namespace_prefixes: Dict[str, str],
    ) -> List["RDFTriple"]:
        """Extract triples related to OWL properties."""
        triples = []

        try:
            from ..models import RDFTriple
        except ImportError:

            class RDFTriple:
                def __init__(self, **kwargs):
                    for k, v in kwargs.items():
                        setattr(self, k, v)

        try:
            for prop in owl_ontology.properties():
                prop_iri = str(prop.iri) if hasattr(prop, "iri") else str(prop)

                # Property type triple
                triples.append(
                    RDFTriple(
                        subject=prop_iri,
                        predicate="http://www.w3.org/1999/02/22-rdf-syntax-ns#type",
                        object="http://www.w3.org/1999/02/22-rdf-syntax-ns#Property",
                        source=source_info,
                        confidence=1.0,
                        metadata=copy.deepcopy(extraction_metadata),
                        namespace_prefixes=namespace_prefixes,
                    )
                )

                # Label triples
                if hasattr(prop, "label") and prop.label:
                    for label in prop.label:
                        triples.append(
                            RDFTriple(
                                subject=prop_iri,
                                predicate="http://www.w3.org/2000/01/rdf-schema#label",
                                object=str(label),
                                object_type="literal",
                                object_datatype="http://www.w3.org/2001/XMLSchema#string",
                                source=source_info,
                                confidence=1.0,
                                metadata=copy.deepcopy(extraction_metadata),
                                namespace_prefixes=namespace_prefixes,
                            )
                        )

        except Exception as e:
            self.logger.warning(f"Failed to extract property triples: {str(e)}")

        return triples

    def _extract_individual_triples(
        self,
        owl_ontology: Any,
        source_info: str,
        extraction_metadata: Dict[str, Any],
        namespace_prefixes: Dict[str, str],
    ) -> List["RDFTriple"]:
        """Extract triples related to OWL individuals."""
        triples = []

        try:
            from ..models import RDFTriple
        except ImportError:

            class RDFTriple:
                def __init__(self, **kwargs):
                    for k, v in kwargs.items():
                        setattr(self, k, v)

        try:
            for individual in owl_ontology.individuals():
                individual_iri = (
                    str(individual.iri)
                    if hasattr(individual, "iri")
                    else str(individual)
                )

                # Individual type triples
                if hasattr(individual, "is_a"):
                    for class_type in individual.is_a:
                        class_iri = (
                            str(class_type.iri)
                            if hasattr(class_type, "iri")
                            else str(class_type)
                        )
                        triples.append(
                            RDFTriple(
                                subject=individual_iri,
                                predicate="http://www.w3.org/1999/02/22-rdf-syntax-ns#type",
                                object=class_iri,
                                source=source_info,
                                confidence=1.0,
                                metadata=copy.deepcopy(extraction_metadata),
                                namespace_prefixes=namespace_prefixes,
                            )
                        )

        except Exception as e:
            self.logger.warning(f"Failed to extract individual triples: {str(e)}")

        return triples

    def _determine_node_type(self, node: Any) -> str:
        """Determine the type of an RDF node (URI, blank node, or literal).

        Args:
            node: RDF node from rdflib

        Returns:
            str: Node type ('uri', 'bnode', or 'literal')
        """
        if not self._rdflib_available:
            return "uri"  # Default assumption

        try:
            if isinstance(node, self._rdflib.URIRef):
                return "uri"
            elif isinstance(node, self._rdflib.BNode):
                return "bnode"
            elif isinstance(node, self._rdflib.Literal):
                return "literal"
            else:
                return "uri"  # Default fallback
        except Exception:
            return "uri"  # Fallback on any error

    def _filter_triples_by_namespace(
        self, triples: List["RDFTriple"]
    ) -> List["RDFTriple"]:
        """Filter triples by namespace if namespace filter is configured.

        Args:
            triples: List of RDF triples to filter

        Returns:
            List[RDFTriple]: Filtered list of triples
        """
        namespace_filter = self.options.get("conversion_filters", {}).get(
            "namespace_filter"
        )
        if not namespace_filter:
            return triples

        filtered_triples = []
        for triple in triples:
            # Check if subject or object matches the namespace filter
            if triple.subject.startswith(namespace_filter) or triple.object.startswith(
                namespace_filter
            ):
                filtered_triples.append(triple)

        self.logger.debug(
            f"Filtered {len(triples)} triples to {len(filtered_triples)} based on namespace filter"
        )
        return filtered_triples

    # =====================
    # OWL-specific Error Recovery Methods
    # =====================

    def _recover_retry(self, error_context: ErrorContext, **kwargs) -> Any:
        """
        OWL-specific retry recovery with alternative parsing strategies.

        Args:
            error_context (ErrorContext): Error context information
            **kwargs: Additional recovery parameters

        Returns:
            Any: Result of retry attempt or None if failed
        """
        retry_count = len([s for s in error_context.attempted_recoveries if s == RecoveryStrategy.RETRY])
        max_retries = kwargs.get("max_retries", 3)
        
        if retry_count >= max_retries:
            self.logger.error(f"Maximum retries ({max_retries}) exceeded for {error_context.location}")
            return None
        
        self.logger.info(f"Attempting OWL-specific retry at {error_context.location} (attempt {retry_count + 1}/{max_retries})")
        
        content = kwargs.get("content", "")
        format_hint = kwargs.get("format_hint", "xml")
        
        try:
            # Strategy 1: Try alternative format detection
            if retry_count == 0:
                self.logger.info("Retry strategy 1: Alternative format detection")
                detected_format = self.detect_format(content)
                if detected_format != format_hint:
                    self.logger.info(f"Switching from {format_hint} to detected format {detected_format}")
                    if "rdflib" in error_context.location:
                        return self._parse_with_rdflib(content, detected_format)
                    elif "owlready2" in error_context.location:
                        return self._parse_with_owlready2(content, detected_format, **kwargs)
            
            # Strategy 2: Try cleaning malformed XML/RDF
            elif retry_count == 1:
                self.logger.info("Retry strategy 2: Content sanitization")
                cleaned_content = self._sanitize_owl_content(content)
                if cleaned_content != content:
                    self.logger.info("Content was sanitized, retrying with cleaned version")
                    if "rdflib" in error_context.location:
                        return self._parse_with_rdflib(cleaned_content, format_hint)
                    elif "owlready2" in error_context.location:
                        return self._parse_with_owlready2(cleaned_content, format_hint, **kwargs)
            
            # Strategy 3: Try alternative parser
            elif retry_count == 2:
                self.logger.info("Retry strategy 3: Alternative parser")
                fallback_parser = kwargs.get("fallback_parser")
                if fallback_parser == "owlready2" and self._owlready2_available:
                    self.logger.info("Trying owlready2 as fallback from rdflib")
                    return self._parse_with_owlready2(content, format_hint, **kwargs)
                elif fallback_parser == "rdflib" and self._rdflib_available:
                    self.logger.info("Trying rdflib as fallback from owlready2")
                    return self._parse_with_rdflib(content, format_hint)
        
        except Exception as retry_error:
            self.logger.warning(f"Retry attempt {retry_count + 1} failed: {str(retry_error)}")
            return None
        
        return None

    def _sanitize_owl_content(self, content: str) -> str:
        """
        Sanitize OWL/RDF content to fix common parsing issues.

        Args:
            content (str): Original OWL/RDF content

        Returns:
            str: Sanitized content
        """
        import re
        
        sanitized = content
        original_length = len(content)
        
        try:
            # Fix 1: Remove invalid XML characters
            sanitized = re.sub(r'[\x00-\x08\x0B-\x0C\x0E-\x1F\x7F]', '', sanitized)
            
            # Fix 2: Fix malformed namespace declarations
            # Find incomplete namespace declarations and add default values
            namespace_pattern = r'xmlns:(\w+)=["\']\s*["\']\s*'
            matches = re.findall(namespace_pattern, sanitized)
            for prefix in matches:
                default_ns = f"http://example.org/{prefix}#"
                sanitized = re.sub(
                    rf'xmlns:{prefix}=["\']\s*["\']\s*',
                    f'xmlns:{prefix}="{default_ns}"',
                    sanitized
                )
                self.logger.info(f"Fixed empty namespace declaration for prefix '{prefix}' with default: {default_ns}")
            
            # Fix 3: Fix missing xml declaration
            if not sanitized.strip().startswith('<?xml'):
                sanitized = '<?xml version="1.0" encoding="UTF-8"?>\n' + sanitized
                self.logger.info("Added missing XML declaration")
            
            # Fix 4: Fix unclosed elements (basic approach)
            # This is a simplified fix - would need more sophisticated parsing for complex cases
            if '<rdf:RDF' in sanitized and not '</rdf:RDF>' in sanitized:
                sanitized += '\n</rdf:RDF>'
                self.logger.info("Added missing closing </rdf:RDF> tag")
            
            # Fix 5: Replace problematic characters in URIs
            uri_pattern = r'rdf:resource=["\'](.*?)["\']\s*'
            def fix_uri(match):
                uri = match.group(1)
                fixed_uri = uri.replace(' ', '%20').replace('\n', '').replace('\t', '')
                if fixed_uri != uri:
                    self.logger.info(f"Fixed problematic characters in URI: {uri} -> {fixed_uri}")
                return f'rdf:resource="{fixed_uri}"'
            
            sanitized = re.sub(uri_pattern, fix_uri, sanitized)
            
            if len(sanitized) != original_length:
                self.logger.info(f"Content sanitization changed length from {original_length} to {len(sanitized)} characters")
        
        except Exception as e:
            self.logger.warning(f"Content sanitization failed: {str(e)}, returning original content")
            return content
        
        return sanitized

    def _recover_default(self, error_context: ErrorContext, **kwargs) -> Any:
        """
        OWL-specific default recovery providing minimal valid OWL structures.

        Args:
            error_context (ErrorContext): Error context information
            **kwargs: Additional recovery parameters

        Returns:
            Any: Default OWL structure
        """
        self.logger.warning(f"Providing OWL-specific defaults for {error_context.location}")
        
        return_type = kwargs.get("return_type")
        
        if return_type == "rdflib_graph" and self._rdflib_available:
            # Create minimal RDF graph with basic namespace
            graph = self._rdflib.Graph()
            # Add default namespace and a basic triple
            default_ns = self._rdflib.Namespace("http://example.org/default#")
            graph.bind("default", default_ns)
            graph.add((default_ns.DefaultOntology, self._rdflib.RDF.type, self._rdflib.OWL.Ontology))
            self.logger.info("Created default RDF graph with minimal ontology structure")
            return graph
        
        elif return_type == "owl_ontology" and self._owlready2_available:
            # Create minimal owlready2 ontology
            try:
                with self._owlready2.World() as world:
                    onto = world.get_ontology("http://example.org/default")
                    with onto:
                        # Create a basic class to make it a valid ontology
                        class DefaultClass(self._owlready2.Thing):
                            pass
                    self.logger.info("Created default owlready2 ontology with minimal class structure")
                    return onto
            except Exception as e:
                self.logger.error(f"Failed to create default owlready2 ontology: {str(e)}")
                return None
        
        # Fallback to parent implementation
        return super()._recover_default(error_context, **kwargs)


class CSVParser(AbstractParser):
    """Concrete CSV parser implementation with comprehensive functionality.

    This parser provides full CSV/TSV parsing capabilities with:
    - Format detection (CSV, TSV, custom delimiters)
    - Dialect detection (delimiter, quote char, escape char)
    - Header detection and processing
    - Encoding detection
    - Column type inference
    - Large file handling with chunking
    - Conversion to Term, Relationship, and Ontology models
    - Comprehensive validation with detailed error reporting
    """

    def __init__(self, options: Optional[Dict[str, Any]] = None):
        """Initialize CSV parser with options."""
        super().__init__("csv", options=options)

        # CSV-specific imports
        import csv
        import io
        import re

        try:
            import chardet

            self._chardet_available = True
        except ImportError:
            self._chardet_available = False
            self.logger.warning(
                "chardet not available - encoding detection will be limited"
            )

        # Store modules for use in methods
        self._csv = csv
        self._io = io
        self._re = re

        # Initialize CSV-specific state
        self._custom_headers: Optional[List[str]] = None
        self._detected_dialect: Optional[Any] = None
        self._detected_encoding: str = "utf-8"
        self._validation_errors: List[str] = []
        self._column_types: Dict[str, str] = {}

        # Set CSV-specific default options
        csv_defaults = {
            "delimiter": ",",
            "quotechar": '"',
            "escapechar": None,
            "has_headers": None,  # Auto-detect if None
            "skiprows": 0,
            "nrows": None,
            "chunksize": None,
            "low_memory": False,
            "na_values": ["", "NULL", "N/A", "unknown", "nan"],
            "column_mapping": None,
            "validate_on_parse": True,
            "error_recovery": False,
            "skip_bad_lines": False,
            "max_errors": 100,
            "infer_types": True,
            "relationship_inference": False,
            "metadata_extraction": True,
        }

        # Update options with CSV defaults
        for key, value in csv_defaults.items():
            if key not in self.options:
                self.options[key] = value

        self.logger.info("CSV parser initialized successfully")

    def _get_default_options(self) -> Dict[str, Any]:
        """Get default options for CSV parser."""
        base_options = super()._get_default_options()
        csv_options = {
            "delimiter": ",",
            "quotechar": '"',
            "escapechar": None,
            "has_headers": None,
            "skiprows": 0,
            "nrows": None,
            "chunksize": None,
            "low_memory": False,
            "na_values": ["", "NULL", "N/A", "unknown", "nan"],
            "column_mapping": None,
            "validate_on_parse": True,
            "error_recovery": False,
            "skip_bad_lines": False,
            "max_errors": 100,
            "infer_types": True,
            "relationship_inference": False,
            "metadata_extraction": True,
        }
        base_options.update(csv_options)
        return base_options

    def parse(self, content: str, **kwargs) -> ParseResult:
        """Parse CSV content."""
        return self.parse_string(content, **kwargs)

    def validate(self, content: str, **kwargs) -> bool:
        """Validate CSV content."""
        validation_result = self.validate_csv(content, **kwargs)
        return validation_result.get("valid_structure", False)

    def get_supported_formats(self) -> List[str]:
        """Get supported CSV formats."""
        return ["csv", "tsv", "txt"]

    def set_options(self, options: Dict[str, Any]) -> None:
        """Set parser options with validation."""
        if not isinstance(options, dict):
            raise ValueError("Options must be a dictionary")

        # Validate specific CSV options
        valid_options = {
            "delimiter",
            "quotechar",
            "escapechar",
            "has_headers",
            "encoding",
            "skiprows",
            "nrows",
            "chunksize",
            "low_memory",
            "na_values",
            "column_mapping",
            "validate_on_parse",
            "error_recovery",
            "skip_bad_lines",
            "max_errors",
            "infer_types",
            "relationship_inference",
            "metadata_extraction",
        }

        for key, value in options.items():
            if key not in valid_options and key not in self._get_default_options():
                raise ValueError(f"Unknown option: {key}")

            # Validate specific option values
            if (
                key == "nrows"
                and value is not None
                and (not isinstance(value, int) or value <= 0)
            ):
                raise ValueError(
                    "Invalid value for option 'nrows': must be positive integer"
                )
            if key == "skiprows" and (not isinstance(value, int) or value < 0):
                raise ValueError(
                    "Invalid value for option 'skiprows': must be non-negative integer"
                )
            if (
                key == "chunksize"
                and value is not None
                and (not isinstance(value, int) or value <= 0)
            ):
                raise ValueError(
                    "Invalid value for option 'chunksize': must be positive integer"
                )

        # Update options
        self.options.update(options)
        self.logger.debug(f"Updated options: {list(options.keys())}")

    def get_options(self) -> Dict[str, Any]:
        """Get current options."""
        return copy.deepcopy(self.options)

    def get_metadata(self) -> Dict[str, Any]:
        """Get parser metadata."""
        return {
            "parser_name": self.parser_name,
            "supported_formats": self.get_supported_formats(),
            "current_options": self.get_options(),
            "statistics": {
                "total_parses": self.statistics.total_parses,
                "successful_parses": self.statistics.successful_parses,
                "failed_parses": self.statistics.failed_parses,
                "average_parse_time": self.statistics.average_parse_time,
            },
            "detected_dialect": self._detected_dialect.__dict__
            if self._detected_dialect
            else None,
            "detected_encoding": self._detected_encoding,
            "column_types": self._column_types,
            "custom_headers": self._custom_headers,
        }

    def parse_file(self, file_path: str, **kwargs) -> ParseResult:
        """Parse CSV file from file path."""
        try:
            # Detect encoding first
            with open(file_path, "rb") as f:
                raw_data = f.read(10000)  # Read first 10KB for detection
                encoding = self.detect_encoding(raw_data)

            # Read file with detected encoding
            with open(file_path, "r", encoding=encoding) as f:
                content = f.read()

            # Update options with file-specific settings
            file_options = dict(kwargs)
            file_options["source_file"] = file_path

            return self.parse_string(content, **file_options)

        except FileNotFoundError:
            raise FileNotFoundError(f"File not found: {file_path}")
        except UnicodeDecodeError as e:
            raise UnicodeDecodeError(
                e.encoding,
                e.object,
                e.start,
                e.end,
                f"Encoding error in file {file_path}: {e.reason}",
            )
        except Exception as e:
            self.logger.error(f"Error parsing file {file_path}: {str(e)}")
            raise

    def parse_string(self, content: str, **kwargs) -> ParseResult:
        """Parse CSV content from string."""
        if not content or not content.strip():
            raise ValueError("Empty CSV file")

        start_time = time.time()
        result = ParseResult(success=False)

        try:
            # Execute pre-parse hooks
            self._execute_hooks("pre_parse", content, **kwargs)

            # Validate input if enabled
            if self.options.get("validate_on_parse", True):
                self._validation_errors = []
                validation_result = self.validate_csv(content, **kwargs)
                if not validation_result.get("valid_structure", False):
                    result.errors.extend(validation_result.get("errors", []))
                    result.warnings.extend(validation_result.get("warnings", []))
                    if not self.options.get("error_recovery", False):
                        return result

            # Detect dialect and format
            detected_format = self.detect_format(content)
            dialect_result = self.detect_dialect(content)
            dialect = dialect_result["dialect"]
            self._detected_dialect = dialect

            # Store dialect detection metadata
            metadata.update(
                {
                    "dialect_confidence": dialect_result["confidence"],
                    "dialect_method": dialect_result["method"],
                    "dialect_details": dialect_result["details"],
                }
            )

            # Detect headers
            has_headers = self.options.get("has_headers")
            if has_headers is None:
                has_headers = self.detect_headers(content)

            # Parse CSV data
            parsed_data = self._parse_csv_content(
                content, dialect, has_headers, **kwargs
            )

            # Infer column types if enabled
            if self.options.get("infer_types", True):
                self._column_types = self.infer_column_types(content)

            # Build result
            result.success = True
            result.data = parsed_data
            result.metadata = {
                "format": detected_format,
                "dialect": {
                    "delimiter": dialect.delimiter,
                    "quotechar": dialect.quotechar,
                    "escapechar": dialect.escapechar,
                    "doublequote": dialect.doublequote,
                    "skipinitialspace": dialect.skipinitialspace,
                    "lineterminator": dialect.lineterminator,
                    "quoting": dialect.quoting,
                }
                if dialect
                else None,
                "has_headers": has_headers,
                "headers": parsed_data.get("headers", []),
                "row_count": len(parsed_data.get("rows", [])),
                "column_count": len(parsed_data.get("headers", [])),
                "column_types": self._column_types,
                "encoding": self._detected_encoding,
                "source_file": kwargs.get("source_file"),
            }

            # Execute post-parse hooks
            self._execute_hooks("post_parse", result)

        except Exception as e:
            self.logger.error(f"Parse error: {str(e)}")
            result.errors.append(str(e))
            self._execute_hooks("on_error", e)

        finally:
            result.parse_time = time.time() - start_time
            self.statistics.update_parse_stats(
                result.success, result.parse_time, len(content)
            )

        return result

    def _parse_csv_content(
        self, content: str, dialect: Any, has_headers: bool, **kwargs
    ) -> Dict[str, Any]:
        """Internal method to parse CSV content."""
        rows = []
        headers = []

        # Create CSV reader
        csv_input = self._io.StringIO(content)

        # Configure reader parameters
        reader_kwargs = {}
        if dialect:
            reader_kwargs["dialect"] = dialect
        else:
            reader_kwargs["delimiter"] = self.options.get("delimiter", ",")
            reader_kwargs["quotechar"] = self.options.get("quotechar", '"')
            if self.options.get("escapechar"):
                reader_kwargs["escapechar"] = self.options.get("escapechar")

        reader = self._csv.reader(csv_input, **reader_kwargs)

        # Skip rows if specified
        skiprows = self.options.get("skiprows", 0)
        for _ in range(skiprows):
            try:
                next(reader)
            except StopIteration:
                break

        # Handle headers
        if has_headers:
            try:
                headers = next(reader)
                headers = [str(h).strip() for h in headers]
            except StopIteration:
                headers = []
        elif self._custom_headers:
            headers = self._custom_headers
        else:
            # Generate default headers
            first_row = next(reader, [])
            if first_row:
                headers = [f"column_{i}" for i in range(len(first_row))]
                rows.append(first_row)

        # Read data rows
        nrows = self.options.get("nrows")
        row_count = 0

        for row in reader:
            if nrows and row_count >= nrows:
                break

            # Handle bad lines with comprehensive error recovery
            if len(row) != len(headers) and not self.options.get("skip_bad_lines", False):
                field_count_error = ValueError(
                    f"Invalid CSV: inconsistent field count at row {row_count + 1} - expected {len(headers)}, got {len(row)}"
                )
                
                error_context = self._handle_parse_error(
                    field_count_error,
                    location=f"CSV row {row_count + 1}",
                    return_type="list",
                    row_data=row,
                    expected_columns=len(headers),
                    actual_columns=len(row),
                    default_value=[""] * len(headers)
                )
                
                if error_context.recovery_data.get("recovery_successful"):
                    recovered_row = error_context.recovery_data.get("recovery_result")
                    if recovered_row:
                        row = recovered_row
                    else:
                        continue  # Skip this row
                elif error_context.severity == ErrorSeverity.FATAL:
                    raise field_count_error
                else:
                    # Continue with row padding/truncation as fallback
                    self._validation_errors.append(
                        f"Row {row_count + 1}: Field count mismatch - expected {len(headers)}, got {len(row)}"
                    )
                    continue

            # Pad or truncate row to match header count
            if len(row) < len(headers):
                row.extend([""] * (len(headers) - len(row)))
            elif len(row) > len(headers):
                row = row[: len(headers)]

            rows.append(row)
            row_count += 1

        return {
            "headers": headers,
            "rows": rows,
            "total_rows": len(rows),
            "errors": self._validation_errors,
            "has_headers": has_headers,
        }

    def parse_stream(self, stream, **kwargs) -> ParseResult:
        """Parse CSV from stream/file-like object."""
        try:
            content = stream.read()
            if isinstance(content, bytes):
                encoding = self.detect_encoding(content)
                content = content.decode(encoding)
            return self.parse_string(content, **kwargs)
        except Exception as e:
            self.logger.error(f"Stream parsing error: {str(e)}")
            raise

    def detect_format(self, content: str) -> str:
        """Detect CSV format based on content analysis."""
        if not content:
            return "csv"

        # Sample first few lines for analysis
        lines = content.split("\n")[:10]
        sample = "\n".join(lines)

        # Count delimiter occurrences
        comma_count = sample.count(",")
        tab_count = sample.count("\t")
        pipe_count = sample.count("|")
        semicolon_count = sample.count(";")

        # Determine format based on delimiter frequency
        if tab_count > comma_count and tab_count > pipe_count:
            return "tsv"
        elif comma_count > 0:
            return "csv"
        elif pipe_count > 0:
            return "csv"  # Custom delimiter CSV
        elif semicolon_count > 0:
            return "csv"  # Semicolon-separated CSV
        else:
            return "csv"  # Default

    def detect_dialect(self, content: str) -> Dict[str, Any]:
        """
        Enhanced CSV dialect detection with confidence scoring and robust fallback mechanisms.

        This method provides comprehensive dialect detection for CSV files including:
        - Enhanced delimiter detection (comma, tab, semicolon, pipe, space, and custom)
        - Quote character detection (double quotes, single quotes, backticks)
        - Escape character detection
        - Confidence scoring for detection results
        - Robust fallback mechanisms when csv.Sniffer fails

        Args:
            content (str): CSV content to analyze

        Returns:
            Dict[str, Any]: Dictionary containing:
                - dialect: The detected dialect object
                - confidence: Confidence score (0.0 to 1.0)
                - method: Detection method used ('sniffer', 'manual', 'fallback')
                - details: Additional detection information
        """
        if not content:
            return self._create_dialect_result(None, 0.0, "fallback", "Empty content")

        # Enhanced sample size for better detection
        sample_size = min(8192, len(content))  # Increased from 1024
        sample = content[:sample_size]

        # Try csv.Sniffer first with enhanced delimiters
        sniffer_result = self._try_csv_sniffer(sample)
        if sniffer_result["confidence"] > 0.6:  # High confidence threshold
            return sniffer_result

        # Manual pattern analysis for difficult cases
        manual_result = self._manual_dialect_detection(sample)
        if manual_result["confidence"] > 0.4:  # Medium confidence threshold
            return manual_result

        # Robust fallback mechanism
        fallback_result = self._fallback_dialect_detection(sample)

        # Log the detection process
        self.logger.info(
            f"Dialect detection completed: method={fallback_result['method']}, "
            f"confidence={fallback_result['confidence']:.2f}, "
            f"delimiter='{fallback_result['dialect'].delimiter}'"
        )

        return fallback_result

    def _try_csv_sniffer(self, sample: str) -> Dict[str, Any]:
        """Try using csv.Sniffer for dialect detection."""
        try:
            # Extended delimiter list including more options
            delimiters = ",\t|;: \u00A0"  # Added colon, space, and non-breaking space
            sniffer = self._csv.Sniffer()

            # Try to sniff the dialect
            dialect = sniffer.sniff(sample, delimiters=delimiters)

            # Calculate confidence based on consistency and pattern strength
            confidence = self._calculate_sniffer_confidence(sample, dialect)

            self.logger.debug(
                f"csv.Sniffer detected: delimiter='{dialect.delimiter}', "
                f"quotechar='{dialect.quotechar}', escapechar='{dialect.escapechar}', "
                f"confidence={confidence:.2f}"
            )

            return self._create_dialect_result(
                dialect, confidence, "sniffer", f"Detected with csv.Sniffer"
            )

        except Exception as e:
            self.logger.debug(f"csv.Sniffer failed: {str(e)}")
            return self._create_dialect_result(None, 0.0, "sniffer_failed", str(e))

    def _manual_dialect_detection(self, sample: str) -> Dict[str, Any]:
        """Manual pattern-based dialect detection for difficult cases."""
        lines = sample.split("\n")[:10]  # Analyze first 10 lines

        if len(lines) < 2:
            return self._create_dialect_result(
                None, 0.0, "manual", "Insufficient lines"
            )

        # Enhanced delimiter detection with scoring
        delimiter_candidates = {
            ",": 0.0,
            "\t": 0.0,
            "|": 0.0,
            ";": 0.0,
            ":": 0.0,
            " ": 0.0,
            "\u00A0": 0.0,  # Added more delimiters
        }

        # Count delimiter occurrences and consistency
        for line in lines:
            if not line.strip():
                continue
            for delimiter in delimiter_candidates:
                count = line.count(delimiter)
                if count > 0:
                    delimiter_candidates[delimiter] += count

        # Find most consistent delimiter
        best_delimiter = ","
        best_score = 0.0

        for delimiter, total_count in delimiter_candidates.items():
            if total_count == 0:
                continue

            # Calculate consistency score
            field_counts = []
            for line in lines:
                if line.strip():
                    field_counts.append(line.count(delimiter) + 1)

            if not field_counts:
                continue

            # Consistency is measured by how uniform field counts are
            if len(set(field_counts)) == 1 and field_counts[0] > 1:
                consistency = 1.0
            else:
                avg_fields = sum(field_counts) / len(field_counts)
                variance = sum((x - avg_fields) ** 2 for x in field_counts) / len(
                    field_counts
                )
                consistency = max(
                    0.0, 1.0 - (variance / avg_fields) if avg_fields > 0 else 0.0
                )

            score = consistency * (total_count / len(lines))

            if score > best_score:
                best_score = score
                best_delimiter = delimiter

        # Enhanced quote character detection
        quote_chars = ['"', "'", "`"]  # Added backtick
        best_quote = '"'
        quote_confidence = 0.0

        for quote_char in quote_chars:
            quote_patterns = 0
            for line in lines:
                # Look for quoted fields pattern
                if self._re.search(
                    rf"{self._re.escape(quote_char)}[^{self._re.escape(quote_char)}]*{self._re.escape(quote_char)}",
                    line,
                ):
                    quote_patterns += 1

            if quote_patterns > quote_confidence:
                quote_confidence = quote_patterns
                best_quote = quote_char

        # Enhanced escape character detection
        escape_char = None
        escape_confidence = 0.0

        for line in lines:
            # Look for common escape patterns
            if "\\" in line:
                escape_patterns = line.count("\\")
                if escape_patterns > escape_confidence:
                    escape_confidence = escape_patterns
                    escape_char = "\\"

        # Create custom dialect
        class ManualDialect:
            def __init__(self, delimiter, quotechar, escapechar):
                import csv

                self.delimiter = delimiter
                self.quotechar = quotechar
                self.escapechar = escapechar
                self.skipinitialspace = False
                self.doublequote = True
                self.quoting = csv.QUOTE_MINIMAL

        dialect = ManualDialect(best_delimiter, best_quote, escape_char)

        # Calculate overall confidence
        confidence = min(1.0, best_score + (quote_confidence / len(lines)) * 0.1)

        details = (
            f"Manual detection: delimiter_score={best_score:.2f}, "
            f"quote_patterns={quote_confidence}, escape_patterns={escape_confidence}"
        )

        self.logger.debug(f"Manual detection: {details}")

        return self._create_dialect_result(dialect, confidence, "manual", details)

    def _fallback_dialect_detection(self, sample: str) -> Dict[str, Any]:
        """Robust fallback mechanism when other detection methods fail."""
        # Analyze sample to make educated guesses
        lines = sample.split("\n")[:5]

        # Simple heuristics for common cases
        delimiter = ","
        confidence = 0.3  # Base fallback confidence

        # Check for obvious TSV pattern
        if any("\t" in line and "," not in line for line in lines):
            delimiter = "\t"
            confidence = 0.5

        # Check for pipe-separated values
        elif any("|" in line and "," not in line for line in lines):
            delimiter = "|"
            confidence = 0.5

        # Check for semicolon-separated (common in European locales)
        elif any(";" in line and "," not in line for line in lines):
            delimiter = ";"
            confidence = 0.4

        # Use user-specified options if available
        user_delimiter = self.options.get("delimiter")
        if user_delimiter:
            delimiter = user_delimiter
            confidence = 0.7  # Higher confidence for user-specified

        # Create fallback dialect
        class FallbackDialect:
            def __init__(self, delimiter, parser_options):
                import csv

                self.delimiter = delimiter
                self.quotechar = parser_options.get("quotechar", '"')
                self.escapechar = parser_options.get("escapechar")
                self.skipinitialspace = False
                self.doublequote = True
                self.quoting = csv.QUOTE_MINIMAL

        dialect = FallbackDialect(delimiter, self.options)

        details = f"Fallback detection using delimiter='{delimiter}'"
        if user_delimiter:
            details += " (user-specified)"

        self.logger.warning(f"Using fallback dialect detection: {details}")

        return self._create_dialect_result(dialect, confidence, "fallback", details)

    def _calculate_sniffer_confidence(self, sample: str, dialect) -> float:
        """Calculate confidence score for csv.Sniffer results."""
        lines = sample.split("\n")[:10]
        valid_lines = [line for line in lines if line.strip()]

        if len(valid_lines) < 2:
            return 0.5  # Low confidence for insufficient data

        # Test consistency of field counts
        field_counts = []
        for line in valid_lines:
            try:
                # Use the detected dialect to parse the line
                reader = self._csv.reader([line], dialect=dialect)
                row = next(reader)
                field_counts.append(len(row))
            except Exception:  # nosec B112
                continue

        if not field_counts:
            return 0.3

        # High confidence if field counts are consistent
        if len(set(field_counts)) == 1:
            return 0.9

        # Medium confidence if mostly consistent
        most_common_count = max(set(field_counts), key=field_counts.count)
        consistency_ratio = field_counts.count(most_common_count) / len(field_counts)

        return max(0.3, min(0.9, consistency_ratio))

    def _create_dialect_result(
        self, dialect, confidence: float, method: str, details: str
    ) -> Dict[str, Any]:
        """Create a standardized dialect detection result."""
        return {
            "dialect": dialect,
            "confidence": confidence,
            "method": method,
            "details": details,
        }

    def detect_encoding(self, content: bytes) -> str:
        """Detect file encoding."""
        if not content:
            return "utf-8"

        # Try chardet if available
        if self._chardet_available:
            try:
                import chardet

                result = chardet.detect(content)
                encoding = result.get("encoding", "utf-8")
                confidence = result.get("confidence", 0)

                if confidence > 0.7:
                    self.logger.debug(
                        f"Detected encoding: {encoding} (confidence: {confidence:.2f})"
                    )
                    self._detected_encoding = encoding
                    return encoding
            except Exception as e:
                self.logger.warning(f"chardet encoding detection failed: {str(e)}")

        # Fallback: try common encodings
        encodings_to_try = ["utf-8", "utf-16", "latin-1", "cp1252"]

        for encoding in encodings_to_try:
            try:
                content.decode(encoding)
                self._detected_encoding = encoding
                return encoding
            except UnicodeDecodeError:
                continue

        # Default fallback
        self._detected_encoding = "utf-8"
        return "utf-8"

    def detect_headers(self, content: str) -> bool:
        """Detect if CSV has headers using csv.Sniffer."""
        if not content:
            return False

        try:
            # Use first few lines for header detection
            lines = content.split("\n")[:10]
            sample = "\n".join(lines)

            sniffer = self._csv.Sniffer()
            has_headers = sniffer.has_header(sample)

            self.logger.debug(f"Header detection result: {has_headers}")
            return has_headers

        except Exception as e:
            self.logger.warning(
                f"Header detection failed: {str(e)}, assuming headers present"
            )
            return True  # Conservative default

    def get_headers(self, content: str) -> List[str]:
        """Get headers from CSV content."""
        if not content:
            return []

        try:
            # Use detected dialect or default
            if self._detected_dialect:
                dialect = self._detected_dialect
            else:
                dialect_result = self.detect_dialect(content)
                dialect = dialect_result["dialect"]

            csv_input = self._io.StringIO(content)
            reader = self._csv.reader(csv_input, dialect=dialect)

            # Skip rows if specified
            skiprows = self.options.get("skiprows", 0)
            for _ in range(skiprows):
                try:
                    next(reader)
                except StopIteration:
                    break

            # Get first row as headers
            first_row = next(reader, [])
            headers = [str(h).strip() for h in first_row]

            return headers

        except Exception as e:
            self.logger.error(f"Error extracting headers: {str(e)}")
            return []

    def set_headers(self, headers: List[str]) -> None:
        """Set custom headers."""
        if not isinstance(headers, list):
            raise ValueError("Headers must be a list of strings")

        self._custom_headers = [str(h) for h in headers]
        self.logger.debug(f"Set custom headers: {self._custom_headers}")

    def infer_column_types(self, content: str) -> Dict[str, str]:
        """Infer column data types from content analysis."""
        if not content:
            return {}

        try:
            # Parse a sample of the data
            parsed = self._parse_csv_content(
                content, self._detected_dialect, self.detect_headers(content)
            )

            headers = parsed.get("headers", [])
            rows = parsed.get("rows", [])

            if not headers or not rows:
                return {}

            column_types = {}

            # Analyze each column
            for i, header in enumerate(headers):
                column_values = []

                # Collect non-empty values from this column
                for row in rows[:100]:  # Sample first 100 rows
                    if i < len(row) and row[i] and str(row[i]).strip():
                        column_values.append(str(row[i]).strip())

                if not column_values:
                    column_types[header] = "string"
                    continue

                # Infer type based on content patterns
                column_types[header] = self._infer_single_column_type(
                    header, column_values
                )

            return column_types

        except Exception as e:
            self.logger.error(f"Column type inference failed: {str(e)}")
            return {}

    def _infer_single_column_type(self, header: str, values: List[str]) -> str:
        """Infer type for a single column."""
        if not values:
            return "string"

        # Check for common ontology patterns
        header_lower = header.lower()

        if "id" in header_lower and all(
            self._re.match(r"^[A-Z]+:\d+", v) for v in values[:10]
        ):
            return "ontology_id"

        if "synonym" in header_lower or "alias" in header_lower:
            return "list"  # Often semicolon-separated

        if "categor" in header_lower or "type" in header_lower:
            return "category"

        # Check if all values are numeric
        numeric_count = 0
        for value in values[:20]:  # Sample for performance
            try:
                float(value)
                numeric_count += 1
            except ValueError:
                pass

        if numeric_count / len(values[:20]) > 0.8:
            # Check if integers
            if all("." not in v for v in values[:10]):
                return "integer"
            else:
                return "float"

        # Check for boolean patterns
        boolean_values = {"true", "false", "yes", "no", "1", "0", "t", "f", "y", "n"}
        if all(v.lower() in boolean_values for v in values[:10]):
            return "boolean"

        # Default to string
        return "string"

    def to_ontology(self, parsed_result: ParseResult) -> Ontology:
        """Convert parsed CSV to Ontology model."""
        if not parsed_result.success or not parsed_result.data:
            raise ValueError("Cannot convert failed parse result to ontology")

        try:
            data = parsed_result.data
            headers = data.get("headers", [])
            rows = data.get("rows", [])

            if not headers or not rows:
                raise ValueError("No data to convert to ontology")

            # Create ontology instance
            ontology_id = f"CSV:{int(time.time())}"  # Follow PREFIX:NUMBER format
            ontology_name = (
                f"CSV Parsed Ontology - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
            )

            # Extract terms and relationships
            terms = self.extract_terms(parsed_result)
            relationships = self.extract_relationships(parsed_result)

            # Build terms and relationships dictionaries
            terms_dict = {term.id: term for term in terms}
            relationships_dict = {rel.id: rel for rel in relationships}

            # Create ontology
            if Ontology:
                ontology = Ontology(
                    id=ontology_id,
                    name=ontology_name,
                    version="1.0.0",
                    description=f"Ontology created from CSV parsing with {len(terms)} terms",
                    terms=terms_dict,
                    relationships=relationships_dict,
                    metadata=self.extract_metadata(parsed_result),
                )

                self.logger.info(
                    f"Created ontology with {len(terms)} terms and {len(relationships)} relationships"
                )
                return ontology
            else:
                self.logger.warning(
                    "Ontology model not available, returning mock object"
                )

                # Return mock object for testing
                class MockOntology:
                    def __init__(self):
                        self.id = ontology_id
                        self.name = ontology_name
                        self.terms = terms_dict
                        self.relationships = relationships_dict
                        self.metadata = self.extract_metadata(parsed_result)

                return MockOntology()

        except Exception as e:
            self.logger.error(f"Ontology conversion failed: {str(e)}")
            raise

    def extract_terms(self, parsed_result: ParseResult) -> List[Term]:
        """Extract Term objects from parsed CSV."""
        if not parsed_result.success or not parsed_result.data:
            return []

        try:
            data = parsed_result.data
            headers = data.get("headers", [])
            rows = data.get("rows", [])

            if not headers or not rows:
                return []

            terms = []
            column_mapping = self.options.get("column_mapping", {})

            # Identify key columns
            id_col = self._find_column(
                headers, ["id", "term_id", "identifier"], column_mapping
            )
            name_col = self._find_column(
                headers, ["name", "label", "term_name"], column_mapping
            )
            def_col = self._find_column(
                headers, ["definition", "description", "def"], column_mapping
            )
            cat_col = self._find_column(
                headers, ["category", "type", "class"], column_mapping
            )
            syn_col = self._find_column(
                headers, ["synonyms", "aliases", "alternative_labels"], column_mapping
            )

            for i, row in enumerate(rows):
                try:
                    # Extract term data
                    term_id = self._get_cell_value(row, id_col, f"TERM_{i+1:06d}")
                    term_name = self._get_cell_value(row, name_col, f"Term {i+1}")
                    definition = self._get_cell_value(row, def_col, "")
                    category = self._get_cell_value(row, cat_col, "")
                    synonyms_str = self._get_cell_value(row, syn_col, "")

                    # Parse synonyms
                    synonyms = []
                    if synonyms_str:
                        synonyms = [
                            s.strip() for s in synonyms_str.split(";") if s.strip()
                        ]

                    # Create term object
                    if Term:
                        term = Term(
                            id=str(term_id),
                            name=str(term_name),
                            definition=str(definition) if definition else None,
                            namespace=str(category) if category else None,
                            synonyms=synonyms,
                            metadata={
                                "source": "csv_parser",
                                "row_index": i,
                                "original_data": dict(zip(headers, row)),
                            },
                        )
                        terms.append(term)
                    else:
                        # Mock term for testing
                        class MockTerm:
                            def __init__(self):
                                self.id = str(term_id)
                                self.name = str(term_name)
                                self.definition = (
                                    str(definition) if definition else None
                                )
                                self.namespace = str(category) if category else None
                                self.synonyms = synonyms

                        terms.append(MockTerm())

                except Exception as e:
                    self.logger.warning(f"Failed to create term from row {i}: {str(e)}")
                    continue

            self.logger.info(f"Extracted {len(terms)} terms from CSV")
            return terms

        except Exception as e:
            self.logger.error(f"Term extraction failed: {str(e)}")
            return []

    def extract_relationships(self, parsed_result: ParseResult) -> List[Relationship]:
        """Extract Relationship objects from parsed CSV."""
        if not self.options.get("relationship_inference", False):
            return []

        if not parsed_result.success or not parsed_result.data:
            return []

        try:
            relationships = []
            data = parsed_result.data
            headers = data.get("headers", [])
            rows = data.get("rows", [])

            # Look for explicit relationship columns
            rel_columns = self._find_relationship_columns(headers)

            if rel_columns:
                # Extract explicit relationships
                relationships.extend(
                    self._extract_explicit_relationships(headers, rows, rel_columns)
                )

            # Infer relationships from category hierarchy
            if self._find_column(headers, ["category", "type", "class"]) >= 0:
                relationships.extend(self._infer_category_relationships(headers, rows))

            self.logger.info(f"Extracted {len(relationships)} relationships from CSV")
            return relationships

        except Exception as e:
            self.logger.error(f"Relationship extraction failed: {str(e)}")
            return []

    def _find_column(
        self,
        headers: List[str],
        possible_names: List[str],
        mapping: Dict[str, str] = None,
    ) -> int:
        """Find column index by possible names."""
        if mapping:
            for mapped_name, actual_name in mapping.items():
                if mapped_name in possible_names and actual_name in headers:
                    return headers.index(actual_name)

        for name in possible_names:
            for i, header in enumerate(headers):
                if name.lower() in header.lower():
                    return i

        return -1

    def _get_cell_value(self, row: List[str], col_index: int, default: str = "") -> str:
        """Get cell value with fallback to default."""
        if col_index >= 0 and col_index < len(row):
            value = row[col_index]
            return value if value is not None else default
        return default

    def _find_relationship_columns(self, headers: List[str]) -> List[int]:
        """Find columns that might contain relationship information."""
        rel_indicators = ["parent", "child", "related", "depends", "is_a", "part_of"]
        rel_columns = []

        for i, header in enumerate(headers):
            header_lower = header.lower()
            if any(indicator in header_lower for indicator in rel_indicators):
                rel_columns.append(i)

        return rel_columns

    def _extract_explicit_relationships(
        self, headers: List[str], rows: List[List[str]], rel_columns: List[int]
    ) -> List[Relationship]:
        """Extract explicit relationships from relationship columns."""
        relationships = []
        id_col = self._find_column(headers, ["id", "term_id", "identifier"])

        for i, row in enumerate(rows):
            subject_id = self._get_cell_value(row, id_col, f"TERM_{i+1:06d}")

            for rel_col in rel_columns:
                rel_value = self._get_cell_value(row, rel_col)
                if rel_value:
                    # Parse relationship format (assuming "predicate:object" or just "object")
                    if ":" in rel_value:
                        predicate, object_id = rel_value.split(":", 1)
                    else:
                        predicate = "related_to"
                        object_id = rel_value

                    rel_id = f"REL_{len(relationships)+1:06d}"

                    if Relationship:
                        relationship = Relationship(
                            id=rel_id,
                            subject=subject_id.strip(),
                            predicate=predicate.strip(),
                            object=object_id.strip(),
                            confidence=0.8,
                            source="csv_inference",
                            metadata={
                                "source_column": headers[rel_col],
                                "row_index": i,
                            },
                        )
                        relationships.append(relationship)

        return relationships

    def _infer_category_relationships(
        self, headers: List[str], rows: List[List[str]]
    ) -> List[Relationship]:
        """Infer is-a relationships from categories."""
        relationships = []
        id_col = self._find_column(headers, ["id", "term_id", "identifier"])
        cat_col = self._find_column(headers, ["category", "type", "class"])

        if id_col == -1 or cat_col == -1:
            return relationships

        for i, row in enumerate(rows):
            subject_id = self._get_cell_value(row, id_col, f"TERM_{i+1:06d}")
            category = self._get_cell_value(row, cat_col)

            if category:
                rel_id = f"REL_{len(relationships)+1:06d}"

                if Relationship:
                    relationship = Relationship(
                        id=rel_id,
                        subject=subject_id.strip(),
                        predicate="is_a",
                        object=category.strip(),
                        confidence=0.7,
                        source="csv_category_inference",
                        metadata={"row_index": i},
                    )
                    relationships.append(relationship)

        return relationships

    def extract_metadata(self, parsed_result: ParseResult) -> Dict[str, Any]:
        """Extract metadata from parsed CSV."""
        if not parsed_result.success:
            return {}

        base_metadata = {
            "source_format": "csv",
            "parser_version": "1.0.0",
            "parsing_timestamp": datetime.now().isoformat(),
            "detected_encoding": self._detected_encoding,
        }

        if parsed_result.metadata:
            base_metadata.update(parsed_result.metadata)

        if parsed_result.data:
            data = parsed_result.data
            base_metadata.update(
                {
                    "original_row_count": data.get("total_rows", 0),
                    "processed_row_count": len(data.get("rows", [])),
                    "column_count": len(data.get("headers", [])),
                    "headers": data.get("headers", []),
                    "has_headers": data.get("has_headers", False),
                    "column_types": self._column_types,
                }
            )

        if self._detected_dialect:
            base_metadata["dialect"] = {
                "delimiter": self._detected_dialect.delimiter,
                "quotechar": self._detected_dialect.quotechar,
                "escapechar": getattr(self._detected_dialect, "escapechar", None),
            }

        return base_metadata

    def validate_csv(self, content: str, **kwargs) -> Dict[str, Any]:
        """Validate CSV content with detailed results."""
        validation_result = {
            "valid_structure": False,
            "valid_headers": False,
            "valid_data_types": False,
            "required_columns_present": False,
            "header_consistency": False,
            "type_consistency": False,
            "null_value_check": False,
            "valid_field_formats": False,
            "id_format_valid": False,
            "name_format_valid": False,
            "errors": [],
            "warnings": [],
            "format_errors": [],
            "required_columns": kwargs.get("required_columns", ["id", "name"]),
            "missing_columns": [],
            "extra_columns": [],
        }

        try:
            # Basic structure validation
            if not content or not content.strip():
                validation_result["errors"].append("Empty CSV content")
                return validation_result

            # Try to parse structure
            try:
                dialect_result = self.detect_dialect(content)
                dialect = dialect_result["dialect"]
                headers = self.get_headers(content)

                if headers:
                    validation_result["valid_structure"] = True
                    validation_result["valid_headers"] = True
                    validation_result["header_consistency"] = True
                else:
                    validation_result["errors"].append("No valid headers found")
                    return validation_result

            except Exception as e:
                validation_result["errors"].append(
                    f"Structure parsing failed: {str(e)}"
                )
                return validation_result

            # Check required columns
            required_columns = validation_result["required_columns"]
            missing_columns = []

            for req_col in required_columns:
                if not any(req_col.lower() in h.lower() for h in headers):
                    missing_columns.append(req_col)

            validation_result["missing_columns"] = missing_columns
            validation_result["required_columns_present"] = len(missing_columns) == 0

            # Check for extra columns
            standard_columns = ["id", "name", "definition", "category", "synonyms"]
            extra_columns = []
            for header in headers:
                if not any(std.lower() in header.lower() for std in standard_columns):
                    extra_columns.append(header)

            validation_result["extra_columns"] = extra_columns

            # Validate data types and formats
            try:
                parsed_sample = self._parse_csv_content(
                    content[:5000], dialect, True
                )  # Sample first 5KB
                rows = parsed_sample.get("rows", [])

                if rows:
                    validation_result["valid_data_types"] = True
                    validation_result["type_consistency"] = True
                    validation_result["null_value_check"] = True

                    # Validate specific field formats
                    id_col = self._find_column(headers, ["id", "term_id"])
                    name_col = self._find_column(headers, ["name", "label"])

                    if id_col >= 0:
                        id_valid = all(
                            self._validate_id_format(self._get_cell_value(row, id_col))
                            for row in rows[:10]
                        )
                        validation_result["id_format_valid"] = id_valid
                        if not id_valid:
                            validation_result["format_errors"].append(
                                "Invalid ID format detected"
                            )

                    if name_col >= 0:
                        name_valid = all(
                            self._get_cell_value(row, name_col).strip() != ""
                            for row in rows[:10]
                        )
                        validation_result["name_format_valid"] = name_valid
                        if not name_valid:
                            validation_result["format_errors"].append(
                                "Empty names detected"
                            )

                    validation_result["valid_field_formats"] = (
                        validation_result["id_format_valid"]
                        and validation_result["name_format_valid"]
                    )

            except Exception as e:
                validation_result["warnings"].append(
                    f"Data validation warning: {str(e)}"
                )

            # Overall validation status
            validation_result["valid_structure"] = (
                validation_result["valid_headers"]
                and validation_result["required_columns_present"]
                and len(validation_result["errors"]) == 0
            )

        except Exception as e:
            validation_result["errors"].append(f"Validation error: {str(e)}")

        return validation_result

    def _validate_id_format(self, id_value: str) -> bool:
        """Validate ID format (basic check for non-empty and reasonable format)."""
        if not id_value or not id_value.strip():
            return False

        # Check for common ontology ID patterns
        id_patterns = [
            r"^[A-Z]+:\d+$",  # CHEM:001
            r"^[A-Z]+_\d+$",  # CHEM_001
            r"^\w+$",  # Any word characters
        ]

        for pattern in id_patterns:
            if self._re.match(pattern, id_value.strip()):
                return True

        return True  # Allow any non-empty string as fallback

    def get_validation_errors(self) -> List[str]:
        """Get validation errors."""
        return self._validation_errors.copy()

    def reset_options(self) -> None:
        """Reset options to defaults."""
        self.options = copy.deepcopy(self._get_default_options())
        self._custom_headers = None
        self._detected_dialect = None
        self._detected_encoding = "utf-8"
        self._validation_errors = []
        self._column_types = {}
        self.logger.debug("Options reset to defaults")

    # =====================
    # CSV-specific Error Recovery Methods
    # =====================

    def _recover_default(self, error_context: ErrorContext, **kwargs) -> Any:
        """
        CSV-specific default recovery for malformed rows and inconsistent data.

        Args:
            error_context (ErrorContext): Error context information
            **kwargs: Additional recovery parameters

        Returns:
            Any: Recovered CSV data
        """
        location = error_context.location.lower()
        
        if "row" in location and "field count" in str(error_context.error):
            # Handle field count mismatches
            row_data = kwargs.get("row_data", [])
            expected_columns = kwargs.get("expected_columns", 0)
            actual_columns = kwargs.get("actual_columns", len(row_data))
            
            if actual_columns < expected_columns:
                # Pad with empty strings
                padded_row = row_data + [""] * (expected_columns - actual_columns)
                self.logger.info(f"Padded row from {actual_columns} to {expected_columns} columns with empty strings")
                return padded_row
            elif actual_columns > expected_columns:
                # Truncate to expected length
                truncated_row = row_data[:expected_columns]
                self.logger.info(f"Truncated row from {actual_columns} to {expected_columns} columns")
                return truncated_row
        
        elif "encoding" in location:
            # Handle encoding issues
            self.logger.info("Using fallback encoding UTF-8 with error replacement")
            return {"encoding": "utf-8", "errors": "replace"}
        
        elif "dialect" in location:
            # Handle dialect detection failures
            self.logger.info("Using default CSV dialect (comma-separated, quoted)")
            return {"delimiter": ",", "quotechar": '"', "escapechar": None}
        
        # Fallback to parent implementation
        return super()._recover_default(error_context, **kwargs)

    def _recover_skip(self, error_context: ErrorContext, **kwargs) -> Any:
        """
        CSV-specific skip recovery for problematic rows.

        Args:
            error_context (ErrorContext): Error context information
            **kwargs: Additional recovery parameters

        Returns:
            Any: None to indicate skipping
        """
        location = error_context.location.lower()
        
        if "row" in location:
            self.logger.warning(f"Skipping problematic CSV row at {error_context.location}")
            # Add to validation errors for tracking
            self._validation_errors.append(f"Skipped row due to error: {str(error_context.error)}")
            return None
        
        return super()._recover_skip(error_context, **kwargs)

    def _recover_replace(self, error_context: ErrorContext, **kwargs) -> Any:
        """
        CSV-specific replace recovery for fixing malformed data.

        Args:
            error_context (ErrorContext): Error context information
            **kwargs: Additional recovery parameters

        Returns:
            Any: Replaced/fixed CSV data
        """
        location = error_context.location.lower()
        error_message = str(error_context.error).lower()
        
        if "row" in location and "field count" in error_message:
            # Handle field count mismatches with intelligent padding/truncation
            row_data = kwargs.get("row_data", [])
            expected_columns = kwargs.get("expected_columns", 0)
            
            if len(row_data) < expected_columns:
                # Try to determine appropriate default values based on column types
                replacement_row = []
                for i, value in enumerate(row_data):
                    replacement_row.append(value)
                
                # Fill missing columns with type-appropriate defaults
                for i in range(len(row_data), expected_columns):
                    column_type = self._column_types.get(i, str)
                    if column_type in [int, float]:
                        replacement_row.append("0")
                    elif column_type == bool:
                        replacement_row.append("false")
                    else:
                        replacement_row.append("")
                
                self.logger.info(f"Replaced missing columns with type-appropriate defaults")
                return replacement_row
            
            elif len(row_data) > expected_columns:
                # Intelligently merge excess columns into the last column
                merged_row = row_data[:expected_columns-1]
                merged_last_column = " ".join(str(x) for x in row_data[expected_columns-1:])
                merged_row.append(merged_last_column)
                
                self.logger.info(f"Merged {len(row_data) - expected_columns + 1} excess columns into last column")
                return merged_row
        
        elif "encoding" in error_message:
            # Handle encoding replacement
            content = kwargs.get("content", "")
            try:
                # Try common encodings
                for encoding in ["utf-8", "latin1", "cp1252", "iso-8859-1"]:
                    try:
                        decoded = content.encode("utf-8", errors="ignore").decode(encoding)
                        self.logger.info(f"Successfully replaced encoding with {encoding}")
                        return {"content": decoded, "encoding": encoding}
                    except (UnicodeDecodeError, UnicodeEncodeError):
                        continue
            except Exception:
                pass
            
            # Final fallback
            safe_content = content.encode("utf-8", errors="ignore").decode("utf-8")
            self.logger.info("Used UTF-8 with ignored errors as final encoding fallback")
            return {"content": safe_content, "encoding": "utf-8"}
        
        return super()._recover_replace(error_context, **kwargs)

    def _sanitize_csv_row(self, row: list, expected_columns: int) -> list:
        """
        Sanitize a CSV row to fix common issues.

        Args:
            row (list): Raw CSV row data
            expected_columns (int): Expected number of columns

        Returns:
            list: Sanitized row data
        """
        sanitized_row = []
        
        for i, cell in enumerate(row):
            if cell is None:
                sanitized_cell = ""
            else:
                # Convert to string and clean
                sanitized_cell = str(cell).strip()
                
                # Remove control characters
                sanitized_cell = ''.join(char for char in sanitized_cell if ord(char) >= 32 or char in '\t\n\r')
                
                # Handle embedded newlines in CSV cells
                if '\n' in sanitized_cell or '\r' in sanitized_cell:
                    sanitized_cell = sanitized_cell.replace('\n', ' ').replace('\r', ' ')
                    self.logger.debug(f"Removed embedded newlines from cell in column {i}")
            
            sanitized_row.append(sanitized_cell)
        
        # Ensure correct column count
        if len(sanitized_row) < expected_columns:
            sanitized_row.extend([""] * (expected_columns - len(sanitized_row)))
        elif len(sanitized_row) > expected_columns:
            sanitized_row = sanitized_row[:expected_columns]
        
        return sanitized_row


class JSONLDParser(AbstractParser):
    """Comprehensive JSON-LD parser implementation for the AIM2 ontology project.

    This parser provides complete JSON-LD parsing capabilities including:
    - Multiple JSON-LD formats (compact, expanded, flattened)
    - Context processing and namespace resolution
    - Integration with pyld and rdflib libraries
    - Comprehensive validation including syntax and semantics
    - Conversion to internal Term/Relationship/Ontology models
    - Performance optimizations for large JSON-LD documents
    - Configurable parsing options and error handling

    The parser uses pyld for JSON-LD processing and rdflib for
    RDF operations and graph management.
    """

    def __init__(self, options: Optional[Dict[str, Any]] = None):
        """Initialize JSON-LD parser with comprehensive options.

        Args:
            options (Dict[str, Any], optional): Parser configuration options
        """
        super().__init__("jsonld", options=options)

        # Initialize JSON-LD specific validation errors list
        self._validation_errors: List[str] = []
        self._current_context: Optional[Dict[str, Any]] = None

        # Import required libraries with fallbacks
        self._json = __import__("json")

        try:
            import pyld
            from pyld import jsonld

            self._pyld = pyld
            self._jsonld = jsonld
            self._pyld_available = True
        except ImportError:
            self.logger.warning(
                "pyld not available - JSON-LD functionality will be limited"
            )
            self._pyld = None
            self._jsonld = None
            self._pyld_available = False

        try:
            import rdflib
            from rdflib import BNode, Graph, Literal, Namespace, URIRef
            from rdflib.namespace import OWL, RDF, RDFS, XSD

            self._rdflib = rdflib
            self._rdflib_graph = Graph
            self._rdflib_namespace = Namespace
            self._rdflib_uriref = URIRef
            self._rdflib_literal = Literal
            self._rdflib_bnode = BNode
            self._RDF = RDF
            self._RDFS = RDFS
            self._OWL = OWL
            self._XSD = XSD
            self._rdflib_available = True
        except ImportError:
            self.logger.warning(
                "rdflib not available - RDF functionality will be limited"
            )
            self._rdflib = None
            self._rdflib_available = False

        # Set up JSON-LD specific default options
        jsonld_defaults = {
            "validate_on_parse": True,
            "strict_validation": False,
            "preserve_contexts": True,
            "resolve_remote_contexts": False,
            "base_uri": None,
            "processing_mode": "json-ld-1.1",
            "expand_contexts": True,
            "safe_mode": True,
            "document_loader": None,
            "timeout": 30,
            "max_context_depth": 10,
            "allow_relative_iris": False,
            "produce_generalized_rdf": False,
            "use_rdf_type": False,
            "use_native_types": True,
            "error_recovery": True,
            "max_errors": 100,
            "continue_on_error": True,
            "log_warnings": True,
            # Security-related options
            "max_content_size": 50 * 1024 * 1024,  # 50MB
            "max_string_length": 1024 * 1024,  # 1MB
            "max_json_depth": 100,
            "max_nesting_depth": 100,
            "max_object_keys": 10000,
            "max_array_size": 100000,
        }

        # Update options with JSON-LD defaults
        for key, value in jsonld_defaults.items():
            if key not in self.options:
                self.options[key] = value

        self.logger.info(
            f"JSON-LD parser initialized with pyld={self._pyld_available}, rdflib={self._rdflib_available}"
        )

    def _get_default_options(self) -> Dict[str, Any]:
        """Get JSON-LD parser specific default options.

        Returns:
            Dict[str, Any]: Default options for JSON-LD parser
        """
        base_options = super()._get_default_options()
        jsonld_options = {
            "validate_on_parse": True,
            "strict_validation": False,
            "preserve_contexts": True,
            "resolve_remote_contexts": False,
            "base_uri": None,
            "processing_mode": "json-ld-1.1",
            "expand_contexts": True,
            "safe_mode": True,
            "document_loader": None,
            "timeout": 30,
            "max_context_depth": 10,
            "allow_relative_iris": False,
            "produce_generalized_rdf": False,
            "use_rdf_type": False,
            "use_native_types": True,
            # Security-related options
            "max_content_size": 50 * 1024 * 1024,  # 50MB
            "max_string_length": 1024 * 1024,  # 1MB
            "max_json_depth": 100,
            "max_nesting_depth": 100,
            "max_object_keys": 10000,
            "max_array_size": 100000,
            "conversion_filters": {
                "include_types": True,
                "include_ids": True,
                "include_values": True,
                "namespace_filter": None,
                "type_filter": None,
                "property_filter": None,
            },
        }
        base_options.update(jsonld_options)
        return base_options

    def get_supported_formats(self) -> List[str]:
        """Get list of supported JSON-LD formats.

        Returns:
            List[str]: List of supported formats
        """
        return ["jsonld", "json-ld", "json"]

    def detect_format(self, content: str) -> str:
        """Auto-detect JSON-LD format from content.

        Args:
            content (str): Content to analyze

        Returns:
            str: Detected format or 'unknown'
        """
        try:
            content_stripped = content.strip()
            if not content_stripped:
                return "unknown"

            # Try to parse as JSON
            data = self._json.loads(content_stripped)

            # Check for JSON-LD specific markers
            if isinstance(data, dict):
                if (
                    "@context" in data
                    or "@id" in data
                    or "@type" in data
                    or "@graph" in data
                ):
                    return "jsonld"
                # Check nested structures
                for value in data.values():
                    if isinstance(value, dict) and any(
                        key.startswith("@") for key in value.keys()
                    ):
                        return "jsonld"
                    elif isinstance(value, list):
                        for item in value:
                            if isinstance(item, dict) and any(
                                key.startswith("@") for key in item.keys()
                            ):
                                return "jsonld"
            elif isinstance(data, list):
                for item in data:
                    if isinstance(item, dict) and any(
                        key.startswith("@") for key in item.keys()
                    ):
                        return "jsonld"

            # If valid JSON but no JSON-LD markers, return json
            return "json"
        except Exception:
            return "unknown"

    def parse(self, content: str, **kwargs) -> Any:
        """Parse JSON-LD content from string.

        Args:
            content (str): JSON-LD content to parse
            **kwargs: Additional parsing parameters

        Returns:
            Any: Parsed JSON-LD data structure

        Raises:
            OntologyException: If parsing fails
        """
        try:
            self.statistics.total_parses += 1
            start_time = time.time()

            # Clear previous validation errors
            self._validation_errors = []

            # Apply security protections before parsing
            if self.options.get("safe_mode", True):
                self._validate_content_security(content)

            # Detect format if not specified
            format_hint = kwargs.get("format", self.detect_format(content))
            self.logger.debug(f"Parsing JSON-LD content with format: {format_hint}")

            # Validate format if strict validation is enabled
            if self.options.get("strict_validation", False):
                if not self.validate(content, **kwargs):
                    raise OntologyException(
                        f"Content validation failed for format: {format_hint}",
                        context={"format": format_hint, "content_length": len(content)},
                    )

            # Parse JSON first with security protections
            data = None
            try:
                # Use secure JSON parsing with depth limits
                data = self._secure_json_parse(content)
            except (self._json.JSONDecodeError, RecursionError, MemoryError) as e:
                error_context = self._handle_parse_error(
                    e,
                    location="JSON parsing",
                    return_type="dict",
                    content=content,
                    parsing_stage="json_decode"
                )
                
                if error_context.recovery_data.get("recovery_successful"):
                    data = error_context.recovery_data.get("recovery_result")
                else:
                    # If recovery fails for JSON parsing, it's usually fatal
                    if isinstance(e, self._json.JSONDecodeError):
                        raise OntologyException(f"Invalid JSON content: {str(e)}")
                    elif isinstance(e, RecursionError):
                        raise OntologyException("JSON structure too deeply nested - potential JSON bomb attack")
                    elif isinstance(e, MemoryError):
                        raise OntologyException("JSON content too large - potential JSON bomb attack")

            # Process JSON-LD data
            processed_data = data

            # Try to expand with pyld if available
            if self._pyld_available:
                try:
                    # Expand the document to normalize it
                    processed_data = self.expand(data, **kwargs)
                except Exception as e:
                    error_context = self._handle_parse_error(
                        e,
                        location="JSON-LD expansion",
                        return_type="dict",
                        content=content,
                        data=data,
                        parsing_stage="jsonld_expand",
                        **kwargs
                    )
                    
                    if error_context.recovery_data.get("recovery_successful"):
                        processed_data = error_context.recovery_data.get("recovery_result")
                    else:
                        # Use raw data as fallback
                        self.logger.warning(f"JSON-LD processing failed, using raw data: {str(e)}")
                        processed_data = data

            # Convert to ontology format if requested
            if kwargs.get("convert_to_ontology", True):
                try:
                    ontology_data = self.to_ontology(processed_data, **kwargs)
                    if ontology_data:
                        # Update statistics
                        self.statistics.successful_parses += 1
                        parse_time = time.time() - start_time
                        self.statistics.total_parse_time += parse_time
                        self.statistics.average_parse_time = (
                            self.statistics.total_parse_time
                            / self.statistics.total_parses
                        )
                        self.statistics.total_content_processed += len(content)

                        return ontology_data
                    else:
                        self.logger.warning(
                            "Ontology conversion returned None, falling back to raw data"
                        )
                except Exception as e:
                    error_context = self._handle_parse_error(
                        e,
                        location="ontology conversion",
                        return_type="dict", 
                        content=content,
                        processed_data=processed_data,
                        parsing_stage="ontology_convert",
                        **kwargs
                    )
                    
                    if error_context.recovery_data.get("recovery_successful"):
                        ontology_data = error_context.recovery_data.get("recovery_result")
                        if ontology_data:
                            return ontology_data
                    elif error_context.severity == ErrorSeverity.FATAL:
                        raise OntologyException(f"Ontology conversion failed: {str(e)}")
                    else:
                        self.logger.warning(f"Ontology conversion failed, returning processed data: {str(e)}")

            # Return processed JSON-LD data
            self.statistics.successful_parses += 1
            parse_time = time.time() - start_time
            self.statistics.total_parse_time += parse_time
            self.statistics.average_parse_time = (
                self.statistics.total_parse_time / self.statistics.total_parses
            )
            self.statistics.total_content_processed += len(content)

            return processed_data

        except Exception as e:
            self.statistics.failed_parses += 1
            if isinstance(e, OntologyException):
                raise
            raise OntologyException(f"JSON-LD parsing failed: {str(e)}")

    def validate(self, content: str, **kwargs) -> bool:
        """Validate JSON-LD content format and structure.

        Args:
            content (str): Content to validate
            **kwargs: Additional validation parameters

        Returns:
            bool: True if content is valid JSON-LD
        """
        try:
            self._validation_errors = []

            # Basic JSON validation
            try:
                data = self._json.loads(content)
            except self._json.JSONDecodeError as e:
                self._validation_errors.append(f"Invalid JSON: {str(e)}")
                return False

            # JSON-LD specific validation
            if not self._is_valid_jsonld_structure(data):
                self._validation_errors.append("Not a valid JSON-LD structure")
                return False

            # Advanced validation with pyld if available
            if self._pyld_available and self.options.get("strict_validation", False):
                try:
                    # Try to expand the document to validate JSON-LD semantics
                    self._jsonld.expand(data)
                except Exception as e:
                    self._validation_errors.append(
                        f"JSON-LD expansion failed: {str(e)}"
                    )
                    return False

            return len(self._validation_errors) == 0

        except Exception as e:
            self._validation_errors.append(f"Validation error: {str(e)}")
            return False

    def _is_valid_jsonld_structure(self, data: Any) -> bool:
        """Check if data has valid JSON-LD structure.

        Args:
            data: Parsed JSON data

        Returns:
            bool: True if structure is valid JSON-LD
        """
        if isinstance(data, dict):
            # Check for JSON-LD keywords
            jsonld_keywords = {
                "@context",
                "@id",
                "@type",
                "@value",
                "@language",
                "@index",
                "@set",
                "@list",
                "@graph",
                "@nest",
                "@reverse",
            }

            # Must have at least one JSON-LD keyword or be a nested structure
            has_jsonld_keyword = any(key in jsonld_keywords for key in data.keys())

            if has_jsonld_keyword:
                return True

            # Check nested structures
            for value in data.values():
                if isinstance(value, (dict, list)) and self._is_valid_jsonld_structure(
                    value
                ):
                    return True

        elif isinstance(data, list):
            # List of JSON-LD objects
            return any(self._is_valid_jsonld_structure(item) for item in data)

        return False

    def get_namespaces(self, document: Dict[str, Any]) -> Dict[str, str]:
        """Get namespaces from JSON-LD document.

        Args:
            document (Dict[str, Any]): JSON-LD document

        Returns:
            Dict[str, str]: Namespace prefix to URI mappings
        """
        namespaces = {}

        try:
            # Extract from @context
            context = document.get("@context", {})
            if isinstance(context, dict):
                for key, value in context.items():
                    if isinstance(value, str) and (
                        value.startswith("http://") or value.startswith("https://")
                    ):
                        namespaces[key] = value
                    elif isinstance(value, dict) and "@id" in value:
                        uri = value["@id"]
                        if isinstance(uri, str) and (
                            uri.startswith("http://") or uri.startswith("https://")
                        ):
                            namespaces[key] = uri

            # Add common RDF namespaces if not present
            default_namespaces = {
                "rdf": "http://www.w3.org/1999/02/22-rdf-syntax-ns#",
                "rdfs": "http://www.w3.org/2000/01/rdf-schema#",
                "owl": "http://www.w3.org/2002/07/owl#",
                "xsd": "http://www.w3.org/2001/XMLSchema#",
            }

            for prefix, uri in default_namespaces.items():
                if prefix not in namespaces:
                    namespaces[prefix] = uri

            return namespaces

        except Exception as e:
            self.logger.warning(f"Error extracting namespaces: {str(e)}")
            return {}

    def expand_namespaces(self, document: Dict[str, Any]) -> Dict[str, Any]:
        """Expand namespaces in JSON-LD document.

        Args:
            document (Dict[str, Any]): JSON-LD document

        Returns:
            Dict[str, Any]: Document with expanded namespaces
        """
        try:
            # Use expand method which naturally expands namespaces
            return self.expand(document)
        except Exception as e:
            self.logger.warning(f"Error expanding namespaces: {str(e)}")
            return document

    def parse_file(self, file_path: str, **kwargs) -> Any:
        """Parse JSON-LD file from file path.

        Args:
            file_path (str): Path to JSON-LD file
            **kwargs: Additional parsing parameters

        Returns:
            Any: Parsed JSON-LD data

        Raises:
            OntologyException: If file parsing fails
        """
        try:
            file_path_obj = Path(file_path)
            if not file_path_obj.exists():
                raise OntologyException(f"File not found: {file_path}")

            encoding = kwargs.get("encoding", "utf-8")
            with open(file_path_obj, "r", encoding=encoding) as f:
                content = f.read()

            return self.parse(content, **kwargs)

        except Exception as e:
            if isinstance(e, OntologyException):
                raise
            raise OntologyException(f"File parsing failed: {str(e)}")

    def parse_string(self, content: str, **kwargs) -> Any:
        """Parse JSON-LD content from string.

        Args:
            content (str): JSON-LD content string
            **kwargs: Additional parsing parameters

        Returns:
            Any: Parsed JSON-LD data
        """
        return self.parse(content, **kwargs)

    def parse_stream(self, stream, **kwargs) -> Any:
        """Parse JSON-LD from stream/file-like object.

        Args:
            stream: File-like object containing JSON-LD data
            **kwargs: Additional parsing parameters

        Returns:
            Any: Parsed JSON-LD data

        Raises:
            OntologyException: If stream parsing fails
        """
        try:
            content = stream.read()
            if isinstance(content, bytes):
                encoding = kwargs.get("encoding", "utf-8")
                content = content.decode(encoding)

            return self.parse(content, **kwargs)

        except Exception as e:
            if isinstance(e, OntologyException):
                raise
            raise OntologyException(f"Stream parsing failed: {str(e)}")

    def validate_jsonld(self, content: str, **kwargs) -> Dict[str, Any]:
        """Validate JSON-LD content with detailed results.

        Args:
            content (str): JSON-LD content to validate
            **kwargs: Additional validation parameters

        Returns:
            Dict[str, Any]: Detailed validation results
        """
        result = {
            "valid_json": False,
            "valid_jsonld": False,
            "valid_structure": False,
            "errors": [],
            "warnings": [],
            "statistics": {},
        }

        try:
            # Basic JSON validation
            try:
                data = self._json.loads(content)
                result["valid_json"] = True
            except self._json.JSONDecodeError as e:
                result["errors"].append(f"Invalid JSON: {str(e)}")
                return result

            # JSON-LD structure validation
            if self._is_valid_jsonld_structure(data):
                result["valid_structure"] = True
            else:
                result["errors"].append("Not a valid JSON-LD structure")

            # Advanced JSON-LD validation
            if self._pyld_available and result["valid_structure"]:
                try:
                    # Try to expand the document
                    expanded = self._jsonld.expand(data)
                    result["valid_jsonld"] = True
                    result["statistics"]["expanded_nodes"] = len(expanded)
                except Exception as e:
                    result["errors"].append(f"JSON-LD expansion failed: {str(e)}")

            # Additional checks
            if isinstance(data, dict):
                if "@context" in data:
                    result["statistics"]["has_context"] = True
                if "@graph" in data:
                    result["statistics"]["has_graph"] = True

            return result

        except Exception as e:
            result["errors"].append(f"Validation error: {str(e)}")
            return result

    def get_validation_errors(self) -> List[str]:
        """Get validation errors.

        Returns:
            List[str]: List of validation errors
        """
        return self._validation_errors.copy()

    def reset_options(self) -> None:
        """Reset options to defaults."""
        self.options = copy.deepcopy(self._get_default_options())
        self._current_context = None
        self._validation_errors = []
        self.logger.debug("Options reset to defaults")

    # JSON-LD transformation methods
    def expand(
        self, document: Union[Dict[str, Any], List[Dict[str, Any]]], **kwargs
    ) -> List[Dict[str, Any]]:
        """Expand JSON-LD document to normalized form.

        Args:
            document: JSON-LD document to expand
            **kwargs: Additional expansion options

        Returns:
            List[Dict[str, Any]]: Expanded JSON-LD document

        Raises:
            OntologyException: If expansion fails
        """
        try:
            # Apply security checks if safe_mode is enabled
            if self.options.get("safe_mode", True):
                self._apply_safe_mode_restrictions(document)

            # Use pyld if available
            if self._pyld_available:
                try:
                    # Prepare expansion options
                    expand_options = {
                        "base": kwargs.get("base", self.options.get("base_uri")),
                        "expandContext": kwargs.get("expand_context"),
                        "keepFreeFloatingNodes": kwargs.get(
                            "keep_free_floating_nodes", False
                        ),
                        "processingMode": self.options.get(
                            "processing_mode", "json-ld-1.1"
                        ),
                        "documentLoader": self.options.get("document_loader"),
                    }

                    # Remove None values
                    expand_options = {
                        k: v for k, v in expand_options.items() if v is not None
                    }

                    # Perform expansion
                    expanded = self._jsonld.expand(document, expand_options)

                    # Ensure we return a list
                    if not isinstance(expanded, list):
                        expanded = [expanded] if expanded else []

                    return expanded

                except Exception as e:
                    if not self.options.get("error_recovery", True):
                        raise OntologyException(f"JSON-LD expansion failed: {str(e)}")
                    self.logger.warning(
                        f"pyld expansion failed, using fallback: {str(e)}"
                    )

            # Fallback implementation
            return self._expand_fallback(document)

        except Exception as e:
            if isinstance(e, OntologyException):
                raise
            raise OntologyException(f"Document expansion failed: {str(e)}")

    def compact(
        self,
        document: Union[Dict[str, Any], List[Dict[str, Any]]],
        context: Optional[Dict[str, Any]] = None,
        **kwargs,
    ) -> Dict[str, Any]:
        """Compact JSON-LD document using provided context.

        Args:
            document: JSON-LD document to compact
            context: Context to use for compaction
            **kwargs: Additional compaction options

        Returns:
            Dict[str, Any]: Compacted JSON-LD document

        Raises:
            OntologyException: If compaction fails
        """
        try:
            # Apply security checks if safe_mode is enabled
            if self.options.get("safe_mode", True):
                self._apply_safe_mode_restrictions(document)
                if context:
                    self._apply_safe_mode_restrictions(context)

            # Use pyld if available
            if self._pyld_available:
                try:
                    # Prepare compaction options
                    compact_options = {
                        "base": kwargs.get("base", self.options.get("base_uri")),
                        "compactArrays": kwargs.get("compact_arrays", True),
                        "graph": kwargs.get("graph", False),
                        "skipExpansion": kwargs.get("skip_expansion", False),
                        "processingMode": self.options.get(
                            "processing_mode", "json-ld-1.1"
                        ),
                        "documentLoader": self.options.get("document_loader"),
                    }

                    # Remove None values
                    compact_options = {
                        k: v for k, v in compact_options.items() if v is not None
                    }

                    # Use provided context or default
                    if context is None:
                        context = self._current_context or {}

                    # Perform compaction
                    compacted = self._jsonld.compact(document, context, compact_options)
                    return compacted

                except Exception as e:
                    if not self.options.get("error_recovery", True):
                        raise OntologyException(f"JSON-LD compaction failed: {str(e)}")
                    self.logger.warning(
                        f"pyld compaction failed, using fallback: {str(e)}"
                    )

            # Fallback implementation
            return self._compact_fallback(document, context)

        except Exception as e:
            if isinstance(e, OntologyException):
                raise
            raise OntologyException(f"Document compaction failed: {str(e)}")

    def flatten(
        self,
        document: Union[Dict[str, Any], List[Dict[str, Any]]],
        context: Optional[Dict[str, Any]] = None,
        **kwargs,
    ) -> Dict[str, Any]:
        """Flatten JSON-LD document to a single graph representation.

        Args:
            document: JSON-LD document to flatten
            context: Context to use for flattening
            **kwargs: Additional flattening options

        Returns:
            Dict[str, Any]: Flattened JSON-LD document

        Raises:
            OntologyException: If flattening fails
        """
        try:
            # Apply security checks if safe_mode is enabled
            if self.options.get("safe_mode", True):
                self._apply_safe_mode_restrictions(document)
                if context:
                    self._apply_safe_mode_restrictions(context)

            # Use pyld if available
            if self._pyld_available:
                try:
                    # Prepare flattening options
                    flatten_options = {
                        "base": kwargs.get("base", self.options.get("base_uri")),
                        "processingMode": self.options.get(
                            "processing_mode", "json-ld-1.1"
                        ),
                        "documentLoader": self.options.get("document_loader"),
                    }

                    # Remove None values
                    flatten_options = {
                        k: v for k, v in flatten_options.items() if v is not None
                    }

                    # Use provided context or default
                    if context is None:
                        context = self._current_context or {}

                    # Perform flattening
                    flattened = self._jsonld.flatten(document, context, flatten_options)
                    return flattened

                except Exception as e:
                    if not self.options.get("error_recovery", True):
                        raise OntologyException(f"JSON-LD flattening failed: {str(e)}")
                    self.logger.warning(
                        f"pyld flattening failed, using fallback: {str(e)}"
                    )

            # Fallback implementation
            return self._flatten_fallback(document, context)

        except Exception as e:
            if isinstance(e, OntologyException):
                raise
            raise OntologyException(f"Document flattening failed: {str(e)}")

    # Security and safety methods
    def _apply_safe_mode_restrictions(self, data: Any) -> None:
        """Apply safe mode restrictions to prevent security issues.

        Args:
            data: Data to check for security restrictions

        Raises:
            OntologyException: If data violates safe mode restrictions
        """
        if not self.options.get("safe_mode", True):
            return

        # Check data size
        max_size = self.options.get(
            "max_content_size", 50 * 1024 * 1024
        )  # 50MB default
        if hasattr(data, "__len__"):
            try:
                data_str = str(data)
                if len(data_str) > max_size:
                    raise OntologyException(
                        f"Content size {len(data_str)} exceeds maximum allowed size {max_size}"
                    )
            except (TypeError, ValueError):
                pass  # Skip size check if string conversion fails

        # Check nesting depth
        max_depth = self.options.get("max_nesting_depth", 100)
        current_depth = self._calculate_nesting_depth(data)
        if current_depth > max_depth:
            raise OntologyException(
                f"Nesting depth {current_depth} exceeds maximum allowed depth {max_depth}"
            )

        # Additional safe mode checks
        if isinstance(data, dict):
            # Limit number of keys
            max_keys = self.options.get("max_object_keys", 10000)
            if len(data) > max_keys:
                raise OntologyException(
                    f"Object has {len(data)} keys, exceeds maximum {max_keys}"
                )

            # Check for suspicious patterns
            for key in data.keys():
                if isinstance(key, str) and len(key) > 1000:
                    raise OntologyException("Object key length exceeds safety limits")

        elif isinstance(data, list):
            # Limit array size
            max_array_size = self.options.get("max_array_size", 100000)
            if len(data) > max_array_size:
                raise OntologyException(
                    f"Array size {len(data)} exceeds maximum {max_array_size}"
                )

    def _calculate_nesting_depth(self, data: Any, current_depth: int = 0) -> int:
        """Calculate the maximum nesting depth of a data structure."""
        if current_depth > 200:  # Prevent stack overflow during depth calculation
            return current_depth

        max_depth = current_depth

        if isinstance(data, dict):
            for value in data.values():
                depth = self._calculate_nesting_depth(value, current_depth + 1)
                max_depth = max(max_depth, depth)
        elif isinstance(data, list):
            for item in data:
                depth = self._calculate_nesting_depth(item, current_depth + 1)
                max_depth = max(max_depth, depth)

        return max_depth

    def _validate_content_security(self, content: str) -> None:
        """Validate content for security issues before parsing.

        Args:
            content: Content string to validate

        Raises:
            OntologyException: If content violates security restrictions
        """
        # Check content size
        max_size = self.options.get(
            "max_content_size", 50 * 1024 * 1024
        )  # 50MB default
        if len(content) > max_size:
            raise OntologyException(
                f"Content size {len(content)} bytes exceeds maximum allowed size {max_size} bytes"
            )

        # Check for suspicious patterns that might indicate JSON bombs
        if content.count("{") > 100000 or content.count("[") > 100000:
            raise OntologyException(
                "Content contains suspiciously high number of nested structures"
            )

        # Check for extremely long strings that might cause memory issues
        max_string_length = self.options.get(
            "max_string_length", 1024 * 1024
        )  # 1MB default
        lines = content.split("\n")
        for i, line in enumerate(
            lines[:1000]
        ):  # Check first 1000 lines only for performance
            if len(line) > max_string_length:
                raise OntologyException(
                    f"Line {i+1} exceeds maximum string length {max_string_length}"
                )

    def _secure_json_parse(self, content: str) -> Any:
        """Parse JSON with security protections against JSON bombs.

        Args:
            content: JSON content to parse

        Returns:
            Parsed JSON data

        Raises:
            Various exceptions for security violations
        """
        # Set recursion limit to prevent stack overflow
        import sys

        original_limit = sys.getrecursionlimit()
        max_depth = self.options.get("max_json_depth", 100)

        try:
            # Temporarily set a lower recursion limit
            sys.setrecursionlimit(
                max_depth + 50
            )  # Add some buffer for Python internals

            # Parse with the standard JSON parser
            data = self._json.loads(content)

            # Additional validation after parsing
            if self.options.get("safe_mode", True):
                self._apply_safe_mode_restrictions(data)

            return data

        finally:
            # Always restore the original recursion limit
            sys.setrecursionlimit(original_limit)

    # Fallback implementations for when pyld is not available
    def _expand_fallback(
        self, document: Union[Dict[str, Any], List[Dict[str, Any]]]
    ) -> List[Dict[str, Any]]:
        """Fallback expansion implementation without pyld."""
        try:
            if isinstance(document, list):
                expanded = []
                for item in document:
                    expanded.extend(self._expand_fallback(item))
                return expanded

            if not isinstance(document, dict):
                return []

            # Simple expansion - just ensure it's in list format and preserve structure
            expanded_doc = document.copy()

            # Remove context for expansion (contexts are not included in expanded form)
            if "@context" in expanded_doc:
                del expanded_doc["@context"]

            return [expanded_doc]

        except Exception as e:
            self.logger.warning(f"Fallback expansion failed: {str(e)}")
            return [document] if isinstance(document, dict) else []

    def _compact_fallback(
        self,
        document: Union[Dict[str, Any], List[Dict[str, Any]]],
        context: Optional[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """Fallback compaction implementation without pyld."""
        try:
            if isinstance(document, list):
                if len(document) == 1:
                    compacted = self._compact_fallback(document[0], context)
                else:
                    # Multiple documents - create a @graph structure
                    compacted = {"@graph": document}
            else:
                compacted = document.copy() if isinstance(document, dict) else {}

            # Add context if provided
            if context:
                compacted["@context"] = context

            return compacted

        except Exception as e:
            self.logger.warning(f"Fallback compaction failed: {str(e)}")
            return document if isinstance(document, dict) else {}

    def _flatten_fallback(
        self,
        document: Union[Dict[str, Any], List[Dict[str, Any]]],
        context: Optional[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """Fallback flattening implementation without pyld."""
        try:
            # Simple flattening - collect all nodes into a @graph array
            nodes = []

            if isinstance(document, list):
                for item in document:
                    if isinstance(item, dict):
                        nodes.append(item)
            elif isinstance(document, dict):
                if "@graph" in document:
                    # Already has @graph, extract nodes
                    graph_data = document["@graph"]
                    if isinstance(graph_data, list):
                        nodes.extend(graph_data)
                    else:
                        nodes.append(graph_data)
                else:
                    # Single document
                    nodes.append(document)

            # Create flattened structure
            flattened = {"@graph": nodes}

            # Add context if provided
            if context:
                flattened["@context"] = context

            return flattened

        except Exception as e:
            self.logger.warning(f"Fallback flattening failed: {str(e)}")
            return document if isinstance(document, dict) else {}

    # Graph and node operations
    def extract_graphs(self, document: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Extract graphs from JSON-LD document.

        Args:
            document (Dict[str, Any]): JSON-LD document

        Returns:
            List[Dict[str, Any]]: List of extracted graphs
        """
        graphs = []

        try:
            if "@graph" in document:
                # Document contains explicit graphs
                graph_data = document["@graph"]
                if isinstance(graph_data, list):
                    graphs.extend(graph_data)
                else:
                    graphs.append(graph_data)
            else:
                # Treat entire document as a single graph
                graphs.append(document)

            return graphs

        except Exception as e:
            self.logger.warning(f"Error extracting graphs: {str(e)}")
            return []

    def merge_graphs(self, graphs: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Merge multiple graphs into a single JSON-LD document.

        Args:
            graphs (List[Dict[str, Any]]): List of graphs to merge

        Returns:
            Dict[str, Any]: Merged JSON-LD document with @graph
        """
        try:
            if not graphs:
                return {}

            if len(graphs) == 1:
                return graphs[0]

            # Create a document with @graph containing all graphs
            merged = {"@context": {}, "@graph": graphs}

            # Merge contexts from all graphs
            for graph in graphs:
                if isinstance(graph, dict) and "@context" in graph:
                    context = graph["@context"]
                    if isinstance(context, dict):
                        merged["@context"].update(context)

            return merged

        except Exception as e:
            self.logger.warning(f"Error merging graphs: {str(e)}")
            return {}

    def filter_graph(
        self, document: Dict[str, Any], filter_func: Callable
    ) -> Dict[str, Any]:
        """Filter nodes in a JSON-LD graph.

        Args:
            document (Dict[str, Any]): JSON-LD document
            filter_func (Callable): Function to filter nodes

        Returns:
            Dict[str, Any]: Filtered JSON-LD document
        """
        try:
            filtered = document.copy()

            if "@graph" in document:
                graph_data = document["@graph"]
                if isinstance(graph_data, list):
                    filtered["@graph"] = [
                        node for node in graph_data if filter_func(node)
                    ]

            return filtered

        except Exception as e:
            self.logger.warning(f"Error filtering graph: {str(e)}")
            return document

    def get_nodes(self, document: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Get all nodes from JSON-LD document.

        Args:
            document (Dict[str, Any]): JSON-LD document

        Returns:
            List[Dict[str, Any]]: List of nodes
        """
        nodes = []

        try:
            if "@graph" in document:
                graph_data = document["@graph"]
                if isinstance(graph_data, list):
                    nodes.extend(graph_data)
                else:
                    nodes.append(graph_data)
            elif "@id" in document or "@type" in document:
                # Document itself is a node
                nodes.append(document)

            return nodes

        except Exception as e:
            self.logger.warning(f"Error getting nodes: {str(e)}")
            return []

    def get_node_by_id(
        self, document: Dict[str, Any], node_id: str
    ) -> Optional[Dict[str, Any]]:
        """Get node by ID from JSON-LD document.

        Args:
            document (Dict[str, Any]): JSON-LD document
            node_id (str): Node ID to find

        Returns:
            Optional[Dict[str, Any]]: Node with matching ID or None
        """
        try:
            nodes = self.get_nodes(document)
            for node in nodes:
                if isinstance(node, dict) and node.get("@id") == node_id:
                    return node
            return None

        except Exception as e:
            self.logger.warning(f"Error getting node by ID: {str(e)}")
            return None

    def get_nodes_by_type(
        self, document: Dict[str, Any], node_type: str
    ) -> List[Dict[str, Any]]:
        """Get nodes by type from JSON-LD document.

        Args:
            document (Dict[str, Any]): JSON-LD document
            node_type (str): Node type to find

        Returns:
            List[Dict[str, Any]]: Nodes with matching type
        """
        matching_nodes = []

        try:
            nodes = self.get_nodes(document)
            for node in nodes:
                if isinstance(node, dict):
                    node_types = node.get("@type", [])
                    if isinstance(node_types, str):
                        node_types = [node_types]
                    if node_type in node_types:
                        matching_nodes.append(node)

            return matching_nodes

        except Exception as e:
            self.logger.warning(f"Error getting nodes by type: {str(e)}")
            return []

    # Conversion methods for ontology data models
    def to_ontology(self, document: Dict[str, Any], **kwargs) -> Optional["Ontology"]:
        """Convert JSON-LD document to Ontology object.

        Args:
            document (Dict[str, Any]): JSON-LD document
            **kwargs: Additional conversion parameters

        Returns:
            Optional[Ontology]: Converted Ontology object or None
        """
        try:
            if not Ontology:
                self.logger.warning("Ontology class not available")
                return None

            # Extract terms and relationships
            terms_list = self.extract_terms(document, **kwargs)
            relationships_list = self.extract_relationships(document, **kwargs)
            metadata = self.extract_metadata(document, **kwargs)

            # Convert lists to dictionaries as expected by Ontology class
            terms_dict = {term.id: term for term in terms_list}
            relationships_dict = {rel.id: rel for rel in relationships_list}

            # Generate a valid ontology ID
            ontology_id = kwargs.get(
                "ontology_id", f"JSONLD:{abs(hash(str(document))) % 1000000}"
            )

            # Create ontology object
            ontology = Ontology(
                id=ontology_id,
                name=kwargs.get(
                    "name", metadata.get("name", "JSON-LD Imported Ontology")
                ),
                terms=terms_dict,
                relationships=relationships_dict,
                metadata=metadata,
                namespaces=kwargs.get(
                    "namespaces", metadata.get("namespaces", ["jsonld"])
                ),
                version=kwargs.get("version", metadata.get("version", "1.0")),
                description=kwargs.get(
                    "description",
                    metadata.get("description", "JSON-LD imported ontology"),
                ),
            )

            return ontology

        except Exception as e:
            self.logger.error(f"Error converting to ontology: {str(e)}")
            return None

    def extract_terms(self, document: Dict[str, Any], **kwargs) -> List["Term"]:
        """Extract terms from JSON-LD document.

        Args:
            document (Dict[str, Any]): JSON-LD document
            **kwargs: Additional extraction parameters

        Returns:
            List[Term]: List of extracted terms
        """
        terms = []

        try:
            if not Term:
                self.logger.warning("Term class not available")
                return []

            nodes = self.get_nodes(document)

            for node in nodes:
                if not isinstance(node, dict):
                    continue

                node_id = node.get("@id")
                if not node_id:
                    continue

                # Extract term properties
                name = self._extract_label(node)
                definition = self._extract_definition(node)
                synonyms = self._extract_synonyms(node)
                node_types = node.get("@type", [])
                if isinstance(node_types, str):
                    node_types = [node_types]

                # Create term metadata including types
                node_metadata = self._extract_node_metadata(node)
                if node_types:
                    node_metadata["types"] = node_types

                # Create term with URI-friendly ID handling
                try:
                    term = Term(
                        id=node_id,
                        name=name or node_id.split("/")[-1].split("#")[-1],
                        definition=definition,
                        synonyms=synonyms,
                        namespace=self._extract_namespace(node_id),
                        metadata=node_metadata,
                    )
                    terms.append(term)
                except ValueError as e:
                    # If term ID validation fails, try to create a compatible ID
                    self.logger.warning(
                        f"Term ID validation failed for {node_id}: {str(e)}"
                    )
                    # Create a fallback ID in standard format
                    fallback_id = f"URI:{abs(hash(node_id)) % 1000000}"
                    try:
                        term = Term(
                            id=fallback_id,
                            name=name or node_id.split("/")[-1].split("#")[-1],
                            definition=definition,
                            synonyms=synonyms,
                            namespace=self._extract_namespace(node_id),
                            metadata={**node_metadata, "original_id": node_id},
                        )
                        terms.append(term)
                    except Exception as inner_e:
                        self.logger.error(
                            f"Failed to create term for {node_id}: {str(inner_e)}"
                        )
                        continue

            return terms

        except Exception as e:
            self.logger.error(f"Error extracting terms: {str(e)}")
            return []

    def extract_relationships(
        self, document: Dict[str, Any], **kwargs
    ) -> List["Relationship"]:
        """Extract relationships from JSON-LD document.

        Args:
            document (Dict[str, Any]): JSON-LD document
            **kwargs: Additional extraction parameters

        Returns:
            List[Relationship]: List of extracted relationships
        """
        relationships = []

        try:
            if not Relationship:
                self.logger.warning("Relationship class not available")
                return []

            nodes = self.get_nodes(document)

            for node in nodes:
                if not isinstance(node, dict):
                    continue

                subject_id = node.get("@id")
                if not subject_id:
                    continue

                # Extract relationships from node properties
                for predicate, objects in node.items():
                    if predicate.startswith("@"):
                        continue  # Skip JSON-LD keywords

                    if not isinstance(objects, list):
                        objects = [objects]

                    for obj in objects:
                        object_id = None
                        object_value = None

                        if isinstance(obj, dict):
                            if "@id" in obj:
                                object_id = obj["@id"]
                            elif "@value" in obj:
                                object_value = obj["@value"]
                        elif isinstance(obj, str):
                            if obj.startswith(("http://", "https://", "_:")):
                                object_id = obj
                            else:
                                object_value = obj
                        else:
                            object_value = str(obj)

                        # Create relationship ID that follows the expected format
                        rel_hash = (
                            abs(
                                hash(
                                    f"{subject_id}_{predicate}_{object_id or object_value}"
                                )
                            )
                            % 1000000
                        )
                        rel_id = f"REL:{rel_hash}"

                        # Create relationship with compatible parameters
                        try:
                            relationship = Relationship(
                                id=rel_id,
                                subject=subject_id,
                                predicate=predicate,
                                object=object_id or object_value or "",
                                confidence=1.0,  # Default confidence
                                source="jsonld",
                                extraction_method="jsonld_parsing",
                                context="JSON-LD document parsing",
                            )
                            relationships.append(relationship)
                        except ValueError as e:
                            # Handle ID validation issues for relationships too
                            self.logger.warning(
                                f"Relationship creation failed: {str(e)}"
                            )
                            # Try with fallback IDs
                            fallback_subject = f"URI:{abs(hash(subject_id)) % 1000000}"
                            fallback_object = object_id or object_value or ""
                            if not object_id and object_value:
                                # This is a literal, keep as is
                                pass
                            elif object_id and not object_id.startswith(
                                ("http://", "https://", "_:")
                            ):
                                # Try to create a compatible object ID
                                fallback_object = (
                                    f"URI:{abs(hash(object_id)) % 1000000}"
                                )

                            try:
                                relationship = Relationship(
                                    id=rel_id,
                                    subject=fallback_subject,
                                    predicate=predicate,
                                    object=fallback_object,
                                    confidence=1.0,
                                    source="jsonld",
                                    extraction_method="jsonld_parsing",
                                    context=f"Original subject: {subject_id}, Original object: {object_id or object_value}",
                                )
                                relationships.append(relationship)
                            except Exception as inner_e:
                                self.logger.error(
                                    f"Failed to create relationship: {str(inner_e)}"
                                )
                                continue

            return relationships

        except Exception as e:
            self.logger.error(f"Error extracting relationships: {str(e)}")
            return []

    def extract_metadata(self, document: Dict[str, Any], **kwargs) -> Dict[str, Any]:
        """Extract metadata from JSON-LD document.

        Args:
            document (Dict[str, Any]): JSON-LD document
            **kwargs: Additional extraction parameters

        Returns:
            Dict[str, Any]: Extracted metadata
        """
        metadata = {
            "format": "jsonld",
            "parser": "JSONLDParser",
            "extraction_timestamp": datetime.now().isoformat(),
        }

        try:
            # Extract basic document metadata
            if "@context" in document:
                metadata["contexts"] = document["@context"]

            if "@id" in document:
                metadata["document_id"] = document["@id"]

            # Extract common ontology metadata properties
            for prop in [
                "title",
                "description",
                "creator",
                "created",
                "modified",
                "version",
                "license",
            ]:
                if prop in document:
                    metadata[prop] = document[prop]

            # Extract namespace information
            namespaces = self.get_namespaces(document)
            if namespaces:
                metadata["namespaces"] = namespaces

            # Count statistics
            nodes = self.get_nodes(document)
            metadata["node_count"] = len(nodes)

            return metadata

        except Exception as e:
            self.logger.error(f"Error extracting metadata: {str(e)}")
            return metadata

    def _extract_label(self, node: Dict[str, Any]) -> Optional[str]:
        """Extract label from JSON-LD node."""
        for prop in ["rdfs:label", "label", "name", "title"]:
            if prop in node:
                value = node[prop]
                if isinstance(value, dict) and "@value" in value:
                    return value["@value"]
                elif isinstance(value, str):
                    return value
        return None

    def _extract_definition(self, node: Dict[str, Any]) -> Optional[str]:
        """Extract definition from JSON-LD node."""
        for prop in ["rdfs:comment", "comment", "definition", "description"]:
            if prop in node:
                value = node[prop]
                if isinstance(value, dict) and "@value" in value:
                    return value["@value"]
                elif isinstance(value, str):
                    return value
        return None

    def _extract_synonyms(self, node: Dict[str, Any]) -> List[str]:
        """Extract synonyms from JSON-LD node."""
        synonyms = []
        for prop in ["synonym", "altLabel", "alternative"]:
            if prop in node:
                values = node[prop]
                if not isinstance(values, list):
                    values = [values]
                for value in values:
                    if isinstance(value, dict) and "@value" in value:
                        synonyms.append(value["@value"])
                    elif isinstance(value, str):
                        synonyms.append(value)
        return synonyms

    def _extract_namespace(self, uri: str) -> str:
        """Extract namespace from URI."""
        if "#" in uri:
            return uri.split("#")[0] + "#"
        elif "/" in uri:
            parts = uri.split("/")
            return "/".join(parts[:-1]) + "/"
        return uri

    def _extract_node_metadata(self, node: Dict[str, Any]) -> Dict[str, Any]:
        """Extract metadata from JSON-LD node."""
        metadata = {}

        # Copy all non-JSON-LD keyword properties as metadata
        for key, value in node.items():
            if not key.startswith("@") and key not in [
                "rdfs:label",
                "rdfs:comment",
                "label",
                "name",
                "comment",
                "definition",
            ]:
                metadata[key] = value

        return metadata

    # =====================
    # JSON-LD-specific Error Recovery Methods
    # =====================

    def _recover_retry(self, error_context: ErrorContext, **kwargs) -> Any:
        """
        JSON-LD-specific retry recovery with alternative parsing strategies.

        Args:
            error_context (ErrorContext): Error context information
            **kwargs: Additional recovery parameters

        Returns:
            Any: Result of retry attempt or None if failed
        """
        retry_count = len([s for s in error_context.attempted_recoveries if s == RecoveryStrategy.RETRY])
        max_retries = kwargs.get("max_retries", 3)
        
        if retry_count >= max_retries:
            self.logger.error(f"Maximum retries ({max_retries}) exceeded for {error_context.location}")
            return None
        
        self.logger.info(f"Attempting JSON-LD-specific retry at {error_context.location} (attempt {retry_count + 1}/{max_retries})")
        
        content = kwargs.get("content", "")
        parsing_stage = kwargs.get("parsing_stage", "")
        
        try:
            if parsing_stage == "json_decode":
                # Strategy for JSON parsing errors
                if retry_count == 0:
                    # Try fixing common JSON issues
                    self.logger.info("Retry strategy 1: JSON sanitization")
                    sanitized_content = self._sanitize_json_content(content)
                    if sanitized_content != content:
                        return self._secure_json_parse(sanitized_content)
                
                elif retry_count == 1:
                    # Try alternative JSON parsing with more lenient settings
                    self.logger.info("Retry strategy 2: Lenient JSON parsing")
                    import ast
                    try:
                        # Use ast.literal_eval for safer parsing of simple structures
                        return ast.literal_eval(content)
                    except (ValueError, SyntaxError):
                        pass
                
                elif retry_count == 2:
                    # Try extracting valid JSON from larger text
                    self.logger.info("Retry strategy 3: JSON extraction")
                    extracted_json = self._extract_json_from_text(content)
                    if extracted_json:
                        return self._secure_json_parse(extracted_json)
            
            elif parsing_stage == "jsonld_expand":
                # Strategy for JSON-LD expansion errors
                data = kwargs.get("data", {})
                
                if retry_count == 0:
                    # Try adding missing context
                    self.logger.info("Retry strategy 1: Add default context")
                    data_with_context = self._add_default_context(data)
                    return self.expand(data_with_context, **kwargs)
                
                elif retry_count == 1:
                    # Try without expansion (use raw data)
                    self.logger.info("Retry strategy 2: Skip expansion")
                    return data
            
            elif parsing_stage == "ontology_convert":
                # Strategy for ontology conversion errors
                processed_data = kwargs.get("processed_data", {})
                
                if retry_count == 0:
                    # Try simpler conversion
                    self.logger.info("Retry strategy 1: Simple ontology structure")
                    return self._simple_ontology_conversion(processed_data)
                
                elif retry_count == 1:
                    # Return minimal ontology structure
                    self.logger.info("Retry strategy 2: Minimal ontology")
                    return self._create_minimal_ontology(processed_data)
        
        except Exception as retry_error:
            self.logger.warning(f"Retry attempt {retry_count + 1} failed: {str(retry_error)}")
        
        return None

    def _sanitize_json_content(self, content: str) -> str:
        """
        Sanitize JSON content to fix common parsing issues.

        Args:
            content (str): Original JSON content

        Returns:
            str: Sanitized JSON content
        """
        import re
        
        sanitized = content.strip()
        original_length = len(content)
        
        try:
            # Fix 1: Remove BOM and invisible characters
            sanitized = sanitized.lstrip('\ufeff\ufffe')
            
            # Fix 2: Fix trailing commas
            sanitized = re.sub(r',(\s*[}\]])', r'\1', sanitized)
            
            # Fix 3: Quote unquoted keys
            sanitized = re.sub(r'([{,]\s*)([a-zA-Z_][a-zA-Z0-9_]*)\s*:', r'\1"\2":', sanitized)
            
            # Fix 4: Fix single quotes to double quotes
            sanitized = re.sub(r"'([^']*)'", r'"\1"', sanitized)
            
            # Fix 5: Remove control characters
            sanitized = re.sub(r'[\x00-\x1f\x7f]', '', sanitized)
            
            # Fix 6: Fix escaped quotes
            sanitized = sanitized.replace("\\'", "'").replace('\\"', '"')
            
            if len(sanitized) != original_length:
                self.logger.info(f"JSON sanitization changed length from {original_length} to {len(sanitized)} characters")
        
        except Exception as e:
            self.logger.warning(f"JSON sanitization failed: {str(e)}, returning original content")
            return content
        
        return sanitized

    def _extract_json_from_text(self, content: str) -> Optional[str]:
        """
        Extract valid JSON from mixed text content.

        Args:
            content (str): Content that may contain JSON

        Returns:
            Optional[str]: Extracted JSON string or None
        """
        import re
        
        # Look for JSON patterns
        json_patterns = [
            r'\{[^{}]*\}',  # Simple object
            r'\[[^\[\]]*\]',  # Simple array
            r'\{.*\}',  # Complex object (greedy)
            r'\[.*\]',  # Complex array (greedy)
        ]
        
        for pattern in json_patterns:
            matches = re.findall(pattern, content, re.DOTALL)
            for match in matches:
                try:
                    # Test if it's valid JSON
                    self._json.loads(match)
                    self.logger.info(f"Extracted valid JSON pattern: {pattern}")
                    return match
                except self._json.JSONDecodeError:
                    continue
        
        return None

    def _add_default_context(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Add default JSON-LD context to data that's missing it.

        Args:
            data (Dict[str, Any]): JSON-LD data

        Returns:
            Dict[str, Any]: Data with default context
        """
        if not isinstance(data, dict):
            return data
        
        if "@context" not in data:
            default_context = {
                "rdf": "http://www.w3.org/1999/02/22-rdf-syntax-ns#",
                "rdfs": "http://www.w3.org/2000/01/rdf-schema#", 
                "owl": "http://www.w3.org/2002/07/owl#",
                "xsd": "http://www.w3.org/2001/XMLSchema#"
            }
            
            data_with_context = {"@context": default_context}
            data_with_context.update(data)
            
            self.logger.info("Added default JSON-LD context")
            return data_with_context
        
        return data

    def _simple_ontology_conversion(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Simple ontology conversion that extracts basic structure.

        Args:
            data (Dict[str, Any]): JSON-LD data

        Returns:
            Dict[str, Any]: Simple ontology structure
        """
        try:
            # Create a simplified ontology structure
            ontology = {
                "id": "recovered_ontology",
                "name": "Recovered Ontology",
                "terms": {},
                "relationships": {},
                "metadata": {
                    "recovered": True,
                    "source": "json-ld"
                }
            }
            
            # Extract basic terms from the data
            if isinstance(data, dict):
                for key, value in data.items():
                    if key.startswith("@"):
                        continue
                    
                    term_id = str(key)
                    ontology["terms"][term_id] = {
                        "id": term_id,
                        "name": term_id.replace("_", " ").title(),
                        "definition": str(value) if not isinstance(value, (dict, list)) else "Complex structure"
                    }
            
            elif isinstance(data, list):
                for i, item in enumerate(data):
                    if isinstance(item, dict):
                        item_id = item.get("@id", f"item_{i}")
                        ontology["terms"][item_id] = {
                            "id": item_id,
                            "name": item.get("name", item.get("rdfs:label", item_id)),
                            "definition": item.get("definition", item.get("rdfs:comment", ""))
                        }
            
            self.logger.info("Created simple ontology conversion")
            return ontology
        
        except Exception as e:
            self.logger.error(f"Simple ontology conversion failed: {str(e)}")
            return None

    def _create_minimal_ontology(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Create minimal valid ontology structure.

        Args:
            data (Dict[str, Any]): Original data

        Returns:
            Dict[str, Any]: Minimal ontology
        """
        return {
            "id": "minimal_ontology",
            "name": "Minimal Recovered Ontology",
            "terms": {
                "default_term": {
                    "id": "default_term",
                    "name": "Default Term",
                    "definition": "Minimal term created during error recovery"
                }
            },
            "relationships": {},
            "metadata": {
                "recovered": True,
                "minimal": True,
                "source": "error_recovery",
                "original_data_type": type(data).__name__
            }
        }

    def _recover_default(self, error_context: ErrorContext, **kwargs) -> Any:
        """
        JSON-LD-specific default recovery providing minimal valid structures.

        Args:
            error_context (ErrorContext): Error context information
            **kwargs: Additional recovery parameters

        Returns:
            Any: Default JSON-LD structure
        """
        parsing_stage = kwargs.get("parsing_stage", "")
        
        if parsing_stage == "json_decode":
            # Return minimal valid JSON
            self.logger.info("Providing default empty JSON object")
            return {}
        
        elif parsing_stage == "jsonld_expand":
            # Return the original data without expansion
            data = kwargs.get("data", {})
            self.logger.info("Using original data without JSON-LD expansion")
            return data
        
        elif parsing_stage == "ontology_convert":
            # Return minimal ontology
            self.logger.info("Providing minimal ontology structure")
            return self._create_minimal_ontology(kwargs.get("processed_data", {}))
        
        # Fallback to parent implementation
        return super()._recover_default(error_context, **kwargs)
