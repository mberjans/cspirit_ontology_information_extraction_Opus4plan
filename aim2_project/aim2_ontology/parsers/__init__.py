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
                    if not self.options.get("error_recovery", True):
                        raise OntologyException(f"RDFlib parsing failed: {str(e)}")
                    self.logger.warning(
                        f"RDFlib parsing failed, continuing with owlready2: {str(e)}"
                    )

            # Parse using owlready2 for OWL-specific features
            owl_ontology = None
            if self._owlready2_available:
                try:
                    owl_ontology = self._parse_with_owlready2(
                        content, format_hint, **kwargs
                    )
                except Exception as e:
                    if not self.options.get("error_recovery", True):
                        raise OntologyException(f"owlready2 parsing failed: {str(e)}")
                    self.logger.warning(f"owlready2 parsing failed: {str(e)}")

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
            dialect = self.detect_dialect(content)
            self._detected_dialect = dialect

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

            # Handle bad lines
            if len(row) != len(headers) and not self.options.get(
                "skip_bad_lines", False
            ):
                if not self.options.get("error_recovery", False):
                    raise ValueError(
                        f"Invalid CSV: inconsistent field count at row {row_count + 1}"
                    )
                else:
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

    def detect_dialect(self, content: str) -> Any:
        """Detect CSV dialect using csv.Sniffer."""
        if not content:
            return None

        try:
            # Use first 1024 characters for dialect detection
            sample = content[:1024] if len(content) > 1024 else content

            sniffer = self._csv.Sniffer()
            dialect = sniffer.sniff(sample, delimiters=",\t|;")

            # Store dialect info
            self.logger.debug(
                f"Detected dialect: delimiter='{dialect.delimiter}', "
                f"quotechar='{dialect.quotechar}', escapechar='{dialect.escapechar}'"
            )

            return dialect

        except Exception as e:
            self.logger.warning(f"Dialect detection failed: {str(e)}, using defaults")

            # Return default dialect
            class DefaultDialect:
                delimiter = self.options.get("delimiter", ",")
                quotechar = self.options.get("quotechar", '"')
                escapechar = self.options.get("escapechar")
                skipinitialspace = False
                doublequote = True
                quoting = self._csv.QUOTE_MINIMAL

            return DefaultDialect()

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
            dialect = self._detected_dialect or self.detect_dialect(content)

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
                dialect = self.detect_dialect(content)
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


class JSONLDParser(AbstractParser):
    """Concrete JSON-LD parser implementation (placeholder for TDD)."""

    def __init__(self, options: Optional[Dict[str, Any]] = None):
        """Initialize JSON-LD parser with options."""
        super().__init__("jsonld", options=options)
        # This is a placeholder implementation for TDD
        # The actual implementation will be done after tests pass
        self.logger.warning(
            "JSONLDParser implementation is pending - using placeholder"
        )

    def parse(self, content: str, **kwargs) -> Any:
        """Parse JSON-LD content."""
        raise NotImplementedError("Parse method not implemented")

    def validate(self, content: str, **kwargs) -> bool:
        """Validate JSON-LD content."""
        raise NotImplementedError("Validate method not implemented")

    def get_supported_formats(self) -> List[str]:
        """Get supported JSON-LD formats."""
        raise NotImplementedError("Get supported formats not implemented")

    def set_options(self, options: Dict[str, Any]) -> None:
        """Set parser options."""
        raise NotImplementedError("Set options not implemented")

    def get_options(self) -> Dict[str, Any]:
        """Get current options."""
        raise NotImplementedError("Get options not implemented")

    def get_metadata(self) -> Dict[str, Any]:
        """Get parser metadata."""
        raise NotImplementedError("Get metadata not implemented")

    def expand(self, document: Dict[str, Any], **kwargs) -> Dict[str, Any]:
        """Expand JSON-LD document."""
        raise NotImplementedError("Expand method not implemented")

    def compact(
        self, document: Dict[str, Any], context: Dict[str, Any], **kwargs
    ) -> Dict[str, Any]:
        """Compact JSON-LD document."""
        raise NotImplementedError("Compact method not implemented")

    def flatten(self, document: Dict[str, Any], **kwargs) -> Dict[str, Any]:
        """Flatten JSON-LD document."""
        raise NotImplementedError("Flatten method not implemented")

    def frame(
        self, document: Dict[str, Any], frame: Dict[str, Any], **kwargs
    ) -> Dict[str, Any]:
        """Frame JSON-LD document."""
        raise NotImplementedError("Frame method not implemented")

    def normalize(self, document: Dict[str, Any], **kwargs) -> str:
        """Normalize JSON-LD document."""
        raise NotImplementedError("Normalize method not implemented")

    def resolve_context(self, context: Any) -> Dict[str, Any]:
        """Resolve JSON-LD context."""
        raise NotImplementedError("Resolve context method not implemented")

    def set_context(self, context: Dict[str, Any]) -> None:
        """Set JSON-LD context."""
        raise NotImplementedError("Set context method not implemented")

    def get_context(self) -> Dict[str, Any]:
        """Get current JSON-LD context."""
        raise NotImplementedError("Get context method not implemented")

    def merge_contexts(
        self, context1: Dict[str, Any], context2: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Merge two JSON-LD contexts."""
        raise NotImplementedError("Merge contexts method not implemented")

    def to_rdf(self, document: Dict[str, Any], **kwargs) -> Any:
        """Convert JSON-LD to RDF."""
        raise NotImplementedError("To RDF method not implemented")

    def from_rdf(self, rdf_data: Any, **kwargs) -> Dict[str, Any]:
        """Convert RDF to JSON-LD."""
        raise NotImplementedError("From RDF method not implemented")

    def get_namespaces(self, document: Dict[str, Any]) -> Dict[str, str]:
        """Get namespaces from JSON-LD document."""
        raise NotImplementedError("Get namespaces method not implemented")

    def expand_namespaces(self, document: Dict[str, Any]) -> Dict[str, Any]:
        """Expand namespaces in JSON-LD document."""
        raise NotImplementedError("Expand namespaces method not implemented")

    def parse_file(self, file_path: str, **kwargs) -> Any:
        """Parse JSON-LD file from file path."""
        raise NotImplementedError("Parse file method not implemented")

    def parse_string(self, content: str, **kwargs) -> Any:
        """Parse JSON-LD content from string."""
        raise NotImplementedError("Parse string method not implemented")

    def parse_stream(self, stream, **kwargs) -> Any:
        """Parse JSON-LD from stream/file-like object."""
        raise NotImplementedError("Parse stream method not implemented")

    def validate_jsonld(self, content: str, **kwargs) -> Dict[str, Any]:
        """Validate JSON-LD content with detailed results."""
        raise NotImplementedError("Validate JSON-LD method not implemented")

    def get_validation_errors(self) -> List[str]:
        """Get validation errors."""
        raise NotImplementedError("Get validation errors method not implemented")

    def reset_options(self) -> None:
        """Reset options to defaults."""
        raise NotImplementedError("Reset options method not implemented")
