#!/usr/bin/env python3
"""
Ontology Manager module for AIM2 project.

This module provides comprehensive ontology management capabilities including
loading, caching, validation, and multi-source integration. The OntologyManager
class serves as the primary interface for ontology operations within the AIM2 project.

Key Features:
- Intelligent format auto-detection for various ontology formats
- Caching system for performance optimization
- Multi-source ontology loading and integration
- Comprehensive error handling and validation
- Statistics generation and reporting
- Integration with existing parser framework
"""

import argparse
import csv
import json
import logging
import time
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

# Import the models and parsers
try:
    from ..aim2_utils.config_manager import ConfigManager
    from .models import Ontology
    from .parsers import auto_detect_parser
    from .validators import ValidationPipeline
except ImportError:
    # For development/testing scenarios
    from models import Ontology
    from parsers import auto_detect_parser
    from validators import ValidationPipeline

    try:
        from aim2_project.aim2_utils.config_manager import ConfigManager
    except ImportError:
        ConfigManager = None


@dataclass
class LoadResult:
    """Result container for ontology load operations."""

    success: bool
    ontology: Optional[Ontology] = None
    source_path: Optional[str] = None
    load_time: float = 0.0
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class CacheEntry:
    """Cache entry for loaded ontologies."""

    ontology: Ontology
    load_time: float
    source_path: str
    file_mtime: Optional[float] = None
    access_count: int = 0
    last_accessed: float = field(default_factory=time.time)


class OntologyManagerError(Exception):
    """Base exception for OntologyManager errors."""


class OntologyLoadError(OntologyManagerError):
    """Exception raised when ontology loading fails."""


class OntologyManager:
    """Comprehensive ontology manager with loading, caching, and integration capabilities."""

    def __init__(self, enable_caching: bool = True, cache_size_limit: int = 100):
        """Initialize ontology manager.

        Args:
            enable_caching: Whether to enable ontology caching
            cache_size_limit: Maximum number of ontologies to cache
        """
        self.logger = logging.getLogger(__name__)

        # Core storage
        self.ontologies: Dict[str, Ontology] = {}

        # Caching system
        self.enable_caching = enable_caching
        self.cache_size_limit = cache_size_limit
        self._cache: Dict[str, CacheEntry] = {}

        # Statistics tracking
        self.load_stats = {
            "total_loads": 0,
            "successful_loads": 0,
            "failed_loads": 0,
            "cache_hits": 0,
            "cache_misses": 0,
            "formats_loaded": defaultdict(int),
        }

        # Multi-source statistics tracking
        self.source_stats = {}  # source_path -> statistics
        self.performance_stats = {
            "average_load_time": 0.0,
            "total_load_time": 0.0,
            "fastest_load_time": None,
            "slowest_load_time": None,
        }

        # Validation pipeline
        self.validation_pipeline = ValidationPipeline()

    def load_ontology(self, source: Union[str, Path]) -> LoadResult:
        """Load a single ontology with format auto-detection.

        Args:
            source: Path to the ontology file or URL

        Returns:
            LoadResult: Result object containing loaded ontology and metadata

        Raises:
            OntologyLoadError: If loading fails critically
        """
        start_time = time.time()
        source_str = str(source)
        self.load_stats["total_loads"] += 1

        try:
            # Check cache first (only if cache size limit allows caching)
            if (
                self.enable_caching
                and self.cache_size_limit > 0
                and self._is_cached_and_valid(source_str)
            ):
                self.load_stats["cache_hits"] += 1
                cache_entry = self._cache[source_str]
                cache_entry.access_count += 1
                cache_entry.last_accessed = time.time()

                load_time = time.time() - start_time
                return LoadResult(
                    success=True,
                    ontology=cache_entry.ontology,
                    source_path=source_str,
                    load_time=load_time,
                    metadata={
                        "cache_hit": True,
                        "access_count": cache_entry.access_count,
                    },
                )

            # Count as cache miss based on caching configuration:
            # - When caching is disabled entirely (enable_caching=False), always count as cache miss
            # - When caching is enabled with size limit > 0, count as cache miss
            # - When caching is enabled but size limit is 0, don't count (no cache system active)
            if not self.enable_caching:
                # Caching disabled entirely - always count as miss for statistics purposes
                self.load_stats["cache_misses"] += 1
            elif self.enable_caching and self.cache_size_limit > 0:
                # Caching enabled with valid limit - count as miss since we didn't hit cache
                self.load_stats["cache_misses"] += 1
            # If enable_caching=True but cache_size_limit=0, don't count misses

            # Auto-detect and create parser
            parser = auto_detect_parser(file_path=source)
            if not parser:
                error_msg = f"No suitable parser found for source: {source_str}"
                self.logger.error(error_msg)
                self.load_stats["failed_loads"] += 1
                failed_load_time = time.time() - start_time

                # Update source-specific statistics for failure
                self._update_source_statistics(
                    source_str, None, failed_load_time, None, False
                )

                return LoadResult(
                    success=False,
                    source_path=source_str,
                    load_time=failed_load_time,
                    errors=[error_msg],
                )

            # Parse ontology using parse_safe for consistent ParseResult interface
            if hasattr(parser, "parse_safe"):
                # Read file content and use parse_safe for consistent ParseResult interface
                try:
                    if str(source).startswith(("http://", "https://")):
                        # For URLs, try to use parser's URL handling if available
                        if hasattr(parser, "parse_url"):
                            parse_result = parser.parse_url(str(source))
                        else:
                            # Fallback: fetch content and parse
                            import requests

                            response = requests.get(str(source), timeout=30)
                            response.raise_for_status()
                            parse_result = parser.parse_safe(response.text)
                    else:
                        # For local files, read content and parse
                        source_path = Path(source)
                        if not source_path.exists():
                            raise FileNotFoundError(f"Source file not found: {source}")

                        with open(source_path, "r", encoding="utf-8") as f:
                            content = f.read()

                        # Clear parser cache to avoid recursion issues
                        if hasattr(parser, "clear_cache"):
                            parser.clear_cache()

                        parse_result = parser.parse_safe(content)
                except Exception as e:
                    # If parse_safe fails, create a ParseResult with the error
                    from .parsers import ParseResult

                    parse_result = ParseResult(
                        success=False,
                        errors=[f"Failed to read or parse source: {str(e)}"],
                    )
            else:
                # Fallback to parse method and wrap result in ParseResult
                try:
                    raw_result = parser.parse(source)
                    # Check if it's already a ParseResult
                    if hasattr(raw_result, "success"):
                        parse_result = raw_result
                    else:
                        # Wrap raw result in ParseResult
                        from .parsers import ParseResult

                        if raw_result is not None:
                            parse_result = ParseResult(success=True, data=raw_result)
                        else:
                            parse_result = ParseResult(
                                success=False, errors=["Parser returned None"]
                            )
                except Exception as e:
                    from .parsers import ParseResult

                    parse_result = ParseResult(
                        success=False, errors=[f"Parser error: {str(e)}"]
                    )

            if not parse_result.success or not parse_result.data:
                error_msg = f"Failed to parse ontology from {source_str}"
                self.logger.error(error_msg)
                if parse_result.errors:
                    self.logger.error(f"Parser errors: {parse_result.errors}")

                self.load_stats["failed_loads"] += 1
                failed_load_time = time.time() - start_time

                # Update source-specific statistics for parse failure
                # Get parser format name
            parser_format = getattr(parser, "parser_name", None)
            if not parser_format:
                # Fallback to get_supported_formats if available
                if hasattr(parser, "get_supported_formats"):
                    formats = parser.get_supported_formats()
                    parser_format = formats[0] if formats else "unknown"
                else:
                    parser_format = "unknown"
                self._update_source_statistics(
                    source_str, None, failed_load_time, parser_format, False
                )

                return LoadResult(
                    success=False,
                    source_path=source_str,
                    load_time=failed_load_time,
                    errors=[error_msg] + parse_result.errors,
                    warnings=parse_result.warnings,
                )

            # Extract or convert to Ontology object
            ontology = parse_result.data

            # If data is not an Ontology object, try to convert it
            if not isinstance(ontology, Ontology):
                if hasattr(parser, "to_ontology") and ontology is not None:
                    try:
                        # Handle nested ParseResult structure
                        conversion_input = parse_result
                        if hasattr(parse_result.data, "success") and hasattr(
                            parse_result.data, "data"
                        ):
                            # Data is itself a ParseResult, use the nested data
                            conversion_input = parse_result.data

                        ontology = parser.to_ontology(conversion_input)
                    except Exception as e:
                        error_msg = (
                            f"Failed to convert parsed data to Ontology: {str(e)}"
                        )
                        self.logger.error(error_msg)
                        self.load_stats["failed_loads"] += 1
                        failed_load_time = time.time() - start_time

                        # Update source-specific statistics for conversion failure
                        # Get parser format name
                        parser_format = getattr(parser, "parser_name", None)
                        if not parser_format:
                            # Fallback to get_supported_formats if available
                            if hasattr(parser, "get_supported_formats"):
                                formats = parser.get_supported_formats()
                                parser_format = formats[0] if formats else "unknown"
                            else:
                                parser_format = "unknown"
                        self._update_source_statistics(
                            source_str, None, failed_load_time, parser_format, False
                        )

                        return LoadResult(
                            success=False,
                            source_path=source_str,
                            load_time=failed_load_time,
                            errors=[error_msg],
                        )
                else:
                    error_msg = f"Parser returned invalid ontology type: {type(ontology)}, and no conversion method available"
                    self.logger.error(error_msg)
                    self.load_stats["failed_loads"] += 1
                    failed_load_time = time.time() - start_time

                    # Update source-specific statistics for validation failure
                    # Get parser format name
            parser_format = getattr(parser, "parser_name", None)
            if not parser_format:
                # Fallback to get_supported_formats if available
                if hasattr(parser, "get_supported_formats"):
                    formats = parser.get_supported_formats()
                    parser_format = formats[0] if formats else "unknown"
                else:
                    parser_format = "unknown"
                    self._update_source_statistics(
                        source_str, None, failed_load_time, parser_format, False
                    )

                    return LoadResult(
                        success=False,
                        source_path=source_str,
                        load_time=failed_load_time,
                        errors=[error_msg],
                    )

            # Final validation that we have an Ontology object
            if not isinstance(ontology, Ontology):
                error_msg = (
                    f"Final validation failed: Expected Ontology, got {type(ontology)}"
                )
                self.logger.error(error_msg)
                self.load_stats["failed_loads"] += 1
                failed_load_time = time.time() - start_time

                # Update source-specific statistics for final validation failure
                # Get parser format name
            parser_format = getattr(parser, "parser_name", None)
            if not parser_format:
                # Fallback to get_supported_formats if available
                if hasattr(parser, "get_supported_formats"):
                    formats = parser.get_supported_formats()
                    parser_format = formats[0] if formats else "unknown"
                else:
                    parser_format = "unknown"
                self._update_source_statistics(
                    source_str, None, failed_load_time, parser_format, False
                )

                return LoadResult(
                    success=False,
                    source_path=source_str,
                    load_time=failed_load_time,
                    errors=[error_msg],
                )

            # Store in manager
            self.ontologies[ontology.id] = ontology

            # Update cache if enabled and cache size limit allows it
            if self.enable_caching and self.cache_size_limit > 0:
                self._update_cache(source_str, ontology, start_time)

            # Calculate load time first
            load_time = time.time() - start_time

            # Update statistics
            self.load_stats["successful_loads"] += 1
            # Get parser format name
            parser_format = getattr(parser, "parser_name", None)
            if not parser_format:
                # Fallback to get_supported_formats if available
                if hasattr(parser, "get_supported_formats"):
                    formats = parser.get_supported_formats()
                    parser_format = formats[0] if formats else "unknown"
                else:
                    parser_format = "unknown"

            # Ensure formats_loaded is a defaultdict
            if not isinstance(self.load_stats["formats_loaded"], defaultdict):
                self.load_stats["formats_loaded"] = defaultdict(
                    int, self.load_stats["formats_loaded"]
                )
            self.load_stats["formats_loaded"][parser_format] += 1

            # Update source-specific statistics
            self._update_source_statistics(
                source_str, ontology, load_time, parser_format, True
            )

            # Update performance statistics
            self._update_performance_statistics(load_time)
            self.logger.info(
                f"Successfully loaded ontology {ontology.id} from {source_str} in {load_time:.3f}s"
            )

            return LoadResult(
                success=True,
                ontology=ontology,
                source_path=source_str,
                load_time=load_time,
                warnings=parse_result.warnings,
                metadata={
                    "format": parser_format,
                    "terms_count": len(ontology.terms),
                    "relationships_count": len(ontology.relationships),
                    "parse_time": parse_result.parse_time,
                },
            )

        except Exception as e:
            error_msg = f"Unexpected error loading ontology from {source_str}: {str(e)}"
            self.logger.exception(error_msg)
            self.load_stats["failed_loads"] += 1
            failed_load_time = time.time() - start_time

            # Update source-specific statistics for unexpected error
            self._update_source_statistics(
                source_str, None, failed_load_time, None, False
            )

            return LoadResult(
                success=False,
                source_path=source_str,
                load_time=failed_load_time,
                errors=[error_msg],
            )

    def load_ontologies(self, sources: List[Union[str, Path]]) -> List[LoadResult]:
        """Load multiple ontology sources.

        Args:
            sources: List of paths to ontology files or URLs

        Returns:
            List[LoadResult]: List of results for each source
        """
        results = []

        for source in sources:
            try:
                result = self.load_ontology(source)
                results.append(result)

                if result.success:
                    self.logger.info(f"Loaded ontology from {source}")
                else:
                    self.logger.warning(
                        f"Failed to load ontology from {source}: {result.errors}"
                    )

            except Exception as e:
                error_msg = f"Critical error loading {source}: {str(e)}"
                self.logger.exception(error_msg)
                results.append(
                    LoadResult(
                        success=False, source_path=str(source), errors=[error_msg]
                    )
                )

        return results

    def load_from_config(
        self,
        config_manager: Optional[ConfigManager] = None,
        config_path: Optional[str] = None,
        source_filter: Optional[List[str]] = None,
        enabled_only: bool = True,
    ) -> List[LoadResult]:
        """Load ontologies from configuration file.

        This method reads ontology sources from the configuration and loads all
        enabled sources automatically. It provides configuration-based automation
        for ontology loading without requiring manual specification of file paths.

        Args:
            config_manager: Optional ConfigManager instance to use. If None, creates a new one.
            config_path: Optional path to configuration file. If None, uses default config.
            source_filter: Optional list of source names to load. If None, loads all enabled sources.
            enabled_only: Whether to load only enabled sources (default: True).

        Returns:
            List[LoadResult]: List of results for each configured source

        Raises:
            OntologyManagerError: If configuration loading fails or no valid sources found
        """
        start_time = time.time()
        results = []

        try:
            # Get or create config manager
            if config_manager is None:
                if ConfigManager is None:
                    raise OntologyManagerError(
                        "ConfigManager not available. Please provide a config_manager instance."
                    )
                config_manager = ConfigManager()

                # Load configuration
                if config_path:
                    config_manager.load_config(config_path)
                else:
                    config_manager.load_default_config()

            # Get ontology sources from configuration
            try:
                ontology_sources = config_manager.get("ontology.sources", {})
            except KeyError:
                raise OntologyManagerError(
                    "No ontology sources configuration found. "
                    "Expected 'ontology.sources' section in configuration."
                )

            if not ontology_sources:
                self.logger.warning("No ontology sources configured")
                return results

            # Filter sources based on criteria
            sources_to_load = self._filter_configured_sources(
                ontology_sources, source_filter, enabled_only
            )

            if not sources_to_load:
                msg = "No valid ontology sources found in configuration"
                if enabled_only:
                    msg += " (only enabled sources were considered)"
                if source_filter:
                    msg += f" (filtered by: {source_filter})"

                self.logger.warning(msg)
                return results

            self.logger.info(
                f"Loading {len(sources_to_load)} ontology sources from configuration"
            )

            # Load each configured source
            for source_name, source_config in sources_to_load.items():
                result = self._load_configured_source(source_name, source_config)
                results.append(result)

            # Log summary
            successful = sum(1 for r in results if r.success)
            failed = len(results) - successful
            total_time = time.time() - start_time

            self.logger.info(
                f"Configuration-based loading completed in {total_time:.3f}s: "
                f"{successful} successful, {failed} failed"
            )

            return results

        except Exception as e:
            error_msg = f"Failed to load ontologies from configuration: {str(e)}"
            self.logger.exception(error_msg)
            raise OntologyManagerError(error_msg)

    def _filter_configured_sources(
        self,
        ontology_sources: Dict[str, Any],
        source_filter: Optional[List[str]],
        enabled_only: bool,
    ) -> Dict[str, Dict[str, Any]]:
        """Filter configured ontology sources based on criteria.

        Args:
            ontology_sources: Dictionary of ontology source configurations
            source_filter: Optional list of source names to include
            enabled_only: Whether to include only enabled sources

        Returns:
            Dict[str, Dict[str, Any]]: Filtered source configurations
        """
        filtered_sources = {}

        for source_name, source_config in ontology_sources.items():
            # Check if source should be filtered by name
            if source_filter and source_name not in source_filter:
                continue

            # Check if source should be filtered by enabled status
            if enabled_only and not source_config.get("enabled", False):
                self.logger.debug(f"Skipping disabled source: {source_name}")
                continue

            # Validate source configuration
            if not isinstance(source_config, dict):
                self.logger.warning(
                    f"Invalid configuration for source {source_name}: not a dictionary"
                )
                continue

            if not source_config.get("local_path") and not source_config.get("url"):
                self.logger.warning(
                    f"Source {source_name} has no local_path or url configured"
                )
                continue

            filtered_sources[source_name] = source_config

        return filtered_sources

    def _load_configured_source(
        self, source_name: str, source_config: Dict[str, Any]
    ) -> LoadResult:
        """Load a single configured ontology source.

        Args:
            source_name: Name of the source (e.g., 'chebi', 'gene_ontology')
            source_config: Configuration dictionary for the source

        Returns:
            LoadResult: Result of the load operation
        """
        start_time = time.time()

        try:
            # Determine the source path to use
            local_path = source_config.get("local_path")
            url = source_config.get("url")

            # Prefer local path if it exists and is accessible
            source_path = None
            if local_path:
                local_path_obj = Path(local_path)
                if local_path_obj.exists():
                    source_path = str(local_path_obj.resolve())
                    self.logger.debug(
                        f"Using local path for {source_name}: {source_path}"
                    )
                else:
                    self.logger.debug(
                        f"Local path not found for {source_name}: {local_path}"
                    )

            # Fall back to URL if local path is not available
            if not source_path and url:
                source_path = url
                self.logger.debug(f"Using URL for {source_name}: {source_path}")

            if not source_path:
                error_msg = f"No valid source path found for {source_name}"
                return LoadResult(
                    success=False,
                    source_path=source_name,
                    load_time=time.time() - start_time,
                    errors=[error_msg],
                    metadata={"source_name": source_name, "config": source_config},
                )

            # Load the ontology
            self.logger.info(
                f"Loading ontology source: {source_name} from {source_path}"
            )
            result = self.load_ontology(source_path)

            # Add source metadata to result
            if result.metadata is None:
                result.metadata = {}

            result.metadata.update(
                {
                    "source_name": source_name,
                    "config": source_config,
                    "configuration_based": True,
                    "url": url,
                    "local_path": local_path,
                }
            )

            if result.success:
                self.logger.info(
                    f"Successfully loaded {source_name} from configuration"
                )
            else:
                self.logger.warning(
                    f"Failed to load {source_name} from configuration: {result.errors}"
                )

            return result

        except Exception as e:
            error_msg = f"Error loading configured source {source_name}: {str(e)}"
            self.logger.exception(error_msg)
            return LoadResult(
                success=False,
                source_path=source_name,
                load_time=time.time() - start_time,
                errors=[error_msg],
                metadata={"source_name": source_name, "config": source_config},
            )

    def validate_ontology_sources_config(
        self,
        config_manager: Optional[ConfigManager] = None,
        config_path: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Validate ontology sources configuration.

        Validates the ontology sources configuration for completeness and correctness.
        Checks for required fields, accessible paths, and valid URLs.

        Args:
            config_manager: Optional ConfigManager instance to use. If None, creates a new one.
            config_path: Optional path to configuration file. If None, uses default config.

        Returns:
            Dict[str, Any]: Validation report with status, errors, and warnings

        Raises:
            OntologyManagerError: If configuration cannot be loaded
        """
        validation_report = {
            "valid": True,
            "errors": [],
            "warnings": [],
            "source_status": {},
            "summary": {
                "total_sources": 0,
                "valid_sources": 0,
                "enabled_sources": 0,
                "accessible_local_paths": 0,
                "valid_urls": 0,
            },
        }

        try:
            # Get or create config manager
            if config_manager is None:
                if ConfigManager is None:
                    raise OntologyManagerError(
                        "ConfigManager not available. Please provide a config_manager instance."
                    )
                config_manager = ConfigManager()

                # Load configuration
                if config_path:
                    config_manager.load_config(config_path)
                else:
                    config_manager.load_default_config()

            # Get ontology sources from configuration
            try:
                ontology_sources = config_manager.get("ontology.sources", {})
            except KeyError:
                validation_report["valid"] = False
                validation_report["errors"].append(
                    "No ontology sources configuration found. Expected 'ontology.sources' section."
                )
                return validation_report

            if not ontology_sources:
                validation_report["warnings"].append("No ontology sources configured")
                return validation_report

            validation_report["summary"]["total_sources"] = len(ontology_sources)

            # Validate each source
            for source_name, source_config in ontology_sources.items():
                source_status = self._validate_single_source(source_name, source_config)
                validation_report["source_status"][source_name] = source_status

                # Update summary counts
                if source_status["valid"]:
                    validation_report["summary"]["valid_sources"] += 1

                if source_config.get("enabled", False):
                    validation_report["summary"]["enabled_sources"] += 1

                if source_status["local_path_accessible"]:
                    validation_report["summary"]["accessible_local_paths"] += 1

                if source_status["url_valid"]:
                    validation_report["summary"]["valid_urls"] += 1

                # Collect errors and warnings
                validation_report["errors"].extend(source_status["errors"])
                validation_report["warnings"].extend(source_status["warnings"])

            # Overall validation status
            if validation_report["errors"]:
                validation_report["valid"] = False

            return validation_report

        except Exception as e:
            error_msg = f"Failed to validate ontology sources configuration: {str(e)}"
            self.logger.exception(error_msg)
            raise OntologyManagerError(error_msg)

    def _validate_single_source(
        self, source_name: str, source_config: Any
    ) -> Dict[str, Any]:
        """Validate a single ontology source configuration.

        Args:
            source_name: Name of the source
            source_config: Configuration for the source

        Returns:
            Dict[str, Any]: Validation status for the source
        """
        status = {
            "valid": True,
            "errors": [],
            "warnings": [],
            "local_path_accessible": False,
            "url_valid": False,
            "has_required_fields": False,
        }

        # Check if config is a dictionary
        if not isinstance(source_config, dict):
            status["valid"] = False
            status["errors"].append(
                f"Source '{source_name}' configuration is not a dictionary"
            )
            return status

        # Check required fields
        required_fields = ["enabled"]
        optional_fields = [
            "local_path",
            "url",
            "update_frequency",
            "include_deprecated",
        ]

        for field in required_fields:
            if field not in source_config:
                status["errors"].append(
                    f"Source '{source_name}' missing required field: {field}"
                )

        # Check that at least one of local_path or url is provided
        has_local_path = bool(source_config.get("local_path"))
        has_url = bool(source_config.get("url"))

        if not has_local_path and not has_url:
            status["valid"] = False
            status["errors"].append(
                f"Source '{source_name}' must have either 'local_path' or 'url'"
            )
        else:
            status["has_required_fields"] = True

        # Validate local path if provided
        if has_local_path:
            local_path = source_config["local_path"]
            try:
                local_path_obj = Path(local_path)
                if local_path_obj.exists():
                    status["local_path_accessible"] = True
                else:
                    status["warnings"].append(
                        f"Source '{source_name}' local path does not exist: {local_path}"
                    )
            except Exception as e:
                status["warnings"].append(
                    f"Source '{source_name}' local path validation error: {str(e)}"
                )

        # Validate URL if provided
        if has_url:
            url = source_config["url"]
            if self._is_valid_url(url):
                status["url_valid"] = True
            else:
                status["warnings"].append(
                    f"Source '{source_name}' has invalid URL format: {url}"
                )

        # Check enabled field type
        enabled = source_config.get("enabled")
        if enabled is not None and not isinstance(enabled, bool):
            status["warnings"].append(
                f"Source '{source_name}' 'enabled' field should be boolean, got {type(enabled)}"
            )

        # Set overall validity
        if status["errors"]:
            status["valid"] = False

        return status

    def _is_valid_url(self, url: str) -> bool:
        """Basic URL validation.

        Args:
            url: URL to validate

        Returns:
            bool: True if URL appears valid
        """
        try:
            import re

            # Basic URL pattern check
            url_pattern = re.compile(
                r"^https?://"  # http:// or https://
                r"(?:(?:[A-Z0-9](?:[A-Z0-9-]{0,61}[A-Z0-9])?\.)+[A-Z]{2,6}\.?|"  # domain...
                r"localhost|"  # localhost...
                r"\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3})"  # ...or ip
                r"(?::\d+)?"  # optional port
                r"(?:/?|[/?]\S+)$",
                re.IGNORECASE,
            )
            return url_pattern.match(url) is not None
        except Exception:
            return False

    def get_ontology(self, ontology_id: str) -> Optional[Ontology]:
        """Get an ontology by ID.

        Args:
            ontology_id: Unique identifier for the ontology

        Returns:
            Optional[Ontology]: The ontology if found, None otherwise
        """
        return self.ontologies.get(ontology_id)

    def list_ontologies(self) -> List[str]:
        """List all loaded ontology IDs.

        Returns:
            List[str]: List of ontology IDs
        """
        return list(self.ontologies.keys())

    def remove_ontology(self, ontology_id: str) -> bool:
        """Remove an ontology from the manager.

        Args:
            ontology_id: Unique identifier for the ontology

        Returns:
            bool: True if removed, False if not found
        """
        removed = ontology_id in self.ontologies
        if removed:
            del self.ontologies[ontology_id]

            # Clean up cache entries for this ontology
            cache_keys_to_remove = []
            for cache_key, cache_entry in self._cache.items():
                if cache_entry.ontology.id == ontology_id:
                    cache_keys_to_remove.append(cache_key)

            for cache_key in cache_keys_to_remove:
                del self._cache[cache_key]

            # Clean up source statistics for removed ontology
            sources_to_remove = []
            for source_path, stats in self.source_stats.items():
                if stats.get("ontology_id") == ontology_id:
                    sources_to_remove.append(source_path)

            for source_path in sources_to_remove:
                del self.source_stats[source_path]

            self.logger.info(f"Removed ontology {ontology_id}")

        return removed

    def add_ontology(self, ontology: Ontology) -> bool:
        """Add an ontology to the manager.

        Args:
            ontology: The ontology to add

        Returns:
            bool: True if added successfully
        """
        if not isinstance(ontology, Ontology):
            self.logger.error(f"Cannot add non-Ontology object: {type(ontology)}")
            return False

        self.ontologies[ontology.id] = ontology
        self.logger.info(f"Added ontology {ontology.id}")
        return True

    def clear_cache(self) -> None:
        """Clear the ontology cache."""
        cache_size = len(self._cache)
        self._cache.clear()
        self.logger.info(f"Cleared cache ({cache_size} entries)")

    def get_statistics(self) -> Dict[str, Any]:
        """Get comprehensive loading, cache, and multi-source statistics.

        Returns:
            Dict[str, Any]: Dictionary containing various statistics including
                          multi-source aggregations and performance metrics
        """
        # Basic cache statistics
        cache_stats = {
            "cache_size": len(self._cache),
            "cache_limit": self.cache_size_limit,
            "cache_enabled": self.enable_caching,
        }

        # Basic ontology statistics (aggregated across all sources)
        ontology_stats = {
            "loaded_ontologies": len(self.ontologies),
            "total_terms": sum(len(ont.terms) for ont in self.ontologies.values()),
            "total_relationships": sum(
                len(ont.relationships) for ont in self.ontologies.values()
            ),
        }

        # Multi-source specific statistics
        multisource_stats = self._get_multisource_statistics()

        # Performance statistics
        performance_stats = self._get_performance_statistics()

        # Format statistics (ensure it's a regular dict, not defaultdict)
        format_stats = dict(self.load_stats["formats_loaded"])

        return {
            **self.load_stats,
            **cache_stats,
            **ontology_stats,
            **multisource_stats,
            **performance_stats,
            "formats_loaded": format_stats,
        }

    def export_ontology(
        self, ontology_id: str, format: str = "json", output_path: Optional[str] = None
    ) -> Union[str, bool]:
        """Export a single ontology in the specified format.

        Args:
            ontology_id: Unique identifier for the ontology to export
            format: Export format ('json', 'csv', 'owl')
            output_path: Optional path to save the exported data

        Returns:
            Union[str, bool]: Serialized data if no output_path, success status if output_path provided

        Raises:
            OntologyManagerError: If ontology not found or export fails
        """
        # Validate ontology exists
        ontology = self.get_ontology(ontology_id)
        if not ontology:
            error_msg = f"Ontology '{ontology_id}' not found"
            self.logger.error(error_msg)
            raise OntologyManagerError(error_msg)

        try:
            # Export based on format
            if format.lower() == "json":
                exported_data = self._export_ontology_json(ontology)
            elif format.lower() == "csv":
                exported_data = self._export_ontology_csv(ontology)
            elif format.lower() == "owl":
                exported_data = self._export_ontology_owl(ontology)
            else:
                error_msg = f"Unsupported export format: {format}"
                self.logger.error(error_msg)
                raise OntologyManagerError(error_msg)

            # Save to file if output_path provided
            if output_path:
                return self._save_exported_data(exported_data, output_path, format)
            else:
                return exported_data

        except Exception as e:
            error_msg = f"Failed to export ontology '{ontology_id}': {str(e)}"
            self.logger.exception(error_msg)
            raise OntologyManagerError(error_msg)

    def export_combined_ontology(
        self, format: str = "json", output_path: Optional[str] = None
    ) -> Union[str, bool]:
        """Export all loaded ontologies as a combined structure.

        Args:
            format: Export format ('json', 'csv', 'owl')
            output_path: Optional path to save the exported data

        Returns:
            Union[str, bool]: Serialized data if no output_path, success status if output_path provided

        Raises:
            OntologyManagerError: If no ontologies loaded or export fails
        """
        if not self.ontologies:
            error_msg = "No ontologies loaded to export"
            self.logger.error(error_msg)
            raise OntologyManagerError(error_msg)

        try:
            # Export based on format
            if format.lower() == "json":
                exported_data = self._export_combined_json()
            elif format.lower() == "csv":
                exported_data = self._export_combined_csv()
            elif format.lower() == "owl":
                exported_data = self._export_combined_owl()
            else:
                error_msg = f"Unsupported export format: {format}"
                self.logger.error(error_msg)
                raise OntologyManagerError(error_msg)

            # Save to file if output_path provided
            if output_path:
                return self._save_exported_data(exported_data, output_path, format)
            else:
                return exported_data

        except Exception as e:
            error_msg = f"Failed to export combined ontologies: {str(e)}"
            self.logger.exception(error_msg)
            raise OntologyManagerError(error_msg)

    def export_statistics(self, output_path: Optional[str] = None) -> Union[str, bool]:
        """Export statistics as JSON.

        Args:
            output_path: Optional path to save the statistics JSON

        Returns:
            Union[str, bool]: JSON string if no output_path, success status if output_path provided

        Raises:
            OntologyManagerError: If export fails
        """
        try:
            stats = self.get_statistics()

            # Add detailed ontology information
            detailed_stats = {**stats, "ontologies": {}}

            for ont_id, ontology in self.ontologies.items():
                detailed_stats["ontologies"][ont_id] = {
                    "id": ontology.id,
                    "name": ontology.name,
                    "version": ontology.version,
                    "description": ontology.description,
                    "terms_count": len(ontology.terms),
                    "relationships_count": len(ontology.relationships),
                    "namespaces": ontology.namespaces,
                    "is_consistent": ontology.is_consistent,
                    "validation_errors": ontology.validation_errors,
                }

            exported_data = json.dumps(detailed_stats, indent=2, default=str)

            # Save to file if output_path provided
            if output_path:
                return self._save_exported_data(exported_data, output_path, "json")
            else:
                return exported_data

        except Exception as e:
            error_msg = f"Failed to export statistics: {str(e)}"
            self.logger.exception(error_msg)
            raise OntologyManagerError(error_msg)

    def _export_ontology_json(self, ontology: Ontology) -> str:
        """Export a single ontology to JSON format.

        Args:
            ontology: The ontology to export

        Returns:
            str: JSON representation of the ontology
        """
        try:
            return ontology.to_json()
        except Exception as e:
            self.logger.error(
                f"Failed to serialize ontology '{ontology.id}' to JSON: {str(e)}"
            )
            raise

    def _export_ontology_csv(self, ontology: Ontology) -> str:
        """Export a single ontology to CSV format.

        Args:
            ontology: The ontology to export

        Returns:
            str: CSV representation of the ontology (terms and relationships)
        """
        import io

        output = io.StringIO()

        # Export terms
        output.write("# TERMS\n")
        if ontology.terms:
            writer = csv.writer(output)
            writer.writerow(
                ["id", "name", "definition", "synonyms", "namespace", "is_obsolete"]
            )

            for term in ontology.terms.values():
                synonyms_str = ";".join(term.synonyms) if term.synonyms else ""
                writer.writerow(
                    [
                        term.id,
                        term.name,
                        term.definition or "",
                        synonyms_str,
                        term.namespace or "",
                        term.is_obsolete,
                    ]
                )

        # Export relationships
        output.write("\n# RELATIONSHIPS\n")
        if ontology.relationships:
            writer = csv.writer(output)
            writer.writerow(
                ["id", "subject", "predicate", "object", "confidence", "evidence"]
            )

            for rel in ontology.relationships.values():
                evidence_str = rel.evidence if rel.evidence else ""
                writer.writerow(
                    [
                        rel.id,
                        rel.subject,
                        rel.predicate,
                        rel.object,
                        rel.confidence,
                        evidence_str,
                    ]
                )

        return output.getvalue()

    def _export_ontology_owl(self, ontology: Ontology) -> str:
        """Export a single ontology to basic OWL format.

        Args:
            ontology: The ontology to export

        Returns:
            str: Basic OWL representation of the ontology
        """
        # Basic OWL template
        owl_lines = [
            '<?xml version="1.0"?>',
            '<rdf:RDF xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#"',
            '         xmlns:rdfs="http://www.w3.org/2000/01/rdf-schema#"',
            '         xmlns:owl="http://www.w3.org/2002/07/owl#">',
            "",
            f'  <owl:Ontology rdf:about="#{ontology.id}">',
            f"    <rdfs:label>{ontology.name}</rdfs:label>",
        ]

        if ontology.description:
            owl_lines.append(f"    <rdfs:comment>{ontology.description}</rdfs:comment>")

        if ontology.version:
            owl_lines.append(
                f"    <owl:versionInfo>{ontology.version}</owl:versionInfo>"
            )

        owl_lines.append("  </owl:Ontology>")
        owl_lines.append("")

        # Export terms as OWL classes
        for term in ontology.terms.values():
            owl_lines.extend(
                [
                    f'  <owl:Class rdf:about="#{term.id}">',
                    f"    <rdfs:label>{term.name}</rdfs:label>",
                ]
            )

            if term.definition:
                owl_lines.append(f"    <rdfs:comment>{term.definition}</rdfs:comment>")

            if term.namespace:
                owl_lines.append(
                    f"    <rdfs:isDefinedBy>{term.namespace}</rdfs:isDefinedBy>"
                )

            owl_lines.append("  </owl:Class>")
            owl_lines.append("")

        # Export relationships as object properties
        for rel in ontology.relationships.values():
            owl_lines.extend(
                [
                    f'  <owl:ObjectProperty rdf:about="#{rel.predicate}">',
                    f'    <rdfs:domain rdf:resource="#{rel.subject}"/>',
                    f'    <rdfs:range rdf:resource="#{rel.object}"/>',
                    "  </owl:ObjectProperty>",
                    "",
                ]
            )

        owl_lines.append("</rdf:RDF>")

        return "\n".join(owl_lines)

    def _export_combined_json(self) -> str:
        """Export all loaded ontologies as a combined JSON structure.

        Returns:
            str: JSON representation of all loaded ontologies
        """
        combined_data = {
            "export_metadata": {
                "export_timestamp": time.time(),
                "ontology_count": len(self.ontologies),
                "export_format": "json",
            },
            "ontologies": {},
        }

        for ont_id, ontology in self.ontologies.items():
            combined_data["ontologies"][ont_id] = ontology.to_dict()

        return json.dumps(combined_data, indent=2, default=str)

    def _export_combined_csv(self) -> str:
        """Export all loaded ontologies as a combined CSV structure.

        Returns:
            str: CSV representation of all loaded ontologies
        """
        import io

        output = io.StringIO()

        # Combined terms from all ontologies
        output.write("# COMBINED TERMS\n")
        writer = csv.writer(output)
        writer.writerow(
            [
                "ontology_id",
                "term_id",
                "name",
                "definition",
                "synonyms",
                "namespace",
                "is_obsolete",
            ]
        )

        for ont_id, ontology in self.ontologies.items():
            for term in ontology.terms.values():
                synonyms_str = ";".join(term.synonyms) if term.synonyms else ""
                writer.writerow(
                    [
                        ont_id,
                        term.id,
                        term.name,
                        term.definition or "",
                        synonyms_str,
                        term.namespace or "",
                        term.is_obsolete,
                    ]
                )

        # Combined relationships from all ontologies
        output.write("\n# COMBINED RELATIONSHIPS\n")
        writer = csv.writer(output)
        writer.writerow(
            [
                "ontology_id",
                "relationship_id",
                "subject",
                "predicate",
                "object",
                "confidence",
                "evidence",
            ]
        )

        for ont_id, ontology in self.ontologies.items():
            for rel in ontology.relationships.values():
                evidence_str = rel.evidence if rel.evidence else ""
                writer.writerow(
                    [
                        ont_id,
                        rel.id,
                        rel.subject,
                        rel.predicate,
                        rel.object,
                        rel.confidence,
                        evidence_str,
                    ]
                )

        return output.getvalue()

    def _export_combined_owl(self) -> str:
        """Export all loaded ontologies as a combined OWL structure.

        Returns:
            str: Basic OWL representation of all loaded ontologies
        """
        # Basic combined OWL template
        owl_lines = [
            '<?xml version="1.0"?>',
            '<rdf:RDF xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#"',
            '         xmlns:rdfs="http://www.w3.org/2000/01/rdf-schema#"',
            '         xmlns:owl="http://www.w3.org/2002/07/owl#">',
            "",
            '  <owl:Ontology rdf:about="#combined_ontologies">',
            "    <rdfs:label>Combined Ontologies Export</rdfs:label>",
            f"    <rdfs:comment>Combined export of {len(self.ontologies)} ontologies</rdfs:comment>",
            "  </owl:Ontology>",
            "",
        ]

        # Export all terms from all ontologies
        for ont_id, ontology in self.ontologies.items():
            owl_lines.append(f"  <!-- Terms from ontology: {ont_id} -->")

            for term in ontology.terms.values():
                owl_lines.extend(
                    [
                        f'  <owl:Class rdf:about="#{ont_id}_{term.id}">',
                        f"    <rdfs:label>{term.name}</rdfs:label>",
                    ]
                )

                if term.definition:
                    owl_lines.append(
                        f"    <rdfs:comment>{term.definition}</rdfs:comment>"
                    )

                if term.namespace:
                    owl_lines.append(
                        f"    <rdfs:isDefinedBy>{term.namespace}</rdfs:isDefinedBy>"
                    )

                owl_lines.append(f'    <rdfs:member rdf:resource="#{ont_id}"/>')
                owl_lines.append("  </owl:Class>")

            owl_lines.append("")

        # Export all relationships from all ontologies
        for ont_id, ontology in self.ontologies.items():
            owl_lines.append(f"  <!-- Relationships from ontology: {ont_id} -->")

            for rel in ontology.relationships.values():
                owl_lines.extend(
                    [
                        f'  <owl:ObjectProperty rdf:about="#{ont_id}_{rel.predicate}">',
                        f'    <rdfs:domain rdf:resource="#{ont_id}_{rel.subject}"/>',
                        f'    <rdfs:range rdf:resource="#{ont_id}_{rel.object}"/>',
                        "  </owl:ObjectProperty>",
                    ]
                )

            owl_lines.append("")

        owl_lines.append("</rdf:RDF>")

        return "\n".join(owl_lines)

    def _save_exported_data(self, data: str, output_path: str, format: str) -> bool:
        """Save exported data to a file.

        Args:
            data: The exported data string
            output_path: Path where to save the file
            format: Export format for logging

        Returns:
            bool: True if saved successfully

        Raises:
            OntologyManagerError: If saving fails
        """
        try:
            # Security: Validate and resolve path to prevent traversal attacks
            output_file = Path(output_path).resolve()

            # Basic path traversal protection - check for '..' components
            if ".." in output_path:
                raise OntologyManagerError(
                    f"Invalid or potentially unsafe path: {output_path}"
                )

            # Create parent directories if they don't exist
            output_file.parent.mkdir(parents=True, exist_ok=True)

            # Write data to file
            with open(output_file, "w", encoding="utf-8") as f:
                f.write(data)

            self.logger.info(f"Successfully exported {format} data to {output_path}")
            return True

        except Exception as e:
            error_msg = f"Failed to save exported data to {output_path}: {str(e)}"
            self.logger.error(error_msg)
            raise OntologyManagerError(error_msg)

    def _is_cached_and_valid(self, source_path: str) -> bool:
        """Check if source is cached and cache entry is valid.

        Args:
            source_path: Path to the source file

        Returns:
            bool: True if cached and valid
        """
        if source_path not in self._cache:
            return False

        cache_entry = self._cache[source_path]

        # Check if source file still exists and hasn't been modified
        try:
            if Path(source_path).exists():
                current_mtime = Path(source_path).stat().st_mtime
                if cache_entry.file_mtime and current_mtime > cache_entry.file_mtime:
                    # File has been modified, invalidate cache
                    del self._cache[source_path]
                    return False
        except (OSError, AttributeError):
            # If we can't check file stats, assume cache is valid
            pass

        return True

    def _update_cache(
        self, source_path: str, ontology: Ontology, load_start_time: float
    ) -> None:
        """Update cache with new ontology.

        Args:
            source_path: Path to the source file
            ontology: The loaded ontology
            load_start_time: When loading started
        """
        # Don't cache if limit is 0 or negative
        if self.cache_size_limit <= 0:
            return

        try:
            file_mtime = None
            if Path(source_path).exists():
                file_mtime = Path(source_path).stat().st_mtime
        except OSError:
            file_mtime = None

        cache_entry = CacheEntry(
            ontology=ontology,
            load_time=load_start_time,
            source_path=source_path,
            file_mtime=file_mtime,
            access_count=1,
        )

        # Implement LRU eviction if cache is full
        if len(self._cache) >= self.cache_size_limit:
            # Remove least recently accessed entry
            lru_key = min(
                self._cache.keys(), key=lambda k: self._cache[k].last_accessed
            )
            del self._cache[lru_key]
            self.logger.debug(f"Evicted cache entry for {lru_key}")

        self._cache[source_path] = cache_entry
        self.logger.debug(f"Cached ontology from {source_path}")

    def _update_source_statistics(
        self,
        source_path: str,
        ontology: Optional[Ontology],
        load_time: float,
        format_name: Optional[str],
        success: bool,
    ) -> None:
        """Update source-specific statistics.

        Args:
            source_path: Path to the source file
            ontology: Loaded ontology (None if failed)
            load_time: Time taken to load
            format_name: Format of the source
            success: Whether loading was successful
        """
        if source_path not in self.source_stats:
            self.source_stats[source_path] = {
                "load_attempts": 0,
                "successful_loads": 0,
                "failed_loads": 0,
                "total_load_time": 0.0,
                "average_load_time": 0.0,
                "format": format_name,
                "terms_count": 0,
                "relationships_count": 0,
                "last_load_time": load_time,
                "ontology_id": None,
            }

        stats = self.source_stats[source_path]
        stats["load_attempts"] += 1
        stats["total_load_time"] += load_time
        stats["average_load_time"] = stats["total_load_time"] / stats["load_attempts"]
        stats["last_load_time"] = load_time

        if success and ontology:
            stats["successful_loads"] += 1
            stats["terms_count"] = len(ontology.terms)
            stats["relationships_count"] = len(ontology.relationships)
            stats["format"] = format_name
            stats["ontology_id"] = ontology.id
        else:
            stats["failed_loads"] += 1

    def _update_performance_statistics(self, load_time: float) -> None:
        """Update overall performance statistics.

        Args:
            load_time: Time taken for this load operation
        """
        self.performance_stats["total_load_time"] += load_time

        if self.load_stats["successful_loads"] > 0:
            self.performance_stats["average_load_time"] = (
                self.performance_stats["total_load_time"]
                / self.load_stats["successful_loads"]
            )

        if (
            self.performance_stats["fastest_load_time"] is None
            or load_time < self.performance_stats["fastest_load_time"]
        ):
            self.performance_stats["fastest_load_time"] = load_time

        if (
            self.performance_stats["slowest_load_time"] is None
            or load_time > self.performance_stats["slowest_load_time"]
        ):
            self.performance_stats["slowest_load_time"] = load_time

    def _get_multisource_statistics(self) -> Dict[str, Any]:
        """Get multi-source specific statistics.

        Returns:
            Dict[str, Any]: Multi-source statistics including per-source breakdowns,
                          source coverage, and overlap analysis
        """
        if not self.source_stats:
            return {
                "sources_loaded": 0,
                "sources_attempted": 0,
                "source_success_rate": 0.0,
                "sources_by_format": {},
                "source_coverage": {},
                "overlap_analysis": {},
            }

        sources_loaded = sum(
            1 for stats in self.source_stats.values() if stats["successful_loads"] > 0
        )
        sources_attempted = len(self.source_stats)
        source_success_rate = (
            sources_loaded / sources_attempted if sources_attempted > 0 else 0.0
        )

        # Group sources by format
        sources_by_format = defaultdict(list)
        for source_path, stats in self.source_stats.items():
            if stats["format"]:
                sources_by_format[stats["format"]].append(source_path)

        # Source coverage analysis (terms and relationships per source)
        source_coverage = {}
        for source_path, stats in self.source_stats.items():
            if stats["successful_loads"] > 0:
                source_coverage[source_path] = {
                    "terms_count": stats["terms_count"],
                    "relationships_count": stats["relationships_count"],
                    "format": stats["format"],
                    "ontology_id": stats["ontology_id"],
                    "average_load_time": stats["average_load_time"],
                }

        # Basic overlap analysis (identify common terms across ontologies)
        overlap_analysis = self._analyze_ontology_overlap()

        return {
            "sources_loaded": sources_loaded,
            "sources_attempted": sources_attempted,
            "source_success_rate": source_success_rate,
            "sources_by_format": dict(sources_by_format),
            "source_coverage": source_coverage,
            "overlap_analysis": overlap_analysis,
        }

    def _get_performance_statistics(self) -> Dict[str, Any]:
        """Get performance-related statistics.

        Returns:
            Dict[str, Any]: Performance statistics including timing metrics
        """
        return {
            "performance": {
                "total_load_time": self.performance_stats["total_load_time"],
                "average_load_time": self.performance_stats["average_load_time"],
                "fastest_load_time": self.performance_stats["fastest_load_time"],
                "slowest_load_time": self.performance_stats["slowest_load_time"],
            }
        }

    def _analyze_ontology_overlap(self) -> Dict[str, Any]:
        """Analyze overlap between loaded ontologies.

        Returns:
            Dict[str, Any]: Overlap analysis including common terms and relationships
        """
        if len(self.ontologies) < 2:
            return {
                "common_terms": [],
                "common_relationships": [],
                "unique_terms_per_ontology": {},
                "overlap_matrix": {},
            }

        # Collect all terms and relationships across ontologies
        ontology_terms = {}
        ontology_relationships = {}

        for ont_id, ontology in self.ontologies.items():
            ontology_terms[ont_id] = set(ontology.terms.keys())
            ontology_relationships[ont_id] = set(ontology.relationships.keys())

        # Find common terms
        all_term_sets = list(ontology_terms.values())
        common_terms = set.intersection(*all_term_sets) if all_term_sets else set()

        # Find common relationships
        all_rel_sets = list(ontology_relationships.values())
        common_relationships = (
            set.intersection(*all_rel_sets) if all_rel_sets else set()
        )

        # Calculate unique terms per ontology
        unique_terms_per_ontology = {}
        for ont_id, terms in ontology_terms.items():
            other_terms = set()
            for other_id, other_terms_set in ontology_terms.items():
                if other_id != ont_id:
                    other_terms.update(other_terms_set)
            unique_terms_per_ontology[ont_id] = list(terms - other_terms)

        # Calculate pairwise overlap matrix
        overlap_matrix = {}
        ontology_ids = list(self.ontologies.keys())
        for i, ont_id1 in enumerate(ontology_ids):
            overlap_matrix[ont_id1] = {}
            for ont_id2 in ontology_ids:
                if ont_id1 == ont_id2:
                    overlap_matrix[ont_id1][ont_id2] = 1.0
                else:
                    terms1 = ontology_terms[ont_id1]
                    terms2 = ontology_terms[ont_id2]
                    intersection = len(terms1.intersection(terms2))
                    union = len(terms1.union(terms2))
                    jaccard_similarity = intersection / union if union > 0 else 0.0
                    overlap_matrix[ont_id1][ont_id2] = jaccard_similarity

        return {
            "common_terms": list(common_terms),
            "common_relationships": list(common_relationships),
            "unique_terms_per_ontology": unique_terms_per_ontology,
            "overlap_matrix": overlap_matrix,
        }

    def get_ontology_statistics(self) -> Dict[str, Any]:
        """Get comprehensive ontology statistics with multi-source support.

        This method provides detailed statistics about loaded ontologies,
        including per-source breakdowns, format analysis, and overlap metrics.

        Returns:
            Dict[str, Any]: Comprehensive statistics dictionary
        """
        return self.get_statistics()  # Delegate to enhanced get_statistics method

    def get_source_statistics(
        self, source_path: Optional[str] = None
    ) -> Dict[str, Any]:
        """Get statistics for a specific source or all sources.

        Args:
            source_path: Path to specific source. If None, returns all source statistics.

        Returns:
            Dict[str, Any]: Source-specific statistics
        """
        if source_path:
            return self.source_stats.get(source_path, {})
        return dict(self.source_stats)

    def validate_ontology(self, ontology_id: str) -> Dict[str, Any]:
        """Validate a specific ontology using the validation pipeline.

        Args:
            ontology_id: ID of the ontology to validate

        Returns:
            Dict[str, Any]: Validation results

        Raises:
            OntologyManagerError: If ontology not found
        """
        ontology = self.get_ontology(ontology_id)
        if not ontology:
            raise OntologyManagerError(f"Ontology '{ontology_id}' not found")

        try:
            validation_result = self.validation_pipeline.validate_ontology(ontology)

            # Update ontology validation status
            if hasattr(ontology, "is_consistent"):
                ontology.is_consistent = validation_result["is_valid"]

            if hasattr(ontology, "validation_errors"):
                ontology.validation_errors = validation_result["errors"]

            self.logger.info(
                f"Validated ontology '{ontology_id}': "
                f"{'PASSED' if validation_result['is_valid'] else 'FAILED'} "
                f"({validation_result['summary']['total_errors']} errors, "
                f"{validation_result['summary']['total_warnings']} warnings)"
            )

            return validation_result

        except Exception as e:
            error_msg = f"Failed to validate ontology '{ontology_id}': {str(e)}"
            self.logger.exception(error_msg)
            raise OntologyManagerError(error_msg)

    def validate_all_ontologies(self) -> Dict[str, Dict[str, Any]]:
        """Validate all loaded ontologies.

        Returns:
            Dict[str, Dict[str, Any]]: Validation results for each ontology
        """
        results = {}

        for ontology_id in self.list_ontologies():
            try:
                results[ontology_id] = self.validate_ontology(ontology_id)
            except Exception as e:
                self.logger.error(
                    f"Failed to validate ontology '{ontology_id}': {str(e)}"
                )
                results[ontology_id] = {
                    "is_valid": False,
                    "errors": [f"Validation failed: {str(e)}"],
                    "warnings": [],
                    "validator_results": {},
                    "summary": {
                        "total_validators": 0,
                        "passed_validators": 0,
                        "failed_validators": 0,
                        "total_errors": 1,
                        "total_warnings": 0,
                    },
                }

        return results

    def get_validation_summary(self) -> Dict[str, Any]:
        """Get a summary of validation results for all ontologies.

        Returns:
            Dict[str, Any]: Summary of validation results
        """
        if not self.ontologies:
            return {
                "total_ontologies": 0,
                "valid_ontologies": 0,
                "invalid_ontologies": 0,
                "total_errors": 0,
                "total_warnings": 0,
                "validation_rate": 0.0,
            }

        validation_results = self.validate_all_ontologies()

        total_ontologies = len(validation_results)
        valid_ontologies = sum(
            1 for result in validation_results.values() if result["is_valid"]
        )
        invalid_ontologies = total_ontologies - valid_ontologies
        total_errors = sum(
            result["summary"]["total_errors"] for result in validation_results.values()
        )
        total_warnings = sum(
            result["summary"]["total_warnings"]
            for result in validation_results.values()
        )
        validation_rate = (
            valid_ontologies / total_ontologies if total_ontologies > 0 else 0.0
        )

        return {
            "total_ontologies": total_ontologies,
            "valid_ontologies": valid_ontologies,
            "invalid_ontologies": invalid_ontologies,
            "total_errors": total_errors,
            "total_warnings": total_warnings,
            "validation_rate": validation_rate,
            "detailed_results": validation_results,
        }

    def add_validation_check(self, validator) -> None:
        """Add a custom validator to the validation pipeline.

        Args:
            validator: The validator to add (must inherit from OntologyValidator)

        Raises:
            OntologyManagerError: If validator is invalid
        """
        try:
            self.validation_pipeline.add_validator(validator)
            self.logger.info(f"Added custom validator: {validator.name}")
        except Exception as e:
            error_msg = f"Failed to add validator: {str(e)}"
            self.logger.error(error_msg)
            raise OntologyManagerError(error_msg)

    def remove_validation_check(self, validator_name: str) -> bool:
        """Remove a validator from the validation pipeline.

        Args:
            validator_name: Name of the validator to remove

        Returns:
            bool: True if validator was removed, False if not found
        """
        removed = self.validation_pipeline.remove_validator(validator_name)
        if removed:
            self.logger.info(f"Removed validator: {validator_name}")
        else:
            self.logger.warning(f"Validator not found: {validator_name}")
        return removed

    def list_validators(self) -> List[str]:
        """Get list of active validators.

        Returns:
            List[str]: List of validator names
        """
        return self.validation_pipeline.list_validators()


def main():
    """Main entry point for the ontology manager CLI."""
    parser = argparse.ArgumentParser(
        description="AIM2 Ontology Manager - Manage ontology operations",
        prog="aim2-ontology-manager",
    )
    parser.add_argument("--version", action="version", version="%(prog)s 0.1.0")
    parser.add_argument(
        "--verbose", "-v", action="store_true", help="Enable verbose output"
    )

    args = parser.parse_args()

    if args.verbose:
        print("AIM2 Ontology Manager - Version 0.1.0")
        print("This is a stub implementation for testing purposes.")
    else:
        print("ontology-manager: Ready to manage ontologies!")

    return 0


if __name__ == "__main__":
    exit(main())
