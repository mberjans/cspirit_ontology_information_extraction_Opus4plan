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
except ImportError:
    # For development/testing scenarios
    from models import Ontology
    from parsers import auto_detect_parser

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
                return LoadResult(
                    success=False,
                    source_path=source_str,
                    load_time=time.time() - start_time,
                    errors=[error_msg],
                )

            # Parse ontology
            parse_result = parser.parse(source)

            if not parse_result.success or not parse_result.data:
                error_msg = f"Failed to parse ontology from {source_str}"
                self.logger.error(error_msg)
                if parse_result.errors:
                    self.logger.error(f"Parser errors: {parse_result.errors}")

                self.load_stats["failed_loads"] += 1
                return LoadResult(
                    success=False,
                    source_path=source_str,
                    load_time=time.time() - start_time,
                    errors=[error_msg] + parse_result.errors,
                    warnings=parse_result.warnings,
                )

            ontology = parse_result.data

            # Validate ontology structure
            if not isinstance(ontology, Ontology):
                error_msg = f"Parser returned invalid ontology type: {type(ontology)}"
                self.logger.error(error_msg)
                self.load_stats["failed_loads"] += 1
                return LoadResult(
                    success=False,
                    source_path=source_str,
                    load_time=time.time() - start_time,
                    errors=[error_msg],
                )

            # Store in manager
            self.ontologies[ontology.id] = ontology

            # Update cache if enabled and cache size limit allows it
            if self.enable_caching and self.cache_size_limit > 0:
                self._update_cache(source_str, ontology, start_time)

            # Update statistics
            self.load_stats["successful_loads"] += 1
            parser_format = getattr(parser, "format_name", "unknown")
            self.load_stats["formats_loaded"][parser_format] += 1

            load_time = time.time() - start_time
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

            return LoadResult(
                success=False,
                source_path=source_str,
                load_time=time.time() - start_time,
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
        """Get loading and cache statistics.

        Returns:
            Dict[str, Any]: Dictionary containing various statistics
        """
        cache_stats = {
            "cache_size": len(self._cache),
            "cache_limit": self.cache_size_limit,
            "cache_enabled": self.enable_caching,
        }

        ontology_stats = {
            "loaded_ontologies": len(self.ontologies),
            "total_terms": sum(len(ont.terms) for ont in self.ontologies.values()),
            "total_relationships": sum(
                len(ont.relationships) for ont in self.ontologies.values()
            ),
        }

        return {
            **self.load_stats,
            **cache_stats,
            **ontology_stats,
            "formats_loaded": dict(self.load_stats["formats_loaded"]),
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
