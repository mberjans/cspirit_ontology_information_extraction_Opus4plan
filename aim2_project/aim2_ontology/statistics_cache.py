#!/usr/bin/env python3
"""
Statistics Caching System for AIM2 Ontology Manager.

This module provides advanced caching capabilities specifically designed for
statistical data including overlap analysis, performance metrics, and source
coverage analysis. The statistics cache builds upon the enhanced cache
infrastructure to provide specialized caching for computational expensive
statistical operations.

Key Features:
- Statistics-specific cache entries with dependency tracking
- TTL and version-based cache invalidation
- Support for different statistics types (overlap, performance, source_coverage)
- Incremental statistics updates where possible
- Batch invalidation for multi-ontology changes
- Memory-efficient storage of statistics data
- Thread-safe operations with proper locking
- Comprehensive error handling and logging

Statistics Types Supported:
- Per-source statistics (load times, success rates, term counts)
- Overlap analysis between ontologies (common terms, Jaccard similarity)
- Performance metrics (aggregated timing data)
- Source coverage analysis (format distribution, success rates)
- Multi-ontology aggregate statistics

Classes:
- StatisticsType: Enum for supported statistics types
- StatisticsCacheEntry: Cache entry specialized for statistics data
- StatisticsDependency: Dependency tracking for cache invalidation
- StatisticsCacheManager: Main cache manager for statistics
- StatisticsInvalidationPolicy: Specialized invalidation strategies
"""

import logging
import threading
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional

try:
    from .enhanced_cache import EnhancedCacheError
except ImportError:
    # For development/testing scenarios
    from enhanced_cache import EnhancedCacheError


class StatisticsCacheError(EnhancedCacheError):
    """Base exception for statistics cache operations."""


class StatisticsTypeError(StatisticsCacheError):
    """Exception raised when statistics type is invalid or unsupported."""


class StatisticsDependencyError(StatisticsCacheError):
    """Exception raised when dependency tracking fails."""


class StatisticsInvalidationError(StatisticsCacheError):
    """Exception raised when cache invalidation fails."""


class StatisticsType(Enum):
    """Supported statistics types for specialized caching."""

    # Per-source statistics
    SOURCE_PERFORMANCE = "source_performance"
    SOURCE_COVERAGE = "source_coverage"
    SOURCE_SUCCESS_RATE = "source_success_rate"

    # Overlap analysis statistics
    ONTOLOGY_OVERLAP = "ontology_overlap"
    TERM_OVERLAP = "term_overlap"
    RELATIONSHIP_OVERLAP = "relationship_overlap"
    JACCARD_SIMILARITY = "jaccard_similarity"

    # Performance metrics
    LOAD_PERFORMANCE = "load_performance"
    CACHE_PERFORMANCE = "cache_performance"
    AGGREGATE_PERFORMANCE = "aggregate_performance"

    # Multi-source aggregations
    FORMAT_DISTRIBUTION = "format_distribution"
    SUCCESS_RATE_ANALYSIS = "success_rate_analysis"
    MULTISOURCE_COVERAGE = "multisource_coverage"

    # Custom/extensible statistics
    CUSTOM = "custom"


@dataclass
class StatisticsDependency:
    """Represents a dependency for statistics cache invalidation.

    Dependencies allow the cache to understand when statistics need to be
    recalculated based on changes to ontologies, sources, or other statistics.

    Attributes:
        dependency_type: Type of dependency (ontology, source, statistics)
        dependency_id: Unique identifier for the dependency
        version: Version/hash of the dependency for change detection
        weight: Weight factor for dependency prioritization (higher = more important)
        created_at: Timestamp when dependency was created
    """

    dependency_type: str
    dependency_id: str
    version: str
    weight: float = 1.0
    created_at: float = field(default_factory=time.time)

    def __post_init__(self):
        """Validate dependency after creation."""
        if not self.dependency_type or not isinstance(self.dependency_type, str):
            raise StatisticsDependencyError(
                "dependency_type must be a non-empty string"
            )

        if not self.dependency_id or not isinstance(self.dependency_id, str):
            raise StatisticsDependencyError("dependency_id must be a non-empty string")

        if not self.version or not isinstance(self.version, str):
            raise StatisticsDependencyError("version must be a non-empty string")

        if not isinstance(self.weight, (int, float)) or self.weight < 0:
            raise StatisticsDependencyError("weight must be a non-negative number")

    def is_valid_for_version(self, current_version: str) -> bool:
        """Check if this dependency is still valid for the given version.

        Args:
            current_version: Current version to check against

        Returns:
            bool: True if dependency is still valid
        """
        return self.version == current_version

    def to_dict(self) -> Dict[str, Any]:
        """Convert dependency to dictionary representation.

        Returns:
            Dict[str, Any]: Dictionary representation
        """
        return {
            "dependency_type": self.dependency_type,
            "dependency_id": self.dependency_id,
            "version": self.version,
            "weight": self.weight,
            "created_at": self.created_at,
        }


@dataclass
class StatisticsCacheEntry:
    """Enhanced cache entry specialized for statistics data.

    This class extends the basic cache entry concept with statistics-specific
    features including dependency tracking, incremental updates, and
    specialized validation logic.

    Attributes:
        statistics_type: Type of statistics stored in this entry
        statistics_data: The actual statistics data (dict/complex object)
        calculation_time: Time taken to calculate these statistics
        dependencies: List of dependencies for cache invalidation
        ontology_versions: Version tracking for related ontologies
        source_versions: Version tracking for related sources
        created_at: Timestamp when the cache entry was created
        last_accessed: Timestamp of the last access
        last_updated: Timestamp of the last update
        access_count: Number of times this entry has been accessed
        ttl_seconds: Time-to-live in seconds (None for no expiration)
        is_incremental: Whether this entry supports incremental updates
        update_count: Number of times this entry has been updated
        memory_footprint: Estimated memory usage in bytes
        _lock: Thread lock for safe concurrent access
    """

    statistics_type: StatisticsType
    statistics_data: Dict[str, Any]
    calculation_time: float
    dependencies: List[StatisticsDependency] = field(default_factory=list)
    ontology_versions: Dict[str, str] = field(default_factory=dict)
    source_versions: Dict[str, str] = field(default_factory=dict)
    created_at: float = field(default_factory=time.time)
    last_accessed: float = field(default_factory=time.time)
    last_updated: float = field(default_factory=time.time)
    access_count: int = 0
    ttl_seconds: Optional[float] = None
    is_incremental: bool = False
    update_count: int = 0
    memory_footprint: int = 0
    _lock: threading.RLock = field(
        default_factory=threading.RLock, init=False, repr=False
    )

    def __post_init__(self):
        """Initialize the statistics cache entry after creation."""
        self.logger = logging.getLogger(__name__)

        # Validate statistics type
        if not isinstance(self.statistics_type, StatisticsType):
            raise StatisticsTypeError(
                f"Invalid statistics type: {self.statistics_type}"
            )

        # Validate statistics data
        if not isinstance(self.statistics_data, dict):
            raise StatisticsCacheError("statistics_data must be a dictionary")

        # Calculate initial memory footprint
        self.memory_footprint = self._calculate_memory_footprint()

        self.logger.debug(
            f"Created statistics cache entry: type={self.statistics_type.value}, "
            f"memory={self.memory_footprint} bytes, dependencies={len(self.dependencies)}"
        )

    def access_data(self) -> Dict[str, Any]:
        """Access the statistics data, updating access tracking.

        Returns:
            Dict[str, Any]: The statistics data
        """
        with self._lock:
            self.access_count += 1
            self.last_accessed = time.time()

            # Return a copy to prevent external modification
            return self.statistics_data.copy()

    def update_data(
        self,
        new_data: Dict[str, Any],
        calculation_time: float,
        is_incremental: bool = False,
    ) -> bool:
        """Update the statistics data in this entry.

        Args:
            new_data: New statistics data
            calculation_time: Time taken to calculate the new data
            is_incremental: Whether this is an incremental update

        Returns:
            bool: True if update was successful
        """
        with self._lock:
            try:
                if not isinstance(new_data, dict):
                    raise StatisticsCacheError("new_data must be a dictionary")

                old_footprint = self.memory_footprint

                if is_incremental and self.is_incremental:
                    # Merge with existing data
                    self._merge_incremental_data(new_data)
                else:
                    # Replace existing data
                    self.statistics_data = new_data.copy()

                self.calculation_time = calculation_time
                self.last_updated = time.time()
                self.update_count += 1
                self.memory_footprint = self._calculate_memory_footprint()

                self.logger.debug(
                    f"Updated statistics cache entry: type={self.statistics_type.value}, "
                    f"incremental={is_incremental}, memory_change={self.memory_footprint - old_footprint}"
                )

                return True

            except Exception as e:
                self.logger.error(f"Failed to update statistics cache entry: {e}")
                return False

    def add_dependency(self, dependency: StatisticsDependency) -> None:
        """Add a dependency to this cache entry.

        Args:
            dependency: Dependency to add
        """
        with self._lock:
            if not isinstance(dependency, StatisticsDependency):
                raise StatisticsDependencyError(
                    "dependency must be a StatisticsDependency instance"
                )

            # Check for duplicate dependencies
            existing_ids = {dep.dependency_id for dep in self.dependencies}
            if dependency.dependency_id not in existing_ids:
                self.dependencies.append(dependency)
                self.logger.debug(
                    f"Added dependency {dependency.dependency_id} to cache entry"
                )

    def remove_dependency(self, dependency_id: str) -> bool:
        """Remove a dependency from this cache entry.

        Args:
            dependency_id: ID of the dependency to remove

        Returns:
            bool: True if dependency was removed
        """
        with self._lock:
            initial_count = len(self.dependencies)
            self.dependencies = [
                dep for dep in self.dependencies if dep.dependency_id != dependency_id
            ]
            removed = len(self.dependencies) < initial_count

            if removed:
                self.logger.debug(
                    f"Removed dependency {dependency_id} from cache entry"
                )

            return removed

    def is_expired(self) -> bool:
        """Check if the cache entry has expired based on TTL.

        Returns:
            bool: True if the entry has expired
        """
        if self.ttl_seconds is None:
            return False

        current_time = time.time()
        return (current_time - self.created_at) > self.ttl_seconds

    def is_valid_for_versions(
        self,
        ontology_versions: Optional[Dict[str, str]] = None,
        source_versions: Optional[Dict[str, str]] = None,
    ) -> bool:
        """Check if the cache entry is valid for the given versions.

        Args:
            ontology_versions: Current ontology versions to check against
            source_versions: Current source versions to check against

        Returns:
            bool: True if the entry is valid for all specified versions
        """
        with self._lock:
            # Check TTL expiration first
            if self.is_expired():
                return False

            # Check ontology versions
            if ontology_versions:
                for ont_id, current_version in ontology_versions.items():
                    if ont_id in self.ontology_versions:
                        if self.ontology_versions[ont_id] != current_version:
                            return False

            # Check source versions
            if source_versions:
                for source_id, current_version in source_versions.items():
                    if source_id in self.source_versions:
                        if self.source_versions[source_id] != current_version:
                            return False

            # Check dependencies
            for dependency in self.dependencies:
                if dependency.dependency_type == "ontology" and ontology_versions:
                    current_version = ontology_versions.get(dependency.dependency_id)
                    if current_version and not dependency.is_valid_for_version(
                        current_version
                    ):
                        return False
                elif dependency.dependency_type == "source" and source_versions:
                    current_version = source_versions.get(dependency.dependency_id)
                    if current_version and not dependency.is_valid_for_version(
                        current_version
                    ):
                        return False

            return True

    def update_ontology_version(self, ontology_id: str, version: str) -> None:
        """Update the version tracking for an ontology.

        Args:
            ontology_id: ID of the ontology
            version: Current version/hash of the ontology
        """
        with self._lock:
            self.ontology_versions[ontology_id] = version

    def update_source_version(self, source_id: str, version: str) -> None:
        """Update the version tracking for a source.

        Args:
            source_id: ID of the source
            version: Current version/hash of the source
        """
        with self._lock:
            self.source_versions[source_id] = version

    def get_age_seconds(self) -> float:
        """Get the age of this cache entry in seconds.

        Returns:
            float: Age in seconds since creation
        """
        return time.time() - self.created_at

    def get_staleness_score(self) -> float:
        """Calculate a staleness score for this cache entry.

        Higher scores indicate more stale entries that should be evicted first.

        Returns:
            float: Staleness score (0.0 = fresh, higher = more stale)
        """
        current_time = time.time()

        # Base score on age
        age_hours = (current_time - self.last_accessed) / 3600.0
        staleness_score = age_hours

        # Adjust for access frequency (frequently accessed = less stale)
        if self.access_count > 0:
            access_bonus = min(10.0, self.access_count / 5.0)
            staleness_score -= access_bonus

        # Adjust for update recency
        hours_since_update = (current_time - self.last_updated) / 3600.0
        staleness_score += hours_since_update * 0.5

        # Penalty for expired entries
        if self.is_expired():
            staleness_score += 1000.0

        return max(0.0, staleness_score)

    def _merge_incremental_data(self, new_data: Dict[str, Any]) -> None:
        """Merge new data with existing data for incremental updates.

        Args:
            new_data: New data to merge
        """
        # Basic merge strategy - can be customized based on statistics type
        for key, value in new_data.items():
            if key in self.statistics_data:
                if isinstance(value, dict) and isinstance(
                    self.statistics_data[key], dict
                ):
                    # Deep merge for nested dictionaries
                    self.statistics_data[key].update(value)
                elif isinstance(value, list) and isinstance(
                    self.statistics_data[key], list
                ):
                    # Extend lists
                    self.statistics_data[key].extend(value)
                elif isinstance(value, (int, float)) and isinstance(
                    self.statistics_data[key], (int, float)
                ):
                    # Sum numeric values
                    self.statistics_data[key] += value
                else:
                    # Replace with new value
                    self.statistics_data[key] = value
            else:
                # Add new key
                self.statistics_data[key] = value

    def _calculate_memory_footprint(self) -> int:
        """Calculate the estimated memory footprint of this cache entry.

        Returns:
            int: Estimated memory usage in bytes
        """
        try:
            import json

            # Use JSON serialization as a rough approximation
            json_str = json.dumps(self.statistics_data, default=str)
            data_size = len(json_str.encode("utf-8"))

            # Add overhead for the entry itself
            overhead = 1024  # Base overhead
            overhead += len(self.dependencies) * 256  # Dependency overhead
            overhead += len(self.ontology_versions) * 128  # Version tracking overhead
            overhead += len(self.source_versions) * 128

            return data_size + overhead

        except Exception:
            # Fallback estimation
            base_size = 2048  # Base entry overhead
            data_keys = len(self.statistics_data) * 64  # Rough key overhead
            dependency_size = len(self.dependencies) * 256
            return base_size + data_keys + dependency_size

    def to_dict(self) -> Dict[str, Any]:
        """Convert cache entry to dictionary representation.

        Returns:
            Dict[str, Any]: Dictionary representation of the cache entry
        """
        return {
            "statistics_type": self.statistics_type.value,
            "calculation_time": self.calculation_time,
            "created_at": self.created_at,
            "last_accessed": self.last_accessed,
            "last_updated": self.last_updated,
            "access_count": self.access_count,
            "ttl_seconds": self.ttl_seconds,
            "is_incremental": self.is_incremental,
            "update_count": self.update_count,
            "memory_footprint": self.memory_footprint,
            "dependencies_count": len(self.dependencies),
            "ontology_versions_count": len(self.ontology_versions),
            "source_versions_count": len(self.source_versions),
            "is_expired": self.is_expired(),
            "age_seconds": self.get_age_seconds(),
            "staleness_score": self.get_staleness_score(),
            "data_keys": list(self.statistics_data.keys()),
        }


class StatisticsInvalidationPolicy(ABC):
    """Abstract base class for statistics cache invalidation policies."""

    def __init__(self, name: str, description: str):
        """Initialize the invalidation policy.

        Args:
            name: Name of the invalidation policy
            description: Description of the policy behavior
        """
        self.name = name
        self.description = description
        self.logger = logging.getLogger(__name__)

    @abstractmethod
    def should_invalidate_entry(
        self, entry: StatisticsCacheEntry, change_context: Dict[str, Any]
    ) -> bool:
        """Determine if a cache entry should be invalidated.

        Args:
            entry: Cache entry to evaluate
            change_context: Context about what changed

        Returns:
            bool: True if the entry should be invalidated
        """

    @abstractmethod
    def get_invalidation_priority(
        self, entry: StatisticsCacheEntry, change_context: Dict[str, Any]
    ) -> float:
        """Get the priority for invalidating this entry.

        Args:
            entry: Cache entry to evaluate
            change_context: Context about what changed

        Returns:
            float: Priority score (higher = higher priority for invalidation)
        """


class DependencyInvalidationPolicy(StatisticsInvalidationPolicy):
    """Invalidation policy based on dependency tracking."""

    def __init__(self):
        """Initialize the dependency-based invalidation policy."""
        super().__init__(
            name="Dependency-Based",
            description="Invalidates entries based on dependency version changes",
        )

    def should_invalidate_entry(
        self, entry: StatisticsCacheEntry, change_context: Dict[str, Any]
    ) -> bool:
        """Determine if entry should be invalidated based on dependencies.

        Args:
            entry: Cache entry to evaluate
            change_context: Context containing version changes

        Returns:
            bool: True if the entry should be invalidated
        """
        # Check if entry is expired
        if entry.is_expired():
            return True

        # Check ontology version changes
        ontology_versions = change_context.get("ontology_versions", {})
        source_versions = change_context.get("source_versions", {})

        return not entry.is_valid_for_versions(ontology_versions, source_versions)

    def get_invalidation_priority(
        self, entry: StatisticsCacheEntry, change_context: Dict[str, Any]
    ) -> float:
        """Get invalidation priority based on dependency weights.

        Args:
            entry: Cache entry to evaluate
            change_context: Context about what changed

        Returns:
            float: Priority score
        """
        if entry.is_expired():
            return 1000.0  # Highest priority for expired entries

        priority = 0.0
        changed_ontologies = set(change_context.get("changed_ontologies", []))
        changed_sources = set(change_context.get("changed_sources", []))

        # Calculate priority based on dependency weights
        for dependency in entry.dependencies:
            if (
                dependency.dependency_type == "ontology"
                and dependency.dependency_id in changed_ontologies
            ):
                priority += dependency.weight * 100
            elif (
                dependency.dependency_type == "source"
                and dependency.dependency_id in changed_sources
            ):
                priority += dependency.weight * 50

        # Add staleness factor
        priority += entry.get_staleness_score()

        return priority
