#!/usr/bin/env python3
"""
Enhanced Caching System for AIM2 Ontology Manager.

This module provides advanced caching capabilities including compression,
TTL management, priority-based eviction, and memory pressure monitoring.
The enhanced caching system builds upon the existing CacheEntry to provide
production-ready caching with better performance and resource management.

Key Features:
- Compression support (zlib, gzip) with automatic fallback
- TTL (Time-To-Live) expiration management
- Priority-based cache entry classification
- Multiple eviction policies (LRU+TTL hybrid, priority-aware, memory pressure)
- Memory management and cross-cache tracking
- Thread-safe operations with appropriate locking
- Comprehensive error handling and logging

Classes:
- CompressionType: Enum for supported compression types
- CachePriority: Enum for cache entry priority levels
- EnhancedCacheEntry: Enhanced cache entry with compression and TTL
- EvictionPolicy: Abstract base class for eviction strategies
- LRUTTLHybridPolicy: LRU eviction with TTL consideration
- PriorityAwarePolicy: Priority-based eviction strategy
- MemoryPressurePolicy: Memory pressure-aware eviction
- CacheConfiguration: Configuration dataclass for cache settings
- MemoryManager: Cross-cache memory tracking and management
"""

import gzip
import logging
import threading
import time
import zlib
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

try:
    from .models import Ontology
except ImportError:
    # For development/testing scenarios
    from models import Ontology


class EnhancedCacheError(Exception):
    """Base exception for enhanced cache operations."""


class CompressionError(EnhancedCacheError):
    """Exception raised when compression/decompression fails."""


class MemoryAllocationError(EnhancedCacheError):
    """Exception raised when memory allocation fails."""


class EvictionPolicyError(EnhancedCacheError):
    """Exception raised when eviction policy operations fail."""


class CacheConfigurationError(EnhancedCacheError):
    """Exception raised when cache configuration is invalid."""


class CompressionType(Enum):
    """Supported compression types for cache entries."""

    NONE = "none"
    ZLIB = "zlib"
    GZIP = "gzip"


class CachePriority(Enum):
    """Priority levels for cache entries."""

    LOW = 1
    MEDIUM = 2
    HIGH = 3
    CRITICAL = 4


@dataclass
class EnhancedCacheEntry:
    """Enhanced cache entry with compression, TTL, and priority support.

    This class extends the basic CacheEntry concept with advanced features
    including data compression, expiration management, and priority classification.
    It maintains backward compatibility while providing significant enhancements.

    Attributes:
        ontology: The cached ontology object
        load_time: Timestamp when the ontology was loaded
        source_path: Path to the source file or URL
        file_mtime: Modification time of the source file (if applicable)
        access_count: Number of times this entry has been accessed
        last_accessed: Timestamp of the last access
        created_at: Timestamp when the cache entry was created
        ttl_seconds: Time-to-live in seconds (None for no expiration)
        priority: Priority level for eviction decisions
        compression_type: Type of compression used for the data
        compressed_data: Compressed serialized data (if compression is used)
        uncompressed_size: Original size of the data before compression
        compressed_size: Size of the data after compression
        compression_ratio: Ratio of compressed to uncompressed size
        is_compressed: Whether the data is currently compressed
        _lock: Thread lock for safe concurrent access
    """

    ontology: Optional[Ontology]
    load_time: float
    source_path: str
    file_mtime: Optional[float] = None
    access_count: int = 0
    last_accessed: float = field(default_factory=time.time)
    created_at: float = field(default_factory=time.time)
    ttl_seconds: Optional[float] = None
    priority: CachePriority = CachePriority.MEDIUM
    compression_type: CompressionType = CompressionType.NONE
    compressed_data: Optional[bytes] = None
    uncompressed_size: int = 0
    compressed_size: int = 0
    compression_ratio: float = 1.0
    is_compressed: bool = False
    _lock: threading.RLock = field(
        default_factory=threading.RLock, init=False, repr=False
    )

    def __post_init__(self):
        """Initialize the cache entry after creation."""
        self.logger = logging.getLogger(__name__)
        if self.ontology is not None and not self.is_compressed:
            # Calculate uncompressed size
            self.uncompressed_size = self._calculate_ontology_size(self.ontology)
            self.compressed_size = self.uncompressed_size

    def compress_data(
        self, compression_type: CompressionType = CompressionType.ZLIB
    ) -> bool:
        """Compress the cached ontology data.

        Args:
            compression_type: Type of compression to use

        Returns:
            bool: True if compression was successful, False otherwise

        Raises:
            CompressionError: If compression fails critically
        """
        with self._lock:
            if self.is_compressed:
                self.logger.debug(f"Entry {self.source_path} is already compressed")
                return True

            if self.ontology is None:
                self.logger.warning(
                    f"Cannot compress entry {self.source_path}: no ontology data"
                )
                return False

            try:
                # Validate compression type
                if not isinstance(compression_type, CompressionType):
                    raise CompressionError(
                        f"Invalid compression type: {compression_type}"
                    )

                # Serialize ontology to JSON
                try:
                    json_data = self.ontology.to_json().encode("utf-8")
                    self.uncompressed_size = len(json_data)
                except Exception as e:
                    raise CompressionError(f"Failed to serialize ontology to JSON: {e}")

                # Apply compression with error handling for each type
                try:
                    if compression_type == CompressionType.ZLIB:
                        self.compressed_data = zlib.compress(json_data, level=6)
                    elif compression_type == CompressionType.GZIP:
                        self.compressed_data = gzip.compress(json_data, compresslevel=6)
                    elif compression_type == CompressionType.NONE:
                        # No compression - store as bytes
                        self.compressed_data = json_data
                    else:
                        raise CompressionError(
                            f"Unsupported compression type: {compression_type}"
                        )

                except (zlib.error, OSError) as e:
                    # Try fallback compression
                    self.logger.warning(
                        f"Primary compression failed for {self.source_path}, trying fallback: {e}"
                    )
                    try:
                        if compression_type != CompressionType.NONE:
                            # Fallback to no compression
                            self.compressed_data = json_data
                            compression_type = CompressionType.NONE
                        else:
                            raise CompressionError(
                                f"Fallback compression also failed: {e}"
                            )
                    except Exception as fallback_error:
                        raise CompressionError(
                            f"All compression methods failed: {fallback_error}"
                        )

                # Validate compression result
                if self.compressed_data is None:
                    raise CompressionError("Compression produced no data")

                self.compressed_size = len(self.compressed_data)
                self.compression_ratio = (
                    self.compressed_size / self.uncompressed_size
                    if self.uncompressed_size > 0
                    else 1.0
                )
                self.compression_type = compression_type
                self.is_compressed = True

                # Clear the uncompressed ontology to save memory
                self.ontology = None

                self.logger.debug(
                    f"Compressed {self.source_path}: {self.uncompressed_size} -> {self.compressed_size} bytes "
                    f"(ratio: {self.compression_ratio:.3f})"
                )

                return True

            except CompressionError:
                # Re-raise compression errors
                raise
            except Exception as e:
                error_msg = (
                    f"Unexpected error during compression of {self.source_path}: {e}"
                )
                self.logger.error(error_msg)
                raise CompressionError(error_msg)

    def decompress_data(self) -> Optional[Ontology]:
        """Decompress and restore the cached ontology data.

        Returns:
            Optional[Ontology]: The decompressed ontology, or None if decompression failed

        Raises:
            CompressionError: If decompression fails critically
        """
        with self._lock:
            if not self.is_compressed:
                self.logger.debug(f"Entry {self.source_path} is not compressed")
                return self.ontology

            if self.compressed_data is None:
                raise CompressionError(
                    f"No compressed data available for {self.source_path}"
                )

            try:
                # Decompress data with type-specific error handling
                json_data = None
                try:
                    if self.compression_type == CompressionType.ZLIB:
                        json_data = zlib.decompress(self.compressed_data).decode(
                            "utf-8"
                        )
                    elif self.compression_type == CompressionType.GZIP:
                        json_data = gzip.decompress(self.compressed_data).decode(
                            "utf-8"
                        )
                    elif self.compression_type == CompressionType.NONE:
                        # No compression - decode directly
                        json_data = self.compressed_data.decode("utf-8")
                    else:
                        raise CompressionError(
                            f"Unknown compression type: {self.compression_type}"
                        )

                except (zlib.error, OSError, UnicodeDecodeError) as e:
                    raise CompressionError(
                        f"Failed to decompress data for {self.source_path}: {e}"
                    )

                if json_data is None:
                    raise CompressionError(
                        f"Decompression produced no data for {self.source_path}"
                    )

                # Reconstruct ontology from JSON
                try:
                    import json

                    data_dict = json.loads(json_data)
                except json.JSONDecodeError as e:
                    raise CompressionError(
                        f"Failed to parse JSON data for {self.source_path}: {e}"
                    )

                # Create ontology from dictionary with fallback handling
                ontology = None
                try:
                    if hasattr(Ontology, "from_dict"):
                        ontology = Ontology.from_dict(data_dict)
                    elif hasattr(Ontology, "from_json"):
                        ontology = Ontology.from_json(json_data)
                    else:
                        # Fallback: create ontology manually with validation
                        required_fields = ["id", "name"]
                        for field in required_fields:
                            if field not in data_dict:
                                raise CompressionError(
                                    f"Missing required field '{field}' in ontology data"
                                )

                        ontology = Ontology(
                            id=data_dict["id"],
                            name=data_dict["name"],
                            version=data_dict.get("version", ""),
                            description=data_dict.get("description", ""),
                        )

                        # Set additional attributes if available
                        if hasattr(ontology, "terms") and "terms" in data_dict:
                            ontology.terms = data_dict["terms"]
                        if (
                            hasattr(ontology, "relationships")
                            and "relationships" in data_dict
                        ):
                            ontology.relationships = data_dict["relationships"]
                        if (
                            hasattr(ontology, "namespaces")
                            and "namespaces" in data_dict
                        ):
                            ontology.namespaces = data_dict["namespaces"]

                except Exception as e:
                    raise CompressionError(
                        f"Failed to reconstruct ontology for {self.source_path}: {e}"
                    )

                if ontology is None:
                    raise CompressionError(
                        f"Failed to create ontology object for {self.source_path}"
                    )

                self.ontology = ontology
                self.logger.debug(f"Successfully decompressed {self.source_path}")
                return ontology

            except CompressionError:
                # Re-raise compression errors
                raise
            except Exception as e:
                error_msg = (
                    f"Unexpected error during decompression of {self.source_path}: {e}"
                )
                self.logger.error(error_msg)
                raise CompressionError(error_msg)

    def get_ontology(self) -> Optional[Ontology]:
        """Get the ontology, decompressing if necessary.

        Returns:
            Optional[Ontology]: The ontology object
        """
        with self._lock:
            self.access_count += 1
            self.last_accessed = time.time()

            if self.is_compressed:
                if self.ontology is None:
                    return self.decompress_data()
                return self.ontology
            else:
                return self.ontology

    def is_expired(self) -> bool:
        """Check if the cache entry has expired based on TTL.

        Returns:
            bool: True if the entry has expired, False otherwise
        """
        if self.ttl_seconds is None:
            return False

        current_time = time.time()
        return (current_time - self.created_at) > self.ttl_seconds

    def time_until_expiry(self) -> Optional[float]:
        """Get the time remaining until expiry.

        Returns:
            Optional[float]: Seconds until expiry, or None if no TTL is set
        """
        if self.ttl_seconds is None:
            return None

        current_time = time.time()
        elapsed_time = current_time - self.created_at
        return max(0, self.ttl_seconds - elapsed_time)

    def get_memory_footprint(self) -> int:
        """Get the current memory footprint of the cache entry.

        Returns:
            int: Memory footprint in bytes
        """
        with self._lock:
            footprint = 0

            # Size of compressed data (if compressed)
            if self.is_compressed and self.compressed_data:
                footprint += len(self.compressed_data)

            # Rough size estimate of the ontology object (if not compressed)
            if not self.is_compressed and self.ontology:
                footprint += self.uncompressed_size

            # Add overhead for the cache entry itself (rough estimate)
            footprint += 1024  # Metadata overhead

            return footprint

    def update_file_mtime(self, source_path: Optional[str] = None) -> bool:
        """Update the file modification time for cache validation.

        Args:
            source_path: Optional path to check (uses self.source_path if None)

        Returns:
            bool: True if mtime was updated successfully
        """
        try:
            path_to_check = source_path or self.source_path
            if Path(path_to_check).exists():
                self.file_mtime = Path(path_to_check).stat().st_mtime
                return True
        except (OSError, AttributeError):
            pass
        return False

    def is_file_modified(self) -> bool:
        """Check if the source file has been modified since caching.

        Returns:
            bool: True if file has been modified or cannot be checked
        """
        if self.file_mtime is None:
            return True  # Assume modified if we don't have mtime info

        try:
            if Path(self.source_path).exists():
                current_mtime = Path(self.source_path).stat().st_mtime
                return current_mtime > self.file_mtime
        except (OSError, AttributeError):
            pass

        return True  # Assume modified if we can't check

    def _calculate_ontology_size(self, ontology: Ontology) -> int:
        """Calculate the approximate size of an ontology object.

        Args:
            ontology: The ontology to measure

        Returns:
            int: Approximate size in bytes
        """
        try:
            # Use JSON serialization as a size approximation
            json_str = ontology.to_json()
            return len(json_str.encode("utf-8"))
        except Exception:
            # Fallback estimation based on terms and relationships
            estimated_size = 0
            estimated_size += len(ontology.terms) * 512  # ~512 bytes per term
            estimated_size += (
                len(ontology.relationships) * 256
            )  # ~256 bytes per relationship
            estimated_size += 1024  # Base overhead
            return estimated_size

    def to_dict(self) -> Dict[str, Any]:
        """Convert cache entry to dictionary representation.

        Returns:
            Dict[str, Any]: Dictionary representation of the cache entry
        """
        return {
            "source_path": self.source_path,
            "load_time": self.load_time,
            "file_mtime": self.file_mtime,
            "access_count": self.access_count,
            "last_accessed": self.last_accessed,
            "created_at": self.created_at,
            "ttl_seconds": self.ttl_seconds,
            "priority": self.priority.name,
            "compression_type": self.compression_type.name,
            "uncompressed_size": self.uncompressed_size,
            "compressed_size": self.compressed_size,
            "compression_ratio": self.compression_ratio,
            "is_compressed": self.is_compressed,
            "is_expired": self.is_expired(),
            "memory_footprint": self.get_memory_footprint(),
            "ontology_id": self.ontology.id if self.ontology else None,
        }


class EvictionPolicy(ABC):
    """Abstract base class for cache eviction policies.

    Eviction policies determine which cache entries should be removed when
    the cache reaches its capacity limits or when memory pressure is detected.
    Different policies can optimize for different scenarios (performance, memory, etc.).
    """

    def __init__(self, name: str, description: str):
        """Initialize the eviction policy.

        Args:
            name: Name of the eviction policy
            description: Description of the policy behavior
        """
        self.name = name
        self.description = description
        self.logger = logging.getLogger(__name__)

    @abstractmethod
    def select_candidates_for_eviction(
        self,
        cache_entries: Dict[str, EnhancedCacheEntry],
        target_count: int,
        memory_pressure: float = 0.0,
        **kwargs,
    ) -> List[str]:
        """Select cache entries for eviction.

        Args:
            cache_entries: Dictionary of cache entries by key
            target_count: Number of entries to evict
            memory_pressure: Current memory pressure level (0.0-1.0)
            **kwargs: Additional policy-specific parameters

        Returns:
            List[str]: List of cache keys to evict, ordered by priority
        """

    def should_evict_entry(
        self, entry: EnhancedCacheEntry, memory_pressure: float = 0.0, **kwargs
    ) -> bool:
        """Check if a specific entry should be evicted.

        Args:
            entry: Cache entry to evaluate
            memory_pressure: Current memory pressure level (0.0-1.0)
            **kwargs: Additional policy-specific parameters

        Returns:
            bool: True if the entry should be evicted
        """
        # Default implementation: evict expired entries
        return entry.is_expired()

    def get_eviction_score(
        self, entry: EnhancedCacheEntry, memory_pressure: float = 0.0, **kwargs
    ) -> float:
        """Calculate an eviction score for a cache entry.

        Higher scores indicate higher priority for eviction.

        Args:
            entry: Cache entry to score
            memory_pressure: Current memory pressure level (0.0-1.0)
            **kwargs: Additional policy-specific parameters

        Returns:
            float: Eviction score (higher = more likely to evict)
        """
        # Default implementation based on expiration and access patterns
        score = 0.0

        # Expired entries get highest score
        if entry.is_expired():
            score += 1000.0

        # Consider access frequency and recency
        current_time = time.time()
        time_since_access = current_time - entry.last_accessed
        score += time_since_access / 3600.0  # Hours since last access

        # Consider access count (inverse relationship)
        if entry.access_count > 0:
            score -= min(10.0, entry.access_count / 10.0)

        return max(0.0, score)


class LRUTTLHybridPolicy(EvictionPolicy):
    """LRU eviction policy with TTL awareness.

    This policy combines Least Recently Used (LRU) eviction with Time-To-Live (TTL)
    expiration. Expired entries are always evicted first, followed by the least
    recently used entries.
    """

    def __init__(self):
        """Initialize the LRU+TTL hybrid policy."""
        super().__init__(
            name="LRU-TTL-Hybrid",
            description="Evicts expired entries first, then least recently used entries",
        )

    def select_candidates_for_eviction(
        self,
        cache_entries: Dict[str, EnhancedCacheEntry],
        target_count: int,
        memory_pressure: float = 0.0,
        **kwargs,
    ) -> List[str]:
        """Select candidates using LRU+TTL strategy.

        Args:
            cache_entries: Dictionary of cache entries by key
            target_count: Number of entries to evict
            memory_pressure: Current memory pressure level (0.0-1.0)

        Returns:
            List[str]: List of cache keys to evict
        """
        if not cache_entries or target_count <= 0:
            return []

        candidates = []

        # First, collect all expired entries
        expired_entries = [
            (key, entry) for key, entry in cache_entries.items() if entry.is_expired()
        ]

        # Add expired entries first
        candidates.extend([key for key, _ in expired_entries])

        # If we need more candidates, use LRU strategy
        remaining_needed = target_count - len(candidates)
        if remaining_needed > 0:
            # Get non-expired entries sorted by last access time (oldest first)
            non_expired = [
                (key, entry)
                for key, entry in cache_entries.items()
                if not entry.is_expired() and key not in candidates
            ]

            non_expired.sort(key=lambda x: x[1].last_accessed)

            # Add LRU entries
            lru_candidates = [key for key, _ in non_expired[:remaining_needed]]
            candidates.extend(lru_candidates)

        return candidates[:target_count]

    def get_eviction_score(
        self, entry: EnhancedCacheEntry, memory_pressure: float = 0.0, **kwargs
    ) -> float:
        """Calculate eviction score for LRU+TTL policy.

        Args:
            entry: Cache entry to score
            memory_pressure: Current memory pressure level

        Returns:
            float: Eviction score
        """
        score = 0.0
        current_time = time.time()

        # Expired entries get maximum score
        if entry.is_expired():
            return 1000.0

        # Score based on last access time (older = higher score)
        time_since_access = current_time - entry.last_accessed
        score += time_since_access / 3600.0  # Convert to hours

        # Adjust for memory pressure
        if memory_pressure > 0.5:
            score *= 1.0 + memory_pressure

        return score


class PriorityAwarePolicy(EvictionPolicy):
    """Priority-aware eviction policy.

    This policy considers the priority level of cache entries when making
    eviction decisions. Lower priority entries are evicted first, with
    additional consideration for TTL and access patterns.
    """

    def __init__(self):
        """Initialize the priority-aware policy."""
        super().__init__(
            name="Priority-Aware",
            description="Evicts lower priority entries first, considering TTL and access patterns",
        )

    def select_candidates_for_eviction(
        self,
        cache_entries: Dict[str, EnhancedCacheEntry],
        target_count: int,
        memory_pressure: float = 0.0,
        **kwargs,
    ) -> List[str]:
        """Select candidates using priority-aware strategy.

        Args:
            cache_entries: Dictionary of cache entries by key
            target_count: Number of entries to evict
            memory_pressure: Current memory pressure level

        Returns:
            List[str]: List of cache keys to evict
        """
        if not cache_entries or target_count <= 0:
            return []

        # Calculate eviction scores for all entries
        scored_entries = [
            (key, self.get_eviction_score(entry, memory_pressure))
            for key, entry in cache_entries.items()
        ]

        # Sort by eviction score (highest first)
        scored_entries.sort(key=lambda x: x[1], reverse=True)

        # Return top candidates
        return [key for key, _ in scored_entries[:target_count]]

    def get_eviction_score(
        self, entry: EnhancedCacheEntry, memory_pressure: float = 0.0, **kwargs
    ) -> float:
        """Calculate eviction score for priority-aware policy.

        Args:
            entry: Cache entry to score
            memory_pressure: Current memory pressure level

        Returns:
            float: Eviction score
        """
        score = 0.0
        current_time = time.time()

        # Expired entries get highest score regardless of priority
        if entry.is_expired():
            return 1000.0

        # Base score on inverse priority (lower priority = higher score)
        priority_score = (5 - entry.priority.value) * 100  # 100-400 range
        score += priority_score

        # Add time-based component (less weight than priority)
        time_since_access = current_time - entry.last_accessed
        score += (time_since_access / 3600.0) * 10  # Max 10 points per hour

        # Reduce score based on access frequency
        if entry.access_count > 0:
            access_bonus = min(50.0, entry.access_count * 2.0)
            score -= access_bonus

        # Adjust for memory pressure
        if memory_pressure > 0.5:
            # Under memory pressure, be more aggressive with low-priority items
            if entry.priority in (CachePriority.LOW, CachePriority.MEDIUM):
                score *= 1.0 + memory_pressure

        return max(0.0, score)


class MemoryPressurePolicy(EvictionPolicy):
    """Memory pressure-aware eviction policy.

    This policy adapts its behavior based on current memory pressure levels.
    Under high memory pressure, it becomes more aggressive in evicting entries,
    prioritizing larger entries and those with lower access frequency.
    """

    def __init__(self, memory_threshold: float = 0.8):
        """Initialize the memory pressure-aware policy.

        Args:
            memory_threshold: Memory usage threshold above which to become aggressive
        """
        super().__init__(
            name="Memory-Pressure-Aware",
            description="Adapts eviction behavior based on memory pressure levels",
        )
        self.memory_threshold = memory_threshold

    def select_candidates_for_eviction(
        self,
        cache_entries: Dict[str, EnhancedCacheEntry],
        target_count: int,
        memory_pressure: float = 0.0,
        **kwargs,
    ) -> List[str]:
        """Select candidates using memory pressure-aware strategy.

        Args:
            cache_entries: Dictionary of cache entries by key
            target_count: Number of entries to evict
            memory_pressure: Current memory pressure level

        Returns:
            List[str]: List of cache keys to evict
        """
        if not cache_entries or target_count <= 0:
            return []

        # Under high memory pressure, prioritize larger entries
        if memory_pressure > self.memory_threshold:
            return self._select_high_pressure_candidates(
                cache_entries, target_count, memory_pressure
            )
        else:
            return self._select_normal_candidates(
                cache_entries, target_count, memory_pressure
            )

    def _select_high_pressure_candidates(
        self,
        cache_entries: Dict[str, EnhancedCacheEntry],
        target_count: int,
        memory_pressure: float,
    ) -> List[str]:
        """Select candidates under high memory pressure.

        Args:
            cache_entries: Dictionary of cache entries by key
            target_count: Number of entries to evict
            memory_pressure: Current memory pressure level

        Returns:
            List[str]: List of cache keys to evict
        """
        # Prioritize larger entries and low-priority entries
        scored_entries = [
            (key, self._get_high_pressure_score(entry, memory_pressure))
            for key, entry in cache_entries.items()
        ]

        scored_entries.sort(key=lambda x: x[1], reverse=True)
        return [key for key, _ in scored_entries[:target_count]]

    def _select_normal_candidates(
        self,
        cache_entries: Dict[str, EnhancedCacheEntry],
        target_count: int,
        memory_pressure: float,
    ) -> List[str]:
        """Select candidates under normal memory conditions.

        Args:
            cache_entries: Dictionary of cache entries by key
            target_count: Number of entries to evict
            memory_pressure: Current memory pressure level

        Returns:
            List[str]: List of cache keys to evict
        """
        # Use standard LRU+TTL approach
        candidates = []

        # First, expired entries
        expired_entries = [
            (key, entry) for key, entry in cache_entries.items() if entry.is_expired()
        ]
        candidates.extend([key for key, _ in expired_entries])

        # Then LRU entries
        remaining_needed = target_count - len(candidates)
        if remaining_needed > 0:
            non_expired = [
                (key, entry)
                for key, entry in cache_entries.items()
                if not entry.is_expired() and key not in candidates
            ]
            non_expired.sort(key=lambda x: x[1].last_accessed)
            candidates.extend([key for key, _ in non_expired[:remaining_needed]])

        return candidates[:target_count]

    def _get_high_pressure_score(
        self, entry: EnhancedCacheEntry, memory_pressure: float
    ) -> float:
        """Get eviction score under high memory pressure.

        Args:
            entry: Cache entry to score
            memory_pressure: Current memory pressure level

        Returns:
            float: Eviction score
        """
        score = 0.0

        # Expired entries always get highest score
        if entry.is_expired():
            return 1000.0

        # Memory footprint weight (larger entries get higher scores)
        footprint = entry.get_memory_footprint()
        score += (
            (footprint / 1024.0) * memory_pressure * 100
        )  # Up to 100 points per KB under full pressure

        # Priority consideration (more aggressive under pressure)
        priority_score = (5 - entry.priority.value) * 50 * memory_pressure
        score += priority_score

        # Access frequency (less frequently accessed = higher score)
        current_time = time.time()
        time_since_access = current_time - entry.last_accessed
        score += (time_since_access / 3600.0) * 20  # Up to 20 points per hour

        if entry.access_count > 0:
            access_penalty = min(100.0, entry.access_count * 5.0)
            score -= access_penalty

        return max(0.0, score)

    def get_eviction_score(
        self, entry: EnhancedCacheEntry, memory_pressure: float = 0.0, **kwargs
    ) -> float:
        """Calculate eviction score for memory pressure policy.

        Args:
            entry: Cache entry to score
            memory_pressure: Current memory pressure level

        Returns:
            float: Eviction score
        """
        if memory_pressure > self.memory_threshold:
            return self._get_high_pressure_score(entry, memory_pressure)
        else:
            # Use standard scoring under normal conditions
            return super().get_eviction_score(entry, memory_pressure, **kwargs)


@dataclass
class CacheConfiguration:
    """Configuration settings for enhanced caching system.

    This dataclass encapsulates all configuration options for the enhanced
    caching system, providing validation and conversion utilities.

    Attributes:
        max_entries: Maximum number of entries in the cache
        max_memory_mb: Maximum memory usage in megabytes
        default_ttl_seconds: Default TTL for cache entries (None for no expiration)
        compression_enabled: Whether to enable data compression
        compression_type: Default compression type to use
        compression_threshold_bytes: Minimum size threshold for compression
        eviction_policy: Name of the eviction policy to use
        memory_pressure_threshold: Memory usage threshold for pressure detection
        enable_statistics: Whether to collect detailed statistics
        statistics_interval_seconds: Interval for statistics collection
        cleanup_interval_seconds: Interval for cache cleanup operations
        thread_safe: Whether to use thread-safe operations
        validate_on_access: Whether to validate entries on access
    """

    max_entries: int = 1000
    max_memory_mb: int = 512
    default_ttl_seconds: Optional[float] = None
    compression_enabled: bool = True
    compression_type: CompressionType = CompressionType.ZLIB
    compression_threshold_bytes: int = 10240  # 10KB
    eviction_policy: str = "LRU-TTL-Hybrid"
    memory_pressure_threshold: float = 0.8
    enable_statistics: bool = True
    statistics_interval_seconds: float = 300.0  # 5 minutes
    cleanup_interval_seconds: float = 60.0  # 1 minute
    thread_safe: bool = True
    validate_on_access: bool = True

    def __post_init__(self):
        """Validate configuration after initialization."""
        self.validate()

    def validate(self) -> None:
        """Validate the configuration settings.

        Raises:
            CacheConfigurationError: If any configuration value is invalid
        """
        errors = []

        # Validate numeric constraints
        if self.max_entries <= 0:
            errors.append("max_entries must be positive")
        elif self.max_entries > 1000000:  # Reasonable upper limit
            errors.append("max_entries is too large (max: 1,000,000)")

        if self.max_memory_mb <= 0:
            errors.append("max_memory_mb must be positive")
        elif self.max_memory_mb > 100000:  # 100GB limit for safety
            errors.append("max_memory_mb is too large (max: 100,000 MB)")

        if self.default_ttl_seconds is not None:
            if self.default_ttl_seconds <= 0:
                errors.append("default_ttl_seconds must be positive or None")
            elif self.default_ttl_seconds > 86400 * 365:  # 1 year limit
                errors.append("default_ttl_seconds is too large (max: 1 year)")

        if self.compression_threshold_bytes < 0:
            errors.append("compression_threshold_bytes must be non-negative")
        elif self.compression_threshold_bytes > 1024 * 1024 * 100:  # 100MB limit
            errors.append("compression_threshold_bytes is too large (max: 100 MB)")

        # Validate enum types
        if not isinstance(self.compression_type, CompressionType):
            errors.append("compression_type must be a CompressionType enum value")

        # Validate float constraints
        if not isinstance(self.memory_pressure_threshold, (int, float)):
            errors.append("memory_pressure_threshold must be a number")
        elif not 0.0 <= self.memory_pressure_threshold <= 1.0:
            errors.append("memory_pressure_threshold must be between 0.0 and 1.0")

        if not isinstance(self.statistics_interval_seconds, (int, float)):
            errors.append("statistics_interval_seconds must be a number")
        elif self.statistics_interval_seconds <= 0:
            errors.append("statistics_interval_seconds must be positive")
        elif self.statistics_interval_seconds > 86400:  # 1 day limit
            errors.append("statistics_interval_seconds is too large (max: 1 day)")

        if not isinstance(self.cleanup_interval_seconds, (int, float)):
            errors.append("cleanup_interval_seconds must be a number")
        elif self.cleanup_interval_seconds <= 0:
            errors.append("cleanup_interval_seconds must be positive")
        elif self.cleanup_interval_seconds > 86400:  # 1 day limit
            errors.append("cleanup_interval_seconds is too large (max: 1 day)")

        # Validate boolean types
        if not isinstance(self.compression_enabled, bool):
            errors.append("compression_enabled must be a boolean")

        if not isinstance(self.enable_statistics, bool):
            errors.append("enable_statistics must be a boolean")

        if not isinstance(self.thread_safe, bool):
            errors.append("thread_safe must be a boolean")

        if not isinstance(self.validate_on_access, bool):
            errors.append("validate_on_access must be a boolean")

        # Validate eviction policy name
        if not isinstance(self.eviction_policy, str):
            errors.append("eviction_policy must be a string")
        else:
            valid_policies = [
                "LRU-TTL-Hybrid",
                "Priority-Aware",
                "Memory-Pressure-Aware",
            ]
            if self.eviction_policy not in valid_policies:
                errors.append(f"eviction_policy must be one of: {valid_policies}")

        # Validate logical consistency
        if self.compression_enabled and self.compression_type == CompressionType.NONE:
            errors.append("compression_enabled is True but compression_type is NONE")

        if self.cleanup_interval_seconds > self.statistics_interval_seconds:
            errors.append(
                "cleanup_interval_seconds should not be larger than statistics_interval_seconds"
            )

        # If there are errors, raise exception with all error messages
        if errors:
            error_msg = "Cache configuration validation failed:\n" + "\n".join(
                f"  - {error}" for error in errors
            )
            raise CacheConfigurationError(error_msg)

    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary.

        Returns:
            Dict[str, Any]: Dictionary representation of the configuration
        """
        return {
            "max_entries": self.max_entries,
            "max_memory_mb": self.max_memory_mb,
            "default_ttl_seconds": self.default_ttl_seconds,
            "compression_enabled": self.compression_enabled,
            "compression_type": self.compression_type.name,
            "compression_threshold_bytes": self.compression_threshold_bytes,
            "eviction_policy": self.eviction_policy,
            "memory_pressure_threshold": self.memory_pressure_threshold,
            "enable_statistics": self.enable_statistics,
            "statistics_interval_seconds": self.statistics_interval_seconds,
            "cleanup_interval_seconds": self.cleanup_interval_seconds,
            "thread_safe": self.thread_safe,
            "validate_on_access": self.validate_on_access,
        }

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> "CacheConfiguration":
        """Create configuration from dictionary.

        Args:
            config_dict: Dictionary containing configuration values

        Returns:
            CacheConfiguration: New configuration instance
        """
        # Handle enum conversion
        config_copy = config_dict.copy()
        if "compression_type" in config_copy and isinstance(
            config_copy["compression_type"], str
        ):
            config_copy["compression_type"] = CompressionType(
                config_copy["compression_type"].lower()
            )

        return cls(**config_copy)

    def get_max_memory_bytes(self) -> int:
        """Get maximum memory in bytes.

        Returns:
            int: Maximum memory in bytes
        """
        return self.max_memory_mb * 1024 * 1024

    def should_compress(self, data_size: int) -> bool:
        """Check if data should be compressed based on configuration.

        Args:
            data_size: Size of the data in bytes

        Returns:
            bool: True if data should be compressed
        """
        return (
            self.compression_enabled
            and data_size >= self.compression_threshold_bytes
            and self.compression_type != CompressionType.NONE
        )


class MemoryManager:
    """Cross-cache memory tracking and management.

    This class provides centralized memory management across multiple cache
    instances, enabling global memory pressure detection and coordinated
    eviction decisions.

    Attributes:
        max_memory_bytes: Maximum allowed memory usage across all caches
        current_memory_bytes: Current memory usage
        cache_registrations: Dictionary of registered cache instances
        memory_pressure_threshold: Threshold for memory pressure detection
        _lock: Thread lock for thread-safe operations
        logger: Logger instance for this class
    """

    def __init__(
        self, max_memory_mb: int = 1024, memory_pressure_threshold: float = 0.8
    ):
        """Initialize the memory manager.

        Args:
            max_memory_mb: Maximum memory usage in megabytes
            memory_pressure_threshold: Threshold for memory pressure detection
        """
        self.max_memory_bytes = max_memory_mb * 1024 * 1024
        self.current_memory_bytes = 0
        self.cache_registrations: Dict[str, Any] = {}  # cache_id -> cache_instance
        self.memory_pressure_threshold = memory_pressure_threshold
        self._lock = threading.RLock()
        self.logger = logging.getLogger(__name__)

        # Statistics
        self.allocation_history: List[
            Tuple[float, int]
        ] = []  # (timestamp, memory_change)
        self.peak_memory_usage = 0
        self.memory_pressure_events = 0

    def register_cache(self, cache_id: str, cache_instance: Any) -> None:
        """Register a cache instance for memory tracking.

        Args:
            cache_id: Unique identifier for the cache
            cache_instance: The cache instance to register
        """
        with self._lock:
            self.cache_registrations[cache_id] = cache_instance
            self.logger.debug(f"Registered cache instance: {cache_id}")

    def unregister_cache(self, cache_id: str) -> None:
        """Unregister a cache instance.

        Args:
            cache_id: Identifier of the cache to unregister
        """
        with self._lock:
            if cache_id in self.cache_registrations:
                del self.cache_registrations[cache_id]
                self.logger.debug(f"Unregistered cache instance: {cache_id}")

    def allocate_memory(self, cache_id: str, bytes_requested: int) -> bool:
        """Request memory allocation for a cache.

        Args:
            cache_id: Identifier of the requesting cache
            bytes_requested: Number of bytes requested

        Returns:
            bool: True if allocation was granted, False if denied

        Raises:
            MemoryAllocationError: If allocation fails due to invalid parameters
        """
        # Validate input parameters
        if not isinstance(cache_id, str) or not cache_id.strip():
            raise MemoryAllocationError("cache_id must be a non-empty string")

        if not isinstance(bytes_requested, int) or bytes_requested < 0:
            raise MemoryAllocationError(
                "bytes_requested must be a non-negative integer"
            )

        # Handle zero allocation
        if bytes_requested == 0:
            return True

        with self._lock:
            if cache_id not in self.cache_registrations:
                self.logger.warning(
                    f"Memory allocation request from unregistered cache: {cache_id}"
                )
                return False

            # Check for reasonable allocation size (prevent abuse)
            max_single_allocation = (
                self.max_memory_bytes // 2
            )  # Max 50% of total memory
            if bytes_requested > max_single_allocation:
                self.logger.error(
                    f"Memory allocation request too large for {cache_id}: "
                    f"{bytes_requested} bytes (max: {max_single_allocation})"
                )
                return False

            projected_usage = self.current_memory_bytes + bytes_requested

            # If allocation would exceed limit, try to free memory
            if projected_usage > self.max_memory_bytes:
                memory_pressure = self.get_memory_pressure()
                self.logger.info(
                    f"Memory allocation for {cache_id} would exceed limit. "
                    f"Current pressure: {memory_pressure:.2%}"
                )

                if memory_pressure > self.memory_pressure_threshold:
                    self.memory_pressure_events += 1
                    try:
                        # Trigger memory pressure response
                        freed_bytes = self._handle_memory_pressure(bytes_requested)
                        if freed_bytes >= bytes_requested:
                            projected_usage = (
                                self.current_memory_bytes + bytes_requested
                            )
                            self.logger.info(
                                f"Freed {freed_bytes} bytes through eviction"
                            )
                        else:
                            self.logger.warning(
                                f"Memory allocation denied for {cache_id}: "
                                f"requested {bytes_requested}, freed only {freed_bytes}"
                            )
                            return False
                    except Exception as e:
                        self.logger.error(f"Error during memory pressure handling: {e}")
                        return False
                else:
                    self.logger.warning(
                        f"Memory allocation denied for {cache_id}: "
                        f"requested {bytes_requested}, would exceed limit "
                        f"(pressure: {memory_pressure:.2%} < threshold: {self.memory_pressure_threshold:.2%})"
                    )
                    return False

            # Final check and allocation
            if projected_usage <= self.max_memory_bytes:
                self.current_memory_bytes += bytes_requested
                self.peak_memory_usage = max(
                    self.peak_memory_usage, self.current_memory_bytes
                )

                # Record allocation with timestamp
                try:
                    self.allocation_history.append((time.time(), bytes_requested))
                    self._cleanup_allocation_history()
                except Exception as e:
                    self.logger.warning(f"Failed to record allocation history: {e}")

                self.logger.debug(
                    f"Memory allocated to {cache_id}: {bytes_requested} bytes "
                    f"(total: {self.current_memory_bytes}/{self.max_memory_bytes}, "
                    f"pressure: {self.get_memory_pressure():.2%})"
                )
                return True

            return False

    def deallocate_memory(self, cache_id: str, bytes_freed: int) -> None:
        """Deallocate memory from a cache.

        Args:
            cache_id: Identifier of the cache freeing memory
            bytes_freed: Number of bytes being freed
        """
        with self._lock:
            if bytes_freed <= 0:
                return

            self.current_memory_bytes = max(0, self.current_memory_bytes - bytes_freed)

            # Record deallocation
            self.allocation_history.append((time.time(), -bytes_freed))
            self._cleanup_allocation_history()

            self.logger.debug(
                f"Memory deallocated from {cache_id}: {bytes_freed} bytes "
                f"(total: {self.current_memory_bytes}/{self.max_memory_bytes})"
            )

    def get_memory_pressure(self) -> float:
        """Get current memory pressure level.

        Returns:
            float: Memory pressure level (0.0-1.0)
        """
        with self._lock:
            if self.max_memory_bytes == 0:
                return 0.0
            return min(1.0, self.current_memory_bytes / self.max_memory_bytes)

    def get_available_memory(self) -> int:
        """Get available memory in bytes.

        Returns:
            int: Available memory in bytes
        """
        with self._lock:
            return max(0, self.max_memory_bytes - self.current_memory_bytes)

    def get_memory_statistics(self) -> Dict[str, Any]:
        """Get memory usage statistics.

        Returns:
            Dict[str, Any]: Memory statistics
        """
        with self._lock:
            return {
                "current_memory_bytes": self.current_memory_bytes,
                "max_memory_bytes": self.max_memory_bytes,
                "memory_pressure": self.get_memory_pressure(),
                "available_memory_bytes": self.get_available_memory(),
                "peak_memory_usage": self.peak_memory_usage,
                "memory_pressure_events": self.memory_pressure_events,
                "registered_caches": len(self.cache_registrations),
                "allocation_history_size": len(self.allocation_history),
            }

    def _handle_memory_pressure(self, bytes_needed: int) -> int:
        """Handle memory pressure by triggering evictions.

        Args:
            bytes_needed: Number of bytes needed to be freed

        Returns:
            int: Number of bytes actually freed
        """
        total_freed = 0
        memory_pressure = self.get_memory_pressure()

        self.logger.info(
            f"Handling memory pressure: {memory_pressure:.2%} usage, "
            f"need to free {bytes_needed} bytes"
        )

        # Request evictions from all registered caches
        for cache_id, cache_instance in self.cache_registrations.items():
            if hasattr(cache_instance, "handle_memory_pressure"):
                try:
                    freed = cache_instance.handle_memory_pressure(
                        memory_pressure, bytes_needed - total_freed
                    )
                    total_freed += freed

                    if total_freed >= bytes_needed:
                        break

                except Exception as e:
                    self.logger.error(
                        f"Error handling memory pressure in cache {cache_id}: {e}"
                    )

        self.logger.info(f"Memory pressure handling freed {total_freed} bytes")
        return total_freed

    def _cleanup_allocation_history(self) -> None:
        """Clean up old allocation history entries."""
        # Keep only the last 1000 entries
        if len(self.allocation_history) > 1000:
            self.allocation_history = self.allocation_history[-1000:]


# Utility functions and factory methods


def create_eviction_policy(policy_name: str, **kwargs) -> EvictionPolicy:
    """Factory function to create eviction policy instances.

    Args:
        policy_name: Name of the eviction policy
        **kwargs: Additional arguments for policy initialization

    Returns:
        EvictionPolicy: The created eviction policy instance

    Raises:
        EvictionPolicyError: If policy name is not recognized
    """
    policy_map = {
        "LRU-TTL-Hybrid": LRUTTLHybridPolicy,
        "Priority-Aware": PriorityAwarePolicy,
        "Memory-Pressure-Aware": MemoryPressurePolicy,
    }

    if policy_name not in policy_map:
        available_policies = list(policy_map.keys())
        raise EvictionPolicyError(
            f"Unknown eviction policy: {policy_name}. Available: {available_policies}"
        )

    try:
        policy_class = policy_map[policy_name]
        if policy_name == "Memory-Pressure-Aware":
            # MemoryPressurePolicy accepts additional parameters
            memory_threshold = kwargs.get("memory_threshold", 0.8)
            return policy_class(memory_threshold=memory_threshold)
        else:
            # Other policies don't take parameters
            return policy_class()
    except Exception as e:
        raise EvictionPolicyError(
            f"Failed to create eviction policy '{policy_name}': {e}"
        )


def create_cache_entry(
    ontology: Ontology,
    source_path: str,
    load_time: float = None,
    priority: CachePriority = CachePriority.MEDIUM,
    ttl_seconds: Optional[float] = None,
    compress: bool = False,
    compression_type: CompressionType = CompressionType.ZLIB,
) -> EnhancedCacheEntry:
    """Factory function to create enhanced cache entries.

    Args:
        ontology: The ontology to cache
        source_path: Path to the source file or URL
        load_time: Load timestamp (uses current time if None)
        priority: Priority level for the cache entry
        ttl_seconds: Time-to-live in seconds
        compress: Whether to compress the entry immediately
        compression_type: Type of compression to use

    Returns:
        EnhancedCacheEntry: The created cache entry

    Raises:
        EnhancedCacheError: If entry creation fails
    """
    if load_time is None:
        load_time = time.time()

    try:
        # Update file mtime if possible
        file_mtime = None
        try:
            if Path(source_path).exists():
                file_mtime = Path(source_path).stat().st_mtime
        except (OSError, AttributeError):
            pass

        # Create the cache entry
        entry = EnhancedCacheEntry(
            ontology=ontology,
            load_time=load_time,
            source_path=source_path,
            file_mtime=file_mtime,
            priority=priority,
            ttl_seconds=ttl_seconds,
        )

        # Compress if requested
        if compress:
            success = entry.compress_data(compression_type)
            if not success:
                logging.getLogger(__name__).warning(
                    f"Failed to compress cache entry for {source_path}"
                )

        return entry

    except Exception as e:
        raise EnhancedCacheError(f"Failed to create cache entry for {source_path}: {e}")


def estimate_ontology_memory_usage(ontology: Ontology) -> int:
    """Estimate the memory usage of an ontology object.

    Args:
        ontology: The ontology to estimate

    Returns:
        int: Estimated memory usage in bytes
    """
    try:
        # Try to get accurate size via JSON serialization
        json_str = ontology.to_json()
        return len(json_str.encode("utf-8"))
    except Exception:
        # Fallback estimation
        estimated_size = 0

        # Base ontology overhead
        estimated_size += 1024

        # Terms estimation
        if hasattr(ontology, "terms") and ontology.terms:
            term_count = len(ontology.terms)
            estimated_size += term_count * 512  # ~512 bytes per term

        # Relationships estimation
        if hasattr(ontology, "relationships") and ontology.relationships:
            rel_count = len(ontology.relationships)
            estimated_size += rel_count * 256  # ~256 bytes per relationship

        # Namespaces estimation
        if hasattr(ontology, "namespaces") and ontology.namespaces:
            estimated_size += len(ontology.namespaces) * 128

        return max(estimated_size, 1024)  # Minimum 1KB


def validate_cache_key(key: str) -> bool:
    """Validate a cache key for safety and consistency.

    Args:
        key: The cache key to validate

    Returns:
        bool: True if the key is valid
    """
    if not isinstance(key, str):
        return False

    if not key or not key.strip():
        return False

    # Check for reasonable length
    if len(key) > 1000:  # 1KB limit for keys
        return False

    # Check for null bytes or other problematic characters
    if "\x00" in key or "\n" in key or "\r" in key:
        return False

    return True


def get_compression_info(
    data_size: int, compression_type: CompressionType
) -> Dict[str, Any]:
    """Get information about compression for given data size and type.

    Args:
        data_size: Size of data in bytes
        compression_type: Type of compression

    Returns:
        Dict[str, Any]: Compression information including estimated ratio
    """
    info = {
        "data_size": data_size,
        "compression_type": compression_type.name,
        "estimated_compressed_size": data_size,
        "estimated_ratio": 1.0,
        "worth_compressing": False,
    }

    # Estimate compression ratios based on typical performance
    if compression_type == CompressionType.ZLIB:
        # ZLIB typically achieves 2-4x compression on JSON text
        estimated_ratio = 0.35  # ~35% of original size
        info["estimated_compressed_size"] = int(data_size * estimated_ratio)
        info["estimated_ratio"] = estimated_ratio
        info["worth_compressing"] = data_size > 1024  # Worth it for >1KB
    elif compression_type == CompressionType.GZIP:
        # GZIP similar to ZLIB but slightly different overhead
        estimated_ratio = 0.38  # ~38% of original size
        info["estimated_compressed_size"] = int(data_size * estimated_ratio)
        info["estimated_ratio"] = estimated_ratio
        info["worth_compressing"] = data_size > 2048  # Worth it for >2KB

    return info
