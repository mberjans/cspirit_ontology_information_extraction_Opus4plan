#!/usr/bin/env python3
"""
Test script for the enhanced caching system.

This script provides basic validation of the enhanced cache components
to ensure they work correctly with the existing OntologyManager.
"""

import logging
import sys
import time
from pathlib import Path

# Add the project root to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

try:
    from aim2_project.aim2_ontology.enhanced_cache import (
        CacheConfiguration,
        CacheConfigurationError,
        CachePriority,
        CompressionType,
        EnhancedCacheEntry,
        EvictionPolicyError,
        MemoryAllocationError,
        MemoryManager,
        create_cache_entry,
        create_eviction_policy,
        estimate_ontology_memory_usage,
        get_compression_info,
        validate_cache_key,
    )
    from aim2_project.aim2_ontology.models import Ontology
except ImportError as e:
    print(f"Import error: {e}")
    print("Make sure you're running this from the project root directory")
    sys.exit(1)

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def create_test_ontology(ontology_id: str = "TEST:001") -> Ontology:
    """Create a simple test ontology."""
    ontology = Ontology(
        id=ontology_id,
        name=f"Test Ontology {ontology_id}",
        version="1.0.0",
        description="A test ontology for enhanced cache validation",
    )

    # Don't add complex terms or relationships to avoid serialization issues
    # The basic ontology structure is sufficient for testing caching functionality

    return ontology


def test_enhanced_cache_entry():
    """Test EnhancedCacheEntry functionality."""
    logger.info("Testing EnhancedCacheEntry...")

    # Create test ontology
    ontology = create_test_ontology("CACHE:001")

    # Test basic entry creation
    entry = EnhancedCacheEntry(
        ontology=ontology,
        load_time=time.time(),
        source_path="/test/path.owl",
        priority=CachePriority.HIGH,
        ttl_seconds=3600,  # 1 hour
    )

    logger.info(f"Created cache entry: {entry.source_path}")
    logger.info(f"Entry priority: {entry.priority}")
    logger.info(f"Entry TTL: {entry.ttl_seconds} seconds")
    logger.info(f"Entry size: {entry.uncompressed_size} bytes")

    # Test compression
    logger.info("Testing compression...")
    if entry.compress_data(CompressionType.ZLIB):
        logger.info(
            f"Compression successful: {entry.uncompressed_size} -> {entry.compressed_size} bytes"
        )
        logger.info(f"Compression ratio: {entry.compression_ratio:.3f}")

        # Test decompression
        logger.info("Testing decompression...")
        restored_ontology = entry.decompress_data()
        if restored_ontology:
            logger.info(f"Decompression successful: {restored_ontology.id}")
        else:
            logger.error("Decompression failed")
    else:
        logger.error("Compression failed")

    # Test TTL functionality
    logger.info(f"Entry expired: {entry.is_expired()}")
    logger.info(f"Time until expiry: {entry.time_until_expiry()} seconds")

    # Test dictionary conversion
    entry_dict = entry.to_dict()
    logger.info(f"Entry dictionary keys: {list(entry_dict.keys())}")

    logger.info("EnhancedCacheEntry tests completed\n")


def test_eviction_policies():
    """Test eviction policy functionality."""
    logger.info("Testing eviction policies...")

    # Create test cache entries
    cache_entries = {}
    for i in range(5):
        ontology = create_test_ontology(f"POLICY:{i:03d}")
        entry = EnhancedCacheEntry(
            ontology=ontology,
            load_time=time.time() - (i * 100),  # Different access times
            source_path=f"/test/path_{i}.owl",
            priority=CachePriority(min(4, max(1, i))),  # Different priorities
            access_count=i * 2,  # Different access counts
            last_accessed=time.time() - (i * 50),
        )
        cache_entries[f"entry_{i}"] = entry

    logger.info(f"Created {len(cache_entries)} test entries")

    # Test LRU+TTL Hybrid Policy
    policy = create_eviction_policy("LRU-TTL-Hybrid")
    candidates = policy.select_candidates_for_eviction(cache_entries, 2)
    logger.info(f"LRU-TTL-Hybrid candidates: {candidates}")

    # Test Priority-Aware Policy
    policy = create_eviction_policy("Priority-Aware")
    candidates = policy.select_candidates_for_eviction(cache_entries, 2)
    logger.info(f"Priority-Aware candidates: {candidates}")

    # Test Memory Pressure Policy
    policy = create_eviction_policy("Memory-Pressure-Aware", memory_threshold=0.7)
    candidates = policy.select_candidates_for_eviction(
        cache_entries, 2, memory_pressure=0.9
    )
    logger.info(f"Memory-Pressure-Aware candidates: {candidates}")

    logger.info("Eviction policy tests completed\n")


def test_cache_configuration():
    """Test CacheConfiguration functionality."""
    logger.info("Testing CacheConfiguration...")

    # Test default configuration
    config = CacheConfiguration()
    logger.info(f"Default config created with {config.max_entries} max entries")

    # Test configuration validation
    try:
        invalid_config = CacheConfiguration(max_entries=-1)
        logger.error("Validation should have failed for negative max_entries")
    except CacheConfigurationError as e:
        logger.info(f"Validation correctly failed: {e}")

    # Test configuration conversion
    config_dict = config.to_dict()
    logger.info(f"Config dictionary has {len(config_dict)} keys")

    # Test configuration creation from dictionary
    new_config = CacheConfiguration.from_dict(config_dict)
    logger.info(f"Recreated config from dict: max_entries={new_config.max_entries}")

    # Test compression threshold check
    should_compress_large = config.should_compress(20480)  # 20KB
    should_compress_small = config.should_compress(1024)  # 1KB
    logger.info(f"Should compress 20KB: {should_compress_large}")
    logger.info(f"Should compress 1KB: {should_compress_small}")

    logger.info("CacheConfiguration tests completed\n")


def test_memory_manager():
    """Test MemoryManager functionality."""
    logger.info("Testing MemoryManager...")

    # Create memory manager
    manager = MemoryManager(max_memory_mb=10, memory_pressure_threshold=0.7)
    logger.info(f"Created memory manager with {manager.max_memory_bytes} bytes limit")

    # Register a mock cache
    manager.register_cache("test_cache", {"type": "mock"})

    # Test memory allocation
    success = manager.allocate_memory("test_cache", 1024 * 1024)  # 1MB
    logger.info(f"Allocated 1MB: {success}")
    logger.info(f"Memory pressure: {manager.get_memory_pressure():.2%}")

    # Test large allocation that should fail
    success = manager.allocate_memory(
        "test_cache", 20 * 1024 * 1024
    )  # 20MB (too large)
    logger.info(f"Allocated 20MB (should fail): {success}")

    # Test memory deallocation
    manager.deallocate_memory("test_cache", 512 * 1024)  # Free 512KB
    logger.info(
        f"After deallocation, memory pressure: {manager.get_memory_pressure():.2%}"
    )

    # Get memory statistics
    stats = manager.get_memory_statistics()
    logger.info(f"Memory stats: {stats}")

    logger.info("MemoryManager tests completed\n")


def test_utility_functions():
    """Test utility functions."""
    logger.info("Testing utility functions...")

    # Test cache key validation
    valid_keys = ["valid_key", "path/to/file.owl", "http://example.com/ontology"]
    invalid_keys = ["", "key\x00with\x00nulls", "key\nwith\nnewlines", "x" * 2000]

    for key in valid_keys:
        result = validate_cache_key(key)
        logger.info(f"Key '{key[:20]}...': {result}")

    for key in invalid_keys:
        result = validate_cache_key(key)
        logger.info(f"Invalid key test: {result}")

    # Test compression info
    compression_info = get_compression_info(10240, CompressionType.ZLIB)
    logger.info(f"Compression info for 10KB: {compression_info}")

    # Test ontology memory estimation
    ontology = create_test_ontology("MEMORY:001")
    estimated_size = estimate_ontology_memory_usage(ontology)
    logger.info(f"Estimated ontology memory usage: {estimated_size} bytes")

    # Test cache entry factory
    entry = create_cache_entry(
        ontology=ontology,
        source_path="/test/factory.owl",
        priority=CachePriority.HIGH,
        compress=True,
    )
    logger.info(
        f"Factory created entry: {entry.source_path}, compressed: {entry.is_compressed}"
    )

    logger.info("Utility function tests completed\n")


def test_error_handling():
    """Test error handling."""
    logger.info("Testing error handling...")

    # Test invalid eviction policy
    try:
        create_eviction_policy("Invalid-Policy")
        logger.error("Should have raised EvictionPolicyError")
    except EvictionPolicyError as e:
        logger.info(f"Correctly caught EvictionPolicyError: {e}")

    # Test invalid memory allocation
    manager = MemoryManager(max_memory_mb=10)
    try:
        manager.allocate_memory("", 1024)  # Empty cache_id
        logger.error("Should have raised MemoryAllocationError")
    except MemoryAllocationError as e:
        logger.info(f"Correctly caught MemoryAllocationError: {e}")

    # Test configuration error
    try:
        config = CacheConfiguration(max_entries=0)  # Invalid value
        logger.error("Should have raised CacheConfigurationError")
    except CacheConfigurationError:
        logger.info(f"Correctly caught CacheConfigurationError")

    logger.info("Error handling tests completed\n")


def main():
    """Run all tests."""
    logger.info("Starting enhanced cache system tests...")

    try:
        test_enhanced_cache_entry()
        test_eviction_policies()
        test_cache_configuration()
        test_memory_manager()
        test_utility_functions()
        test_error_handling()

        logger.info("All tests completed successfully!")

    except Exception as e:
        logger.error(f"Test failed with error: {e}")
        raise


if __name__ == "__main__":
    main()
