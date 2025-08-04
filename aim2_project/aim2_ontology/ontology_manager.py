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
import logging
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
from dataclasses import dataclass, field
from collections import defaultdict

# Import the models and parsers
try:
    from .models import Ontology, Term, Relationship
    from .parsers import auto_detect_parser, ParseResult
except ImportError:
    # For development/testing scenarios
    from models import Ontology, Term, Relationship
    from parsers import auto_detect_parser, ParseResult


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
    pass


class OntologyLoadError(OntologyManagerError):
    """Exception raised when ontology loading fails."""
    pass


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
            'total_loads': 0,
            'successful_loads': 0,
            'failed_loads': 0,
            'cache_hits': 0,
            'cache_misses': 0,
            'formats_loaded': defaultdict(int)
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
        self.load_stats['total_loads'] += 1
        
        try:
            # Check cache first
            if self.enable_caching and self._is_cached_and_valid(source_str):
                self.load_stats['cache_hits'] += 1
                cache_entry = self._cache[source_str]
                cache_entry.access_count += 1
                cache_entry.last_accessed = time.time()
                
                load_time = time.time() - start_time
                return LoadResult(
                    success=True,
                    ontology=cache_entry.ontology,
                    source_path=source_str,
                    load_time=load_time,
                    metadata={'cache_hit': True, 'access_count': cache_entry.access_count}
                )
            
            self.load_stats['cache_misses'] += 1
            
            # Auto-detect and create parser
            parser = auto_detect_parser(file_path=source)
            if not parser:
                error_msg = f"No suitable parser found for source: {source_str}"
                self.logger.error(error_msg)
                self.load_stats['failed_loads'] += 1
                return LoadResult(
                    success=False,
                    source_path=source_str,
                    load_time=time.time() - start_time,
                    errors=[error_msg]
                )
            
            # Parse ontology
            parse_result = parser.parse(source)
            
            if not parse_result.success or not parse_result.data:
                error_msg = f"Failed to parse ontology from {source_str}"
                self.logger.error(error_msg)
                if parse_result.errors:
                    self.logger.error(f"Parser errors: {parse_result.errors}")
                
                self.load_stats['failed_loads'] += 1
                return LoadResult(
                    success=False,
                    source_path=source_str,
                    load_time=time.time() - start_time,
                    errors=[error_msg] + parse_result.errors,
                    warnings=parse_result.warnings
                )
            
            ontology = parse_result.data
            
            # Validate ontology structure
            if not isinstance(ontology, Ontology):
                error_msg = f"Parser returned invalid ontology type: {type(ontology)}"
                self.logger.error(error_msg)
                self.load_stats['failed_loads'] += 1
                return LoadResult(
                    success=False,
                    source_path=source_str,
                    load_time=time.time() - start_time,
                    errors=[error_msg]
                )
            
            # Store in manager
            self.ontologies[ontology.id] = ontology
            
            # Update cache if enabled
            if self.enable_caching:
                self._update_cache(source_str, ontology, start_time)
            
            # Update statistics
            self.load_stats['successful_loads'] += 1
            parser_format = getattr(parser, 'format_name', 'unknown')
            self.load_stats['formats_loaded'][parser_format] += 1
            
            load_time = time.time() - start_time
            self.logger.info(f"Successfully loaded ontology {ontology.id} from {source_str} in {load_time:.3f}s")
            
            return LoadResult(
                success=True,
                ontology=ontology,
                source_path=source_str,
                load_time=load_time,
                warnings=parse_result.warnings,
                metadata={
                    'format': parser_format,
                    'terms_count': len(ontology.terms),
                    'relationships_count': len(ontology.relationships),
                    'parse_time': parse_result.parse_time
                }
            )
            
        except Exception as e:
            error_msg = f"Unexpected error loading ontology from {source_str}: {str(e)}"
            self.logger.exception(error_msg)
            self.load_stats['failed_loads'] += 1
            
            return LoadResult(
                success=False,
                source_path=source_str,
                load_time=time.time() - start_time,
                errors=[error_msg]
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
                    self.logger.warning(f"Failed to load ontology from {source}: {result.errors}")
                    
            except Exception as e:
                error_msg = f"Critical error loading {source}: {str(e)}"
                self.logger.exception(error_msg)
                results.append(LoadResult(
                    success=False,
                    source_path=str(source),
                    errors=[error_msg]
                ))
        
        return results

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
            'cache_size': len(self._cache),
            'cache_limit': self.cache_size_limit,
            'cache_enabled': self.enable_caching
        }
        
        ontology_stats = {
            'loaded_ontologies': len(self.ontologies),
            'total_terms': sum(len(ont.terms) for ont in self.ontologies.values()),
            'total_relationships': sum(len(ont.relationships) for ont in self.ontologies.values())
        }
        
        return {
            **self.load_stats,
            **cache_stats,
            **ontology_stats,
            'formats_loaded': dict(self.load_stats['formats_loaded'])
        }

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

    def _update_cache(self, source_path: str, ontology: Ontology, load_start_time: float) -> None:
        """Update cache with new ontology.
        
        Args:
            source_path: Path to the source file
            ontology: The loaded ontology
            load_start_time: When loading started
        """
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
            access_count=1
        )
        
        # Implement LRU eviction if cache is full
        if len(self._cache) >= self.cache_size_limit:
            # Remove least recently accessed entry
            lru_key = min(self._cache.keys(), 
                         key=lambda k: self._cache[k].last_accessed)
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
