#!/usr/bin/env python3
"""
Comprehensive unit tests for OntologyManager caching system.

This module contains extensive unit tests specifically focused on caching
functionality within the OntologyManager class. The tests ensure that all
caching scenarios, edge cases, and error conditions are properly handled,
including file system edge cases, cache state consistency, LRU behavior,
concurrent access safety, and error recovery.

Test Coverage:
1. File System Edge Cases (Critical):
   - File deletion after caching
   - File recreation with same name
   - Permission changes
   - Symbolic link handling
   - File modification detection

2. Cache State Consistency (Critical):
   - Cache update atomicity
   - State consistency after failures
   - Rollback scenarios
   - Partial failure recovery

3. LRU Edge Cases (Critical):
   - Tie-breaking scenarios
   - Cache size limits (0, 1, boundary conditions)
   - Concurrent eviction
   - Access pattern tracking

4. Concurrent Access Safety (Critical):
   - Thread safety testing
   - Race condition prevention
   - Concurrent cache updates
   - Lock contention scenarios

5. Error Handling and Recovery (Important):
   - Filesystem errors during cache operations
   - Cache corruption recovery
   - I/O error handling
   - Memory pressure scenarios

Key Design Principles:
- All tests are deterministic and reliable
- Comprehensive fixture setup for repeatable scenarios
- Proper mocking of external dependencies
- Thread safety validation where applicable
- Memory and resource cleanup verification
"""

import gc
import os
import stat
import tempfile
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from unittest.mock import Mock, patch

import pytest

# Import the modules to be tested
try:
    from aim2_project.aim2_ontology.models import Ontology, Relationship, Term
    from aim2_project.aim2_ontology.ontology_manager import OntologyManager
    from aim2_project.aim2_ontology.parsers import ParseResult
except ImportError:
    import warnings

    warnings.warn("Some imports failed - tests may be skipped", ImportWarning)


class TestOntologyManagerCaching:
    """Comprehensive test suite for OntologyManager caching functionality."""

    @pytest.fixture
    def temp_dir(self):
        """Create a temporary directory for test files."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            yield Path(tmp_dir)

    @pytest.fixture
    def ontology_manager(self):
        """Create an OntologyManager instance with caching enabled."""
        return OntologyManager(enable_caching=True, cache_size_limit=10)

    @pytest.fixture
    def ontology_manager_no_cache(self):
        """Create an OntologyManager instance with caching disabled."""
        return OntologyManager(enable_caching=False, cache_size_limit=0)

    @pytest.fixture
    def sample_ontology(self):
        """Create a sample ontology for testing."""
        terms = {
            "TEST:001": Term(
                id="TEST:001",
                name="Test Term 1",
                definition="A test term for caching",
                synonyms=["test1", "example1"],
                namespace="test",
            ),
            "TEST:002": Term(
                id="TEST:002",
                name="Test Term 2",
                definition="Another test term for caching",
                synonyms=["test2", "example2"],
                namespace="test",
            ),
        }

        relationships = {
            "REL:001": Relationship(
                id="REL:001",
                subject="TEST:001",
                predicate="regulates",
                object="TEST:002",
                confidence=0.95,
            )
        }

        return Ontology(
            id="CACHETEST:001",
            name="Cache Test Ontology",
            version="1.0",
            description="Test ontology for caching scenarios",
            terms=terms,
            relationships=relationships,
            namespaces=["test"],
        )

    @pytest.fixture
    def create_test_file(self, temp_dir, sample_ontology):
        """Create a test file for caching scenarios."""

        def _create_file(filename="test_ontology.owl", ontology=None):
            if ontology is None:
                ontology = sample_ontology

            test_file = temp_dir / filename
            # Create a simple OWL content
            owl_content = f"""<?xml version="1.0"?>
<rdf:RDF xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#"
         xmlns:owl="http://www.w3.org/2002/07/owl#"
         xmlns:rdfs="http://www.w3.org/2000/01/rdf-schema#">
    <owl:Ontology rdf:about="http://example.org/{ontology.id}">
        <rdfs:label>{ontology.name}</rdfs:label>
        <rdfs:comment>{ontology.description}</rdfs:comment>
    </owl:Ontology>
</rdf:RDF>"""
            test_file.write_text(owl_content)
            return str(test_file), ontology

        return _create_file

    def test_basic_cache_functionality(self, ontology_manager, create_test_file):
        """Test basic cache hit and miss functionality."""
        file_path, ontology = create_test_file()

        with patch(
            "aim2_project.aim2_ontology.ontology_manager.auto_detect_parser"
        ) as mock_auto_detect:
            # Mock parser for first load
            parser = Mock()
            parser.format_name = "owl"
            parser.parse.return_value = ParseResult(
                success=True,
                data=ontology,
                errors=[],
                warnings=[],
                metadata={},
                parse_time=0.1,
            )
            mock_auto_detect.return_value = parser

            # First load - should be cache miss
            result1 = ontology_manager.load_ontology(file_path)

            assert result1.success is True
            assert result1.ontology.id == ontology.id
            assert result1.metadata.get("cache_hit", False) is False
            assert ontology_manager.load_stats["cache_misses"] == 1
            assert ontology_manager.load_stats["cache_hits"] == 0
            assert len(ontology_manager._cache) == 1

            # Verify cache entry was created correctly
            cache_entry = ontology_manager._cache[file_path]
            assert cache_entry.ontology.id == ontology.id
            assert cache_entry.source_path == file_path
            assert cache_entry.access_count == 1
            assert cache_entry.file_mtime is not None

            # Second load - should be cache hit
            result2 = ontology_manager.load_ontology(file_path)

            assert result2.success is True
            assert result2.ontology.id == ontology.id
            assert result2.metadata.get("cache_hit", True) is True
            assert ontology_manager.load_stats["cache_misses"] == 1
            assert ontology_manager.load_stats["cache_hits"] == 1

            # Verify cache entry was updated
            cache_entry = ontology_manager._cache[file_path]
            assert cache_entry.access_count == 2

            # Verify parser was only called once
            assert parser.parse.call_count == 1

    def test_cache_disabled_functionality(
        self, ontology_manager_no_cache, create_test_file
    ):
        """Test that caching can be disabled."""
        file_path, ontology = create_test_file()

        with patch(
            "aim2_project.aim2_ontology.ontology_manager.auto_detect_parser"
        ) as mock_auto_detect:
            parser = Mock()
            parser.format_name = "owl"
            parser.parse.return_value = ParseResult(
                success=True,
                data=ontology,
                errors=[],
                warnings=[],
                metadata={},
                parse_time=0.1,
            )
            mock_auto_detect.return_value = parser

            # Load twice - both should hit parser
            result1 = ontology_manager_no_cache.load_ontology(file_path)
            result2 = ontology_manager_no_cache.load_ontology(file_path)

            assert result1.success is True
            assert result2.success is True
            assert ontology_manager_no_cache.load_stats["cache_hits"] == 0
            # When caching is disabled, cache misses should be tracked for statistics
            assert ontology_manager_no_cache.load_stats["cache_misses"] == 2
            assert len(ontology_manager_no_cache._cache) == 0

            # Parser should be called twice
            assert parser.parse.call_count == 2

    def test_file_deletion_after_caching(self, ontology_manager, create_test_file):
        """Test cache behavior when file is deleted after caching."""
        file_path, ontology = create_test_file()

        with patch(
            "aim2_project.aim2_ontology.ontology_manager.auto_detect_parser"
        ) as mock_auto_detect:
            parser = Mock()
            parser.format_name = "owl"
            parser.parse.return_value = ParseResult(
                success=True,
                data=ontology,
                errors=[],
                warnings=[],
                metadata={},
                parse_time=0.1,
            )
            mock_auto_detect.return_value = parser

            # First load - cache the file
            result1 = ontology_manager.load_ontology(file_path)
            assert result1.success is True
            assert len(ontology_manager._cache) == 1

            # Delete the file
            Path(file_path).unlink()

            # Second load - should still hit cache since file validation handles missing files gracefully
            result2 = ontology_manager.load_ontology(file_path)
            assert result2.success is True
            assert result2.metadata.get("cache_hit", True) is True
            assert ontology_manager.load_stats["cache_hits"] == 1

    def test_file_recreation_with_same_name(
        self, ontology_manager, create_test_file, temp_dir
    ):
        """Test cache behavior when file is recreated with same name but different content."""
        file_path, original_ontology = create_test_file()

        # Create a different ontology for recreation
        new_ontology = Ontology(
            id="CACHENEW:001",
            name="New Cache Test Ontology",
            version="2.0",
            description="Recreated test ontology",
            terms={},
            relationships={},
            namespaces=["new_test"],
        )

        with patch(
            "aim2_project.aim2_ontology.ontology_manager.auto_detect_parser"
        ) as mock_auto_detect:
            # First parser for original file
            parser1 = Mock()
            parser1.format_name = "owl"
            parser1.parse.return_value = ParseResult(
                success=True,
                data=original_ontology,
                errors=[],
                warnings=[],
                metadata={},
                parse_time=0.1,
            )

            # Second parser for recreated file
            parser2 = Mock()
            parser2.format_name = "owl"
            parser2.parse.return_value = ParseResult(
                success=True,
                data=new_ontology,
                errors=[],
                warnings=[],
                metadata={},
                parse_time=0.1,
            )

            mock_auto_detect.side_effect = [parser1, parser2]

            # First load - cache the original file
            result1 = ontology_manager.load_ontology(file_path)
            assert result1.success is True
            assert result1.ontology.id == original_ontology.id

            original_mtime = Path(file_path).stat().st_mtime

            # Wait a moment to ensure different mtime
            time.sleep(0.1)

            # Delete and recreate file with different content
            Path(file_path).unlink()

            new_owl_content = f"""<?xml version="1.0"?>
<rdf:RDF xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#"
         xmlns:owl="http://www.w3.org/2002/07/owl#"
         xmlns:rdfs="http://www.w3.org/2000/01/rdf-schema#">
    <owl:Ontology rdf:about="http://example.org/{new_ontology.id}">
        <rdfs:label>{new_ontology.name}</rdfs:label>
        <rdfs:comment>{new_ontology.description}</rdfs:comment>
    </owl:Ontology>
</rdf:RDF>"""
            Path(file_path).write_text(new_owl_content)

            new_mtime = Path(file_path).stat().st_mtime
            assert new_mtime > original_mtime

            # Second load - should detect file change and reload
            result2 = ontology_manager.load_ontology(file_path)
            assert result2.success is True
            assert result2.ontology.id == new_ontology.id
            assert result2.metadata.get("cache_hit", False) is False
            assert ontology_manager.load_stats["cache_misses"] == 2

    def test_permission_changes(self, ontology_manager, create_test_file):
        """Test cache behavior when file permissions change."""
        file_path, ontology = create_test_file()
        original_mode = Path(file_path).stat().st_mode

        with patch(
            "aim2_project.aim2_ontology.ontology_manager.auto_detect_parser"
        ) as mock_auto_detect:
            parser = Mock()
            parser.format_name = "owl"
            parser.parse.return_value = ParseResult(
                success=True,
                data=ontology,
                errors=[],
                warnings=[],
                metadata={},
                parse_time=0.1,
            )
            mock_auto_detect.return_value = parser

            # First load - cache the file
            result1 = ontology_manager.load_ontology(file_path)
            assert result1.success is True
            assert len(ontology_manager._cache) == 1

            # Change file permissions (remove read permission)
            try:
                os.chmod(file_path, stat.S_IWUSR)  # Write only

                # Second load - cache should still work (permission check happens during validation)
                result2 = ontology_manager.load_ontology(file_path)
                assert result2.success is True
                assert result2.metadata.get("cache_hit", True) is True

            finally:
                # Restore original permissions
                os.chmod(file_path, original_mode)

    def test_symbolic_link_handling(self, ontology_manager, create_test_file, temp_dir):
        """Test cache behavior with symbolic links."""
        file_path, ontology = create_test_file("original.owl")
        symlink_path = temp_dir / "symlink.owl"

        # Create symbolic link
        os.symlink(file_path, str(symlink_path))

        with patch(
            "aim2_project.aim2_ontology.ontology_manager.auto_detect_parser"
        ) as mock_auto_detect:
            parser = Mock()
            parser.format_name = "owl"
            parser.parse.return_value = ParseResult(
                success=True,
                data=ontology,
                errors=[],
                warnings=[],
                metadata={},
                parse_time=0.1,
            )
            mock_auto_detect.return_value = parser

            # Load via original path
            result1 = ontology_manager.load_ontology(file_path)
            assert result1.success is True

            # Load via symlink path - should be treated as separate cache entry
            result2 = ontology_manager.load_ontology(str(symlink_path))
            assert result2.success is True

            # Both should be cache misses since they're different paths
            assert ontology_manager.load_stats["cache_misses"] == 2
            assert len(ontology_manager._cache) == 2

    def test_cache_size_limit_zero(self):
        """Test cache behavior with size limit of zero."""
        manager = OntologyManager(enable_caching=True, cache_size_limit=0)

        # Should not cache anything
        with tempfile.TemporaryDirectory() as tmp_dir:
            test_file = Path(tmp_dir) / "test.owl"
            test_file.write_text("<?xml version='1.0'?><rdf:RDF></rdf:RDF>")

            with patch(
                "aim2_project.aim2_ontology.ontology_manager.auto_detect_parser"
            ) as mock_auto_detect:
                parser = Mock()
                parser.format_name = "owl"
                parser.parse.return_value = ParseResult(
                    success=True,
                    data=Ontology(
                        id="TEST:001",
                        name="Test",
                        version="1.0",
                        terms={},
                        relationships={},
                        namespaces=[],
                    ),
                    errors=[],
                    warnings=[],
                    metadata={},
                    parse_time=0.1,
                )
                mock_auto_detect.return_value = parser

                result1 = manager.load_ontology(str(test_file))
                result2 = manager.load_ontology(str(test_file))

                assert result1.success is True
                assert result2.success is True
                assert len(manager._cache) == 0
                # When cache size limit is 0, no cache statistics are tracked
                assert manager.load_stats["cache_misses"] == 0
                assert manager.load_stats["cache_hits"] == 0
                assert parser.parse.call_count == 2  # No caching, so called twice

    def test_cache_size_limit_one(self):
        """Test cache behavior with size limit of one."""
        manager = OntologyManager(enable_caching=True, cache_size_limit=1)

        with tempfile.TemporaryDirectory() as tmp_dir:
            # Create two test files
            file1 = Path(tmp_dir) / "test1.owl"
            file2 = Path(tmp_dir) / "test2.owl"
            file1.write_text("<?xml version='1.0'?><rdf:RDF></rdf:RDF>")
            file2.write_text("<?xml version='1.0'?><rdf:RDF></rdf:RDF>")

            ontology1 = Ontology(
                id="TEST:001",
                name="Test1",
                version="1.0",
                terms={},
                relationships={},
                namespaces=[],
            )
            ontology2 = Ontology(
                id="TEST:002",
                name="Test2",
                version="1.0",
                terms={},
                relationships={},
                namespaces=[],
            )

            with patch(
                "aim2_project.aim2_ontology.ontology_manager.auto_detect_parser"
            ) as mock_auto_detect:
                parser1 = Mock()
                parser1.format_name = "owl"
                parser1.parse.return_value = ParseResult(
                    success=True,
                    data=ontology1,
                    errors=[],
                    warnings=[],
                    metadata={},
                    parse_time=0.1,
                )

                parser2 = Mock()
                parser2.format_name = "owl"
                parser2.parse.return_value = ParseResult(
                    success=True,
                    data=ontology2,
                    errors=[],
                    warnings=[],
                    metadata={},
                    parse_time=0.1,
                )

                # Create a function that returns the appropriate parser based on file path
                def get_parser(*args, **kwargs):
                    file_path = (
                        str(args[0]) if args else str(kwargs.get("file_path", ""))
                    )
                    if "test1.owl" in file_path:
                        return parser1
                    else:
                        return parser2

                mock_auto_detect.side_effect = get_parser

                # Load first file
                result1 = manager.load_ontology(str(file1))
                assert result1.success is True
                assert len(manager._cache) == 1
                assert str(file1) in manager._cache

                # Load second file - should evict first
                result2 = manager.load_ontology(str(file2))
                assert result2.success is True
                assert len(manager._cache) == 1
                assert str(file1) not in manager._cache
                assert str(file2) in manager._cache

    def test_lru_eviction_tie_breaking(self, ontology_manager):
        """Test LRU eviction behavior with tie-breaking scenarios."""
        # Set small cache limit for testing
        ontology_manager.cache_size_limit = 3

        with tempfile.TemporaryDirectory() as tmp_dir:
            files = []
            ontologies = []

            # Create test files and ontologies
            for i in range(4):
                file_path = Path(tmp_dir) / f"test{i}.owl"
                file_path.write_text("<?xml version='1.0'?><rdf:RDF></rdf:RDF>")
                files.append(str(file_path))
                ontologies.append(
                    Ontology(
                        id=f"TEST:{i:03d}",
                        name=f"Test{i}",
                        version="1.0",
                        terms={},
                        relationships={},
                        namespaces=[],
                    )
                )

            with patch(
                "aim2_project.aim2_ontology.ontology_manager.auto_detect_parser"
            ) as mock_auto_detect:
                parsers = []
                for i, ont in enumerate(ontologies):
                    parser = Mock()
                    parser.format_name = "owl"
                    parser.parse.return_value = ParseResult(
                        success=True,
                        data=ont,
                        errors=[],
                        warnings=[],
                        metadata={},
                        parse_time=0.1,
                    )
                    parsers.append(parser)

                # Create a function that returns the appropriate parser based on file path
                def get_parser(*args, **kwargs):
                    file_path = (
                        str(args[0]) if args else str(kwargs.get("file_path", ""))
                    )
                    for i, expected_file in enumerate(files):
                        if expected_file in file_path:
                            return parsers[i]
                    return parsers[0]  # Default fallback

                mock_auto_detect.side_effect = get_parser

                # Load first 3 files (fill cache)
                for i in range(3):
                    result = ontology_manager.load_ontology(files[i])
                    assert result.success is True
                    time.sleep(0.01)  # Ensure different timestamps

                assert len(ontology_manager._cache) == 3

                # Access files in specific order to create known LRU state
                # Access file1, then file2 (file0 becomes LRU)
                ontology_manager.load_ontology(files[1])  # Cache hit
                time.sleep(0.01)
                ontology_manager.load_ontology(files[2])  # Cache hit

                # Load 4th file - should evict file0 (least recently used)
                result = ontology_manager.load_ontology(files[3])
                assert result.success is True
                assert len(ontology_manager._cache) == 3
                assert files[0] not in ontology_manager._cache
                assert files[3] in ontology_manager._cache

    def test_concurrent_cache_access(self, ontology_manager, create_test_file):
        """Test thread safety of cache operations."""
        file_path, ontology = create_test_file()

        with patch(
            "aim2_project.aim2_ontology.ontology_manager.auto_detect_parser"
        ) as mock_auto_detect:
            parser = Mock()
            parser.format_name = "owl"
            parser.parse.return_value = ParseResult(
                success=True,
                data=ontology,
                errors=[],
                warnings=[],
                metadata={},
                parse_time=0.1,
            )

            # Create a function that returns the parser instead of using return_value
            def get_parser(*args, **kwargs):
                return parser

            mock_auto_detect.side_effect = get_parser

            results = []
            errors = []

            def load_ontology_worker():
                try:
                    result = ontology_manager.load_ontology(file_path)
                    results.append(result)
                except Exception as e:
                    errors.append(e)

            # Create multiple threads to access cache concurrently
            threads = []
            for _ in range(10):
                thread = threading.Thread(target=load_ontology_worker)
                threads.append(thread)

            # Start all threads
            for thread in threads:
                thread.start()

            # Wait for all threads to complete
            for thread in threads:
                thread.join()

            # Verify no errors occurred
            assert len(errors) == 0, f"Concurrent access errors: {errors}"
            assert len(results) == 10
            assert all(r.success for r in results)

            # All should return the same ontology
            assert all(r.ontology.id == ontology.id for r in results)

            # Due to race conditions in concurrent access, parser may be called multiple times
            # but should be called at least once and at most once per thread
            assert parser.parse.call_count >= 1
            assert parser.parse.call_count <= 10

    def test_concurrent_cache_eviction(self, ontology_manager):
        """Test concurrent cache eviction scenarios."""
        # Set very small cache limit
        ontology_manager.cache_size_limit = 2

        with tempfile.TemporaryDirectory() as tmp_dir:
            files = []
            ontologies = []

            # Create multiple test files
            for i in range(5):
                file_path = Path(tmp_dir) / f"concurrent{i}.owl"
                file_path.write_text("<?xml version='1.0'?><rdf:RDF></rdf:RDF>")
                files.append(str(file_path))
                ontologies.append(
                    Ontology(
                        id=f"CONCURRENT:{i:03d}",
                        name=f"Concurrent{i}",
                        version="1.0",
                        terms={},
                        relationships={},
                        namespaces=[],
                    )
                )

            with patch(
                "aim2_project.aim2_ontology.ontology_manager.auto_detect_parser"
            ) as mock_auto_detect:
                parsers = []
                for ont in ontologies:
                    parser = Mock()
                    parser.format_name = "owl"
                    parser.parse.return_value = ParseResult(
                        success=True,
                        data=ont,
                        errors=[],
                        warnings=[],
                        metadata={},
                        parse_time=0.1,
                    )
                    parsers.append(parser)

                # Create a function that returns the appropriate parser based on file path
                def get_parser(*args, **kwargs):
                    file_path = (
                        str(args[0]) if args else str(kwargs.get("file_path", ""))
                    )
                    for i, expected_file in enumerate(files):
                        if expected_file in file_path:
                            return parsers[i]
                    return parsers[0]  # Default fallback

                mock_auto_detect.side_effect = get_parser

                results = []
                errors = []

                def concurrent_load_worker(file_index):
                    try:
                        result = ontology_manager.load_ontology(files[file_index])
                        results.append((file_index, result))
                    except Exception as e:
                        errors.append((file_index, e))

                # Load files concurrently to trigger eviction
                with ThreadPoolExecutor(max_workers=5) as executor:
                    futures = [
                        executor.submit(concurrent_load_worker, i) for i in range(5)
                    ]

                    for future in as_completed(futures):
                        future.result()  # Wait for completion

                # Verify no errors and cache limit respected
                assert len(errors) == 0, f"Concurrent eviction errors: {errors}"
                assert len(results) == 5
                assert all(r[1].success for r in results)
                assert len(ontology_manager._cache) <= ontology_manager.cache_size_limit

    def test_cache_corruption_recovery(self, ontology_manager, create_test_file):
        """Test recovery from cache corruption scenarios."""
        file_path, ontology = create_test_file()

        with patch(
            "aim2_project.aim2_ontology.ontology_manager.auto_detect_parser"
        ) as mock_auto_detect:
            parser = Mock()
            parser.format_name = "owl"
            parser.parse.return_value = ParseResult(
                success=True,
                data=ontology,
                errors=[],
                warnings=[],
                metadata={},
                parse_time=0.1,
            )
            mock_auto_detect.return_value = parser

            # First load - populate cache
            result1 = ontology_manager.load_ontology(file_path)
            assert result1.success is True
            assert len(ontology_manager._cache) == 1

            # Simulate cache corruption by modifying cache entry
            cache_entry = ontology_manager._cache[file_path]
            cache_entry.ontology = None  # Corrupt the cached ontology

            # Second load - should handle corruption gracefully
            result2 = ontology_manager.load_ontology(file_path)
            assert result2.success is True
            # Should still work even with corrupted cache

    def test_filesystem_errors_during_cache_operations(
        self, ontology_manager, create_test_file
    ):
        """Test handling of filesystem errors during cache operations."""
        file_path, ontology = create_test_file()

        with patch(
            "aim2_project.aim2_ontology.ontology_manager.auto_detect_parser"
        ) as mock_auto_detect:
            parser = Mock()
            parser.format_name = "owl"
            parser.parse.return_value = ParseResult(
                success=True,
                data=ontology,
                errors=[],
                warnings=[],
                metadata={},
                parse_time=0.1,
            )
            mock_auto_detect.return_value = parser

            # First load - should work normally
            result1 = ontology_manager.load_ontology(file_path)
            assert result1.success is True

            # Mock filesystem error during cache validation
            with patch.object(Path, "stat", side_effect=OSError("Filesystem error")):
                # Second load - should handle filesystem error gracefully
                result2 = ontology_manager.load_ontology(file_path)
                assert result2.success is True
                # Should fall back to cache even with stat error

    def test_io_error_handling_during_cache_updates(
        self, ontology_manager, create_test_file
    ):
        """Test I/O error handling during cache update operations."""
        file_path, ontology = create_test_file()

        with patch(
            "aim2_project.aim2_ontology.ontology_manager.auto_detect_parser"
        ) as mock_auto_detect:
            parser = Mock()
            parser.format_name = "owl"
            parser.parse.return_value = ParseResult(
                success=True,
                data=ontology,
                errors=[],
                warnings=[],
                metadata={},
                parse_time=0.1,
            )
            mock_auto_detect.return_value = parser

            # Mock I/O error during mtime check in cache update
            with patch.object(Path, "stat", side_effect=OSError("I/O error")):
                result = ontology_manager.load_ontology(file_path)
                assert result.success is True
                # Should still cache successfully despite mtime error
                assert len(ontology_manager._cache) == 1
                cache_entry = ontology_manager._cache[file_path]
                assert cache_entry.file_mtime is None  # Should be None due to error

    def test_memory_pressure_cache_behavior(self, ontology_manager):
        """Test cache behavior under memory pressure scenarios."""
        # Create large ontologies to simulate memory pressure
        large_ontologies = []
        file_paths = []

        with tempfile.TemporaryDirectory() as tmp_dir:
            for i in range(3):
                # Create ontology with many terms to use more memory
                terms = {}
                for j in range(100):
                    term_id = f"MEMORY:{i*100+j:06d}"
                    terms[term_id] = Term(
                        id=term_id,
                        name=f"Memory Test Term {j}" * 10,  # Long names
                        definition=f"Long definition for memory test term {j}"
                        * 20,  # Long definitions
                        synonyms=[
                            f"synonym{j}_{k}" * 5 for k in range(10)
                        ],  # Many long synonyms
                        namespace=f"memorytest{i}",
                    )

                ontology = Ontology(
                    id=f"MEMORYTEST:{i:06d}",
                    name=f"Memory Test Ontology {i}",
                    version="1.0",
                    description="Large ontology for memory testing",
                    terms=terms,
                    relationships={},
                    namespaces=[f"memorytest{i}"],
                )
                large_ontologies.append(ontology)

                file_path = Path(tmp_dir) / f"memory_test_{i}.owl"
                file_path.write_text("<?xml version='1.0'?><rdf:RDF></rdf:RDF>")
                file_paths.append(str(file_path))

            with patch(
                "aim2_project.aim2_ontology.ontology_manager.auto_detect_parser"
            ) as mock_auto_detect:
                parsers = []
                for ont in large_ontologies:
                    parser = Mock()
                    parser.format_name = "owl"
                    parser.parse.return_value = ParseResult(
                        success=True,
                        data=ont,
                        errors=[],
                        warnings=[],
                        metadata={},
                        parse_time=0.1,
                    )
                    parsers.append(parser)

                mock_auto_detect.side_effect = parsers

                # Load all large ontologies
                results = []
                for file_path in file_paths:
                    result = ontology_manager.load_ontology(file_path)
                    results.append(result)
                    gc.collect()  # Force garbage collection

                # Verify all loaded successfully despite memory pressure
                assert all(r.success for r in results)
                assert len(ontology_manager._cache) == len(file_paths)

                # Verify cache entries are valid
                for file_path in file_paths:
                    cache_entry = ontology_manager._cache[file_path]
                    assert cache_entry.ontology is not None
                    assert len(cache_entry.ontology.terms) == 100

    def test_cache_clear_functionality(self, ontology_manager, create_test_file):
        """Test cache clearing functionality."""
        file_paths = []
        ontologies = []

        # Create multiple test files
        for i in range(3):
            file_path, ontology = create_test_file(f"clear_test_{i}.owl")
            file_paths.append(file_path)
            ontologies.append(ontology)

        with patch(
            "aim2_project.aim2_ontology.ontology_manager.auto_detect_parser"
        ) as mock_auto_detect:
            parsers = []
            for ont in ontologies:
                parser = Mock()
                parser.format_name = "owl"
                parser.parse.return_value = ParseResult(
                    success=True,
                    data=ont,
                    errors=[],
                    warnings=[],
                    metadata={},
                    parse_time=0.1,
                )
                parsers.append(parser)

            # Create a function that returns the appropriate parser based on file path
            def get_parser(*args, **kwargs):
                file_path = str(args[0]) if args else str(kwargs.get("file_path", ""))
                for i in range(len(ontologies)):
                    if f"clear_test_{i}.owl" in file_path:
                        return parsers[i]
                return parsers[0]  # Default fallback

            mock_auto_detect.side_effect = get_parser

            # Load all files to populate cache
            for file_path in file_paths:
                result = ontology_manager.load_ontology(file_path)
                assert result.success is True

            assert len(ontology_manager._cache) == 3

            # Clear cache
            ontology_manager.clear_cache()
            assert len(ontology_manager._cache) == 0

            # Subsequent loads should be cache misses
            initial_misses = ontology_manager.load_stats["cache_misses"]
            result = ontology_manager.load_ontology(file_paths[0])
            assert result.success is True
            assert ontology_manager.load_stats["cache_misses"] == initial_misses + 1

    def test_cache_statistics_accuracy(self, ontology_manager, create_test_file):
        """Test accuracy of cache-related statistics."""
        file_path, ontology = create_test_file()

        with patch(
            "aim2_project.aim2_ontology.ontology_manager.auto_detect_parser"
        ) as mock_auto_detect:
            parser = Mock()
            parser.format_name = "owl"
            parser.parse.return_value = ParseResult(
                success=True,
                data=ontology,
                errors=[],
                warnings=[],
                metadata={},
                parse_time=0.1,
            )
            mock_auto_detect.return_value = parser

            # Initial state
            stats = ontology_manager.get_statistics()
            assert stats["cache_hits"] == 0
            assert stats["cache_misses"] == 0
            assert stats["cache_size"] == 0
            assert stats["cache_limit"] == 10
            assert stats["cache_enabled"] is True

            # First load - cache miss
            ontology_manager.load_ontology(file_path)
            stats = ontology_manager.get_statistics()
            assert stats["cache_hits"] == 0
            assert stats["cache_misses"] == 1
            assert stats["cache_size"] == 1

            # Second load - cache hit
            ontology_manager.load_ontology(file_path)
            stats = ontology_manager.get_statistics()
            assert stats["cache_hits"] == 1
            assert stats["cache_misses"] == 1
            assert stats["cache_size"] == 1

            # Clear cache
            ontology_manager.clear_cache()
            stats = ontology_manager.get_statistics()
            assert stats["cache_size"] == 0
            # Hit/miss counts should remain (cumulative)
            assert stats["cache_hits"] == 1
            assert stats["cache_misses"] == 1

    def test_cache_entry_access_tracking(self, ontology_manager, create_test_file):
        """Test that cache entries properly track access patterns."""
        file_path, ontology = create_test_file()

        with patch(
            "aim2_project.aim2_ontology.ontology_manager.auto_detect_parser"
        ) as mock_auto_detect:
            parser = Mock()
            parser.format_name = "owl"
            parser.parse.return_value = ParseResult(
                success=True,
                data=ontology,
                errors=[],
                warnings=[],
                metadata={},
                parse_time=0.1,
            )
            mock_auto_detect.return_value = parser

            # First load
            start_time = time.time()
            result1 = ontology_manager.load_ontology(file_path)
            assert result1.success is True

            cache_entry = ontology_manager._cache[file_path]
            assert cache_entry.access_count == 1
            assert cache_entry.last_accessed >= start_time
            first_access_time = cache_entry.last_accessed

            # Wait a moment
            time.sleep(0.1)

            # Second load
            result2 = ontology_manager.load_ontology(file_path)
            assert result2.success is True
            assert result2.metadata.get("cache_hit", True) is True
            assert result2.metadata.get("access_count") == 2

            cache_entry = ontology_manager._cache[file_path]
            assert cache_entry.access_count == 2
            assert cache_entry.last_accessed > first_access_time

    def test_cache_with_failed_loads(self, ontology_manager, create_test_file):
        """Test that failed loads don't corrupt cache state."""
        valid_file, valid_ontology = create_test_file("valid.owl")

        with tempfile.TemporaryDirectory() as tmp_dir:
            invalid_file = Path(tmp_dir) / "invalid.owl"
            invalid_file.write_text("invalid content")

            with patch(
                "aim2_project.aim2_ontology.ontology_manager.auto_detect_parser"
            ) as mock_auto_detect:

                def mock_parser_behavior(*args, **kwargs):
                    # Get file_path from args or kwargs
                    file_path = (
                        str(args[0]) if args else str(kwargs.get("file_path", ""))
                    )
                    # Check for invalid.owl first since it contains "valid.owl" as substring
                    if file_path.endswith("invalid.owl"):
                        parser = Mock()
                        parser.format_name = "owl"
                        parser.parse.return_value = ParseResult(
                            success=False,
                            data=None,
                            errors=["Invalid content"],
                            warnings=[],
                            metadata={},
                            parse_time=0.1,
                        )
                        return parser
                    elif file_path.endswith("valid.owl"):
                        parser = Mock()
                        parser.format_name = "owl"
                        parser.parse.return_value = ParseResult(
                            success=True,
                            data=valid_ontology,
                            errors=[],
                            warnings=[],
                            metadata={},
                            parse_time=0.1,
                        )
                        return parser
                    else:
                        return None

                mock_auto_detect.side_effect = mock_parser_behavior

                # Load valid file - should cache
                result1 = ontology_manager.load_ontology(valid_file)
                assert result1.success is True
                assert len(ontology_manager._cache) == 1

                # Load invalid file - should fail without affecting cache
                result2 = ontology_manager.load_ontology(str(invalid_file))
                assert result2.success is False
                assert len(ontology_manager._cache) == 1  # Cache unchanged

                # Load valid file again - should hit cache
                result3 = ontology_manager.load_ontology(valid_file)
                assert result3.success is True
                assert result3.metadata.get("cache_hit", True) is True

    def test_cache_state_after_ontology_removal(
        self, ontology_manager, create_test_file
    ):
        """Test cache state consistency after ontology removal."""
        file_path, ontology = create_test_file()

        with patch(
            "aim2_project.aim2_ontology.ontology_manager.auto_detect_parser"
        ) as mock_auto_detect:
            parser = Mock()
            parser.format_name = "owl"
            parser.parse.return_value = ParseResult(
                success=True,
                data=ontology,
                errors=[],
                warnings=[],
                metadata={},
                parse_time=0.1,
            )
            mock_auto_detect.return_value = parser

            # Load and cache ontology
            result = ontology_manager.load_ontology(file_path)
            assert result.success is True
            assert len(ontology_manager._cache) == 1
            assert len(ontology_manager.ontologies) == 1

            # Remove ontology - should also clean up cache
            removed = ontology_manager.remove_ontology(ontology.id)
            assert removed is True
            assert len(ontology_manager.ontologies) == 0
            assert len(ontology_manager._cache) == 0  # Cache should be cleaned up


if __name__ == "__main__":
    pytest.main([__file__])
