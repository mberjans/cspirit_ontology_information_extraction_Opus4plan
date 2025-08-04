#!/usr/bin/env python3
"""
Comprehensive unit tests for OntologyManager statistics functionality.

This module contains extensive unit tests specifically focused on statistics
functionality within the OntologyManager class. The tests ensure that all
statistical calculations, tracking mechanisms, and edge cases are properly
handled across various operational scenarios.

Test Coverage:
1. Initial Statistics State (Critical):
   - Verify all statistics start at expected zero/default values
   - Test statistics structure and completeness
   - Verify statistics types and formats

2. Load Statistics Tracking (Critical):
   - Total loads, successful loads, failed loads counting
   - Cache hits and misses tracking accuracy
   - Format loading statistics accuracy
   - Statistics consistency across multiple operations

3. Cache Statistics Tracking (Critical):
   - Cache size, limit, and enabled status reporting
   - Cache statistics with different cache configurations
   - Statistics accuracy with cache disabled/enabled
   - Cache statistics during eviction scenarios

4. Ontology Statistics Calculations (Critical):
   - Loaded ontologies count accuracy
   - Total terms calculation across all ontologies
   - Total relationships calculation across all ontologies
   - Statistics accuracy with empty ontologies

5. Statistics Edge Cases (Important):
   - Zero cache limit scenarios
   - Empty ontology handling
   - Failed load impact on statistics
   - Statistics consistency after cache clearing

6. Statistics Integration (Important):
   - Statistics persistence across operations
   - Multi-format loading statistics
   - Complex operational scenarios
   - Statistics accuracy under concurrent access

Key Design Principles:
- All tests are deterministic and reliable
- Comprehensive fixture setup for repeatable scenarios
- Proper mocking of external dependencies
- Statistics validation at every operational step
- Edge case coverage for all statistical calculations
"""

import tempfile
import time
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


class TestOntologyManagerStatistics:
    """Comprehensive test suite for OntologyManager statistics functionality."""

    @pytest.fixture
    def ontology_manager(self):
        """Create an OntologyManager instance with caching enabled."""
        return OntologyManager(enable_caching=True, cache_size_limit=10)

    @pytest.fixture
    def ontology_manager_no_cache(self):
        """Create an OntologyManager instance with caching disabled."""
        return OntologyManager(enable_caching=False, cache_size_limit=0)

    @pytest.fixture
    def ontology_manager_zero_cache(self):
        """Create an OntologyManager instance with zero cache limit."""
        return OntologyManager(enable_caching=True, cache_size_limit=0)

    @pytest.fixture
    def sample_ontology_small(self):
        """Create a small sample ontology for testing."""
        terms = {
            "TEST:001": Term(
                id="TEST:001",
                name="Test Term 1",
                definition="A test term",
                synonyms=["test1"],
                namespace="test",
            ),
            "TEST:002": Term(
                id="TEST:002",
                name="Test Term 2",
                definition="Another test term",
                synonyms=["test2"],
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
            id="STATSTEST:001",
            name="Statistics Test Ontology Small",
            version="1.0",
            description="Small test ontology for statistics",
            terms=terms,
            relationships=relationships,
            namespaces=["test"],
        )

    @pytest.fixture
    def sample_ontology_large(self):
        """Create a larger sample ontology for testing."""
        terms = {}
        relationships = {}

        # Create 10 terms
        for i in range(10):
            term_id = f"LARGE:{'%03d' % (i + 1)}"
            terms[term_id] = Term(
                id=term_id,
                name=f"Large Test Term {i + 1}",
                definition=f"Large test term definition {i + 1}",
                synonyms=[f"large{i + 1}", f"big{i + 1}"],
                namespace="large_test",
            )

        # Create 5 relationships
        for i in range(5):
            rel_id = f"LARGE:{'%03d' % (i + 101)}"  # Use LARGE:101, LARGE:102, etc. for relationships
            subject_idx = i
            object_idx = (i + 1) % 10
            relationships[rel_id] = Relationship(
                id=rel_id,
                subject=f"LARGE:{'%03d' % (subject_idx + 1)}",
                predicate="regulates",
                object=f"LARGE:{'%03d' % (object_idx + 1)}",
                confidence=0.9,
            )

        return Ontology(
            id="STATSTEST:002",
            name="Statistics Test Ontology Large",
            version="1.0",
            description="Large test ontology for statistics",
            terms=terms,
            relationships=relationships,
            namespaces=["large_test"],
        )

    @pytest.fixture
    def empty_ontology(self):
        """Create an empty ontology for testing."""
        return Ontology(
            id="STATSTEST:003",
            name="Empty Test Ontology",
            version="1.0",
            description="Empty test ontology for statistics",
            terms={},
            relationships={},
            namespaces=["empty"],
        )

    @pytest.fixture
    def create_test_file(self):
        """Create a test file factory for statistics testing."""

        def _create_file(filename="test_stats.owl", ontology=None):
            tmp_dir = Path(tempfile.mkdtemp())
            test_file = tmp_dir / filename
            # Create simple OWL content
            owl_content = """<?xml version="1.0"?>
<rdf:RDF xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#"
         xmlns:owl="http://www.w3.org/2002/07/owl#"
         xmlns:rdfs="http://www.w3.org/2000/01/rdf-schema#">
    <owl:Ontology rdf:about="http://example.org/test">
        <rdfs:label>Test Ontology</rdfs:label>
    </owl:Ontology>
</rdf:RDF>"""
            test_file.write_text(owl_content)
            return str(test_file), ontology, tmp_dir

        return _create_file

    def test_initial_statistics_state(self, ontology_manager):
        """Test that all statistics start at expected zero/default values."""
        stats = ontology_manager.get_statistics()

        # Load statistics should all be zero
        assert stats["total_loads"] == 0
        assert stats["successful_loads"] == 0
        assert stats["failed_loads"] == 0
        assert stats["cache_hits"] == 0
        assert stats["cache_misses"] == 0

        # Format statistics should be empty
        assert stats["formats_loaded"] == {}
        assert isinstance(stats["formats_loaded"], dict)

        # Cache statistics should reflect initial state
        assert stats["cache_size"] == 0
        assert stats["cache_limit"] == 10
        assert stats["cache_enabled"] is True

        # Ontology statistics should be zero
        assert stats["loaded_ontologies"] == 0
        assert stats["total_terms"] == 0
        assert stats["total_relationships"] == 0

        # Verify all expected keys are present (including new multi-source statistics)
        required_keys = {
            "total_loads",
            "successful_loads",
            "failed_loads",
            "cache_hits",
            "cache_misses",
            "formats_loaded",
            "cache_size",
            "cache_limit",
            "cache_enabled",
            "loaded_ontologies",
            "total_terms",
            "total_relationships",
        }
        
        # New multi-source statistics keys
        multisource_keys = {
            "sources_loaded",
            "sources_attempted",
            "source_success_rate",
            "sources_by_format",
            "source_coverage",
            "overlap_analysis",
            "performance",
        }
        
        all_expected_keys = required_keys | multisource_keys
        actual_keys = set(stats.keys())
        
        # Check that all required keys are present
        assert required_keys.issubset(actual_keys), f"Missing required keys: {required_keys - actual_keys}"
        
        # Check that all multisource keys are present
        assert multisource_keys.issubset(actual_keys), f"Missing multisource keys: {multisource_keys - actual_keys}"
        
        # Verify no unexpected keys (allow for future extensions)
        assert actual_keys <= all_expected_keys, f"Unexpected keys: {actual_keys - all_expected_keys}"

    def test_statistics_disabled_caching(self, ontology_manager_no_cache):
        """Test statistics with caching disabled."""
        stats = ontology_manager_no_cache.get_statistics()

        # Cache-related statistics should reflect disabled state
        assert stats["cache_size"] == 0
        assert stats["cache_limit"] == 0
        assert stats["cache_enabled"] is False
        assert stats["cache_hits"] == 0
        assert stats["cache_misses"] == 0

        # Other statistics should still be tracked normally
        assert stats["total_loads"] == 0
        assert stats["successful_loads"] == 0
        assert stats["failed_loads"] == 0

    def test_statistics_zero_cache_limit(self, ontology_manager_zero_cache):
        """Test statistics with zero cache limit."""
        stats = ontology_manager_zero_cache.get_statistics()

        # Cache should be enabled but with zero limit
        assert stats["cache_size"] == 0
        assert stats["cache_limit"] == 0
        assert stats["cache_enabled"] is True
        assert stats["cache_hits"] == 0
        assert stats["cache_misses"] == 0

    def test_successful_load_statistics(
        self, ontology_manager, create_test_file, sample_ontology_small
    ):
        """Test statistics tracking for successful loads."""
        file_path, _, tmp_dir = create_test_file("success_test.owl")

        try:
            with patch(
                "aim2_project.aim2_ontology.ontology_manager.auto_detect_parser"
            ) as mock_auto_detect:
                parser = Mock()
                parser.format_name = "owl"
                parser.parse.return_value = ParseResult(
                    success=True,
                    data=sample_ontology_small,
                    errors=[],
                    warnings=[],
                    metadata={},
                    parse_time=0.1,
                )
                mock_auto_detect.return_value = parser

                # First load - should be cache miss
                result = ontology_manager.load_ontology(file_path)
                assert result.success is True

                stats = ontology_manager.get_statistics()
                assert stats["total_loads"] == 1
                assert stats["successful_loads"] == 1
                assert stats["failed_loads"] == 0
                assert stats["cache_hits"] == 0
                assert stats["cache_misses"] == 1
                assert stats["formats_loaded"]["owl"] == 1
                assert stats["cache_size"] == 1
                assert stats["loaded_ontologies"] == 1
                assert stats["total_terms"] == 2
                assert stats["total_relationships"] == 1

                # Second load - should be cache hit
                result = ontology_manager.load_ontology(file_path)
                assert result.success is True

                stats = ontology_manager.get_statistics()
                assert stats["total_loads"] == 2
                assert stats["successful_loads"] == 1  # Only one actual successful load (cache hit doesn't count)
                assert stats["failed_loads"] == 0
                assert stats["cache_hits"] == 1
                assert stats["cache_misses"] == 1
                assert stats["formats_loaded"]["owl"] == 1  # Format count shouldn't change on cache hit
                assert stats["cache_size"] == 1
                assert stats["loaded_ontologies"] == 1  # Same ontology, so count stays same
                assert stats["total_terms"] == 2
                assert stats["total_relationships"] == 1

        finally:
            # Cleanup
            import shutil

            shutil.rmtree(tmp_dir, ignore_errors=True)

    def test_failed_load_statistics(self, ontology_manager, create_test_file):
        """Test statistics tracking for failed loads."""
        file_path, _, tmp_dir = create_test_file("failure_test.owl")

        try:
            with patch(
                "aim2_project.aim2_ontology.ontology_manager.auto_detect_parser"
            ) as mock_auto_detect:
                # Test no parser found
                mock_auto_detect.return_value = None

                result = ontology_manager.load_ontology(file_path)
                assert result.success is False

                stats = ontology_manager.get_statistics()
                assert stats["total_loads"] == 1
                assert stats["successful_loads"] == 0
                assert stats["failed_loads"] == 1
                assert stats["cache_hits"] == 0
                assert stats["cache_misses"] == 1  # Cache miss is counted even when parser detection fails
                assert stats["formats_loaded"] == {}
                assert stats["cache_size"] == 0
                assert stats["loaded_ontologies"] == 0
                assert stats["total_terms"] == 0
                assert stats["total_relationships"] == 0

                # Test parser failure
                parser = Mock()
                parser.format_name = "owl"
                parser.parse.return_value = ParseResult(
                    success=False,
                    data=None,
                    errors=["Parse error"],
                    warnings=[],
                    metadata={},
                    parse_time=0.1,
                )
                mock_auto_detect.return_value = parser

                result = ontology_manager.load_ontology(file_path)
                assert result.success is False

                stats = ontology_manager.get_statistics()
                assert stats["total_loads"] == 2
                assert stats["successful_loads"] == 0
                assert stats["failed_loads"] == 2
                assert stats["cache_hits"] == 0
                assert stats["cache_misses"] == 2  # Both calls count as cache misses
                assert stats["formats_loaded"] == {}  # No successful format loads
                assert stats["cache_size"] == 0
                assert stats["loaded_ontologies"] == 0
                assert stats["total_terms"] == 0
                assert stats["total_relationships"] == 0

        finally:
            # Cleanup
            import shutil

            shutil.rmtree(tmp_dir, ignore_errors=True)

    def test_multiple_format_statistics(
        self,
        ontology_manager,
        create_test_file,
        sample_ontology_small,
        sample_ontology_large,
    ):
        """Test statistics tracking for multiple ontology formats."""
        owl_file, _, tmp_dir1 = create_test_file("test.owl")
        rdf_file, _, tmp_dir2 = create_test_file("test.rdf")

        try:
            with patch(
                "aim2_project.aim2_ontology.ontology_manager.auto_detect_parser"
            ) as mock_auto_detect:

                def create_parser(format_name, ontology):
                    parser = Mock()
                    parser.format_name = format_name
                    parser.parse.return_value = ParseResult(
                        success=True,
                        data=ontology,
                        errors=[],
                        warnings=[],
                        metadata={},
                        parse_time=0.1,
                    )
                    return parser

                # Mock parser selection based on file extension
                def get_parser(*args, **kwargs):
                    file_path = str(args[0]) if args else str(kwargs.get("file_path", ""))
                    if file_path.endswith(".owl"):
                        return create_parser("owl", sample_ontology_small)
                    elif file_path.endswith(".rdf"):
                        return create_parser("rdf", sample_ontology_large)
                    return None

                mock_auto_detect.side_effect = get_parser

                # Load OWL file
                result1 = ontology_manager.load_ontology(owl_file)
                assert result1.success is True

                stats = ontology_manager.get_statistics()
                assert stats["total_loads"] == 1
                assert stats["successful_loads"] == 1
                assert stats["formats_loaded"]["owl"] == 1
                assert "rdf" not in stats["formats_loaded"]
                assert stats["loaded_ontologies"] == 1
                assert stats["total_terms"] == 2  # small ontology
                assert stats["total_relationships"] == 1

                # Load RDF file
                result2 = ontology_manager.load_ontology(rdf_file)
                assert result2.success is True

                stats = ontology_manager.get_statistics()
                assert stats["total_loads"] == 2
                assert stats["successful_loads"] == 2
                assert stats["formats_loaded"]["owl"] == 1
                assert stats["formats_loaded"]["rdf"] == 1
                assert stats["loaded_ontologies"] == 2
                assert stats["total_terms"] == 12  # 2 + 10
                assert stats["total_relationships"] == 6  # 1 + 5

                # Load OWL file again
                result3 = ontology_manager.load_ontology(owl_file)
                assert result3.success is True

                stats = ontology_manager.get_statistics()
                assert stats["total_loads"] == 3
                assert stats["successful_loads"] == 2  # Two successful parses, third is cache hit
                assert stats["cache_hits"] == 1
                assert stats["formats_loaded"]["owl"] == 1  # No change on cache hit
                assert stats["formats_loaded"]["rdf"] == 1

        finally:
            # Cleanup
            import shutil

            shutil.rmtree(tmp_dir1, ignore_errors=True)
            shutil.rmtree(tmp_dir2, ignore_errors=True)

    def test_empty_ontology_statistics(
        self, ontology_manager, create_test_file, empty_ontology
    ):
        """Test statistics with empty ontologies."""
        file_path, _, tmp_dir = create_test_file("empty.owl")

        try:
            with patch(
                "aim2_project.aim2_ontology.ontology_manager.auto_detect_parser"
            ) as mock_auto_detect:
                parser = Mock()
                parser.format_name = "owl"
                parser.parse.return_value = ParseResult(
                    success=True,
                    data=empty_ontology,
                    errors=[],
                    warnings=[],
                    metadata={},
                    parse_time=0.1,
                )
                mock_auto_detect.return_value = parser

                result = ontology_manager.load_ontology(file_path)
                assert result.success is True

                stats = ontology_manager.get_statistics()
                assert stats["total_loads"] == 1
                assert stats["successful_loads"] == 1
                assert stats["failed_loads"] == 0
                assert stats["loaded_ontologies"] == 1
                assert stats["total_terms"] == 0  # Empty ontology
                assert stats["total_relationships"] == 0  # Empty ontology
                assert stats["formats_loaded"]["owl"] == 1

        finally:
            # Cleanup
            import shutil

            shutil.rmtree(tmp_dir, ignore_errors=True)

    def test_cache_eviction_statistics(
        self, sample_ontology_small, sample_ontology_large, empty_ontology
    ):
        """Test statistics during cache eviction scenarios."""
        # Create manager with small cache limit
        manager = OntologyManager(enable_caching=True, cache_size_limit=2)

        with tempfile.TemporaryDirectory() as tmp_dir:
            # Create test files
            files = []
            ontologies = [sample_ontology_small, sample_ontology_large, empty_ontology]
            
            for i, ont in enumerate(ontologies):
                file_path = Path(tmp_dir) / f"eviction_test_{i}.owl"
                file_path.write_text("<?xml version='1.0'?><rdf:RDF></rdf:RDF>")
                files.append(str(file_path))

            with patch(
                "aim2_project.aim2_ontology.ontology_manager.auto_detect_parser"
            ) as mock_auto_detect:

                def get_parser(*args, **kwargs):
                    file_path = str(args[0]) if args else str(kwargs.get("file_path", ""))
                    for i, expected_file in enumerate(files):
                        if expected_file in file_path:
                            parser = Mock()
                            parser.format_name = "owl"
                            parser.parse.return_value = ParseResult(
                                success=True,
                                data=ontologies[i],
                                errors=[],
                                warnings=[],
                                metadata={},
                                parse_time=0.1,
                            )
                            return parser
                    return None

                mock_auto_detect.side_effect = get_parser

                # Load first two files (fill cache)
                result1 = manager.load_ontology(files[0])
                result2 = manager.load_ontology(files[1])
                assert result1.success is True
                assert result2.success is True

                stats = manager.get_statistics()
                assert stats["cache_size"] == 2
                assert stats["cache_misses"] == 2
                assert stats["cache_hits"] == 0
                assert stats["loaded_ontologies"] == 2
                assert stats["total_terms"] == 12  # 2 + 10
                assert stats["total_relationships"] == 6  # 1 + 5

                # Load third file (should evict first)
                result3 = manager.load_ontology(files[2])
                assert result3.success is True

                stats = manager.get_statistics()
                assert stats["cache_size"] == 2  # Still at limit
                assert stats["cache_misses"] == 3
                assert stats["cache_hits"] == 0
                assert stats["loaded_ontologies"] == 3
                assert stats["total_terms"] == 12  # 2 + 10 + 0
                assert stats["total_relationships"] == 6  # 1 + 5 + 0

                # Load first file again (should be cache miss due to eviction)
                result4 = manager.load_ontology(files[0])
                assert result4.success is True

                stats = manager.get_statistics()
                assert stats["cache_size"] == 2
                assert stats["cache_misses"] == 4
                assert stats["cache_hits"] == 0
                assert stats["total_loads"] == 4
                assert stats["successful_loads"] == 4

    def test_statistics_after_ontology_removal(
        self, ontology_manager, create_test_file, sample_ontology_small
    ):
        """Test statistics accuracy after ontology removal."""
        file_path, _, tmp_dir = create_test_file("removal_test.owl")

        try:
            with patch(
                "aim2_project.aim2_ontology.ontology_manager.auto_detect_parser"
            ) as mock_auto_detect:
                parser = Mock()
                parser.format_name = "owl"
                parser.parse.return_value = ParseResult(
                    success=True,
                    data=sample_ontology_small,
                    errors=[],
                    warnings=[],
                    metadata={},
                    parse_time=0.1,
                )
                mock_auto_detect.return_value = parser

                # Load ontology
                result = ontology_manager.load_ontology(file_path)
                assert result.success is True

                stats_before = ontology_manager.get_statistics()
                assert stats_before["loaded_ontologies"] == 1
                assert stats_before["total_terms"] == 2
                assert stats_before["total_relationships"] == 1
                assert stats_before["cache_size"] == 1

                # Remove ontology
                removed = ontology_manager.remove_ontology(sample_ontology_small.id)
                assert removed is True

                stats_after = ontology_manager.get_statistics()
                assert stats_after["loaded_ontologies"] == 0
                assert stats_after["total_terms"] == 0
                assert stats_after["total_relationships"] == 0
                assert stats_after["cache_size"] == 0  # Cache should be cleaned up

                # Load statistics should remain unchanged
                assert stats_after["total_loads"] == stats_before["total_loads"]
                assert stats_after["successful_loads"] == stats_before["successful_loads"]
                assert stats_after["failed_loads"] == stats_before["failed_loads"]
                assert stats_after["cache_hits"] == stats_before["cache_hits"]
                assert stats_after["cache_misses"] == stats_before["cache_misses"]

        finally:
            # Cleanup
            import shutil

            shutil.rmtree(tmp_dir, ignore_errors=True)

    def test_statistics_after_cache_clear(
        self, ontology_manager, create_test_file, sample_ontology_small
    ):
        """Test statistics after cache clearing."""
        file_path, _, tmp_dir = create_test_file("cache_clear_test.owl")

        try:
            with patch(
                "aim2_project.aim2_ontology.ontology_manager.auto_detect_parser"
            ) as mock_auto_detect:
                parser = Mock()
                parser.format_name = "owl"
                parser.parse.return_value = ParseResult(
                    success=True,
                    data=sample_ontology_small,
                    errors=[],
                    warnings=[],
                    metadata={},
                    parse_time=0.1,
                )
                mock_auto_detect.return_value = parser

                # Load ontology
                result = ontology_manager.load_ontology(file_path)
                assert result.success is True

                stats_before = ontology_manager.get_statistics()
                assert stats_before["cache_size"] == 1
                assert stats_before["loaded_ontologies"] == 1

                # Clear cache
                ontology_manager.clear_cache()

                stats_after = ontology_manager.get_statistics()
                assert stats_after["cache_size"] == 0
                assert stats_after["loaded_ontologies"] == 1  # Ontologies still loaded in manager

                # Load statistics should remain unchanged
                assert stats_after["total_loads"] == stats_before["total_loads"]
                assert stats_after["successful_loads"] == stats_before["successful_loads"]
                assert stats_after["cache_hits"] == stats_before["cache_hits"]
                assert stats_after["cache_misses"] == stats_before["cache_misses"]

        finally:
            # Cleanup
            import shutil

            shutil.rmtree(tmp_dir, ignore_errors=True)

    def test_statistics_with_mixed_success_failure(
        self, ontology_manager, create_test_file, sample_ontology_small
    ):
        """Test statistics tracking with mix of successful and failed loads."""
        success_file, _, tmp_dir1 = create_test_file("success.owl")
        failure_file, _, tmp_dir2 = create_test_file("failure.owl")

        try:
            with patch(
                "aim2_project.aim2_ontology.ontology_manager.auto_detect_parser"
            ) as mock_auto_detect:

                def get_parser(*args, **kwargs):
                    file_path = str(args[0]) if args else str(kwargs.get("file_path", ""))
                    if "success.owl" in file_path:
                        parser = Mock()
                        parser.format_name = "owl"
                        parser.parse.return_value = ParseResult(
                            success=True,
                            data=sample_ontology_small,
                            errors=[],
                            warnings=[],
                            metadata={},
                            parse_time=0.1,
                        )
                        return parser
                    elif "failure.owl" in file_path:
                        parser = Mock()
                        parser.format_name = "owl"
                        parser.parse.return_value = ParseResult(
                            success=False,
                            data=None,
                            errors=["Parse error"],
                            warnings=[],
                            metadata={},
                            parse_time=0.1,
                        )
                        return parser
                    return None

                mock_auto_detect.side_effect = get_parser

                # Load successful file
                result1 = ontology_manager.load_ontology(success_file)
                assert result1.success is True

                # Load failed file
                result2 = ontology_manager.load_ontology(failure_file)
                assert result2.success is False

                # Load successful file again (cache hit)
                result3 = ontology_manager.load_ontology(success_file)
                assert result3.success is True

                # Load failed file again
                result4 = ontology_manager.load_ontology(failure_file)
                assert result4.success is False

                stats = ontology_manager.get_statistics()
                assert stats["total_loads"] == 4
                assert stats["successful_loads"] == 1  # Only one actual successful parse (cache hit doesn't count)
                assert stats["failed_loads"] == 2
                assert stats["cache_hits"] == 1
                assert stats["cache_misses"] == 3  # success:1, failure:1, success:cache_hit, failure:1 (failures not cached)
                assert stats["formats_loaded"]["owl"] == 1  # Only successful loads count
                assert stats["loaded_ontologies"] == 1
                assert stats["total_terms"] == 2
                assert stats["total_relationships"] == 1
                assert stats["cache_size"] == 1

        finally:
            # Cleanup
            import shutil

            shutil.rmtree(tmp_dir1, ignore_errors=True)
            shutil.rmtree(tmp_dir2, ignore_errors=True)

    def test_statistics_with_disabled_caching_loads(
        self, ontology_manager_no_cache, create_test_file, sample_ontology_small
    ):
        """Test statistics accuracy when caching is disabled."""
        file_path, _, tmp_dir = create_test_file("no_cache_test.owl")

        try:
            with patch(
                "aim2_project.aim2_ontology.ontology_manager.auto_detect_parser"
            ) as mock_auto_detect:
                parser = Mock()
                parser.format_name = "owl"
                parser.parse.return_value = ParseResult(
                    success=True,
                    data=sample_ontology_small,
                    errors=[],
                    warnings=[],
                    metadata={},
                    parse_time=0.1,
                )
                mock_auto_detect.return_value = parser

                # Load same file multiple times
                for i in range(3):
                    result = ontology_manager_no_cache.load_ontology(file_path)
                    assert result.success is True

                stats = ontology_manager_no_cache.get_statistics()
                assert stats["total_loads"] == 3
                assert stats["successful_loads"] == 3
                assert stats["failed_loads"] == 0
                assert stats["cache_hits"] == 0  # No caching
                assert stats["cache_misses"] == 3  # Cache misses tracked for statistics when disabled
                assert stats["formats_loaded"]["owl"] == 3  # Each load counts
                assert stats["cache_size"] == 0
                assert stats["cache_enabled"] is False
                assert stats["loaded_ontologies"] == 1  # Same ontology loaded multiple times
                assert stats["total_terms"] == 2
                assert stats["total_relationships"] == 1

        finally:
            # Cleanup
            import shutil

            shutil.rmtree(tmp_dir, ignore_errors=True)

    def test_statistics_data_types_and_structure(self, ontology_manager):
        """Test that statistics return correct data types and structure."""
        stats = ontology_manager.get_statistics()

        # Test integer types
        assert isinstance(stats["total_loads"], int)
        assert isinstance(stats["successful_loads"], int)
        assert isinstance(stats["failed_loads"], int)
        assert isinstance(stats["cache_hits"], int)
        assert isinstance(stats["cache_misses"], int)
        assert isinstance(stats["cache_size"], int)
        assert isinstance(stats["cache_limit"], int)
        assert isinstance(stats["loaded_ontologies"], int)
        assert isinstance(stats["total_terms"], int)
        assert isinstance(stats["total_relationships"], int)

        # Test boolean type
        assert isinstance(stats["cache_enabled"], bool)

        # Test dictionary type
        assert isinstance(stats["formats_loaded"], dict)

        # Test that all numeric values are non-negative
        for key in [
            "total_loads",
            "successful_loads",
            "failed_loads",
            "cache_hits",
            "cache_misses",
            "cache_size",
            "cache_limit",
            "loaded_ontologies",
            "total_terms",
            "total_relationships",
        ]:
            assert stats[key] >= 0, f"{key} should be non-negative"

    def test_statistics_mathematical_consistency(
        self, ontology_manager, create_test_file, sample_ontology_small
    ):
        """Test mathematical consistency of statistics."""
        file_path, _, tmp_dir = create_test_file("consistency_test.owl")

        try:
            with patch(
                "aim2_project.aim2_ontology.ontology_manager.auto_detect_parser"
            ) as mock_auto_detect:
                parser = Mock()
                parser.format_name = "owl"
                parser.parse.return_value = ParseResult(
                    success=True,
                    data=sample_ontology_small,
                    errors=[],
                    warnings=[],
                    metadata={},
                    parse_time=0.1,
                )
                mock_auto_detect.return_value = parser

                # Perform various operations
                ontology_manager.load_ontology(file_path)  # Success + cache miss
                ontology_manager.load_ontology(file_path)  # Success + cache hit
                
                # Test failure by mocking parser to return None for nonexistent file
                with patch(
                    "aim2_project.aim2_ontology.ontology_manager.auto_detect_parser",
                    return_value=None
                ):
                    ontology_manager.load_ontology("nonexistent.owl")  # Failure

                stats = ontology_manager.get_statistics()

                # Mathematical consistency checks
                # Note: successful_loads only counts actual parsing, not cache hits
                # So successful_loads + failed_loads may be less than total_loads due to cache hits
                assert (
                    stats["successful_loads"] + stats["failed_loads"] + stats["cache_hits"]
                    == stats["total_loads"]
                )
                assert stats["cache_hits"] + stats["cache_misses"] <= stats["total_loads"]
                assert stats["cache_size"] <= stats["cache_limit"]
                assert stats["loaded_ontologies"] <= stats["successful_loads"]

                # Format count consistency
                total_format_loads = sum(stats["formats_loaded"].values())
                assert total_format_loads <= stats["successful_loads"]

        finally:
            # Cleanup
            import shutil

            shutil.rmtree(tmp_dir, ignore_errors=True)

    def test_statistics_persistence_across_operations(
        self, ontology_manager, create_test_file, sample_ontology_small, sample_ontology_large
    ):
        """Test that statistics persist correctly across various operations."""
        file1, _, tmp_dir1 = create_test_file("persist1.owl")
        file2, _, tmp_dir2 = create_test_file("persist2.owl")

        try:
            with patch(
                "aim2_project.aim2_ontology.ontology_manager.auto_detect_parser"
            ) as mock_auto_detect:

                def get_parser(*args, **kwargs):
                    file_path = str(args[0]) if args else str(kwargs.get("file_path", ""))
                    if "persist1.owl" in file_path:
                        parser = Mock()
                        parser.format_name = "owl"
                        parser.parse.return_value = ParseResult(
                            success=True,
                            data=sample_ontology_small,
                            errors=[],
                            warnings=[],
                            metadata={},
                            parse_time=0.1,
                        )
                        return parser
                    elif "persist2.owl" in file_path:
                        parser = Mock()
                        parser.format_name = "rdf"
                        parser.parse.return_value = ParseResult(
                            success=True,
                            data=sample_ontology_large,
                            errors=[],
                            warnings=[],
                            metadata={},
                            parse_time=0.1,
                        )
                        return parser
                    return None

                mock_auto_detect.side_effect = get_parser

                # Track statistics through operations
                checkpoints = []

                # Initial state
                checkpoints.append(("initial", ontology_manager.get_statistics()))

                # Load first file
                ontology_manager.load_ontology(file1)
                checkpoints.append(("after_load1", ontology_manager.get_statistics()))

                # Load second file
                ontology_manager.load_ontology(file2)
                checkpoints.append(("after_load2", ontology_manager.get_statistics()))

                # Cache hit on first file
                ontology_manager.load_ontology(file1)
                checkpoints.append(("after_cache_hit", ontology_manager.get_statistics()))

                # Clear cache
                ontology_manager.clear_cache()
                checkpoints.append(("after_cache_clear", ontology_manager.get_statistics()))

                # Remove ontology
                ontology_manager.remove_ontology(sample_ontology_small.id)
                checkpoints.append(("after_removal", ontology_manager.get_statistics()))

                # Verify statistical progression
                for i in range(1, len(checkpoints)):
                    prev_name, prev_stats = checkpoints[i - 1]
                    curr_name, curr_stats = checkpoints[i]

                    # Total loads should only increase or stay same
                    assert (
                        curr_stats["total_loads"] >= prev_stats["total_loads"]
                    ), f"Total loads decreased from {prev_name} to {curr_name}"

                    # Cache size should be within limits
                    assert (
                        curr_stats["cache_size"] <= curr_stats["cache_limit"]
                    ), f"Cache size exceeded limit at {curr_name}"

        finally:
            # Cleanup
            import shutil

            shutil.rmtree(tmp_dir1, ignore_errors=True)
            shutil.rmtree(tmp_dir2, ignore_errors=True)


if __name__ == "__main__":
    pytest.main([__file__])