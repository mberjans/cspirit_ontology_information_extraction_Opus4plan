#!/usr/bin/env python3
"""
Comprehensive unit tests for OntologyManager loading logic.

This module contains extensive unit tests for the OntologyManager class,
focusing on the loading functionality and integration with the existing
parser framework. The tests follow established project patterns using
pytest with fixtures, mocking, and temporary directories.

Test Coverage:
- Single ontology loading with format auto-detection
- Multi-source ontology loading
- Caching functionality and cache validation
- Error handling for invalid files and edge cases
- Statistics generation and tracking
- Integration with existing parsers (OWL, CSV, JSON-LD)
- Cache eviction and LRU management
- File modification detection
- Parser auto-detection failure scenarios
"""

import json
import tempfile
import time
from pathlib import Path
from unittest.mock import Mock, patch

import pytest

# Import the modules to be tested
try:
    from aim2_project.aim2_ontology.models import Ontology, Relationship, Term
    from aim2_project.aim2_ontology.ontology_manager import (
        CacheEntry,
        LoadResult,
        OntologyLoadError,
        OntologyManager,
        OntologyManagerError,
    )
    from aim2_project.aim2_ontology.parsers import ParseResult
except ImportError:
    # For development scenarios where modules might not be fully available
    import warnings

    warnings.warn("Some imports failed - tests may be skipped", ImportWarning)


class TestOntologyManagerLoading:
    """Comprehensive test suite for OntologyManager loading functionality."""

    @pytest.fixture
    def temp_dir(self):
        """Create a temporary directory for test files."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            yield Path(tmp_dir)

    @pytest.fixture
    def sample_ontology(self):
        """Create a sample ontology for testing."""
        terms = {
            "CHEBI:15422": Term(
                id="CHEBI:15422",
                name="ATP",
                definition="Adenosine 5'-triphosphate",
                synonyms=["adenosine triphosphate", "adenosine 5'-triphosphate"],
                namespace="chemical",
            ),
            "GO:0008152": Term(
                id="GO:0008152",
                name="metabolic process",
                definition="The chemical reactions and pathways",
                namespace="biological_process",
            ),
        }

        relationships = {
            "REL:001": Relationship(
                id="REL:001",
                subject="CHEBI:15422",
                predicate="regulates",
                object="GO:0008152",
                confidence=0.95,
            )
        }

        return Ontology(
            id="TEST:001",
            name="Test Ontology",
            version="1.0",
            description="A test ontology for unit testing",
            terms=terms,
            relationships=relationships,
            namespaces=["chemical", "biological_process"],
        )

    @pytest.fixture
    def ontology_manager(self):
        """Create an OntologyManager instance for testing."""
        return OntologyManager(enable_caching=True, cache_size_limit=5)

    @pytest.fixture
    def mock_parser(self, sample_ontology):
        """Create a mock parser for testing."""
        parser = Mock()
        parser.format_name = "test_format"
        parser.parse.return_value = ParseResult(
            success=True,
            data=sample_ontology,
            errors=[],
            warnings=[],
            metadata={"test": True},
            parse_time=0.1,
        )
        return parser

    @pytest.fixture
    def sample_owl_file(self, temp_dir, sample_ontology):
        """Create a sample OWL file for testing."""
        owl_content = """<?xml version="1.0"?>
<rdf:RDF xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#"
         xmlns:owl="http://www.w3.org/2002/07/owl#"
         xmlns:rdfs="http://www.w3.org/2000/01/rdf-schema#">
    <owl:Ontology rdf:about="http://example.org/test"/>
    <owl:Class rdf:about="http://example.org/ATP">
        <rdfs:label>ATP</rdfs:label>
        <rdfs:comment>Adenosine 5'-triphosphate</rdfs:comment>
    </owl:Class>
</rdf:RDF>"""

        owl_file = temp_dir / "test_ontology.owl"
        owl_file.write_text(owl_content)
        return owl_file

    @pytest.fixture
    def sample_csv_file(self, temp_dir):
        """Create a sample CSV file for testing."""
        csv_content = """id,name,definition,namespace
CHEBI:15422,ATP,Adenosine 5'-triphosphate,chemical
GO:0008152,metabolic process,The chemical reactions and pathways,biological_process"""

        csv_file = temp_dir / "test_ontology.csv"
        csv_file.write_text(csv_content)
        return csv_file

    @pytest.fixture
    def sample_json_file(self, temp_dir):
        """Create a sample JSON-LD file for testing."""
        json_content = {
            "@context": {
                "@vocab": "http://example.org/",
                "rdfs": "http://www.w3.org/2000/01/rdf-schema#",
            },
            "@id": "test-ontology",
            "@type": "Ontology",
            "name": "Test Ontology",
            "terms": [
                {
                    "@id": "CHEBI:15422",
                    "rdfs:label": "ATP",
                    "definition": "Adenosine 5'-triphosphate",
                }
            ],
        }

        json_file = temp_dir / "test_ontology.jsonld"
        json_file.write_text(json.dumps(json_content, indent=2))
        return json_file

    def test_initialization(self):
        """Test OntologyManager initialization with different parameters."""
        # Test default initialization
        manager = OntologyManager()
        assert manager.enable_caching is True
        assert manager.cache_size_limit == 100
        assert len(manager.ontologies) == 0
        assert len(manager._cache) == 0
        assert manager.load_stats["total_loads"] == 0

        # Test custom initialization
        manager = OntologyManager(enable_caching=False, cache_size_limit=50)
        assert manager.enable_caching is False
        assert manager.cache_size_limit == 50

    @patch("aim2_project.aim2_ontology.ontology_manager.auto_detect_parser")
    def test_load_ontology_success(
        self, mock_auto_detect, ontology_manager, mock_parser, sample_ontology
    ):
        """Test successful ontology loading with auto-detection."""
        mock_auto_detect.return_value = mock_parser

        result = ontology_manager.load_ontology("test_file.owl")

        assert result.success is True
        assert result.ontology == sample_ontology
        assert result.source_path == "test_file.owl"
        assert result.load_time > 0
        assert len(result.errors) == 0
        assert result.metadata["format"] == "test_format"
        assert result.metadata["terms_count"] == 2
        assert result.metadata["relationships_count"] == 1

        # Verify ontology was stored
        assert sample_ontology.id in ontology_manager.ontologies
        assert ontology_manager.ontologies[sample_ontology.id] == sample_ontology

        # Verify statistics were updated
        assert ontology_manager.load_stats["total_loads"] == 1
        assert ontology_manager.load_stats["successful_loads"] == 1
        assert ontology_manager.load_stats["failed_loads"] == 0
        assert ontology_manager.load_stats["formats_loaded"]["test_format"] == 1

    @patch("aim2_project.aim2_ontology.ontology_manager.auto_detect_parser")
    def test_load_ontology_parser_not_found(self, mock_auto_detect, ontology_manager):
        """Test loading failure when no suitable parser is found."""
        mock_auto_detect.return_value = None

        result = ontology_manager.load_ontology("unknown_file.xyz")

        assert result.success is False
        assert result.ontology is None
        assert result.source_path == "unknown_file.xyz"
        assert len(result.errors) == 1
        assert "No suitable parser found" in result.errors[0]

        # Verify statistics were updated
        assert ontology_manager.load_stats["total_loads"] == 1
        assert ontology_manager.load_stats["successful_loads"] == 0
        assert ontology_manager.load_stats["failed_loads"] == 1

    @patch("aim2_project.aim2_ontology.ontology_manager.auto_detect_parser")
    def test_load_ontology_parse_failure(self, mock_auto_detect, ontology_manager):
        """Test loading failure when parser fails."""
        mock_parser = Mock()
        mock_parser.format_name = "test_format"
        mock_parser.parse.return_value = ParseResult(
            success=False,
            data=None,
            errors=["Parse error: Invalid syntax"],
            warnings=["Warning: Missing namespace"],
            metadata={},
            parse_time=0.05,
        )
        mock_auto_detect.return_value = mock_parser

        result = ontology_manager.load_ontology("bad_file.owl")

        assert result.success is False
        assert result.ontology is None
        assert len(result.errors) >= 1
        assert "Failed to parse ontology" in result.errors[0]
        assert "Parse error: Invalid syntax" in result.errors
        assert len(result.warnings) == 1
        assert "Warning: Missing namespace" in result.warnings

    @patch("aim2_project.aim2_ontology.ontology_manager.auto_detect_parser")
    def test_load_ontology_invalid_type(self, mock_auto_detect, ontology_manager):
        """Test loading failure when parser returns wrong type."""
        mock_parser = Mock()
        mock_parser.parse.return_value = ParseResult(
            success=True,
            data="not_an_ontology",  # Wrong type
            errors=[],
            warnings=[],
            metadata={},
            parse_time=0.1,
        )
        mock_auto_detect.return_value = mock_parser

        result = ontology_manager.load_ontology("bad_type.owl")

        assert result.success is False
        assert result.ontology is None
        assert len(result.errors) == 1
        assert "invalid ontology type" in result.errors[0]

    @patch("aim2_project.aim2_ontology.ontology_manager.auto_detect_parser")
    def test_load_ontology_exception_handling(self, mock_auto_detect, ontology_manager):
        """Test handling of unexpected exceptions during loading."""
        mock_auto_detect.side_effect = Exception("Unexpected error")

        result = ontology_manager.load_ontology("error_file.owl")

        assert result.success is False
        assert result.ontology is None
        assert len(result.errors) == 1
        assert "Unexpected error loading ontology" in result.errors[0]
        assert "Unexpected error" in result.errors[0]

    @patch("aim2_project.aim2_ontology.ontology_manager.auto_detect_parser")
    def test_load_ontologies_multiple_sources(
        self, mock_auto_detect, ontology_manager, mock_parser
    ):
        """Test loading multiple ontology sources."""
        mock_auto_detect.return_value = mock_parser

        sources = ["file1.owl", "file2.csv", "file3.jsonld"]
        results = ontology_manager.load_ontologies(sources)

        assert len(results) == 3
        assert all(result.success for result in results)
        assert (
            len(ontology_manager.ontologies) == 1
        )  # Same ontology loaded 3 times (overwrites)
        assert ontology_manager.load_stats["total_loads"] == 3
        assert ontology_manager.load_stats["successful_loads"] == 3

    @patch("aim2_project.aim2_ontology.ontology_manager.auto_detect_parser")
    def test_caching_functionality(
        self, mock_auto_detect, ontology_manager, mock_parser, temp_dir
    ):
        """Test caching functionality with cache hits and misses."""
        mock_auto_detect.return_value = mock_parser

        # Create a test file
        test_file = temp_dir / "test.owl"
        test_file.write_text("test content")

        # First load - should be cache miss
        result1 = ontology_manager.load_ontology(str(test_file))
        assert result1.success is True
        assert (
            "cache_hit" not in result1.metadata
            or result1.metadata["cache_hit"] is False
        )
        assert ontology_manager.load_stats["cache_misses"] == 1
        assert ontology_manager.load_stats["cache_hits"] == 0

        # Second load - should be cache hit
        result2 = ontology_manager.load_ontology(str(test_file))
        assert result2.success is True
        assert result2.metadata["cache_hit"] is True
        assert result2.metadata["access_count"] == 2
        assert ontology_manager.load_stats["cache_misses"] == 1
        assert ontology_manager.load_stats["cache_hits"] == 1

        # Verify parser was only called once
        assert mock_parser.parse.call_count == 1

    def test_caching_disabled(self, sample_ontology):
        """Test behavior when caching is disabled."""
        manager = OntologyManager(enable_caching=False)

        with patch(
            "aim2_project.aim2_ontology.ontology_manager.auto_detect_parser"
        ) as mock_auto_detect:
            mock_parser = Mock()
            mock_parser.format_name = "test_format"
            mock_parser.parse.return_value = ParseResult(
                success=True,
                data=sample_ontology,
                errors=[],
                warnings=[],
                metadata={},
                parse_time=0.1,
            )
            mock_auto_detect.return_value = mock_parser

            # Load same file twice
            result1 = manager.load_ontology("test.owl")
            result2 = manager.load_ontology("test.owl")

            assert result1.success is True
            assert result2.success is True
            assert "cache_hit" not in result1.metadata
            assert "cache_hit" not in result2.metadata
            assert manager.load_stats["cache_hits"] == 0
            assert manager.load_stats["cache_misses"] == 2

            # Parser should be called twice
            assert mock_parser.parse.call_count == 2

    @patch("aim2_project.aim2_ontology.ontology_manager.auto_detect_parser")
    def test_cache_eviction(self, mock_auto_detect, sample_ontology, temp_dir):
        """Test LRU cache eviction when cache limit is reached."""
        manager = OntologyManager(enable_caching=True, cache_size_limit=2)

        # Create different sample ontologies
        ontologies = []
        for i in range(3):
            ontology = Ontology(
                id=f"TEST:{i:03d}",
                name=f"Test Ontology {i}",
                version="1.0",
                terms={},
                relationships={},
                namespaces=[],
            )
            ontologies.append(ontology)

        def create_mock_parser(ontology):
            mock_parser = Mock()
            mock_parser.format_name = "test_format"
            mock_parser.parse.return_value = ParseResult(
                success=True,
                data=ontology,
                errors=[],
                warnings=[],
                metadata={},
                parse_time=0.1,
            )
            return mock_parser

        # Create test files
        test_files = []
        for i in range(3):
            test_file = temp_dir / f"test{i}.owl"
            test_file.write_text(f"test content {i}")
            test_files.append(str(test_file))

        # Mock parser to return different ontologies
        mock_auto_detect.side_effect = [
            create_mock_parser(ontologies[0]),
            create_mock_parser(ontologies[1]),
            create_mock_parser(ontologies[2]),
        ]

        # Load files to fill cache
        manager.load_ontology(test_files[0])
        manager.load_ontology(test_files[1])
        assert len(manager._cache) == 2

        # Load third file - should evict first one
        manager.load_ontology(test_files[2])
        assert len(manager._cache) == 2
        assert test_files[0] not in manager._cache  # Should be evicted
        assert test_files[1] in manager._cache
        assert test_files[2] in manager._cache

    @patch("aim2_project.aim2_ontology.ontology_manager.auto_detect_parser")
    def test_cache_invalidation_on_file_modification(
        self, mock_auto_detect, ontology_manager, mock_parser, temp_dir
    ):
        """Test cache invalidation when source file is modified."""
        mock_auto_detect.return_value = mock_parser

        # Create test file
        test_file = temp_dir / "test.owl"
        test_file.write_text("original content")
        test_file.stat().st_mtime

        # First load
        result1 = ontology_manager.load_ontology(str(test_file))
        assert result1.success is True
        assert ontology_manager.load_stats["cache_misses"] == 1

        # Modify file (ensure different mtime)
        time.sleep(0.1)
        test_file.write_text("modified content")

        # Second load - should detect modification and reload
        result2 = ontology_manager.load_ontology(str(test_file))
        assert result2.success is True
        assert ontology_manager.load_stats["cache_misses"] == 2  # Cache was invalidated
        assert mock_parser.parse.call_count == 2

    def test_add_ontology_success(self, ontology_manager, sample_ontology):
        """Test successful addition of ontology to manager."""
        result = ontology_manager.add_ontology(sample_ontology)

        assert result is True
        assert sample_ontology.id in ontology_manager.ontologies
        assert ontology_manager.ontologies[sample_ontology.id] == sample_ontology

    def test_add_ontology_invalid_type(self, ontology_manager):
        """Test addition of invalid ontology type."""
        result = ontology_manager.add_ontology("not_an_ontology")

        assert result is False
        assert len(ontology_manager.ontologies) == 0

    def test_get_ontology(self, ontology_manager, sample_ontology):
        """Test retrieval of ontology by ID."""
        ontology_manager.add_ontology(sample_ontology)

        retrieved = ontology_manager.get_ontology(sample_ontology.id)
        assert retrieved == sample_ontology

        not_found = ontology_manager.get_ontology("NONEXISTENT:001")
        assert not_found is None

    def test_list_ontologies(self, ontology_manager, sample_ontology):
        """Test listing of ontology IDs."""
        assert ontology_manager.list_ontologies() == []

        ontology_manager.add_ontology(sample_ontology)
        ontology_ids = ontology_manager.list_ontologies()

        assert len(ontology_ids) == 1
        assert sample_ontology.id in ontology_ids

    def test_remove_ontology(self, ontology_manager, sample_ontology):
        """Test removal of ontology and cache cleanup."""
        ontology_manager.add_ontology(sample_ontology)

        # Add to cache as well
        cache_entry = CacheEntry(
            ontology=sample_ontology, load_time=time.time(), source_path="test.owl"
        )
        ontology_manager._cache["test.owl"] = cache_entry

        result = ontology_manager.remove_ontology(sample_ontology.id)

        assert result is True
        assert sample_ontology.id not in ontology_manager.ontologies
        assert "test.owl" not in ontology_manager._cache

        # Test removing non-existent ontology
        result = ontology_manager.remove_ontology("NONEXISTENT:001")
        assert result is False

    def test_clear_cache(self, ontology_manager, sample_ontology):
        """Test cache clearing functionality."""
        # Add entries to cache
        for i in range(3):
            cache_entry = CacheEntry(
                ontology=sample_ontology,
                load_time=time.time(),
                source_path=f"test{i}.owl",
            )
            ontology_manager._cache[f"test{i}.owl"] = cache_entry

        assert len(ontology_manager._cache) == 3

        ontology_manager.clear_cache()

        assert len(ontology_manager._cache) == 0

    def test_get_statistics(self, ontology_manager, sample_ontology):
        """Test statistics generation."""
        # Add some data
        ontology_manager.add_ontology(sample_ontology)
        ontology_manager.load_stats["total_loads"] = 5
        ontology_manager.load_stats["successful_loads"] = 4
        ontology_manager.load_stats["failed_loads"] = 1
        ontology_manager.load_stats["cache_hits"] = 2
        ontology_manager.load_stats["cache_misses"] = 3
        ontology_manager.load_stats["formats_loaded"]["owl"] = 3
        ontology_manager.load_stats["formats_loaded"]["csv"] = 1

        stats = ontology_manager.get_statistics()

        # Check load stats
        assert stats["total_loads"] == 5
        assert stats["successful_loads"] == 4
        assert stats["failed_loads"] == 1
        assert stats["cache_hits"] == 2
        assert stats["cache_misses"] == 3

        # Check cache stats
        assert stats["cache_size"] == 0
        assert stats["cache_limit"] == 5
        assert stats["cache_enabled"] is True

        # Check ontology stats
        assert stats["loaded_ontologies"] == 1
        assert stats["total_terms"] == 2
        assert stats["total_relationships"] == 1

        # Check format stats
        assert stats["formats_loaded"]["owl"] == 3
        assert stats["formats_loaded"]["csv"] == 1

    def test_pathlib_path_support(self, ontology_manager, mock_parser, temp_dir):
        """Test support for pathlib.Path objects as source."""
        with patch(
            "aim2_project.aim2_ontology.ontology_manager.auto_detect_parser"
        ) as mock_auto_detect:
            mock_auto_detect.return_value = mock_parser

            test_file = temp_dir / "test.owl"
            test_file.write_text("test content")

            result = ontology_manager.load_ontology(test_file)  # Pass Path object

            assert result.success is True
            assert result.source_path == str(test_file)

    def test_concurrent_access_safety(self, ontology_manager, sample_ontology):
        """Test thread safety of cache operations."""
        import threading
        import time

        def add_to_cache(manager, ontology, file_path):
            cache_entry = CacheEntry(
                ontology=ontology, load_time=time.time(), source_path=file_path
            )
            manager._cache[file_path] = cache_entry

        # Create multiple threads that access cache simultaneously
        threads = []
        for i in range(10):
            thread = threading.Thread(
                target=add_to_cache,
                args=(ontology_manager, sample_ontology, f"test{i}.owl"),
            )
            threads.append(thread)

        # Start all threads
        for thread in threads:
            thread.start()

        # Wait for all threads to complete
        for thread in threads:
            thread.join()

        # Verify all entries were added
        assert len(ontology_manager._cache) == 10

    @pytest.mark.parametrize(
        "file_extension,expected_calls",
        [
            ("test.owl", 1),
            ("test.csv", 1),
            ("test.jsonld", 1),
            ("test.json", 1),
            ("test.unknown", 1),  # Should still try to parse
        ],
    )
    def test_format_detection_coverage(
        self, ontology_manager, mock_parser, file_extension, expected_calls
    ):
        """Test format detection for various file extensions."""
        with patch(
            "aim2_project.aim2_ontology.ontology_manager.auto_detect_parser"
        ) as mock_auto_detect:
            mock_auto_detect.return_value = mock_parser

            result = ontology_manager.load_ontology(file_extension)

            assert mock_auto_detect.call_count == expected_calls
            if mock_parser:
                assert result.success is True

    def test_memory_cleanup_on_failure(self, ontology_manager):
        """Test that failed loads don't leave artifacts in memory."""
        with patch(
            "aim2_project.aim2_ontology.ontology_manager.auto_detect_parser"
        ) as mock_auto_detect:
            mock_auto_detect.return_value = None  # Simulate parser not found

            initial_ontology_count = len(ontology_manager.ontologies)
            initial_cache_size = len(ontology_manager._cache)

            result = ontology_manager.load_ontology("nonexistent.xyz")

            assert result.success is False
            assert len(ontology_manager.ontologies) == initial_ontology_count
            assert len(ontology_manager._cache) == initial_cache_size

    def test_load_result_dataclass(self):
        """Test LoadResult dataclass functionality."""
        # Test with minimal data
        result = LoadResult(success=True)
        assert result.success is True
        assert result.ontology is None
        assert result.source_path is None
        assert result.load_time == 0.0
        assert result.errors == []
        assert result.warnings == []
        assert result.metadata == {}

        # Test with full data
        ontology = Mock()
        result = LoadResult(
            success=True,
            ontology=ontology,
            source_path="test.owl",
            load_time=1.5,
            errors=["error1"],
            warnings=["warning1"],
            metadata={"key": "value"},
        )
        assert result.ontology == ontology
        assert result.source_path == "test.owl"
        assert result.load_time == 1.5
        assert "error1" in result.errors
        assert "warning1" in result.warnings
        assert result.metadata["key"] == "value"

    def test_cache_entry_dataclass(self, sample_ontology):
        """Test CacheEntry dataclass functionality."""
        load_time = time.time()

        cache_entry = CacheEntry(
            ontology=sample_ontology, load_time=load_time, source_path="test.owl"
        )

        assert cache_entry.ontology == sample_ontology
        assert cache_entry.load_time == load_time
        assert cache_entry.source_path == "test.owl"
        assert cache_entry.file_mtime is None
        assert cache_entry.access_count == 0
        assert cache_entry.last_accessed > 0  # Should be set to current time

    def test_exception_classes(self):
        """Test custom exception classes."""
        # Test base exception
        base_error = OntologyManagerError("Base error")
        assert str(base_error) == "Base error"
        assert isinstance(base_error, Exception)

        # Test load error
        load_error = OntologyLoadError("Load error")
        assert str(load_error) == "Load error"
        assert isinstance(load_error, OntologyManagerError)
        assert isinstance(load_error, Exception)
