#!/usr/bin/env python3
"""
Comprehensive unit tests for OntologyManager multi-source loading functionality.

This module contains specialized unit tests focusing specifically on multi-source
ontology loading capabilities, including batch operations, mixed format scenarios,
performance aspects, memory management, concurrent access patterns, cache efficiency,
and comprehensive error handling for batch operations.

Test Coverage:
- Multi-source loading from different formats (OWL, CSV, JSON-LD)
- Mixed success/failure scenarios during batch loading
- Performance characteristics of batch loading operations
- Memory management when loading multiple large ontologies
- Concurrent access patterns and thread safety
- Cache efficiency with multiple sources
- Statistics aggregation for multi-source operations
- Error handling and reporting for batch operations
- Resource cleanup and memory optimization
- Load balancing and prioritization strategies
"""

import os
import json
import tempfile
import pytest
import time
import threading
import concurrent.futures
import gc
import psutil
from unittest.mock import patch, Mock, MagicMock
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, Any, List, Optional
import queue
import weakref

# Import the modules to be tested
try:
    from aim2_project.aim2_ontology.ontology_manager import (
        OntologyManager,
        LoadResult,
        CacheEntry,
        OntologyManagerError,
        OntologyLoadError,
    )
    from aim2_project.aim2_ontology.models import Ontology, Term, Relationship
    from aim2_project.aim2_ontology.parsers import ParseResult
except ImportError:
    import warnings
    warnings.warn("Some imports failed - tests may be skipped", ImportWarning)


class TestOntologyManagerMultiSource:
    """Comprehensive test suite for OntologyManager multi-source loading functionality."""

    @pytest.fixture
    def temp_dir(self):
        """Create a temporary directory for test files."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            yield Path(tmp_dir)

    @pytest.fixture
    def ontology_manager(self):
        """Create an OntologyManager instance for testing."""
        return OntologyManager(enable_caching=True, cache_size_limit=20)

    @pytest.fixture
    def sample_ontologies(self):
        """Create multiple sample ontologies for multi-source testing."""
        ontologies = []
        
        for i in range(5):
            terms = {}
            relationships = {}
            
            # Create diverse terms for each ontology
            for j in range(10):
                term_id = f"TEST:{i*10+j:06d}"
                terms[term_id] = Term(
                    id=term_id,
                    name=f"Term {j} from Ontology {i}",
                    definition=f"Definition for term {j} in ontology {i}",
                    synonyms=[f"synonym{j}_1", f"synonym{j}_2"],
                    namespace=f"namespace_{i}"
                )
            
            # Create relationships within each ontology
            for j in range(5):
                rel_id = f"REL:{i*10+j:06d}"
                relationships[rel_id] = Relationship(
                    id=rel_id,
                    subject=f"TEST:{i*10+j:06d}",
                    predicate="regulates",
                    object=f"TEST:{i*10+(j+1)%10:06d}",
                    confidence=0.8 + (j * 0.04)
                )
            
            ontology = Ontology(
                id=f"TESTMULTI:{i:06d}",
                name=f"Multi-Source Test Ontology {i}",
                version=f"1.{i}",
                description=f"Test ontology {i} for multi-source loading scenarios",
                terms=terms,
                relationships=relationships,
                namespaces=[f"namespace_{i}"]
            )
            ontologies.append(ontology)
        
        return ontologies

    @pytest.fixture
    def create_test_files(self, temp_dir, sample_ontologies):
        """Create test files in different formats for multi-source testing."""
        files = {}
        
        # Create OWL files
        for i in range(2):
            ontology = sample_ontologies[i]
            owl_content = f'''<?xml version="1.0"?>
<rdf:RDF xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#"
         xmlns:owl="http://www.w3.org/2002/07/owl#"
         xmlns:rdfs="http://www.w3.org/2000/01/rdf-schema#">
    <owl:Ontology rdf:about="http://example.org/{ontology.id}">
        <rdfs:label>{ontology.name}</rdfs:label>
        <rdfs:comment>{ontology.description}</rdfs:comment>
    </owl:Ontology>
    <!-- Terms would be here in real OWL -->
</rdf:RDF>'''
            
            owl_file = temp_dir / f"ontology_{i}.owl"
            owl_file.write_text(owl_content)
            files[f"owl_{i}"] = (str(owl_file), ontology)
        
        # Create CSV files
        for i in range(2, 4):
            ontology = sample_ontologies[i]
            csv_lines = ["id,name,definition,namespace,synonyms"]
            
            for term_id, term in list(ontology.terms.items())[:5]:  # First 5 terms
                synonyms = "|".join(term.synonyms)
                csv_lines.append(f"{term_id},{term.name},{term.definition},{term.namespace},{synonyms}")
            
            csv_file = temp_dir / f"ontology_{i}.csv"
            csv_file.write_text("\n".join(csv_lines))
            files[f"csv_{i}"] = (str(csv_file), ontology)
        
        # Create JSON-LD file
        ontology = sample_ontologies[4]
        jsonld_content = {
            "@context": {
                "@vocab": "http://example.org/",
                "rdfs": "http://www.w3.org/2000/01/rdf-schema#"
            },
            "@id": ontology.id,
            "@type": "Ontology",
            "rdfs:label": ontology.name,
            "rdfs:comment": ontology.description,
            "terms": [
                {
                    "@id": term_id,
                    "rdfs:label": term.name,
                    "definition": term.definition,
                    "namespace": term.namespace
                }
                for term_id, term in list(ontology.terms.items())[:3]
            ]
        }
        
        jsonld_file = temp_dir / "ontology_4.jsonld"
        jsonld_file.write_text(json.dumps(jsonld_content, indent=2))
        files["jsonld_4"] = (str(jsonld_file), ontology)
        
        return files

    def test_load_multiple_sources_mixed_formats(self, ontology_manager, create_test_files):
        """Test loading multiple ontologies from different formats simultaneously."""
        # Get all file paths
        sources = [file_info[0] for file_info in create_test_files.values()]
        
        with patch('aim2_project.aim2_ontology.ontology_manager.auto_detect_parser') as mock_auto_detect:
            # Mock successful parsing for all files
            def create_mock_parser(ontology):
                parser = Mock()
                parser.format_name = "mock_format"
                parser.parse.return_value = ParseResult(
                    success=True,
                    data=ontology,
                    errors=[],
                    warnings=[],
                    metadata={"test": True},
                    parse_time=0.1
                )
                return parser
            
            # Return appropriate ontology for each file
            mock_parsers = []
            for key, (file_path, ontology) in create_test_files.items():
                mock_parsers.append(create_mock_parser(ontology))
            mock_auto_detect.side_effect = mock_parsers
            
            # Load all sources
            results = ontology_manager.load_ontologies(sources)
            
            # Verify results
            assert len(results) == len(sources)
            successful_results = [r for r in results if r.success]
            assert len(successful_results) == len(sources), f"Expected all to succeed, got: {[r.errors for r in results if not r.success]}"
            
            # Verify each ontology was loaded
            assert len(ontology_manager.ontologies) == len(sources)
            
            # Verify statistics
            stats = ontology_manager.get_statistics()
            assert stats['total_loads'] == len(sources)
            assert stats['successful_loads'] == len(sources)
            assert stats['failed_loads'] == 0

    def test_mixed_success_failure_scenarios(self, ontology_manager, create_test_files):
        """Test batch loading with some files succeeding and others failing."""
        sources = list(create_test_files.keys())
        file_paths = [create_test_files[key][0] for key in sources]
        
        # Add some non-existent files to the mix
        file_paths.append("/nonexistent/file1.owl")
        file_paths.append("/nonexistent/file2.csv")
        
        with patch('aim2_project.aim2_ontology.ontology_manager.auto_detect_parser') as mock_auto_detect:
            def mock_parser_factory(call_count=[0]):
                call_count[0] += 1
                if call_count[0] <= len(sources):  # First N calls succeed
                    key = sources[call_count[0] - 1]
                    ontology = create_test_files[key][1]
                    parser = Mock()
                    parser.format_name = f"format_{call_count[0]}"
                    parser.parse.return_value = ParseResult(
                        success=True,
                        data=ontology,
                        errors=[],
                        warnings=[],
                        metadata={},
                        parse_time=0.1
                    )
                    return parser
                else:  # Later calls fail (for nonexistent files)
                    return None
            
            mock_auto_detect.side_effect = lambda **kwargs: mock_parser_factory()
            
            # Load with mixed success/failure
            results = ontology_manager.load_ontologies(file_paths)
            
            # Verify mixed results
            assert len(results) == len(file_paths)
            successful_results = [r for r in results if r.success]
            failed_results = [r for r in results if not r.success]
            
            assert len(successful_results) == len(sources)  # Original files succeed
            assert len(failed_results) == 2  # Nonexistent files fail
            
            # Verify only successful ontologies were stored
            assert len(ontology_manager.ontologies) == len(sources)
            
            # Verify statistics reflect mixed results
            stats = ontology_manager.get_statistics()
            assert stats['total_loads'] == len(file_paths)
            assert stats['successful_loads'] == len(sources)
            assert stats['failed_loads'] == 2

    def test_performance_batch_loading_large_ontologies(self, ontology_manager, temp_dir):
        """Test performance aspects of batch loading multiple large ontologies."""
        # Create larger test ontologies
        large_ontologies = []
        file_paths = []
        
        for i in range(3):  # 3 large ontologies
            # Create ontology with many terms
            terms = {}
            for j in range(500):  # 500 terms each
                term_id = f"LARGE:{i*1000+j:06d}"
                terms[term_id] = Term(
                    id=term_id,
                    name=f"Large Term {j} from Ontology {i}",
                    definition=f"Definition for large term {j} in ontology {i}",
                    synonyms=[f"syn{j}_1", f"syn{j}_2", f"syn{j}_3"],
                    namespace=f"large_namespace_{i}"
                )
            
            # Create many relationships
            relationships = {}
            for j in range(100):  # 100 relationships each
                rel_id = f"REL:{i*1000+j:06d}"
                relationships[rel_id] = Relationship(
                    id=rel_id,
                    subject=f"LARGE:{i*1000+j:06d}",
                    predicate="regulates",
                    object=f"LARGE:{i*1000+(j+1)%500:06d}",
                    confidence=0.9
                )
            
            ontology = Ontology(
                id=f"LARGETEST:{i:06d}",
                name=f"Large Test Ontology {i}",
                version="1.0",
                description=f"Large test ontology {i} with 500 terms",
                terms=terms,
                relationships=relationships,
                namespaces=[f"large_namespace_{i}"]
            )
            large_ontologies.append(ontology)
            
            # Create CSV file for this ontology
            csv_lines = ["id,name,definition,namespace"]
            for term_id, term in terms.items():
                csv_lines.append(f"{term_id},{term.name},{term.definition},{term.namespace}")
            
            csv_file = temp_dir / f"large_ontology_{i}.csv"
            csv_file.write_text("\n".join(csv_lines))
            file_paths.append(str(csv_file))
        
        with patch('aim2_project.aim2_ontology.ontology_manager.auto_detect_parser') as mock_auto_detect:
            def create_large_parser(ontology):
                parser = Mock()
                parser.format_name = "csv"
                parser.parse.return_value = ParseResult(
                    success=True,
                    data=ontology,
                    errors=[],
                    warnings=[],
                    metadata={"large": True},
                    parse_time=0.5  # Simulate longer parse time
                )
                return parser
            
            mock_auto_detect.side_effect = [create_large_parser(ont) for ont in large_ontologies]
            
            # Measure batch loading performance
            start_time = time.time()
            results = ontology_manager.load_ontologies(file_paths)
            total_time = time.time() - start_time
            
            # Verify all succeeded
            assert all(r.success for r in results)
            assert len(ontology_manager.ontologies) == 3
            
            # Performance assertions
            assert total_time < 30.0, f"Batch loading took too long: {total_time:.2f}s"
            average_time = total_time / len(file_paths)
            assert average_time < 10.0, f"Average load time too high: {average_time:.2f}s"
            
            # Verify memory usage is reasonable
            stats = ontology_manager.get_statistics()
            assert stats['total_terms'] == 1500  # 500 * 3
            assert stats['total_relationships'] == 300  # 100 * 3

    def test_memory_management_multi_source_loading(self, ontology_manager, temp_dir):
        """Test memory management when loading multiple large ontologies."""
        import gc
        import psutil
        
        process = psutil.Process()
        initial_memory = process.memory_info().rss
        
        # Create multiple ontologies of moderate size
        file_paths = []
        expected_ontologies = []
        
        for i in range(5):
            # Create ontology with moderate number of terms
            terms = {}
            for j in range(200):  # 200 terms each
                term_id = f"MEMORY:{i*1000+j:06d}"
                terms[term_id] = Term(
                    id=term_id,
                    name=f"Memory Test Term {j}",
                    definition=f"Definition {j}" * 10,  # Longer definitions
                    synonyms=[f"synonym{j}_{k}" for k in range(5)],  # More synonyms
                    namespace=f"memory_ns_{i}"
                )
            
            ontology = Ontology(
                id=f"MEMTEST:{i:06d}",
                name=f"Memory Test Ontology {i}",
                version="1.0",
                terms=terms,
                relationships={},
                namespaces=[f"memory_ns_{i}"]
            )
            expected_ontologies.append(ontology)
            
            # Create CSV file
            csv_lines = ["id,name,definition,namespace,synonyms"]
            for term_id, term in terms.items():
                synonyms = "|".join(term.synonyms)
                csv_lines.append(f"{term_id},{term.name},{term.definition},{term.namespace},{synonyms}")
            
            csv_file = temp_dir / f"memory_test_{i}.csv"
            csv_file.write_text("\n".join(csv_lines))
            file_paths.append(str(csv_file))
        
        with patch('aim2_project.aim2_ontology.ontology_manager.auto_detect_parser') as mock_auto_detect:
            mock_auto_detect.side_effect = [
                Mock(format_name="csv", parse=Mock(return_value=ParseResult(
                    success=True, data=ont, errors=[], warnings=[], metadata={}, parse_time=0.1
                )))
                for ont in expected_ontologies
            ]
            
            # Load all ontologies
            results = ontology_manager.load_ontologies(file_paths)
            
            # Force garbage collection
            gc.collect()
            
            # Check memory usage
            after_load_memory = process.memory_info().rss
            memory_increase = after_load_memory - initial_memory
            
            # Verify loading succeeded
            assert all(r.success for r in results)
            assert len(ontology_manager.ontologies) == 5
            
            # Memory increase should be reasonable (less than 500MB for this test)
            assert memory_increase < 500 * 1024 * 1024, f"Memory increase too high: {memory_increase / 1024 / 1024:.2f}MB"
            
            # Test cache cleanup to free memory
            ontology_manager.clear_cache()
            gc.collect()
            
            after_cleanup_memory = process.memory_info().rss
            memory_freed = after_load_memory - after_cleanup_memory
            
            # Should free some memory (at least 10MB)
            assert memory_freed > 10 * 1024 * 1024, f"Cache cleanup freed too little memory: {memory_freed / 1024 / 1024:.2f}MB"

    def test_concurrent_multi_source_loading(self, ontology_manager, create_test_files):
        """Test concurrent loading of multiple sources with thread safety."""
        file_groups = [
            [create_test_files[key][0] for key in list(create_test_files.keys())[:2]],
            [create_test_files[key][0] for key in list(create_test_files.keys())[2:4]],
            [create_test_files[key][0] for key in list(create_test_files.keys())[4:]]
        ]
        
        with patch('aim2_project.aim2_ontology.ontology_manager.auto_detect_parser') as mock_auto_detect:
            # Create parsers for all ontologies
            all_parsers = []
            for key, (file_path, ontology) in create_test_files.items():
                parser = Mock()
                parser.format_name = "concurrent_test"
                parser.parse.return_value = ParseResult(
                    success=True,
                    data=ontology,
                    errors=[],
                    warnings=[],
                    metadata={"concurrent": True},
                    parse_time=0.2
                )
                all_parsers.append(parser)
            
            # Use thread-local counter for parser assignment
            import threading
            thread_local = threading.local()
            
            def get_parser(*args, **kwargs):
                if not hasattr(thread_local, 'counter'):
                    thread_local.counter = 0
                parser = all_parsers[thread_local.counter % len(all_parsers)]
                thread_local.counter += 1
                return parser
            
            mock_auto_detect.side_effect = get_parser
            
            # Function to load a group of files
            def load_group(group):
                return ontology_manager.load_ontologies(group)
            
            # Use ThreadPoolExecutor for concurrent loading
            with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
                # Submit all loading tasks
                future_to_group = {
                    executor.submit(load_group, group): i 
                    for i, group in enumerate(file_groups)
                }
                
                # Collect results
                all_results = []
                for future in concurrent.futures.as_completed(future_to_group):
                    group_index = future_to_group[future]
                    try:
                        group_results = future.result()
                        all_results.extend(group_results)
                    except Exception as exc:
                        pytest.fail(f'Group {group_index} generated an exception: {exc}')
            
            # Verify concurrent loading results
            assert len(all_results) == len(create_test_files)
            successful_results = [r for r in all_results if r.success]
            assert len(successful_results) >= len(create_test_files) - 1, "Most concurrent loads should succeed"
            
            # Verify thread safety - no data corruption
            assert len(ontology_manager.ontologies) >= 1, "At least some ontologies should be loaded"
            
            # Verify statistics are consistent
            stats = ontology_manager.get_statistics()
            assert stats['total_loads'] >= len(create_test_files)

    def test_cache_efficiency_multi_source(self, ontology_manager, create_test_files):
        """Test cache efficiency with multiple sources and repeated loads."""
        sources = [file_info[0] for file_info in create_test_files.values()]
        
        with patch('aim2_project.aim2_ontology.ontology_manager.auto_detect_parser') as mock_auto_detect:
            # Create parsers
            parsers = []
            for key, (file_path, ontology) in create_test_files.items():
                parser = Mock()
                parser.format_name = f"cache_test_{key}"
                parser.parse.return_value = ParseResult(
                    success=True,
                    data=ontology,
                    errors=[],
                    warnings=[],
                    metadata={"cache_test": True},
                    parse_time=0.1
                )
                parsers.append(parser)
            
            # First load - all cache misses
            mock_auto_detect.side_effect = parsers
            results1 = ontology_manager.load_ontologies(sources)
            
            assert all(r.success for r in results1)
            assert ontology_manager.load_stats['cache_misses'] == len(sources)
            assert ontology_manager.load_stats['cache_hits'] == 0
            
            # Second load - should be all cache hits
            results2 = ontology_manager.load_ontologies(sources)
            
            assert all(r.success for r in results2)
            assert all(r.metadata.get('cache_hit', False) for r in results2)
            assert ontology_manager.load_stats['cache_hits'] == len(sources)
            
            # Verify parsers were only called once for each file
            for parser in parsers:
                assert parser.parse.call_count == 1
            
            # Test cache efficiency with partial overlaps
            partial_sources = sources[:3]  # First 3 sources
            new_source = "/tmp/new_ontology.owl"
            
            # Mock for new source
            new_ontology = Ontology(id="NEW:001", name="New Ontology", version="1.0", terms={}, relationships={}, namespaces=[])
            new_parser = Mock()
            new_parser.format_name = "new_format"
            new_parser.parse.return_value = ParseResult(
                success=True, data=new_ontology, errors=[], warnings=[], metadata={}, parse_time=0.1
            )
            mock_auto_detect.side_effect = [new_parser]
            
            # Load mix of cached and new
            mixed_sources = partial_sources + [new_source]
            initial_cache_hits = ontology_manager.load_stats['cache_hits']
            initial_cache_misses = ontology_manager.load_stats['cache_misses']
            
            results3 = ontology_manager.load_ontologies(mixed_sources)
            
            assert all(r.success for r in results3)
            # Should have 3 more cache hits (partial_sources) and 1 more miss (new_source)
            assert ontology_manager.load_stats['cache_hits'] == initial_cache_hits + 3
            assert ontology_manager.load_stats['cache_misses'] == initial_cache_misses + 1

    def test_statistics_aggregation_multi_source(self, ontology_manager, create_test_files):
        """Test statistics aggregation for multi-source operations."""
        sources = [file_info[0] for file_info in create_test_files.values()]
        
        with patch('aim2_project.aim2_ontology.ontology_manager.auto_detect_parser') as mock_auto_detect:
            # Create diverse parsers with different formats and characteristics
            format_names = ["owl", "csv", "jsonld", "rdf", "custom"]
            parsers = []
            
            for i, (key, (file_path, ontology)) in enumerate(create_test_files.items()):
                parser = Mock()
                parser.format_name = format_names[i % len(format_names)]
                parser.parse.return_value = ParseResult(
                    success=True,
                    data=ontology,
                    errors=[],
                    warnings=[f"Warning for {key}"] if i % 2 == 0 else [],
                    metadata={"terms_parsed": len(ontology.terms)},
                    parse_time=0.1 + (i * 0.05)  # Varying parse times
                )
                parsers.append(parser)
            
            mock_auto_detect.side_effect = parsers
            
            # Load all sources
            results = ontology_manager.load_ontologies(sources)
            
            # Get comprehensive statistics
            stats = ontology_manager.get_statistics()
            
            # Verify basic load statistics
            assert stats['total_loads'] == len(sources)
            assert stats['successful_loads'] == len(sources)
            assert stats['failed_loads'] == 0
            
            # Verify format-specific statistics
            formats_loaded = stats['formats_loaded']
            expected_formats = set(format_names[:len(sources)])
            actual_formats = set(formats_loaded.keys())
            assert expected_formats.issubset(actual_formats), f"Missing formats: {expected_formats - actual_formats}"
            
            # Verify aggregated ontology statistics
            expected_total_terms = sum(len(ont.terms) for _, (_, ont) in create_test_files.items())
            expected_total_relationships = sum(len(ont.relationships) for _, (_, ont) in create_test_files.items())
            
            assert stats['total_terms'] == expected_total_terms
            assert stats['total_relationships'] == expected_total_relationships
            assert stats['loaded_ontologies'] == len(sources)
            
            # Test statistics after partial failures
            # Add some failing sources
            failing_sources = ["/nonexistent1.owl", "/nonexistent2.csv"]
            all_sources = sources + failing_sources
            
            # Mock will return None for failing sources (auto_detect_parser failure)
            mock_auto_detect.side_effect = parsers + [None, None]
            
            # Reset stats and load again
            ontology_manager.load_stats = {
                'total_loads': 0, 'successful_loads': 0, 'failed_loads': 0,
                'cache_hits': 0, 'cache_misses': 0, 'formats_loaded': {}
            }
            ontology_manager.ontologies.clear()
            ontology_manager.clear_cache()
            
            results_with_failures = ontology_manager.load_ontologies(all_sources)
            stats_with_failures = ontology_manager.get_statistics()
            
            # Verify mixed statistics
            assert stats_with_failures['total_loads'] == len(all_sources)
            assert stats_with_failures['successful_loads'] == len(sources)
            assert stats_with_failures['failed_loads'] == len(failing_sources)
            
            # Success rate should be calculable
            success_rate = stats_with_failures['successful_loads'] / stats_with_failures['total_loads']
            assert 0 < success_rate < 1

    def test_error_handling_batch_operations(self, ontology_manager, temp_dir):
        """Test comprehensive error handling and reporting for batch operations."""
        # Create a mix of valid and invalid files
        test_files = []
        expected_results = []
        
        # Valid OWL file
        valid_owl = temp_dir / "valid.owl"
        valid_owl.write_text('''<?xml version="1.0"?>
<rdf:RDF xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#"
         xmlns:owl="http://www.w3.org/2002/07/owl#">
    <owl:Ontology rdf:about="http://example.org/valid"/>
</rdf:RDF>''')
        test_files.append(str(valid_owl))
        expected_results.append("success")
        
        # Invalid XML file
        invalid_xml = temp_dir / "invalid.owl"
        invalid_xml.write_text("<<invalid xml content>>")
        test_files.append(str(invalid_xml))
        expected_results.append("parse_failure")
        
        # Valid CSV file
        valid_csv = temp_dir / "valid.csv"
        valid_csv.write_text("id,name,definition\nTEST:001,Test Term,Test Definition")
        test_files.append(str(valid_csv))
        expected_results.append("success")
        
        # Empty file
        empty_file = temp_dir / "empty.jsonld"
        empty_file.write_text("")
        test_files.append(str(empty_file))
        expected_results.append("parse_failure")
        
        # Nonexistent file
        test_files.append("/totally/nonexistent/file.owl")
        expected_results.append("file_not_found")
        
        # File with unsupported format
        unsupported = temp_dir / "unsupported.xyz"
        unsupported.write_text("unsupported content")
        test_files.append(str(unsupported))
        expected_results.append("no_parser")
        
        with patch('aim2_project.aim2_ontology.ontology_manager.auto_detect_parser') as mock_auto_detect:
            def mock_parser_behavior(file_path=None, **kwargs):
                file_path_str = str(file_path) if file_path else ""
                
                if "valid.owl" in file_path_str:
                    # Valid OWL
                    parser = Mock()
                    parser.format_name = "owl"
                    parser.parse.return_value = ParseResult(
                        success=True,
                        data=Ontology(id="VALID:001", name="Valid", version="1.0", terms={}, relationships={}, namespaces=[]),
                        errors=[], warnings=[], metadata={}, parse_time=0.1
                    )
                    return parser
                elif "invalid.owl" in file_path_str:
                    # Parser found but parsing fails
                    parser = Mock()
                    parser.format_name = "owl"
                    parser.parse.return_value = ParseResult(
                        success=False, data=None,
                        errors=["XML parsing error: Invalid syntax"],
                        warnings=[], metadata={}, parse_time=0.05
                    )
                    return parser
                elif "valid.csv" in file_path_str:
                    # Valid CSV
                    parser = Mock()
                    parser.format_name = "csv"
                    parser.parse.return_value = ParseResult(
                        success=True,
                        data=Ontology(id="VALIDCSV:001", name="Valid CSV", version="1.0", terms={}, relationships={}, namespaces=[]),
                        errors=[], warnings=["Minor formatting issue"], metadata={}, parse_time=0.08
                    )
                    return parser
                elif "empty.jsonld" in file_path_str:
                    # Parser found but fails on empty content
                    parser = Mock()
                    parser.format_name = "jsonld"
                    parser.parse.return_value = ParseResult(
                        success=False, data=None,
                        errors=["Empty file or invalid JSON"],
                        warnings=[], metadata={}, parse_time=0.01
                    )
                    return parser
                elif "unsupported.xyz" in file_path_str:
                    # No parser available
                    return None
                else:
                    # For nonexistent files, this won't be called due to file check
                    return None
            
            mock_auto_detect.side_effect = mock_parser_behavior
            
            # Load all files with comprehensive error handling
            results = ontology_manager.load_ontologies(test_files)
            
            # Verify we got results for all files
            assert len(results) == len(test_files)
            
            # Analyze results by type
            success_count = 0
            parse_failure_count = 0
            file_not_found_count = 0
            no_parser_count = 0
            
            for i, result in enumerate(results):
                expected = expected_results[i]
                
                if expected == "success":
                    assert result.success is True, f"Expected success for {test_files[i]}, got: {result.errors}"
                    success_count += 1
                elif expected == "parse_failure":
                    assert result.success is False
                    assert len(result.errors) > 0
                    assert any("parsing" in error.lower() or "invalid" in error.lower() or "empty" in error.lower() 
                             for error in result.errors)
                    parse_failure_count += 1
                elif expected == "file_not_found":
                    assert result.success is False
                    assert len(result.errors) > 0
                    file_not_found_count += 1
                elif expected == "no_parser":
                    assert result.success is False
                    assert len(result.errors) > 0
                    assert any("no suitable parser" in error.lower() for error in result.errors)
                    no_parser_count += 1
            
            # Verify error categorization
            assert success_count >= 2, "Should have at least 2 successful loads"
            assert parse_failure_count >= 2, "Should have at least 2 parse failures"
            assert file_not_found_count >= 1, "Should have at least 1 file not found"
            assert no_parser_count >= 1, "Should have at least 1 no parser available"
            
            # Verify only successful ontologies were stored
            successful_ontologies = [r.ontology for r in results if r.success and r.ontology]
            assert len(ontology_manager.ontologies) == len(successful_ontologies)
            
            # Verify comprehensive statistics
            stats = ontology_manager.get_statistics()
            assert stats['total_loads'] == len(test_files)
            assert stats['successful_loads'] == success_count
            assert stats['failed_loads'] == len(test_files) - success_count

    def test_resource_cleanup_and_optimization(self, ontology_manager, temp_dir):
        """Test resource cleanup and memory optimization during multi-source loading."""
        # Create manager with limited cache to test eviction
        limited_manager = OntologyManager(enable_caching=True, cache_size_limit=3)
        
        # Create more files than cache limit
        file_paths = []
        ontologies = []
        
        for i in range(5):  # More than cache limit
            ontology = Ontology(
                id=f"CLEANUP:{i:03d}",
                name=f"Cleanup Test {i}",
                version="1.0",
                terms={f"TERM:{j+i*1000:06d}": Term(
                    id=f"TERM:{j+i*1000:06d}",
                    name=f"Term {j}",
                    definition=f"Definition {j}",
                    namespace=f"ns_{i}"
                ) for j in range(50)},  # 50 terms each
                relationships={},
                namespaces=[f"ns_{i}"]
            )
            ontologies.append(ontology)
            
            csv_file = temp_dir / f"cleanup_test_{i}.csv"
            csv_file.write_text(f"id,name,definition\nTEST:{i:03d},Test {i},Definition {i}")
            file_paths.append(str(csv_file))
        
        with patch('aim2_project.aim2_ontology.ontology_manager.auto_detect_parser') as mock_auto_detect:
            parsers = [
                Mock(format_name="csv", parse=Mock(return_value=ParseResult(
                    success=True, data=ont, errors=[], warnings=[], metadata={}, parse_time=0.1
                )))
                for ont in ontologies
            ]
            mock_auto_detect.side_effect = parsers
            
            # Load all files
            results = limited_manager.load_ontologies(file_paths)
            
            # Verify all loaded successfully
            assert all(r.success for r in results)
            assert len(limited_manager.ontologies) == 5
            
            # Verify cache was limited and eviction occurred
            assert len(limited_manager._cache) == limited_manager.cache_size_limit
            
            # Test explicit cache cleanup
            initial_cache_size = len(limited_manager._cache)
            limited_manager.clear_cache()
            assert len(limited_manager._cache) == 0
            
            # Test memory cleanup after removing ontologies
            initial_ontology_count = len(limited_manager.ontologies)
            removed_count = 0
            
            # Remove some ontologies
            for ontology_id in list(limited_manager.ontologies.keys())[:2]:
                if limited_manager.remove_ontology(ontology_id):
                    removed_count += 1
            
            assert len(limited_manager.ontologies) == initial_ontology_count - removed_count
            
            # Verify cache was also cleaned up
            remaining_cache_entries = len(limited_manager._cache)
            assert remaining_cache_entries == 0  # Already cleared above

    def test_load_balancing_and_prioritization(self, ontology_manager, temp_dir):
        """Test load balancing and prioritization strategies for multi-source operations."""
        # Create files with different sizes/complexity
        test_scenarios = [
            ("small", 10, 5),    # 10 terms, 5 relationships
            ("medium", 100, 25), # 100 terms, 25 relationships  
            ("large", 500, 100), # 500 terms, 100 relationships
            ("tiny", 2, 1),      # 2 terms, 1 relationship
            ("mediumtwo", 150, 30) # 150 terms, 30 relationships
        ]
        
        file_paths = []
        expected_ontologies = []
        
        for name, term_count, rel_count in test_scenarios:
            # Create ontology with specified complexity
            terms = {}
            for i in range(term_count):
                term_id = f"{name.upper()}:{i+1:06d}"
                terms[term_id] = Term(
                    id=term_id,
                    name=f"{name} Term {i}",
                    definition=f"Definition for {name} term {i}",
                    namespace=f"{name}_namespace"
                )
            
            relationships = {}
            for i in range(rel_count):
                rel_id = f"{name.upper()}REL:{i+1:06d}"
                relationships[rel_id] = Relationship(
                    id=rel_id,
                    subject=f"{name.upper()}:{i+1:06d}",
                    predicate="participates_in",
                    object=f"{name.upper()}:{((i+1)%term_count)+1:06d}",
                    confidence=0.9
                )
            
            ontology = Ontology(
                id=f"{name.upper()}PRIORITY:000001",
                name=f"{name.title()} Priority Test",
                version="1.0",
                terms=terms,
                relationships=relationships,
                namespaces=[f"{name}_namespace"]
            )
            expected_ontologies.append(ontology)
            
            # Create CSV file
            csv_lines = ["id,name,definition,namespace"]
            for term_id, term in terms.items():
                csv_lines.append(f"{term_id},{term.name},{term.definition},{term.namespace}")
            
            csv_file = temp_dir / f"priority_{name}.csv"
            csv_file.write_text("\n".join(csv_lines))
            file_paths.append(str(csv_file))
        
        with patch('aim2_project.aim2_ontology.ontology_manager.auto_detect_parser') as mock_auto_detect:
            # Create parsers with different processing times based on complexity
            parsers = []
            for i, (name, term_count, rel_count) in enumerate(test_scenarios):
                parser = Mock()
                parser.format_name = "csv"
                # Simulate longer processing time for larger ontologies
                processing_time = 0.1 + (term_count / 1000.0)
                parser.parse.return_value = ParseResult(
                    success=True,
                    data=expected_ontologies[i],
                    errors=[],
                    warnings=[],
                    metadata={"complexity": term_count + rel_count, "processing_time": processing_time},
                    parse_time=processing_time
                )
                parsers.append(parser)
            
            mock_auto_detect.side_effect = parsers
            
            # Test different loading strategies
            
            # Strategy 1: Load all at once (baseline)
            start_time = time.time()
            results_all = ontology_manager.load_ontologies(file_paths)
            total_time_all = time.time() - start_time
            
            assert all(r.success for r in results_all)
            assert len(results_all) == len(file_paths)
            
            # Strategy 2: Load in priority order (smallest first for faster feedback)
            ontology_manager.ontologies.clear()
            ontology_manager.clear_cache()
            
            # Sort by complexity (smallest first)
            complexity_sorted = sorted(enumerate(test_scenarios), key=lambda x: x[1][1] + x[1][2])
            priority_file_paths = [file_paths[i] for i, _ in complexity_sorted]
            
            # Reset mock for second test
            mock_auto_detect.side_effect = [parsers[i] for i, _ in complexity_sorted]
            
            start_time = time.time()
            results_priority = ontology_manager.load_ontologies(priority_file_paths)
            total_time_priority = time.time() - start_time
            
            assert all(r.success for r in results_priority)
            
            # Verify that smaller ontologies loaded first would give faster initial feedback
            # (This is more of a conceptual test - in practice, you might implement actual prioritization)
            
            # Strategy 3: Batch loading with size limits (simulate chunked processing)
            ontology_manager.ontologies.clear()
            ontology_manager.clear_cache()
            
            chunk_size = 2
            chunks = [file_paths[i:i+chunk_size] for i in range(0, len(file_paths), chunk_size)]
            
            # Reset mock for chunked test
            mock_auto_detect.side_effect = parsers * 2  # Ensure enough parsers for all chunks
            
            chunked_results = []
            start_time = time.time()
            for chunk in chunks:
                chunk_results = ontology_manager.load_ontologies(chunk)
                chunked_results.extend(chunk_results)
                # Small delay between chunks (simulate processing time)
                time.sleep(0.01)
            total_time_chunked = time.time() - start_time
            
            assert len(chunked_results) == len(file_paths)
            assert all(r.success for r in chunked_results)
            
            # Verify statistics for different strategies
            stats = ontology_manager.get_statistics()
            assert stats['total_loads'] >= len(file_paths)  # May be higher due to multiple strategies
            
            # All strategies should complete in reasonable time
            max_reasonable_time = 5.0
            assert total_time_all < max_reasonable_time
            assert total_time_priority < max_reasonable_time
            assert total_time_chunked < max_reasonable_time

    def test_error_recovery_and_resilience(self, ontology_manager, temp_dir):
        """Test error recovery and resilience during multi-source loading."""
        # Create a mix of files with different types of issues
        sources_and_behaviors = []
        
        # Good file that should always work
        good_file = temp_dir / "good.csv"
        good_file.write_text("id,name,definition\nGOOD:001,Good Term,Good Definition")
        sources_and_behaviors.append((str(good_file), "success"))
        
        # File that fails initially but succeeds on retry
        retry_file = temp_dir / "retry.csv"
        retry_file.write_text("id,name,definition\nRETRY:001,Retry Term,Retry Definition")
        sources_and_behaviors.append((str(retry_file), "retry_success"))
        
        # File that always fails
        bad_file = temp_dir / "bad.csv"
        bad_file.write_text("invalid,csv,content,without,proper,structure")
        sources_and_behaviors.append((str(bad_file), "always_fail"))
        
        # File with intermittent issues
        intermittent_file = temp_dir / "intermittent.csv"
        intermittent_file.write_text("id,name,definition\nINTER:001,Intermittent Term,Intermittent Definition")
        sources_and_behaviors.append((str(intermittent_file), "intermittent"))
        
        # File that succeeds with warnings
        warning_file = temp_dir / "warning.csv"
        warning_file.write_text("id,name,definition\nWARN:001,Warning Term,Warning Definition")
        sources_and_behaviors.append((str(warning_file), "success_with_warnings"))
        
        sources = [source for source, _ in sources_and_behaviors]
        
        with patch('aim2_project.aim2_ontology.ontology_manager.auto_detect_parser') as mock_auto_detect:
            call_counts = {}
            
            def mock_parser_with_behavior(file_path=None, **kwargs):
                file_path_str = str(file_path) if file_path else ""
                
                # Track call counts for retry logic
                if file_path_str not in call_counts:
                    call_counts[file_path_str] = 0
                call_counts[file_path_str] += 1
                
                parser = Mock()
                parser.format_name = "resilient_csv"
                
                if "good.csv" in file_path_str:
                    # Always succeeds
                    parser.parse.return_value = ParseResult(
                        success=True,
                        data=Ontology(id="GOOD:001", name="Good", version="1.0", terms={}, relationships={}, namespaces=[]),
                        errors=[], warnings=[], metadata={}, parse_time=0.1
                    )
                    return parser
                    
                elif "retry.csv" in file_path_str:
                    # Fails first time, succeeds second time
                    if call_counts[file_path_str] == 1:
                        parser.parse.return_value = ParseResult(
                            success=False, data=None,
                            errors=["Temporary parsing error"], warnings=[], metadata={}, parse_time=0.05
                        )
                    else:
                        parser.parse.return_value = ParseResult(
                            success=True,
                            data=Ontology(id="RETRY:001", name="Retry", version="1.0", terms={}, relationships={}, namespaces=[]),
                            errors=[], warnings=[], metadata={}, parse_time=0.1
                        )
                    return parser
                    
                elif "bad.csv" in file_path_str:
                    # Always fails
                    parser.parse.return_value = ParseResult(
                        success=False, data=None,
                        errors=["Persistent parsing error", "Invalid CSV structure"], 
                        warnings=[], metadata={}, parse_time=0.02
                    )
                    return parser
                    
                elif "intermittent.csv" in file_path_str:
                    # Succeeds every other time
                    if call_counts[file_path_str] % 2 == 1:
                        parser.parse.return_value = ParseResult(
                            success=False, data=None,
                            errors=["Intermittent network error"], warnings=[], metadata={}, parse_time=0.03
                        )
                    else:
                        parser.parse.return_value = ParseResult(
                            success=True,
                            data=Ontology(id="INTER:001", name="Intermittent", version="1.0", terms={}, relationships={}, namespaces=[]),
                            errors=[], warnings=[], metadata={}, parse_time=0.1
                        )
                    return parser
                    
                elif "warning.csv" in file_path_str:
                    # Succeeds but with warnings
                    parser.parse.return_value = ParseResult(
                        success=True,
                        data=Ontology(id="WARN:001", name="Warning", version="1.0", terms={}, relationships={}, namespaces=[]),
                        errors=[], 
                        warnings=["Deprecated field format", "Missing optional field"], 
                        metadata={}, parse_time=0.1
                    )
                    return parser
                    
                return None
            
            mock_auto_detect.side_effect = mock_parser_with_behavior
            
            # First attempt - some may fail
            results1 = ontology_manager.load_ontologies(sources)
            
            # Count outcomes
            successes1 = [r for r in results1 if r.success]
            failures1 = [r for r in results1 if not r.success]
            
            # Should have at least some successes and some failures
            assert len(successes1) >= 2, "Should have at least 2 successes on first attempt"
            assert len(failures1) >= 1, "Should have at least 1 failure on first attempt"
            
            # Retry failed sources (simulate retry logic)
            failed_sources = [r.source_path for r in failures1 if r.source_path]
            
            if failed_sources:
                # Reset mock call behavior for retry
                results_retry = ontology_manager.load_ontologies(failed_sources)
                
                # Some retries should succeed (retry.csv, intermittent.csv on even calls)
                retry_successes = [r for r in results_retry if r.success]
                assert len(retry_successes) >= 1, "At least some retries should succeed"
            
            # Verify resilience - manager should still be operational
            stats = ontology_manager.get_statistics()
            assert stats['total_loads'] > 0
            assert stats['successful_loads'] > 0
            assert len(ontology_manager.ontologies) >= 2  # At least good and warning files
            
            # Test warning handling
            warning_results = [r for r in results1 if r.success and r.warnings]
            assert len(warning_results) >= 1, "Should have at least one result with warnings"
            
            # Verify warning result details
            for result in warning_results:
                assert len(result.warnings) > 0
                assert result.ontology is not None  # Should still have loaded despite warnings

    def test_progressive_loading_feedback(self, ontology_manager, temp_dir):
        """Test progressive loading with feedback mechanisms."""
        # Create multiple files for progressive loading
        file_count = 8
        file_paths = []
        expected_ontologies = []
        
        for i in range(file_count):
            ontology = Ontology(
                id=f"PROGRESSIVE:{i:06d}",
                name=f"Progressive Ontology {i}",
                version="1.0",
                terms={f"PROG:{i*100+j:06d}": Term(
                    id=f"PROG:{i*100+j:06d}",
                    name=f"Progressive Term {j}",
                    definition=f"Definition {j}",
                    namespace=f"progressive_{i}"
                ) for j in range(20)},  # 20 terms each
                relationships={},
                namespaces=[f"progressive_{i}"]
            )
            expected_ontologies.append(ontology)
            
            csv_file = temp_dir / f"progressive_{i}.csv"
            csv_file.write_text(f"id,name,definition\nPROG:{i*100+1:06d},Progressive Term,Progressive Definition")
            file_paths.append(str(csv_file))
        
        # Track loading progress
        progress_callbacks = []
        
        def progress_callback(completed, total, current_file=None, result=None):
            progress_callbacks.append({
                'completed': completed,
                'total': total,
                'current_file': current_file,
                'result': result,
                'timestamp': time.time()
            })
        
        with patch('aim2_project.aim2_ontology.ontology_manager.auto_detect_parser') as mock_auto_detect:
            parsers = [
                Mock(format_name="csv", parse=Mock(return_value=ParseResult(
                    success=True, data=ont, errors=[], warnings=[], metadata={}, parse_time=0.1
                )))
                for ont in expected_ontologies
            ]
            mock_auto_detect.side_effect = parsers
            
            # Simulate progressive loading with callbacks
            # Note: This would require modification to OntologyManager to support callbacks
            # For this test, we'll simulate the concept
            
            results = []
            start_time = time.time()
            
            for i, file_path in enumerate(file_paths):
                # Simulate progress callback
                progress_callback(i, len(file_paths), file_path, None)
                
                # Load individual file
                result = ontology_manager.load_ontology(file_path)
                results.append(result)
                
                # Callback with result
                progress_callback(i + 1, len(file_paths), file_path, result)
                
                # Small delay to simulate processing time
                time.sleep(0.01)
            
            end_time = time.time()
            
            # Verify progressive loading worked
            assert len(results) == file_count
            assert all(r.success for r in results)
            
            # Verify progress tracking
            assert len(progress_callbacks) == file_count * 2  # Before and after each file
            
            # Check progress increments
            completion_callbacks = [cb for cb in progress_callbacks if cb['result'] is not None]
            assert len(completion_callbacks) == file_count
            
            for i, callback in enumerate(completion_callbacks):
                assert callback['completed'] == i + 1
                assert callback['total'] == file_count
                assert callback['result'].success is True
            
            # Verify timing progression
            timestamps = [cb['timestamp'] for cb in progress_callbacks]
            assert timestamps == sorted(timestamps), "Timestamps should be in chronological order"
            
            # Total time should be reasonable
            total_time = end_time - start_time
            assert total_time < 2.0, f"Progressive loading took too long: {total_time:.2f}s"
            
            # Verify all ontologies were loaded
            assert len(ontology_manager.ontologies) == file_count
            
            # Verify statistics reflect progressive loading
            stats = ontology_manager.get_statistics()
            assert stats['total_loads'] == file_count
            assert stats['successful_loads'] == file_count
            assert stats['failed_loads'] == 0


if __name__ == "__main__":
    pytest.main([__file__])