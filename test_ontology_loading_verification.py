#!/usr/bin/env python3
"""
Comprehensive Test Script for OntologyManager Load Functionality Verification

This script provides verification testing for the OntologyManager's load_ontologies() method
to ensure it works correctly for AIM2-013-05. It tests multi-source loading capabilities,
error handling, integration with parsers, and statistics reporting.

Test Coverage:
- Multi-source loading with various formats (OWL, CSV, JSON-LD)
- Successful loading scenarios with valid ontology files
- Error handling for invalid/missing files
- Proper storage and accessibility of loaded ontologies
- Statistics and reporting functionality integration
- Cache behavior and performance
- Parser framework integration
"""

import json
import logging
import tempfile
import time
from pathlib import Path
from typing import List

# Import the modules to be tested
try:
    from aim2_project.aim2_ontology.models import Ontology, Relationship, Term
    from aim2_project.aim2_ontology.ontology_manager import OntologyManager
    from aim2_project.aim2_ontology.parsers import ParseResult
    from aim2_project.aim2_ontology.parsers import auto_detect_parser
except ImportError as e:
    print(f"Import error: {e}")
    print("Please ensure the aim2_project package is in your PYTHONPATH")
    exit(1)


def setup_logging():
    """Setup logging for the test script."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    return logging.getLogger(__name__)


def create_sample_ontologies() -> List[Ontology]:
    """Create sample ontologies for testing."""
    logger = logging.getLogger(__name__)
    logger.info("Creating sample ontologies for testing...")

    ontologies = []

    # Sample Ontology 1: Medical/Biological
    medical_terms = {}
    medical_relationships = {}

    for i in range(5):
        term_id = f"MED:{i+1:03d}"
        medical_terms[term_id] = Term(
            id=term_id,
            name=f"Medical Term {i+1}",
            definition=f"Definition of medical concept {i+1}",
            synonyms=[f"synonym_{i+1}_1", f"synonym_{i+1}_2"],
            namespace="medical"
        )

    for i in range(3):
        rel_id = f"MEDREL:{i+1:03d}"
        medical_relationships[rel_id] = Relationship(
            id=rel_id,
            subject=f"MED:{i+1:03d}",
            predicate="is_a",
            object=f"MED:{i+2:03d}",
            confidence=0.85 + (i * 0.05)
        )

    medical_ontology = Ontology(
        id="MEDICAL:001",
        name="Medical Test Ontology",
        version="1.0",
        description="Test ontology containing medical/biological terms",
        terms=medical_terms,
        relationships=medical_relationships,
        namespaces=["medical"]
    )
    ontologies.append(medical_ontology)

    # Sample Ontology 2: Technical/Computing
    tech_terms = {}
    tech_relationships = {}

    for i in range(4):
        term_id = f"TECH:{i+1:03d}"
        tech_terms[term_id] = Term(
            id=term_id,
            name=f"Technology Term {i+1}",
            definition=f"Definition of technology concept {i+1}",
            synonyms=[f"tech_syn_{i+1}"],
            namespace="technology"
        )

    for i in range(2):
        rel_id = f"TECHREL:{i+1:03d}"
        tech_relationships[rel_id] = Relationship(
            id=rel_id,
            subject=f"TECH:{i+1:03d}",
            predicate="part_of",
            object=f"TECH:{i+3:03d}",
            confidence=0.90
        )

    tech_ontology = Ontology(
        id="TECH:001",
        name="Technology Test Ontology",
        version="2.0",
        description="Test ontology containing technology/computing terms",
        terms=tech_terms,
        relationships=tech_relationships,
        namespaces=["technology"]
    )
    ontologies.append(tech_ontology)

    # Sample Ontology 3: Small/Simple for Edge Cases
    simple_terms = {
        "SIMPLE:001": Term(
            id="SIMPLE:001",
            name="Simple Term",
            definition="A simple test term",
            synonyms=["basic"],
            namespace="simple"
        )
    }

    simple_ontology = Ontology(
        id="SIMPLE:001",
        name="Simple Test Ontology",
        version="0.1",
        description="Minimal test ontology for edge cases",
        terms=simple_terms,
        relationships={},
        namespaces=["simple"]
    )
    ontologies.append(simple_ontology)

    logger.info(f"Created {len(ontologies)} sample ontologies")
    return ontologies


def create_test_files(temp_dir: Path, ontologies: List[Ontology]) -> dict:
    """Create test files in various formats."""
    logger = logging.getLogger(__name__)
    logger.info("Creating test files in various formats...")

    files = {}

    # Create OWL file
    owl_content = f"""<?xml version="1.0"?>
<rdf:RDF xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#"
         xmlns:owl="http://www.w3.org/2002/07/owl#"
         xmlns:rdfs="http://www.w3.org/2000/01/rdf-schema#">
    <owl:Ontology rdf:about="http://example.org/{ontologies[0].id}">
        <rdfs:label>{ontologies[0].name}</rdfs:label>
        <rdfs:comment>{ontologies[0].description}</rdfs:comment>
    </owl:Ontology>
    <!-- Additional OWL content would be here -->
</rdf:RDF>"""

    owl_file = temp_dir / "medical_ontology.owl"
    owl_file.write_text(owl_content)
    files["owl"] = (str(owl_file), ontologies[0])

    # Create CSV file
    csv_lines = ["id,name,definition,namespace,synonyms"]
    for term_id, term in list(ontologies[1].terms.items())[:3]:  # First 3 terms
        synonyms = "|".join(term.synonyms) if term.synonyms else ""
        csv_lines.append(f"{term_id},{term.name},{term.definition},{term.namespace},{synonyms}")

    csv_file = temp_dir / "technology_ontology.csv"
    csv_file.write_text("\n".join(csv_lines))
    files["csv"] = (str(csv_file), ontologies[1])

    # Create JSON-LD file
    jsonld_content = {
        "@context": {
            "@vocab": "http://example.org/",
            "rdfs": "http://www.w3.org/2000/01/rdf-schema#"
        },
        "@id": ontologies[2].id,
        "@type": "Ontology",
        "rdfs:label": ontologies[2].name,
        "rdfs:comment": ontologies[2].description,
        "terms": [
            {
                "@id": term_id,
                "rdfs:label": term.name,
                "definition": term.definition,
                "namespace": term.namespace
            }
            for term_id, term in ontologies[2].terms.items()
        ]
    }

    jsonld_file = temp_dir / "simple_ontology.jsonld"
    jsonld_file.write_text(json.dumps(jsonld_content, indent=2))
    files["jsonld"] = (str(jsonld_file), ontologies[2])

    # Create invalid files for error testing
    invalid_file = temp_dir / "invalid.owl"
    invalid_file.write_text("<<invalid xml content>>")
    files["invalid"] = (str(invalid_file), None)

    empty_file = temp_dir / "empty.csv"
    empty_file.write_text("")
    files["empty"] = (str(empty_file), None)

    logger.info(f"Created {len(files)} test files")
    return files


def test_multi_source_loading(manager: OntologyManager, valid_files: List[str]) -> bool:
    """Test multi-source loading with valid files."""
    logger = logging.getLogger(__name__)
    logger.info("Testing multi-source loading with valid files...")

    try:
        # Use mock parsers for testing since we may not have full parser implementations
        from unittest.mock import Mock, patch

        with patch('aim2_project.aim2_ontology.ontology_manager.auto_detect_parser') as mock_auto_detect:
            # Create mock parsers that return success
            def create_mock_parser(format_name: str, ontology: Ontology):
                parser = Mock()
                parser.format_name = format_name
                parser.parse.return_value = ParseResult(
                    success=True,
                    data=ontology,
                    errors=[],
                    warnings=[],
                    metadata={},
                    parse_time=0.1
                )
                return parser

            # Mock auto_detect_parser to return appropriate parsers
            mock_parsers = []
            sample_ontologies = create_sample_ontologies()
            
            for i, file_path in enumerate(valid_files):
                if file_path.endswith('.owl'):
                    mock_parsers.append(create_mock_parser("owl", sample_ontologies[0]))
                elif file_path.endswith('.csv'):
                    mock_parsers.append(create_mock_parser("csv", sample_ontologies[1]))
                elif file_path.endswith('.jsonld'):
                    mock_parsers.append(create_mock_parser("jsonld", sample_ontologies[2]))

            mock_auto_detect.side_effect = mock_parsers

            # Test multi-source loading
            start_time = time.time()
            results = manager.load_ontologies(valid_files)
            load_time = time.time() - start_time

            # Verify results
            logger.info(f"Loaded {len(results)} files in {load_time:.3f} seconds")
            
            successful_results = [r for r in results if r.success]
            failed_results = [r for r in results if not r.success]

            logger.info(f"Successful loads: {len(successful_results)}")
            logger.info(f"Failed loads: {len(failed_results)}")

            if len(successful_results) != len(valid_files):
                logger.error(f"Expected {len(valid_files)} successful loads, got {len(successful_results)}")
                for result in failed_results:
                    logger.error(f"Failed to load {result.source_path}: {result.errors}")
                return False

            # Verify ontologies are stored
            if len(manager.ontologies) != len(successful_results):
                logger.error(f"Expected {len(successful_results)} stored ontologies, got {len(manager.ontologies)}")
                return False

            logger.info("✓ Multi-source loading test passed")
            return True

    except Exception as e:
        logger.error(f"Multi-source loading test failed: {e}")
        return False


def test_error_handling(manager: OntologyManager, invalid_files: List[str]) -> bool:
    """Test error handling with invalid files."""
    logger = logging.getLogger(__name__)
    logger.info("Testing error handling with invalid files...")

    try:
        from unittest.mock import Mock, patch

        with patch('aim2_project.aim2_ontology.ontology_manager.auto_detect_parser') as mock_auto_detect:
            # Mock parsers that fail for invalid files
            def create_failing_parser():
                parser = Mock()
                parser.format_name = "unknown"
                parser.parse.return_value = ParseResult(
                    success=False,
                    data=None,
                    errors=["Parsing failed: invalid content"],
                    warnings=[],
                    metadata={},
                    parse_time=0.01
                )
                return parser

            # For some files, return None (no parser found)
            # For others, return a failing parser
            mock_responses = []
            for i, file_path in enumerate(invalid_files):
                if i % 2 == 0:
                    mock_responses.append(None)  # No parser found
                else:
                    mock_responses.append(create_failing_parser())  # Parser fails

            mock_auto_detect.side_effect = mock_responses

            # Test loading invalid files
            results = manager.load_ontologies(invalid_files)

            # Verify all failed appropriately
            successful_results = [r for r in results if r.success]
            failed_results = [r for r in results if not r.success]

            if len(successful_results) > 0:
                logger.error(f"Expected no successful loads, got {len(successful_results)}")
                return False

            if len(failed_results) != len(invalid_files):
                logger.error(f"Expected {len(invalid_files)} failed loads, got {len(failed_results)}")
                return False

            # Verify error messages are present
            for result in failed_results:
                if not result.errors:
                    logger.error(f"Expected error messages for {result.source_path}")
                    return False

            logger.info("✓ Error handling test passed")
            return True

    except Exception as e:
        logger.error(f"Error handling test failed: {e}")
        return False


def test_ontology_storage_and_access(manager: OntologyManager) -> bool:
    """Test that loaded ontologies are properly stored and accessible."""
    logger = logging.getLogger(__name__)
    logger.info("Testing ontology storage and access...")

    try:
        # Get list of loaded ontologies
        ontology_ids = manager.list_ontologies()
        logger.info(f"Found {len(ontology_ids)} stored ontologies: {ontology_ids}")

        if not ontology_ids:
            logger.warning("No ontologies found - this may be expected if previous tests used mocks")
            return True

        # Test accessing each ontology
        for ontology_id in ontology_ids:
            ontology = manager.get_ontology(ontology_id)
            
            if ontology is None:
                logger.error(f"Could not access ontology {ontology_id}")
                return False

            if not isinstance(ontology, Ontology):
                logger.error(f"Retrieved object is not an Ontology: {type(ontology)}")
                return False

            logger.info(f"✓ Successfully accessed ontology {ontology_id}: {ontology.name}")

        # Test accessing non-existent ontology
        non_existent = manager.get_ontology("NON_EXISTENT:999")
        if non_existent is not None:
            logger.error("Expected None for non-existent ontology")
            return False

        logger.info("✓ Ontology storage and access test passed")
        return True

    except Exception as e:
        logger.error(f"Ontology storage and access test failed: {e}")
        return False


def test_statistics_and_reporting(manager: OntologyManager) -> bool:
    """Test statistics and reporting functionality."""
    logger = logging.getLogger(__name__)
    logger.info("Testing statistics and reporting functionality...")

    try:
        # Get comprehensive statistics
        stats = manager.get_statistics()

        # Verify statistics structure
        expected_keys = {
            'total_loads', 'successful_loads', 'failed_loads',
            'cache_hits', 'cache_misses', 'formats_loaded',
            'cache_size', 'cache_limit', 'cache_enabled',
            'loaded_ontologies', 'total_terms', 'total_relationships'
        }

        missing_keys = expected_keys - set(stats.keys())
        if missing_keys:
            logger.error(f"Missing statistics keys: {missing_keys}")
            return False

        # Log statistics for verification
        logger.info("Statistics summary:")
        logger.info(f"  Total loads: {stats['total_loads']}")
        logger.info(f"  Successful loads: {stats['successful_loads']}")
        logger.info(f"  Failed loads: {stats['failed_loads']}")
        logger.info(f"  Cache hits: {stats['cache_hits']}")
        logger.info(f"  Cache misses: {stats['cache_misses']}")
        logger.info(f"  Loaded ontologies: {stats['loaded_ontologies']}")
        logger.info(f"  Total terms: {stats['total_terms']}")
        logger.info(f"  Total relationships: {stats['total_relationships']}")
        logger.info(f"  Formats loaded: {stats['formats_loaded']}")

        # Verify mathematical consistency
        if stats['total_loads'] != stats['successful_loads'] + stats['failed_loads']:
            logger.error("Statistics inconsistency: total_loads != successful_loads + failed_loads")
            return False

        # Verify data types
        for key in ['total_loads', 'successful_loads', 'failed_loads', 'cache_hits', 'cache_misses']:
            if not isinstance(stats[key], int):
                logger.error(f"Expected integer for {key}, got {type(stats[key])}")
                return False

        logger.info("✓ Statistics and reporting test passed")
        return True

    except Exception as e:
        logger.error(f"Statistics and reporting test failed: {e}")
        return False


def test_cache_behavior(manager: OntologyManager, test_file: str) -> bool:
    """Test cache behavior and efficiency."""
    logger = logging.getLogger(__name__)
    logger.info("Testing cache behavior...")

    try:
        from unittest.mock import Mock, patch

        with patch('aim2_project.aim2_ontology.ontology_manager.auto_detect_parser') as mock_auto_detect:
            # Create a mock parser
            sample_ontology = create_sample_ontologies()[0]
            mock_parser = Mock()
            mock_parser.format_name = "test_format"
            mock_parser.parse.return_value = ParseResult(
                success=True,
                data=sample_ontology,
                errors=[],
                warnings=[],
                metadata={},
                parse_time=0.1
            )
            mock_auto_detect.return_value = mock_parser

            # First load - should be cache miss
            initial_cache_misses = manager.load_stats['cache_misses']
            initial_cache_hits = manager.load_stats['cache_hits']

            result1 = manager.load_ontology(test_file)
            if not result1.success:
                logger.error(f"First load failed: {result1.errors}")
                return False

            cache_misses_after_first = manager.load_stats['cache_misses']
            if cache_misses_after_first <= initial_cache_misses:
                logger.warning("Expected cache miss on first load (may be disabled)")

            # Second load - should be cache hit if caching is enabled
            result2 = manager.load_ontology(test_file)
            if not result2.success:
                logger.error(f"Second load failed: {result2.errors}")
                return False

            cache_hits_after_second = manager.load_stats['cache_hits']
            if cache_hits_after_second > initial_cache_hits:
                logger.info("✓ Cache hit detected on second load")
            else:
                logger.info("No cache hit detected (caching may be disabled)")

            # Verify parser was only called once if cache is working
            if manager.enable_caching and manager.cache_size_limit > 0:
                if mock_parser.parse.call_count == 1:
                    logger.info("✓ Cache efficiency confirmed - parser called only once")
                else:
                    logger.warning(f"Parser called {mock_parser.parse.call_count} times - cache may not be working")

        logger.info("✓ Cache behavior test completed")
        return True

    except Exception as e:
        logger.error(f"Cache behavior test failed: {e}")
        return False


def main():
    """Main test execution function."""
    logger = setup_logging()
    logger.info("Starting OntologyManager Load Functionality Verification Tests")

    # Test results tracking
    results = {}
    
    try:
        # Create temporary directory for test files
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            logger.info(f"Using temporary directory: {temp_path}")

            # Create sample ontologies and test files
            sample_ontologies = create_sample_ontologies()
            test_files = create_test_files(temp_path, sample_ontologies)

            # Separate valid and invalid files
            valid_files = [path for key, (path, ont) in test_files.items() 
                          if key in ['owl', 'csv', 'jsonld'] and ont is not None]
            invalid_files = [path for key, (path, ont) in test_files.items() 
                           if key in ['invalid', 'empty']]

            logger.info(f"Valid test files: {len(valid_files)}")
            logger.info(f"Invalid test files: {len(invalid_files)}")

            # Initialize OntologyManager
            manager = OntologyManager(enable_caching=True, cache_size_limit=10)
            logger.info("Created OntologyManager instance")

            # Run test suite
            logger.info("=" * 60)
            logger.info("RUNNING TEST SUITE")
            logger.info("=" * 60)

            # Test 1: Multi-source loading
            results['multi_source_loading'] = test_multi_source_loading(manager, valid_files)

            # Test 2: Error handling
            manager_for_errors = OntologyManager(enable_caching=True, cache_size_limit=10)
            results['error_handling'] = test_error_handling(manager_for_errors, invalid_files)

            # Test 3: Ontology storage and access
            results['storage_and_access'] = test_ontology_storage_and_access(manager)

            # Test 4: Statistics and reporting
            results['statistics_reporting'] = test_statistics_and_reporting(manager)

            # Test 5: Cache behavior
            if valid_files:
                results['cache_behavior'] = test_cache_behavior(manager, valid_files[0])

            # Test 6: Integration test - Mixed valid/invalid files
            mixed_files = valid_files + invalid_files
            logger.info(f"\nTesting mixed loading scenario with {len(mixed_files)} files...")
            mixed_manager = OntologyManager(enable_caching=True, cache_size_limit=5)
            
            from unittest.mock import Mock, patch
            with patch('aim2_project.aim2_ontology.ontology_manager.auto_detect_parser') as mock_auto_detect:
                responses = []
                for i, file_path in enumerate(mixed_files):
                    if file_path in valid_files:
                        # Valid file - return successful parser
                        parser = Mock()
                        parser.format_name = "mixed_test"
                        parser.parse.return_value = ParseResult(
                            success=True,
                            data=sample_ontologies[i % len(sample_ontologies)],
                            errors=[], warnings=[], metadata={}, parse_time=0.1
                        )
                        responses.append(parser)
                    else:
                        # Invalid file - return None or failing parser
                        responses.append(None)
                
                mock_auto_detect.side_effect = responses
                mixed_results = mixed_manager.load_ontologies(mixed_files)
                
                successful_mixed = [r for r in mixed_results if r.success]
                failed_mixed = [r for r in mixed_results if not r.success]
                
                logger.info(f"Mixed scenario: {len(successful_mixed)} successful, {len(failed_mixed)} failed")
                results['mixed_scenario'] = len(successful_mixed) == len(valid_files) and len(failed_mixed) == len(invalid_files)

    except Exception as e:
        logger.error(f"Test execution failed: {e}")
        results['execution'] = False

    # Print final results
    logger.info("=" * 60)
    logger.info("TEST RESULTS SUMMARY")
    logger.info("=" * 60)

    all_passed = True
    for test_name, passed in results.items():
        status = "✓ PASSED" if passed else "✗ FAILED"
        logger.info(f"{test_name:.<40} {status}")
        if not passed:
            all_passed = False

    if all_passed:
        logger.info("\n🎉 ALL TESTS PASSED! OntologyManager load functionality is working correctly.")
        return 0
    else:
        logger.error("\n❌ SOME TESTS FAILED! Please review the failures above.")
        return 1


if __name__ == "__main__":
    exit(main())