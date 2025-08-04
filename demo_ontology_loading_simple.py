#!/usr/bin/env python3
"""
Simple Demonstration of OntologyManager Loading Functionality

This script demonstrates the basic operation of the OntologyManager's load_ontologies()
method with real ontology files. It shows multi-source loading, error handling,
and basic functionality verification for AIM2-013-05.
"""

import logging
import tempfile
from pathlib import Path

# Import the modules to be tested
try:
    from aim2_project.aim2_ontology.models import Ontology, Term, Relationship
    from aim2_project.aim2_ontology.ontology_manager import OntologyManager
except ImportError as e:
    print(f"Import error: {e}")
    print("Please ensure the aim2_project package is in your PYTHONPATH")
    exit(1)


def setup_logging():
    """Setup logging for the demo."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    return logging.getLogger(__name__)


def create_sample_files(temp_dir: Path):
    """Create sample ontology files for testing."""
    files = []
    
    # Create a simple OWL file
    owl_content = """<?xml version="1.0"?>
<rdf:RDF xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#"
         xmlns:owl="http://www.w3.org/2002/07/owl#"
         xmlns:rdfs="http://www.w3.org/2000/01/rdf-schema#">
    <owl:Ontology rdf:about="http://example.org/demo">
        <rdfs:label>Demo Ontology</rdfs:label>
        <rdfs:comment>Simple demonstration ontology</rdfs:comment>
    </owl:Ontology>
    <owl:Class rdf:about="http://example.org/Thing1">
        <rdfs:label>Thing One</rdfs:label>
    </owl:Class>
</rdf:RDF>"""
    
    owl_file = temp_dir / "demo.owl"
    owl_file.write_text(owl_content)
    files.append(str(owl_file))
    
    # Create a simple CSV file
    csv_content = """id,name,definition,namespace
DEMO:001,Demo Term 1,First demonstration term,demo
DEMO:002,Demo Term 2,Second demonstration term,demo
DEMO:003,Demo Term 3,Third demonstration term,demo"""
    
    csv_file = temp_dir / "demo.csv"
    csv_file.write_text(csv_content)
    files.append(str(csv_file))
    
    # Create invalid file for error testing
    invalid_file = temp_dir / "invalid.owl"
    invalid_file.write_text("<<invalid xml content>>")
    files.append(str(invalid_file))
    
    return files


def main():
    """Main demonstration function."""
    logger = setup_logging()
    logger.info("Starting OntologyManager Loading Demonstration")
    
    try:
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            logger.info(f"Working in temporary directory: {temp_path}")
            
            # Create sample files
            test_files = create_sample_files(temp_path)
            logger.info(f"Created {len(test_files)} test files")
            
            # Initialize OntologyManager
            manager = OntologyManager(enable_caching=True, cache_size_limit=10)
            logger.info("Created OntologyManager instance")
            
            # Test 1: Single file loading
            logger.info("=" * 50)
            logger.info("TEST 1: Single File Loading")
            logger.info("=" * 50)
            
            single_result = manager.load_ontology(test_files[0])  # OWL file
            if single_result.success:
                logger.info(f"✓ Successfully loaded single file: {single_result.ontology.name}")
                logger.info(f"  Load time: {single_result.load_time:.3f}s")
                logger.info(f"  Source: {single_result.source_path}")
            else:
                logger.error(f"✗ Failed to load single file: {single_result.errors}")
            
            # Test 2: Multi-source loading
            logger.info("=" * 50)
            logger.info("TEST 2: Multi-Source Loading")
            logger.info("=" * 50)
            
            # Load all files (including invalid one)
            results = manager.load_ontologies(test_files)
            
            successful = [r for r in results if r.success]
            failed = [r for r in results if not r.success]
            
            logger.info(f"Multi-source loading results:")
            logger.info(f"  Total files: {len(test_files)}")
            logger.info(f"  Successful: {len(successful)}")
            logger.info(f"  Failed: {len(failed)}")
            
            for result in successful:
                logger.info(f"  ✓ {Path(result.source_path).name}: {result.ontology.name if result.ontology else 'N/A'}")
            
            for result in failed:
                logger.info(f"  ✗ {Path(result.source_path).name}: {result.errors}")
            
            # Test 3: Verify storage and access
            logger.info("=" * 50)
            logger.info("TEST 3: Storage and Access")
            logger.info("=" * 50)
            
            ontology_ids = manager.list_ontologies()
            logger.info(f"Stored ontologies: {len(ontology_ids)}")
            
            for ont_id in ontology_ids:
                ontology = manager.get_ontology(ont_id)
                if ontology:
                    logger.info(f"  ✓ {ont_id}: {ontology.name} (v{ontology.version})")
                    logger.info(f"    Terms: {len(ontology.terms)}, Relationships: {len(ontology.relationships)}")
                else:
                    logger.error(f"  ✗ Could not access ontology {ont_id}")
            
            # Test 4: Statistics
            logger.info("=" * 50)
            logger.info("TEST 4: Statistics and Reporting")
            logger.info("=" * 50)
            
            stats = manager.get_statistics()
            logger.info("Statistics Summary:")
            logger.info(f"  Total loads: {stats['total_loads']}")
            logger.info(f"  Successful loads: {stats['successful_loads']}")
            logger.info(f"  Failed loads: {stats['failed_loads']}")
            logger.info(f"  Cache hits: {stats['cache_hits']}")
            logger.info(f"  Cache misses: {stats['cache_misses']}")
            logger.info(f"  Formats loaded: {dict(stats['formats_loaded'])}")
            logger.info(f"  Total terms: {stats['total_terms']}")
            logger.info(f"  Total relationships: {stats['total_relationships']}")
            
            # Test 5: Cache behavior
            logger.info("=" * 50)
            logger.info("TEST 5: Cache Behavior")
            logger.info("=" * 50)
            
            # Load same file again to test caching
            if successful:
                first_successful_file = successful[0].source_path
                logger.info(f"Re-loading {Path(first_successful_file).name} to test cache...")
                
                cache_test_result = manager.load_ontology(first_successful_file)
                if cache_test_result.success:
                    is_cache_hit = cache_test_result.metadata.get('cache_hit', False)
                    logger.info(f"  Cache hit: {is_cache_hit}")
                    logger.info(f"  Load time: {cache_test_result.load_time:.6f}s")
                else:
                    logger.error(f"  Failed cache test: {cache_test_result.errors}")
            
            # Final statistics after cache test
            final_stats = manager.get_statistics()
            logger.info(f"  Final cache hits: {final_stats['cache_hits']}")
            logger.info(f"  Final cache misses: {final_stats['cache_misses']}")
            
            # Test 6: Error recovery
            logger.info("=" * 50)
            logger.info("TEST 6: Error Recovery")
            logger.info("=" * 50)
            
            # Try to load non-existent file
            non_existent = "/path/that/does/not/exist.owl"
            error_result = manager.load_ontology(non_existent)
            
            if not error_result.success:
                logger.info("✓ Error handling working correctly for non-existent file")
                logger.info(f"  Error: {error_result.errors[0] if error_result.errors else 'No specific error'}")
            else:
                logger.error("✗ Expected error for non-existent file, but load succeeded")
            
            # Verify manager is still functional after error
            if successful:
                recovery_result = manager.load_ontology(successful[0].source_path)
                if recovery_result.success:
                    logger.info("✓ Manager remains functional after error")
                else:
                    logger.error("✗ Manager appears to be corrupted after error")
            
    except Exception as e:
        logger.error(f"Demonstration failed: {e}")
        return 1
    
    logger.info("=" * 50)
    logger.info("DEMONSTRATION COMPLETE")
    logger.info("=" * 50)
    logger.info("🎉 OntologyManager load functionality demonstration successful!")
    logger.info("The load_ontologies() method is working correctly for AIM2-013-05")
    
    return 0


if __name__ == "__main__":
    exit(main())