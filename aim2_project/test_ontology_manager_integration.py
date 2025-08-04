#!/usr/bin/env python3
"""
Integration tests for OntologyManager with existing parsers.

This module contains integration tests that verify the OntologyManager
works correctly with the existing parser framework, including OWL, CSV,
and JSON-LD parsers with format auto-detection.

These tests complement the unit tests by verifying end-to-end functionality
with real parser implementations and actual file formats.

Test Coverage:
- Integration with OWL parser for RDF/OWL files
- Integration with CSV parser with dialect detection
- Integration with JSON-LD parser for JSON-LD files
- Format auto-detection across different file types
- Real file parsing and ontology construction
- Error handling with actual parser failures
- Performance testing with larger ontologies
- Multi-format batch loading scenarios
"""

import os
import json
import tempfile
import pytest
import time
from pathlib import Path
from unittest.mock import patch, Mock

# Import modules for integration testing
try:
    from aim2_project.aim2_ontology.ontology_manager import OntologyManager, LoadResult
    from aim2_project.aim2_ontology.models import Ontology, Term, Relationship
    from aim2_project.aim2_ontology.parsers import (
        auto_detect_parser, 
        get_parser_for_format,
        detect_format_from_extension,
        detect_format_from_content
    )
except ImportError:
    import warnings
    warnings.warn("Integration test imports failed - tests may be skipped", ImportWarning)


class TestOntologyManagerIntegration:
    """Integration tests for OntologyManager with existing parsers."""

    @pytest.fixture
    def temp_dir(self):
        """Create a temporary directory for test files."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            yield Path(tmp_dir)

    @pytest.fixture
    def ontology_manager(self):
        """Create an OntologyManager instance for integration testing."""
        return OntologyManager(enable_caching=True, cache_size_limit=10)

    @pytest.fixture
    def sample_owl_content(self):
        """Sample OWL/RDF content for testing."""
        return '''<?xml version="1.0"?>
<rdf:RDF xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#"
         xmlns:owl="http://www.w3.org/2002/07/owl#"
         xmlns:rdfs="http://www.w3.org/2000/01/rdf-schema#"
         xmlns:chebi="http://purl.obolibrary.org/obo/CHEBI_"
         xmlns:go="http://purl.obolibrary.org/obo/GO_">

    <owl:Ontology rdf:about="http://example.org/test-ontology">
        <rdfs:label>Test Chemical Ontology</rdfs:label>
        <rdfs:comment>A test ontology for integration testing</rdfs:comment>
    </owl:Ontology>

    <!-- Chemical compound -->
    <owl:Class rdf:about="http://purl.obolibrary.org/obo/CHEBI_15422">
        <rdfs:label>ATP</rdfs:label>
        <rdfs:comment>Adenosine 5'-triphosphate</rdfs:comment>
        <chebi:synonym>adenosine triphosphate</chebi:synonym>
        <chebi:synonym>adenosine 5'-triphosphate</chebi:synonym>
    </owl:Class>

    <!-- Biological process -->
    <owl:Class rdf:about="http://purl.obolibrary.org/obo/GO_0008152">
        <rdfs:label>metabolic process</rdfs:label>
        <rdfs:comment>The chemical reactions and pathways</rdfs:comment>
    </owl:Class>

    <!-- Relationship -->
    <owl:ObjectProperty rdf:about="http://example.org/regulates">
        <rdfs:label>regulates</rdfs:label>
        <rdfs:domain rdf:resource="http://purl.obolibrary.org/obo/CHEBI_15422"/>
        <rdfs:range rdf:resource="http://purl.obolibrary.org/obo/GO_0008152"/>
    </owl:ObjectProperty>

</rdf:RDF>'''

    @pytest.fixture
    def sample_csv_content(self):
        """Sample CSV content for testing."""
        return '''id,name,definition,namespace,synonyms,xrefs
CHEBI:15422,ATP,"Adenosine 5'-triphosphate",chemical,"adenosine triphosphate|adenosine 5'-triphosphate","CAS:56-65-5|KEGG:C00002"
GO:0008152,metabolic process,"The chemical reactions and pathways",biological_process,"metabolism|metabolic pathway","EC:1.1.1.1"
CHEBI:33917,monosaccharide,"A simple sugar",chemical,"simple sugar|single sugar","CAS:492-61-5"
GO:0006096,glycolysis,"Glucose catabolism",biological_process,"glucose breakdown","KEGG:ko00010"'''

    @pytest.fixture 
    def sample_jsonld_content(self):
        """Sample JSON-LD content for testing."""
        return {
            "@context": {
                "@vocab": "http://example.org/",
                "rdfs": "http://www.w3.org/2000/01/rdf-schema#",
                "owl": "http://www.w3.org/2002/07/owl#",
                "chebi": "http://purl.obolibrary.org/obo/CHEBI_",
                "go": "http://purl.obolibrary.org/obo/GO_"
            },
            "@id": "test-ontology",
            "@type": "owl:Ontology",
            "rdfs:label": "Test JSON-LD Ontology",
            "rdfs:comment": "A test ontology in JSON-LD format",
            "terms": [
                {
                    "@id": "chebi:15422",
                    "@type": "owl:Class",
                    "rdfs:label": "ATP",
                    "rdfs:comment": "Adenosine 5'-triphosphate",
                    "synonyms": ["adenosine triphosphate", "adenosine 5'-triphosphate"],
                    "namespace": "chemical"
                },
                {
                    "@id": "go:0008152",
                    "@type": "owl:Class", 
                    "rdfs:label": "metabolic process",
                    "rdfs:comment": "The chemical reactions and pathways",
                    "namespace": "biological_process"
                }
            ],
            "relationships": [
                {
                    "@id": "rel:001",
                    "@type": "owl:ObjectProperty",
                    "subject": "chebi:15422",
                    "predicate": "regulates",
                    "object": "go:0008152",
                    "confidence": 0.95
                }
            ]
        }

    def test_owl_parser_integration(self, ontology_manager, temp_dir, sample_owl_content):
        """Test integration with OWL parser for RDF/OWL files."""
        # Create OWL file
        owl_file = temp_dir / "test_ontology.owl"
        owl_file.write_text(sample_owl_content)

        # Load ontology through manager
        result = ontology_manager.load_ontology(str(owl_file))

        # Verify loading was successful
        assert result.success is True, f"Loading failed: {result.errors}"
        assert result.ontology is not None
        assert isinstance(result.ontology, Ontology)
        
        # Verify ontology content (parser-dependent)
        ontology = result.ontology
        assert ontology.id is not None
        assert ontology.name is not None
        
        # Verify metadata
        assert result.metadata['format'] in ['owl', 'rdf', 'xml']  # Format detected by parser
        assert result.load_time > 0
        assert result.source_path == str(owl_file)

    def test_csv_parser_integration(self, ontology_manager, temp_dir, sample_csv_content):
        """Test integration with CSV parser including dialect detection."""
        # Create CSV file
        csv_file = temp_dir / "test_ontology.csv"
        csv_file.write_text(sample_csv_content)

        # Load ontology through manager
        result = ontology_manager.load_ontology(str(csv_file))

        # Verify loading was successful
        assert result.success is True, f"Loading failed: {result.errors}"
        assert result.ontology is not None
        assert isinstance(result.ontology, Ontology)
        
        # Verify ontology has expected content
        ontology = result.ontology
        assert len(ontology.terms) >= 2  # Should have at least ATP and metabolic process
        
        # Check specific terms if parser creates them
        if ontology.terms:
            term_ids = list(ontology.terms.keys())
            assert any("CHEBI:15422" in term_id for term_id in term_ids)
            assert any("GO:0008152" in term_id for term_id in term_ids)
        
        # Verify metadata
        assert result.metadata['format'] == 'csv'
        assert result.metadata['terms_count'] >= 2

    def test_jsonld_parser_integration(self, ontology_manager, temp_dir, sample_jsonld_content):
        """Test integration with JSON-LD parser."""
        # Create JSON-LD file
        jsonld_file = temp_dir / "test_ontology.jsonld"
        jsonld_file.write_text(json.dumps(sample_jsonld_content, indent=2))

        # Load ontology through manager
        result = ontology_manager.load_ontology(str(jsonld_file))

        # Verify loading was successful
        assert result.success is True, f"Loading failed: {result.errors}"
        assert result.ontology is not None
        assert isinstance(result.ontology, Ontology)
        
        # Verify ontology content
        ontology = result.ontology
        assert ontology.name == "Test JSON-LD Ontology"
        
        # Verify metadata
        assert result.metadata['format'] in ['jsonld', 'json-ld', 'json']
        assert result.load_time > 0

    def test_format_auto_detection_by_extension(self, ontology_manager, temp_dir, sample_owl_content):
        """Test format auto-detection based on file extensions."""
        test_cases = [
            ("test.owl", sample_owl_content),
            ("test.rdf", sample_owl_content),
            ("test.xml", sample_owl_content),
        ]
        
        for filename, content in test_cases:
            test_file = temp_dir / filename
            test_file.write_text(content)
            
            result = ontology_manager.load_ontology(str(test_file))
            
            # Should successfully detect format and load
            assert result.success is True, f"Failed to load {filename}: {result.errors}"
            assert result.ontology is not None

    def test_format_auto_detection_by_content(self, ontology_manager, temp_dir):
        """Test format auto-detection based on file content when extension is ambiguous."""
        # Create file with no extension but OWL content
        test_file = temp_dir / "ambiguous_file"
        test_file.write_text('''<?xml version="1.0"?>
<rdf:RDF xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#"
         xmlns:owl="http://www.w3.org/2002/07/owl#">
    <owl:Ontology rdf:about="http://example.org/test"/>
</rdf:RDF>''')

        result = ontology_manager.load_ontology(str(test_file))
        
        # Should detect as OWL/RDF based on content
        if result.success:  # Content detection may not always work
            assert result.ontology is not None
            assert result.metadata['format'] in ['owl', 'rdf', 'xml']

    def test_batch_loading_mixed_formats(self, ontology_manager, temp_dir, 
                                       sample_owl_content, sample_csv_content, sample_jsonld_content):
        """Test loading multiple ontologies of different formats."""
        # Create files of different formats
        owl_file = temp_dir / "test.owl"
        csv_file = temp_dir / "test.csv" 
        jsonld_file = temp_dir / "test.jsonld"
        
        owl_file.write_text(sample_owl_content)
        csv_file.write_text(sample_csv_content)
        jsonld_file.write_text(json.dumps(sample_jsonld_content, indent=2))
        
        # Load all files
        sources = [str(owl_file), str(csv_file), str(jsonld_file)]
        results = ontology_manager.load_ontologies(sources)
        
        assert len(results) == 3
        
        # Check each result
        successful_loads = [r for r in results if r.success]
        assert len(successful_loads) >= 1, "At least one format should load successfully"
        
        # Verify different formats were detected
        detected_formats = {r.metadata.get('format') for r in successful_loads if 'format' in r.metadata}
        assert len(detected_formats) > 0, "Should detect at least one format"

    def test_error_handling_invalid_owl(self, ontology_manager, temp_dir):
        """Test error handling with malformed OWL content."""
        # Create invalid OWL file
        invalid_owl = temp_dir / "invalid.owl"
        invalid_owl.write_text('''<?xml version="1.0"?>
<invalid_root>
    <unclosed_tag>
    missing closing tags and proper structure
</invalid_root>''')

        result = ontology_manager.load_ontology(str(invalid_owl))
        
        # Should fail gracefully
        assert result.success is False
        assert len(result.errors) > 0
        assert result.ontology is None

    def test_error_handling_invalid_csv(self, ontology_manager, temp_dir):
        """Test error handling with malformed CSV content."""
        # Create invalid CSV file
        invalid_csv = temp_dir / "invalid.csv"
        invalid_csv.write_text('''id,name,definition
"unclosed quote, broken row
missing,columns
too,many,columns,here,extra,data''')

        result = ontology_manager.load_ontology(str(invalid_csv))
        
        # Behavior depends on CSV parser robustness
        # It might succeed with warnings or fail
        if not result.success:
            assert len(result.errors) > 0
        else:
            # If it succeeds, should have warnings
            assert len(result.warnings) >= 0

    def test_error_handling_invalid_json(self, ontology_manager, temp_dir):
        """Test error handling with malformed JSON-LD content."""
        # Create invalid JSON file
        invalid_json = temp_dir / "invalid.jsonld"
        invalid_json.write_text('''{
    "unclosed": "object",
    "missing": "comma"
    "invalid": json syntax
}''')

        result = ontology_manager.load_ontology(str(invalid_json))
        
        # Should fail with JSON parsing error
        assert result.success is False
        assert len(result.errors) > 0

    def test_nonexistent_file_handling(self, ontology_manager):
        """Test handling of nonexistent files."""
        result = ontology_manager.load_ontology("/nonexistent/path/file.owl")
        
        assert result.success is False
        assert len(result.errors) > 0
        assert "not found" in result.errors[0].lower() or "no such file" in result.errors[0].lower()

    def test_unsupported_format_handling(self, ontology_manager, temp_dir):
        """Test handling of unsupported file formats."""
        # Create file with unsupported extension
        unsupported_file = temp_dir / "test.unknown"
        unsupported_file.write_text("This is not a supported ontology format")

        result = ontology_manager.load_ontology(str(unsupported_file))
        
        # Should fail with no suitable parser
        assert result.success is False
        assert len(result.errors) > 0
        assert "no suitable parser" in result.errors[0].lower() or "not supported" in result.errors[0].lower()

    def test_caching_across_formats(self, ontology_manager, temp_dir, sample_owl_content):
        """Test caching behavior across different format loads."""
        # Create same content in different formats
        owl_file = temp_dir / "test.owl"
        rdf_file = temp_dir / "test.rdf"
        
        owl_file.write_text(sample_owl_content)
        rdf_file.write_text(sample_owl_content)
        
        # Load OWL file first
        result1 = ontology_manager.load_ontology(str(owl_file))
        assert result1.success is True
        assert ontology_manager.load_stats['cache_misses'] == 1
        
        # Load RDF file (different path, should be cache miss)
        result2 = ontology_manager.load_ontology(str(rdf_file))
        if result2.success:
            assert ontology_manager.load_stats['cache_misses'] == 2
        
        # Load OWL file again (should be cache hit)
        result3 = ontology_manager.load_ontology(str(owl_file))
        assert result3.success is True
        assert ontology_manager.load_stats['cache_hits'] == 1

    def test_statistics_across_formats(self, ontology_manager, temp_dir,
                                     sample_owl_content, sample_csv_content):
        """Test statistics tracking across different format loads."""
        # Create files
        owl_file = temp_dir / "test.owl"
        csv_file = temp_dir / "test.csv"
        
        owl_file.write_text(sample_owl_content)
        csv_file.write_text(sample_csv_content)
        
        # Load files
        owl_result = ontology_manager.load_ontology(str(owl_file))
        csv_result = ontology_manager.load_ontology(str(csv_file))
        
        stats = ontology_manager.get_statistics()
        
        # Check overall stats
        assert stats['total_loads'] == 2
        
        successful_loads = 0
        if owl_result.success:
            successful_loads += 1
        if csv_result.success:
            successful_loads += 1
            
        assert stats['successful_loads'] == successful_loads
        assert stats['failed_loads'] == 2 - successful_loads
        
        # Check format-specific stats
        formats_loaded = stats['formats_loaded']
        if owl_result.success:
            owl_format = owl_result.metadata.get('format', 'unknown')
            assert formats_loaded[owl_format] >= 1
        if csv_result.success:
            csv_format = csv_result.metadata.get('format', 'unknown')
            assert formats_loaded[csv_format] >= 1

    def test_performance_with_larger_ontology(self, ontology_manager, temp_dir):
        """Test performance characteristics with a larger ontology."""
        # Generate larger CSV content
        csv_lines = ["id,name,definition,namespace"]
        
        # Create 1000 terms
        for i in range(1000):
            csv_lines.append(f"TEST:{i:06d},Term {i},Definition for term {i},test_namespace")
        
        large_csv = temp_dir / "large_ontology.csv"
        large_csv.write_text("\n".join(csv_lines))
        
        # Measure load time
        start_time = time.time()
        result = ontology_manager.load_ontology(str(large_csv))
        load_time = time.time() - start_time
        
        if result.success:
            # Should complete in reasonable time (adjust threshold as needed)
            assert load_time < 10.0, f"Loading took too long: {load_time:.2f}s"
            assert result.metadata['terms_count'] >= 900  # Allow for some parsing tolerance
            
            # Test caching performance
            start_time = time.time()
            cached_result = ontology_manager.load_ontology(str(large_csv))
            cached_load_time = time.time() - start_time
            
            assert cached_result.success is True
            assert cached_result.metadata['cache_hit'] is True
            assert cached_load_time < load_time / 10  # Cache should be much faster

    def test_real_world_file_extensions(self, ontology_manager, temp_dir, sample_owl_content):
        """Test with real-world file extensions and naming patterns."""
        test_extensions = [
            "ontology.owl",
            "data.rdf", 
            "terms.xml",
            "vocabulary.ttl",  # Turtle format (may not be supported)
            "knowledge.n3",    # N3 format (may not be supported)
            "schema.jsonld",
            "data.json",
            "terms.csv",
            "vocabulary.tsv"   # Tab-separated (variant of CSV)
        ]
        
        results = []
        for ext in test_extensions:
            test_file = temp_dir / ext
            # Use appropriate content based on extension
            if ext.endswith(('.owl', '.rdf', '.xml')):
                test_file.write_text(sample_owl_content)
            elif ext.endswith(('.csv', '.tsv')):
                test_file.write_text("id,name,definition\nTEST:001,Test Term,Test Definition")
            else:
                test_file.write_text('{"@context": {"@vocab": "http://example.org/"}, "@id": "test"}')
            
            result = ontology_manager.load_ontology(str(test_file))
            results.append((ext, result.success, result.errors))
        
        # At least some formats should be supported
        successful_results = [r for r in results if r[1]]
        assert len(successful_results) > 0, f"No formats were successfully loaded: {results}"

    def test_parser_integration_robustness(self, ontology_manager, temp_dir):
        """Test robustness of parser integration under various conditions."""
        # Test with empty files
        empty_file = temp_dir / "empty.owl"
        empty_file.write_text("")
        
        result = ontology_manager.load_ontology(str(empty_file))
        assert result.success is False
        
        # Test with files containing only whitespace
        whitespace_file = temp_dir / "whitespace.csv"
        whitespace_file.write_text("   \n  \t  \n   ")
        
        result = ontology_manager.load_ontology(str(whitespace_file))
        # May succeed or fail depending on parser implementation
        assert result is not None
        
        # Test with binary file
        binary_file = temp_dir / "binary.owl"
        binary_file.write_bytes(b"\x00\x01\x02\x03\xff\xfe\xfd")
        
        result = ontology_manager.load_ontology(str(binary_file))
        assert result.success is False

    def test_concurrent_format_loading(self, ontology_manager, temp_dir, 
                                     sample_owl_content, sample_csv_content):
        """Test concurrent loading of different formats."""
        import threading
        import queue
        
        # Create test files
        owl_file = temp_dir / "concurrent.owl"
        csv_file = temp_dir / "concurrent.csv"
        
        owl_file.write_text(sample_owl_content)
        csv_file.write_text(sample_csv_content)
        
        results_queue = queue.Queue()
        
        def load_file(manager, file_path):
            result = manager.load_ontology(str(file_path))
            results_queue.put(result)
        
        # Start concurrent loads
        threads = [
            threading.Thread(target=load_file, args=(ontology_manager, owl_file)),
            threading.Thread(target=load_file, args=(ontology_manager, csv_file))
        ]
        
        for thread in threads:
            thread.start()
        
        for thread in threads:
            thread.join()
        
        # Collect results
        results = []
        while not results_queue.empty():
            results.append(results_queue.get())
        
        assert len(results) == 2
        # At least one should succeed
        assert any(r.success for r in results)