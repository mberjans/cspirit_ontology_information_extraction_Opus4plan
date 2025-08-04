#!/usr/bin/env python3
"""
Integration tests for OntologyManager multi-source loading with real parsers.

This module contains integration tests that verify the OntologyManager's multi-source
loading capabilities work correctly with the actual parser implementations (OWL, CSV,
JSON-LD) in real-world scenarios. These tests complement the unit tests by using
actual file formats and parser implementations rather than mocks.

Test Coverage:
- Real-world multi-format batch loading scenarios
- Large-scale ontology integration from diverse sources
- Performance benchmarking with actual parsing overhead
- Memory usage patterns with real ontology data
- Cache behavior with actual file I/O operations
- Error recovery with real parser failures
- Cross-format compatibility and data consistency
- Production-like workload simulation
"""

import os
import json
import tempfile
import pytest
import time
import threading
import gc
import psutil
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, Any, List, Optional

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


class TestOntologyManagerMultiSourceIntegration:
    """Integration tests for OntologyManager multi-source loading with real parsers."""

    @pytest.fixture
    def temp_dir(self):
        """Create a temporary directory for test files."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            yield Path(tmp_dir)

    @pytest.fixture
    def ontology_manager(self):
        """Create an OntologyManager instance for integration testing."""
        return OntologyManager(enable_caching=True, cache_size_limit=25)

    @pytest.fixture
    def comprehensive_owl_content(self):
        """Comprehensive OWL/RDF content for integration testing."""
        return '''<?xml version="1.0"?>
<rdf:RDF xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#"
         xmlns:owl="http://www.w3.org/2002/07/owl#"
         xmlns:rdfs="http://www.w3.org/2000/01/rdf-schema#"
         xmlns:chebi="http://purl.obolibrary.org/obo/CHEBI_"
         xmlns:go="http://purl.obolibrary.org/obo/GO_"
         xmlns:uniprot="http://purl.uniprot.org/core/">

    <owl:Ontology rdf:about="http://example.org/integration-test-chemical">
        <rdfs:label>Integration Test Chemical Ontology</rdfs:label>
        <rdfs:comment>Comprehensive chemical ontology for multi-source integration testing</rdfs:comment>
        <owl:versionInfo>2.0</owl:versionInfo>
    </owl:Ontology>

    <!-- Chemical compounds -->
    <owl:Class rdf:about="http://purl.obolibrary.org/obo/CHEBI_15422">
        <rdfs:label>ATP</rdfs:label>
        <rdfs:comment>Adenosine 5'-triphosphate</rdfs:comment>
        <chebi:synonym>adenosine triphosphate</chebi:synonym>
        <chebi:synonym>adenosine 5'-triphosphate</chebi:synonym>
        <chebi:formula>C10H16N5O13P3</chebi:formula>
        <chebi:mass>507.181</chebi:mass>
    </owl:Class>

    <owl:Class rdf:about="http://purl.obolibrary.org/obo/CHEBI_17234">
        <rdfs:label>glucose</rdfs:label>
        <rdfs:comment>A simple sugar and an important energy source</rdfs:comment>
        <chebi:synonym>D-glucose</chebi:synonym>
        <chebi:synonym>dextrose</chebi:synonym>
        <chebi:formula>C6H12O6</chebi:formula>
        <chebi:mass>180.156</chebi:mass>
    </owl:Class>

    <owl:Class rdf:about="http://purl.obolibrary.org/obo/CHEBI_16541">
        <rdfs:label>protein</rdfs:label>
        <rdfs:comment>A biological macromolecule</rdfs:comment>
        <chebi:synonym>polypeptide</chebi:synonym>
    </owl:Class>

    <!-- Biological processes -->
    <owl:Class rdf:about="http://purl.obolibrary.org/obo/GO_0008152">
        <rdfs:label>metabolic process</rdfs:label>
        <rdfs:comment>The chemical reactions and pathways</rdfs:comment>
    </owl:Class>

    <owl:Class rdf:about="http://purl.obolibrary.org/obo/GO_0006096">
        <rdfs:label>glycolysis</rdfs:label>
        <rdfs:comment>The chemical reactions and pathways resulting in the breakdown of glucose</rdfs:comment>
    </owl:Class>

    <!-- Relationships -->
    <owl:ObjectProperty rdf:about="http://example.org/participates_in">
        <rdfs:label>participates in</rdfs:label>
        <rdfs:domain rdf:resource="http://purl.obolibrary.org/obo/CHEBI_15422"/>
        <rdfs:range rdf:resource="http://purl.obolibrary.org/obo/GO_0008152"/>
    </owl:ObjectProperty>

    <owl:ObjectProperty rdf:about="http://example.org/substrate_of">
        <rdfs:label>substrate of</rdfs:label>
        <rdfs:domain rdf:resource="http://purl.obolibrary.org/obo/CHEBI_17234"/>
        <rdfs:range rdf:resource="http://purl.obolibrary.org/obo/GO_0006096"/>
    </owl:ObjectProperty>

</rdf:RDF>'''

    @pytest.fixture
    def comprehensive_csv_content(self):
        """Comprehensive CSV content for integration testing."""
        return '''id,name,definition,namespace,synonyms,xrefs,molecular_formula,molecular_weight
CHEBI:15422,ATP,"Adenosine 5'-triphosphate",chemical,"adenosine triphosphate|adenosine 5'-triphosphate","CAS:56-65-5|KEGG:C00002|PubChem:5957",C10H16N5O13P3,507.181
CHEBI:17234,glucose,"A simple sugar and an important energy source",chemical,"D-glucose|dextrose|grape sugar","CAS:50-99-7|KEGG:C00031|PubChem:5793",C6H12O6,180.156
CHEBI:16541,protein,"A biological macromolecule",chemical,"polypeptide|protein chain","UniProt:P0DP23","",""
GO:0008152,metabolic process,"The chemical reactions and pathways",biological_process,"metabolism|metabolic pathway","EC:1.1.1.1","",""
GO:0006096,glycolysis,"The chemical reactions and pathways resulting in the breakdown of glucose",biological_process,"glucose breakdown|glucose catabolism","KEGG:ko00010|REACTOME:R-HSA-70171","",""
UNIPROT:P69905,hemoglobin subunit alpha,"Hemoglobin subunit alpha",protein,"HBA1|hemoglobin alpha 1","UniProt:P69905|HGNC:4823","","15126"
UNIPROT:P68871,hemoglobin subunit beta,"Hemoglobin subunit beta",protein,"HBB|hemoglobin beta","UniProt:P68871|HGNC:4827","","15867"
MESH:D005947,Glucose,"A primary source of energy for living organisms",chemical,"D-Glucose|Dextrose","MESH:D005947|CAS:50-99-7",C6H12O6,180.16
MESH:D006454,Hemoglobins,"The oxygen-carrying proteins of red blood cells",protein,"Hemoglobin|Hb","MESH:D006454","","64500"
NCIT:C389,Protein,"A biological macromolecule consisting of one or more chains of amino acids",chemical,"Proteins|Polypeptide","NCIT:C389","",""'''

    @pytest.fixture
    def comprehensive_jsonld_content(self):
        """Comprehensive JSON-LD content for integration testing."""
        return {
            "@context": {
                "@vocab": "http://example.org/",
                "rdfs": "http://www.w3.org/2000/01/rdf-schema#",
                "owl": "http://www.w3.org/2002/07/owl#",
                "chebi": "http://purl.obolibrary.org/obo/CHEBI_",
                "go": "http://purl.obolibrary.org/obo/GO_",
                "uniprot": "http://purl.uniprot.org/core/",
                "xsd": "http://www.w3.org/2001/XMLSchema#"
            },
            "@id": "integration-test-molecular",
            "@type": "owl:Ontology",
            "rdfs:label": "Integration Test Molecular Ontology",
            "rdfs:comment": "Molecular biology ontology for multi-source integration testing",
            "owl:versionInfo": "3.0",
            "terms": [
                {
                    "@id": "chebi:15422",
                    "@type": "owl:Class",
                    "rdfs:label": "ATP",
                    "rdfs:comment": "Adenosine 5'-triphosphate - the energy currency of cells",
                    "synonyms": ["adenosine triphosphate", "adenosine 5'-triphosphate", "5'-ATP"],
                    "namespace": "chemical",
                    "molecular_formula": "C10H16N5O13P3",
                    "molecular_weight": {"@value": "507.181", "@type": "xsd:float"},
                    "roles": ["energy carrier", "phosphate donor", "allosteric regulator"]
                },
                {
                    "@id": "go:0008152",
                    "@type": "owl:Class",
                    "rdfs:label": "metabolic process",
                    "rdfs:comment": "The chemical reactions and pathways, including anabolism and catabolism",
                    "synonyms": ["metabolism", "metabolic pathway", "biochemical process"],
                    "namespace": "biological_process",
                    "subprocesses": ["go:0006096", "go:0006091", "go:0019319"]
                },
                {
                    "@id": "uniprot:P69905",
                    "@type": "owl:Class", 
                    "rdfs:label": "hemoglobin subunit alpha",
                    "rdfs:comment": "Alpha globin chain of hemoglobin",
                    "synonyms": ["HBA1", "hemoglobin alpha 1", "alpha-globin"],
                    "namespace": "protein",
                    "organism": "Homo sapiens",
                    "function": "oxygen transport",
                    "length": {"@value": "141", "@type": "xsd:int"}
                },
                {
                    "@id": "go:0006096",
                    "@type": "owl:Class",
                    "rdfs:label": "glycolysis",
                    "rdfs:comment": "The chemical reactions and pathways resulting in the breakdown of glucose to pyruvate",
                    "synonyms": ["glucose breakdown", "Embden-Meyerhof pathway", "glucose catabolism"],
                    "namespace": "biological_process",
                    "part_of": "go:0008152",
                    "substrates": ["chebi:17234"],
                    "products": ["chebi:15361", "chebi:15422"]
                },
                {
                    "@id": "chebi:17234",
                    "@type": "owl:Class",
                    "rdfs:label": "glucose",
                    "rdfs:comment": "A monosaccharide sugar that is an important energy source",
                    "synonyms": ["D-glucose", "dextrose", "grape sugar", "blood sugar"],
                    "namespace": "chemical",
                    "molecular_formula": "C6H12O6",
                    "molecular_weight": {"@value": "180.156", "@type": "xsd:float"},
                    "stereoisomers": ["alpha-D-glucose", "beta-D-glucose"]
                }
            ],
            "relationships": [
                {
                    "@id": "rel:atp_participates_metabolism",
                    "@type": "owl:ObjectProperty",
                    "subject": "chebi:15422",
                    "predicate": "participates_in",
                    "object": "go:0008152",
                    "confidence": {"@value": "0.98", "@type": "xsd:float"},
                    "evidence": "experimental"
                },
                {
                    "@id": "rel:glucose_substrate_glycolysis",
                    "@type": "owl:ObjectProperty", 
                    "subject": "chebi:17234",
                    "predicate": "substrate_of",
                    "object": "go:0006096",
                    "confidence": {"@value": "0.99", "@type": "xsd:float"},
                    "evidence": "experimental"
                },
                {
                    "@id": "rel:glycolysis_part_of_metabolism",
                    "@type": "owl:ObjectProperty",
                    "subject": "go:0006096", 
                    "predicate": "part_of",
                    "object": "go:0008152",
                    "confidence": {"@value": "1.0", "@type": "xsd:float"},
                    "evidence": "logical"
                }
            ]
        }

    @pytest.fixture
    def create_comprehensive_test_files(self, temp_dir, comprehensive_owl_content, 
                                       comprehensive_csv_content, comprehensive_jsonld_content):
        """Create comprehensive test files in multiple formats."""
        files = {}
        
        # Create OWL files with different characteristics
        owl_files = [
            ("chemical_ontology.owl", comprehensive_owl_content),
            ("small_ontology.rdf", '''<?xml version="1.0"?>
<rdf:RDF xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#"
         xmlns:owl="http://www.w3.org/2002/07/owl#"
         xmlns:rdfs="http://www.w3.org/2000/01/rdf-schema#">
    <owl:Ontology rdf:about="http://example.org/small-test">
        <rdfs:label>Small Test Ontology</rdfs:label>
    </owl:Ontology>
    <owl:Class rdf:about="http://example.org/SmallTerm">
        <rdfs:label>Small Term</rdfs:label>
    </owl:Class>
</rdf:RDF>'''),
            ("minimal.xml", '''<?xml version="1.0"?>
<rdf:RDF xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#"
         xmlns:owl="http://www.w3.org/2002/07/owl#">
    <owl:Ontology rdf:about="http://example.org/minimal"/>
</rdf:RDF>''')
        ]
        
        for filename, content in owl_files:
            file_path = temp_dir / filename
            file_path.write_text(content)
            files[filename] = str(file_path)
        
        # Create CSV files with different structures
        csv_files = [
            ("comprehensive_terms.csv", comprehensive_csv_content),
            ("simple_terms.csv", '''id,name,definition
SIMPLE:001,Simple Term One,First simple term definition
SIMPLE:002,Simple Term Two,Second simple term definition
SIMPLE:003,Simple Term Three,Third simple term definition'''),
            ("protein_data.tsv", '''id\tname\tdefinition\torganism\tfunction
PROT:001\tMyoglobin\tOxygen-binding protein\tHomo sapiens\toxygen storage
PROT:002\tInsulin\tHormone regulating glucose\tHomo sapiens\tglucose regulation
PROT:003\tHemoglobin\tOxygen transport protein\tHomo sapiens\toxygen transport'''),
            ("relationships.csv", '''subject,predicate,object,confidence
CHEBI:15422,regulates,GO:0008152,0.95
CHEBI:17234,participates_in,GO:0006096,0.98
GO:0006096,part_of,GO:0008152,1.0''')
        ]
        
        for filename, content in csv_files:
            file_path = temp_dir / filename
            file_path.write_text(content)
            files[filename] = str(file_path)
        
        # Create JSON-LD files
        jsonld_files = [
            ("molecular_ontology.jsonld", comprehensive_jsonld_content),
            ("simple_data.json", {
                "@context": {"@vocab": "http://example.org/"},
                "@id": "simple-json-ontology",
                "@type": "Ontology",
                "name": "Simple JSON Ontology",
                "terms": [
                    {"@id": "JSON:001", "name": "JSON Term", "definition": "A term from JSON"}
                ]
            }),
            ("context_free.jsonld", {
                "id": "context-free-ontology",
                "type": "Ontology",
                "name": "Context Free Ontology",
                "description": "Ontology without @context",
                "terms": [
                    {"id": "CF:001", "name": "Context Free Term", "definition": "Term without context"}
                ]
            })
        ]
        
        for filename, content in jsonld_files:
            file_path = temp_dir / filename
            file_path.write_text(json.dumps(content, indent=2))
            files[filename] = str(file_path)
        
        return files

    def test_comprehensive_multi_format_loading(self, ontology_manager, create_comprehensive_test_files):
        """Test comprehensive loading of multiple ontologies from different formats."""
        all_files = list(create_comprehensive_test_files.values())
        
        # Load all files using actual parsers
        start_time = time.time()
        results = ontology_manager.load_ontologies(all_files)
        total_time = time.time() - start_time
        
        # Analyze results
        successful_results = [r for r in results if r.success]
        failed_results = [r for r in results if not r.success]
        
        # We expect most formats to be supported
        assert len(successful_results) >= len(all_files) * 0.6, f"Too many failures: {len(failed_results)} out of {len(all_files)}"
        
        # Verify performance is reasonable
        assert total_time < 30.0, f"Loading took too long: {total_time:.2f}s"
        
        # Verify different formats were detected
        detected_formats = set()
        for result in successful_results:
            if 'format' in result.metadata:
                detected_formats.add(result.metadata['format'])
        
        assert len(detected_formats) >= 2, f"Should detect multiple formats, got: {detected_formats}"
        
        # Verify ontologies were actually loaded
        assert len(ontology_manager.ontologies) >= len(successful_results) * 0.8
        
        # Check statistics
        stats = ontology_manager.get_statistics()
        assert stats['total_loads'] == len(all_files)
        assert stats['successful_loads'] == len(successful_results)
        assert stats['failed_loads'] == len(failed_results)
        
        # Verify format-specific statistics
        formats_loaded = stats['formats_loaded']
        assert len(formats_loaded) >= 2, "Should have loaded multiple formats"

    def test_large_scale_batch_loading_performance(self, ontology_manager, temp_dir):
        """Test performance with large-scale batch loading of diverse ontologies."""
        # Create many files of different formats and sizes
        large_batch_files = []
        
        # Create 20 CSV files with varying sizes
        for i in range(20):
            term_count = 50 + (i * 25)  # 50 to 525 terms
            csv_lines = ["id,name,definition,namespace"]
            
            for j in range(term_count):
                csv_lines.append(f"BATCH{i}:TERM{j:04d},Batch {i} Term {j},Definition for batch {i} term {j},batch_{i}")
            
            csv_file = temp_dir / f"batch_{i:02d}.csv"
            csv_file.write_text("\n".join(csv_lines))
            large_batch_files.append(str(csv_file))
        
        # Create 10 OWL files
        for i in range(10):
            owl_content = f'''<?xml version="1.0"?>
<rdf:RDF xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#"
         xmlns:owl="http://www.w3.org/2002/07/owl#"
         xmlns:rdfs="http://www.w3.org/2000/01/rdf-schema#">
    <owl:Ontology rdf:about="http://example.org/batch-owl-{i}">
        <rdfs:label>Batch OWL Ontology {i}</rdfs:label>
    </owl:Ontology>'''
            
            # Add varying numbers of terms
            for j in range(10 + (i * 5)):  # 10 to 55 terms
                owl_content += f'''
    <owl:Class rdf:about="http://example.org/BATCH_OWL{i}_TERM{j}">
        <rdfs:label>Batch OWL {i} Term {j}</rdfs:label>
        <rdfs:comment>Definition for batch OWL {i} term {j}</rdfs:comment>
    </owl:Class>'''
            
            owl_content += "\n</rdf:RDF>"
            
            owl_file = temp_dir / f"batch_owl_{i:02d}.owl"
            owl_file.write_text(owl_content)
            large_batch_files.append(str(owl_file))
        
        # Create 5 JSON-LD files
        for i in range(5):
            terms = []
            for j in range(20 + (i * 10)):  # 20 to 60 terms
                terms.append({
                    "@id": f"BATCH_JSON{i}:TERM{j:03d}",
                    "name": f"Batch JSON {i} Term {j}",
                    "definition": f"Definition for batch JSON {i} term {j}",
                    "namespace": f"batch_json_{i}"
                })
            
            jsonld_content = {
                "@context": {"@vocab": "http://example.org/"},
                "@id": f"batch-json-{i}",
                "@type": "Ontology",
                "name": f"Batch JSON Ontology {i}",
                "terms": terms
            }
            
            json_file = temp_dir / f"batch_json_{i:02d}.jsonld"
            json_file.write_text(json.dumps(jsonld_content, indent=2))
            large_batch_files.append(str(json_file))
        
        # Measure memory before loading
        process = psutil.Process()
        initial_memory = process.memory_info().rss
        
        # Perform large-scale batch loading
        start_time = time.time()
        results = ontology_manager.load_ontologies(large_batch_files)
        end_time = time.time()
        
        # Measure memory after loading
        final_memory = process.memory_info().rss
        memory_increase = final_memory - initial_memory
        
        # Analyze performance results
        total_time = end_time - start_time
        successful_results = [r for r in results if r.success]
        
        # Performance assertions
        assert len(results) == len(large_batch_files)
        assert len(successful_results) >= len(large_batch_files) * 0.8, "Most files should load successfully"
        assert total_time < 60.0, f"Batch loading took too long: {total_time:.2f}s"
        assert memory_increase < 1024 * 1024 * 1024, f"Memory increase too high: {memory_increase / 1024 / 1024:.2f}MB"
        
        # Throughput analysis
        avg_time_per_file = total_time / len(large_batch_files)
        assert avg_time_per_file < 2.0, f"Average time per file too high: {avg_time_per_file:.3f}s"
        
        # Verify data integrity
        stats = ontology_manager.get_statistics()
        assert stats['total_loads'] == len(large_batch_files)
        assert stats['loaded_ontologies'] >= len(successful_results) * 0.9
        
        # Verify format diversity
        formats_loaded = stats['formats_loaded']
        assert len(formats_loaded) >= 3, f"Should detect multiple formats: {formats_loaded}"

    def test_memory_usage_with_real_ontologies(self, ontology_manager, temp_dir):
        """Test memory management with real ontology data and actual parsing."""
        # Create ontologies with realistic data volumes
        memory_test_files = []
        
        # Large CSV ontology (medical terms)
        medical_csv_lines = ["id,name,definition,category,synonyms"]
        for i in range(1000):  # 1000 medical terms
            synonyms = f"synonym_{i}_1|synonym_{i}_2|synonym_{i}_3"
            medical_csv_lines.append(
                f"MED:{i:06d},Medical Term {i},Clinical definition for medical term {i} with detailed description,medical_category_{i%10},{synonyms}"
            )
        
        large_medical_csv = temp_dir / "large_medical_ontology.csv"
        large_medical_csv.write_text("\n".join(medical_csv_lines))
        memory_test_files.append(str(large_medical_csv))
        
        # Complex OWL ontology (biological processes)
        complex_owl_terms = []
        for i in range(500):  # 500 biological process terms
            complex_owl_terms.append(f'''
    <owl:Class rdf:about="http://purl.obolibrary.org/obo/GO_{i:07d}">
        <rdfs:label>biological process {i}</rdfs:label>
        <rdfs:comment>Detailed description of biological process {i} including molecular mechanisms, cellular context, and regulatory pathways involved in this complex biological phenomenon</rdfs:comment>
        <go:synonym>BP{i}</go:synonym>
        <go:synonym>process_{i}</go:synonym>
        <go:category>biological_process</go:category>
        <go:evidence_code>IEA</go:evidence_code>
    </owl:Class>''')
        
        complex_owl_content = f'''<?xml version="1.0"?>
<rdf:RDF xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#"
         xmlns:owl="http://www.w3.org/2002/07/owl#"
         xmlns:rdfs="http://www.w3.org/2000/01/rdf-schema#"
         xmlns:go="http://purl.obolibrary.org/obo/go#">
    <owl:Ontology rdf:about="http://example.org/complex-biological-processes">
        <rdfs:label>Complex Biological Processes Ontology</rdfs:label>
        <rdfs:comment>Comprehensive ontology of biological processes for memory testing</rdfs:comment>
    </owl:Ontology>
    {"".join(complex_owl_terms)}
</rdf:RDF>'''
        
        complex_owl_file = temp_dir / "complex_biological_processes.owl"
        complex_owl_file.write_text(complex_owl_content)
        memory_test_files.append(str(complex_owl_file))
        
        # Rich JSON-LD ontology (chemical compounds)
        chemical_terms = []
        for i in range(750):  # 750 chemical compounds
            chemical_terms.append({
                "@id": f"CHEBI:{i:06d}",
                "@type": "ChemicalCompound",
                "name": f"Chemical Compound {i}",
                "definition": f"Detailed chemical description of compound {i} including structural formula, molecular properties, biological activity, and pharmacological characteristics",
                "molecular_formula": f"C{i%20+1}H{i%30+1}N{i%5+1}O{i%10+1}",
                "molecular_weight": 100.0 + (i * 0.5),
                "synonyms": [f"compound_{i}", f"chemical_{i}", f"substance_{i}"],
                "categories": [f"category_{i%15}", f"class_{i%8}"],
                "properties": {
                    "solubility": f"soluble_{i%3}",
                    "toxicity": f"toxicity_level_{i%5}",
                    "stability": f"stable_{i%4}"
                },
                "cross_references": [f"CAS:{i:06d}", f"PubChem:{i+100000}", f"KEGG:C{i:05d}"]
            })
        
        rich_jsonld_content = {
            "@context": {
                "@vocab": "http://example.org/",
                "chebi": "http://purl.obolibrary.org/obo/CHEBI_"
            },
            "@id": "rich-chemical-ontology",
            "@type": "ChemicalOntology",
            "name": "Rich Chemical Compounds Ontology",
            "description": "Comprehensive chemical ontology for memory usage testing",
            "compounds": chemical_terms
        }
        
        rich_json_file = temp_dir / "rich_chemical_ontology.jsonld"
        rich_json_file.write_text(json.dumps(rich_jsonld_content, indent=2))
        memory_test_files.append(str(rich_json_file))
        
        # Monitor memory usage during loading
        process = psutil.Process()
        memory_snapshots = []
        
        # Initial memory
        initial_memory = process.memory_info().rss
        memory_snapshots.append(("initial", initial_memory))
        
        # Load files one by one and monitor memory
        results = []
        for i, file_path in enumerate(memory_test_files):
            gc.collect()  # Force garbage collection before measurement
            pre_load_memory = process.memory_info().rss
            
            result = ontology_manager.load_ontology(file_path)
            results.append(result)
            
            gc.collect()  # Force garbage collection after loading
            post_load_memory = process.memory_info().rss
            
            memory_snapshots.append((f"after_file_{i}", post_load_memory))
            
            # Verify successful loading
            assert result.success, f"Failed to load {file_path}: {result.errors}"
        
        # Final memory measurement
        final_memory = process.memory_info().rss
        memory_snapshots.append(("final", final_memory))
        
        # Analyze memory usage patterns
        total_memory_increase = final_memory - initial_memory
        max_memory_per_file = max(
            memory_snapshots[i+1][1] - memory_snapshots[i][1]
            for i in range(len(memory_snapshots) - 1)
        )
        
        # Memory usage assertions
        assert total_memory_increase < 2 * 1024 * 1024 * 1024, f"Total memory increase too high: {total_memory_increase / 1024 / 1024:.2f}MB"
        assert max_memory_per_file < 500 * 1024 * 1024, f"Per-file memory increase too high: {max_memory_per_file / 1024 / 1024:.2f}MB"
        
        # Test cache effectiveness in reducing memory pressure
        ontology_manager.clear_cache()
        gc.collect()
        
        after_cache_clear = process.memory_info().rss
        memory_freed = final_memory - after_cache_clear
        
        # Should free some memory
        assert memory_freed > 10 * 1024 * 1024, f"Cache clear freed too little memory: {memory_freed / 1024 / 1024:.2f}MB"
        
        # Verify all ontologies are still accessible
        assert len(ontology_manager.ontologies) == len(memory_test_files)
        stats = ontology_manager.get_statistics()
        assert stats['loaded_ontologies'] == len(memory_test_files)

    def test_concurrent_real_parser_loading(self, ontology_manager, create_comprehensive_test_files):
        """Test concurrent loading using real parsers with thread safety."""
        all_files = list(create_comprehensive_test_files.values())
        
        # Divide files into groups for concurrent processing
        file_groups = [
            all_files[:len(all_files)//3],          # First third
            all_files[len(all_files)//3:2*len(all_files)//3],  # Second third
            all_files[2*len(all_files)//3:]         # Last third
        ]
        
        def load_group_with_timing(group_files):
            """Load a group of files and return results with timing."""
            start_time = time.time()
            group_results = ontology_manager.load_ontologies(group_files)
            end_time = time.time()
            return {
                'results': group_results,
                'load_time': end_time - start_time,
                'group_size': len(group_files),
                'thread_id': threading.current_thread().ident
            }
        
        # Execute concurrent loading
        with ThreadPoolExecutor(max_workers=3) as executor:
            # Submit all group loading tasks
            future_to_group = {
                executor.submit(load_group_with_timing, group): i
                for i, group in enumerate(file_groups)
            }
            
            # Collect results as they complete
            group_results = []
            for future in as_completed(future_to_group):
                group_index = future_to_group[future]
                try:
                    result = future.result()
                    result['group_index'] = group_index
                    group_results.append(result)
                except Exception as exc:
                    pytest.fail(f'Group {group_index} generated an exception: {exc}')
        
        # Analyze concurrent loading results
        all_results = []
        total_concurrent_time = 0
        
        for group_result in group_results:
            all_results.extend(group_result['results'])
            total_concurrent_time = max(total_concurrent_time, group_result['load_time'])
        
        # Verify results
        assert len(all_results) == len(all_files)
        successful_concurrent = [r for r in all_results if r.success]
        
        # Should have reasonable success rate with real parsers
        success_rate = len(successful_concurrent) / len(all_results)
        assert success_rate >= 0.7, f"Concurrent success rate too low: {success_rate:.2%}"
        
        # Verify thread safety - no data corruption
        unique_ontology_ids = set()
        for result in successful_concurrent:
            if result.ontology and result.ontology.id:
                assert result.ontology.id not in unique_ontology_ids, f"Duplicate ontology ID: {result.ontology.id}"
                unique_ontology_ids.add(result.ontology.id)
        
        # Compare with sequential loading performance
        ontology_manager_sequential = OntologyManager(enable_caching=True, cache_size_limit=25)
        
        start_sequential = time.time()
        sequential_results = ontology_manager_sequential.load_ontologies(all_files)
        sequential_time = time.time() - start_sequential
        
        successful_sequential = [r for r in sequential_results if r.success]
        
        # Concurrent loading should not be significantly slower
        # (May not be faster due to I/O bound operations and Python GIL)
        assert total_concurrent_time < sequential_time * 2, f"Concurrent loading too slow: {total_concurrent_time:.2f}s vs {sequential_time:.2f}s sequential"
        
        # Results should be comparable
        assert abs(len(successful_concurrent) - len(successful_sequential)) <= 2, "Concurrent and sequential results should be similar"

    def test_cache_behavior_with_real_files(self, ontology_manager, create_comprehensive_test_files):
        """Test caching behavior with real file I/O and parsing operations."""
        test_files = list(create_comprehensive_test_files.values())[:5]  # Use first 5 files
        
        # First load - all should be cache misses
        results_first = ontology_manager.load_ontologies(test_files)
        successful_first = [r for r in results_first if r.success]
        
        assert len(successful_first) >= 3, "Should have at least 3 successful loads"
        
        # Verify cache misses
        cache_misses_after_first = ontology_manager.load_stats['cache_misses']
        cache_hits_after_first = ontology_manager.load_stats['cache_hits']
        assert cache_misses_after_first >= len(successful_first)
        assert cache_hits_after_first == 0
        
        # Second load - should hit cache for successful files
        results_second = ontology_manager.load_ontologies(test_files)
        successful_second = [r for r in results_second if r.success]
        
        # Verify cache hits
        cache_hits_after_second = ontology_manager.load_stats['cache_hits']
        expected_hits = len([r for r in results_first if r.success])
        assert cache_hits_after_second >= expected_hits * 0.8, f"Expected ~{expected_hits} cache hits, got {cache_hits_after_second}"
        
        # Verify cached results have correct metadata
        cached_results = [r for r in results_second if r.success and r.metadata.get('cache_hit', False)]
        assert len(cached_results) >= len(successful_first) * 0.8
        
        for cached_result in cached_results:
            assert cached_result.metadata['cache_hit'] is True
            assert 'access_count' in cached_result.metadata
            assert cached_result.metadata['access_count'] >= 2
        
        # Test cache invalidation with file modification
        if successful_first:
            # Modify one of the successfully loaded files
            first_successful_file = successful_first[0].source_path
            
            # Get original content and modify it
            original_content = Path(first_successful_file).read_text()
            modified_content = original_content + "\n# Modified for cache invalidation test"
            
            # Wait a bit to ensure different mtime
            time.sleep(0.1)
            Path(first_successful_file).write_text(modified_content)
            
            # Load again - should detect modification and reload
            initial_cache_misses = ontology_manager.load_stats['cache_misses']
            result_after_modification = ontology_manager.load_ontology(first_successful_file)
            
            if result_after_modification.success:
                # Should have triggered cache miss due to file modification
                final_cache_misses = ontology_manager.load_stats['cache_misses']
                assert final_cache_misses > initial_cache_misses, "File modification should trigger cache miss"
        
        # Test cache efficiency with repeated partial loads
        subset_files = test_files[:3]
        
        # Load subset multiple times
        for i in range(3):
            subset_results = ontology_manager.load_ontologies(subset_files)
            successful_subset = [r for r in subset_results if r.success]
            
            if i > 0:  # After first load, should be cache hits
                cache_hit_results = [r for r in successful_subset if r.metadata.get('cache_hit', False)]
                assert len(cache_hit_results) >= len(successful_subset) * 0.8, f"Iteration {i}: Expected cache hits"

    def test_error_recovery_real_world_scenarios(self, ontology_manager, temp_dir):
        """Test error recovery with real-world file issues and parser failures."""
        # Create files with various real-world problems
        problematic_files = []
        
        # Valid reference file
        valid_csv = temp_dir / "valid_reference.csv"
        valid_csv.write_text("id,name,definition\nVALID:001,Valid Term,Valid definition")
        problematic_files.append(("valid", str(valid_csv)))
        
        # Corrupted XML file (invalid XML syntax)
        corrupted_xml = temp_dir / "corrupted.owl"
        corrupted_xml.write_text('''<?xml version="1.0"?>
<rdf:RDF xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#"
         xmlns:owl="http://www.w3.org/2002/07/owl#">
    <owl:Ontology rdf:about="http://example.org/corrupted">
        <rdfs:label>Corrupted Ontology</rdfs:label>
        <!-- Missing closing tag and broken structure -->
    <owl:Class rdf:about="http://example.org/BrokenTerm">
        <rdfs:label>Broken Term</rdfs:label>
        <!-- Unclosed element -->
    </owl:Class
</rdf:RDF>''')
        problematic_files.append(("corrupted_xml", str(corrupted_xml)))
        
        # CSV with encoding issues
        encoding_csv = temp_dir / "encoding_issues.csv"
        with open(encoding_csv, 'w', encoding='latin1') as f:
            f.write("id,name,definition\nENC:001,Tëst Tërm,Définition with spëcial charactërs\n")
        problematic_files.append(("encoding_issues", str(encoding_csv)))
        
        # JSON with syntax errors
        malformed_json = temp_dir / "malformed.jsonld"
        malformed_json.write_text('''{
    "@context": {"@vocab": "http://example.org/"},
    "@id": "malformed-ontology",
    "name": "Malformed Ontology",
    "terms": [
        {
            "@id": "MAL:001",
            "name": "Malformed Term",
            "definition": "Term with JSON syntax errors"
        },
        {
            "@id": "MAL:002",
            "name": "Another Term",
            "definition": "Missing comma above"
            "extra_field": "this causes JSON error"
        }
    ]
}''')
        problematic_files.append(("malformed_json", str(malformed_json)))
        
        # Empty file
        empty_file = temp_dir / "empty.csv"
        empty_file.write_text("")
        problematic_files.append(("empty", str(empty_file)))
        
        # File with only headers
        headers_only_csv = temp_dir / "headers_only.csv"
        headers_only_csv.write_text("id,name,definition\n")
        problematic_files.append(("headers_only", str(headers_only_csv)))
        
        # Binary file with wrong extension
        binary_file = temp_dir / "binary_data.owl"
        binary_file.write_bytes(b'\x00\x01\x02\x03\xff\xfe\xfd\xfc\x89PNG\r\n\x1a\n')
        problematic_files.append(("binary", str(binary_file)))
        
        # Extremely large file (simulate memory issues)
        large_csv_lines = ["id,name,definition"]
        for i in range(50000):  # 50K lines
            large_csv_lines.append(f"LARGE:{i:06d},Large Term {i},Definition {i} " + "x" * 200)  # Long definitions
        
        large_file = temp_dir / "extremely_large.csv"
        large_file.write_text("\n".join(large_csv_lines))
        problematic_files.append(("extremely_large", str(large_file)))
        
        # File with permission issues (simulate by making directory read-only on Unix systems)
        permission_file = temp_dir / "permission_test.csv"
        permission_file.write_text("id,name,definition\nPERM:001,Permission Term,Permission definition")
        
        try:
            # Try to make file unreadable (may not work on all systems)
            permission_file.chmod(0o000)
            problematic_files.append(("permission_denied", str(permission_file)))
        except (OSError, PermissionError):
            # Skip if we can't modify permissions
            pass
        
        # Load all problematic files
        file_paths = [file_info[1] for file_info in problematic_files]
        
        start_time = time.time()
        results = ontology_manager.load_ontologies(file_paths)
        end_time = time.time()
        
        # Restore file permissions if modified
        try:
            permission_file.chmod(0o644)
        except (OSError, PermissionError):
            pass
        
        # Analyze error recovery
        successful_results = [r for r in results if r.success]
        failed_results = [r for r in results if not r.success]
        
        # Should handle at least the valid file
        assert len(successful_results) >= 1, "Should successfully load at least the valid reference file"
        
        # Should gracefully handle failures
        assert len(failed_results) >= 3, "Should detect multiple problematic files"
        
        # Should complete in reasonable time despite errors
        assert end_time - start_time < 60.0, f"Error recovery took too long: {end_time - start_time:.2f}s"
        
        # Analyze error types
        error_categories = {
            'xml_errors': 0,
            'json_errors': 0,
            'encoding_errors': 0,
            'empty_file_errors': 0,
            'permission_errors': 0,
            'size_errors': 0,
            'unknown_errors': 0
        }
        
        for result in failed_results:
            error_text = " ".join(result.errors).lower()
            
            if any(keyword in error_text for keyword in ['xml', 'parsing', 'syntax']):
                error_categories['xml_errors'] += 1
            elif any(keyword in error_text for keyword in ['json', 'invalid json']):
                error_categories['json_errors'] += 1
            elif any(keyword in error_text for keyword in ['encoding', 'decode']):
                error_categories['encoding_errors'] += 1
            elif any(keyword in error_text for keyword in ['empty', 'no data']):
                error_categories['empty_file_errors'] += 1
            elif any(keyword in error_text for keyword in ['permission', 'access']):
                error_categories['permission_errors'] += 1
            elif any(keyword in error_text for keyword in ['memory', 'size', 'large']):
                error_categories['size_errors'] += 1
            else:
                error_categories['unknown_errors'] += 1
        
        # Verify error categorization makes sense
        total_categorized = sum(error_categories.values())
        assert total_categorized >= len(failed_results) * 0.8, "Most errors should be categorized"
        
        # Manager should remain operational after errors
        stats = ontology_manager.get_statistics()
        assert stats['total_loads'] == len(file_paths)
        assert stats['successful_loads'] == len(successful_results)
        assert stats['failed_loads'] == len(failed_results)
        
        # Should be able to load new valid files after errors
        recovery_csv = temp_dir / "recovery_test.csv"
        recovery_csv.write_text("id,name,definition\nREC:001,Recovery Term,Recovery definition")
        
        recovery_result = ontology_manager.load_ontology(str(recovery_csv))
        assert recovery_result.success, f"Should recover and load new files: {recovery_result.errors}"

    def test_production_workload_simulation(self, ontology_manager, temp_dir):
        """Simulate production-like workload with mixed operations and realistic data."""
        # Create diverse ontology sources mimicking production usage
        production_files = []
        
        # Medical terminology (large CSV)
        medical_terms = ["id,name,definition,category,icd10_code,synonyms"]
        medical_categories = ["disease", "symptom", "procedure", "anatomy", "drug"]
        
        for i in range(2000):  # 2000 medical terms
            category = medical_categories[i % len(medical_categories)]
            icd_code = f"{chr(65 + i%26)}{i%99:02d}.{i%10}"
            synonyms = f"med_syn_{i}_1|med_syn_{i}_2"
            medical_terms.append(
                f"MED:{i:06d},Medical Term {i},Clinical definition for {category} term {i},{category},{icd_code},{synonyms}"
            )
        
        medical_file = temp_dir / "medical_terminology.csv"
        medical_file.write_text("\n".join(medical_terms))
        production_files.append(("medical_csv", str(medical_file)))
        
        # Chemical compounds (OWL format)
        chemical_owl_classes = []
        for i in range(1500):  # 1500 chemical compounds
            chemical_owl_classes.append(f'''
    <owl:Class rdf:about="http://purl.obolibrary.org/obo/CHEBI_{i:06d}">
        <rdfs:label>Chemical Compound {i}</rdfs:label>
        <rdfs:comment>Chemical compound {i} with molecular formula and biological activity</rdfs:comment>
        <chebi:formula>C{i%25+1}H{i%40+1}N{i%8+1}O{i%15+1}</chebi:formula>
        <chebi:mass>{100.0 + (i * 0.123):.3f}</chebi:mass>
        <chebi:charge>{(i % 5) - 2}</chebi:charge>
        <chebi:smiles>CCO{i%10}</chebi:smiles>
    </owl:Class>''')
        
        chemical_owl = f'''<?xml version="1.0"?>
<rdf:RDF xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#"
         xmlns:owl="http://www.w3.org/2002/07/owl#"
         xmlns:rdfs="http://www.w3.org/2000/01/rdf-schema#"
         xmlns:chebi="http://purl.obolibrary.org/obo/chebi#">
    <owl:Ontology rdf:about="http://example.org/production-chemicals">
        <rdfs:label>Production Chemical Compounds Database</rdfs:label>
        <owl:versionInfo>4.0</owl:versionInfo>
    </owl:Ontology>
    {"".join(chemical_owl_classes)}
</rdf:RDF>'''
        
        chemical_file = temp_dir / "chemical_compounds.owl"
        chemical_file.write_text(chemical_owl)
        production_files.append(("chemical_owl", str(chemical_file)))
        
        # Biological processes (JSON-LD)
        biological_processes = []
        process_types = ["metabolic", "signaling", "transport", "regulation", "development"]
        
        for i in range(1000):  # 1000 biological processes
            process_type = process_types[i % len(process_types)]
            biological_processes.append({
                "@id": f"GO:{i:07d}",
                "@type": "BiologicalProcess",
                "name": f"{process_type.title()} Process {i}",
                "definition": f"Biological {process_type} process {i} involving cellular mechanisms and molecular interactions",
                "namespace": "biological_process",
                "process_type": process_type,
                "evidence_codes": [f"IEA_{i%10}", f"IDA_{i%15}"],
                "annotations": {
                    "complexity": f"level_{i%5+1}",
                    "conservation": f"conserved_{i%3}",
                    "regulation": f"regulated_by_{i%20}"
                },
                "related_processes": [f"GO:{(i+j)%1000:07d}" for j in range(1, min(4, i%5+1))]
            })
        
        biological_jsonld = {
            "@context": {
                "@vocab": "http://example.org/",
                "go": "http://purl.obolibrary.org/obo/GO_"
            },
            "@id": "production-biological-processes",
            "@type": "BiologicalProcessOntology",
            "name": "Production Biological Processes Database",
            "version": "3.1",
            "description": "Comprehensive biological processes for production use",
            "processes": biological_processes
        }
        
        biological_file = temp_dir / "biological_processes.jsonld"
        biological_file.write_text(json.dumps(biological_jsonld, indent=2))
        production_files.append(("biological_jsonld", str(biological_file)))
        
        # Protein sequences (TSV format)
        protein_lines = ["id\tname\tdefinition\torganism\tsequence_length\tfunction\tdomain"]
        organisms = ["Homo sapiens", "Mus musculus", "Drosophila melanogaster", "Saccharomyces cerevisiae", "Escherichia coli"]
        functions = ["enzyme", "structural", "transport", "signaling", "regulatory"]
        
        for i in range(800):  # 800 proteins
            organism = organisms[i % len(organisms)]
            function = functions[i % len(functions)]
            sequence_length = 100 + (i * 3) % 2000
            protein_lines.append(
                f"PROT:{i:06d}\tProtein {i}\tProtein {i} from {organism} with {function} function\t{organism}\t{sequence_length}\t{function}\tdomain_{i%50}"
            )
        
        protein_file = temp_dir / "protein_sequences.tsv"
        protein_file.write_text("\n".join(protein_lines))
        production_files.append(("protein_tsv", str(protein_file)))
        
        # Measure baseline system performance
        process = psutil.Process()
        initial_memory = process.memory_info().rss
        initial_cpu_percent = process.cpu_percent()
        
        # Simulate production workload with timing
        workload_start = time.time()
        
        # Phase 1: Batch load all files
        print("Phase 1: Initial batch loading...")
        batch_start = time.time()
        batch_results = ontology_manager.load_ontologies([file_info[1] for file_info in production_files])
        batch_time = time.time() - batch_start
        
        # Phase 2: Simulate repeated access patterns
        print("Phase 2: Simulating access patterns...")
        access_start = time.time()
        
        # Simulate 50 random access operations
        import random
        file_paths = [file_info[1] for file_info in production_files]
        
        for _ in range(50):
            # Randomly select 1-3 files to load
            selected_files = random.sample(file_paths, random.randint(1, 3))
            access_results = ontology_manager.load_ontologies(selected_files)
            
            # Verify cache hits
            cache_hits = [r for r in access_results if r.success and r.metadata.get('cache_hit', False)]
            assert len(cache_hits) >= len(access_results) * 0.8, "Should have high cache hit rate in access pattern"
        
        access_time = time.time() - access_start
        
        # Phase 3: Simulate concurrent usage
        print("Phase 3: Simulating concurrent usage...")
        concurrent_start = time.time()
        
        def concurrent_worker(worker_id, iterations=10):
            worker_results = []
            for i in range(iterations):
                # Each worker randomly accesses files
                selected_files = random.sample(file_paths, random.randint(1, 2))
                results = ontology_manager.load_ontologies(selected_files)
                worker_results.extend(results)
                time.sleep(0.01)  # Small delay between operations
            return worker_results
        
        # Run 5 concurrent workers
        with ThreadPoolExecutor(max_workers=5) as executor:
            worker_futures = [
                executor.submit(concurrent_worker, i, 10)
                for i in range(5)
            ]
            
            concurrent_results = []
            for future in as_completed(worker_futures):
                worker_results = future.result()
                concurrent_results.extend(worker_results)
        
        concurrent_time = time.time() - concurrent_start
        total_workload_time = time.time() - workload_start
        
        # Measure final system state
        final_memory = process.memory_info().rss
        final_cpu_percent = process.cpu_percent()
        
        # Analyze production workload results
        successful_batch = [r for r in batch_results if r.success]
        successful_concurrent = [r for r in concurrent_results if r.success]
        
        # Performance assertions
        assert len(successful_batch) >= len(production_files) * 0.8, f"Batch loading success rate too low: {len(successful_batch)}/{len(production_files)}"
        assert len(successful_concurrent) >= len(concurrent_results) * 0.9, "Concurrent access success rate too low"
        
        # Timing assertions
        assert batch_time < 120.0, f"Initial batch loading too slow: {batch_time:.2f}s"
        assert access_time < 30.0, f"Access pattern simulation too slow: {access_time:.2f}s"
        assert concurrent_time < 60.0, f"Concurrent usage too slow: {concurrent_time:.2f}s"
        assert total_workload_time < 200.0, f"Total workload time too long: {total_workload_time:.2f}s"
        
        # Memory usage assertions
        memory_increase = final_memory - initial_memory
        assert memory_increase < 3 * 1024 * 1024 * 1024, f"Memory usage too high: {memory_increase / 1024 / 1024:.2f}MB"
        
        # Verify data integrity after production workload
        stats = ontology_manager.get_statistics()
        
        # Should have loaded multiple formats
        assert len(stats['formats_loaded']) >= 3, f"Should detect multiple formats: {stats['formats_loaded']}"
        
        # Cache should be effective
        cache_hit_rate = stats['cache_hits'] / (stats['cache_hits'] + stats['cache_misses'])
        assert cache_hit_rate >= 0.7, f"Cache hit rate too low: {cache_hit_rate:.2%}"
        
        # Should have processed significant volume
        assert stats['total_loads'] >= 100, f"Should have processed many loads: {stats['total_loads']}"
        
        # Verify system remains responsive
        final_ontology_count = len(ontology_manager.ontologies)
        assert final_ontology_count >= len(successful_batch), "All successful ontologies should be available"
        
        # Test final operations still work
        test_files = [file_info[1] for file_info in production_files[:2]]
        final_test_results = ontology_manager.load_ontologies(test_files)
        assert all(r.success for r in final_test_results), "System should remain operational after production workload"
        
        print(f"Production workload completed:")
        print(f"  Total time: {total_workload_time:.2f}s")
        print(f"  Memory usage: {memory_increase / 1024 / 1024:.2f}MB")
        print(f"  Cache hit rate: {cache_hit_rate:.2%}")
        print(f"  Total loads: {stats['total_loads']}")
        print(f"  Ontologies loaded: {final_ontology_count}")


if __name__ == "__main__":
    pytest.main([__file__])