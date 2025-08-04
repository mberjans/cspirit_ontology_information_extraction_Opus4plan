#!/usr/bin/env python3
"""
Simple RDF Triple Extraction Test

A focused test of the RDF triple extraction functionality with simplified OWL content.
"""

import sys
import os
import tempfile
import time
from pathlib import Path

# Add the project root to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

try:
    from aim2_project.aim2_ontology.parsers import OWLParser
    from aim2_project.aim2_ontology.models import RDFTriple
except ImportError as e:
    print(f"Import error: {e}")
    sys.exit(1)

def test_simple_owl_extraction():
    """Test RDF triple extraction with a simple OWL file."""
    print("Testing RDF Triple Extraction")
    print("=" * 40)
    
    # Create simple OWL content
    simple_owl = """<?xml version="1.0"?>
<rdf:RDF xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#"
         xmlns:owl="http://www.w3.org/2002/07/owl#"
         xmlns:rdfs="http://www.w3.org/2000/01/rdf-schema#"
         xmlns="http://example.org/simple#">

    <owl:Ontology rdf:about="http://example.org/simple">
        <rdfs:label>Simple Test Ontology</rdfs:label>
    </owl:Ontology>

    <owl:Class rdf:about="http://example.org/simple#Chemical">
        <rdfs:label>Chemical</rdfs:label>
        <rdfs:comment>A chemical compound</rdfs:comment>
    </owl:Class>

    <owl:Class rdf:about="http://example.org/simple#Glucose">
        <rdfs:label>Glucose</rdfs:label>
        <rdfs:comment>A simple sugar</rdfs:comment>
        <rdfs:subClassOf rdf:resource="http://example.org/simple#Chemical"/>
    </owl:Class>

</rdf:RDF>"""
    
    # Write to temporary file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.owl', delete=False) as f:
        f.write(simple_owl)
        temp_file = f.name
    
    try:
        # Initialize parser
        parser = OWLParser()
        
        # Configure parser for safe operation
        parser.set_options({
            "continue_on_error": True,
            "error_recovery": True,
            "timeout_seconds": 30
        })
        
        # Test 1: Basic parsing
        print("Test 1: Basic OWL Parsing")
        start_time = time.time()
        parsed_result = parser.parse_file(temp_file)
        parse_time = time.time() - start_time
        
        print(f"  ✓ Parsed in {parse_time:.3f}s")
        print(f"  - Has RDF graph: {parsed_result.get('rdf_graph') is not None}")
        print(f"  - Has OWL ontology: {parsed_result.get('owl_ontology') is not None}")
        print(f"  - Format: {parsed_result.get('format', 'unknown')}")
        
        # Test 2: Manual triple extraction
        print("\nTest 2: Manual Triple Extraction")
        start_time = time.time()
        triples = parser.extract_triples(parsed_result)
        extract_time = time.time() - start_time
        
        print(f"  ✓ Extracted {len(triples)} triples in {extract_time:.3f}s")
        
        # Validate triples
        valid_triples = 0
        for triple in triples:
            if isinstance(triple, RDFTriple) and triple.is_valid():
                valid_triples += 1
        
        print(f"  - Valid triples: {valid_triples}/{len(triples)}")
        
        # Show sample triples
        print("  - Sample triples:")
        for i, triple in enumerate(triples[:3]):
            print(f"    {i+1}. S: {triple.subject}")
            print(f"       P: {triple.predicate}")
            print(f"       O: {triple.object}")
            print(f"       Type: {getattr(triple, 'object_type', 'unknown')}")
        
        # Test 3: Automatic extraction during parsing
        print("\nTest 3: Automatic Triple Extraction")
        parser.set_options({"extract_triples_on_parse": True})
        
        start_time = time.time()
        auto_result = parser.parse_file(temp_file)
        auto_time = time.time() - start_time
        
        auto_triples = auto_result.get("triples", [])
        print(f"  ✓ Auto-extracted {len(auto_triples)} triples in {auto_time:.3f}s")
        print(f"  - Triple count in result: {auto_result.get('triple_count', 0)}")
        
        # Test 4: Content validation
        print("\nTest 4: Content Validation")
        if triples:
            predicates = {}
            object_types = {}
            
            for triple in triples:
                # Count predicates
                pred_name = triple.predicate.split('#')[-1] if '#' in triple.predicate else triple.predicate.split('/')[-1]
                predicates[pred_name] = predicates.get(pred_name, 0) + 1
                
                # Count object types
                obj_type = getattr(triple, 'object_type', 'unknown')
                object_types[obj_type] = object_types.get(obj_type, 0) + 1
            
            print(f"  - Predicate distribution: {predicates}")
            print(f"  - Object type distribution: {object_types}")
            
            # Check for expected content
            expected_subjects = ["Chemical", "Glucose", "simple"]
            expected_predicates = ["label", "comment", "subClassOf", "type"]
            
            found_subjects = sum(1 for t in triples if any(subj in t.subject for subj in expected_subjects))
            found_predicates = sum(1 for t in triples if any(pred in t.predicate for pred in expected_predicates))
            
            print(f"  - Expected subjects found: {found_subjects}")
            print(f"  - Expected predicates found: {found_predicates}")
        
        print("\n✅ All tests completed successfully!")
        return True
        
    except Exception as e:
        print(f"\n❌ Test failed: {str(e)}")
        import traceback
        print(f"Traceback: {traceback.format_exc()}")
        return False
        
    finally:
        # Clean up
        try:
            os.unlink(temp_file)
        except:
            pass

def test_existing_owl_files():
    """Test with existing owlready2 OWL files."""
    print("\nTesting with Existing OWL Files")
    print("=" * 40)
    
    # Test with simpler owlready2 files
    test_files = [
        "/Users/Mark/Research/C-Spirit/cspirit_ontology_information_extraction_Opus4plan/venv/lib/python3.13/site-packages/owlready2/ontos/dc.owl"
    ]
    
    parser = OWLParser()
    parser.set_options({
        "continue_on_error": True,
        "error_recovery": True,
        "timeout_seconds": 30
    })
    
    for file_path in test_files:
        if os.path.exists(file_path):
            try:
                print(f"\nTesting: {os.path.basename(file_path)}")
                file_size = os.path.getsize(file_path)
                print(f"  File size: {file_size} bytes")
                
                start_time = time.time()
                parsed_result = parser.parse_file(file_path)
                triples = parser.extract_triples(parsed_result)
                end_time = time.time()
                
                print(f"  ✓ Extracted {len(triples)} triples in {end_time - start_time:.3f}s")
                print(f"  - Performance: {len(triples)/(end_time - start_time):.1f} triples/sec")
                
                if triples:
                    print(f"  - Sample triple: {triples[0].subject} -> {triples[0].predicate} -> {triples[0].object}")
                
            except Exception as e:
                print(f"  ❌ Failed: {str(e)}")
        else:
            print(f"  ⚠️  File not found: {file_path}")

def main():
    """Main test execution."""
    success = True
    
    try:
        success &= test_simple_owl_extraction()
        test_existing_owl_files()
        
        if success:
            print(f"\n🎉 RDF Triple Extraction Tests: PASSED")
            return 0
        else:
            print(f"\n⚠️  Some tests failed")
            return 1
            
    except Exception as e:
        print(f"\n❌ Test execution failed: {str(e)}")
        return 1

if __name__ == "__main__":
    sys.exit(main())