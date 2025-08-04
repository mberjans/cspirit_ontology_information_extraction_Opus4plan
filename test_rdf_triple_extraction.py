#!/usr/bin/env python3
"""
RDF Triple Extraction Testing Script

This script tests the RDF triple extraction functionality with actual sample OWL files
to ensure everything works correctly. It validates the extract_triples() method and
integration with the parse() method.

Usage:
    python test_rdf_triple_extraction.py

Author: AIM2 Development Team
Date: 2025-01-04
"""

import sys
import os
import tempfile
import time
from pathlib import Path
from typing import List, Dict, Any

# Add the project root to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

try:
    from aim2_project.aim2_ontology.parsers import OWLParser
    from aim2_project.aim2_ontology.models import RDFTriple
    from aim2_project.aim2_utils.logger_factory import get_logger
except ImportError as e:
    print(f"Import error: {e}")
    print("Make sure the project is properly installed")
    sys.exit(1)

# Set up logging
logger = get_logger("rdf_triple_extraction_test")

class RDFTripleExtractionTester:
    """Test class for RDF triple extraction functionality."""
    
    def __init__(self):
        self.parser = OWLParser()
        self.results = []
        self.errors = []
        
    def create_sample_owl_files(self) -> Dict[str, str]:
        """Create sample OWL files for testing."""
        temp_dir = tempfile.mkdtemp()
        files = {}
        
        # Simple OWL/XML file
        simple_owl = f"""<?xml version="1.0"?>
<rdf:RDF xmlns="http://example.org/simple#"
         xml:base="http://example.org/simple"
         xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#"
         xmlns:owl="http://www.w3.org/2002/07/owl#"
         xmlns:rdfs="http://www.w3.org/2000/01/rdf-schema#">

    <owl:Ontology rdf:about="http://example.org/simple">
        <rdfs:label>Simple Test Ontology</rdfs:label>
        <rdfs:comment>A simple ontology for testing RDF triple extraction</rdfs:comment>
    </owl:Ontology>

    <owl:Class rdf:about="http://example.org/simple#Chemical">
        <rdfs:label>Chemical</rdfs:label>
        <rdfs:comment>A chemical compound</rdfs:comment>
    </owl:Class>

    <owl:Class rdf:about="http://example.org/simple#Glucose">
        <rdfs:label>Glucose</rdfs:label>
        <rdfs:subClassOf rdf:resource="http://example.org/simple#Chemical"/>
        <rdfs:comment>A simple sugar molecule</rdfs:comment>
    </owl:Class>

</rdf:RDF>"""
        
        simple_file = Path(temp_dir) / "simple_ontology.owl"
        simple_file.write_text(simple_owl)
        files["simple"] = str(simple_file)
        
        # Complex OWL/XML file
        complex_owl = f"""<?xml version="1.0"?>
<rdf:RDF xmlns="http://example.org/complex#"
         xml:base="http://example.org/complex"
         xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#"
         xmlns:owl="http://www.w3.org/2002/07/owl#"
         xmlns:rdfs="http://www.w3.org/2000/01/rdf-schema#"
         xmlns:dc="http://purl.org/dc/elements/1.1/">

    <owl:Ontology rdf:about="http://example.org/complex">
        <dc:title>Complex Test Ontology</dc:title>
        <dc:description>A complex ontology for comprehensive testing</dc:description>
        <owl:versionInfo>1.0.0</owl:versionInfo>
    </owl:Ontology>

    <!-- Classes -->
    <owl:Class rdf:about="http://example.org/complex#Entity">
        <rdfs:label>Entity</rdfs:label>
        <rdfs:comment>Base class for all entities</rdfs:comment>
    </owl:Class>

    <owl:Class rdf:about="http://example.org/complex#Chemical">
        <rdfs:label>Chemical</rdfs:label>
        <rdfs:subClassOf rdf:resource="http://example.org/complex#Entity"/>
        <owl:disjointWith rdf:resource="http://example.org/complex#BiologicalProcess"/>
    </owl:Class>

    <owl:Class rdf:about="http://example.org/complex#BiologicalProcess">
        <rdfs:label>Biological Process</rdfs:label>
        <rdfs:subClassOf rdf:resource="http://example.org/complex#Entity"/>
    </owl:Class>

    <!-- Properties -->
    <owl:ObjectProperty rdf:about="http://example.org/complex#participatesIn">
        <rdfs:label>participates in</rdfs:label>
        <rdfs:domain rdf:resource="http://example.org/complex#Chemical"/>
        <rdfs:range rdf:resource="http://example.org/complex#BiologicalProcess"/>
    </owl:ObjectProperty>

    <owl:DatatypeProperty rdf:about="http://example.org/complex#hasName">
        <rdfs:label>has name</rdfs:label>
        <rdfs:domain rdf:resource="http://example.org/complex#Entity"/>
        <rdfs:range rdf:resource="http://www.w3.org/2001/XMLSchema#string"/>
    </owl:DatatypeProperty>

    <!-- Individuals -->
    <owl:NamedIndividual rdf:about="http://example.org/complex#glucose">
        <rdf:type rdf:resource="http://example.org/complex#Chemical"/>
        <rdfs:label>Glucose Instance</rdfs:label>
    </owl:NamedIndividual>

</rdf:RDF>"""
        
        complex_file = Path(temp_dir) / "complex_ontology.owl"
        complex_file.write_text(complex_owl)
        files["complex"] = str(complex_file)
        
        # Turtle format file
        turtle_content = """@prefix : <http://example.org/turtle#> .
@prefix owl: <http://www.w3.org/2002/07/owl#> .
@prefix rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#> .
@prefix rdfs: <http://www.w3.org/2000/01/rdf-schema#> .

<http://example.org/turtle> a owl:Ontology ;
    rdfs:label "Turtle Test Ontology" ;
    rdfs:comment "A test ontology in Turtle format" .

:Chemical a owl:Class ;
    rdfs:label "Chemical" ;
    rdfs:comment "A chemical compound in Turtle format" .

:Glucose a owl:Class ;
    rdfs:label "Glucose" ;
    rdfs:subClassOf :Chemical .

:hasFormula a owl:DatatypeProperty ;
    rdfs:label "has formula" ;
    rdfs:domain :Chemical ;
    rdfs:range <http://www.w3.org/2001/XMLSchema#string> .
"""
        
        turtle_file = Path(temp_dir) / "turtle_ontology.ttl"
        turtle_file.write_text(turtle_content)
        files["turtle"] = str(turtle_file)
        
        return files
    
    def test_basic_triple_extraction(self, file_path: str, file_type: str) -> Dict[str, Any]:
        """Test basic triple extraction from a file."""
        logger.info(f"Testing basic triple extraction from {file_type} file: {file_path}")
        
        try:
            start_time = time.time()
            
            # Configure parser with timeout and error recovery
            self.parser.set_options({
                "continue_on_error": True,
                "error_recovery": True,
                "timeout_seconds": 30
            })
            
            # Parse the file
            parsed_result = self.parser.parse_file(file_path)
            
            # Extract triples directly
            triples = self.parser.extract_triples(parsed_result)
            
            end_time = time.time()
            processing_time = end_time - start_time
            
            result = {
                "file_type": file_type,
                "file_path": file_path,
                "success": True,
                "triple_count": len(triples),
                "processing_time": processing_time,
                "triples": triples[:10],  # Store first 10 triples for inspection
                "has_rdf_graph": parsed_result.get("rdf_graph") is not None,
                "has_owl_ontology": parsed_result.get("owl_ontology") is not None,
                "errors": []
            }
            
            logger.info(f"Successfully extracted {len(triples)} triples in {processing_time:.3f}s")
            
            # Validate triples
            for i, triple in enumerate(triples[:5]):  # Check first 5 triples
                if not isinstance(triple, RDFTriple):
                    result["errors"].append(f"Triple {i} is not an RDFTriple instance")
                elif not triple.is_valid():
                    result["errors"].append(f"Triple {i} failed validation")
            
            return result
            
        except Exception as e:
            logger.error(f"Error in basic triple extraction: {str(e)}")
            return {
                "file_type": file_type,
                "file_path": file_path,
                "success": False,
                "error": str(e),
                "triple_count": 0,
                "processing_time": 0,
                "errors": [str(e)]
            }
    
    def test_parse_with_auto_extraction(self, file_path: str, file_type: str) -> Dict[str, Any]:
        """Test parsing with automatic triple extraction enabled."""
        logger.info(f"Testing parse with auto-extraction from {file_type} file: {file_path}")
        
        try:
            # Configure parser to extract triples automatically
            self.parser.set_options({"extract_triples_on_parse": True})
            
            start_time = time.time()
            parsed_result = self.parser.parse_file(file_path)
            end_time = time.time()
            
            processing_time = end_time - start_time
            
            result = {
                "file_type": file_type,
                "file_path": file_path,
                "success": True,
                "processing_time": processing_time,
                "has_triples_in_result": "triples" in parsed_result,
                "has_triple_count": "triple_count" in parsed_result,
                "auto_extracted_count": parsed_result.get("triple_count", 0),
                "errors": []
            }
            
            if "triples" in parsed_result:
                triples = parsed_result["triples"]
                result["sample_triples"] = triples[:5]  # First 5 for inspection
                logger.info(f"Auto-extraction produced {len(triples)} triples in {processing_time:.3f}s")
            else:
                result["errors"].append("No triples found in parsed result")
                
            return result
            
        except Exception as e:
            logger.error(f"Error in parse with auto-extraction: {str(e)}")
            return {
                "file_type": file_type,
                "file_path": file_path,
                "success": False,
                "error": str(e),
                "processing_time": 0,
                "errors": [str(e)]
            }
        finally:
            # Reset parser options
            self.parser.set_options({"extract_triples_on_parse": False})
    
    def test_triple_content_validation(self, triples: List[RDFTriple]) -> Dict[str, Any]:
        """Validate the content and quality of extracted triples."""
        logger.info(f"Validating content of {len(triples)} triples")
        
        validation_result = {
            "total_count": len(triples),
            "valid_count": 0,
            "invalid_count": 0,
            "type_distribution": {},
            "predicate_distribution": {},
            "confidence_stats": {"min": 1.0, "max": 0.0, "avg": 0.0},
            "has_metadata": 0,
            "has_namespaces": 0,
            "issues": []
        }
        
        total_confidence = 0
        
        for i, triple in enumerate(triples):
            try:
                # Basic validation
                if triple.is_valid():
                    validation_result["valid_count"] += 1
                else:
                    validation_result["invalid_count"] += 1
                    validation_result["issues"].append(f"Triple {i} failed validation")
                
                # Collect statistics
                obj_type = getattr(triple, 'object_type', 'unknown')
                validation_result["type_distribution"][obj_type] = \
                    validation_result["type_distribution"].get(obj_type, 0) + 1
                
                # Predicate distribution (simplified)
                pred = triple.predicate
                if pred:
                    pred_name = pred.split('#')[-1] if '#' in pred else pred.split('/')[-1]
                    validation_result["predicate_distribution"][pred_name] = \
                        validation_result["predicate_distribution"].get(pred_name, 0) + 1
                
                # Confidence stats
                conf = getattr(triple, 'confidence', 1.0)
                validation_result["confidence_stats"]["min"] = min(
                    validation_result["confidence_stats"]["min"], conf
                )
                validation_result["confidence_stats"]["max"] = max(
                    validation_result["confidence_stats"]["max"], conf
                )
                total_confidence += conf
                
                # Metadata presence
                if hasattr(triple, 'metadata') and triple.metadata:
                    validation_result["has_metadata"] += 1
                
                if hasattr(triple, 'namespace_prefixes') and triple.namespace_prefixes:
                    validation_result["has_namespaces"] += 1
                    
            except Exception as e:
                validation_result["issues"].append(f"Error validating triple {i}: {str(e)}")
        
        # Calculate average confidence
        if len(triples) > 0:
            validation_result["confidence_stats"]["avg"] = total_confidence / len(triples)
        
        logger.info(f"Validation complete: {validation_result['valid_count']} valid, "
                   f"{validation_result['invalid_count']} invalid triples")
        
        return validation_result
    
    def test_error_handling(self) -> Dict[str, Any]:
        """Test error handling with malformed OWL content."""
        logger.info("Testing error handling with malformed content")
        
        malformed_owl = """<?xml version="1.0"?>
<rdf:RDF xmlns="http://example.org/malformed#"
         xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#"
         xmlns:owl="http://www.w3.org/2002/07/owl#">
    
    <owl:Ontology rdf:about="http://example.org/malformed">
        <rdfs:label>Malformed Ontology</rdfs:label>
        <!-- Missing namespace for rdfs -->
    </owl:Ontology>
    
    <owl:Class rdf:about="http://example.org/malformed#TestClass">
        <!-- Unclosed tag -->
        <owl:someProperty>
    </owl:Class>
    
<!-- Missing closing RDF tag -->
"""
        
        temp_file = tempfile.NamedTemporaryFile(mode='w', suffix='.owl', delete=False)
        temp_file.write(malformed_owl)
        temp_file.close()
        
        try:
            # Test with continue_on_error enabled
            self.parser.set_options({"continue_on_error": True})
            
            result = {
                "test_type": "error_handling",
                "continue_on_error": True,
                "success": False,
                "errors": []
            }
            
            try:
                parsed_result = self.parser.parse_file(temp_file.name)
                triples = self.parser.extract_triples(parsed_result)
                result["success"] = True
                result["triple_count"] = len(triples)
                logger.info(f"Error handling test: extracted {len(triples)} triples despite malformed input")
            except Exception as e:
                result["errors"].append(str(e))
                logger.info(f"Error handling test: caught expected error: {str(e)}")
            
            return result
            
        finally:
            # Clean up
            os.unlink(temp_file.name)
            self.parser.set_options({"continue_on_error": False})
    
    def test_performance_with_owlready2_files(self) -> Dict[str, Any]:
        """Test performance with actual owlready2 sample files."""
        logger.info("Testing performance with owlready2 sample files")
        
        # Use the owlready2 sample files we found
        owlready2_files = [
            "/Users/Mark/Research/C-Spirit/cspirit_ontology_information_extraction_Opus4plan/venv/lib/python3.13/site-packages/owlready2/ontos/dc.owl",
            "/Users/Mark/Research/C-Spirit/cspirit_ontology_information_extraction_Opus4plan/venv/lib/python3.13/site-packages/owlready2/ontos/dcterms.owl",
        ]
        
        results = []
        
        for file_path in owlready2_files:
            if os.path.exists(file_path):
                try:
                    start_time = time.time()
                    parsed_result = self.parser.parse_file(file_path)
                    triples = self.parser.extract_triples(parsed_result)
                    end_time = time.time()
                    
                    result = {
                        "file": os.path.basename(file_path),
                        "file_size": os.path.getsize(file_path),
                        "success": True,
                        "triple_count": len(triples),
                        "processing_time": end_time - start_time,
                        "triples_per_second": len(triples) / (end_time - start_time) if (end_time - start_time) > 0 else 0
                    }
                    
                    logger.info(f"Performance test {os.path.basename(file_path)}: "
                               f"{len(triples)} triples in {end_time - start_time:.3f}s "
                               f"({result['triples_per_second']:.1f} triples/sec)")
                    
                    results.append(result)
                    
                except Exception as e:
                    logger.error(f"Performance test failed for {file_path}: {str(e)}")
                    results.append({
                        "file": os.path.basename(file_path),
                        "success": False,
                        "error": str(e)
                    })
        
        return {"performance_tests": results}
    
    def run_all_tests(self) -> Dict[str, Any]:
        """Run all RDF triple extraction tests."""
        logger.info("Starting comprehensive RDF triple extraction tests")
        
        # Create sample files
        sample_files = self.create_sample_owl_files()
        
        all_results = {
            "test_summary": {
                "start_time": time.time(),
                "total_tests": 0,
                "successful_tests": 0,
                "failed_tests": 0
            },
            "basic_extraction_tests": [],
            "auto_extraction_tests": [],
            "validation_results": [],
            "error_handling_test": {},
            "performance_tests": {}
        }
        
        try:
            # Test basic extraction for each file type
            for file_type, file_path in sample_files.items():
                all_results["test_summary"]["total_tests"] += 1
                result = self.test_basic_triple_extraction(file_path, file_type)
                all_results["basic_extraction_tests"].append(result)
                
                if result["success"]:
                    all_results["test_summary"]["successful_tests"] += 1
                    
                    # Validate extracted triples
                    if result["triples"]:
                        validation = self.test_triple_content_validation(result["triples"])
                        validation["source_file"] = file_type
                        all_results["validation_results"].append(validation)
                else:
                    all_results["test_summary"]["failed_tests"] += 1
            
            # Test auto-extraction
            for file_type, file_path in sample_files.items():
                all_results["test_summary"]["total_tests"] += 1
                result = self.test_parse_with_auto_extraction(file_path, file_type)
                all_results["auto_extraction_tests"].append(result)
                
                if result["success"]:
                    all_results["test_summary"]["successful_tests"] += 1
                else:
                    all_results["test_summary"]["failed_tests"] += 1
            
            # Test error handling
            all_results["test_summary"]["total_tests"] += 1
            error_result = self.test_error_handling()
            all_results["error_handling_test"] = error_result
            
            if error_result.get("success", False):
                all_results["test_summary"]["successful_tests"] += 1
            else:
                all_results["test_summary"]["failed_tests"] += 1
            
            # Test performance with real files
            all_results["test_summary"]["total_tests"] += 1
            perf_result = self.test_performance_with_owlready2_files()
            all_results["performance_tests"] = perf_result
            
            if perf_result and perf_result.get("performance_tests"):
                all_results["test_summary"]["successful_tests"] += 1
            else:
                all_results["test_summary"]["failed_tests"] += 1
                
        except Exception as e:
            logger.error(f"Test execution error: {str(e)}")
            all_results["execution_error"] = str(e)
            import traceback
            logger.error(f"Traceback: {traceback.format_exc()}")
        
        finally:
            # Clean up sample files
            for file_path in sample_files.values():
                try:
                    os.unlink(file_path)
                except:
                    pass
        
        all_results["test_summary"]["end_time"] = time.time()
        all_results["test_summary"]["total_time"] = (
            all_results["test_summary"]["end_time"] - all_results["test_summary"]["start_time"]
        )
        
        return all_results
    
    def print_results(self, results: Dict[str, Any]):
        """Print formatted test results."""
        print("\n" + "="*80)
        print("RDF TRIPLE EXTRACTION TEST RESULTS")
        print("="*80)
        
        summary = results["test_summary"]
        print(f"\nTEST SUMMARY:")
        print(f"  Total Tests: {summary['total_tests']}")
        print(f"  Successful: {summary['successful_tests']}")
        print(f"  Failed: {summary['failed_tests']}")
        success_rate = (summary['successful_tests']/summary['total_tests']*100) if summary['total_tests'] > 0 else 0
        print(f"  Success Rate: {success_rate:.1f}%")
        print(f"  Total Time: {summary['total_time']:.2f} seconds")
        
        print(f"\nBASIC EXTRACTION TESTS:")
        for test in results["basic_extraction_tests"]:
            status = "✓" if test["success"] else "✗"
            print(f"  {status} {test['file_type']}: {test['triple_count']} triples "
                  f"({test['processing_time']:.3f}s)")
            if test.get("errors"):
                for error in test["errors"]:
                    print(f"    Error: {error}")
        
        print(f"\nAUTO-EXTRACTION TESTS:")
        for test in results["auto_extraction_tests"]:
            status = "✓" if test["success"] else "✗"
            count = test.get("auto_extracted_count", "N/A")
            print(f"  {status} {test['file_type']}: {count} triples auto-extracted "
                  f"({test['processing_time']:.3f}s)")
        
        print(f"\nTRIPLE VALIDATION RESULTS:")
        for validation in results["validation_results"]:
            print(f"  {validation['source_file']}: {validation['valid_count']}/{validation['total_count']} valid")
            print(f"    Types: {validation['type_distribution']}")
            print(f"    Avg Confidence: {validation['confidence_stats']['avg']:.2f}")
            if validation["issues"]:
                print(f"    Issues: {len(validation['issues'])}")
        
        error_test = results["error_handling_test"]
        if error_test:
            status = "✓" if error_test.get("success", False) else "✗"
            print(f"\nERROR HANDLING TEST: {status}")
            if error_test.get("errors"):
                print(f"  Errors handled: {len(error_test['errors'])}")
        
        perf_tests = results["performance_tests"].get("performance_tests", [])
        if perf_tests:
            print(f"\nPERFORMANCE TESTS:")
            for test in perf_tests:
                if test["success"]:
                    print(f"  ✓ {test['file']}: {test['triple_count']} triples "
                          f"({test['triples_per_second']:.1f} triples/sec)")
                else:
                    print(f"  ✗ {test['file']}: {test.get('error', 'Unknown error')}")
        
        print("\n" + "="*80)


def main():
    """Main test execution function."""
    print("RDF Triple Extraction Testing Script")
    print("====================================")
    
    try:
        tester = RDFTripleExtractionTester()
        results = tester.run_all_tests()
        tester.print_results(results)
        
        # Return appropriate exit code
        summary = results["test_summary"]
        if summary["failed_tests"] == 0:
            print("\n🎉 All tests passed!")
            return 0
        else:
            print(f"\n⚠️  {summary['failed_tests']} tests failed")
            return 1
            
    except Exception as e:
        logger.error(f"Test execution failed: {str(e)}")
        print(f"\n❌ Test execution failed: {str(e)}")
        return 1


if __name__ == "__main__":
    sys.exit(main())