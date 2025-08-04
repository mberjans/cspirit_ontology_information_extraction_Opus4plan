#!/usr/bin/env python3
"""
RDF Triple Extraction Demo

Demonstrates the working RDF triple extraction functionality with real examples.
"""

import sys
from pathlib import Path

# Add the project root to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from aim2_project.aim2_ontology.parsers import OWLParser
from aim2_project.aim2_ontology.models import RDFTriple

def demo_basic_extraction():
    """Demonstrate basic RDF triple extraction."""
    print("🔬 RDF Triple Extraction Demo")
    print("=" * 50)
    
    # Create a sample ontology in RDF/XML
    sample_ontology = '''<?xml version="1.0"?>
<rdf:RDF xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#"
         xmlns:owl="http://www.w3.org/2002/07/owl#"
         xmlns:rdfs="http://www.w3.org/2000/01/rdf-schema#"
         xmlns:chebi="http://purl.obolibrary.org/obo/">

    <owl:Ontology rdf:about="http://example.org/chemistry">
        <rdfs:label>Chemistry Ontology Demo</rdfs:label>
        <rdfs:comment>A demo ontology for testing RDF triple extraction</rdfs:comment>
    </owl:Ontology>

    <owl:Class rdf:about="http://example.org/chemistry#Chemical">
        <rdfs:label>Chemical</rdfs:label>
        <rdfs:comment>A chemical compound or substance</rdfs:comment>
    </owl:Class>

    <owl:Class rdf:about="http://example.org/chemistry#Glucose">
        <rdfs:label>Glucose</rdfs:label>
        <rdfs:comment>A simple sugar with the formula C6H12O6</rdfs:comment>
        <rdfs:subClassOf rdf:resource="http://example.org/chemistry#Chemical"/>
    </owl:Class>

    <owl:ObjectProperty rdf:about="http://example.org/chemistry#hasFormula">
        <rdfs:label>has formula</rdfs:label>
        <rdfs:domain rdf:resource="http://example.org/chemistry#Chemical"/>
        <rdfs:range rdf:resource="http://www.w3.org/2001/XMLSchema#string"/>
    </owl:ObjectProperty>

</rdf:RDF>'''

    # Initialize parser
    parser = OWLParser()
    parser.set_options({
        "continue_on_error": True,
        "error_recovery": True
    })
    
    print("📝 Sample Ontology:")
    print("  - Chemistry ontology with Chemical and Glucose classes")
    print("  - Contains labels, comments, and relationships")
    print("  - Includes object property hasFormula")
    
    # Parse the ontology
    print("\n🔍 Step 1: Parsing OWL content...")
    parsed_result = parser.parse(sample_ontology)
    
    print(f"  ✓ Successfully parsed")
    print(f"  ✓ Format detected: {parsed_result.get('format')}")
    print(f"  ✓ Content size: {parsed_result.get('content_size')} characters")
    print(f"  ✓ Has RDF graph: {parsed_result.get('rdf_graph') is not None}")
    print(f"  ✓ Has OWL ontology: {parsed_result.get('owl_ontology') is not None}")
    
    # Extract triples manually
    print("\n🔍 Step 2: Extracting RDF triples...")
    triples = parser.extract_triples(parsed_result)
    
    print(f"  ✓ Extracted {len(triples)} RDF triples")
    
    # Analyze the triples
    print("\n📊 Step 3: Analyzing extracted triples...")
    
    predicates = {}
    object_types = {}
    subjects = set()
    
    for triple in triples:
        # Count predicates
        pred_name = triple.predicate.split('#')[-1] if '#' in triple.predicate else triple.predicate.split('/')[-1]
        predicates[pred_name] = predicates.get(pred_name, 0) + 1
        
        # Count object types
        obj_type = getattr(triple, 'object_type', 'unknown')
        object_types[obj_type] = object_types.get(obj_type, 0) + 1
        
        # Collect subjects
        subj_name = triple.subject.split('#')[-1] if '#' in triple.subject else triple.subject.split('/')[-1]
        subjects.add(subj_name)
    
    print(f"  📈 Predicate distribution: {dict(list(predicates.items())[:5])}")
    print(f"  📈 Object type distribution: {object_types}")
    print(f"  📈 Unique subjects: {len(subjects)}")
    print(f"  📈 Sample subjects: {list(subjects)[:5]}")
    
    # Show sample triples
    print("\n🔍 Step 4: Sample extracted triples...")
    for i, triple in enumerate(triples[:5]):
        subj = triple.subject.split('#')[-1] if '#' in triple.subject else triple.subject.split('/')[-1]
        pred = triple.predicate.split('#')[-1] if '#' in triple.predicate else triple.predicate.split('/')[-1]
        obj = triple.object[:50] + "..." if len(triple.object) > 50 else triple.object
        
        print(f"  {i+1}. {subj} --{pred}--> {obj}")
        print(f"     Type: {getattr(triple, 'object_type', 'unknown')}")
        print(f"     Confidence: {triple.confidence}")
    
    # Test automatic extraction
    print("\n🔍 Step 5: Testing automatic extraction...")
    parser.set_options({"extract_triples_on_parse": True})
    
    auto_result = parser.parse(sample_ontology)
    auto_triples = auto_result.get("triples", [])
    
    print(f"  ✓ Auto-extracted {len(auto_triples)} triples")
    print(f"  ✓ Triple count in result: {auto_result.get('triple_count', 0)}")
    print(f"  ✓ Matches manual extraction: {len(auto_triples) == len(triples)}")
    
    return len(triples) > 0

def demo_triple_features():
    """Demonstrate RDFTriple model features."""
    print("\n🧬 RDFTriple Model Features Demo")
    print("=" * 50)
    
    # Create sample triples with different features
    triples = [
        # URI triple
        RDFTriple(
            subject="http://example.org/Glucose",
            predicate="http://www.w3.org/1999/02/22-rdf-syntax-ns#type",
            object="http://example.org/Chemical",
            source="demo",
            confidence=1.0
        ),
        
        # Literal triple with datatype
        RDFTriple(
            subject="http://example.org/Glucose",
            predicate="http://example.org/hasFormula",
            object="C6H12O6",
            object_type="literal",
            object_datatype="http://www.w3.org/2001/XMLSchema#string",
            source="demo",
            confidence=0.95
        ),
        
        # Literal triple with language
        RDFTriple(
            subject="http://example.org/Glucose",
            predicate="http://www.w3.org/2000/01/rdf-schema#label",
            object="Glucose",
            object_type="literal",
            object_language="en",
            source="demo",
            confidence=1.0,
            metadata={"extraction_method": "manual", "source_line": 42}
        )
    ]
    
    print("📋 Created sample triples with different features:")
    
    for i, triple in enumerate(triples, 1):
        print(f"\n  Triple {i}:")
        print(f"    Subject: {triple.subject}")
        print(f"    Predicate: {triple.predicate}")
        print(f"    Object: {triple.object}")
        print(f"    Object Type: {getattr(triple, 'object_type', 'uri')}")
        print(f"    Confidence: {triple.confidence}")
        print(f"    Is Valid: {triple.is_valid()}")
        
        if hasattr(triple, 'object_datatype') and triple.object_datatype:
            print(f"    Datatype: {triple.object_datatype}")
        
        if hasattr(triple, 'object_language') and triple.object_language:
            print(f"    Language: {triple.object_language}")
        
        if hasattr(triple, 'metadata') and triple.metadata:
            print(f"    Metadata: {triple.metadata}")
    
    # Test serialization features
    print("\n🔄 Testing serialization features:")
    sample_triple = triples[0]
    
    try:
        # Test dictionary serialization
        triple_dict = sample_triple.to_dict()
        print(f"  ✓ Dictionary serialization: {len(triple_dict)} fields")
        
        # Test JSON serialization
        triple_json = sample_triple.to_json()
        print(f"  ✓ JSON serialization: {len(triple_json)} characters")
        
        # Test Turtle serialization
        turtle_str = sample_triple.to_turtle()
        print(f"  ✓ Turtle serialization: {turtle_str}")
        
    except Exception as e:
        print(f"  ❌ Serialization error: {e}")
    
    return True

def main():
    """Run the complete demo."""
    try:
        success1 = demo_basic_extraction()
        success2 = demo_triple_features()
        
        print("\n" + "=" * 50)
        if success1 and success2:
            print("🎉 RDF Triple Extraction Demo: SUCCESS!")
            print("\n✅ Key capabilities demonstrated:")
            print("  • Parse OWL/RDF content from strings")
            print("  • Extract RDF triples with comprehensive metadata")
            print("  • Automatic triple extraction during parsing")
            print("  • Rich RDFTriple model with validation")
            print("  • Multiple serialization formats")
            print("  • Confidence scoring and source tracking")
            return 0
        else:
            print("⚠️ Some demo features failed")
            return 1
            
    except Exception as e:
        print(f"\n❌ Demo failed: {e}")
        import traceback
        print(f"Traceback: {traceback.format_exc()}")
        return 1

if __name__ == "__main__":
    sys.exit(main())