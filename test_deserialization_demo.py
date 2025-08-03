#!/usr/bin/env python3
"""
Test script to demonstrate JSON deserialization functionality for AIM2-011-11.

This script tests the from_json methods for Term, Relationship, and Ontology classes
to verify that JSON deserialization is working correctly.
"""

import json
from aim2_project.aim2_ontology.models import Term, Relationship, Ontology

def test_term_deserialization():
    """Test Term JSON deserialization."""
    print("Testing Term deserialization...")
    
    # Create a test term
    original_term = Term(
        id="CHEBI:15422",
        name="ATP",
        definition="Adenosine 5'-triphosphate",
        synonyms=["adenosine triphosphate", "adenosine 5'-triphosphate"],
        namespace="chemical",
        xrefs=["CAS:56-65-5", "KEGG:C00002"],
        relationships={"is_a": ["CHEBI:33019"]}
    )
    
    # Serialize to JSON
    json_str = original_term.to_json()
    print(f"Serialized JSON: {json_str}")
    
    # Deserialize from JSON
    deserialized_term = Term.from_json(json_str)
    print(f"Deserialized term: {deserialized_term}")
    
    # Verify they are equal
    assert original_term == deserialized_term, "Term serialization/deserialization failed"
    print("✓ Term deserialization test passed!\n")
    
    return True

def test_relationship_deserialization():
    """Test Relationship JSON deserialization."""
    print("Testing Relationship deserialization...")
    
    # Create a test relationship
    original_rel = Relationship(
        id="REL:001",
        subject="CHEBI:15422",
        predicate="is_a",
        object="CHEBI:33019",
        confidence=0.95,
        source="ChEBI",
        evidence="experimental_evidence",
        context="chemical classification"
    )
    
    # Serialize to JSON
    json_str = original_rel.to_json()
    print(f"Serialized JSON: {json_str}")
    
    # Deserialize from JSON
    deserialized_rel = Relationship.from_json(json_str)
    print(f"Deserialized relationship: {deserialized_rel}")
    
    # Verify they are equal
    assert original_rel == deserialized_rel, "Relationship serialization/deserialization failed"
    print("✓ Relationship deserialization test passed!\n")
    
    return True

def test_ontology_deserialization():
    """Test Ontology JSON deserialization."""
    print("Testing Ontology deserialization...")
    
    # Create test terms and relationships
    term1 = Term(id="CHEBI:15422", name="ATP")
    term2 = Term(id="CHEBI:33019", name="nucleotide")
    rel1 = Relationship(id="REL:001", subject="CHEBI:15422", predicate="is_a", object="CHEBI:33019")
    
    # Create a test ontology
    original_ontology = Ontology(
        id="ONTOLOGY:001",
        name="Test Chemical Ontology",
        description="A test ontology for chemicals",
        version="1.0",
        terms={"CHEBI:15422": term1, "CHEBI:33019": term2},
        relationships={"REL:001": rel1},
        metadata={"created_by": "test_script"}
    )
    
    # Serialize to JSON
    json_str = original_ontology.to_json()
    print(f"Serialized JSON (truncated): {json_str[:200]}...")
    
    # Deserialize from JSON
    deserialized_ontology = Ontology.from_json(json_str)
    print(f"Deserialized ontology: {deserialized_ontology.name} ({deserialized_ontology.id})")
    
    # Verify they are equal
    assert original_ontology == deserialized_ontology, "Ontology serialization/deserialization failed"
    print("✓ Ontology deserialization test passed!\n")
    
    return True

def main():
    """Run all deserialization tests."""
    print("=" * 60)
    print("AIM2-011-11: JSON Deserialization Test Demo")
    print("=" * 60)
    
    try:
        # Test all three classes
        test_term_deserialization()
        test_relationship_deserialization()
        test_ontology_deserialization()
        
        print("=" * 60) 
        print("✓ ALL DESERIALIZATION TESTS PASSED!")
        print("JSON deserialization functionality has been successfully implemented and tested.")
        print("=" * 60)
        
        return True
        
    except Exception as e:
        print(f"❌ Test failed with error: {e}")
        print("=" * 60)
        return False

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)