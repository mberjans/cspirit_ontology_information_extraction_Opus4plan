#!/usr/bin/env python3
"""
Demonstration script for the comprehensive OWL parser implementation.

This script showcases the full functionality of the enhanced OWL parser
including format detection, parsing, validation, and conversion capabilities.
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "aim2_project"))

from aim2_project.aim2_ontology.parsers import OWLParser
import tempfile


def demo_owl_parser():
    """Demonstrate the OWL parser with various formats and features."""

    print("OWL Parser Implementation Demo")
    print("=" * 40)

    # Create parser instance
    parser = OWLParser()
    print(f"✓ Created OWL parser: {parser.parser_name}")
    print(f"  Supported formats: {parser.get_supported_formats()}")
    print()

    # Test sample OWL content
    owl_xml = """<?xml version="1.0"?>
<rdf:RDF xmlns="http://example.org/ontology#"
         xml:base="http://example.org/ontology"
         xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#"
         xmlns:owl="http://www.w3.org/2002/07/owl#"
         xmlns:rdfs="http://www.w3.org/2000/01/rdf-schema#">

    <owl:Ontology rdf:about="http://example.org/ontology">
        <rdfs:label>Demo Chemical Ontology</rdfs:label>
        <rdfs:comment>A demonstration ontology for chemical compounds</rdfs:comment>
    </owl:Ontology>

    <owl:Class rdf:about="http://example.org/ontology#Chemical">
        <rdfs:label>Chemical</rdfs:label>
        <rdfs:comment>A chemical compound or substance</rdfs:comment>
    </owl:Class>

    <owl:Class rdf:about="http://example.org/ontology#Glucose">
        <rdfs:label>Glucose</rdfs:label>
        <rdfs:comment>A simple sugar important in biology</rdfs:comment>
        <rdfs:subClassOf rdf:resource="http://example.org/ontology#Chemical"/>
    </owl:Class>

    <owl:Class rdf:about="http://example.org/ontology#Protein">
        <rdfs:label>Protein</rdfs:label>
        <rdfs:comment>A complex biomolecule</rdfs:comment>
        <rdfs:subClassOf rdf:resource="http://example.org/ontology#Chemical"/>
    </owl:Class>

</rdf:RDF>"""

    # Demo 1: Format Detection
    print("Demo 1: Format Detection")
    print("-" * 25)

    detected_format = parser.detect_format(owl_xml)
    print(f"  Detected format: {detected_format}")

    # Validate the format
    is_valid_format = parser.validate_format(owl_xml, detected_format)
    print(f"  Format validation: {'✓ Valid' if is_valid_format else '✗ Invalid'}")
    print()

    # Demo 2: Basic Parsing
    print("Demo 2: OWL Parsing")
    print("-" * 19)

    try:
        # Parse the OWL content
        parsed_result = parser.parse(owl_xml)
        print(f"✓ Successfully parsed OWL content")
        print(f"  Result type: {type(parsed_result).__name__}")
        print(f"  Format: {parsed_result.get('format', 'unknown')}")
        print(f"  Content size: {parsed_result.get('content_size', 0)} bytes")

        # Check available parsers
        has_rdf = parsed_result.get("rdf_graph") is not None
        has_owl = parsed_result.get("owl_ontology") is not None
        print(f"  RDF graph available: {has_rdf}")
        print(f"  OWL ontology available: {has_owl}")

    except Exception as e:
        print(f"✗ Parsing failed: {str(e)}")
        parsed_result = None
    print()

    # Demo 3: Validation
    print("Demo 3: OWL Validation")
    print("-" * 20)

    # Basic validation
    is_valid = parser.validate(owl_xml)
    print(f"Basic validation: {'✓ Valid' if is_valid else '✗ Invalid'}")

    # Comprehensive validation
    try:
        validation_result = parser.validate_owl(owl_xml)
        print(f"Comprehensive validation:")
        print(f"  Valid: {'✓' if validation_result.get('is_valid', False) else '✗'}")
        print(f"  Format: {validation_result.get('format', 'unknown')}")
        print(f"  Errors: {len(validation_result.get('errors', []))}")
        print(f"  Warnings: {len(validation_result.get('warnings', []))}")

        # Show any errors or warnings
        errors = validation_result.get("errors", [])
        if errors:
            print(f"  Error details: {errors[0]}")

        warnings = validation_result.get("warnings", [])
        if warnings:
            print(f"  Warning details: {warnings[0]}")

    except Exception as e:
        print(f"✗ Comprehensive validation failed: {str(e)}")
    print()

    # Demo 4: Model Conversion (if parsing succeeded)
    if parsed_result:
        print("Demo 4: Model Conversion")
        print("-" * 23)

        try:
            # Convert to internal ontology model
            ontology = parser.to_ontology(parsed_result)
            print(f"✓ Converted to Ontology model")
            print(f"  ID: {ontology.id}")
            print(f"  Name: {ontology.name}")
            print(f"  Metadata entries: {len(ontology.metadata)}")

            # Extract terms
            terms = parser.extract_terms(parsed_result)
            print(f"✓ Extracted {len(terms)} terms")
            if terms:
                for i, term in enumerate(terms[:3]):  # Show first 3
                    print(f"    Term {i+1}: {term.id} - '{term.name}'")

            # Extract relationships
            relationships = parser.extract_relationships(parsed_result)
            print(f"✓ Extracted {len(relationships)} relationships")
            if relationships:
                for i, rel in enumerate(relationships[:3]):  # Show first 3
                    print(
                        f"    Rel {i+1}: {rel.subject} --{rel.predicate}--> {rel.object}"
                    )

            # Extract metadata
            metadata = parser.extract_metadata(parsed_result)
            print(f"✓ Extracted metadata with {len(metadata)} entries")
            key_metadata = ["format", "parser_name", "ontology_iri"]
            for key in key_metadata:
                if key in metadata:
                    value = metadata[key]
                    if isinstance(value, str) and len(value) > 50:
                        value = value[:47] + "..."
                    print(f"    {key}: {value}")

        except Exception as e:
            print(f"✗ Model conversion failed: {str(e)}")
        print()

    # Demo 5: File Operations
    print("Demo 5: File Operations")
    print("-" * 20)

    try:
        # Create temporary file
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".owl", delete=False
        ) as tmp_file:
            tmp_file.write(owl_xml)
            tmp_file_path = tmp_file.name

        print(f"✓ Created temporary file: {tmp_file_path}")

        # Parse from file
        file_result = parser.parse_file(tmp_file_path)
        print(f"✓ Successfully parsed from file")
        print(f"  Format: {file_result.get('format', 'unknown')}")
        print(
            f"  Source file: {file_result.get('options_used', {}).get('source_file', 'unknown')}"
        )

        # Clean up
        os.unlink(tmp_file_path)
        print(f"✓ Cleaned up temporary file")

    except Exception as e:
        print(f"✗ File operations failed: {str(e)}")
    print()

    # Demo 6: Parser Configuration
    print("Demo 6: Parser Configuration")
    print("-" * 25)

    try:
        # Show current options
        current_options = parser.get_options()
        print(f"✓ Current configuration has {len(current_options)} options")

        # Show some key options
        key_options = [
            "validate_on_parse",
            "include_imports",
            "strict_validation",
            "error_recovery",
        ]
        for option in key_options:
            if option in current_options:
                print(f"    {option}: {current_options[option]}")

        # Set custom options
        custom_options = {
            "validate_on_parse": True,
            "strict_validation": False,
            "preserve_namespaces": True,
        }
        parser.set_options(custom_options)
        print(f"✓ Applied custom configuration")

        # Get validation errors (if any)
        errors = parser.get_validation_errors()
        print(f"✓ Validation errors: {len(errors)}")

    except Exception as e:
        print(f"✗ Configuration management failed: {str(e)}")
    print()

    # Demo 7: Parser Metadata and Statistics
    print("Demo 7: Parser Metadata")
    print("-" * 21)

    try:
        metadata = parser.get_metadata()
        print(f"✓ Parser metadata:")
        print(f"  Parser name: {metadata.get('parser_name')}")
        print(f"  Supported formats: {len(metadata.get('supported_formats', []))}")

        # Show statistics
        stats = metadata.get("statistics", {})
        print(f"  Usage statistics:")
        print(f"    Total parses: {stats.get('total_parses', 0)}")
        print(f"    Successful parses: {stats.get('successful_parses', 0)}")
        print(f"    Success rate: {stats.get('success_rate', 0):.1f}%")

        # Show cache information
        cache_info = metadata.get("cache_info", {})
        if cache_info:
            print(f"  Cache status:")
            print(f"    Enabled: {cache_info.get('enabled', False)}")
            print(
                f"    Size: {cache_info.get('size', 0)}/{cache_info.get('max_size', 0)}"
            )

    except Exception as e:
        print(f"✗ Metadata retrieval failed: {str(e)}")
    print()

    # Demo 8: Different Format Testing
    print("Demo 8: Multiple Format Support")
    print("-" * 30)

    # Turtle format
    turtle_content = """@prefix : <http://example.org/ontology#> .
@prefix owl: <http://www.w3.org/2002/07/owl#> .
@prefix rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#> .
@prefix rdfs: <http://www.w3.org/2000/01/rdf-schema#> .

<http://example.org/ontology> a owl:Ontology ;
    rdfs:label "Demo Turtle Ontology" .

:Chemical a owl:Class ;
    rdfs:label "Chemical" .

:Glucose a owl:Class ;
    rdfs:label "Glucose" ;
    rdfs:subClassOf :Chemical .
"""

    formats_to_test = [
        ("Turtle", turtle_content),
        ("OWL/XML", owl_xml[:200] + "..."),  # Truncated for display
    ]

    for format_name, content in formats_to_test:
        try:
            detected = parser.detect_format(content)
            print(f"  {format_name}: detected as '{detected}'")

            if format_name == "Turtle":  # Only parse the complete turtle content
                parser.parse(turtle_content)
                print(f"    ✓ Parsed successfully")
            else:
                print(f"    (content truncated for demo)")

        except Exception as e:
            print(f"    ✗ Failed: {str(e)}")
    print()

    print("OWL Parser Demo Complete!")
    print("=" * 40)
    print()
    print("Summary:")
    print("  ✓ Format detection and validation")
    print("  ✓ OWL parsing with multiple libraries")
    print("  ✓ Comprehensive validation")
    print("  ✓ Model conversion (Ontology, Terms, Relationships)")
    print("  ✓ File operations")
    print("  ✓ Configuration management")
    print("  ✓ Statistics and metadata")
    print("  ✓ Multiple format support")


if __name__ == "__main__":
    demo_owl_parser()
