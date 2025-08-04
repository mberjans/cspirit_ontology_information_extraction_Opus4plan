#!/usr/bin/env python3
"""
Demonstration script for OntologyManager loading functionality.

This script demonstrates the comprehensive ontology loading capabilities
of the OntologyManager class, including format auto-detection, caching,
multi-source loading, and statistics generation.

Usage:
    python demo_ontology_manager.py

The script will create sample ontology files in different formats and
demonstrate various loading scenarios.
"""

import json
import logging
import tempfile
from pathlib import Path

# Configure logging for demonstration
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)

try:
    from aim2_project.aim2_ontology.models import Ontology, Term
    from aim2_project.aim2_ontology.ontology_manager import OntologyManager
    from aim2_project.aim2_utils.config_manager import ConfigManager
except ImportError as e:
    print(f"Import error: {e}")
    print("This demo requires the AIM2 ontology modules to be available.")
    print(
        "Make sure you're running from the correct directory and the modules are properly installed."
    )
    exit(1)


def create_sample_files(temp_dir: Path):
    """Create sample ontology files in different formats for demonstration."""

    # Sample OWL/RDF content
    owl_content = """<?xml version="1.0"?>
<rdf:RDF xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#"
         xmlns:owl="http://www.w3.org/2002/07/owl#"
         xmlns:rdfs="http://www.w3.org/2000/01/rdf-schema#">

    <owl:Ontology rdf:about="http://example.org/demo-ontology">
        <rdfs:label>Demo Chemical Ontology</rdfs:label>
        <rdfs:comment>A demonstration ontology for OntologyManager</rdfs:comment>
    </owl:Ontology>

    <owl:Class rdf:about="http://purl.obolibrary.org/obo/CHEBI_15422">
        <rdfs:label>ATP</rdfs:label>
        <rdfs:comment>Adenosine 5'-triphosphate</rdfs:comment>
    </owl:Class>

    <owl:Class rdf:about="http://purl.obolibrary.org/obo/GO_0008152">
        <rdfs:label>metabolic process</rdfs:label>
        <rdfs:comment>The chemical reactions and pathways</rdfs:comment>
    </owl:Class>

</rdf:RDF>"""

    # Sample CSV content
    csv_content = '''id,name,definition,namespace,synonyms
CHEBI:15422,ATP,"Adenosine 5'-triphosphate",chemical,"adenosine triphosphate|ATP"
GO:0008152,metabolic process,"The chemical reactions and pathways",biological_process,"metabolism"
CHEBI:33917,monosaccharide,"A simple sugar",chemical,"simple sugar"
GO:0006096,glycolysis,"Glucose catabolism pathway",biological_process,"glucose breakdown"'''

    # Sample JSON-LD content
    jsonld_content = {
        "@context": {
            "@vocab": "http://example.org/",
            "rdfs": "http://www.w3.org/2000/01/rdf-schema#",
            "owl": "http://www.w3.org/2002/07/owl#",
        },
        "@id": "demo-jsonld-ontology",
        "@type": "owl:Ontology",
        "rdfs:label": "Demo JSON-LD Ontology",
        "rdfs:comment": "A demonstration ontology in JSON-LD format",
        "terms": [
            {
                "@id": "PROTEIN:001",
                "@type": "owl:Class",
                "rdfs:label": "hemoglobin",
                "rdfs:comment": "Oxygen-carrying protein in red blood cells",
                "namespace": "protein",
            },
            {
                "@id": "DISEASE:001",
                "@type": "owl:Class",
                "rdfs:label": "anemia",
                "rdfs:comment": "Condition with low red blood cell count",
                "namespace": "disease",
            },
        ],
    }

    # Create files
    files = {}

    # OWL file
    owl_file = temp_dir / "demo_ontology.owl"
    owl_file.write_text(owl_content)
    files["owl"] = owl_file

    # CSV file
    csv_file = temp_dir / "demo_ontology.csv"
    csv_file.write_text(csv_content)
    files["csv"] = csv_file

    # JSON-LD file
    jsonld_file = temp_dir / "demo_ontology.jsonld"
    jsonld_file.write_text(json.dumps(jsonld_content, indent=2))
    files["jsonld"] = jsonld_file

    # Invalid file for error demonstration
    invalid_file = temp_dir / "invalid_ontology.owl"
    invalid_file.write_text("This is not valid RDF/OWL content")
    files["invalid"] = invalid_file

    return files


def demonstrate_basic_loading(manager: OntologyManager, files: dict):
    """Demonstrate basic ontology loading with different formats."""
    print("\n" + "=" * 60)
    print("DEMONSTRATION: Basic Ontology Loading")
    print("=" * 60)

    for format_name, file_path in files.items():
        if format_name == "invalid":  # Skip invalid file for now
            continue

        print(f"\n--- Loading {format_name.upper()} file: {file_path.name} ---")

        result = manager.load_ontology(str(file_path))

        if result.success:
            print(f"✓ Successfully loaded ontology: {result.ontology.id}")
            print(f"  - Load time: {result.load_time:.3f} seconds")
            print(f"  - Format detected: {result.metadata.get('format', 'unknown')}")
            print(f"  - Terms count: {result.metadata.get('terms_count', 0)}")
            print(
                f"  - Relationships count: {result.metadata.get('relationships_count', 0)}"
            )
            if result.warnings:
                print(f"  - Warnings: {len(result.warnings)}")
        else:
            print(f"✗ Failed to load ontology")
            print(f"  - Errors: {result.errors}")


def demonstrate_caching(manager: OntologyManager, files: dict):
    """Demonstrate caching functionality."""
    print("\n" + "=" * 60)
    print("DEMONSTRATION: Caching Functionality")
    print("=" * 60)

    # Use the OWL file for caching demo
    owl_file = files["owl"]

    print(f"\n--- First load (should be cache miss) ---")
    result1 = manager.load_ontology(str(owl_file))
    if result1.success:
        print(f"✓ Loaded ontology in {result1.load_time:.3f} seconds")
        print(f"  - Cache hit: {result1.metadata.get('cache_hit', False)}")

    print(f"\n--- Second load (should be cache hit) ---")
    result2 = manager.load_ontology(str(owl_file))
    if result2.success:
        print(f"✓ Loaded ontology in {result2.load_time:.3f} seconds")
        print(f"  - Cache hit: {result2.metadata.get('cache_hit', False)}")
        print(f"  - Access count: {result2.metadata.get('access_count', 1)}")

    # Show cache statistics
    stats = manager.get_statistics()
    print(f"\n--- Cache Statistics ---")
    print(f"Cache hits: {stats['cache_hits']}")
    print(f"Cache misses: {stats['cache_misses']}")
    print(f"Cache size: {stats['cache_size']}")


def demonstrate_multi_source_loading(manager: OntologyManager, files: dict):
    """Demonstrate loading multiple ontology sources."""
    print("\n" + "=" * 60)
    print("DEMONSTRATION: Multi-Source Loading")
    print("=" * 60)

    # Get all valid files
    sources = [str(files[key]) for key in ["owl", "csv", "jsonld"]]

    print(f"\n--- Loading {len(sources)} ontology sources ---")
    for i, source in enumerate(sources, 1):
        print(f"{i}. {Path(source).name}")

    results = manager.load_ontologies(sources)

    print(f"\n--- Results ---")
    successful = 0
    failed = 0

    for i, result in enumerate(results, 1):
        source_name = Path(result.source_path).name
        if result.success:
            successful += 1
            print(f"✓ {source_name}: Loaded successfully")
            print(f"  - Format: {result.metadata.get('format', 'unknown')}")
            print(f"  - Load time: {result.load_time:.3f}s")
        else:
            failed += 1
            print(f"✗ {source_name}: Failed to load")
            print(f"  - Errors: {result.errors[:2]}...")  # Show first 2 errors

    print(f"\nSummary: {successful} successful, {failed} failed")


def demonstrate_error_handling(manager: OntologyManager, files: dict):
    """Demonstrate error handling with invalid files."""
    print("\n" + "=" * 60)
    print("DEMONSTRATION: Error Handling")
    print("=" * 60)

    # Try to load invalid file
    invalid_file = files["invalid"]
    print(f"\n--- Loading invalid file: {invalid_file.name} ---")

    result = manager.load_ontology(str(invalid_file))

    if result.success:
        print("✓ Unexpectedly succeeded (parser was very tolerant)")
    else:
        print("✗ Failed as expected")
        print(f"  - Number of errors: {len(result.errors)}")
        print(
            f"  - First error: {result.errors[0] if result.errors else 'No specific error'}"
        )

    # Try to load non-existent file
    print(f"\n--- Loading non-existent file ---")
    result = manager.load_ontology("/path/that/does/not/exist.owl")

    if result.success:
        print("✓ Unexpectedly succeeded")
    else:
        print("✗ Failed as expected")
        print(
            f"  - Error: {result.errors[0] if result.errors else 'No specific error'}"
        )


def demonstrate_statistics(manager: OntologyManager):
    """Demonstrate statistics generation."""
    print("\n" + "=" * 60)
    print("DEMONSTRATION: Statistics and Reporting")
    print("=" * 60)

    stats = manager.get_statistics()

    print(f"\n--- Load Statistics ---")
    print(f"Total loads attempted: {stats['total_loads']}")
    print(f"Successful loads: {stats['successful_loads']}")
    print(f"Failed loads: {stats['failed_loads']}")

    print(f"\n--- Cache Statistics ---")
    print(f"Cache enabled: {stats['cache_enabled']}")
    print(f"Cache hits: {stats['cache_hits']}")
    print(f"Cache misses: {stats['cache_misses']}")
    print(f"Current cache size: {stats['cache_size']}")
    print(f"Cache size limit: {stats['cache_limit']}")

    print(f"\n--- Ontology Statistics ---")
    print(f"Loaded ontologies: {stats['loaded_ontologies']}")
    print(f"Total terms: {stats['total_terms']}")
    print(f"Total relationships: {stats['total_relationships']}")

    print(f"\n--- Format Statistics ---")
    formats_loaded = stats["formats_loaded"]
    if formats_loaded:
        for format_name, count in formats_loaded.items():
            print(f"{format_name}: {count} files")
    else:
        print("No formats loaded yet")


def demonstrate_ontology_management(manager: OntologyManager):
    """Demonstrate ontology management operations."""
    print("\n" + "=" * 60)
    print("DEMONSTRATION: Ontology Management")
    print("=" * 60)

    print(f"\n--- Listing loaded ontologies ---")
    ontology_ids = manager.list_ontologies()
    print(f"Number of ontologies: {len(ontology_ids)}")

    for i, ontology_id in enumerate(ontology_ids, 1):
        ontology = manager.get_ontology(ontology_id)
        print(f"{i}. {ontology_id}")
        if ontology:
            print(f"   Name: {ontology.name}")
            print(f"   Version: {ontology.version}")
            print(f"   Terms: {len(ontology.terms)}")
            print(f"   Relationships: {len(ontology.relationships)}")

    # Demonstrate adding a programmatically created ontology
    print(f"\n--- Adding programmatically created ontology ---")

    # Create a simple ontology
    terms = {
        "DEMO:001": Term(
            id="DEMO:001",
            name="demo term",
            definition="A term created for demonstration",
            namespace="demo",
        )
    }

    demo_ontology = Ontology(
        id="DEMO:ONTOLOGY",
        name="Programmatic Demo Ontology",
        version="1.0",
        description="An ontology created programmatically for demonstration",
        terms=terms,
        relationships={},
        namespaces=["demo"],
    )

    success = manager.add_ontology(demo_ontology)
    if success:
        print(f"✓ Successfully added ontology: {demo_ontology.id}")
    else:
        print(f"✗ Failed to add ontology")

    # Show updated list
    print(f"\n--- Updated ontology list ---")
    updated_ids = manager.list_ontologies()
    print(f"Number of ontologies: {len(updated_ids)}")
    for ontology_id in updated_ids:
        print(f"  - {ontology_id}")


def demonstrate_configuration_based_loading():
    """Demonstrate configuration-based ontology loading."""
    print("\n" + "=" * 60)
    print("DEMONSTRATION: Configuration-Based Loading")
    print("=" * 60)

    try:
        # Initialize config manager and load default configuration
        print("\n--- Loading default configuration ---")
        config_manager = ConfigManager()
        config_manager.load_default_config()
        print("✓ Default configuration loaded")

        # Initialize ontology manager
        manager = OntologyManager(enable_caching=True, cache_size_limit=10)

        # Validate configuration first
        print("\n--- Validating ontology sources configuration ---")
        try:
            validation_report = manager.validate_ontology_sources_config(config_manager)

            print(f"Configuration valid: {validation_report['valid']}")
            print(
                f"Total sources configured: {validation_report['summary']['total_sources']}"
            )
            print(f"Enabled sources: {validation_report['summary']['enabled_sources']}")
            print(f"Valid URLs: {validation_report['summary']['valid_urls']}")
            print(
                f"Accessible local paths: {validation_report['summary']['accessible_local_paths']}"
            )

            if validation_report["warnings"]:
                print(
                    f"\nConfiguration warnings ({len(validation_report['warnings'])}):"
                )
                for warning in validation_report["warnings"][:3]:  # Show first 3
                    print(f"  - {warning}")
                if len(validation_report["warnings"]) > 3:
                    print(f"  ... and {len(validation_report['warnings']) - 3} more")

        except Exception as e:
            print(f"✗ Configuration validation failed: {e}")
            return

        # Attempt to load from configuration
        print(f"\n--- Loading ontologies from configuration ---")
        print(
            "Note: This will likely fail due to missing local files and network restrictions,"
        )
        print("but demonstrates the configuration-based loading functionality.")

        try:
            results = manager.load_from_config(config_manager)

            print(f"\nAttempted to load {len(results)} configured sources:")

            successful = 0
            for result in results:
                source_name = result.metadata.get("source_name", "unknown")
                if result.success:
                    successful += 1
                    print(f"✓ {source_name}: Loaded successfully")
                    print(f"  - Load time: {result.load_time:.3f}s")
                    print(f"  - Terms: {result.metadata.get('terms_count', 0)}")
                else:
                    print(f"✗ {source_name}: Failed to load")
                    if result.errors:
                        print(f"  - First error: {result.errors[0][:100]}...")

            print(
                f"\nSummary: {successful} successful, {len(results) - successful} failed"
            )

            if successful > 0:
                print(f"Total ontologies now loaded: {len(manager.list_ontologies())}")

        except Exception as e:
            print(f"✗ Configuration-based loading failed: {e}")

        # Demonstrate filtered loading
        print(f"\n--- Demonstrating filtered loading ---")
        print("Loading only specific sources (if they exist)...")

        try:
            # Try to load only chebi and gene_ontology
            filtered_results = manager.load_from_config(
                config_manager, source_filter=["chebi", "gene_ontology"]
            )

            print(f"Filtered loading attempted {len(filtered_results)} sources:")
            for result in filtered_results:
                source_name = result.metadata.get("source_name", "unknown")
                status = "✓ Loaded" if result.success else "✗ Failed"
                print(f"  {status}: {source_name}")

        except Exception as e:
            print(f"✗ Filtered loading failed: {e}")

    except Exception as e:
        print(f"✗ Configuration-based loading demonstration failed: {e}")


def demonstrate_configuration_validation():
    """Demonstrate configuration validation capabilities."""
    print("\n" + "=" * 60)
    print("DEMONSTRATION: Configuration Validation")
    print("=" * 60)

    try:
        # Load default configuration
        config_manager = ConfigManager()
        config_manager.load_default_config()

        manager = OntologyManager()

        print("\n--- Validating default ontology sources configuration ---")

        validation_report = manager.validate_ontology_sources_config(config_manager)

        print(f"\nValidation Summary:")
        print(f"  Overall valid: {validation_report['valid']}")
        print(f"  Total sources: {validation_report['summary']['total_sources']}")
        print(f"  Valid sources: {validation_report['summary']['valid_sources']}")
        print(f"  Enabled sources: {validation_report['summary']['enabled_sources']}")
        print(
            f"  Accessible local paths: {validation_report['summary']['accessible_local_paths']}"
        )
        print(f"  Valid URLs: {validation_report['summary']['valid_urls']}")

        print(f"\n--- Individual Source Validation ---")
        for source_name, status in validation_report["source_status"].items():
            validity = "✓ Valid" if status["valid"] else "✗ Invalid"
            local = "✓" if status["local_path_accessible"] else "✗"
            url = "✓" if status["url_valid"] else "✗"
            print(f"  {source_name}: {validity} (local: {local}, url: {url})")

        if validation_report["errors"]:
            print(
                f"\n--- Configuration Errors ({len(validation_report['errors'])}) ---"
            )
            for error in validation_report["errors"][:5]:  # Show first 5
                print(f"  - {error}")
            if len(validation_report["errors"]) > 5:
                print(f"  ... and {len(validation_report['errors']) - 5} more errors")

        if validation_report["warnings"]:
            print(
                f"\n--- Configuration Warnings ({len(validation_report['warnings'])}) ---"
            )
            for warning in validation_report["warnings"][:5]:  # Show first 5
                print(f"  - {warning}")
            if len(validation_report["warnings"]) > 5:
                print(
                    f"  ... and {len(validation_report['warnings']) - 5} more warnings"
                )

    except Exception as e:
        print(f"✗ Configuration validation demonstration failed: {e}")


def main():
    """Main demonstration function."""
    print("OntologyManager Comprehensive Demonstration")
    print("=" * 80)
    print("This demonstration shows the capabilities of the OntologyManager class")
    print("including format auto-detection, caching, multi-source loading, and more.")

    # Create temporary directory for demo files
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        print(f"\nUsing temporary directory: {temp_path}")

        # Create sample files
        print("\nCreating sample ontology files...")
        files = create_sample_files(temp_path)
        print(f"Created {len(files)} sample files:")
        for format_name, file_path in files.items():
            print(f"  - {format_name}: {file_path.name}")

        # Initialize OntologyManager
        print("\nInitializing OntologyManager...")
        manager = OntologyManager(enable_caching=True, cache_size_limit=10)
        print("✓ OntologyManager initialized")

        # Run demonstrations
        try:
            demonstrate_basic_loading(manager, files)
            demonstrate_caching(manager, files)
            demonstrate_multi_source_loading(manager, files)
            demonstrate_configuration_validation()
            demonstrate_configuration_based_loading()
            demonstrate_error_handling(manager, files)
            demonstrate_statistics(manager)
            demonstrate_ontology_management(manager)

        except Exception as e:
            print(f"\n✗ Demonstration failed with error: {e}")
            print(
                "This may be due to missing parser implementations or other dependencies."
            )
            return

        # Final statistics
        print("\n" + "=" * 60)
        print("FINAL SUMMARY")
        print("=" * 60)

        final_stats = manager.get_statistics()
        print(f"Total operations: {final_stats['total_loads']}")
        print(
            f"Success rate: {final_stats['successful_loads']}/{final_stats['total_loads']}"
        )
        print(
            f"Cache efficiency: {final_stats['cache_hits']}/{final_stats['cache_hits'] + final_stats['cache_misses']} hits"
        )
        print(f"Ontologies managed: {final_stats['loaded_ontologies']}")
        print(f"Total terms: {final_stats['total_terms']}")
        print(f"Total relationships: {final_stats['total_relationships']}")

        print("\n✓ Demonstration completed successfully!")
        print("\nConfiguration-based loading features demonstrated:")
        print("- Automatic loading from configuration files")
        print("- Configuration validation and error reporting")
        print("- Filtered loading by source name")
        print("- Proper error handling for missing sources")
        print("\nNote: Some operations may fail if the corresponding parsers")
        print("are not fully implemented or if there are import dependencies.")
        print("Network-based loading may fail due to connectivity or missing files.")


if __name__ == "__main__":
    main()
