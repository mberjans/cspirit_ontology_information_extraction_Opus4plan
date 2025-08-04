#!/usr/bin/env python3
"""
Demonstration script showing the new export functionality in OntologyManager.

This script demonstrates the three new export methods:
1. export_ontology() - Export a single ontology
2. export_combined_ontology() - Export all loaded ontologies
3. export_statistics() - Export statistics as JSON
"""

import json

# Add the aim2_project to the path
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, "aim2_project")

from aim2_ontology.models import Ontology, Relationship, Term
from aim2_ontology.ontology_manager import OntologyManager


def create_sample_ontologies():
    """Create sample ontologies for demonstration."""

    # Create first ontology - Chemical Ontology
    chem_ont = Ontology(
        id="CHEM:001",
        name="Chemical Ontology",
        version="1.0",
        description="A sample chemical ontology",
    )

    # Add chemical terms
    glucose = Term(
        id="CHEM:001",
        name="glucose",
        definition="A simple sugar with molecular formula C6H12O6",
        synonyms=["dextrose", "grape sugar"],
        namespace="chemical",
    )

    sugar = Term(
        id="CHEM:002",
        name="sugar",
        definition="A class of sweet carbohydrates",
        synonyms=["saccharide"],
        namespace="chemical",
    )

    chem_ont.add_term(glucose)
    chem_ont.add_term(sugar)

    # Add relationship
    chem_rel = Relationship(
        id="REL:001",
        subject="CHEM:001",
        predicate="is_a",
        object="CHEM:002",
        confidence=1.0,
        evidence="structural_analysis",
    )
    chem_ont.add_relationship(chem_rel)

    # Create second ontology - Biological Process Ontology
    bio_ont = Ontology(
        id="BIO:001",
        name="Biological Process Ontology",
        version="1.0",
        description="A sample biological process ontology",
    )

    # Add biological process terms
    metabolism = Term(
        id="BIO:001",
        name="metabolism",
        definition="The chemical reactions that occur within a living organism",
        synonyms=["metabolic process"],
        namespace="biological_process",
    )

    glycolysis = Term(
        id="BIO:002",
        name="glycolysis",
        definition="The breakdown of glucose to pyruvate",
        synonyms=["glucose catabolism"],
        namespace="biological_process",
    )

    bio_ont.add_term(metabolism)
    bio_ont.add_term(glycolysis)

    # Add relationship
    bio_rel = Relationship(
        id="REL:002",
        subject="BIO:002",
        predicate="is_a",
        object="BIO:001",
        confidence=0.95,
        evidence="literature_review",
    )
    bio_ont.add_relationship(bio_rel)

    return chem_ont, bio_ont


def demonstrate_export_functionality():
    """Demonstrate the export functionality."""
    print("=== OntologyManager Export Functionality Demo ===\n")

    # Create manager and load sample ontologies
    manager = OntologyManager()
    chem_ont, bio_ont = create_sample_ontologies()

    manager.add_ontology(chem_ont)
    manager.add_ontology(bio_ont)

    print(f"Loaded ontologies:")
    for ont_id in manager.list_ontologies():
        ont = manager.get_ontology(ont_id)
        print(
            f"  - {ont_id}: {ont.name} ({len(ont.terms)} terms, {len(ont.relationships)} relationships)"
        )
    print()

    # Demo 1: Export single ontology in different formats
    print("1. Export Single Ontology (Chemical Ontology):")
    print("=" * 50)

    # JSON export
    print("JSON Export (first 200 chars):")
    json_data = manager.export_ontology("CHEM:001", format="json")
    print(json_data[:200] + "...")
    print()

    # CSV export
    print("CSV Export:")
    csv_data = manager.export_ontology("CHEM:001", format="csv")
    print(csv_data)

    # OWL export (first few lines)
    print("OWL Export (first 10 lines):")
    owl_data = manager.export_ontology("CHEM:001", format="owl")
    print("\n".join(owl_data.split("\n")[:10]))
    print("...\n")

    # Demo 2: Export combined ontologies
    print("2. Export Combined Ontologies:")
    print("=" * 50)

    combined_json = manager.export_combined_ontology(format="json")
    combined_data = json.loads(combined_json)

    print(f"Combined export metadata:")
    metadata = combined_data["export_metadata"]
    print(f"  - Ontology count: {metadata['ontology_count']}")
    print(f"  - Export format: {metadata['export_format']}")

    print(f"\nIncluded ontologies:")
    for ont_id in combined_data["ontologies"]:
        ont_data = combined_data["ontologies"][ont_id]
        print(f"  - {ont_id}: {ont_data['name']} (v{ont_data['version']})")
    print()

    # Demo 3: Export statistics
    print("3. Export Statistics:")
    print("=" * 50)

    stats_json = manager.export_statistics()
    stats = json.loads(stats_json)

    print(f"Loading Statistics:")
    print(f"  - Total loads: {stats['total_loads']}")
    print(f"  - Successful loads: {stats['successful_loads']}")
    print(f"  - Failed loads: {stats['failed_loads']}")

    print(f"\nOntology Statistics:")
    print(f"  - Loaded ontologies: {stats['loaded_ontologies']}")
    print(f"  - Total terms: {stats['total_terms']}")
    print(f"  - Total relationships: {stats['total_relationships']}")

    print(f"\nDetailed Ontology Information:")
    for ont_id, ont_info in stats["ontologies"].items():
        print(f"  - {ont_id}:")
        print(f"    * Name: {ont_info['name']}")
        print(f"    * Terms: {ont_info['terms_count']}")
        print(f"    * Relationships: {ont_info['relationships_count']}")
        print(f"    * Consistent: {ont_info['is_consistent']}")
    print()

    # Demo 4: Export to files
    print("4. Export to Files:")
    print("=" * 50)

    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        print(f"Exporting to temporary directory: {temp_path}")

        # Export single ontology to files
        json_file = temp_path / "chemical_ontology.json"
        csv_file = temp_path / "chemical_ontology.csv"
        owl_file = temp_path / "chemical_ontology.owl"

        manager.export_ontology("CHEM:001", format="json", output_path=str(json_file))
        manager.export_ontology("CHEM:001", format="csv", output_path=str(csv_file))
        manager.export_ontology("CHEM:001", format="owl", output_path=str(owl_file))

        # Export combined ontologies
        combined_file = temp_path / "combined_ontologies.json"
        manager.export_combined_ontology(format="json", output_path=str(combined_file))

        # Export statistics
        stats_file = temp_path / "statistics.json"
        manager.export_statistics(output_path=str(stats_file))

        print("Files created:")
        for file_path in temp_path.iterdir():
            size = file_path.stat().st_size
            print(f"  - {file_path.name}: {size} bytes")

        # Show content of statistics file
        print(f"\nStatistics file content (formatted):")
        with open(stats_file, "r") as f:
            stats_content = json.load(f)
        print(json.dumps(stats_content, indent=2)[:500] + "...")

    print("\n=== Demo completed successfully! ===")


if __name__ == "__main__":
    demonstrate_export_functionality()
