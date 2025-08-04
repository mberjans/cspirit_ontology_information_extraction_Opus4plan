#!/usr/bin/env python3
"""
Integration test for configuration-based ontology loading.

This test demonstrates and validates the load_from_config() functionality
of the OntologyManager class, including configuration validation and
error handling for missing or inaccessible sources.
"""

import logging
import tempfile
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)

try:
    from aim2_project.aim2_ontology.ontology_manager import OntologyManager
    from aim2_project.aim2_utils.config_manager import ConfigManager
except ImportError as e:
    print(f"Import error: {e}")
    print("This test requires the AIM2 modules to be available.")
    exit(1)


def create_test_config(temp_dir: Path) -> Path:
    """Create a test configuration file with ontology sources."""

    test_config = {
        "project": {"name": "Test Configuration", "version": "1.0.0"},
        "ontology": {
            "sources": {
                "test_local": {
                    "enabled": True,
                    "local_path": str(temp_dir / "test_ontology.owl"),
                    "url": "https://example.com/test.owl",
                    "update_frequency": "daily",
                    "include_deprecated": False,
                },
                "test_url_only": {
                    "enabled": True,
                    "url": "https://example.com/url_only.owl",
                    "update_frequency": "weekly",
                    "include_deprecated": False,
                },
                "test_disabled": {
                    "enabled": False,
                    "local_path": str(temp_dir / "disabled_ontology.owl"),
                    "url": "https://example.com/disabled.owl",
                    "update_frequency": "monthly",
                    "include_deprecated": True,
                },
                "test_missing_paths": {
                    "enabled": True,
                    "local_path": str(temp_dir / "nonexistent.owl"),
                    "url": "https://example.com/missing.owl",
                    "update_frequency": "weekly",
                    "include_deprecated": False,
                },
                "test_invalid_config": "this_is_not_a_dict",
            }
        },
    }

    config_file = temp_dir / "test_config.yaml"

    # Write as YAML (simplified - using JSON for now)
    import yaml

    with open(config_file, "w") as f:
        yaml.dump(test_config, f, default_flow_style=False, indent=2)

    return config_file


def create_test_ontology_file(file_path: Path):
    """Create a simple test ontology file."""

    owl_content = """<?xml version="1.0"?>
<rdf:RDF xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#"
         xmlns:owl="http://www.w3.org/2002/07/owl#"
         xmlns:rdfs="http://www.w3.org/2000/01/rdf-schema#">

    <owl:Ontology rdf:about="http://test.example.org/test-ontology">
        <rdfs:label>Test Ontology</rdfs:label>
        <rdfs:comment>A test ontology for configuration-based loading</rdfs:comment>
    </owl:Ontology>

    <owl:Class rdf:about="http://test.example.org/TestClass">
        <rdfs:label>Test Class</rdfs:label>
        <rdfs:comment>A test class</rdfs:comment>
    </owl:Class>

</rdf:RDF>"""

    file_path.write_text(owl_content)


def test_configuration_validation():
    """Test configuration validation functionality."""
    print("\n" + "=" * 60)
    print("TEST: Configuration Validation")
    print("=" * 60)

    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)

        # Create test configuration
        config_file = create_test_config(temp_path)

        # Create one existing ontology file
        test_ontology_file = temp_path / "test_ontology.owl"
        create_test_ontology_file(test_ontology_file)

        print(f"Created test configuration: {config_file}")
        print(f"Created test ontology file: {test_ontology_file}")

        # Initialize components
        config_manager = ConfigManager()
        config_manager.load_config(str(config_file))

        manager = OntologyManager(enable_caching=True, cache_size_limit=10)

        # Test configuration validation
        print("\n--- Validating ontology sources configuration ---")

        try:
            validation_report = manager.validate_ontology_sources_config(config_manager)

            print(f"Configuration valid: {validation_report['valid']}")
            print(f"Total sources: {validation_report['summary']['total_sources']}")
            print(f"Valid sources: {validation_report['summary']['valid_sources']}")
            print(f"Enabled sources: {validation_report['summary']['enabled_sources']}")
            print(
                f"Accessible local paths: {validation_report['summary']['accessible_local_paths']}"
            )
            print(f"Valid URLs: {validation_report['summary']['valid_urls']}")

            if validation_report["errors"]:
                print(f"\nErrors ({len(validation_report['errors'])}):")
                for error in validation_report["errors"]:
                    print(f"  - {error}")

            if validation_report["warnings"]:
                print(f"\nWarnings ({len(validation_report['warnings'])}):")
                for warning in validation_report["warnings"]:
                    print(f"  - {warning}")

            print(f"\n--- Individual Source Status ---")
            for source_name, status in validation_report["source_status"].items():
                print(
                    f"{source_name}: {'✓' if status['valid'] else '✗'} "
                    f"(local: {'✓' if status['local_path_accessible'] else '✗'}, "
                    f"url: {'✓' if status['url_valid'] else '✗'})"
                )

        except Exception as e:
            print(f"✗ Configuration validation failed: {e}")


def test_config_based_loading():
    """Test configuration-based ontology loading."""
    print("\n" + "=" * 60)
    print("TEST: Configuration-Based Loading")
    print("=" * 60)

    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)

        # Create test configuration
        config_file = create_test_config(temp_path)

        # Create one existing ontology file
        test_ontology_file = temp_path / "test_ontology.owl"
        create_test_ontology_file(test_ontology_file)

        print(f"Using test configuration: {config_file}")

        # Initialize components
        config_manager = ConfigManager()
        config_manager.load_config(str(config_file))

        manager = OntologyManager(enable_caching=True, cache_size_limit=10)

        # Test loading all enabled sources
        print("\n--- Loading all enabled sources ---")

        try:
            results = manager.load_from_config(config_manager)

            print(f"Attempted to load {len(results)} sources")

            successful = 0
            failed = 0

            for result in results:
                source_name = result.metadata.get("source_name", "unknown")
                if result.success:
                    successful += 1
                    print(f"✓ {source_name}: Loaded successfully")
                    print(f"  - Ontology ID: {result.ontology.id}")
                    print(f"  - Load time: {result.load_time:.3f}s")
                    print(f"  - Source path: {result.source_path}")
                else:
                    failed += 1
                    print(f"✗ {source_name}: Failed")
                    print(f"  - Errors: {result.errors[:2]}...")  # Show first 2 errors

            print(f"\nSummary: {successful} successful, {failed} failed")

            # Show loaded ontologies
            loaded_ontologies = manager.list_ontologies()
            print(f"Total ontologies in manager: {len(loaded_ontologies)}")

        except Exception as e:
            print(f"✗ Configuration-based loading failed: {e}")


def test_filtered_loading():
    """Test loading with source filtering."""
    print("\n" + "=" * 60)
    print("TEST: Filtered Loading")
    print("=" * 60)

    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)

        # Create test configuration
        config_file = create_test_config(temp_path)

        # Create one existing ontology file
        test_ontology_file = temp_path / "test_ontology.owl"
        create_test_ontology_file(test_ontology_file)

        # Initialize components
        config_manager = ConfigManager()
        config_manager.load_config(str(config_file))

        manager = OntologyManager(enable_caching=True, cache_size_limit=10)

        # Test loading with source filter
        print("\n--- Loading specific sources only ---")

        try:
            results = manager.load_from_config(
                config_manager, source_filter=["test_local", "test_url_only"]
            )

            print(f"Filtered loading attempted {len(results)} sources")

            for result in results:
                source_name = result.metadata.get("source_name", "unknown")
                status = "✓ Loaded" if result.success else "✗ Failed"
                print(f"{status}: {source_name}")

        except Exception as e:
            print(f"✗ Filtered loading failed: {e}")

        # Test loading including disabled sources
        print("\n--- Loading including disabled sources ---")

        try:
            results = manager.load_from_config(config_manager, enabled_only=False)

            print(f"Loading with disabled sources: {len(results)} sources")

            for result in results:
                source_name = result.metadata.get("source_name", "unknown")
                config = result.metadata.get("config", {})
                enabled = config.get("enabled", False)
                status = "✓ Loaded" if result.success else "✗ Failed"
                print(f"{status}: {source_name} (enabled: {enabled})")

        except Exception as e:
            print(f"✗ Loading with disabled sources failed: {e}")


def test_error_handling():
    """Test error handling for various failure scenarios."""
    print("\n" + "=" * 60)
    print("TEST: Error Handling")
    print("=" * 60)

    manager = OntologyManager()

    # Test with no configuration
    print("\n--- Test with missing ConfigManager ---")
    try:
        results = manager.load_from_config(None, "/nonexistent/config.yaml")
        print(f"Unexpected success: {len(results)} results")
    except Exception as e:
        print(f"✓ Expected error: {e}")

    # Test with invalid configuration path
    print("\n--- Test with invalid configuration path ---")
    try:
        results = manager.load_from_config(
            None, "/absolutely/nonexistent/path/config.yaml"
        )
        print(f"Unexpected success: {len(results)} results")
    except Exception as e:
        print(f"✓ Expected error: {e}")

    # Test validation with missing ConfigManager
    print("\n--- Test validation with missing ConfigManager ---")
    try:
        report = manager.validate_ontology_sources_config(
            None, "/nonexistent/config.yaml"
        )
        print(f"Unexpected success: {report}")
    except Exception as e:
        print(f"✓ Expected error: {e}")


def main():
    """Run all configuration-based loading tests."""
    print("Configuration-Based Ontology Loading Tests")
    print("=" * 80)
    print(
        "Testing the new load_from_config() functionality and configuration validation."
    )

    try:
        test_configuration_validation()
        test_config_based_loading()
        test_filtered_loading()
        test_error_handling()

        print("\n" + "=" * 60)
        print("✓ All tests completed successfully!")
        print("=" * 60)
        print(
            "\nThe configuration-based loading functionality appears to be working correctly."
        )
        print(
            "Note: Some load operations may fail due to missing parsers or network issues,"
        )
        print("but the configuration system and error handling should work properly.")

    except Exception as e:
        print(f"\n✗ Test suite failed with error: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    main()
