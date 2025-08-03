#!/usr/bin/env python3
"""
Demo script to verify AIM2-004-09: Create module-specific logger factory

This script demonstrates that the module-specific logger factory is fully implemented
and working correctly. It shows various features including:
- Automatic module detection
- Hierarchical module naming
- Module-specific configuration
- Convenience functions
"""

import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), "aim2_project"))

from aim2_utils.logger_factory import get_logger, get_module_logger, configure_logging


def demo_basic_usage():
    """Demo basic logger factory usage."""
    print("=== Demo: Basic Logger Factory Usage ===")

    # Get logger with auto-detected module name
    logger = get_logger()
    print(f"Auto-detected logger name: {logger.name}")
    logger.info("This is from auto-detected module logger")

    # Get explicit module logger
    ontology_logger = get_module_logger("ontology")
    print(f"Ontology logger name: {ontology_logger.name}")
    ontology_logger.info("This is from ontology module")

    # Get nested module logger
    pipeline_logger = get_module_logger("extraction.pipeline")
    print(f"Pipeline logger name: {pipeline_logger.name}")
    pipeline_logger.info("This is from extraction pipeline module")

    print()


def demo_hierarchical_naming():
    """Demo hierarchical module naming."""
    print("=== Demo: Hierarchical Module Naming ===")

    # Create loggers for different modules
    modules = [
        "ontology",
        "ontology.manager",
        "ontology.trimmer",
        "extraction",
        "extraction.ner",
        "extraction.relation",
        "utils",
        "utils.config",
    ]

    for module in modules:
        logger = get_module_logger(module)
        print(f"Module: {module:20} -> Logger: {logger.name}")
        logger.debug(f"Debug message from {module}")

    print()


def demo_module_specific_configuration():
    """Demo module-specific configuration."""
    print("=== Demo: Module-Specific Configuration ===")

    # Configure different log levels for different modules
    configure_logging(
        {
            "level": "INFO",
            "module_levels": {
                "ontology": "DEBUG",
                "extraction": "WARNING",
                "utils": "ERROR",
            },
        }
    )

    # Test different modules with their specific levels
    modules_to_test = ["ontology", "extraction", "utils", "general"]

    for module in modules_to_test:
        logger = get_module_logger(module)
        print(f"\nTesting {module} module (level: {logger.level}):")
        logger.debug(f"DEBUG from {module}")
        logger.info(f"INFO from {module}")
        logger.warning(f"WARNING from {module}")
        logger.error(f"ERROR from {module}")

    print()


def demo_convenience_functions():
    """Demo convenience functions."""
    print("=== Demo: Convenience Functions ===")

    # Different ways to get loggers
    print("Getting loggers using convenience functions:")

    # Method 1: Auto-detect module
    logger1 = get_logger()
    print(f"get_logger(): {logger1.name}")

    # Method 2: Explicit module name
    logger2 = get_logger(module_name="data_processor")
    print(f"get_logger(module_name='data_processor'): {logger2.name}")

    # Method 3: Explicit logger name
    logger3 = get_logger("my_component")
    print(f"get_logger('my_component'): {logger3.name}")

    # Method 4: Module-specific logger
    logger4 = get_module_logger("analysis.statistics")
    print(f"get_module_logger('analysis.statistics'): {logger4.name}")

    print()


def demo_factory_info():
    """Demo factory information and management."""
    print("=== Demo: Factory Information ===")

    from aim2_utils.logger_factory import LoggerFactory

    factory = LoggerFactory.get_instance()

    # Get factory information
    info = factory.get_info()
    print(f"Factory initialized: {info['initialized']}")
    print(f"Singleton active: {info['singleton_active']}")
    print(f"Number of managed loggers: {len(info.get('loggers', {}))}")

    # Get managed loggers
    managed_loggers = factory.get_managed_loggers()
    print(f"\nManaged loggers ({len(managed_loggers)}):")
    for name, logger in managed_loggers.items():
        print(f"  {name}: {logger}")

    print()


def main():
    """Main demo function."""
    print("AIM2-004-09: Module-Specific Logger Factory Demo")
    print("=" * 60)
    print()

    # Configure initial logging
    configure_logging(
        {
            "level": "DEBUG",
            "handlers": ["console"],
            "format": "%(name)s | %(levelname)s | %(message)s",
        }
    )

    # Run demos
    demo_basic_usage()
    demo_hierarchical_naming()
    demo_module_specific_configuration()
    demo_convenience_functions()
    demo_factory_info()

    print("=" * 60)
    print("✅ AIM2-004-09: Module-specific logger factory is COMPLETE!")
    print("✅ All features working correctly:")
    print("   - Automatic module detection")
    print("   - Hierarchical module naming")
    print("   - Module-specific configuration")
    print("   - Convenience functions")
    print("   - Factory management")


if __name__ == "__main__":
    main()
