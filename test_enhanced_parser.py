#!/usr/bin/env python3
"""
Test script to verify the enhanced AbstractParser implementation.

This script tests the new AbstractParser base class to ensure it works
correctly with the existing project structure and provides all the
enhanced functionality.
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "aim2_project"))

from aim2_project.aim2_ontology.parsers import AbstractParser, ParseResult
from aim2_project.exceptions import OntologyException
from typing import List, Any


class TestParser(AbstractParser):
    """Simple test parser implementation for verification."""

    def parse(self, content: str, **kwargs) -> Any:
        """Simple parsing that just returns content length and first 100 chars."""
        if "error" in content.lower():
            raise OntologyException("Test error condition detected")

        return {
            "content_length": len(content),
            "preview": content[:100],
            "kwargs": kwargs,
        }

    def validate(self, content: str, **kwargs) -> bool:
        """Simple validation - content must not be empty and under 1MB."""
        if not content.strip():
            return False
        if len(content) > 1024 * 1024:  # 1MB limit
            return False
        return True

    def get_supported_formats(self) -> List[str]:
        """Return test formats."""
        return ["test", "txt"]


def test_basic_functionality():
    """Test basic AbstractParser functionality."""
    print("Testing basic AbstractParser functionality...")

    # Create test parser instance
    parser = TestParser()

    # Test basic attributes
    assert parser.parser_name == "test"
    assert parser.get_supported_formats() == ["test", "txt"]
    print("✓ Basic attributes working")

    # Test parsing
    test_content = "This is test content for parsing."
    result = parser.parse_safe(test_content)

    assert isinstance(result, ParseResult)
    assert result.success is True
    assert result.data is not None
    assert result.data["content_length"] == len(test_content)
    print("✓ Basic parsing working")

    # Test validation
    assert parser.validate(test_content) is True
    assert parser.validate("") is False
    print("✓ Validation working")

    # Test options management
    parser.get_options()
    parser.set_options({"test_option": "test_value"})
    updated_options = parser.get_options()
    assert updated_options["test_option"] == "test_value"
    print("✓ Options management working")

    # Test metadata
    metadata = parser.get_metadata()
    assert "parser_name" in metadata
    assert "statistics" in metadata
    assert "cache_info" in metadata
    print("✓ Metadata generation working")


def test_error_handling():
    """Test error handling and exception integration."""
    print("\nTesting error handling...")

    parser = TestParser()

    # Test parsing error
    result = parser.parse_safe("This content contains ERROR")
    assert result.success is False
    assert len(result.errors) > 0
    print("✓ Error handling working")

    # Test validation errors
    result = parser.parse_safe("")  # Empty content should fail validation
    assert result.success is False or len(result.warnings) > 0
    print("✓ Validation error handling working")


def test_statistics_tracking():
    """Test statistics tracking functionality."""
    print("\nTesting statistics tracking...")

    parser = TestParser()

    # Perform several operations
    parser.parse_safe("Test content 1")
    parser.parse_safe("Test content 2")
    parser.parse_safe("Content with ERROR")  # This should fail

    stats = parser.get_metadata()["statistics"]
    assert stats["total_parses"] == 3
    assert stats["successful_parses"] == 2
    assert stats["failed_parses"] == 1
    print("✓ Statistics tracking working")


def test_caching():
    """Test caching functionality."""
    print("\nTesting caching functionality...")

    parser = TestParser()

    # Parse same content twice
    content = "Repeated content for caching test"
    parser.parse_safe(content)
    result2 = parser.parse_safe(content)

    # Second result should be from cache
    assert result2.metadata.get("from_cache") is True
    print("✓ Caching working")

    # Test cache info
    cache_info = parser.get_cache_info()
    assert cache_info["enabled"] is True
    assert cache_info["size"] > 0
    print("✓ Cache info working")


def test_hooks_and_validation_rules():
    """Test hooks and custom validation rules."""
    print("\nTesting hooks and validation rules...")

    parser = TestParser()

    # Add custom validation rule
    def custom_rule(content: str) -> List[str]:
        errors = []
        if "forbidden" in content.lower():
            errors.append("Forbidden word detected")
        return errors

    parser.add_validation_rule("forbidden_word", custom_rule)

    # Test custom validation
    result = parser.parse_safe("This contains FORBIDDEN word")
    # Should have warning or error about forbidden word
    assert len(result.errors) > 0 or len(result.warnings) > 0
    print("✓ Custom validation rules working")

    # Add hook
    hook_called = []

    def test_hook(*args, **kwargs):
        hook_called.append(True)

    parser.add_hook("pre_parse", test_hook)
    parser.parse_safe("Test content")

    assert len(hook_called) > 0
    print("✓ Hooks working")


def test_performance_monitoring():
    """Test performance monitoring features."""
    print("\nTesting performance monitoring...")

    parser = TestParser()

    # Parse content and check timing
    result = parser.parse_safe("Performance test content")
    assert result.parse_time > 0
    print("✓ Performance timing working")

    # Get performance hints
    hints = parser.get_performance_hints()
    assert isinstance(hints, list)
    print("✓ Performance hints working")


def main():
    """Run all tests."""
    print("Starting AbstractParser Enhancement Tests")
    print("=" * 50)

    try:
        test_basic_functionality()
        test_error_handling()
        test_statistics_tracking()
        test_caching()
        test_hooks_and_validation_rules()
        test_performance_monitoring()

        print("\n" + "=" * 50)
        print("🎉 All tests passed successfully!")
        print("The enhanced AbstractParser is working correctly.")

    except Exception as e:
        print(f"\n❌ Test failed: {str(e)}")
        import traceback

        traceback.print_exc()
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
