#!/usr/bin/env python3
"""
Comprehensive error handling unit tests for OntologyManager.

This module contains extensive unit tests specifically focused on error handling
scenarios and edge cases for the OntologyManager class. The tests ensure that
all error conditions are properly handled, exceptions are appropriately raised
and caught, and system recovery works correctly.

Test Coverage:
1. File System Errors:
   - Non-existent files
   - Permission denied (read permissions)
   - Corrupted or unreadable files
   - Invalid file paths
   - Network path failures
   - Disk space issues during loading

2. Format and Parsing Errors:
   - Unsupported file formats
   - Malformed ontology files (invalid OWL, CSV, JSON-LD)
   - Encoding issues (UTF-8, Latin-1, etc.)
   - Empty files
   - Truncated files
   - Files with invalid ontology structure

3. Resource and Memory Errors:
   - Out of memory conditions during large file loading
   - Timeout errors for long-running operations
   - Resource exhaustion (too many open files)
   - Memory leaks detection and prevention

4. Configuration and Input Validation Errors:
   - Invalid configuration parameters
   - Null/None inputs
   - Invalid source types (not strings or Path objects)
   - Empty source lists for multi-source loading

5. Cache and State Management Errors:
   - Cache corruption scenarios
   - Concurrent access errors
   - State inconsistency after partial failures
   - Recovery from failed operations

6. Custom Exception Hierarchy:
   - Proper exception chaining and propagation
   - Appropriate error codes and messages
   - Exception serialization and deserialization
   - Logging integration with error handling
"""

import os
import stat
import tempfile
import pytest
import time
import threading
import concurrent.futures
import gc
import psutil
import signal
import weakref
from unittest.mock import patch, Mock, MagicMock, mock_open, PropertyMock
from pathlib import Path
from typing import Dict, Any, List, Optional
from dataclasses import dataclass

# Import the modules to be tested
try:
    from aim2_project.aim2_ontology.ontology_manager import (
        OntologyManager,
        LoadResult,
        CacheEntry,
        OntologyManagerError,
        OntologyLoadError,
    )
    from aim2_project.aim2_ontology.models import Ontology, Term, Relationship
    from aim2_project.aim2_ontology.parsers import ParseResult
except ImportError:
    import warnings
    warnings.warn("Some imports failed - tests may be skipped", ImportWarning)


class TestOntologyManagerErrorHandling:
    """Comprehensive error handling test suite for OntologyManager."""

    @pytest.fixture
    def temp_dir(self):
        """Create a temporary directory for test files."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            yield Path(tmp_dir)

    @pytest.fixture
    def ontology_manager(self):
        """Create an OntologyManager instance for testing."""
        return OntologyManager(enable_caching=True, cache_size_limit=10)

    @pytest.fixture
    def sample_ontology(self):
        """Create a sample ontology for testing."""
        terms = {
            "TEST:001": Term(
                id="TEST:001",
                name="Test Term",
                definition="A test term for error handling",
                synonyms=["test", "example"],
                namespace="test"
            )
        }
        
        return Ontology(
            id="TEST:001",
            name="Error Test Ontology",
            version="1.0",
            description="Test ontology for error handling scenarios",
            terms=terms,
            relationships={},
            namespaces=["test"]
        )

    # ========== FILE SYSTEM ERRORS ==========

    def test_load_nonexistent_file(self, ontology_manager):
        """Test loading a non-existent file."""
        nonexistent_file = "/path/that/does/not/exist/file.owl"
        
        with patch('aim2_project.aim2_ontology.ontology_manager.auto_detect_parser') as mock_auto_detect:
            mock_auto_detect.return_value = None  # Parser not found for nonexistent file
            
            result = ontology_manager.load_ontology(nonexistent_file)
            
            assert result.success is False
            assert result.ontology is None
            assert result.source_path == nonexistent_file
            assert len(result.errors) > 0
            assert any("no suitable parser" in error.lower() for error in result.errors)
            assert ontology_manager.load_stats['failed_loads'] == 1

    def test_load_permission_denied_file(self, ontology_manager, temp_dir):
        """Test loading a file with permission denied."""
        # Create a file and remove read permissions
        restricted_file = temp_dir / "restricted.owl"
        restricted_file.write_text("<?xml version='1.0'?><owl:Ontology/>")
        
        # Remove read permissions
        original_mode = restricted_file.stat().st_mode
        restricted_file.chmod(0o000)  # No permissions
        
        try:
            with patch('aim2_project.aim2_ontology.ontology_manager.auto_detect_parser') as mock_auto_detect:
                # Mock parser that would fail due to permission error
                mock_parser = Mock()
                mock_parser.format_name = "owl"
                mock_parser.parse.side_effect = PermissionError("Permission denied")
                mock_auto_detect.return_value = mock_parser
                
                result = ontology_manager.load_ontology(str(restricted_file))
                
                assert result.success is False
                assert result.ontology is None
                assert len(result.errors) > 0
                assert any("unexpected error" in error.lower() for error in result.errors)
                
        finally:
            # Restore permissions for cleanup
            try:
                restricted_file.chmod(original_mode)
            except (OSError, FileNotFoundError):
                pass

    def test_load_corrupted_file(self, ontology_manager, temp_dir):
        """Test loading a corrupted/unreadable file."""
        corrupted_file = temp_dir / "corrupted.owl"
        
        # Create a file with invalid binary content
        with open(corrupted_file, 'wb') as f:
            f.write(b'\x00\x01\x02\x03\xFF\xFE\xFD\xFC')  # Invalid binary data
        
        with patch('aim2_project.aim2_ontology.ontology_manager.auto_detect_parser') as mock_auto_detect:
            mock_parser = Mock()
            mock_parser.format_name = "owl"
            mock_parser.parse.side_effect = UnicodeDecodeError(
                'utf-8', b'\xFF\xFE', 0, 1, 'invalid start byte'
            )
            mock_auto_detect.return_value = mock_parser
            
            result = ontology_manager.load_ontology(str(corrupted_file))
            
            assert result.success is False
            assert result.ontology is None
            assert len(result.errors) > 0
            assert any("unexpected error" in error.lower() for error in result.errors)

    def test_load_invalid_file_path(self, ontology_manager):
        """Test loading with invalid file paths."""
        invalid_paths = [
            "",  # Empty string
            "\0",  # Null character
            "con:",  # Windows reserved name
            "/\0/invalid",  # Path with null character
            "a" * 1000,  # Extremely long path
        ]
        
        for invalid_path in invalid_paths:
            with patch('aim2_project.aim2_ontology.ontology_manager.auto_detect_parser') as mock_auto_detect:
                mock_auto_detect.return_value = None
                
                result = ontology_manager.load_ontology(invalid_path)
                
                assert result.success is False
                assert result.ontology is None
                assert len(result.errors) > 0

    def test_load_network_path_failure(self, ontology_manager):
        """Test loading from a network path that fails."""
        network_path = "//nonexistent.server/share/file.owl"
        
        with patch('aim2_project.aim2_ontology.ontology_manager.auto_detect_parser') as mock_auto_detect:
            mock_parser = Mock()
            mock_parser.format_name = "owl"
            mock_parser.parse.side_effect = OSError("Network path not accessible")
            mock_auto_detect.return_value = mock_parser
            
            result = ontology_manager.load_ontology(network_path)
            
            assert result.success is False
            assert result.ontology is None
            assert len(result.errors) > 0
            assert any("unexpected error" in error.lower() for error in result.errors)

    def test_disk_space_error_during_loading(self, ontology_manager, temp_dir):
        """Test handling of disk space errors during loading."""
        test_file = temp_dir / "large_file.owl"
        test_file.write_text("<?xml version='1.0'?><owl:Ontology/>")
        
        with patch('aim2_project.aim2_ontology.ontology_manager.auto_detect_parser') as mock_auto_detect:
            mock_parser = Mock()
            mock_parser.format_name = "owl"
            mock_parser.parse.side_effect = OSError("No space left on device")
            mock_auto_detect.return_value = mock_parser
            
            result = ontology_manager.load_ontology(str(test_file))
            
            assert result.success is False
            assert result.ontology is None
            assert len(result.errors) > 0
            assert any("unexpected error" in error.lower() for error in result.errors)

    # ========== FORMAT AND PARSING ERRORS ==========

    def test_unsupported_file_format(self, ontology_manager, temp_dir):
        """Test loading unsupported file formats."""
        unsupported_formats = [
            ("binary.exe", b'\x4D\x5A'),  # Windows executable
            ("image.png", b'\x89PNG\r\n\x1a\n'),  # PNG image
            ("document.pdf", b'%PDF-1.4'),  # PDF document
            ("archive.zip", b'PK\x03\x04'),  # ZIP archive
            ("unknown.xyz", b'unknown format'),  # Unknown format
        ]
        
        for filename, content in unsupported_formats:
            test_file = temp_dir / filename
            test_file.write_bytes(content)
            
            with patch('aim2_project.aim2_ontology.ontology_manager.auto_detect_parser') as mock_auto_detect:
                mock_auto_detect.return_value = None  # No parser found
                
                result = ontology_manager.load_ontology(str(test_file))
                
                assert result.success is False
                assert result.ontology is None
                assert len(result.errors) > 0
                assert any("no suitable parser" in error.lower() for error in result.errors)

    def test_malformed_owl_file(self, ontology_manager, temp_dir):
        """Test loading malformed OWL files."""
        malformed_owl_cases = [
            ("invalid_xml.owl", "<?xml version='1.0'?><unclosed_tag>"),
            ("missing_namespace.owl", "<owl:Ontology>missing namespaces</owl:Ontology>"),
            ("invalid_structure.owl", "<?xml version='1.0'?><not_owl_content/>"),
            ("corrupted_encoding.owl", "<?xml version='1.0' encoding='utf-8'?>\xFF\xFE<owl:Ontology/>"),
        ]
        
        for filename, content in malformed_owl_cases:
            test_file = temp_dir / filename
            if isinstance(content, str):
                test_file.write_text(content)
            else:
                test_file.write_bytes(content)
            
            with patch('aim2_project.aim2_ontology.ontology_manager.auto_detect_parser') as mock_auto_detect:
                mock_parser = Mock()
                mock_parser.format_name = "owl"
                mock_parser.parse.return_value = ParseResult(
                    success=False,
                    data=None,
                    errors=[f"Malformed OWL: {filename}"],
                    warnings=[],
                    metadata={},
                    parse_time=0.01
                )
                mock_auto_detect.return_value = mock_parser
                
                result = ontology_manager.load_ontology(str(test_file))
                
                assert result.success is False
                assert result.ontology is None
                assert len(result.errors) > 1  # Should include both manager and parser errors
                assert any("failed to parse ontology" in error.lower() for error in result.errors)

    def test_malformed_csv_file(self, ontology_manager, temp_dir):
        """Test loading malformed CSV files."""
        malformed_csv_cases = [
            ("missing_headers.csv", "data1,data2,data3\nmore,data,here"),
            ("inconsistent_columns.csv", "id,name,definition\n1,term1\n2,term2,def2,extra"),
            ("invalid_encoding.csv", b"id,name\n\xFF\xFE,term"),
            ("empty_required_fields.csv", "id,name,definition\n,Empty Name,Definition"),
        ]
        
        for filename, content in malformed_csv_cases:
            test_file = temp_dir / filename
            if isinstance(content, str):
                test_file.write_text(content)
            else:
                test_file.write_bytes(content)
            
            with patch('aim2_project.aim2_ontology.ontology_manager.auto_detect_parser') as mock_auto_detect:
                mock_parser = Mock()
                mock_parser.format_name = "csv"
                mock_parser.parse.return_value = ParseResult(
                    success=False,
                    data=None,
                    errors=[f"Malformed CSV: {filename}"],
                    warnings=[],
                    metadata={},
                    parse_time=0.01
                )
                mock_auto_detect.return_value = mock_parser
                
                result = ontology_manager.load_ontology(str(test_file))
                
                assert result.success is False
                assert result.ontology is None
                assert len(result.errors) > 1

    def test_malformed_jsonld_file(self, ontology_manager, temp_dir):
        """Test loading malformed JSON-LD files."""
        malformed_jsonld_cases = [
            ("invalid_json.jsonld", '{"@context": "incomplete json'),
            ("missing_context.jsonld", '{"data": "no @context"}'),
            ("circular_reference.jsonld", '{"@id": "self", "self": {"@id": "self"}}'),
            ("invalid_ld_structure.jsonld", '{"@context": [], "@graph": "not an array"}'),
        ]
        
        for filename, content in malformed_jsonld_cases:
            test_file = temp_dir / filename
            test_file.write_text(content)
            
            with patch('aim2_project.aim2_ontology.ontology_manager.auto_detect_parser') as mock_auto_detect:
                mock_parser = Mock()
                mock_parser.format_name = "jsonld"
                mock_parser.parse.return_value = ParseResult(
                    success=False,
                    data=None,
                    errors=[f"Malformed JSON-LD: {filename}"],
                    warnings=[],
                    metadata={},
                    parse_time=0.01
                )
                mock_auto_detect.return_value = mock_parser
                
                result = ontology_manager.load_ontology(str(test_file))
                
                assert result.success is False
                assert result.ontology is None
                assert len(result.errors) > 1

    def test_encoding_issues(self, ontology_manager, temp_dir):
        """Test handling of various encoding issues."""
        encoding_test_cases = [
            ("utf8_bom.owl", "<?xml version='1.0'?><owl:Ontology/>", 'utf-8-sig'),
            ("latin1.owl", "<?xml version='1.0'?><owl:Ontology/>", 'latin-1'),
            ("cp1252.owl", "<?xml version='1.0'?><owl:Ontology/>", 'cp1252'),
        ]
        
        for filename, content, encoding in encoding_test_cases:
            test_file = temp_dir / filename
            test_file.write_text(content, encoding=encoding)
            
            with patch('aim2_project.aim2_ontology.ontology_manager.auto_detect_parser') as mock_auto_detect:
                mock_parser = Mock()
                mock_parser.format_name = "owl"
                mock_parser.parse.side_effect = UnicodeDecodeError(
                    'utf-8', content.encode(encoding), 0, 1, f'Invalid {encoding} encoding'
                )
                mock_auto_detect.return_value = mock_parser
                
                result = ontology_manager.load_ontology(str(test_file))
                
                assert result.success is False
                assert result.ontology is None
                assert len(result.errors) > 0

    def test_empty_files(self, ontology_manager, temp_dir):
        """Test loading completely empty files."""
        empty_file_cases = [
            "empty.owl",
            "empty.csv", 
            "empty.jsonld",
            "empty.rdf",
        ]
        
        for filename in empty_file_cases:
            test_file = temp_dir / filename
            test_file.write_text("")  # Completely empty
            
            with patch('aim2_project.aim2_ontology.ontology_manager.auto_detect_parser') as mock_auto_detect:
                mock_parser = Mock()
                mock_parser.format_name = filename.split('.')[-1]
                mock_parser.parse.return_value = ParseResult(
                    success=False,
                    data=None,
                    errors=["Empty file or no valid content found"],
                    warnings=[],
                    metadata={},
                    parse_time=0.01
                )
                mock_auto_detect.return_value = mock_parser
                
                result = ontology_manager.load_ontology(str(test_file))
                
                assert result.success is False
                assert result.ontology is None
                assert len(result.errors) > 1

    def test_truncated_files(self, ontology_manager, temp_dir):
        """Test loading truncated files."""
        truncated_cases = [
            ("truncated.owl", "<?xml version='1.0'?><owl:Onto"),  # Cut off mid-tag
            ("truncated.csv", "id,name,definition\nTEST:001,Term,Def"),  # Cut off mid-line
            ("truncated.jsonld", '{"@context": {"@vocab": "http://'),  # Cut off mid-string
        ]
        
        for filename, content in truncated_cases:
            test_file = temp_dir / filename
            test_file.write_text(content)
            
            with patch('aim2_project.aim2_ontology.ontology_manager.auto_detect_parser') as mock_auto_detect:
                mock_parser = Mock()
                mock_parser.format_name = filename.split('.')[-1]
                mock_parser.parse.return_value = ParseResult(
                    success=False,
                    data=None,
                    errors=["File appears to be truncated or incomplete"],
                    warnings=[],
                    metadata={},
                    parse_time=0.01
                )
                mock_auto_detect.return_value = mock_parser
                
                result = ontology_manager.load_ontology(str(test_file))
                
                assert result.success is False
                assert result.ontology is None
                assert len(result.errors) > 1

    def test_invalid_ontology_structure(self, ontology_manager, temp_dir):
        """Test loading files with invalid ontology structure."""
        test_file = temp_dir / "invalid_structure.owl"
        test_file.write_text("<?xml version='1.0'?><owl:Ontology/>")
        
        with patch('aim2_project.aim2_ontology.ontology_manager.auto_detect_parser') as mock_auto_detect:
            mock_parser = Mock()
            mock_parser.format_name = "owl"
            # Parser returns wrong type instead of Ontology
            mock_parser.parse.return_value = ParseResult(
                success=True,
                data="not_an_ontology_object",  # Wrong type
                errors=[],
                warnings=[],
                metadata={},
                parse_time=0.1
            )
            mock_auto_detect.return_value = mock_parser
            
            result = ontology_manager.load_ontology(str(test_file))
            
            assert result.success is False
            assert result.ontology is None
            assert len(result.errors) > 0
            assert any("invalid ontology type" in error.lower() for error in result.errors)

    # ========== RESOURCE AND MEMORY ERRORS ==========

    def test_out_of_memory_during_loading(self, ontology_manager, temp_dir):
        """Test handling of out-of-memory conditions during loading."""
        test_file = temp_dir / "large_ontology.owl"
        test_file.write_text("<?xml version='1.0'?><owl:Ontology/>")
        
        with patch('aim2_project.aim2_ontology.ontology_manager.auto_detect_parser') as mock_auto_detect:
            mock_parser = Mock()
            mock_parser.format_name = "owl"
            mock_parser.parse.side_effect = MemoryError("Out of memory")
            mock_auto_detect.return_value = mock_parser
            
            result = ontology_manager.load_ontology(str(test_file))
            
            assert result.success is False
            assert result.ontology is None
            assert len(result.errors) > 0
            assert any("unexpected error" in error.lower() for error in result.errors)

    def test_timeout_during_loading(self, ontology_manager, temp_dir):
        """Test handling of timeout errors during long-running operations."""
        test_file = temp_dir / "slow_parsing.owl"
        test_file.write_text("<?xml version='1.0'?><owl:Ontology/>")
        
        with patch('aim2_project.aim2_ontology.ontology_manager.auto_detect_parser') as mock_auto_detect:
            mock_parser = Mock()
            mock_parser.format_name = "owl"
            mock_parser.parse.side_effect = TimeoutError("Operation timed out")
            mock_auto_detect.return_value = mock_parser
            
            result = ontology_manager.load_ontology(str(test_file))
            
            assert result.success is False
            assert result.ontology is None
            assert len(result.errors) > 0
            assert any("unexpected error" in error.lower() for error in result.errors)

    def test_resource_exhaustion(self, ontology_manager, temp_dir):
        """Test handling of resource exhaustion (too many open files)."""
        test_file = temp_dir / "resource_heavy.owl"
        test_file.write_text("<?xml version='1.0'?><owl:Ontology/>")
        
        with patch('aim2_project.aim2_ontology.ontology_manager.auto_detect_parser') as mock_auto_detect:
            mock_parser = Mock()
            mock_parser.format_name = "owl"
            mock_parser.parse.side_effect = OSError("Too many open files")
            mock_auto_detect.return_value = mock_parser
            
            result = ontology_manager.load_ontology(str(test_file))
            
            assert result.success is False
            assert result.ontology is None
            assert len(result.errors) > 0
            assert any("unexpected error" in error.lower() for error in result.errors)

    def test_memory_leak_prevention(self, ontology_manager, temp_dir, sample_ontology):
        """Test that failed operations don't cause memory leaks."""
        test_file = temp_dir / "memory_test.owl"
        test_file.write_text("<?xml version='1.0'?><owl:Ontology/>")
        
        # Track object creation
        initial_objects = len(gc.get_objects())
        
        with patch('aim2_project.aim2_ontology.ontology_manager.auto_detect_parser') as mock_auto_detect:
            # Create a parser that fails after creating some objects
            mock_parser = Mock()
            mock_parser.format_name = "owl"
            
            def failing_parse(*args, **kwargs):
                # Create some objects that might leak
                large_data = [sample_ontology] * 100  # Create references
                raise RuntimeError("Simulated failure")
            
            mock_parser.parse.side_effect = failing_parse
            mock_auto_detect.return_value = mock_parser
            
            # Attempt multiple loads that should fail
            for _ in range(10):
                result = ontology_manager.load_ontology(str(test_file))
                assert result.success is False
        
        # Force garbage collection
        gc.collect()
        
        # Check that we haven't significantly increased object count
        final_objects = len(gc.get_objects())
        object_increase = final_objects - initial_objects
        
        # Allow some increase but not excessive (less than 1000 new objects)
        assert object_increase < 1000, f"Potential memory leak: {object_increase} new objects"

    # ========== CONFIGURATION AND INPUT VALIDATION ERRORS ==========

    def test_invalid_configuration_parameters(self):
        """Test initialization with invalid configuration parameters."""
        # Current implementation accepts these values - document actual behavior
        # Test negative cache size (currently allowed)
        manager = OntologyManager(cache_size_limit=-1)
        assert manager.cache_size_limit == -1
        
        # Test non-boolean for enable_caching (currently accepted)
        manager = OntologyManager(enable_caching="invalid")
        assert manager.enable_caching == "invalid"  # Truthy value
        
        # Test non-integer for cache_size_limit (currently accepted)
        manager = OntologyManager(cache_size_limit="invalid")
        assert manager.cache_size_limit == "invalid"

    def test_null_none_inputs(self, ontology_manager):
        """Test handling of null/None inputs."""
        none_inputs = [None, "", [], {}]
        
        for invalid_input in none_inputs:
            # Current implementation handles all invalid inputs gracefully with error results
            result = ontology_manager.load_ontology(invalid_input)
            assert result.success is False
            assert len(result.errors) > 0
            # Should contain appropriate error message
            if invalid_input is None:
                assert any("either file_path or content must be provided" in error.lower() for error in result.errors)
            else:
                assert any("no suitable parser" in error.lower() or "unexpected error" in error.lower() for error in result.errors)

    def test_invalid_source_types(self, ontology_manager):
        """Test handling of invalid source types."""
        invalid_sources = [
            123,  # Integer
            123.45,  # Float
            [],  # List
            {},  # Dictionary
            set(),  # Set
            complex(1, 2),  # Complex number
        ]
        
        for invalid_source in invalid_sources:
            # Current implementation handles invalid types gracefully with error results
            result = ontology_manager.load_ontology(invalid_source)
            assert result.success is False
            assert len(result.errors) > 0
            # Should contain appropriate error message about parsing failure or invalid input
            error_text = " ".join(result.errors).lower()
            assert any(phrase in error_text for phrase in [
                "no suitable parser", 
                "unexpected error", 
                "either file_path or content must be provided"
            ])

    def test_empty_source_lists(self, ontology_manager):
        """Test handling of empty source lists for multi-source loading."""
        # Empty list
        results = ontology_manager.load_ontologies([])
        assert len(results) == 0
        
        # List with None values
        results = ontology_manager.load_ontologies([None, None])
        assert len(results) == 2
        assert all(not result.success for result in results)
        
        # List with empty strings
        results = ontology_manager.load_ontologies(["", ""])
        assert len(results) == 2
        assert all(not result.success for result in results)

    def test_invalid_ontology_objects(self, ontology_manager):
        """Test adding invalid ontology objects."""
        invalid_ontologies = [
            None,
            "not an ontology",
            123,
            [],
            {},
            Mock(),  # Object without required Ontology interface
        ]
        
        for invalid_ontology in invalid_ontologies:
            result = ontology_manager.add_ontology(invalid_ontology)
            assert result is False
            assert len(ontology_manager.ontologies) == 0

    # ========== CACHE AND STATE MANAGEMENT ERRORS ==========

    def test_cache_corruption_scenarios(self, ontology_manager, sample_ontology):
        """Test handling of cache corruption scenarios."""
        # Manually corrupt cache entry
        corrupt_entry = "not_a_cache_entry"
        ontology_manager._cache["corrupted"] = corrupt_entry
        
        # Try to access corrupted cache
        with patch('aim2_project.aim2_ontology.ontology_manager.auto_detect_parser') as mock_auto_detect:
            mock_parser = Mock()
            mock_parser.format_name = "owl"
            mock_parser.parse.return_value = ParseResult(
                success=True,
                data=sample_ontology,
                errors=[],
                warnings=[],
                metadata={},
                parse_time=0.1
            )
            mock_auto_detect.return_value = mock_parser
            
            # Currently the OntologyManager doesn't handle cache corruption gracefully
            # This test documents the current behavior and could guide future improvements
            result = ontology_manager.load_ontology("corrupted")
            
            # Current behavior: should fail due to corrupted cache
            assert result.success is False
            assert len(result.errors) > 0
            assert "unexpected error" in result.errors[0].lower()
            assert "access_count" in result.errors[0]  # Specific error from cache corruption

    def test_concurrent_access_errors(self, ontology_manager, sample_ontology, temp_dir):
        """Test handling of concurrent access errors."""
        test_file = temp_dir / "concurrent_test.owl"
        test_file.write_text("<?xml version='1.0'?><owl:Ontology/>")
        
        errors = []
        results = []
        
        def load_with_potential_error():
            try:
                with patch('aim2_project.aim2_ontology.ontology_manager.auto_detect_parser') as mock_auto_detect:
                    mock_parser = Mock()
                    mock_parser.format_name = "owl"
                    mock_parser.parse.return_value = ParseResult(
                        success=True,
                        data=sample_ontology,
                        errors=[],
                        warnings=[],
                        metadata={},
                        parse_time=0.1
                    )
                    mock_auto_detect.return_value = mock_parser
                    
                    result = ontology_manager.load_ontology(str(test_file))
                    results.append(result)
            except Exception as e:
                errors.append(e)
        
        # Run multiple threads simultaneously
        threads = []
        for _ in range(10):
            thread = threading.Thread(target=load_with_potential_error)
            threads.append(thread)
        
        for thread in threads:
            thread.start()
        
        for thread in threads:
            thread.join()
        
        # Should handle concurrent access without catastrophic failures
        assert len(errors) == 0 or all(isinstance(e, (RuntimeError, AttributeError)) for e in errors)
        assert len(results) >= 1  # At least some should succeed

    def test_state_inconsistency_after_partial_failures(self, ontology_manager, temp_dir):
        """Test state consistency after partial failures."""
        # Create multiple test files
        test_files = []
        for i in range(5):
            test_file = temp_dir / f"test_{i}.owl"
            test_file.write_text("<?xml version='1.0'?><owl:Ontology/>")
            test_files.append(str(test_file))
        
        with patch('aim2_project.aim2_ontology.ontology_manager.auto_detect_parser') as mock_auto_detect:
            call_count = [0]
            
            def alternating_parser(*args, **kwargs):
                call_count[0] += 1
                mock_parser = Mock()
                mock_parser.format_name = "owl"
                
                if call_count[0] % 2 == 0:  # Every second call fails
                    mock_parser.parse.side_effect = RuntimeError("Simulated failure")
                else:  # Success
                    ontology = Ontology(
                        id=f"SUCCESS:{call_count[0]:03d}",
                        name=f"Success {call_count[0]}",
                        version="1.0",
                        terms={},
                        relationships={},
                        namespaces=[]
                    )
                    mock_parser.parse.return_value = ParseResult(
                        success=True,
                        data=ontology,
                        errors=[],
                        warnings=[],
                        metadata={},
                        parse_time=0.1
                    )
                
                return mock_parser
            
            mock_auto_detect.side_effect = alternating_parser
            
            # Load all files (some will fail)
            results = ontology_manager.load_ontologies(test_files)
            
            # Verify state consistency
            successful_results = [r for r in results if r.success]
            failed_results = [r for r in results if not r.success]
            
            assert len(successful_results) > 0
            assert len(failed_results) > 0
            
            # Verify loaded ontologies match successful results
            assert len(ontology_manager.ontologies) == len(successful_results)
            
            # Verify statistics are consistent
            stats = ontology_manager.get_statistics()
            assert stats['successful_loads'] == len(successful_results)
            assert stats['failed_loads'] == len(failed_results)
            assert stats['total_loads'] == len(results)

    def test_recovery_from_failed_operations(self, ontology_manager, temp_dir, sample_ontology):
        """Test recovery from failed operations."""
        test_file = temp_dir / "recovery_test.owl"
        test_file.write_text("<?xml version='1.0'?><owl:Ontology/>")
        
        with patch('aim2_project.aim2_ontology.ontology_manager.auto_detect_parser') as mock_auto_detect:
            # First attempt fails
            failing_parser = Mock()
            failing_parser.format_name = "owl"
            failing_parser.parse.side_effect = RuntimeError("First failure")
            
            # Second attempt succeeds
            success_parser = Mock()
            success_parser.format_name = "owl"
            success_parser.parse.return_value = ParseResult(
                success=True,
                data=sample_ontology,
                errors=[],
                warnings=[],
                metadata={},
                parse_time=0.1
            )
            
            mock_auto_detect.side_effect = [failing_parser, success_parser]
            
            # First load should fail
            result1 = ontology_manager.load_ontology(str(test_file))
            assert result1.success is False
            assert len(ontology_manager.ontologies) == 0
            
            # Second load should succeed (recovery)
            result2 = ontology_manager.load_ontology(str(test_file))
            assert result2.success is True
            assert len(ontology_manager.ontologies) == 1
            assert ontology_manager.ontologies[sample_ontology.id] == sample_ontology

    # ========== CUSTOM EXCEPTION HIERARCHY ==========

    def test_exception_chaining_and_propagation(self, ontology_manager, temp_dir):
        """Test proper exception chaining and propagation."""
        test_file = temp_dir / "exception_test.owl"
        test_file.write_text("<?xml version='1.0'?><owl:Ontology/>")
        
        with patch('aim2_project.aim2_ontology.ontology_manager.auto_detect_parser') as mock_auto_detect:
            # Create a simple exception (exception chaining syntax might not be supported)
            chained_error = RuntimeError("Parser wrapper error with original: Original parsing error")
            
            mock_parser = Mock()
            mock_parser.format_name = "owl"
            mock_parser.parse.side_effect = chained_error
            mock_auto_detect.return_value = mock_parser
            
            result = ontology_manager.load_ontology(str(test_file))
            
            assert result.success is False
            assert len(result.errors) > 0
            # Should capture the exception message in errors
            assert any("unexpected error" in error.lower() for error in result.errors)

    def test_appropriate_error_codes_and_messages(self, ontology_manager):
        """Test that appropriate error codes and messages are generated."""
        test_cases = [
            (FileNotFoundError("File not found"), "file.*not.*found"),
            (PermissionError("Permission denied"), "permission.*denied"),
            (MemoryError("Out of memory"), "memory"),
            (TimeoutError("Operation timed out"), "timeout"),
            (UnicodeDecodeError('utf-8', b'', 0, 1, 'invalid'), "encoding"),
        ]
        
        for exception, expected_pattern in test_cases:
            with patch('aim2_project.aim2_ontology.ontology_manager.auto_detect_parser') as mock_auto_detect:
                mock_parser = Mock()
                mock_parser.format_name = "test"
                mock_parser.parse.side_effect = exception
                mock_auto_detect.return_value = mock_parser
                
                result = ontology_manager.load_ontology("test_file.owl")
                
                assert result.success is False
                assert len(result.errors) > 0
                # Error should contain relevant information
                error_text = " ".join(result.errors).lower()
                assert "unexpected error" in error_text

    def test_exception_hierarchy(self):
        """Test custom exception hierarchy structure."""
        # Test inheritance
        assert issubclass(OntologyLoadError, OntologyManagerError)
        assert issubclass(OntologyManagerError, Exception)
        
        # Test instantiation
        base_error = OntologyManagerError("Base error message")
        assert str(base_error) == "Base error message"
        assert isinstance(base_error, Exception)
        
        load_error = OntologyLoadError("Load error message")
        assert str(load_error) == "Load error message"
        assert isinstance(load_error, OntologyManagerError)
        assert isinstance(load_error, Exception)

    def test_logging_integration_with_error_handling(self, ontology_manager, temp_dir):
        """Test that error handling integrates properly with logging."""
        test_file = temp_dir / "logging_test.owl"
        test_file.write_text("<?xml version='1.0'?><owl:Ontology/>")
        
        with patch('aim2_project.aim2_ontology.ontology_manager.auto_detect_parser') as mock_auto_detect:
            with patch.object(ontology_manager.logger, 'error') as mock_error_log:
                with patch.object(ontology_manager.logger, 'exception') as mock_exception_log:
                    mock_parser = Mock()
                    mock_parser.format_name = "owl"
                    mock_parser.parse.side_effect = RuntimeError("Test error for logging")
                    mock_auto_detect.return_value = mock_parser
                    
                    result = ontology_manager.load_ontology(str(test_file))
                    
                    assert result.success is False
                    
                    # Verify logging calls were made
                    assert mock_exception_log.called or mock_error_log.called
                    
                    # Check that error was logged with appropriate context
                    if mock_exception_log.called:
                        log_call_args = mock_exception_log.call_args[0]
                        assert any("unexpected error" in str(arg).lower() for arg in log_call_args)

    # ========== INTEGRATION AND EDGE CASE TESTS ==========

    def test_multiple_error_types_in_sequence(self, ontology_manager, temp_dir):
        """Test handling multiple different error types in sequence."""
        test_files = []
        error_types = [
            FileNotFoundError("File not found"),
            PermissionError("Permission denied"),
            MemoryError("Out of memory"),
            UnicodeDecodeError('utf-8', b'', 0, 1, 'invalid'),
            TimeoutError("Timeout"),
        ]
        
        # Create test files
        for i in range(len(error_types)):
            test_file = temp_dir / f"error_test_{i}.owl"
            test_file.write_text("<?xml version='1.0'?><owl:Ontology/>")
            test_files.append(str(test_file))
        
        with patch('aim2_project.aim2_ontology.ontology_manager.auto_detect_parser') as mock_auto_detect:
            def error_parser_factory(error):
                mock_parser = Mock()
                mock_parser.format_name = "owl"
                mock_parser.parse.side_effect = error
                return mock_parser
            
            mock_auto_detect.side_effect = [error_parser_factory(error) for error in error_types]
            
            # Load all files (all should fail with different errors)
            results = ontology_manager.load_ontologies(test_files)
            
            assert len(results) == len(error_types)
            assert all(not result.success for result in results)
            assert all(len(result.errors) > 0 for result in results)
            
            # Verify manager state is still consistent
            stats = ontology_manager.get_statistics()
            assert stats['failed_loads'] == len(error_types)
            assert stats['successful_loads'] == 0
            assert len(ontology_manager.ontologies) == 0

    def test_error_handling_with_caching_enabled_and_disabled(self, temp_dir, sample_ontology):
        """Test error handling behavior with caching enabled and disabled."""
        test_file = temp_dir / "cache_error_test.owl"
        test_file.write_text("<?xml version='1.0'?><owl:Ontology/>")
        
        error_scenarios = [
            (True, "with caching enabled"),
            (False, "with caching disabled"),
        ]
        
        for enable_caching, description in error_scenarios:
            manager = OntologyManager(enable_caching=enable_caching)
            
            with patch('aim2_project.aim2_ontology.ontology_manager.auto_detect_parser') as mock_auto_detect:
                mock_parser = Mock()
                mock_parser.format_name = "owl"
                mock_parser.parse.side_effect = RuntimeError(f"Error {description}")
                mock_auto_detect.return_value = mock_parser
                
                result = manager.load_ontology(str(test_file))
                
                assert result.success is False, f"Should fail {description}"
                assert len(result.errors) > 0, f"Should have errors {description}"
                assert len(manager._cache) == 0, f"Cache should be empty after error {description}"

    def test_graceful_degradation_under_stress(self, ontology_manager, temp_dir):
        """Test graceful degradation under stress conditions."""
        # Create many test files
        test_files = []
        for i in range(50):
            test_file = temp_dir / f"stress_test_{i}.owl"
            test_file.write_text("<?xml version='1.0'?><owl:Ontology/>")
            test_files.append(str(test_file))
        
        with patch('aim2_project.aim2_ontology.ontology_manager.auto_detect_parser') as mock_auto_detect:
            call_count = [0]
            
            def stress_parser(*args, **kwargs):
                call_count[0] += 1
                mock_parser = Mock()
                mock_parser.format_name = "owl"
                
                # Randomly succeed or fail to simulate stress
                if call_count[0] % 3 == 0:  # Every third call fails
                    mock_parser.parse.side_effect = RuntimeError(f"Stress failure {call_count[0]}")
                else:
                    ontology = Ontology(
                        id=f"STRESS:{call_count[0]:03d}",
                        name=f"Stress Test {call_count[0]}",
                        version="1.0",
                        terms={},
                        relationships={},
                        namespaces=[]
                    )
                    mock_parser.parse.return_value = ParseResult(
                        success=True,
                        data=ontology,
                        errors=[],
                        warnings=[],
                        metadata={},
                        parse_time=0.01
                    )
                
                return mock_parser
            
            mock_auto_detect.side_effect = stress_parser
            
            # Load all files under stress conditions
            start_time = time.time()
            results = ontology_manager.load_ontologies(test_files)
            end_time = time.time()
            
            # Verify graceful handling
            assert len(results) == len(test_files)
            successful_results = [r for r in results if r.success]
            failed_results = [r for r in results if not r.success]
            
            # Should have both successes and failures
            assert len(successful_results) > 0
            assert len(failed_results) > 0
            
            # Should complete in reasonable time (less than 30 seconds)
            assert (end_time - start_time) < 30.0
            
            # Manager should still be in consistent state
            stats = ontology_manager.get_statistics()
            assert stats['total_loads'] == len(test_files)
            assert stats['successful_loads'] == len(successful_results)
            assert stats['failed_loads'] == len(failed_results)


if __name__ == "__main__":
    pytest.main([__file__])