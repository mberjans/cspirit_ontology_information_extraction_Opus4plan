"""
pytest configuration file for figure/table extraction tests.

This module provides global test configuration, fixtures, and utilities
for the comprehensive test suite.
"""

import pytest
import sys
import logging
from pathlib import Path
from unittest.mock import Mock, patch

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Import test fixtures
from tests.fixtures.test_data import (
    sample_pdf_content, sample_xml_content, edge_case_pdf_content,
    edge_case_xml_content, performance_test_content, mock_pdf_libraries,
    mock_xml_libraries, content_extractor, quality_assessor,
    validate_figure_metadata, validate_table_metadata, validate_quality_metrics
)


def pytest_configure(config):
    """Configure pytest settings."""
    # Configure logging for tests
    logging.basicConfig(
        level=logging.WARNING,  # Reduce noise during testing
        format='%(levelname)s: %(name)s: %(message)s'
    )
    
    # Add custom markers
    config.addinivalue_line(
        "markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')"
    )
    config.addinivalue_line(
        "markers", "integration: marks tests as integration tests"
    )
    config.addinivalue_line(
        "markers", "unit: marks tests as unit tests"
    )
    config.addinivalue_line(
        "markers", "performance: marks tests as performance benchmarks"
    )


def pytest_collection_modifyitems(config, items):
    """Modify test collection to add markers automatically."""
    for item in items:
        # Mark integration tests
        if "integration" in str(item.fspath):
            item.add_marker(pytest.mark.integration)
        
        # Mark unit tests
        if "unit" in str(item.fspath):
            item.add_marker(pytest.mark.unit)
        
        # Mark slow tests based on name patterns
        if any(keyword in item.name.lower() for keyword in ["performance", "large", "benchmark"]):
            item.add_marker(pytest.mark.slow)
        
        # Mark performance tests
        if "performance" in item.name.lower():
            item.add_marker(pytest.mark.performance)


@pytest.fixture(scope="session")
def test_config():
    """Test configuration settings."""
    return {
        "timeout": 30,  # Default test timeout in seconds
        "max_figures": 100,  # Maximum expected figures in test documents
        "max_tables": 100,   # Maximum expected tables in test documents
        "min_quality_score": 0.0,  # Minimum acceptable quality score
        "max_quality_score": 1.0   # Maximum quality score
    }


@pytest.fixture(autouse=True)
def mock_external_dependencies():
    """Mock external dependencies that aren't needed for testing."""
    with patch('aim2_project.aim2_utils.config_manager.ConfigManager') as mock_config:
        # Mock config manager
        mock_config_instance = Mock()
        mock_config_instance.get.return_value = {}
        mock_config.return_value = mock_config_instance
        yield


@pytest.fixture
def mock_file_system():
    """Mock file system operations for testing."""
    with patch('pathlib.Path.exists') as mock_exists, \
         patch('pathlib.Path.is_file') as mock_isfile, \
         patch('builtins.open') as mock_open_file:
        
        mock_exists.return_value = True
        mock_isfile.return_value = True
        mock_open_file.return_value.__enter__.return_value.read.return_value = "mock file content"
        
        yield {
            'exists': mock_exists,
            'is_file': mock_isfile,
            'open': mock_open_file
        }


@pytest.fixture
def temporary_test_files(tmp_path):
    """Create temporary test files."""
    # Create test PDF content
    pdf_file = tmp_path / "test.pdf"
    pdf_file.write_text("Mock PDF content for testing")
    
    # Create test XML content
    xml_file = tmp_path / "test.xml"
    xml_file.write_text("<?xml version='1.0'?><article><fig id='test'/></article>")
    
    return {
        'pdf_file': pdf_file,
        'xml_file': xml_file,
        'temp_dir': tmp_path
    }


# Performance monitoring fixtures
@pytest.fixture
def performance_monitor():
    """Monitor test performance metrics."""
    import time
    import psutil
    import os
    
    class PerformanceMonitor:
        def __init__(self):
            self.start_time = None
            self.end_time = None
            self.start_memory = None
            self.end_memory = None
            self.process = psutil.Process(os.getpid())
        
        def start(self):
            self.start_time = time.time()
            self.start_memory = self.process.memory_info().rss
        
        def stop(self):
            self.end_time = time.time()
            self.end_memory = self.process.memory_info().rss
        
        @property
        def duration(self):
            if self.start_time and self.end_time:
                return self.end_time - self.start_time
            return None
        
        @property
        def memory_delta(self):
            if self.start_memory and self.end_memory:
                return self.end_memory - self.start_memory
            return None
        
        def get_stats(self):
            return {
                'duration': self.duration,
                'memory_delta': self.memory_delta,
                'start_memory_mb': self.start_memory / 1024 / 1024 if self.start_memory else None,
                'end_memory_mb': self.end_memory / 1024 / 1024 if self.end_memory else None
            }
    
    return PerformanceMonitor()


# Test result validation utilities
class TestResultValidator:
    """Utilities for validating test results."""
    
    @staticmethod
    def validate_extraction_result(result):
        """Validate extraction result structure."""
        assert isinstance(result, dict), "Result must be a dictionary"
        assert "figures" in result, "Result must contain 'figures' key"
        assert "tables" in result, "Result must contain 'tables' key"
        assert isinstance(result["figures"], list), "Figures must be a list"
        assert isinstance(result["tables"], list), "Tables must be a list"
        
        if "summary" in result:
            summary = result["summary"]
            assert isinstance(summary, dict), "Summary must be a dictionary"
            if "total_figures" in summary:
                assert summary["total_figures"] >= 0, "Total figures must be non-negative"
            if "total_tables" in summary:
                assert summary["total_tables"] >= 0, "Total tables must be non-negative"
    
    @staticmethod
    def validate_figure_structure(figure):
        """Validate individual figure structure."""
        assert isinstance(figure, dict), "Figure must be a dictionary"
        
        # Basic required fields
        required_fields = ["id", "caption"]
        for field in required_fields:
            assert field in figure, f"Figure missing required field: {field}"
        
        # Validate ID
        assert isinstance(figure["id"], str), "Figure ID must be string"
        assert len(figure["id"]) > 0, "Figure ID must not be empty"
        
        # Validate caption
        assert isinstance(figure["caption"], str), "Figure caption must be string"
    
    @staticmethod
    def validate_table_structure(table):
        """Validate individual table structure."""
        assert isinstance(table, dict), "Table must be a dictionary"
        
        # Basic required fields
        required_fields = ["id", "caption"]
        for field in required_fields:
            assert field in table, f"Table missing required field: {field}"
        
        # Validate ID
        assert isinstance(table["id"], str), "Table ID must be string"
        assert len(table["id"]) > 0, "Table ID must not be empty"
        
        # Validate caption
        assert isinstance(table["caption"], str), "Table caption must be string"
        
        # Validate structure if present
        if "metadata" in table and "structure" in table["metadata"]:
            structure = table["metadata"]["structure"]
            if "rows" in structure:
                assert structure["rows"] >= 0, "Row count must be non-negative"
            if "columns" in structure:
                assert structure["columns"] >= 0, "Column count must be non-negative"
    
    @staticmethod
    def validate_quality_metrics(quality):
        """Validate quality metrics structure."""
        if not isinstance(quality, dict):
            return  # Quality metrics might not be present
        
        # Check score ranges
        score_fields = ["extraction_confidence", "completeness_score", "parsing_accuracy", "overall_quality"]
        for field in score_fields:
            if field in quality:
                score = quality[field]
                assert isinstance(score, (int, float)), f"{field} must be numeric"
                assert 0.0 <= score <= 1.0, f"{field} must be between 0.0 and 1.0, got {score}"


@pytest.fixture
def result_validator():
    """Provide test result validator."""
    return TestResultValidator()


# Error handling fixtures
@pytest.fixture
def error_tracker():
    """Track errors during testing."""
    class ErrorTracker:
        def __init__(self):
            self.errors = []
            self.warnings = []
        
        def add_error(self, error, context=""):
            self.errors.append({
                'error': str(error),
                'type': type(error).__name__,
                'context': context
            })
        
        def add_warning(self, warning, context=""):
            self.warnings.append({
                'warning': str(warning),
                'context': context
            })
        
        def has_errors(self):
            return len(self.errors) > 0
        
        def has_warnings(self):
            return len(self.warnings) > 0
        
        def get_summary(self):
            return {
                'error_count': len(self.errors),
                'warning_count': len(self.warnings),
                'errors': self.errors,
                'warnings': self.warnings
            }
    
    return ErrorTracker()


# Parametrized test data
@pytest.fixture(params=[
    "pdf_simple",
    "pdf_complex", 
    "xml_simple",
    "xml_complex"
])
def parser_test_data(request, sample_pdf_content, sample_xml_content):
    """Parametrized test data for different parser types."""
    data_map = {
        "pdf_simple": {
            "type": "pdf",
            "content": {
                "text": "Figure 1. Simple test figure. Table 1. Simple test table.",
                "extraction_method": "test"
            },
            "expected_figures": 1,
            "expected_tables": 1
        },
        "pdf_complex": {
            "type": "pdf", 
            "content": sample_pdf_content,
            "expected_figures": 4,
            "expected_tables": 4
        },
        "xml_simple": {
            "type": "xml",
            "content": {
                "xml_content": """
                <article>
                    <fig id="f1"><caption><p>Simple figure</p></caption></fig>
                    <table-wrap id="t1"><caption><p>Simple table</p></caption></table-wrap>
                </article>""",
                "extraction_method": "etree"
            },
            "expected_figures": 1,
            "expected_tables": 1
        },
        "xml_complex": {
            "type": "xml",
            "content": sample_xml_content,
            "expected_figures": 4,
            "expected_tables": 4
        }
    }
    
    return data_map[request.param]


# Cleanup fixtures
@pytest.fixture(autouse=True)
def cleanup_logging():
    """Clean up logging after each test."""
    yield
    # Reset logging configuration
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)


@pytest.fixture(autouse=True)
def reset_mocks():
    """Reset all mocks after each test."""
    yield
    # Any mock cleanup can go here if needed


# Test skip conditions
def pytest_runtest_setup(item):
    """Set up conditions for running tests."""
    # Skip slow tests unless explicitly requested
    if "slow" in item.keywords:
        if not item.config.getoption("--runslow", default=False):
            pytest.skip("need --runslow option to run")
    
    # Skip performance tests in normal runs
    if "performance" in item.keywords:
        if not item.config.getoption("--performance", default=False):
            pytest.skip("need --performance option to run")


def pytest_addoption(parser):
    """Add custom command line options."""
    parser.addoption(
        "--runslow", action="store_true", default=False, 
        help="run slow tests"
    )
    parser.addoption(
        "--performance", action="store_true", default=False,
        help="run performance tests"
    )


# Test reporting hooks
def pytest_terminal_summary(terminalreporter, exitstatus, config):
    """Add custom information to test summary."""
    if hasattr(terminalreporter.config, '_performance_stats'):
        stats = terminalreporter.config._performance_stats
        terminalreporter.write_sep("=", "Performance Summary")
        for test_name, test_stats in stats.items():
            terminalreporter.write_line(
                f"{test_name}: {test_stats['duration']:.2f}s, "
                f"Memory: {test_stats.get('memory_delta', 0) / 1024 / 1024:.1f}MB"
            )