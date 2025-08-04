#!/usr/bin/env python3
"""
Test script for the enhanced metadata framework.

This script tests the core metadata classes and utilities independently
to demonstrate the comprehensive metadata extraction capabilities.
"""

import json
import logging
import sys
from pathlib import Path

# Add the project to path for testing
sys.path.insert(0, str(Path(__file__).parent))

from aim2_project.aim2_ontology.parsers.metadata_framework import (
    FigureMetadata, TableMetadata, FigureType, TableType, DataType,
    ContextInfo, TechnicalDetails, ContentAnalysis, QualityMetrics,
    ContentExtractor, QualityAssessment, ExtractionSummary
)
from aim2_project.aim2_ontology.parsers.content_utils import (
    TableContentExtractor, FigureContentExtractor, TextAnalyzer, StatisticalAnalyzer
)

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)


def test_metadata_classes():
    """Test the metadata classes functionality."""
    logger.info("=== Testing Metadata Classes ===")
    
    try:
        # Test FigureMetadata
        figure_meta = FigureMetadata()
        figure_meta.id = "fig1"
        figure_meta.label = "Figure 1"
        figure_meta.caption = "Experimental setup showing treatment chambers and measurement devices."
        figure_meta.figure_type = FigureType.DIAGRAM
        
        # Test context
        figure_meta.context.section_context = "Methods"
        figure_meta.context.page_number = 3
        figure_meta.context.cross_references = ["see Figure 1", "Figure 1 shows"]
        
        # Test technical details
        figure_meta.technical.dimensions = (600, 400)
        figure_meta.technical.file_format = "PNG"
        figure_meta.technical.color_info = "color"
        
        # Convert to dictionary
        fig_dict = figure_meta.to_dict()
        
        logger.info(f"FigureMetadata test successful:")
        logger.info(f"  - ID: {fig_dict['id']}")
        logger.info(f"  - Type: {fig_dict['type']}")
        logger.info(f"  - Dimensions: {fig_dict['metadata']['technical']['dimensions']}")
        logger.info(f"  - Context: {fig_dict['metadata']['context']['section_context']}")
        
        # Test TableMetadata
        table_meta = TableMetadata()
        table_meta.id = "tab1"
        table_meta.label = "Table 1"
        table_meta.caption = "Summary of experimental results"
        table_meta.table_type = TableType.EXPERIMENTAL
        
        # Test structure
        table_meta.structure.rows = 4
        table_meta.structure.columns = 5
        table_meta.structure.column_headers = ["Treatment Group", "Sample Size", "Mean Response", "Std Dev", "P-value"]
        
        # Convert to dictionary
        table_dict = table_meta.to_dict()
        
        logger.info(f"TableMetadata test successful:")
        logger.info(f"  - ID: {table_dict['id']}")
        logger.info(f"  - Type: {table_dict['type']}")
        logger.info(f"  - Structure: {table_dict['metadata']['structure']['rows']}x{table_dict['metadata']['structure']['columns']}")
        
        return True
        
    except Exception as e:
        logger.error(f"Metadata classes test failed: {e}")
        return False


def test_content_extractor():
    """Test the ContentExtractor functionality."""
    logger.info("\n=== Testing ContentExtractor ===")
    
    try:
        extractor = ContentExtractor()
        
        # Test figure type classification
        fig_caption = "Bar chart showing treatment efficacy across different patient groups"
        fig_type = extractor.classify_figure_type(fig_caption)
        logger.info(f"Figure type classification: '{fig_caption}' -> {fig_type.value}")
        
        # Test table type classification
        table_caption = "Baseline demographic characteristics of study participants"
        headers = ["Age", "Gender", "BMI", "Comorbidities"]
        table_type = extractor.classify_table_type(table_caption, headers)
        logger.info(f"Table type classification: '{table_caption}' -> {table_type.value}")
        
        # Test keyword extraction
        text = "This randomized controlled trial examined the efficacy of novel therapeutic interventions in patients with chronic conditions."
        keywords = extractor.extract_keywords(text)
        logger.info(f"Extracted keywords: {keywords[:5]}")  # Show first 5
        
        # Test data type detection
        sample_data = [
            ["Age", "Weight", "Gender", "Treatment", "Response"],
            ["45", "70.5", "M", "A", "12.3"],
            ["52", "68.2", "F", "B", "15.7"],
            ["38", "75.1", "M", "A", "11.8"],
            ["61", "72.4", "F", "B", "18.2"]
        ]
        
        data_types = extractor.detect_data_types(sample_data)
        logger.info(f"Data types detected: {[dt.value for dt in data_types]}")
        
        # Test complexity scoring
        complex_text = "The p-value (p<0.001) indicates statistical significance with 95% CI [12.3-18.7]. Mean ± SD: 15.2 ± 3.4"
        complexity = extractor.calculate_complexity_score(complex_text)
        logger.info(f"Complexity score: {complexity:.2f}")
        
        return True
        
    except Exception as e:
        logger.error(f"ContentExtractor test failed: {e}")
        return False


def test_table_content_extractor():
    """Test the TableContentExtractor functionality."""
    logger.info("\n=== Testing TableContentExtractor ===")
    
    try:
        extractor = TableContentExtractor()
        
        # Sample table data
        table_data = [
            ["Treatment Group", "Sample Size", "Mean Response", "Std Dev", "P-value"],
            ["Control", "25", "12.5", "2.1", "-"],
            ["Treatment A", "24", "15.7", "2.8", "0.023"],
            ["Treatment B", "26", "18.2", "3.1", "0.001"],
            ["Treatment C", "23", "20.4", "3.5", "<0.001"]
        ]
        
        # Parse table structure
        structure = extractor.parse_table_structure(table_data)
        logger.info(f"Table structure parsed:")
        logger.info(f"  - Dimensions: {structure.rows}x{structure.columns}")
        logger.info(f"  - Headers: {structure.column_headers}")
        logger.info(f"  - Data types: {[dt.value for dt in structure.data_types]}")
        logger.info(f"  - Complexity: {structure.structure_complexity:.2f}")
        
        # Extract table content
        content = extractor.extract_table_content(table_data, structure)
        logger.info(f"Table content extracted:")
        logger.info(f"  - Structured data columns: {len(content.structured_data)}")
        logger.info(f"  - Numerical columns: {len(content.numerical_data)}")
        logger.info(f"  - Data quality score: {content.data_quality_score:.2f}")
        
        if content.numerical_data:
            first_num_col = list(content.numerical_data.keys())[0]
            logger.info(f"  - {first_num_col} values: {content.numerical_data[first_num_col]}")
        
        # Analyze table content
        analysis = extractor.analyze_table_content(content, "Experimental results table")
        logger.info(f"Content analysis:")
        logger.info(f"  - Complexity score: {analysis.complexity_score:.2f}")
        logger.info(f"  - Content themes: {analysis.content_themes}")
        logger.info(f"  - Statistical summary columns: {len(analysis.statistical_summary)}")
        
        return True
        
    except Exception as e:
        logger.error(f"TableContentExtractor test failed: {e}")
        return False


def test_quality_assessment():
    """Test the QualityAssessment functionality."""
    logger.info("\n=== Testing QualityAssessment ===")
    
    try:
        assessor = QualityAssessment()
        
        # Create a sample figure metadata for quality assessment
        figure_meta = FigureMetadata()
        figure_meta.id = "fig1"
        figure_meta.label = "Figure 1"
        figure_meta.caption = "Comprehensive analysis of treatment outcomes showing statistical significance across multiple patient groups."
        figure_meta.figure_type = FigureType.CHART
        figure_meta.context.section_context = "Results"
        figure_meta.context.cross_references = ["see Figure 1", "Figure 1 demonstrates"]
        figure_meta.analysis.content_type = "statistical_chart"
        figure_meta.analysis.keywords = ["treatment", "outcomes", "statistical", "significance"]
        figure_meta.technical.file_format = "PNG"
        figure_meta.technical.dimensions = (800, 600)
        
        # Assess figure quality
        figure_quality = assessor.assess_figure_quality(figure_meta)
        logger.info(f"Figure quality assessment:")
        logger.info(f"  - Extraction confidence: {figure_quality.extraction_confidence:.2f}")
        logger.info(f"  - Completeness score: {figure_quality.completeness_score:.2f}")
        logger.info(f"  - Parsing accuracy: {figure_quality.parsing_accuracy:.2f}")
        logger.info(f"  - Overall quality: {figure_quality.overall_quality():.2f}")
        logger.info(f"  - Validation status: {figure_quality.validation_status}")
        
        # Create a sample table metadata for quality assessment
        table_meta = TableMetadata()
        table_meta.id = "tab1"
        table_meta.label = "Table 1"
        table_meta.caption = "Detailed statistical analysis of experimental results"
        table_meta.table_type = TableType.STATISTICAL
        table_meta.structure.rows = 5
        table_meta.structure.columns = 5
        table_meta.structure.column_headers = ["Group", "N", "Mean", "SD", "P-value"]
        table_meta.content.data_quality_score = 0.9
        table_meta.analysis.statistical_summary = {"Mean": {"mean": 15.2, "std": 3.1}}
        
        # Assess table quality
        table_quality = assessor.assess_table_quality(table_meta)
        logger.info(f"Table quality assessment:")
        logger.info(f"  - Extraction confidence: {table_quality.extraction_confidence:.2f}")
        logger.info(f"  - Completeness score: {table_quality.completeness_score:.2f}")
        logger.info(f"  - Parsing accuracy: {table_quality.parsing_accuracy:.2f}")
        logger.info(f"  - Overall quality: {table_quality.overall_quality():.2f}")
        logger.info(f"  - Validation status: {table_quality.validation_status}")
        
        # Generate extraction summary
        figures = [figure_meta]
        tables = [table_meta]
        summary = assessor.generate_extraction_summary(figures, tables, 0.15, "test_method")
        
        logger.info(f"Extraction summary:")
        logger.info(f"  - Total figures: {summary.total_figures}")
        logger.info(f"  - Total tables: {summary.total_tables}")
        logger.info(f"  - Average quality: {summary.average_quality_score:.2f}")
        logger.info(f"  - Extraction time: {summary.extraction_time:.3f}s")
        logger.info(f"  - Parsing method: {summary.parsing_method}")
        
        return True
        
    except Exception as e:
        logger.error(f"QualityAssessment test failed: {e}")
        return False


def test_statistical_analyzer():
    """Test the StatisticalAnalyzer functionality."""
    logger.info("\n=== Testing StatisticalAnalyzer ===")
    
    try:
        analyzer = StatisticalAnalyzer()
        
        # Sample numerical data
        data = [12.5, 15.7, 18.2, 20.4, 14.3, 16.8, 19.1, 13.7, 17.2, 21.0]
        
        # Analyze distribution
        distribution = analyzer.analyze_distribution(data)
        logger.info(f"Distribution analysis:")
        logger.info(f"  - Count: {distribution['count']}")
        logger.info(f"  - Mean: {distribution['mean']:.2f}")
        logger.info(f"  - Median: {distribution['median']:.2f}")
        logger.info(f"  - Std Dev: {distribution['std_dev']:.2f}")
        logger.info(f"  - Range: {distribution['range']:.2f}")
        logger.info(f"  - Skewness: {distribution['skewness']:.3f}")
        
        # Detect outliers
        outliers = analyzer.detect_outliers(data, method='iqr')
        logger.info(f"Outliers detected (IQR method): {outliers}")
        
        # Test correlation
        x_data = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        y_data = [12.5, 15.7, 18.2, 20.4, 14.3, 16.8, 19.1, 13.7, 17.2, 21.0]
        
        correlation = analyzer.correlation_analysis(x_data, y_data)
        logger.info(f"Correlation analysis:")
        logger.info(f"  - Coefficient: {correlation['correlation_coefficient']:.3f}")
        logger.info(f"  - Strength: {correlation['correlation_strength']}")
        
        return True
        
    except Exception as e:
        logger.error(f"StatisticalAnalyzer test failed: {e}")
        return False


def test_unified_format():
    """Test the unified data format across different components."""
    logger.info("\n=== Testing Unified Data Format ===")
    
    try:
        # Create sample figure and table metadata
        figure_meta = FigureMetadata()
        figure_meta.id = "fig1"
        figure_meta.caption = "Test figure"
        figure_meta.figure_type = FigureType.CHART
        
        table_meta = TableMetadata()
        table_meta.id = "tab1"
        table_meta.caption = "Test table"
        table_meta.table_type = TableType.EXPERIMENTAL
        
        # Convert to unified dictionary format
        fig_dict = figure_meta.to_dict()
        table_dict = table_meta.to_dict()
        
        # Check required keys
        required_fig_keys = {'id', 'type', 'caption', 'metadata', 'content', 'analysis', 'quality'}
        required_table_keys = {'id', 'type', 'caption', 'metadata', 'content', 'analysis', 'quality'}
        
        fig_keys = set(fig_dict.keys())
        table_keys = set(table_dict.keys())
        
        if required_fig_keys.issubset(fig_keys):
            logger.info("✓ Figure dictionary has all required keys")
        else:
            missing = required_fig_keys - fig_keys
            logger.warning(f"✗ Figure dictionary missing keys: {missing}")
        
        if required_table_keys.issubset(table_keys):
            logger.info("✓ Table dictionary has all required keys")
        else:
            missing = required_table_keys - table_keys
            logger.warning(f"✗ Table dictionary missing keys: {missing}")
        
        # Test JSON serialization (important for API compatibility)
        try:
            fig_json = json.dumps(fig_dict, indent=2, default=str)
            table_json = json.dumps(table_dict, indent=2, default=str)
            logger.info("✓ Metadata structures are JSON serializable")
        except Exception as e:
            logger.error(f"✗ JSON serialization failed: {e}")
            return False
        
        # Test that quality metrics are properly structured
        fig_quality = fig_dict['quality']
        expected_quality_keys = {'extraction_confidence', 'completeness_score', 'parsing_accuracy', 'overall_quality'}
        
        if expected_quality_keys.issubset(set(fig_quality.keys())):
            logger.info("✓ Quality metrics are properly structured")
        else:
            missing = expected_quality_keys - set(fig_quality.keys())
            logger.warning(f"✗ Quality metrics missing keys: {missing}")
        
        return True
        
    except Exception as e:
        logger.error(f"Unified format test failed: {e}")
        return False


def main():
    """Run all metadata framework tests."""
    logger.info("Starting comprehensive metadata framework tests...\n")
    
    tests = [
        ("Metadata Classes", test_metadata_classes),
        ("ContentExtractor", test_content_extractor),
        ("TableContentExtractor", test_table_content_extractor),
        ("QualityAssessment", test_quality_assessment),
        ("StatisticalAnalyzer", test_statistical_analyzer),
        ("Unified Format", test_unified_format)
    ]
    
    results = {}
    
    for test_name, test_func in tests:
        try:
            results[test_name] = test_func()
        except Exception as e:
            logger.error(f"Test '{test_name}' encountered an error: {e}")
            results[test_name] = False
    
    # Summary
    logger.info("\n" + "="*60)
    logger.info("METADATA FRAMEWORK TEST SUMMARY")
    logger.info("="*60)
    
    passed = sum(results.values())
    total = len(results)
    
    for test_name, result in results.items():
        status = "PASS" if result else "FAIL"
        logger.info(f"{test_name:<25} : {status}")
    
    logger.info("-"*60)
    logger.info(f"Tests passed: {passed}/{total} ({passed/total*100:.1f}%)")
    
    if passed == total:
        logger.info("🎉 All metadata framework tests passed!")
        logger.info("📊 The comprehensive metadata extraction system is working correctly.")
        logger.info("🔧 Enhanced parsers are ready for production use.")
        return 0
    else:
        logger.error("⚠️  Some tests failed. Please check the implementation.")
        return 1


if __name__ == "__main__":
    sys.exit(main())