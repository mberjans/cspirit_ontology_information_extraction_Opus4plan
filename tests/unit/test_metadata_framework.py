"""
Comprehensive unit tests for the metadata framework module.

This module contains thorough tests for all metadata classes, enumerations,
and utility functions in the metadata framework.
"""

import json
import pytest
import statistics
from datetime import datetime
from typing import Dict, List, Any

from aim2_project.aim2_ontology.parsers.metadata_framework import (
    FigureType, TableType, DataType,
    QualityMetrics, ContextInfo, TechnicalDetails, ContentAnalysis,
    TableStructure, TableContent, FigureMetadata, TableMetadata,
    ExtractionSummary, ContentExtractor, QualityAssessment
)


class TestEnumerations:
    """Test enumeration classes."""
    
    def test_figure_type_enum(self):
        """Test FigureType enumeration values."""
        expected_types = {
            'chart', 'diagram', 'photo', 'schematic', 'flowchart', 
            'graph', 'map', 'illustration', 'composite', 'unknown'
        }
        actual_types = {ft.value for ft in FigureType}
        assert actual_types == expected_types
    
    def test_table_type_enum(self):
        """Test TableType enumeration values."""
        expected_types = {
            'data_summary', 'statistical', 'demographic', 'experimental',
            'comparison', 'reference', 'metadata', 'results', 'unknown'
        }
        actual_types = {tt.value for tt in TableType}
        assert actual_types == expected_types
    
    def test_data_type_enum(self):
        """Test DataType enumeration values."""
        expected_types = {
            'numeric', 'text', 'categorical', 'datetime', 'boolean',
            'percentage', 'currency', 'mixed', 'unknown'
        }
        actual_types = {dt.value for dt in DataType}
        assert actual_types == expected_types


class TestDataClasses:
    """Test data class functionality."""
    
    def test_quality_metrics_creation(self):
        """Test QualityMetrics data class creation and methods."""
        metrics = QualityMetrics()
        assert metrics.extraction_confidence == 0.0
        assert metrics.completeness_score == 0.0
        assert metrics.parsing_accuracy == 0.0
        assert metrics.validation_status == "pending"
        assert metrics.quality_issues == []
        assert metrics.processing_time == 0.0
        
        # Test overall quality calculation
        assert metrics.overall_quality() == 0.0
        
        # Test with values
        metrics.extraction_confidence = 0.8
        metrics.completeness_score = 0.9
        metrics.parsing_accuracy = 0.7
        assert metrics.overall_quality() == pytest.approx(0.8, rel=1e-2)
    
    def test_context_info_creation(self):
        """Test ContextInfo data class creation."""
        context = ContextInfo()
        assert context.section_context == ""
        assert context.page_number is None
        assert context.section_number is None
        assert context.cross_references == []
        assert context.related_text == ""
        assert context.document_position == 0.0
        
        # Test with values
        context.section_context = "Methods"
        context.page_number = 5
        context.cross_references = ["see Figure 1", "Table 1 shows"]
        assert context.section_context == "Methods"
        assert context.page_number == 5
        assert len(context.cross_references) == 2
    
    def test_technical_details_creation(self):
        """Test TechnicalDetails data class creation."""
        tech = TechnicalDetails()
        assert tech.file_format is None
        assert tech.dimensions is None
        assert tech.resolution is None
        assert tech.color_info is None
        assert tech.file_size is None
        
        # Test with values
        tech.file_format = "PNG"
        tech.dimensions = (800, 600)
        tech.resolution = 300
        tech.color_info = "color"
        tech.file_size = 1024000
        
        assert tech.file_format == "PNG"
        assert tech.dimensions == (800, 600)
        assert tech.resolution == 300
        assert tech.color_info == "color"
        assert tech.file_size == 1024000
    
    def test_content_analysis_creation(self):
        """Test ContentAnalysis data class creation."""
        analysis = ContentAnalysis()
        assert analysis.content_type == ""
        assert analysis.complexity_score == 0.0
        assert analysis.visual_elements == []
        assert analysis.text_content == []
        assert analysis.numerical_content == []
        assert analysis.statistical_summary == {}
        assert analysis.content_themes == []
        assert analysis.keywords == []
    
    def test_table_structure_creation(self):
        """Test TableStructure data class creation."""
        structure = TableStructure()
        assert structure.rows == 0
        assert structure.columns == 0
        assert structure.header_rows == 0
        assert structure.header_columns == 0
        assert structure.merged_cells == []
        assert structure.column_headers == []
        assert structure.row_headers == []
        assert structure.data_types == []
        assert structure.structure_complexity == 0.0
    
    def test_table_content_creation(self):
        """Test TableContent data class creation."""
        content = TableContent()
        assert content.raw_data == []
        assert content.structured_data == {}
        assert content.numerical_data == {}
        assert content.categorical_data == {}
        assert content.data_summary == {}
        assert content.missing_values == 0
        assert content.data_quality_score == 0.0


class TestFigureMetadata:
    """Test FigureMetadata class functionality."""
    
    def test_figure_metadata_creation(self):
        """Test basic FigureMetadata creation."""
        figure = FigureMetadata()
        assert figure.id == ""
        assert figure.label == ""
        assert figure.caption == ""
        assert figure.figure_type == FigureType.UNKNOWN
        assert figure.position == 0
        assert isinstance(figure.context, ContextInfo)
        assert isinstance(figure.technical, TechnicalDetails)
        assert isinstance(figure.analysis, ContentAnalysis)
        assert isinstance(figure.quality, QualityMetrics)
    
    def test_figure_metadata_to_dict(self):
        """Test FigureMetadata to_dict conversion."""
        figure = FigureMetadata()
        figure.id = "fig1"
        figure.label = "Figure 1"
        figure.caption = "Test figure caption"
        figure.figure_type = FigureType.CHART
        figure.position = 1
        
        figure.context.section_context = "Results"
        figure.context.page_number = 3
        figure.technical.file_format = "PNG"
        figure.technical.dimensions = (800, 600)
        figure.analysis.content_type = "statistical_chart"
        figure.analysis.keywords = ["test", "data"]
        figure.quality.extraction_confidence = 0.8
        
        fig_dict = figure.to_dict()
        
        # Test basic fields
        assert fig_dict["id"] == "fig1"
        assert fig_dict["label"] == "Figure 1"
        assert fig_dict["caption"] == "Test figure caption"
        assert fig_dict["type"] == "chart"
        assert fig_dict["position"] == 1
        
        # Test nested structures
        assert fig_dict["metadata"]["context"]["section_context"] == "Results"
        assert fig_dict["metadata"]["context"]["page_number"] == 3
        assert fig_dict["metadata"]["technical"]["file_format"] == "PNG"
        assert fig_dict["metadata"]["technical"]["dimensions"] == (800, 600)
        assert fig_dict["content"]["content_type"] == "statistical_chart"
        assert fig_dict["content"]["keywords"] == ["test", "data"]
        assert fig_dict["quality"]["extraction_confidence"] == 0.8
    
    def test_figure_metadata_json_serializable(self):
        """Test that FigureMetadata can be JSON serialized."""
        figure = FigureMetadata()
        figure.id = "fig1"
        figure.caption = "Test caption"
        figure.technical.dimensions = (800, 600)
        
        fig_dict = figure.to_dict()
        
        # Should not raise an exception
        json_str = json.dumps(fig_dict, default=str)
        assert isinstance(json_str, str)
        assert "fig1" in json_str


class TestTableMetadata:
    """Test TableMetadata class functionality."""
    
    def test_table_metadata_creation(self):
        """Test basic TableMetadata creation."""
        table = TableMetadata()
        assert table.id == ""
        assert table.label == ""
        assert table.caption == ""
        assert table.table_type == TableType.UNKNOWN
        assert table.position == 0
        assert isinstance(table.context, ContextInfo)
        assert isinstance(table.structure, TableStructure)
        assert isinstance(table.content, TableContent)
        assert isinstance(table.technical, TechnicalDetails)
        assert isinstance(table.analysis, ContentAnalysis)
        assert isinstance(table.quality, QualityMetrics)
    
    def test_table_metadata_to_dict(self):
        """Test TableMetadata to_dict conversion."""
        table = TableMetadata()
        table.id = "tab1"
        table.label = "Table 1"
        table.caption = "Test table caption"
        table.table_type = TableType.STATISTICAL
        table.position = 1
        
        table.context.section_context = "Results"
        table.structure.rows = 5
        table.structure.columns = 3
        table.structure.column_headers = ["A", "B", "C"]
        table.structure.data_types = [DataType.TEXT, DataType.NUMERIC, DataType.PERCENTAGE]
        table.content.missing_values = 2
        table.content.data_quality_score = 0.9
        table.quality.extraction_confidence = 0.85
        
        table_dict = table.to_dict()
        
        # Test basic fields
        assert table_dict["id"] == "tab1"
        assert table_dict["label"] == "Table 1"
        assert table_dict["caption"] == "Test table caption"
        assert table_dict["type"] == "statistical"
        assert table_dict["position"] == 1
        
        # Test nested structures
        assert table_dict["metadata"]["context"]["section_context"] == "Results"
        assert table_dict["metadata"]["structure"]["rows"] == 5
        assert table_dict["metadata"]["structure"]["columns"] == 3
        assert table_dict["metadata"]["structure"]["column_headers"] == ["A", "B", "C"]
        assert table_dict["metadata"]["structure"]["data_types"] == ["text", "numeric", "percentage"]
        assert table_dict["content"]["missing_values"] == 2
        assert table_dict["content"]["data_quality_score"] == 0.9
        assert table_dict["quality"]["extraction_confidence"] == 0.85
    
    def test_table_metadata_json_serializable(self):
        """Test that TableMetadata can be JSON serialized."""
        table = TableMetadata()
        table.id = "tab1"
        table.caption = "Test caption"
        table.structure.rows = 5
        table.structure.columns = 3
        
        table_dict = table.to_dict()
        
        # Should not raise an exception
        json_str = json.dumps(table_dict, default=str)
        assert isinstance(json_str, str)
        assert "tab1" in json_str


class TestContentExtractor:
    """Test ContentExtractor functionality."""
    
    @pytest.fixture
    def extractor(self):
        """Create ContentExtractor instance for testing."""
        return ContentExtractor()
    
    def test_classify_figure_type(self, extractor):
        """Test figure type classification."""
        test_cases = [
            ("Bar chart showing results", FigureType.CHART),
            ("Scatter plot of data points", FigureType.GRAPH),
            ("Schematic diagram of the system", FigureType.DIAGRAM),
            ("Photograph of the specimen", FigureType.PHOTO),
            ("Geographic map of the region", FigureType.MAP),
            ("Flowchart of the process", FigureType.DIAGRAM),
            ("Unknown content here", FigureType.UNKNOWN),
        ]
        
        for caption, expected_type in test_cases:
            result = extractor.classify_figure_type(caption)
            assert result == expected_type, f"Failed for caption: {caption}"
    
    def test_classify_table_type(self, extractor):
        """Test table type classification."""
        test_cases = [
            ("Baseline demographics of patients", ["Age", "Gender"], TableType.DEMOGRAPHIC),
            ("Statistical analysis results", ["Mean", "P-value"], TableType.STATISTICAL),
            ("Comparison of treatment groups", ["Control", "Treatment"], TableType.COMPARISON),
            ("Experimental results summary", ["Trial", "Outcome"], TableType.EXPERIMENTAL),
            ("Unknown table content", [], TableType.UNKNOWN),
        ]
        
        for caption, headers, expected_type in test_cases:
            result = extractor.classify_table_type(caption, headers)
            assert result == expected_type, f"Failed for caption: {caption}"
    
    def test_extract_keywords(self, extractor):
        """Test keyword extraction."""
        text = "This study examines treatment efficacy in randomized controlled trials with statistical significance."
        keywords = extractor.extract_keywords(text)
        
        assert isinstance(keywords, list)
        assert len(keywords) <= 10  # Should return max 10 keywords
        assert all(len(word) > 3 for word in keywords)  # All keywords should be > 3 chars
        assert "study" in keywords or "treatment" in keywords or "trials" in keywords
    
    def test_analyze_numerical_data(self, extractor):
        """Test numerical data analysis."""
        # Test with valid data
        data = [
            ["Age", "Weight", "Height"],
            ["25", "70.5", "175"],
            ["30", "68.2", "168"],
            ["35", "75.1", "180"]
        ]
        
        analysis = extractor.analyze_numerical_data(data)
        assert isinstance(analysis, dict)
        
        # Should have analysis for Weight and Height columns (numeric)
        if "Weight" in analysis:
            weight_stats = analysis["Weight"]
            assert "mean" in weight_stats
            assert "median" in weight_stats
            assert "count" in weight_stats
            assert weight_stats["count"] == 3
            assert weight_stats["mean"] == pytest.approx(71.27, rel=1e-2)
        
        # Test with empty data
        empty_analysis = extractor.analyze_numerical_data([])
        assert empty_analysis == {}
        
        # Test with non-numeric data
        text_data = [["Name"], ["Alice"], ["Bob"]]
        text_analysis = extractor.analyze_numerical_data(text_data)
        assert text_analysis == {}
    
    def test_detect_data_types(self, extractor):
        """Test data type detection."""
        data = [
            ["Name", "Age", "Active", "Score", "Percentage"],
            ["Alice", "25", "yes", "85.5", "90%"],
            ["Bob", "30", "no", "92.1", "95%"],
            ["Carol", "28", "true", "78.3", "85%"]
        ]
        
        data_types = extractor.detect_data_types(data)
        assert len(data_types) == 5
        
        # Name should be TEXT
        assert data_types[0] == DataType.TEXT
        # Age should be NUMERIC
        assert data_types[1] == DataType.NUMERIC
        # Active should be BOOLEAN
        assert data_types[2] == DataType.BOOLEAN
        # Score should be NUMERIC
        assert data_types[3] == DataType.NUMERIC
        # Percentage should be PERCENTAGE
        assert data_types[4] == DataType.PERCENTAGE
    
    def test_calculate_complexity_score(self, extractor):
        """Test complexity score calculation."""
        # Simple text
        simple_text = "This is a simple sentence."
        simple_score = extractor.calculate_complexity_score(simple_text)
        assert 0.0 <= simple_score <= 1.0
        
        # Complex scientific text
        complex_text = "The p-value (p<0.001) shows statistical significance with 95% CI [12.3-18.7]. Mean ± SD: 15.2 ± 3.4 µg/mL."
        complex_score = extractor.calculate_complexity_score(complex_text)
        assert 0.0 <= complex_score <= 1.0
        assert complex_score > simple_score
        
        # Empty text
        empty_score = extractor.calculate_complexity_score("")
        assert empty_score == 0.0
        
        # Test with structure info
        structure_info = {"rows": 10, "columns": 5, "merged_cells": []}
        structure_score = extractor.calculate_complexity_score("simple text", structure_info)
        assert structure_score > simple_score


class TestQualityAssessment:
    """Test QualityAssessment functionality."""
    
    @pytest.fixture
    def assessor(self):
        """Create QualityAssessment instance for testing."""
        return QualityAssessment()
    
    def test_assess_figure_quality_complete(self, assessor):
        """Test figure quality assessment with complete metadata."""
        figure = FigureMetadata()
        figure.id = "fig1"
        figure.label = "Figure 1"
        figure.caption = "Comprehensive analysis showing statistical significance"
        figure.figure_type = FigureType.CHART
        figure.context.section_context = "Results"
        figure.context.cross_references = ["see Figure 1"]
        figure.analysis.content_type = "chart"
        figure.analysis.keywords = ["analysis", "statistical"]
        figure.technical.file_format = "PNG"
        figure.technical.dimensions = (800, 600)
        figure.analysis.visual_elements = ["axes", "legend"]
        figure.analysis.complexity_score = 0.5
        
        quality = assessor.assess_figure_quality(figure)
        
        assert isinstance(quality, QualityMetrics)
        assert quality.extraction_confidence > 0.7  # Should be high with complete data
        assert quality.completeness_score > 0.5
        assert quality.parsing_accuracy > 0.8
        assert quality.validation_status in ["passed", "partial"]
        assert quality.overall_quality() > 0.6
    
    def test_assess_figure_quality_minimal(self, assessor):
        """Test figure quality assessment with minimal metadata."""
        figure = FigureMetadata()
        # Only set minimal fields
        figure.caption = "Simple figure"
        
        quality = assessor.assess_figure_quality(figure)
        
        assert isinstance(quality, QualityMetrics)
        assert quality.extraction_confidence < 0.5  # Should be low with minimal data
        assert quality.completeness_score < 0.5
        assert len(quality.quality_issues) > 0  # Should have issues
        assert "Missing figure ID" in quality.quality_issues
    
    def test_assess_table_quality_complete(self, assessor):
        """Test table quality assessment with complete metadata."""
        table = TableMetadata()
        table.id = "tab1"
        table.label = "Table 1"
        table.caption = "Statistical analysis of experimental results"
        table.table_type = TableType.STATISTICAL
        table.structure.rows = 5
        table.structure.columns = 4
        table.structure.column_headers = ["Group", "Mean", "SD", "P-value"]
        table.structure.data_types = [DataType.TEXT, DataType.NUMERIC, DataType.NUMERIC, DataType.NUMERIC]
        table.content.raw_data = [["header"] * 4, ["data"] * 4, ["more"] * 4]
        table.content.structured_data = {"Group": ["A", "B"], "Mean": [1.0, 2.0]}
        table.content.data_summary = {"completeness": 0.95}
        table.content.data_quality_score = 0.9
        table.context.section_context = "Results"
        table.context.cross_references = ["Table 1"]
        table.analysis.statistical_summary = {"Mean": {"mean": 1.5}}
        table.analysis.keywords = ["statistical", "analysis"]
        
        quality = assessor.assess_table_quality(table)
        
        assert isinstance(quality, QualityMetrics)
        assert quality.extraction_confidence > 0.7
        assert quality.completeness_score > 0.7
        assert quality.parsing_accuracy > 0.7
        assert quality.validation_status in ["passed", "partial"]
        assert quality.overall_quality() > 0.7
    
    def test_assess_table_quality_structure_mismatch(self, assessor):
        """Test table quality assessment with structure/content mismatch."""
        table = TableMetadata()
        table.id = "tab1"
        table.caption = "Test table"
        table.structure.rows = 5
        table.structure.columns = 3
        table.content.raw_data = [["A", "B"], ["1", "2"]]  # Mismatch: 2 cols vs 3 expected
        
        quality = assessor.assess_table_quality(table)
        
        assert quality.parsing_accuracy < 1.0  # Should be penalized for mismatch
        assert any("mismatch" in issue.lower() for issue in quality.quality_issues)
    
    def test_generate_extraction_summary(self, assessor):
        """Test extraction summary generation."""
        # Create sample figure and table
        figure = FigureMetadata()
        figure.figure_type = FigureType.CHART
        figure.quality.extraction_confidence = 0.8
        figure.quality.completeness_score = 0.7
        figure.quality.parsing_accuracy = 0.9
        
        table = TableMetadata()
        table.table_type = TableType.STATISTICAL
        table.quality.extraction_confidence = 0.9
        table.quality.completeness_score = 0.8
        table.quality.parsing_accuracy = 0.85
        
        summary = assessor.generate_extraction_summary(
            figures=[figure],
            tables=[table],
            extraction_time=0.5,
            parsing_method="enhanced_parser"
        )
        
        assert isinstance(summary, ExtractionSummary)
        assert summary.total_figures == 1
        assert summary.total_tables == 1
        assert summary.figures_by_type["chart"] == 1
        assert summary.tables_by_type["statistical"] == 1
        assert summary.extraction_time == 0.5
        assert summary.parsing_method == "enhanced_parser"
        
        # Average quality should be calculated correctly
        expected_avg = (0.8 + 0.7 + 0.9 + 0.9 + 0.8 + 0.85) / 6
        assert summary.average_quality_score == pytest.approx(expected_avg, rel=1e-2)
    
    def test_generate_extraction_summary_empty(self, assessor):
        """Test extraction summary with no figures or tables."""
        summary = assessor.generate_extraction_summary([], [], 0.1, "test")
        
        assert summary.total_figures == 0
        assert summary.total_tables == 0
        assert summary.average_quality_score == 0.0
        assert "No figures or tables found" in summary.processing_notes


class TestEdgeCases:
    """Test edge cases and error conditions."""
    
    def test_invalid_enum_values(self):
        """Test handling of invalid enum values."""
        # These should not raise exceptions due to enum design
        assert FigureType.UNKNOWN.value == "unknown"
        assert TableType.UNKNOWN.value == "unknown"
        assert DataType.UNKNOWN.value == "unknown"
    
    def test_empty_data_structures(self):
        """Test behavior with empty data structures."""
        extractor = ContentExtractor()
        
        # Empty data type detection
        data_types = extractor.detect_data_types([])
        assert data_types == []
        
        # Empty numerical analysis
        analysis = extractor.analyze_numerical_data([])
        assert analysis == {}
        
        # Empty keyword extraction
        keywords = extractor.extract_keywords("")
        assert keywords == []
    
    def test_malformed_data(self):
        """Test handling of malformed data."""
        extractor = ContentExtractor()
        
        # Inconsistent row lengths
        malformed_data = [
            ["A", "B", "C"],
            ["1", "2"],  # Missing column
            ["3", "4", "5", "6"]  # Extra column
        ]
        
        # Should not crash
        data_types = extractor.detect_data_types(malformed_data)
        assert isinstance(data_types, list)
        
        analysis = extractor.analyze_numerical_data(malformed_data)
        assert isinstance(analysis, dict)
    
    def test_large_data_handling(self):
        """Test handling of large datasets."""
        extractor = ContentExtractor()
        
        # Create large dataset
        large_data = [["col1", "col2"]]
        for i in range(1000):
            large_data.append([str(i), str(i * 2)])
        
        # Should complete without issues
        data_types = extractor.detect_data_types(large_data)
        assert len(data_types) == 2
        
        analysis = extractor.analyze_numerical_data(large_data)
        assert isinstance(analysis, dict)
    
    def test_unicode_handling(self):
        """Test handling of Unicode characters."""
        extractor = ContentExtractor()
        
        unicode_text = "Étude statistique avec données spécialisées: α=0.05, β-test, ±2.5°C"
        keywords = extractor.extract_keywords(unicode_text)
        assert isinstance(keywords, list)
        
        complexity = extractor.calculate_complexity_score(unicode_text)
        assert 0.0 <= complexity <= 1.0


class TestIntegration:
    """Integration tests for metadata framework components."""
    
    def test_end_to_end_figure_processing(self):
        """Test complete figure processing workflow."""
        # Create and populate figure metadata
        figure = FigureMetadata()
        figure.id = "fig1"
        figure.caption = "Bar chart showing p<0.001 statistical significance across treatment groups"
        figure.figure_type = FigureType.CHART
        
        # Add context
        figure.context.section_context = "Results"
        figure.context.page_number = 5
        figure.context.cross_references = ["Figure 1 shows", "see Fig. 1"]
        
        # Add technical details
        figure.technical.file_format = "PNG"
        figure.technical.dimensions = (800, 600)
        figure.technical.color_info = "color"
        
        # Analyze content
        extractor = ContentExtractor()
        figure.analysis.keywords = extractor.extract_keywords(figure.caption)
        figure.analysis.complexity_score = extractor.calculate_complexity_score(figure.caption)
        
        # Assess quality
        assessor = QualityAssessment()
        figure.quality = assessor.assess_figure_quality(figure)
        
        # Convert to dict and verify
        fig_dict = figure.to_dict()
        
        assert fig_dict["id"] == "fig1"
        assert fig_dict["type"] == "chart"
        assert "statistical" in fig_dict["content"]["keywords"]
        assert fig_dict["quality"]["overall_quality"] > 0.5
        
        # Verify JSON serialization
        json_str = json.dumps(fig_dict, default=str)
        assert isinstance(json_str, str)
    
    def test_end_to_end_table_processing(self):
        """Test complete table processing workflow."""
        # Create sample table data
        table_data = [
            ["Treatment", "N", "Mean", "SD", "P-value"],
            ["Control", "25", "12.5", "2.1", "-"],
            ["Drug A", "24", "15.7", "2.8", "0.023"],
            ["Drug B", "26", "18.2", "3.1", "0.001"]
        ]
        
        # Create and populate table metadata
        table = TableMetadata()
        table.id = "tab1"
        table.caption = "Statistical comparison of treatment efficacy"
        table.table_type = TableType.STATISTICAL
        
        # Process structure and content
        from aim2_project.aim2_ontology.parsers.content_utils import TableContentExtractor
        content_extractor = TableContentExtractor()
        
        table.structure = content_extractor.parse_table_structure(table_data)
        table.content = content_extractor.extract_table_content(table_data, table.structure)
        table.analysis = content_extractor.analyze_table_content(table.content, table.caption)
        
        # Assess quality
        assessor = QualityAssessment()
        table.quality = assessor.assess_table_quality(table)
        
        # Convert to dict and verify
        table_dict = table.to_dict()
        
        assert table_dict["id"] == "tab1"
        assert table_dict["type"] == "statistical"
        assert table_dict["metadata"]["structure"]["rows"] == 4
        assert table_dict["metadata"]["structure"]["columns"] == 5
        assert table_dict["quality"]["overall_quality"] > 0.5
        
        # Verify JSON serialization
        json_str = json.dumps(table_dict, default=str)
        assert isinstance(json_str, str)
    
    def test_cross_parser_compatibility(self):
        """Test that metadata format is compatible across different parsers."""
        # Create figure metadata (simulating PDF parser output)
        pdf_figure = FigureMetadata()
        pdf_figure.id = "pdf_fig1"
        pdf_figure.source_parser = "pdf_parser"
        pdf_figure.extraction_method = "pdfplumber"
        
        # Create figure metadata (simulating XML parser output)
        xml_figure = FigureMetadata()
        xml_figure.id = "xml_fig1"
        xml_figure.source_parser = "xml_parser"
        xml_figure.extraction_method = "lxml"
        
        # Both should have same dict structure
        pdf_dict = pdf_figure.to_dict()
        xml_dict = xml_figure.to_dict()
        
        # Same keys at top level
        assert set(pdf_dict.keys()) == set(xml_dict.keys())
        
        # Same metadata structure
        assert set(pdf_dict["metadata"].keys()) == set(xml_dict["metadata"].keys())
        assert set(pdf_dict["quality"].keys()) == set(xml_dict["quality"].keys())
        
        # Both should be JSON serializable
        pdf_json = json.dumps(pdf_dict, default=str)
        xml_json = json.dumps(xml_dict, default=str)
        assert isinstance(pdf_json, str)
        assert isinstance(xml_json, str)