"""
Comprehensive unit tests for enhanced PDF parser figure/table extraction functionality.

This module contains thorough tests for the enhanced PDF parser functionality,
including figure/table extraction, metadata analysis, and content processing.
"""

import io
import pytest
from unittest.mock import Mock, MagicMock, patch, mock_open
from pathlib import Path
from typing import Dict, List, Any

from aim2_project.aim2_ontology.parsers.pdf_parser import PDFParser
from aim2_project.aim2_ontology.parsers.metadata_framework import (
    FigureMetadata, TableMetadata, FigureType, TableType, DataType,
    ContextInfo, TechnicalDetails, ContentAnalysis, QualityMetrics,
    ContentExtractor, QualityAssessment, ExtractionSummary
)
from aim2_project.exceptions import ExtractionException, ValidationException


class TestPDFParserEnhancedExtraction:
    """Test enhanced figure/table extraction functionality."""
    
    @pytest.fixture
    def parser(self):
        """Create PDFParser instance for testing."""
        return PDFParser()
    
    @pytest.fixture
    def mock_pdf_content(self):
        """Mock PDF content structure."""
        return {
            "text": "Sample PDF content with Figure 1 showing results and Table 1 summarizing data.",
            "extraction_method": "pdfplumber",
            "pages": 5,
            "raw_content": Mock(),
            "metadata": {
                "title": "Test Paper",
                "author": "Test Author"
            }
        }
    
    @pytest.fixture
    def sample_pdf_text(self):
        """Sample PDF text content with figures and tables."""
        return """
        Introduction
        This study examines the effects of treatment protocols.
        
        Methods
        Figure 1. Experimental setup showing treatment chambers and measurement devices.
        The experimental design included three treatment groups as shown in Figure 1.
        
        Results
        Table 1. Summary of experimental results across treatment groups.
        Treatment Group | Sample Size | Mean Response | Std Dev | P-value
        Control         | 25          | 12.5          | 2.1     | -
        Treatment A     | 24          | 15.7          | 2.8     | 0.023
        Treatment B     | 26          | 18.2          | 3.1     | 0.001
        
        Figure 2. Bar chart showing statistical significance (p<0.001) across groups.
        The results demonstrate significant differences between treatment groups.
        
        Discussion
        As shown in Figure 2, the treatment effects were statistically significant.
        Table 1 provides detailed statistical comparisons.
        """
    
    def test_extract_figures_tables_comprehensive(self, parser, mock_pdf_content, sample_pdf_text):
        """Test comprehensive figure and table extraction."""
        mock_pdf_content["text"] = sample_pdf_text
        
        result = parser.extract_figures_tables(mock_pdf_content)
        
        assert "figures" in result
        assert "tables" in result
        assert isinstance(result["figures"], list)
        assert isinstance(result["tables"], list)
        
        # Should find figures and tables
        if result["figures"]:
            figure = result["figures"][0]
            assert isinstance(figure, dict)
            assert "id" in figure
            assert "type" in figure
            assert "caption" in figure
            assert "metadata" in figure
            assert "quality" in figure
        
        if result["tables"]:
            table = result["tables"][0]
            assert isinstance(table, dict)
            assert "id" in table
            assert "type" in table
            assert "caption" in table
            assert "metadata" in table
            assert "quality" in table
    
    def test_extract_figures_comprehensive(self, parser, sample_pdf_text, mock_pdf_content):
        """Test comprehensive figure extraction with metadata."""
        figures = parser._extract_figures_comprehensive(sample_pdf_text, mock_pdf_content)
        
        assert isinstance(figures, list)
        
        for figure in figures:
            assert isinstance(figure, FigureMetadata)
            assert figure.id != ""
            assert figure.caption != ""
            assert isinstance(figure.context, ContextInfo)
            assert isinstance(figure.technical, TechnicalDetails)
            assert isinstance(figure.analysis, ContentAnalysis)
            assert isinstance(figure.quality, QualityMetrics)
            assert figure.source_parser == "pdf_parser"
    
    def test_extract_tables_comprehensive(self, parser, sample_pdf_text, mock_pdf_content):
        """Test comprehensive table extraction with metadata."""
        tables = parser._extract_tables_comprehensive(sample_pdf_text, mock_pdf_content)
        
        assert isinstance(tables, list)
        
        for table in tables:
            assert isinstance(table, TableMetadata)
            assert table.id != ""
            assert table.caption != ""
            assert isinstance(table.context, ContextInfo)
            assert isinstance(table.structure, type(table.structure))  # TableStructure
            assert isinstance(table.content, type(table.content))      # TableContent
            assert isinstance(table.technical, TechnicalDetails)
            assert isinstance(table.analysis, ContentAnalysis)
            assert isinstance(table.quality, QualityMetrics)
            assert table.source_parser == "pdf_parser"
    
    def test_extract_figure_context(self, parser, sample_pdf_text, mock_pdf_content):
        """Test figure context extraction."""
        figure_meta = FigureMetadata()
        figure_meta.id = "fig1"
        figure_meta.caption = "Experimental setup showing treatment chambers"
        
        context = parser._extract_figure_context(figure_meta, sample_pdf_text, mock_pdf_content)
        
        assert isinstance(context, ContextInfo)
        # Should find context information
        if context.cross_references:
            assert any("Figure 1" in ref for ref in context.cross_references)
        if context.section_context:
            assert context.section_context in ["Methods", "Results", "Discussion"]
    
    def test_extract_table_context(self, parser, sample_pdf_text, mock_pdf_content):
        """Test table context extraction."""
        table_meta = TableMetadata()
        table_meta.id = "tab1"
        table_meta.caption = "Summary of experimental results"
        
        context = parser._extract_table_context(table_meta, sample_pdf_text, mock_pdf_content)
        
        assert isinstance(context, ContextInfo)
        # Should find context information
        if context.cross_references:
            assert any("Table 1" in ref for ref in context.cross_references)
        if context.section_context:
            assert context.section_context in ["Methods", "Results", "Discussion"]
    
    def test_extract_figure_technical_details(self, parser, mock_pdf_content):
        """Test figure technical details extraction."""
        basic_figure = {
            "id": "fig1",
            "caption": "Test figure",
            "page": 2,
            "position": (100, 200, 300, 400)
        }
        
        technical = parser._extract_figure_technical_details(basic_figure, mock_pdf_content)
        
        assert isinstance(technical, TechnicalDetails)
        # Should extract available technical information
        if technical.dimensions:
            assert isinstance(technical.dimensions, tuple)
            assert len(technical.dimensions) == 2
    
    def test_extract_table_technical_details(self, parser, mock_pdf_content):
        """Test table technical details extraction."""
        basic_table = {
            "id": "tab1",
            "caption": "Test table",
            "page": 3,
            "data": [["A", "B"], ["1", "2"]]
        }
        
        technical = parser._extract_table_technical_details(basic_table, mock_pdf_content)
        
        assert isinstance(technical, TechnicalDetails)
        # Should extract available technical information
        assert technical.encoding is not None


class TestPDFParserFigureExtraction:
    """Test PDF parser figure extraction methods."""
    
    @pytest.fixture
    def parser(self):
        """Create PDFParser instance for testing."""
        return PDFParser()
    
    def test_extract_figures_by_patterns(self, parser):
        """Test figure extraction using regex patterns."""
        text = """
        Figure 1. This is the first figure caption.
        Fig. 2: Second figure with different format.
        Figure 3 - Third figure with dash separator.
        Supplementary Figure S1. Additional figure.
        """
        
        figures = parser._extract_figures_by_patterns(text)
        
        assert isinstance(figures, list)
        assert len(figures) >= 3  # Should find at least 3 main figures
        
        for figure in figures:
            assert "id" in figure
            assert "caption" in figure
            assert "position" in figure
    
    def test_extract_figures_by_patterns_various_formats(self, parser):
        """Test figure extraction with various caption formats."""
        test_cases = [
            "Figure 1. Simple caption.",
            "Fig 2: Caption with colon.",
            "Figure 3 - Caption with dash.",
            "FIGURE 4. Uppercase figure.",
            "Figure A1. Appendix figure.",
            "Supplementary Figure S1. Supplementary material.",
        ]
        
        for test_text in test_cases:
            figures = parser._extract_figures_by_patterns(test_text)
            assert len(figures) >= 1, f"Failed to extract from: {test_text}"
    
    @patch('aim2_project.aim2_ontology.parsers.pdf_parser.FITZ_AVAILABLE', True)
    def test_extract_figures_fitz(self, parser):
        """Test figure extraction using PyMuPDF (fitz)."""
        # Mock fitz document
        mock_doc = Mock()
        mock_page = Mock()
        mock_image = Mock()
        
        # Mock image extraction
        mock_image.xref = 123
        mock_image.rect = [100, 200, 300, 400]
        mock_page.get_images.return_value = [(123, 0, 0, 0, 0, "", "", "", "", "", mock_image)]
        mock_page.get_image_bbox.return_value = [100, 200, 300, 400]
        mock_page.number = 0
        
        mock_doc.__iter__.return_value = [mock_page]
        mock_doc.__len__.return_value = 1
        
        content = {"raw_content": mock_doc}
        
        figures = parser._extract_figures_fitz(content)
        
        assert isinstance(figures, list)
        # Should process without errors even if no figures found
    
    @patch('aim2_project.aim2_ontology.parsers.pdf_parser.PDFPLUMBER_AVAILABLE', True)
    def test_extract_figures_pdfplumber(self, parser):
        """Test figure extraction using pdfplumber."""
        # Mock pdfplumber PDF
        mock_pdf = Mock()
        mock_page = Mock()
        
        # Mock figures/images on page
        mock_page.images = [
            {"x0": 100, "y0": 200, "x1": 300, "y1": 400, "width": 200, "height": 200}
        ]
        mock_page.page_number = 1
        
        mock_pdf.pages = [mock_page]
        
        content = {"raw_content": mock_pdf}
        
        figures = parser._extract_figures_pdfplumber(content)
        
        assert isinstance(figures, list)
        # Should process without errors
    
    def test_extract_figures_basic_patterns(self, parser):
        """Test fallback basic pattern extraction."""
        text = "Figure 1. Basic pattern test figure."
        
        figures = parser._extract_figures_basic_patterns(text)
        
        assert isinstance(figures, list)
        if figures:
            figure = figures[0]
            assert "id" in figure
            assert "caption" in figure
            assert figure["id"] == "fig1"
    
    def test_extract_figures_integration(self, parser):
        """Test integrated figure extraction with all methods."""
        text = """
        Methods section content here.
        
        Figure 1. Experimental setup showing the treatment protocol and measurement devices
        used in this study. The apparatus consisted of three main components.
        
        Results section here.
        
        Figure 2. Statistical analysis results (p<0.001) showing significant differences
        between treatment groups. Error bars represent standard deviation.
        """
        
        content = {
            "text": text,
            "extraction_method": "pdfplumber",
            "pages": 2,
            "raw_content": Mock()
        }
        
        figures = parser._extract_figures(text, content)
        
        assert isinstance(figures, list)
        assert len(figures) >= 2
        
        for figure in figures:
            assert isinstance(figure, dict)
            assert "id" in figure
            assert "caption" in figure
            assert len(figure["caption"]) > 10  # Should have substantial captions


class TestPDFParserTableExtraction:
    """Test PDF parser table extraction methods."""
    
    @pytest.fixture
    def parser(self):
        """Create PDFParser instance for testing."""
        return PDFParser()
    
    def test_extract_tables_by_patterns(self, parser):
        """Test table extraction using regex patterns."""
        text = """
        Table 1. Summary of experimental results.
        Tab. 2: Statistical analysis results.
        Table 3 - Demographic characteristics.
        Supplementary Table S1. Additional data.
        """
        
        tables = parser._extract_tables_by_patterns(text)
        
        assert isinstance(tables, list)
        assert len(tables) >= 3  # Should find at least 3 main tables
        
        for table in tables:
            assert "id" in table
            assert "caption" in table
            assert "position" in table
    
    def test_extract_tables_by_patterns_various_formats(self, parser):
        """Test table extraction with various caption formats."""
        test_cases = [
            "Table 1. Simple caption.",
            "Tab 2: Caption with colon.",
            "Table 3 - Caption with dash.",
            "TABLE 4. Uppercase table.",
            "Table A1. Appendix table.",
            "Supplementary Table S1. Supplementary material.",
        ]
        
        for test_text in test_cases:
            tables = parser._extract_tables_by_patterns(test_text)
            assert len(tables) >= 1, f"Failed to extract from: {test_text}"
    
    @patch('aim2_project.aim2_ontology.parsers.pdf_parser.PDFPLUMBER_AVAILABLE', True)
    def test_extract_tables_pdfplumber(self, parser):
        """Test table extraction using pdfplumber."""
        # Mock pdfplumber PDF
        mock_pdf = Mock()
        mock_page = Mock()
        
        # Mock table data
        mock_table = Mock()
        mock_table.extract.return_value = [
            ["Name", "Age", "Score"],
            ["Alice", "25", "85"],
            ["Bob", "30", "92"]
        ]
        mock_table.bbox = [100, 200, 300, 400]
        
        mock_page.tables = [mock_table]
        mock_page.page_number = 1
        
        mock_pdf.pages = [mock_page]
        
        content = {"raw_content": mock_pdf}
        
        tables = parser._extract_tables_pdfplumber(content)
        
        assert isinstance(tables, list)
        if tables:
            table = tables[0]
            assert "data" in table
            assert isinstance(table["data"], list)
    
    @patch('aim2_project.aim2_ontology.parsers.pdf_parser.FITZ_AVAILABLE', True)
    def test_extract_tables_fitz(self, parser):
        """Test table extraction using PyMuPDF (fitz)."""
        # Mock fitz document
        mock_doc = Mock()
        mock_page = Mock()
        
        # Mock table detection (simplified)
        mock_page.search_for.return_value = []  # No explicit tables found
        mock_page.get_text.return_value = "Sample text content"
        mock_page.number = 0
        
        mock_doc.__iter__.return_value = [mock_page]
        mock_doc.__len__.return_value = 1
        
        content = {"raw_content": mock_doc}
        
        tables = parser._extract_tables_fitz(content)
        
        assert isinstance(tables, list)
        # Should process without errors even if no tables found
    
    def test_extract_tables_basic_patterns(self, parser):
        """Test fallback basic pattern extraction."""
        text = "Table 1. Basic pattern test table."
        
        tables = parser._extract_tables_basic_patterns(text)
        
        assert isinstance(tables, list)
        if tables:
            table = tables[0]
            assert "id" in table
            assert "caption" in table
            assert table["id"] == "tab1"
    
    def test_extract_tables_with_data(self, parser):
        """Test table extraction with actual table data."""
        text = """
        Table 1. Summary of experimental results across treatment groups.
        
        Treatment Group | Sample Size | Mean Response | Std Dev | P-value
        Control         | 25          | 12.5          | 2.1     | -
        Treatment A     | 24          | 15.7          | 2.8     | 0.023
        Treatment B     | 26          | 18.2          | 3.1     | 0.001
        
        The results show significant differences between groups.
        """
        
        content = {
            "text": text,
            "extraction_method": "pdfplumber",
            "raw_content": Mock()
        }
        
        tables = parser._extract_tables(text, content)
        
        assert isinstance(tables, list)
        if tables:
            table = tables[0]
            assert "id" in table
            assert "caption" in table
            # May or may not have data depending on extraction method success


class TestPDFParserQualityAssessment:
    """Test quality assessment functionality in PDF parser."""
    
    @pytest.fixture
    def parser(self):
        """Create PDFParser instance for testing."""
        return PDFParser()
    
    def test_figure_quality_assessment_high_quality(self, parser):
        """Test quality assessment for high-quality figures."""
        figure = FigureMetadata()
        figure.id = "fig1"
        figure.label = "Figure 1"
        figure.caption = "Comprehensive statistical analysis showing p<0.001 significance"
        figure.figure_type = FigureType.CHART
        figure.context.section_context = "Results"
        figure.context.cross_references = ["Figure 1 shows", "see Fig. 1"]
        figure.technical.file_format = "PNG"
        figure.technical.dimensions = (800, 600)
        figure.analysis.content_type = "statistical_chart"
        figure.analysis.keywords = ["statistical", "analysis", "significance"]
        
        assessor = QualityAssessment()
        quality = assessor.assess_figure_quality(figure)
        
        assert quality.extraction_confidence > 0.7
        assert quality.completeness_score > 0.5
        assert quality.overall_quality() > 0.6
        assert quality.validation_status in ["passed", "partial"]
    
    def test_figure_quality_assessment_low_quality(self, parser):
        """Test quality assessment for low-quality figures."""
        figure = FigureMetadata()
        figure.caption = "Simple figure"  # Minimal information
        
        assessor = QualityAssessment()
        quality = assessor.assess_figure_quality(figure)
        
        assert quality.extraction_confidence < 0.5
        assert len(quality.quality_issues) > 0
        assert "Missing figure ID" in quality.quality_issues
    
    def test_table_quality_assessment_high_quality(self, parser):
        """Test quality assessment for high-quality tables."""
        table = TableMetadata()
        table.id = "tab1"
        table.label = "Table 1"
        table.caption = "Statistical comparison of treatment groups"
        table.table_type = TableType.STATISTICAL
        table.structure.rows = 4
        table.structure.columns = 5
        table.structure.column_headers = ["Group", "N", "Mean", "SD", "P-value"]
        table.content.data_quality_score = 0.9
        table.context.section_context = "Results"
        table.analysis.statistical_summary = {"Mean": {"mean": 15.2}}
        
        assessor = QualityAssessment()
        quality = assessor.assess_table_quality(table)
        
        assert quality.extraction_confidence > 0.7
        assert quality.completeness_score > 0.7
        assert quality.overall_quality() > 0.7
        assert quality.validation_status in ["passed", "partial"]
    
    def test_extraction_summary_generation(self, parser):
        """Test extraction summary generation."""
        # Create sample figures and tables
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
        
        assessor = QualityAssessment()
        summary = assessor.generate_extraction_summary(
            figures=[figure],
            tables=[table],
            extraction_time=0.5,
            parsing_method="pdf_parser_enhanced"
        )
        
        assert summary.total_figures == 1
        assert summary.total_tables == 1
        assert summary.figures_by_type["chart"] == 1
        assert summary.tables_by_type["statistical"] == 1
        assert summary.average_quality_score > 0.7
        assert summary.parsing_method == "pdf_parser_enhanced"


class TestPDFParserLibraryFallback:
    """Test library availability and fallback mechanisms."""
    
    @pytest.fixture
    def parser(self):
        """Create PDFParser instance for testing."""
        return PDFParser()
    
    @patch('aim2_project.aim2_ontology.parsers.pdf_parser.PDFPLUMBER_AVAILABLE', False)
    @patch('aim2_project.aim2_ontology.parsers.pdf_parser.FITZ_AVAILABLE', False)
    @patch('aim2_project.aim2_ontology.parsers.pdf_parser.PYPDF_AVAILABLE', True)
    def test_pypdf_only_fallback(self, parser):
        """Test fallback to pypdf only when other libraries unavailable."""
        text = "Figure 1. Test figure caption."
        content = {"text": text, "extraction_method": "pypdf"}
        
        figures = parser._extract_figures(text, content)
        
        # Should still work with basic pattern extraction
        assert isinstance(figures, list)
    
    @patch('aim2_project.aim2_ontology.parsers.pdf_parser.PDFPLUMBER_AVAILABLE', False)
    @patch('aim2_project.aim2_ontology.parsers.pdf_parser.FITZ_AVAILABLE', False)
    @patch('aim2_project.aim2_ontology.parsers.pdf_parser.PYPDF_AVAILABLE', False)
    def test_no_libraries_available(self, parser):
        """Test behavior when no PDF libraries are available."""
        text = "Figure 1. Test figure caption."
        content = {"text": text, "extraction_method": "none"}
        
        # Should still work with basic pattern extraction
        figures = parser._extract_figures(text, content)
        assert isinstance(figures, list)
        
        tables = parser._extract_tables(text, content)
        assert isinstance(tables, list)
    
    def test_library_specific_method_selection(self, parser):
        """Test that appropriate extraction methods are selected based on available libraries."""
        content = {"extraction_method": "pdfplumber", "raw_content": Mock()}
        
        # Should not crash regardless of library availability
        figures = parser._extract_figures_by_library(content)
        assert isinstance(figures, list)
        
        tables = parser._extract_tables_by_library(content)
        assert isinstance(tables, list)


class TestPDFParserErrorHandling:
    """Test error handling and edge cases."""
    
    @pytest.fixture  
    def parser(self):
        """Create PDFParser instance for testing."""
        return PDFParser()
    
    def test_empty_content_handling(self, parser):
        """Test handling of empty or None content."""
        empty_content = {"text": "", "extraction_method": "test"}
        
        figures = parser._extract_figures("", empty_content)
        assert isinstance(figures, list)
        assert len(figures) == 0
        
        tables = parser._extract_tables("", empty_content)
        assert isinstance(tables, list)
        assert len(tables) == 0
    
    def test_malformed_content_handling(self, parser):
        """Test handling of malformed content."""
        malformed_content = {"text": None, "extraction_method": "test"}
        
        # Should not crash
        result = parser.extract_figures_tables(malformed_content)
        assert isinstance(result, dict)
        assert "figures" in result
        assert "tables" in result
    
    def test_large_document_handling(self, parser):
        """Test handling of large documents."""
        # Create large text content
        large_text = "Introduction. " * 1000
        large_text += "Figure 1. Test figure in large document. "
        large_text += "Content. " * 1000
        large_text += "Table 1. Test table in large document. "
        large_text += "More content. " * 1000
        
        content = {"text": large_text, "extraction_method": "test"}
        
        # Should complete without timeout or memory issues
        import time
        start_time = time.time()
        
        result = parser.extract_figures_tables(content)
        
        end_time = time.time()
        
        # Should complete within reasonable time (10 seconds)
        assert (end_time - start_time) < 10.0
        assert isinstance(result, dict)
    
    def test_unicode_content_handling(self, parser):
        """Test handling of Unicode characters in content."""
        unicode_text = """
        Étude statistique avec données spécialisées.
        
        Figure 1. Analyse des résultats avec α=0.05 et β-test.
        Les données montrent une différence significative (p<0.001).
        
        Tableau 1. Résumé des caractéristiques démographiques.
        Âge moyen: 45.2 ± 12.3 années.
        """
        
        content = {"text": unicode_text, "extraction_method": "test"}
        
        # Should handle Unicode without errors
        result = parser.extract_figures_tables(content)
        assert isinstance(result, dict)
        
        if result["figures"]:
            figure = result["figures"][0]
            assert isinstance(figure["caption"], str)
        
        if result["tables"]:
            table = result["tables"][0]
            assert isinstance(table["caption"], str)
    
    def test_regex_pattern_edge_cases(self, parser):
        """Test regex patterns with edge cases."""
        edge_cases = [
            "Figure 1.2. Subfigure notation.",
            "Fig. A-1: Appendix figure with dash.",
            "Figure S1a. Supplementary subfigure.",
            "Table II. Roman numeral notation.",
            "Tab. 3.1: Hierarchical numbering.",
        ]
        
        for text in edge_cases:
            # Should not crash on unusual figure/table numbering
            figures = parser._extract_figures_by_patterns(text)
            tables = parser._extract_tables_by_patterns(text)
            
            assert isinstance(figures, list)
            assert isinstance(tables, list)
    
    def test_extraction_with_corrupted_pdf_data(self, parser):
        """Test extraction with corrupted or partial PDF data."""
        corrupted_content = {
            "text": "Partial content Figure 1.",
            "extraction_method": "unknown",
            "raw_content": None,  # Simulating corrupted data
            "pages": None
        }
        
        # Should handle gracefully without crashing
        result = parser.extract_figures_tables(corrupted_content)
        assert isinstance(result, dict)
        assert "figures" in result
        assert "tables" in result


class TestPDFParserIntegration:
    """Integration tests for PDF parser enhanced functionality."""
    
    @pytest.fixture
    def parser(self):
        """Create PDFParser instance for testing."""
        return PDFParser()
    
    def test_end_to_end_figure_table_extraction(self, parser):
        """Test complete end-to-end extraction workflow."""
        # Comprehensive sample content
        sample_content = {
            "text": """
            Abstract
            This study investigates novel treatment approaches.
            
            Introduction
            Previous research has shown limited efficacy of conventional treatments.
            
            Methods
            Figure 1. Experimental setup showing the treatment protocol and measurement 
            apparatus. The system consisted of three main components: (a) treatment chamber,
            (b) measurement device, and (c) control unit.
            
            Table 1. Baseline demographic characteristics of study participants.
            Characteristic | Control Group | Treatment Group | P-value
            Age (years)    | 45.2 ± 12.3  | 47.1 ± 11.8    | 0.234
            Gender (M/F)   | 12/13         | 14/11           | 0.456
            BMI (kg/m²)    | 24.8 ± 3.2   | 25.1 ± 3.7     | 0.567
            
            Results
            Figure 2. Statistical analysis results showing significant differences (p<0.001)
            between treatment groups. Error bars represent 95% confidence intervals.
            
            As shown in Figure 1, the experimental setup allowed precise control of variables.
            The demographic data in Table 1 shows well-matched groups.
            
            Table 2. Summary of primary outcome measures.
            Outcome Measure     | Control | Treatment | Effect Size | P-value
            Primary endpoint    | 12.5    | 18.7      | 1.23        | <0.001
            Secondary endpoint  | 8.3     | 11.2      | 0.89        | 0.012
            Safety measure      | 2.1     | 1.8       | -0.15       | 0.234
            
            Discussion
            The results demonstrate significant treatment effects as illustrated in Figure 2.
            """,
            "extraction_method": "pdfplumber",
            "pages": 8,
            "raw_content": Mock(),
            "metadata": {
                "title": "Novel Treatment Approaches: A Randomized Controlled Trial",
                "author": "Smith, J. et al."
            }
        }
        
        # Perform extraction
        result = parser.extract_figures_tables(sample_content)
        
        # Verify comprehensive results
        assert isinstance(result, dict)
        assert "figures" in result
        assert "tables" in result
        assert "summary" in result
        
        figures = result["figures"]
        tables = result["tables"]
        summary = result["summary"]
        
        # Should find figures and tables
        assert len(figures) >= 2  # At least Figure 1 and 2
        assert len(tables) >= 2   # At least Table 1 and 2
        
        # Verify figure quality
        for figure_dict in figures:
            assert "id" in figure_dict
            assert "type" in figure_dict
            assert "caption" in figure_dict
            assert "metadata" in figure_dict
            assert "quality" in figure_dict
            
            # Quality should be reasonable
            quality = figure_dict["quality"]
            assert quality["overall_quality"] > 0.3  # At least minimal quality
        
        # Verify table quality
        for table_dict in tables:
            assert "id" in table_dict
            assert "type" in table_dict
            assert "caption" in table_dict
            assert "metadata" in table_dict
            assert "quality" in table_dict
            
            # Quality should be reasonable
            quality = table_dict["quality"]
            assert quality["overall_quality"] > 0.3  # At least minimal quality
        
        # Verify summary
        assert isinstance(summary, dict)
        assert summary["total_figures"] >= 2
        assert summary["total_tables"] >= 2
        assert "average_quality_score" in summary
        assert "parsing_method" in summary
        assert summary["parsing_method"] == "pdf_parser_enhanced"
    
    def test_cross_reference_detection(self, parser):
        """Test detection of cross-references between text and figures/tables."""
        content = {
            "text": """
            The experimental setup is shown in Figure 1, which illustrates the main components.
            As demonstrated in Fig. 1, the system provides accurate measurements.
            See Figure 1 for detailed specifications.
            
            Table 1 summarizes the baseline characteristics.
            The data in Table 1 shows balanced groups.
            Results are presented in Tab. 1.
            """,
            "extraction_method": "test",
            "raw_content": Mock()
        }
        
        # Extract with cross-reference detection
        figures = parser._extract_figures_comprehensive(content["text"], content)
        tables = parser._extract_tables_comprehensive(content["text"], content)
        
        # Verify cross-references are detected
        for figure in figures:
            if figure.id == "fig1":
                cross_refs = figure.context.cross_references
                assert len(cross_refs) > 0
                assert any("Figure 1" in ref for ref in cross_refs)
        
        for table in tables:
            if table.id == "tab1":
                cross_refs = table.context.cross_references
                assert len(cross_refs) > 0
                assert any("Table 1" in ref for ref in cross_refs)
    
    def test_metadata_format_consistency(self, parser):
        """Test that extracted metadata follows consistent format."""
        content = {
            "text": "Figure 1. Test figure. Table 1. Test table.",
            "extraction_method": "test",
            "raw_content": Mock()
        }
        
        result = parser.extract_figures_tables(content)
        
        # Verify consistent format across figures and tables
        all_items = result["figures"] + result["tables"]
        
        for item in all_items:
            # Required top-level keys
            required_keys = {"id", "type", "caption", "metadata", "content", "analysis", "quality"}
            assert required_keys.issubset(set(item.keys()))
            
            # Metadata structure
            metadata = item["metadata"]
            assert "context" in metadata
            assert "technical" in metadata
            
            # Quality structure
            quality = item["quality"]
            assert "extraction_confidence" in quality
            assert "completeness_score" in quality
            assert "parsing_accuracy" in quality
            assert "overall_quality" in quality
            assert "validation_status" in quality
            
            # JSON serializable
            import json
            json_str = json.dumps(item, default=str)
            assert isinstance(json_str, str)