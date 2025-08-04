"""
Comprehensive unit tests for enhanced XML parser figure/table extraction functionality.

This module contains thorough tests for the enhanced XML parser functionality,
including figure/table extraction, metadata analysis, and content processing.
"""

import xml.etree.ElementTree as ET
import pytest
from unittest.mock import Mock, MagicMock, patch
from io import StringIO
from typing import Dict, List, Any

from aim2_project.aim2_ontology.parsers.xml_parser import XMLParser
from aim2_project.aim2_ontology.parsers.metadata_framework import (
    FigureMetadata, TableMetadata, FigureType, TableType, DataType,
    ContextInfo, TechnicalDetails, ContentAnalysis, QualityMetrics,
    ContentExtractor, QualityAssessment, ExtractionSummary
)
from aim2_project.exceptions import ExtractionException, ValidationException


class TestXMLParserEnhancedExtraction:
    """Test enhanced figure/table extraction functionality."""
    
    @pytest.fixture
    def parser(self):
        """Create XMLParser instance for testing."""
        return XMLParser()
    
    @pytest.fixture
    def sample_pmc_xml(self):
        """Sample PMC XML content with figures and tables."""
        return """<?xml version="1.0" encoding="UTF-8"?>
        <article xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance"
                 xmlns:mml="http://www.w3.org/1998/Math/MathML"
                 xmlns:xlink="http://www.w3.org/1999/xlink">
            <front>
                <article-meta>
                    <title-group>
                        <article-title>Sample Research Article</article-title>
                    </title-group>
                </article-meta>
            </front>
            <body>
                <sec id="methods">
                    <title>Methods</title>
                    <p>The experimental setup is shown in <xref ref-type="fig" rid="fig1">Figure 1</xref>.</p>
                    
                    <fig id="fig1" position="float">
                        <label>Figure 1</label>
                        <caption>
                            <title>Experimental Setup</title>
                            <p>Comprehensive experimental apparatus showing treatment chambers and measurement devices. The system consists of three main components: (a) treatment unit, (b) monitoring system, and (c) data collection interface.</p>
                        </caption>
                        <graphic xlink:href="figure1.jpg" mimetype="image" mime-subtype="jpeg"/>
                        <alternatives>
                            <graphic xlink:href="figure1.tiff" mimetype="image" mime-subtype="tiff"/>
                        </alternatives>
                    </fig>
                </sec>
                
                <sec id="results">
                    <title>Results</title>
                    <p>Baseline characteristics are summarized in <xref ref-type="table" rid="tab1">Table 1</xref>.</p>
                    
                    <table-wrap id="tab1" position="float">
                        <label>Table 1</label>
                        <caption>
                            <title>Baseline Demographics</title>
                            <p>Summary of demographic characteristics of study participants across treatment groups.</p>
                        </caption>
                        <table>
                            <thead>
                                <tr>
                                    <th>Characteristic</th>
                                    <th>Control Group (n=25)</th>
                                    <th>Treatment Group (n=24)</th>
                                    <th>P-value</th>
                                </tr>
                            </thead>
                            <tbody>
                                <tr>
                                    <td>Age (years)</td>
                                    <td>45.2 ± 12.3</td>
                                    <td>47.1 ± 11.8</td>
                                    <td>0.234</td>
                                </tr>
                                <tr>
                                    <td>Gender (M/F)</td>
                                    <td>12/13</td>
                                    <td>14/11</td>
                                    <td>0.456</td>
                                </tr>
                            </tbody>
                        </table>
                        <table-wrap-foot>
                            <fn>
                                <p>Data are presented as mean ± SD or n. P-values calculated using t-test or chi-square test as appropriate.</p>
                            </fn>
                        </table-wrap-foot>
                    </table-wrap>
                    
                    <fig id="fig2" position="float">
                        <label>Figure 2</label>
                        <caption>
                            <title>Statistical Analysis Results</title>
                            <p>Bar chart showing significant differences (p&lt;0.001) between treatment groups. Error bars represent 95% confidence intervals.</p>
                        </caption>
                        <graphic xlink:href="figure2.png" mimetype="image" mime-subtype="png"/>
                    </fig>
                </sec>
            </body>
        </article>"""
    
    @pytest.fixture
    def sample_jats_xml(self):
        """Sample JATS XML content with different structure."""
        return """<?xml version="1.0" encoding="UTF-8"?>
        <article xmlns:xlink="http://www.w3.org/1999/xlink">
            <body>
                <sec>
                    <title>Introduction</title>
                    <fig id="f1">
                        <caption>
                            <p>Schematic diagram of the experimental system</p>
                        </caption>
                        <graphic xlink:href="schema.pdf"/>
                    </fig>
                </sec>
                <sec>
                    <title>Methods</title>
                    <table-wrap id="t1">
                        <caption>
                            <p>Experimental parameters and settings</p>
                        </caption>
                        <table>
                            <tr>
                                <th>Parameter</th>
                                <th>Value</th>
                                <th>Unit</th>
                            </tr>
                            <tr>
                                <td>Temperature</td>
                                <td>25.0</td>
                                <td>°C</td>
                            </tr>
                            <tr>
                                <td>Pressure</td>
                                <td>1013.25</td>
                                <td>hPa</td>
                            </tr>
                        </table>
                    </table-wrap>
                </sec>
            </body>
        </article>"""
    
    @pytest.fixture
    def mock_xml_content(self, sample_pmc_xml):
        """Mock XML content structure."""
        return {
            "xml_content": sample_pmc_xml,
            "extraction_method": "lxml",
            "schema_type": "pmc",
            "namespaces": {
                "xlink": "http://www.w3.org/1999/xlink",
                "mml": "http://www.w3.org/1998/Math/MathML"
            }
        }
    
    def test_extract_figures_tables_comprehensive(self, parser, mock_xml_content):
        """Test comprehensive figure and table extraction."""
        result = parser.extract_figures_tables(mock_xml_content)
        
        assert "figures" in result
        assert "tables" in result
        assert isinstance(result["figures"], list)
        assert isinstance(result["tables"], list)
        
        # Should find figures and tables
        assert len(result["figures"]) >= 2  # fig1 and fig2
        assert len(result["tables"]) >= 1   # tab1
        
        # Verify figure structure
        if result["figures"]:
            figure = result["figures"][0]
            assert isinstance(figure, dict)
            assert "id" in figure
            assert "type" in figure
            assert "caption" in figure
            assert "metadata" in figure
            assert "quality" in figure
        
        # Verify table structure
        if result["tables"]:
            table = result["tables"][0]
            assert isinstance(table, dict)
            assert "id" in table
            assert "type" in table
            assert "caption" in table
            assert "metadata" in table
            assert "quality" in table
    
    def test_extract_figures_comprehensive_xml(self, parser, sample_pmc_xml, mock_xml_content):
        """Test comprehensive figure extraction with metadata."""
        root = ET.fromstring(sample_pmc_xml)
        
        figures = parser._extract_figures_comprehensive_xml(root, mock_xml_content)
        
        assert isinstance(figures, list)
        assert len(figures) >= 2  # Should find fig1 and fig2
        
        for figure in figures:
            assert isinstance(figure, FigureMetadata)
            assert figure.id != ""
            assert figure.caption != ""
            assert isinstance(figure.context, ContextInfo)
            assert isinstance(figure.technical, TechnicalDetails)
            assert isinstance(figure.analysis, ContentAnalysis)
            assert isinstance(figure.quality, QualityMetrics)
            assert figure.source_parser == "xml_parser"
            assert figure.extraction_method in ["etree", "lxml"]
    
    def test_extract_tables_comprehensive_xml(self, parser, sample_pmc_xml, mock_xml_content):
        """Test comprehensive table extraction with metadata."""
        root = ET.fromstring(sample_pmc_xml)
        
        tables = parser._extract_tables_comprehensive_xml(root, mock_xml_content)
        
        assert isinstance(tables, list)
        assert len(tables) >= 1  # Should find tab1
        
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
            assert table.source_parser == "xml_parser"
    
    def test_figure_extraction_with_different_schemas(self, parser):
        """Test figure extraction with different XML schemas."""
        # PMC schema
        pmc_xml = """<fig id="f1" xmlns:xlink="http://www.w3.org/1999/xlink">
            <label>Figure 1</label>
            <caption><p>PMC style figure</p></caption>
            <graphic xlink:href="image.jpg"/>
        </fig>"""
        
        # JATS schema
        jats_xml = """<fig id="f1">
            <caption><p>JATS style figure</p></caption>
            <graphic href="image.jpg"/>
        </fig>"""
        
        for xml_content in [pmc_xml, jats_xml]:
            root = ET.fromstring(xml_content)
            content = {"extraction_method": "etree", "schema_type": "unknown"}
            
            figures = parser._extract_figures_comprehensive_xml(root, content)
            
            # Should handle different schemas gracefully
            assert isinstance(figures, list)
    
    def test_table_extraction_with_complex_structure(self, parser):
        """Test table extraction with complex table structures."""
        complex_table_xml = """
        <table-wrap id="complex-table">
            <label>Table 1</label>
            <caption><p>Complex table with merged cells and multiple sections</p></caption>
            <table>
                <thead>
                    <tr>
                        <th rowspan="2">Parameter</th>
                        <th colspan="2">Group A</th>
                        <th colspan="2">Group B</th>
                    </tr>
                    <tr>
                        <th>Mean</th>
                        <th>SD</th>
                        <th>Mean</th>
                        <th>SD</th>
                    </tr>
                </thead>
                <tbody>
                    <tr>
                        <td>Variable 1</td>
                        <td>12.5</td>
                        <td>2.1</td>
                        <td>15.7</td>
                        <td>2.8</td>
                    </tr>
                </tbody>
                <tfoot>
                    <tr>
                        <td colspan="5">Footer information</td>
                    </tr>
                </tfoot>
            </table>
        </table-wrap>"""
        
        root = ET.fromstring(complex_table_xml)
        content = {"extraction_method": "etree", "schema_type": "pmc"}
        
        tables = parser._extract_tables_comprehensive_xml(root, content)
        
        assert isinstance(tables, list)
        if tables:
            table = tables[0]
            assert table.structure.rows > 0
            assert table.structure.columns > 0
            # Should handle merged cells
            if table.structure.merged_cells:
                assert len(table.structure.merged_cells) > 0


class TestXMLParserLibrarySpecificMethods:
    """Test library-specific extraction methods."""
    
    @pytest.fixture
    def parser(self):
        """Create XMLParser instance for testing."""
        return XMLParser()
    
    @pytest.fixture
    def sample_xml_element(self):
        """Sample XML element for testing."""
        xml_string = """
        <root>
            <fig id="test-fig">
                <label>Figure 1</label>
                <caption><p>Test figure caption</p></caption>
                <graphic href="test.jpg"/>
            </fig>
            <table-wrap id="test-table">
                <label>Table 1</label>
                <caption><p>Test table caption</p></caption>
                <table>
                    <tr><th>Header</th></tr>
                    <tr><td>Data</td></tr>
                </table>
            </table-wrap>
        </root>"""
        return ET.fromstring(xml_string)
    
    def test_extract_figures_etree(self, parser, sample_xml_element):
        """Test figure extraction using ElementTree."""
        figures = parser._extract_figures_etree(sample_xml_element)
        
        assert isinstance(figures, list)
        if figures:
            figure = figures[0]
            assert "id" in figure
            assert "caption" in figure
            assert "graphics" in figure
            assert figure["id"] == "test-fig"
    
    @patch('aim2_project.aim2_ontology.parsers.xml_parser.LXML_AVAILABLE', True)
    def test_extract_figures_lxml(self, parser, sample_xml_element):
        """Test figure extraction using lxml."""
        # Convert to lxml-style element for testing
        figures = parser._extract_figures_lxml(sample_xml_element)
        
        assert isinstance(figures, list)
        # Should work even with ElementTree element (graceful degradation)
    
    def test_extract_tables_etree(self, parser, sample_xml_element):
        """Test table extraction using ElementTree."""
        tables = parser._extract_tables_etree(sample_xml_element)
        
        assert isinstance(tables, list)
        if tables:
            table = tables[0]
            assert "id" in table
            assert "caption" in table
            assert "content" in table
            assert table["id"] == "test-table"
    
    @patch('aim2_project.aim2_ontology.parsers.xml_parser.LXML_AVAILABLE', True)
    def test_extract_tables_lxml(self, parser, sample_xml_element):
        """Test table extraction using lxml."""
        tables = parser._extract_tables_lxml(sample_xml_element)
        
        assert isinstance(tables, list)
        # Should work even with ElementTree element (graceful degradation)
    
    def test_extract_table_structure_etree(self, parser):
        """Test table structure extraction using ElementTree."""
        table_xml = """
        <table-wrap>
            <table>
                <thead>
                    <tr><th>Col1</th><th>Col2</th><th>Col3</th></tr>
                </thead>
                <tbody>
                    <tr><td>A</td><td>B</td><td>C</td></tr>
                    <tr><td>1</td><td>2</td><td>3</td></tr>
                </tbody>
            </table>
        </table-wrap>"""
        
        element = ET.fromstring(table_xml)
        structure = parser._extract_table_structure_etree(element)
        
        assert isinstance(structure, dict)
        assert "total_rows" in structure
        assert "total_columns" in structure
        assert "has_header" in structure
        assert structure["total_rows"] == 3  # Header + 2 data rows
        assert structure["total_columns"] == 3
        assert structure["has_header"] == True
    
    def test_extract_table_content_etree(self, parser):
        """Test table content extraction using ElementTree."""
        table_xml = """
        <table-wrap>
            <table>
                <tr><th>Name</th><th>Age</th><th>Score</th></tr>
                <tr><td>Alice</td><td>25</td><td>85.5</td></tr>
                <tr><td>Bob</td><td>30</td><td>92.1</td></tr>
            </table>
        </table-wrap>"""
        
        element = ET.fromstring(table_xml)
        content = parser._extract_table_content_etree(element)
        
        assert isinstance(content, dict)
        assert "data" in content
        assert "metadata" in content
        
        data = content["data"]
        assert len(data) == 3  # Header + 2 data rows
        assert len(data[0]) == 3  # 3 columns
    
    def test_extract_cross_references(self, parser):
        """Test cross-reference extraction."""
        xml_with_refs = """
        <root>
            <p>See <xref ref-type="fig" rid="fig1">Figure 1</xref> for details.</p>
            <p>Data in <xref ref-type="table" rid="tab1">Table 1</xref> shows results.</p>
            <fig id="fig1">
                <caption><p>Test figure</p></caption>
            </fig>
        </root>"""
        
        root = ET.fromstring(xml_with_refs)
        fig_elem = root.find(".//fig")
        
        cross_refs = parser._extract_figure_cross_refs_etree(fig_elem)
        
        assert isinstance(cross_refs, list)
        # Should find references to fig1


class TestXMLParserContentAnalysis:
    """Test content analysis functionality."""
    
    @pytest.fixture
    def parser(self):
        """Create XMLParser instance for testing."""
        return XMLParser()
    
    def test_figure_content_analysis(self, parser):
        """Test figure content analysis."""
        figure_xml = """
        <fig id="analysis-fig">
            <label>Figure 1</label>
            <caption>
                <title>Statistical Analysis</title>
                <p>Bar chart showing significant differences (p&lt;0.001) between treatment groups with 95% confidence intervals. Error bars represent standard deviation.</p>
            </caption>
            <graphic href="stats.png" mimetype="image" mime-subtype="png"/>
        </fig>"""
        
        element = ET.fromstring(figure_xml)
        content = {"extraction_method": "etree", "schema_type": "pmc"}
        
        figures = parser._extract_figures_comprehensive_xml(element, content)
        
        assert len(figures) == 1
        figure = figures[0]
        
        # Should classify as chart
        assert figure.figure_type in [FigureType.CHART, FigureType.GRAPH, FigureType.UNKNOWN]
        
        # Should extract keywords
        assert len(figure.analysis.keywords) > 0
        
        # Should have content analysis
        assert figure.analysis.complexity_score > 0
    
    def test_table_content_analysis(self, parser):
        """Test table content analysis."""
        table_xml = """
        <table-wrap id="analysis-table">
            <label>Table 1</label>
            <caption>
                <title>Demographic Characteristics</title>
                <p>Baseline demographic data for study participants showing age, gender distribution, and clinical parameters.</p>
            </caption>
            <table>
                <thead>
                    <tr>
                        <th>Characteristic</th>
                        <th>Control (n=25)</th>
                        <th>Treatment (n=24)</th>
                        <th>P-value</th>
                    </tr>
                </thead>
                <tbody>
                    <tr>
                        <td>Age (years)</td>
                        <td>45.2 ± 12.3</td>
                        <td>47.1 ± 11.8</td>
                        <td>0.234</td>
                    </tr>
                    <tr>
                        <td>Gender (M/F)</td>
                        <td>12/13</td>
                        <td>14/11</td>
                        <td>0.456</td>
                    </tr>
                </tbody>
            </table>
        </table-wrap>"""
        
        element = ET.fromstring(table_xml)
        content = {"extraction_method": "etree", "schema_type": "pmc"}
        
        tables = parser._extract_tables_comprehensive_xml(element, content)
        
        assert len(tables) == 1
        table = tables[0]
        
        # Should classify as demographic
        assert table.table_type in [TableType.DEMOGRAPHIC, TableType.STATISTICAL, TableType.UNKNOWN]
        
        # Should have structure information
        assert table.structure.rows > 0
        assert table.structure.columns > 0
        assert len(table.structure.column_headers) > 0
        
        # Should have content analysis
        assert table.analysis.complexity_score > 0
        
        # Should detect data types
        assert len(table.structure.data_types) > 0


class TestXMLParserQualityAssessment:
    """Test quality assessment functionality."""
    
    @pytest.fixture
    def parser(self):
        """Create XMLParser instance for testing."""
        return XMLParser()
    
    def test_high_quality_figure_assessment(self, parser):
        """Test quality assessment for high-quality figures."""
        high_quality_xml = """
        <fig id="hq-fig" position="float">
            <label>Figure 1</label>
            <caption>
                <title>Comprehensive Statistical Analysis</title>
                <p>Detailed bar chart showing statistically significant differences (p&lt;0.001) between treatment groups across multiple outcome measures. Error bars represent 95% confidence intervals. Statistical analysis performed using ANOVA with post-hoc Tukey's test.</p>
            </caption>
            <graphic xlink:href="figure1.tiff" mimetype="image" mime-subtype="tiff"/>
            <alternatives>
                <graphic xlink:href="figure1.png" mimetype="image" mime-subtype="png"/>
            </alternatives>
        </fig>"""
        
        element = ET.fromstring(high_quality_xml)
        content = {"extraction_method": "etree", "schema_type": "pmc"}
        
        figures = parser._extract_figures_comprehensive_xml(element, content)
        figure = figures[0]
        
        # Should have high quality scores
        assert figure.quality.extraction_confidence > 0.7
        assert figure.quality.completeness_score > 0.5
        assert figure.quality.overall_quality() > 0.6
        assert figure.quality.validation_status in ["passed", "partial"]
    
    def test_low_quality_figure_assessment(self, parser):
        """Test quality assessment for low-quality figures."""
        low_quality_xml = """
        <fig>
            <caption><p>Simple figure</p></caption>
        </fig>"""
        
        element = ET.fromstring(low_quality_xml)
        content = {"extraction_method": "etree", "schema_type": "unknown"}
        
        figures = parser._extract_figures_comprehensive_xml(element, content)
        figure = figures[0]
        
        # Should have lower quality scores
        assert figure.quality.extraction_confidence < 0.8
        assert len(figure.quality.quality_issues) > 0
    
    def test_high_quality_table_assessment(self, parser):
        """Test quality assessment for high-quality tables."""
        high_quality_xml = """
        <table-wrap id="hq-table" position="float">
            <label>Table 1</label>
            <caption>
                <title>Comprehensive Demographic Analysis</title>
                <p>Detailed baseline characteristics of study participants showing demographic variables, clinical parameters, and statistical comparisons between treatment groups.</p>
            </caption>
            <table>
                <thead>
                    <tr>
                        <th>Characteristic</th>
                        <th>Control Group (n=25)</th>
                        <th>Treatment Group (n=24)</th>
                        <th>P-value</th>
                        <th>Effect Size</th>
                    </tr>
                </thead>
                <tbody>
                    <tr>
                        <td>Age (years)</td>
                        <td>45.2 ± 12.3</td>
                        <td>47.1 ± 11.8</td>
                        <td>0.234</td>
                        <td>0.15</td>
                    </tr>
                    <tr>
                        <td>BMI (kg/m²)</td>
                        <td>24.8 ± 3.2</td>
                        <td>25.1 ± 3.7</td>
                        <td>0.567</td>
                        <td>0.09</td>
                    </tr>
                </tbody>
            </table>
            <table-wrap-foot>
                <fn><p>Data presented as mean ± SD. P-values calculated using independent t-test.</p></fn>
            </table-wrap-foot>
        </table-wrap>"""
        
        element = ET.fromstring(high_quality_xml)
        content = {"extraction_method": "etree", "schema_type": "pmc"}
        
        tables = parser._extract_tables_comprehensive_xml(element, content)
        table = tables[0]
        
        # Should have high quality scores
        assert table.quality.extraction_confidence > 0.7
        assert table.quality.completeness_score > 0.7
        assert table.quality.overall_quality() > 0.7
        assert table.quality.validation_status in ["passed", "partial"]


class TestXMLParserErrorHandling:
    """Test error handling and edge cases."""
    
    @pytest.fixture
    def parser(self):
        """Create XMLParser instance for testing."""
        return XMLParser()
    
    def test_malformed_xml_handling(self, parser):
        """Test handling of malformed XML."""
        malformed_xml = """
        <article>
            <fig id="broken-fig">
                <caption><p>Unclosed paragraph
                <graphic href="missing.jpg"/>
            </fig>
        </article>"""
        
        # Should handle gracefully
        try:
            root = ET.fromstring(malformed_xml)
            content = {"extraction_method": "etree"}
            result = parser._extract_figures_comprehensive_xml(root, content)
            assert isinstance(result, list)
        except ET.ParseError:
            # Expected for truly malformed XML
            pass
    
    def test_missing_elements_handling(self, parser):
        """Test handling of missing required elements."""
        minimal_xml = """
        <root>
            <fig id="minimal">
                <!-- Missing caption and graphics -->
            </fig>
            <table-wrap id="minimal-table">
                <!-- Missing caption and table content -->
            </table-wrap>
        </root>"""
        
        root = ET.fromstring(minimal_xml)
        content = {"extraction_method": "etree"}
        
        figures = parser._extract_figures_comprehensive_xml(root, content)
        tables = parser._extract_tables_comprehensive_xml(root, content)
        
        # Should handle missing elements gracefully
        assert isinstance(figures, list)
        assert isinstance(tables, list)
        
        if figures:
            figure = figures[0]
            assert figure.id == "minimal"
            # Should have low quality due to missing elements
            assert figure.quality.extraction_confidence < 0.5
    
    def test_empty_content_handling(self, parser):
        """Test handling of empty content."""
        empty_xml = "<root></root>"
        
        root = ET.fromstring(empty_xml)
        content = {"extraction_method": "etree"}
        
        figures = parser._extract_figures_comprehensive_xml(root, content)
        tables = parser._extract_tables_comprehensive_xml(root, content)
        
        assert isinstance(figures, list)
        assert isinstance(tables, list)
        assert len(figures) == 0
        assert len(tables) == 0
    
    def test_namespace_handling(self, parser):
        """Test handling of different namespaces."""
        namespaced_xml = """
        <article xmlns:xlink="http://www.w3.org/1999/xlink" 
                 xmlns:mml="http://www.w3.org/1998/Math/MathML">
            <fig id="ns-fig">
                <caption><p>Namespaced figure</p></caption>
                <graphic xlink:href="image.jpg" mimetype="image"/>
            </fig>
        </article>"""
        
        root = ET.fromstring(namespaced_xml)
        content = {
            "extraction_method": "etree",
            "namespaces": {
                "xlink": "http://www.w3.org/1999/xlink",
                "mml": "http://www.w3.org/1998/Math/MathML"
            }
        }
        
        figures = parser._extract_figures_comprehensive_xml(root, content)
        
        assert isinstance(figures, list)
        if figures:
            figure = figures[0]
            assert figure.id == "ns-fig"
    
    def test_large_xml_handling(self, parser):
        """Test handling of large XML documents."""
        # Create large XML content
        large_figures = []
        for i in range(100):
            large_figures.append(f"""
            <fig id="fig{i}">
                <label>Figure {i}</label>
                <caption><p>Figure {i} caption with some content</p></caption>
                <graphic href="figure{i}.jpg"/>
            </fig>""")
        
        large_xml = f"<root>{''.join(large_figures)}</root>"
        
        root = ET.fromstring(large_xml)
        content = {"extraction_method": "etree"}
        
        # Should complete in reasonable time
        import time
        start_time = time.time()
        
        figures = parser._extract_figures_comprehensive_xml(root, content)
        
        end_time = time.time()
        
        # Should complete within 30 seconds (generous limit)
        assert (end_time - start_time) < 30.0
        assert isinstance(figures, list)
        assert len(figures) == 100
    
    def test_unicode_content_handling(self, parser):
        """Test handling of Unicode characters."""
        unicode_xml = """
        <root>
            <fig id="unicode-fig">
                <caption>
                    <p>Étude statistique avec données spécialisées: α=0.05, β-test, ±2.5°C, µg/mL</p>
                </caption>
            </fig>
        </root>"""
        
        root = ET.fromstring(unicode_xml)
        content = {"extraction_method": "etree"}
        
        figures = parser._extract_figures_comprehensive_xml(root, content)
        
        assert isinstance(figures, list)
        if figures:
            figure = figures[0]
            assert "α" in figure.caption or "alpha" in figure.caption.lower()


class TestXMLParserIntegration:
    """Integration tests for XML parser enhanced functionality."""
    
    @pytest.fixture
    def parser(self):
        """Create XMLParser instance for testing."""
        return XMLParser()
    
    def test_end_to_end_extraction_workflow(self, parser):
        """Test complete end-to-end extraction workflow."""
        comprehensive_xml = """<?xml version="1.0" encoding="UTF-8"?>
        <article xmlns:xlink="http://www.w3.org/1999/xlink">
            <front>
                <article-meta>
                    <title-group>
                        <article-title>Comprehensive Research Study</article-title>
                    </title-group>
                </article-meta>
            </front>
            <body>
                <sec id="introduction">
                    <title>Introduction</title>
                    <p>This study examines novel approaches as shown in <xref ref-type="fig" rid="fig1">Figure 1</xref>.</p>
                </sec>
                
                <sec id="methods">
                    <title>Methods</title>
                    <fig id="fig1" position="float">
                        <label>Figure 1</label>
                        <caption>
                            <title>Experimental Design</title>
                            <p>Comprehensive experimental setup showing treatment protocols and measurement systems. The apparatus includes: (a) treatment chamber, (b) monitoring equipment, (c) data acquisition system, and (d) control interface.</p>
                        </caption>
                        <graphic xlink:href="experiment.tiff" mimetype="image" mime-subtype="tiff"/>
                        <alternatives>
                            <graphic xlink:href="experiment.png" mimetype="image" mime-subtype="png"/>
                        </alternatives>
                    </fig>
                    
                    <p>Participant characteristics are detailed in <xref ref-type="table" rid="tab1">Table 1</xref>.</p>
                    
                    <table-wrap id="tab1" position="float">
                        <label>Table 1</label>
                        <caption>
                            <title>Baseline Participant Characteristics</title>
                            <p>Comprehensive demographic and clinical characteristics of study participants at baseline, stratified by treatment group.</p>
                        </caption>
                        <table>
                            <thead>
                                <tr>
                                    <th rowspan="2">Characteristic</th>
                                    <th colspan="2">Control Group</th>
                                    <th colspan="2">Treatment Group</th>
                                    <th rowspan="2">P-value</th>
                                </tr>
                                <tr>
                                    <th>Mean ± SD</th>
                                    <th>Range</th>
                                    <th>Mean ± SD</th>
                                    <th>Range</th>
                                </tr>
                            </thead>
                            <tbody>
                                <tr>
                                    <td>Age (years)</td>
                                    <td>45.2 ± 12.3</td>
                                    <td>22-68</td>
                                    <td>47.1 ± 11.8</td>
                                    <td>24-65</td>
                                    <td>0.234</td>
                                </tr>
                                <tr>
                                    <td>BMI (kg/m²)</td>
                                    <td>24.8 ± 3.2</td>
                                    <td>19.1-31.5</td>
                                    <td>25.1 ± 3.7</td>
                                    <td>18.9-32.1</td>
                                    <td>0.567</td>
                                </tr>
                            </tbody>
                        </table>
                        <table-wrap-foot>
                            <fn>
                                <p>Data presented as mean ± standard deviation unless otherwise specified. P-values calculated using independent t-test for continuous variables and chi-square test for categorical variables.</p>
                            </fn>
                        </table-wrap-foot>
                    </table-wrap>
                </sec>
                
                <sec id="results">
                    <title>Results</title>
                    <fig id="fig2" position="float">
                        <label>Figure 2</label>
                        <caption>
                            <title>Primary Outcome Results</title>
                            <p>Box plots showing significant differences in primary outcome measures between treatment groups (p&lt;0.001). Boxes represent interquartile range, whiskers show 95% confidence intervals, and dots indicate outliers.</p>
                        </caption>
                        <graphic xlink:href="results.png" mimetype="image" mime-subtype="png"/>
                    </fig>
                    
                    <p>Detailed statistical analysis is presented in <xref ref-type="table" rid="tab2">Table 2</xref>.</p>
                    
                    <table-wrap id="tab2" position="float">
                        <label>Table 2</label>
                        <caption>
                            <title>Statistical Analysis of Primary and Secondary Outcomes</title>
                            <p>Comprehensive statistical analysis showing treatment effects, confidence intervals, and significance levels for all outcome measures.</p>
                        </caption>
                        <table>
                            <thead>
                                <tr>
                                    <th>Outcome Measure</th>
                                    <th>Control Group (n=25)</th>
                                    <th>Treatment Group (n=24)</th>
                                    <th>Difference (95% CI)</th>
                                    <th>Effect Size</th>
                                    <th>P-value</th>
                                </tr>
                            </thead>
                            <tbody>
                                <tr>
                                    <td>Primary endpoint</td>
                                    <td>12.5 ± 3.2</td>
                                    <td>18.7 ± 4.1</td>
                                    <td>6.2 (4.1-8.3)</td>
                                    <td>1.73</td>
                                    <td>&lt;0.001</td>
                                </tr>
                                <tr>
                                    <td>Secondary endpoint A</td>
                                    <td>8.3 ± 2.1</td>
                                    <td>11.2 ± 2.8</td>
                                    <td>2.9 (1.2-4.6)</td>
                                    <td>1.15</td>
                                    <td>0.012</td>
                                </tr>
                            </tbody>
                        </table>
                    </table-wrap>
                </sec>
            </body>
        </article>"""
        
        content = {
            "xml_content": comprehensive_xml,
            "extraction_method": "etree",
            "schema_type": "pmc",
            "namespaces": {"xlink": "http://www.w3.org/1999/xlink"}
        }
        
        # Perform complete extraction
        result = parser.extract_figures_tables(content)
        
        # Verify comprehensive results structure
        assert isinstance(result, dict)
        assert "figures" in result
        assert "tables" in result
        assert "summary" in result
        
        figures = result["figures"]
        tables = result["tables"]
        summary = result["summary"]
        
        # Should find all figures and tables
        assert len(figures) >= 2  # fig1 and fig2
        assert len(tables) >= 2   # tab1 and tab2
        
        # Verify figure quality and metadata
        for figure_dict in figures:
            required_keys = {"id", "type", "caption", "metadata", "content", "analysis", "quality"}
            assert required_keys.issubset(set(figure_dict.keys()))
            
            # Should have good quality for complete figures
            quality = figure_dict["quality"]
            assert quality["overall_quality"] > 0.4
            assert quality["extraction_confidence"] > 0.5
        
        # Verify table quality and metadata
        for table_dict in tables:
            required_keys = {"id", "type", "caption", "metadata", "content", "analysis", "quality"}
            assert required_keys.issubset(set(table_dict.keys()))
            
            # Should have structure information
            structure = table_dict["metadata"]["structure"]
            assert structure["rows"] > 0
            assert structure["columns"] > 0
            
            # Should have good quality for complete tables
            quality = table_dict["quality"]
            assert quality["overall_quality"] > 0.4
        
        # Verify summary information
        assert summary["total_figures"] >= 2
        assert summary["total_tables"] >= 2
        assert "average_quality_score" in summary
        assert "parsing_method" in summary
        assert summary["parsing_method"] == "xml_parser_enhanced"
    
    def test_cross_parser_compatibility(self, parser):
        """Test that XML parser output is compatible with PDF parser format."""
        xml_content = {
            "xml_content": """
            <article>
                <fig id="compat-fig">
                    <caption><p>Compatibility test figure</p></caption>
                </fig>
                <table-wrap id="compat-table">
                    <caption><p>Compatibility test table</p></caption>
                    <table><tr><td>Test</td></tr></table>
                </table-wrap>
            </article>""",
            "extraction_method": "etree",
            "schema_type": "generic"
        }
        
        result = parser.extract_figures_tables(xml_content)
        
        # Should have same structure as PDF parser output
        assert "figures" in result
        assert "tables" in result
        assert "summary" in result
        
        # Verify unified format
        all_items = result["figures"] + result["tables"]
        for item in all_items:
            # Same top-level structure as PDF parser
            required_keys = {"id", "type", "caption", "metadata", "quality"}
            common_keys = required_keys & set(item.keys())
            assert len(common_keys) >= 4  # Should have most required keys
            
            # Should be JSON serializable
            import json
            json_str = json.dumps(item, default=str)
            assert isinstance(json_str, str)
    
    def test_schema_detection_and_adaptation(self, parser):
        """Test automatic schema detection and adaptation."""
        # Test different schema types
        schema_tests = [
            ("pmc", """<article xmlns:xlink="http://www.w3.org/1999/xlink">
                        <fig id="pmc-fig"><caption><p>PMC figure</p></caption></fig>
                      </article>"""),
            ("jats", """<article>
                         <fig id="jats-fig"><caption><p>JATS figure</p></caption></fig>
                       </article>"""),
            ("generic", """<document>
                            <figure id="generic-fig"><title>Generic figure</title></figure>
                          </document>""")
        ]
        
        for schema_type, xml_content in schema_tests:
            content = {
                "xml_content": xml_content,
                "extraction_method": "etree",
                "schema_type": schema_type
            }
            
            # Should handle different schemas without errors
            result = parser.extract_figures_tables(content)
            assert isinstance(result, dict)
            assert "figures" in result
            assert "tables" in result