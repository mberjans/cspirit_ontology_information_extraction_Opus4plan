"""
Integration tests for enhanced figure and table extraction functionality.

This module contains end-to-end integration tests that verify the complete
figure/table extraction workflow across both PDF and XML parsers, ensuring
consistent output formats, quality assessment, and cross-parser compatibility.
"""

import json
from unittest.mock import Mock

import pytest

from aim2_project.aim2_ontology.parsers.pdf_parser import PDFParser
from aim2_project.aim2_ontology.parsers.xml_parser import XMLParser


class TestCrossParserCompatibility:
    """Test compatibility and consistency between PDF and XML parsers."""

    @pytest.fixture
    def pdf_parser(self):
        """Create PDF parser instance."""
        return PDFParser()

    @pytest.fixture
    def xml_parser(self):
        """Create XML parser instance."""
        return XMLParser()

    @pytest.fixture
    def sample_pdf_content(self):
        """Sample PDF content with figures and tables."""
        # Create mock page objects
        mock_pages = []
        for i in range(5):
            mock_page = Mock()
            mock_page.extract_text.return_value = f"Page {i+1} content"
            mock_pages.append(mock_page)

        return {
            "text": """
            Introduction
            This study examines novel treatment approaches.

            Methods
            Figure 1. Experimental setup showing treatment chambers and measurement devices.
            The system consists of three main components for accurate measurement.

            Table 1. Baseline demographic characteristics of study participants.
            Characteristic | Control Group | Treatment Group | P-value
            Age (years)    | 45.2 ± 12.3  | 47.1 ± 11.8    | 0.234
            Gender (M/F)   | 12/13         | 14/11           | 0.456

            Results
            Figure 2. Statistical analysis results showing significant differences (p<0.001).
            Error bars represent 95% confidence intervals.

            Table 2. Summary of primary outcome measures across treatment groups.
            Outcome        | Control | Treatment | Effect Size | P-value
            Primary        | 12.5    | 18.7      | 1.23        | <0.001
            Secondary      | 8.3     | 11.2      | 0.89        | 0.012
            """,
            "extraction_method": "pdfplumber",
            "pages": mock_pages,
            "page_count": 5,
            "raw_content": Mock(),
        }

    @pytest.fixture
    def sample_xml_content(self):
        """Sample XML content with figures and tables."""
        import xml.etree.ElementTree as ET

        xml_string = """<?xml version="1.0" encoding="UTF-8"?>
            <article xmlns:xlink="http://www.w3.org/1999/xlink">
                <body>
                    <sec id="introduction">
                        <title>Introduction</title>
                        <p>This study examines novel treatment approaches.</p>
                    </sec>

                    <sec id="methods">
                        <title>Methods</title>
                        <fig id="fig1" position="float">
                            <label>Figure 1</label>
                            <caption>
                                <p>Experimental setup showing treatment chambers and measurement devices. The system consists of three main components for accurate measurement.</p>
                            </caption>
                            <graphic xlink:href="setup.jpg" mimetype="image" mime-subtype="jpeg"/>
                        </fig>

                        <table-wrap id="tab1" position="float">
                            <label>Table 1</label>
                            <caption>
                                <p>Baseline demographic characteristics of study participants.</p>
                            </caption>
                            <table>
                                <thead>
                                    <tr>
                                        <th>Characteristic</th>
                                        <th>Control Group</th>
                                        <th>Treatment Group</th>
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
                        </table-wrap>
                    </sec>

                    <sec id="results">
                        <title>Results</title>
                        <fig id="fig2" position="float">
                            <label>Figure 2</label>
                            <caption>
                                <p>Statistical analysis results showing significant differences (p&lt;0.001). Error bars represent 95% confidence intervals.</p>
                            </caption>
                            <graphic xlink:href="results.png" mimetype="image" mime-subtype="png"/>
                        </fig>

                        <table-wrap id="tab2" position="float">
                            <label>Table 2</label>
                            <caption>
                                <p>Summary of primary outcome measures across treatment groups.</p>
                            </caption>
                            <table>
                                <thead>
                                    <tr>
                                        <th>Outcome</th>
                                        <th>Control</th>
                                        <th>Treatment</th>
                                        <th>Effect Size</th>
                                        <th>P-value</th>
                                    </tr>
                                </thead>
                                <tbody>
                                    <tr>
                                        <td>Primary</td>
                                        <td>12.5</td>
                                        <td>18.7</td>
                                        <td>1.23</td>
                                        <td>&lt;0.001</td>
                                    </tr>
                                    <tr>
                                        <td>Secondary</td>
                                        <td>8.3</td>
                                        <td>11.2</td>
                                        <td>0.89</td>
                                        <td>0.012</td>
                                    </tr>
                                </tbody>
                            </table>
                        </table-wrap>
                    </sec>
                </body>
            </article>"""

        # Parse the XML to create proper structure
        root_element = ET.fromstring(xml_string)

        return {
            "raw_content": xml_string,
            "root_element": root_element,
            "namespaces": {"xlink": "http://www.w3.org/1999/xlink"},
            "enhanced_namespaces": {
                "declared": {"xlink": "http://www.w3.org/1999/xlink"},
                "used": ["xlink"],
                "prefixes": {"http://www.w3.org/1999/xlink": ["xlink"]},
                "conflicts": [],
            },
            "schema_info": {
                "schema": "pmc",
                "version": None,
                "confidence": 0.8,
                "indicators": ["article", "fig", "table-wrap"],
            },
            "structure_metadata": {
                "root_element": "article",
                "depth": 4,
                "element_counts": {
                    "article": 1,
                    "body": 1,
                    "sec": 3,
                    "fig": 2,
                    "table-wrap": 2,
                },
                "text_density": 0.6,
                "structure_complexity": 0.7,
                "has_mixed_content": True,
            },
            "parsing_method": "etree",
            "element_count": 50,
            "extraction_method": "etree",
            "schema_type": "pmc",
        }

    def test_unified_output_format(
        self, pdf_parser, xml_parser, sample_pdf_content, sample_xml_content
    ):
        """Test that both parsers produce the same output format."""
        # Extract from both parsers
        pdf_result = pdf_parser.extract_figures_tables(sample_pdf_content)
        xml_result = xml_parser.extract_figures_tables(sample_xml_content)

        # Both should have same top-level structure
        required_keys = {"figures", "tables", "extraction_summary"}
        assert set(pdf_result.keys()) == required_keys
        assert set(xml_result.keys()) == required_keys

        # Both should have lists of figures and tables
        assert isinstance(pdf_result["figures"], list)
        assert isinstance(pdf_result["tables"], list)
        assert isinstance(xml_result["figures"], list)
        assert isinstance(xml_result["tables"], list)

        # Test figure format consistency
        for figures in [pdf_result["figures"], xml_result["figures"]]:
            for figure in figures:
                required_figure_keys = {"id", "type", "caption", "metadata", "quality"}
                common_keys = required_figure_keys & set(figure.keys())
                assert len(common_keys) >= 4  # Should have most required keys

                # Metadata structure
                if "metadata" in figure:
                    metadata = figure["metadata"]
                    expected_metadata_keys = {"context", "technical"}
                    assert any(key in metadata for key in expected_metadata_keys)

                # Quality structure
                if "quality" in figure:
                    quality = figure["quality"]
                    expected_quality_keys = {"extraction_confidence", "overall_quality"}
                    assert any(key in quality for key in expected_quality_keys)

        # Test table format consistency
        for tables in [pdf_result["tables"], xml_result["tables"]]:
            for table in tables:
                required_table_keys = {"id", "type", "caption", "metadata", "quality"}
                common_keys = required_table_keys & set(table.keys())
                assert len(common_keys) >= 4  # Should have most required keys

                # Table-specific metadata
                if "metadata" in table and "structure" in table["metadata"]:
                    structure = table["metadata"]["structure"]
                    expected_structure_keys = {"rows", "columns"}
                    assert any(key in structure for key in expected_structure_keys)

    def test_json_serialization_compatibility(
        self, pdf_parser, xml_parser, sample_pdf_content, sample_xml_content
    ):
        """Test that outputs from both parsers are JSON serializable."""
        pdf_result = pdf_parser.extract_figures_tables(sample_pdf_content)
        xml_result = xml_parser.extract_figures_tables(sample_xml_content)

        # Both results should be JSON serializable
        pdf_json = json.dumps(pdf_result, default=str)
        xml_json = json.dumps(xml_result, default=str)

        assert isinstance(pdf_json, str)
        assert isinstance(xml_json, str)

        # Should be able to parse back
        pdf_parsed = json.loads(pdf_json)
        xml_parsed = json.loads(xml_json)

        assert isinstance(pdf_parsed, dict)
        assert isinstance(xml_parsed, dict)
        assert "figures" in pdf_parsed
        assert "figures" in xml_parsed

    def test_quality_metrics_consistency(
        self, pdf_parser, xml_parser, sample_pdf_content, sample_xml_content
    ):
        """Test that quality metrics are consistent across parsers."""
        pdf_result = pdf_parser.extract_figures_tables(sample_pdf_content)
        xml_result = xml_parser.extract_figures_tables(sample_xml_content)

        # Both should have quality metrics
        for result in [pdf_result, xml_result]:
            summary = result["extraction_summary"]
            assert "average_quality_score" in summary
            assert 0.0 <= summary["average_quality_score"] <= 1.0

            # Individual items should have quality scores
            all_items = result["figures"] + result["tables"]
            for item in all_items:
                if "quality" in item:
                    quality = item["quality"]
                    if "overall_quality" in quality:
                        assert 0.0 <= quality["overall_quality"] <= 1.0

    def test_content_extraction_accuracy(
        self, pdf_parser, xml_parser, sample_pdf_content, sample_xml_content
    ):
        """Test accuracy of content extraction across parsers."""
        pdf_result = pdf_parser.extract_figures_tables(sample_pdf_content)
        xml_result = xml_parser.extract_figures_tables(sample_xml_content)

        # Should find similar number of figures and tables
        pdf_fig_count = len(pdf_result["figures"])
        xml_fig_count = len(xml_result["figures"])
        pdf_table_count = len(pdf_result["tables"])
        xml_table_count = len(xml_result["tables"])

        # Counts should be reasonably close (within 1-2 items)
        assert abs(pdf_fig_count - xml_fig_count) <= 2
        assert abs(pdf_table_count - xml_table_count) <= 2

        # Should extract meaningful captions
        for result in [pdf_result, xml_result]:
            for figure in result["figures"]:
                if "caption" in figure:
                    assert len(figure["caption"]) > 5  # Should have substantial caption

            for table in result["tables"]:
                if "caption" in table:
                    assert len(table["caption"]) > 5  # Should have substantial caption


class TestEndToEndWorkflow:
    """Test complete end-to-end extraction workflow."""

    @pytest.fixture
    def pdf_parser(self):
        """Create PDF parser instance."""
        return PDFParser()

    @pytest.fixture
    def xml_parser(self):
        """Create XML parser instance."""
        return XMLParser()

    def _create_scientific_paper_xml_content(self):
        """Helper method to create scientific paper XML content structure."""
        import xml.etree.ElementTree as ET

        xml_string = """<?xml version="1.0" encoding="UTF-8"?>
        <article xmlns:xlink="http://www.w3.org/1999/xlink">
            <front>
                <article-meta>
                    <title-group>
                        <article-title>Novel Therapeutic Approaches in Cardiovascular Medicine</article-title>
                    </title-group>
                </article-meta>
            </front>
            <body>
                <sec id="introduction">
                    <title>Introduction</title>
                    <p>Cardiovascular disease remains a leading cause of mortality worldwide.</p>
                </sec>

                <sec id="methods">
                    <title>Methods</title>

                    <fig id="fig1" position="float">
                        <label>Figure 1</label>
                        <caption>
                            <p>Study flow diagram showing participant recruitment, randomization, and follow-up procedures. A total of 150 participants were screened, with 120 meeting inclusion criteria.</p>
                        </caption>
                        <graphic xlink:href="flowchart.tiff" mimetype="image" mime-subtype="tiff"/>
                    </fig>

                    <table-wrap id="tab1" position="float">
                        <label>Table 1</label>
                        <caption>
                            <p>Baseline demographic and clinical characteristics of study participants.</p>
                        </caption>
                        <table>
                            <thead>
                                <tr>
                                    <th>Characteristic</th>
                                    <th>Control (n=60)</th>
                                    <th>Treatment (n=60)</th>
                                    <th>P-value</th>
                                </tr>
                            </thead>
                            <tbody>
                                <tr>
                                    <td>Age (years)</td>
                                    <td>58.3 ± 12.7</td>
                                    <td>59.1 ± 11.9</td>
                                    <td>0.712</td>
                                </tr>
                                <tr>
                                    <td>Male gender, n (%)</td>
                                    <td>35 (58.3)</td>
                                    <td>38 (63.3)</td>
                                    <td>0.584</td>
                                </tr>
                            </tbody>
                        </table>
                    </table-wrap>

                    <fig id="fig2" position="float">
                        <label>Figure 2</label>
                        <caption>
                            <p>Detailed intervention protocol flowchart showing three treatment phases over 24 weeks.</p>
                        </caption>
                        <graphic xlink:href="protocol.png" mimetype="image" mime-subtype="png"/>
                    </fig>
                </sec>

                <sec id="results">
                    <title>Results</title>

                    <fig id="fig3" position="float">
                        <label>Figure 3</label>
                        <caption>
                            <p>Primary outcome results showing significant improvement in treatment group. Box plots with 95% confidence intervals.</p>
                        </caption>
                        <graphic xlink:href="outcomes.tiff" mimetype="image" mime-subtype="tiff"/>
                    </fig>

                    <table-wrap id="tab2" position="float">
                        <label>Table 2</label>
                        <caption>
                            <p>Primary and secondary outcome measures at 24-week follow-up.</p>
                        </caption>
                        <table>
                            <thead>
                                <tr>
                                    <th>Outcome Measure</th>
                                    <th>Control</th>
                                    <th>Treatment</th>
                                    <th>Difference (95% CI)</th>
                                    <th>P-value</th>
                                </tr>
                            </thead>
                            <tbody>
                                <tr>
                                    <td>Primary endpoint</td>
                                    <td>42.1 ± 8.3</td>
                                    <td>58.7 ± 9.2</td>
                                    <td>16.6 (13.2-20.0)</td>
                                    <td>&lt;0.001</td>
                                </tr>
                            </tbody>
                        </table>
                    </table-wrap>

                    <fig id="fig4" position="float">
                        <label>Figure 4</label>
                        <caption>
                            <p>Subgroup analysis results stratified by baseline characteristics. Forest plot with hazard ratios.</p>
                        </caption>
                        <graphic xlink:href="subgroup.png" mimetype="image" mime-subtype="png"/>
                    </fig>
                </sec>
            </body>
        </article>"""

        # Parse the XML to create proper structure
        root_element = ET.fromstring(xml_string)

        return {
            "raw_content": xml_string,
            "root_element": root_element,
            "namespaces": {"xlink": "http://www.w3.org/1999/xlink"},
            "enhanced_namespaces": {
                "declared": {"xlink": "http://www.w3.org/1999/xlink"},
                "used": ["xlink"],
                "prefixes": {"http://www.w3.org/1999/xlink": ["xlink"]},
                "conflicts": [],
            },
            "schema_info": {
                "schema": "pmc",
                "version": None,
                "confidence": 0.9,
                "indicators": ["article", "fig", "table-wrap", "front", "body"],
            },
            "structure_metadata": {
                "root_element": "article",
                "depth": 5,
                "element_counts": {
                    "article": 1,
                    "front": 1,
                    "body": 1,
                    "sec": 3,
                    "fig": 4,
                    "table-wrap": 2,
                },
                "text_density": 0.7,
                "structure_complexity": 0.8,
                "has_mixed_content": True,
            },
            "parsing_method": "etree",
            "element_count": 80,
            "extraction_method": "etree",
            "schema_type": "pmc",
        }

    def test_complete_scientific_paper_extraction(self, pdf_parser, xml_parser):
        """Test extraction from a complete scientific paper."""
        # Comprehensive scientific paper content
        # Create mock page objects for scientific paper PDF
        paper_mock_pages = []
        for i in range(12):
            mock_page = Mock()
            mock_page.extract_text.return_value = f"Scientific paper page {i+1} content"
            paper_mock_pages.append(mock_page)

        paper_content = {
            "pdf": {
                "text": """
                Title: Novel Therapeutic Approaches in Cardiovascular Medicine

                Abstract
                Background: Current treatments show limited efficacy.
                Methods: Randomized controlled trial with 150 participants.
                Results: Significant improvements observed (p<0.001).
                Conclusions: Novel approach demonstrates superior outcomes.

                Introduction
                Cardiovascular disease remains a leading cause of mortality worldwide.
                Previous studies have shown limitations in current therapeutic approaches.

                Methods
                Study Design
                This was a double-blind, randomized controlled trial conducted at three centers.

                Figure 1. Study flow diagram showing participant recruitment, randomization,
                and follow-up procedures. A total of 150 participants were screened, with
                120 meeting inclusion criteria and randomized to treatment (n=60) or control (n=60) groups.

                Participants
                Table 1. Baseline demographic and clinical characteristics of study participants.
                Characteristic          | Control (n=60) | Treatment (n=60) | P-value
                Age (years)            | 58.3 ± 12.7    | 59.1 ± 11.9      | 0.712
                Male gender, n (%)     | 35 (58.3)      | 38 (63.3)        | 0.584
                BMI (kg/m²)           | 28.4 ± 4.2     | 27.9 ± 4.1       | 0.486
                Diabetes, n (%)        | 18 (30.0)      | 21 (35.0)        | 0.553
                Hypertension, n (%)    | 42 (70.0)      | 44 (73.3)        | 0.686

                Intervention Protocol
                The treatment protocol consisted of three phases as illustrated in Figure 2.

                Figure 2. Detailed intervention protocol flowchart showing three treatment phases:
                Phase I (weeks 1-4): Initial assessment and baseline measurements
                Phase II (weeks 5-12): Active intervention with dose escalation
                Phase III (weeks 13-24): Maintenance therapy and monitoring

                Statistical Analysis
                All analyses were performed using intention-to-treat principles.

                Results
                Primary Outcomes
                Figure 3. Primary outcome results showing significant improvement in treatment group.
                Box plots demonstrate median values with interquartile ranges and 95% confidence intervals.
                Statistical significance: ***p<0.001, **p<0.01, *p<0.05.

                Table 2. Primary and secondary outcome measures at 24-week follow-up.
                Outcome Measure                | Control        | Treatment      | Difference (95% CI) | P-value
                Primary endpoint (score)      | 42.1 ± 8.3     | 58.7 ± 9.2     | 16.6 (13.2-20.0)   | <0.001
                Secondary endpoint A (%)      | 23.4 ± 5.1     | 31.8 ± 6.2     | 8.4 (5.9-10.9)     | <0.001
                Secondary endpoint B (units)  | 156.3 ± 24.7   | 189.2 ± 28.1   | 32.9 (21.4-44.4)   | <0.001
                Quality of life score         | 6.2 ± 1.4      | 7.8 ± 1.6      | 1.6 (1.0-2.2)      | <0.001

                Subgroup Analysis
                Figure 4. Subgroup analysis results stratified by age, gender, and comorbidities.
                Forest plot showing hazard ratios with 95% confidence intervals for predefined subgroups.

                Table 3. Subgroup analysis of primary outcome by baseline characteristics.
                Subgroup              | Control (n)  | Treatment (n) | Effect Size | P-interaction
                Age <65 years         | 28.3 ± 6.1   | 35.2 ± 7.4   | 1.12        | 0.234
                Age ≥65 years         | 31.7 ± 7.8   | 42.1 ± 8.9   | 1.38        | 0.156
                Male                  | 29.4 ± 6.9   | 37.8 ± 8.2   | 1.24        | 0.045
                Female                | 30.2 ± 7.1   | 39.1 ± 8.5   | 1.29        | 0.067

                Safety Analysis
                Table 4. Adverse events and safety profile comparison between groups.
                Adverse Event         | Control n (%) | Treatment n (%) | Risk Ratio (95% CI) | P-value
                Any adverse event     | 45 (75.0)     | 42 (70.0)       | 0.93 (0.74-1.18)   | 0.558
                Serious adverse event | 8 (13.3)      | 6 (10.0)        | 0.75 (0.28-2.03)   | 0.571
                Study discontinuation | 5 (8.3)       | 3 (5.0)         | 0.60 (0.15-2.41)   | 0.470

                Discussion
                The results demonstrate significant therapeutic benefits as shown in Figure 3.
                The safety profile was acceptable as detailed in Table 4.
                Subgroup analyses (Table 3, Figure 4) suggest consistent benefits across populations.

                Limitations
                Several limitations should be acknowledged in interpreting these results.

                Conclusions
                This study provides strong evidence for the efficacy and safety of the novel intervention.
                """,
                "extraction_method": "pdfplumber",
                "pages": paper_mock_pages,
                "page_count": 12,
                "raw_content": Mock(),
            },
            "xml": {
                # Create proper XML structure with parsed root element
                **self._create_scientific_paper_xml_content()
            },
        }

        # Extract from both formats
        pdf_result = pdf_parser.extract_figures_tables(paper_content["pdf"])
        xml_result = xml_parser.extract_figures_tables(paper_content["xml"])

        # Verify comprehensive extraction
        for result, parser_name in [(pdf_result, "PDF"), (xml_result, "XML")]:
            # Should find multiple figures and tables
            assert (
                len(result["figures"]) >= 3
            ), f"{parser_name}: Should find at least 3 figures"
            assert (
                len(result["tables"]) >= 2
            ), f"{parser_name}: Should find at least 2 tables"

            # Verify figure quality
            for i, figure in enumerate(result["figures"]):
                assert "id" in figure, f"{parser_name}: Figure {i} missing ID"
                assert "caption" in figure, f"{parser_name}: Figure {i} missing caption"
                assert (
                    len(figure["caption"]) > 20
                ), f"{parser_name}: Figure {i} caption too short"

                if "quality" in figure:
                    quality = figure["quality"]
                    if "overall_quality" in quality:
                        assert (
                            quality["overall_quality"] > 0.3
                        ), f"{parser_name}: Figure {i} quality too low"

            # Verify table quality
            for i, table in enumerate(result["tables"]):
                assert "id" in table, f"{parser_name}: Table {i} missing ID"
                assert "caption" in table, f"{parser_name}: Table {i} missing caption"
                assert (
                    len(table["caption"]) > 15
                ), f"{parser_name}: Table {i} caption too short"

                if "metadata" in table and "structure" in table["metadata"]:
                    structure = table["metadata"]["structure"]
                    if "rows" in structure and "columns" in structure:
                        assert (
                            structure["rows"] > 0
                        ), f"{parser_name}: Table {i} should have rows"
                        assert (
                            structure["columns"] > 0
                        ), f"{parser_name}: Table {i} should have columns"

            # Verify summary statistics
            summary = result["extraction_summary"]
            assert (
                summary["total_figures"] >= 3
            ), f"{parser_name}: Summary figures count incorrect"
            assert (
                summary["total_tables"] >= 2
            ), f"{parser_name}: Summary tables count incorrect"

            if "average_quality_score" in summary:
                assert (
                    0.0 <= summary["average_quality_score"] <= 1.0
                ), f"{parser_name}: Invalid quality score"

    def test_performance_benchmarking(self, pdf_parser, xml_parser):
        """Test performance with large documents."""
        import time

        # Create large document content
        # Create mock page objects for large PDF
        large_mock_pages = []
        for i in range(10):
            mock_page = Mock()
            mock_page.extract_text.return_value = f"Large page {i+1} content"
            large_mock_pages.append(mock_page)

        large_pdf_content = {
            "text": "Introduction. " * 100,
            "extraction_method": "pdfplumber",
            "pages": large_mock_pages,
            "page_count": 10,
            "raw_content": Mock(),
        }

        # Add many figures and tables
        for i in range(50):
            large_pdf_content[
                "text"
            ] += f"\nFigure {i+1}. Test figure {i+1} with detailed caption describing experimental setup and results. "
            large_pdf_content[
                "text"
            ] += f"\nTable {i+1}. Test table {i+1} with comprehensive data analysis and statistical comparisons. "
            large_pdf_content["text"] += "Content text. " * 20

        # Build large XML content string
        large_xml_string = "<article><body>"

        # Add many figures and tables to XML
        for i in range(50):
            large_xml_string += f"""
            <fig id="fig{i+1}">
                <label>Figure {i+1}</label>
                <caption><p>Test figure {i+1} with detailed caption.</p></caption>
            </fig>
            <table-wrap id="tab{i+1}">
                <label>Table {i+1}</label>
                <caption><p>Test table {i+1} with data analysis.</p></caption>
                <table><tr><td>Data</td></tr></table>
            </table-wrap>
            """

        large_xml_string += "</body></article>"

        # Parse the large XML to create proper structure
        import xml.etree.ElementTree as ET

        large_root_element = ET.fromstring(large_xml_string)

        large_xml_content = {
            "raw_content": large_xml_string,
            "root_element": large_root_element,
            "namespaces": {},
            "enhanced_namespaces": {
                "declared": {},
                "used": [],
                "prefixes": {},
                "conflicts": [],
            },
            "schema_info": {
                "schema": "generic",
                "version": None,
                "confidence": 0.6,
                "indicators": ["article", "fig", "table-wrap"],
            },
            "structure_metadata": {
                "root_element": "article",
                "depth": 3,
                "element_counts": {
                    "article": 1,
                    "body": 1,
                    "fig": 50,
                    "table-wrap": 50,
                },
                "text_density": 0.4,
                "structure_complexity": 0.8,
                "has_mixed_content": False,
            },
            "parsing_method": "etree",
            "element_count": 300,
            "extraction_method": "etree",
            "schema_type": "generic",
        }

        # Benchmark PDF parser
        start_time = time.time()
        pdf_result = pdf_parser.extract_figures_tables(large_pdf_content)
        pdf_time = time.time() - start_time

        # Benchmark XML parser
        start_time = time.time()
        xml_result = xml_parser.extract_figures_tables(large_xml_content)
        xml_time = time.time() - start_time

        # Performance assertions (generous limits)
        assert pdf_time < 30.0, f"PDF parser too slow: {pdf_time:.2f}s"
        assert xml_time < 30.0, f"XML parser too slow: {xml_time:.2f}s"

        # Should still extract content correctly
        assert (
            len(pdf_result["figures"]) > 0
        ), "PDF parser should find figures in large document"
        assert (
            len(xml_result["figures"]) > 0
        ), "XML parser should find figures in large document"

        # Record performance metrics
        print(f"\nPerformance Benchmarks:")
        print(
            f"PDF Parser: {pdf_time:.2f}s for {len(pdf_result['figures'])} figures, {len(pdf_result['tables'])} tables"
        )
        print(
            f"XML Parser: {xml_time:.2f}s for {len(xml_result['figures'])} figures, {len(xml_result['tables'])} tables"
        )


class TestQualityThresholdFiltering:
    """Test quality threshold filtering functionality."""

    @pytest.fixture
    def pdf_parser(self):
        """Create PDF parser instance."""
        return PDFParser()

    @pytest.fixture
    def xml_parser(self):
        """Create XML parser instance."""
        return XMLParser()

    def _create_mixed_quality_xml_content(self):
        """Helper method to create mixed quality XML content structure."""
        import xml.etree.ElementTree as ET

        xml_string = """
        <article>
            <fig id="fig1">
                <label>Figure 1</label>
                <caption>
                    <p>High-quality figure with comprehensive caption describing experimental methodology and detailed statistical analysis.</p>
                </caption>
                <graphic href="detailed.tiff" mimetype="image"/>
            </fig>

            <fig id="fig2">
                <caption><p>Simple.</p></caption>
            </fig>

            <table-wrap id="tab1">
                <label>Table 1</label>
                <caption>
                    <p>Comprehensive demographic and clinical characteristics with statistical comparisons.</p>
                </caption>
                <table>
                    <tr><th>Characteristic</th><th>Control</th><th>Treatment</th><th>P-value</th></tr>
                    <tr><td>Age</td><td>45.2</td><td>47.1</td><td>0.234</td></tr>
                </table>
            </table-wrap>

            <table-wrap id="tab2">
                <caption><p>Simple.</p></caption>
            </table-wrap>
        </article>"""

        # Parse the XML to create proper structure
        root_element = ET.fromstring(xml_string)

        return {
            "raw_content": xml_string,
            "root_element": root_element,
            "namespaces": {},
            "enhanced_namespaces": {
                "declared": {},
                "used": [],
                "prefixes": {},
                "conflicts": [],
            },
            "schema_info": {
                "schema": "generic",
                "version": None,
                "confidence": 0.6,
                "indicators": ["article", "fig", "table-wrap"],
            },
            "structure_metadata": {
                "root_element": "article",
                "depth": 3,
                "element_counts": {"article": 1, "fig": 2, "table-wrap": 2},
                "text_density": 0.5,
                "structure_complexity": 0.6,
                "has_mixed_content": False,
            },
            "parsing_method": "etree",
            "element_count": 20,
            "extraction_method": "etree",
            "schema_type": "generic",
        }

    def test_quality_threshold_application(self, pdf_parser, xml_parser):
        """Test that quality thresholds filter out low-quality extractions."""
        # Mixed quality content
        # Create mock page for mixed quality test
        mixed_mock_page = Mock()
        mixed_mock_page.extract_text.return_value = "Mixed quality test page content"

        mixed_content = {
            "pdf": {
                "text": """
                Figure 1. High-quality figure with comprehensive caption describing experimental methodology, statistical analysis (p<0.001), and detailed results with 95% confidence intervals.

                Fig. 2. Low quality.

                Table 1. Comprehensive demographic and clinical characteristics of study participants including age, gender, BMI, comorbidities, and baseline measurements with statistical comparisons between treatment groups.

                Tab. 2. Simple.
                """,
                "extraction_method": "pdfplumber",
                "pages": [mixed_mock_page],
                "raw_content": Mock(),
            },
            "xml": {
                # Create proper mixed quality XML structure
                **self._create_mixed_quality_xml_content()
            },
        }

        # Extract without filtering
        pdf_result = pdf_parser.extract_figures_tables(mixed_content["pdf"])
        xml_result = xml_parser.extract_figures_tables(mixed_content["xml"])

        # Should find all items initially
        total_pdf_items = len(pdf_result["figures"]) + len(pdf_result["tables"])
        total_xml_items = len(xml_result["figures"]) + len(xml_result["tables"])

        assert total_pdf_items >= 2, "Should find multiple items in PDF"
        assert total_xml_items >= 2, "Should find multiple items in XML"

        # Apply quality filtering (simulate)
        def filter_by_quality(items, min_quality=0.5):
            """Filter items by quality threshold."""
            filtered = []
            for item in items:
                if "quality" in item and "overall_quality" in item["quality"]:
                    if item["quality"]["overall_quality"] >= min_quality:
                        filtered.append(item)
                else:
                    # Include items without quality scores for now
                    filtered.append(item)
            return filtered

        # Apply filtering to both results
        pdf_high_quality_figures = filter_by_quality(pdf_result["figures"], 0.5)
        pdf_high_quality_tables = filter_by_quality(pdf_result["tables"], 0.5)
        xml_high_quality_figures = filter_by_quality(xml_result["figures"], 0.5)
        xml_high_quality_tables = filter_by_quality(xml_result["tables"], 0.5)

        # High-quality items should have better captions
        for figures in [pdf_high_quality_figures, xml_high_quality_figures]:
            for figure in figures:
                if "caption" in figure:
                    # High-quality figures should have longer, more detailed captions
                    caption_length = len(figure["caption"])
                    # Allow for variation in extraction quality
                    if caption_length > 0:
                        assert (
                            caption_length > 15
                        ), f"High-quality figure should have substantial caption: '{figure['caption']}'"

        for tables in [pdf_high_quality_tables, xml_high_quality_tables]:
            for table in tables:
                if "caption" in table:
                    # High-quality tables should have longer, more detailed captions
                    caption_length = len(table["caption"])
                    if caption_length > 0:
                        assert (
                            caption_length > 15
                        ), f"High-quality table should have substantial caption: '{table['caption']}'"


class TestErrorHandlingAndRobustness:
    """Test error handling and robustness across parsers."""

    @pytest.fixture
    def pdf_parser(self):
        """Create PDF parser instance."""
        return PDFParser()

    @pytest.fixture
    def xml_parser(self):
        """Create XML parser instance."""
        return XMLParser()

    def test_corrupted_content_handling(self, pdf_parser, xml_parser):
        """Test handling of corrupted or incomplete content."""
        corrupted_contents = [
            # PDF with None content
            {
                "pdf": {
                    "text": None,
                    "extraction_method": "pdfplumber",
                    "pages": [],
                    "raw_content": Mock(),
                }
            },
            # XML with malformed content - Note: this should be a proper structure but with bad XML content
            {
                "xml": {
                    "raw_content": "<article><fig><caption>Incomplete",
                    "root_element": None,
                    "extraction_method": "etree",
                }
            },
            # Empty content
            {
                "pdf": {
                    "text": "",
                    "extraction_method": "pdfplumber",
                    "pages": [],
                    "raw_content": Mock(),
                }
            },
            {
                "xml": {
                    "raw_content": "",
                    "root_element": None,
                    "extraction_method": "etree",
                }
            },
        ]

        for content_set in corrupted_contents:
            if "pdf" in content_set:
                # Should not crash
                try:
                    result = pdf_parser.extract_figures_tables(content_set["pdf"])
                    assert isinstance(result, dict)
                    assert "figures" in result
                    assert "tables" in result
                except Exception as e:
                    # Some exceptions may be expected for severely corrupted content
                    assert isinstance(e, (ValueError, TypeError, AttributeError))

            if "xml" in content_set:
                # Should not crash
                try:
                    result = xml_parser.extract_figures_tables(content_set["xml"])
                    assert isinstance(result, dict)
                    assert "figures" in result
                    assert "tables" in result
                except Exception as e:
                    # Some exceptions may be expected for malformed XML
                    assert isinstance(e, (ValueError, TypeError))

    def test_memory_efficiency_large_documents(self, pdf_parser, xml_parser):
        """Test memory efficiency with very large documents."""
        import sys

        # Create memory-intensive content
        large_text = "Text content with figures and tables. " * 10000
        large_text += "Figure 1. Memory test figure. " * 100
        large_text += "Table 1. Memory test table. " * 100

        large_xml = "<article><body>"
        for i in range(1000):
            large_xml += f"<p>Content paragraph {i}</p>"
        large_xml += (
            "<fig id='mem-fig'><caption><p>Memory test figure</p></caption></fig>"
        )
        large_xml += "<table-wrap id='mem-tab'><caption><p>Memory test table</p></caption></table-wrap>"
        large_xml += "</body></article>"

        # Create mock page for memory test
        memory_mock_page = Mock()
        memory_mock_page.extract_text.return_value = "Memory test page content"

        pdf_content = {
            "text": large_text,
            "extraction_method": "pdfplumber",
            "pages": [memory_mock_page],
            "raw_content": Mock(),
        }

        # Parse large XML for proper structure
        import xml.etree.ElementTree as ET

        large_xml_root = ET.fromstring(large_xml)

        xml_content = {
            "raw_content": large_xml,
            "root_element": large_xml_root,
            "namespaces": {},
            "enhanced_namespaces": {
                "declared": {},
                "used": [],
                "prefixes": {},
                "conflicts": [],
            },
            "schema_info": {
                "schema": "generic",
                "version": None,
                "confidence": 0.5,
                "indicators": ["article", "fig", "table-wrap"],
            },
            "structure_metadata": {
                "root_element": "article",
                "depth": 3,
                "element_counts": {"article": 1, "body": 1, "fig": 1, "table-wrap": 1},
                "text_density": 0.3,
                "structure_complexity": 0.5,
                "has_mixed_content": True,
            },
            "parsing_method": "etree",
            "element_count": 2000,
            "extraction_method": "etree",
        }

        # Monitor memory usage (basic check)
        initial_size = sys.getsizeof(pdf_content) + sys.getsizeof(xml_content)

        # Process large documents
        pdf_result = pdf_parser.extract_figures_tables(pdf_content)
        xml_result = xml_parser.extract_figures_tables(xml_content)

        # Should complete successfully
        assert isinstance(pdf_result, dict)
        assert isinstance(xml_result, dict)

        # Results should be much smaller than input
        result_size = sys.getsizeof(pdf_result) + sys.getsizeof(xml_result)
        assert result_size < initial_size, "Results should be more compact than input"

    def test_concurrent_processing_safety(self, pdf_parser, xml_parser):
        """Test thread safety for concurrent processing."""
        import threading
        import time

        # Sample content for concurrent processing
        # Create mock page for concurrent test
        concurrent_mock_page = Mock()
        concurrent_mock_page.extract_text.return_value = "Concurrent test page content"

        sample_pdf = {
            "text": "Figure 1. Concurrent test figure. Table 1. Concurrent test table.",
            "extraction_method": "pdfplumber",
            "pages": [concurrent_mock_page],
            "raw_content": Mock(),
        }

        # Create proper XML structure for concurrent test
        import xml.etree.ElementTree as ET

        concurrent_xml_string = """
        <article>
            <fig id="concurrent-fig">
                <caption><p>Concurrent test figure</p></caption>
            </fig>
            <table-wrap id="concurrent-tab">
                <caption><p>Concurrent test table</p></caption>
            </table-wrap>
        </article>"""

        concurrent_root_element = ET.fromstring(concurrent_xml_string)

        sample_xml = {
            "raw_content": concurrent_xml_string,
            "root_element": concurrent_root_element,
            "namespaces": {},
            "enhanced_namespaces": {
                "declared": {},
                "used": [],
                "prefixes": {},
                "conflicts": [],
            },
            "schema_info": {
                "schema": "generic",
                "version": None,
                "confidence": 0.6,
                "indicators": ["article", "fig", "table-wrap"],
            },
            "structure_metadata": {
                "root_element": "article",
                "depth": 3,
                "element_counts": {"article": 1, "fig": 1, "table-wrap": 1},
                "text_density": 0.5,
                "structure_complexity": 0.4,
                "has_mixed_content": False,
            },
            "parsing_method": "etree",
            "element_count": 10,
            "extraction_method": "etree",
        }

        results = []
        errors = []

        def process_pdf():
            try:
                result = pdf_parser.extract_figures_tables(sample_pdf)
                results.append(("pdf", result))
            except Exception as e:
                errors.append(("pdf", e))

        def process_xml():
            try:
                result = xml_parser.extract_figures_tables(sample_xml)
                results.append(("xml", result))
            except Exception as e:
                errors.append(("xml", e))

        # Create multiple threads
        threads = []
        for _ in range(5):  # 5 PDF threads
            thread = threading.Thread(target=process_pdf)
            threads.append(thread)

        for _ in range(5):  # 5 XML threads
            thread = threading.Thread(target=process_xml)
            threads.append(thread)

        # Start all threads
        start_time = time.time()
        for thread in threads:
            thread.start()

        # Wait for completion
        for thread in threads:
            thread.join(timeout=30)  # 30 second timeout

        end_time = time.time()

        # Should complete within reasonable time
        assert (end_time - start_time) < 60.0, "Concurrent processing took too long"

        # Should have minimal errors
        assert len(errors) == 0, f"Concurrent processing errors: {errors}"

        # Should have results from all threads
        assert len(results) == 10, f"Expected 10 results, got {len(results)}"

        # All results should be valid
        for parser_type, result in results:
            assert isinstance(result, dict), f"{parser_type} result not a dict"
            assert "figures" in result, f"{parser_type} missing figures"
            assert "tables" in result, f"{parser_type} missing tables"
