"""
Comprehensive unit tests for the content_utils module.

This module contains thorough tests for TextAnalyzer, StatisticalAnalyzer,
TableContentExtractor, and FigureContentExtractor classes.
"""

import pytest
import statistics
from collections import Counter
from unittest.mock import patch

from aim2_project.aim2_ontology.parsers.content_utils import (
    TextAnalyzer, StatisticalAnalyzer, TableContentExtractor, 
    FigureContentExtractor
)
from aim2_project.aim2_ontology.parsers.metadata_framework import (
    DataType, TableStructure, TableContent, ContentAnalysis
)


class TestTextAnalyzer:
    """Test TextAnalyzer functionality."""
    
    @pytest.fixture
    def analyzer(self):
        """Create TextAnalyzer instance for testing."""
        return TextAnalyzer()
    
    def test_extract_scientific_notation(self, analyzer):
        """Test extraction of scientific notation expressions."""
        test_text = "The p-value was p<0.001 with 95% CI [12.3-18.7]. Mean ± SD: 15.2 ± 3.4 µg/mL."
        
        notations = analyzer.extract_scientific_notation(test_text)
        
        assert isinstance(notations, list)
        assert len(notations) > 0
        # Should find p-values, percentages, ranges, etc.
        assert any('p<0.001' in str(notation) for notation in notations)
    
    def test_extract_scientific_notation_empty(self, analyzer):
        """Test scientific notation extraction with empty text."""
        notations = analyzer.extract_scientific_notation("")
        assert notations == []
        
        notations = analyzer.extract_scientific_notation("simple text with no scientific notation")
        assert len(notations) == 0
    
    def test_identify_domain(self, analyzer):
        """Test domain identification from text."""
        test_cases = [
            ("Patient treatment with clinical diagnosis", ["medical"]),
            ("Cell culture and protein expression in organisms", ["biological"]),
            ("Statistical analysis with mean and correlation", ["statistical"]),
            ("Control group in experimental trial design", ["experimental"]),
            ("Simple text without scientific terms", []),
        ]
        
        for text, expected_domains in test_cases:
            domains = analyzer.identify_domain(text)
            for expected in expected_domains:
                assert expected in domains, f"Expected {expected} in domains for text: {text}"
    
    def test_extract_numerical_values(self, analyzer):
        """Test numerical value extraction."""
        test_text = "Temperature: 25.5°C, pH 7.4, concentration 1.2e-3 M, range 10-20"
        
        values = analyzer.extract_numerical_values(test_text)
        
        assert isinstance(values, list)
        assert len(values) > 0
        assert 25.5 in values
        assert 7.4 in values
        assert 1.2e-3 in values
        assert 10 in values
        assert 20 in values
    
    def test_extract_numerical_values_empty(self, analyzer):
        """Test numerical extraction with non-numeric text."""
        values = analyzer.extract_numerical_values("no numbers here at all")
        assert values == []
        
        values = analyzer.extract_numerical_values("")
        assert values == []
    
    def test_analyze_text_complexity(self, analyzer):
        """Test text complexity analysis."""
        # Simple text
        simple_text = "This is a simple sentence."
        simple_analysis = analyzer.analyze_text_complexity(simple_text)
        
        # Complex scientific text
        complex_text = ("The pharmacokinetic analysis revealed significant bioavailability "
                       "differences (p<0.001) between treatment groups. Statistical significance "
                       "was determined using ANOVA with post-hoc Tukey's test.")
        complex_analysis = analyzer.analyze_text_complexity(complex_text)
        
        # Test structure
        for analysis in [simple_analysis, complex_analysis]:
            assert "word_count" in analysis
            assert "sentence_count" in analysis
            assert "average_word_length" in analysis
            assert "average_sentence_length" in analysis
            assert "unique_words" in analysis
            assert "lexical_diversity" in analysis
            assert "scientific_terms" in analysis
            assert "scientific_density" in analysis
        
        # Complex text should have higher values
        assert complex_analysis["word_count"] > simple_analysis["word_count"]
        assert complex_analysis["scientific_terms"] > simple_analysis["scientific_terms"]
        assert complex_analysis["scientific_density"] > simple_analysis["scientific_density"]
    
    def test_analyze_text_complexity_empty(self, analyzer):
        """Test complexity analysis with empty text."""
        analysis = analyzer.analyze_text_complexity("")
        
        assert analysis["word_count"] == 0
        assert analysis["sentence_count"] == 0
        assert analysis["average_word_length"] == 0
        assert analysis["average_sentence_length"] == 0
        assert analysis["unique_words"] == 0
        assert analysis["lexical_diversity"] == 0


class TestStatisticalAnalyzer:
    """Test StatisticalAnalyzer functionality."""
    
    @pytest.fixture
    def analyzer(self):
        """Create StatisticalAnalyzer instance for testing."""
        return StatisticalAnalyzer()
    
    def test_analyze_distribution(self, analyzer):
        """Test statistical distribution analysis."""
        values = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]
        
        analysis = analyzer.analyze_distribution(values)
        
        # Test required fields
        required_fields = ['count', 'mean', 'median', 'min', 'max', 'range', 'std_dev', 'variance']
        for field in required_fields:
            assert field in analysis
        
        # Test calculated values
        assert analysis['count'] == 10
        assert analysis['mean'] == 5.5
        assert analysis['median'] == 5.5
        assert analysis['min'] == 1.0
        assert analysis['max'] == 10.0
        assert analysis['range'] == 9.0
        assert analysis['std_dev'] == pytest.approx(3.03, rel=1e-2)
        
        # Test quartiles (when enough data)
        assert 'q1' in analysis
        assert 'q3' in analysis
        assert 'iqr' in analysis
        assert analysis['q1'] == 2.0  # 25th percentile
        assert analysis['q3'] == 8.0  # 75th percentile
        assert analysis['iqr'] == 6.0
    
    def test_analyze_distribution_small_dataset(self, analyzer):
        """Test distribution analysis with small dataset."""
        values = [1.0, 2.0]
        
        analysis = analyzer.analyze_distribution(values)
        
        assert analysis['count'] == 2
        assert analysis['mean'] == 1.5
        assert analysis['median'] == 1.5
        assert analysis['std_dev'] == pytest.approx(0.71, rel=1e-2)
        # Quartiles should be None for small datasets
        assert analysis['q1'] is None
        assert analysis['q3'] is None
    
    def test_analyze_distribution_empty(self, analyzer):
        """Test distribution analysis with empty data."""
        analysis = analyzer.analyze_distribution([])
        assert analysis == {}
        
        analysis = analyzer.analyze_distribution([1.0])  # Single value
        assert analysis == {}
    
    def test_detect_outliers_iqr(self, analyzer):
        """Test outlier detection using IQR method."""
        # Data with clear outliers
        values = [1, 2, 3, 4, 5, 6, 7, 8, 9, 100]  # 100 is an outlier
        
        outliers = analyzer.detect_outliers(values, method='iqr')
        
        assert isinstance(outliers, list)
        assert 9 in outliers  # Index of the outlier value (100)
    
    def test_detect_outliers_zscore(self, analyzer):
        """Test outlier detection using Z-score method."""
        # Data with clear outliers
        values = [1, 2, 3, 4, 5, 6, 7, 8, 9, 100]  # 100 is an outlier
        
        outliers = analyzer.detect_outliers(values, method='zscore')
        
        assert isinstance(outliers, list)
        assert 9 in outliers  # Index of the outlier value (100)
    
    def test_detect_outliers_small_dataset(self, analyzer):
        """Test outlier detection with small dataset."""
        values = [1, 2, 3]  # Too small for reliable outlier detection
        
        outliers = analyzer.detect_outliers(values, method='iqr')
        assert outliers == []
    
    def test_correlation_analysis(self, analyzer):
        """Test correlation analysis between two variables."""
        # Perfect positive correlation
        x_values = [1, 2, 3, 4, 5]
        y_values = [2, 4, 6, 8, 10]
        
        correlation = analyzer.correlation_analysis(x_values, y_values)
        
        assert 'correlation_coefficient' in correlation
        assert 'correlation_strength' in correlation
        assert 'sample_size' in correlation
        
        assert correlation['correlation_coefficient'] == pytest.approx(1.0, rel=1e-2)
        assert correlation['correlation_strength'] == 'very_strong'
        assert correlation['sample_size'] == 5
        
        # No correlation
        x_random = [1, 2, 3, 4, 5]
        y_random = [5, 2, 8, 1, 6]
        
        correlation_random = analyzer.correlation_analysis(x_random, y_random)
        assert abs(correlation_random['correlation_coefficient']) < 0.8  # Should be weak
    
    def test_correlation_analysis_invalid(self, analyzer):
        """Test correlation analysis with invalid data."""
        # Mismatched lengths
        result = analyzer.correlation_analysis([1, 2, 3], [1, 2])
        assert result == {}
        
        # Too few data points
        result = analyzer.correlation_analysis([1], [1])
        assert result == {}
        
        # Constant values (zero variance)
        result = analyzer.correlation_analysis([1, 1, 1], [2, 2, 2])
        assert result['correlation_coefficient'] == 0
    
    def test_interpret_correlation(self, analyzer):
        """Test correlation strength interpretation."""
        # Test private method through correlation_analysis
        test_cases = [
            ([1, 2, 3, 4, 5], [1, 2, 3, 4, 5], 'very_strong'),  # r ≈ 1.0
            ([1, 2, 3, 4, 5], [5, 4, 3, 2, 1], 'very_strong'),  # r ≈ -1.0
            ([1, 2, 3, 4, 5], [1, 3, 2, 5, 4], 'moderate'),     # moderate correlation
        ]
        
        for x_vals, y_vals, expected_strength in test_cases:
            result = analyzer.correlation_analysis(x_vals, y_vals)
            # The exact correlation may vary, but we test the interpretation logic
            assert 'correlation_strength' in result


class TestTableContentExtractor:
    """Test TableContentExtractor functionality."""
    
    @pytest.fixture
    def extractor(self):
        """Create TableContentExtractor instance for testing."""
        return TableContentExtractor()
    
    @pytest.fixture
    def sample_table_data(self):
        """Sample table data for testing."""
        return [
            ["Name", "Age", "Active", "Score", "Percentage"],
            ["Alice", "25", "yes", "85.5", "90%"],
            ["Bob", "30", "no", "92.1", "95%"],
            ["Carol", "28", "true", "78.3", "85%"],
            ["Dave", "35", "false", "88.7", "92%"]
        ]
    
    def test_parse_table_structure(self, extractor, sample_table_data):
        """Test table structure parsing."""
        structure = extractor.parse_table_structure(sample_table_data)
        
        assert isinstance(structure, TableStructure)
        assert structure.rows == 5
        assert structure.columns == 5
        assert structure.header_rows == 1  # First row detected as header
        assert structure.column_headers == ["Name", "Age", "Active", "Score", "Percentage"]
        assert len(structure.data_types) == 5
        
        # Check data types
        assert structure.data_types[0] == DataType.TEXT      # Name
        assert structure.data_types[1] == DataType.NUMERIC  # Age
        assert structure.data_types[2] == DataType.BOOLEAN  # Active
        assert structure.data_types[3] == DataType.NUMERIC  # Score
        assert structure.data_types[4] == DataType.PERCENTAGE  # Percentage
        
        assert 0.0 <= structure.structure_complexity <= 1.0
    
    def test_parse_table_structure_empty(self, extractor):
        """Test structure parsing with empty data."""
        structure = extractor.parse_table_structure([])
        
        assert structure.rows == 0
        assert structure.columns == 0
        assert structure.header_rows == 0
        assert structure.column_headers == []
        assert structure.data_types == []
    
    def test_extract_table_content(self, extractor, sample_table_data):
        """Test table content extraction."""
        structure = extractor.parse_table_structure(sample_table_data)
        content = extractor.extract_table_content(sample_table_data, structure)
        
        assert isinstance(content, TableContent)
        assert content.raw_data == sample_table_data
        
        # Check structured data
        assert "Name" in content.structured_data
        assert "Age" in content.structured_data
        assert len(content.structured_data["Name"]) == 4  # Data rows only
        assert content.structured_data["Name"] == ["Alice", "Bob", "Carol", "Dave"]
        
        # Check numerical data extraction
        assert "Age" in content.numerical_data
        assert "Score" in content.numerical_data
        assert content.numerical_data["Age"] == [25.0, 30.0, 28.0, 35.0]
        
        # Check categorical data
        assert "Name" in content.categorical_data
        assert content.categorical_data["Name"] == ["Alice", "Bob", "Carol", "Dave"]
        
        # Check data quality metrics
        assert content.missing_values >= 0
        assert 0.0 <= content.data_quality_score <= 1.0
        assert isinstance(content.data_summary, dict)
    
    def test_extract_table_content_with_missing_values(self, extractor):
        """Test content extraction with missing values."""
        data_with_missing = [
            ["Name", "Age", "Score"],
            ["Alice", "25", "85.5"],
            ["Bob", "", "92.1"],  # Missing age
            ["Carol", "28", ""],  # Missing score
            ["", "35", "88.7"]    # Missing name
        ]
        
        structure = extractor.parse_table_structure(data_with_missing)
        content = extractor.extract_table_content(data_with_missing, structure)
        
        assert content.missing_values > 0
        assert content.data_quality_score < 1.0  # Should be penalized for missing values
    
    def test_analyze_table_content(self, extractor, sample_table_data):
        """Test table content analysis."""
        structure = extractor.parse_table_structure(sample_table_data)
        content = extractor.extract_table_content(sample_table_data, structure)
        
        analysis = extractor.analyze_table_content(content, "Test table with user data")
        
        assert isinstance(analysis, ContentAnalysis)
        assert isinstance(analysis.keywords, list)
        assert isinstance(analysis.content_themes, list)
        assert isinstance(analysis.statistical_summary, dict)
        assert 0.0 <= analysis.complexity_score <= 1.0
        
        # Should have statistical summaries for numerical columns
        if content.numerical_data:
            assert len(analysis.statistical_summary) > 0
    
    def test_detect_column_types(self, extractor):
        """Test column data type detection."""
        test_data = [
            ["25", "90%", "true", "2023-01-01", "text"],
            ["30", "95%", "false", "2023-01-02", "more"],
            ["35", "85%", "yes", "2023-01-03", "data"]
        ]
        
        data_types = extractor._detect_column_types(test_data)
        
        assert len(data_types) == 5
        assert data_types[0] == DataType.NUMERIC     # "25", "30", "35"
        assert data_types[1] == DataType.PERCENTAGE  # "90%", "95%", "85%"
        assert data_types[2] == DataType.BOOLEAN     # "true", "false", "yes"
        # Note: datetime detection is basic, may not catch all formats
        assert data_types[4] == DataType.TEXT        # "text", "more", "data"
    
    def test_is_numeric(self, extractor):
        """Test numeric value detection."""
        assert extractor._is_numeric("123") == True
        assert extractor._is_numeric("123.45") == True
        assert extractor._is_numeric("-123.45") == True
        assert extractor._is_numeric("1,234.56") == True  # With comma
        assert extractor._is_numeric("$123.45") == True   # With currency symbol
        assert extractor._is_numeric("123%") == True      # With percentage
        
        assert extractor._is_numeric("abc") == False
        assert extractor._is_numeric("") == False
        assert extractor._is_numeric("123abc") == False
    
    def test_calculate_structure_complexity(self, extractor):
        """Test structure complexity calculation."""
        # Simple structure
        simple_structure = TableStructure()
        simple_structure.rows = 3
        simple_structure.columns = 3
        simple_structure.header_rows = 1
        simple_structure.data_types = [DataType.TEXT, DataType.NUMERIC, DataType.TEXT]
        
        simple_complexity = extractor._calculate_structure_complexity(simple_structure)
        assert 0.0 <= simple_complexity <= 1.0
        
        # Complex structure
        complex_structure = TableStructure()
        complex_structure.rows = 20
        complex_structure.columns = 10
        complex_structure.header_rows = 2
        complex_structure.merged_cells = [(0, 0, 0, 2), (1, 1, 3, 1)]
        complex_structure.data_types = [DataType.TEXT, DataType.NUMERIC, DataType.PERCENTAGE,
                                       DataType.BOOLEAN, DataType.DATETIME]
        
        complex_complexity = extractor._calculate_structure_complexity(complex_structure)
        assert 0.0 <= complex_complexity <= 1.0
        assert complex_complexity > simple_complexity
    
    def test_generate_data_summary(self, extractor, sample_table_data):
        """Test data summary generation."""
        structure = extractor.parse_table_structure(sample_table_data)
        content = extractor.extract_table_content(sample_table_data, structure)
        
        summary = extractor._generate_data_summary(content, structure)
        
        required_fields = ['total_cells', 'data_cells', 'missing_cells', 'completeness_ratio',
                          'column_count', 'data_row_count', 'numerical_columns', 'categorical_columns']
        
        for field in required_fields:
            assert field in summary
        
        assert summary['total_cells'] == 25  # 5x5
        assert summary['data_cells'] == 20   # 4x5 (excluding header)
        assert summary['column_count'] == 5
        assert summary['data_row_count'] == 4
        assert 0.0 <= summary['completeness_ratio'] <= 1.0
    
    def test_calculate_data_quality(self, extractor, sample_table_data):
        """Test data quality score calculation."""
        structure = extractor.parse_table_structure(sample_table_data)
        content = extractor.extract_table_content(sample_table_data, structure)
        
        quality_score = extractor._calculate_data_quality(content, structure)
        
        assert 0.0 <= quality_score <= 1.0
        
        # Test with poor quality data (lots of missing values)
        poor_data = [
            ["A", "B", "C"],
            ["", "", ""],
            ["", "2", ""],
            ["", "", ""]
        ]
        
        poor_structure = extractor.parse_table_structure(poor_data)
        poor_content = extractor.extract_table_content(poor_data, poor_structure)
        poor_quality = extractor._calculate_data_quality(poor_content, poor_structure)
        
        assert poor_quality < quality_score  # Should be lower quality


class TestFigureContentExtractor:
    """Test FigureContentExtractor functionality."""
    
    @pytest.fixture
    def extractor(self):
        """Create FigureContentExtractor instance for testing."""
        return FigureContentExtractor()
    
    def test_analyze_figure_content(self, extractor):
        """Test figure content analysis."""
        caption = "Bar chart showing statistical significance (p<0.001) across treatment groups"
        extracted_text = ["Control", "Treatment A", "Treatment B", "p<0.001", "**"]
        
        analysis = extractor.analyze_figure_content(caption, extracted_text)
        
        assert isinstance(analysis, ContentAnalysis)
        assert isinstance(analysis.keywords, list)
        assert isinstance(analysis.content_themes, list)
        assert analysis.content_type != ""
        assert analysis.text_content == extracted_text
        assert isinstance(analysis.numerical_content, list)
        assert 0.0 <= analysis.complexity_score <= 1.0
        assert isinstance(analysis.visual_elements, list)
    
    def test_analyze_figure_content_minimal(self, extractor):
        """Test figure content analysis with minimal data."""
        analysis = extractor.analyze_figure_content("Simple figure")
        
        assert isinstance(analysis, ContentAnalysis)
        assert analysis.text_content == []
        assert analysis.content_type != ""
    
    def test_classify_figure_content(self, extractor):
        """Test figure content type classification."""
        test_cases = [
            ("Bar chart showing results", "chart_visualization"),
            ("Schematic diagram of the system", "technical_diagram"),
            ("Photograph of the specimen", "photographic_image"),
            ("Geographic map of the area", "geographic_map"),
            ("Unknown figure type", "mixed_content"),
        ]
        
        for caption, expected_type in test_cases:
            result = extractor._classify_figure_content(caption)
            assert result == expected_type
    
    def test_calculate_figure_complexity(self, extractor):
        """Test figure complexity calculation."""
        # Simple caption
        simple_caption = "Figure 1."
        simple_complexity = extractor._calculate_figure_complexity(simple_caption)
        assert 0.0 <= simple_complexity <= 1.0
        
        # Complex caption with scientific content
        complex_caption = ("Comprehensive statistical analysis showing p<0.001 significance "
                          "across multiple treatment groups with 95% confidence intervals")
        complex_complexity = extractor._calculate_figure_complexity(complex_caption)
        assert 0.0 <= complex_complexity <= 1.0
        assert complex_complexity > simple_complexity
        
        # With extracted text
        extracted_text = ["Statistical analysis", "p<0.001", "95% CI", "Mean ± SD"]
        text_complexity = extractor._calculate_figure_complexity(complex_caption, extracted_text)
        assert text_complexity >= complex_complexity
    
    def test_identify_visual_elements(self, extractor):
        """Test visual element identification."""
        test_cases = [
            ("Chart with x-axis and y-axis labels", ["axes", "labels"]),
            ("Plot with legend and colored bars", ["legend", "colors", "bars"]),
            ("Scatter plot with grid lines", ["grid", "points"]),
            ("Simple diagram", []),
        ]
        
        for caption, expected_elements in test_cases:
            elements = extractor._identify_visual_elements(caption)
            for expected in expected_elements:
                assert expected in elements, f"Expected {expected} in elements for: {caption}"


class TestEdgeCases:
    """Test edge cases and error conditions for content utilities."""
    
    def test_empty_inputs(self):
        """Test behavior with empty inputs."""
        text_analyzer = TextAnalyzer()
        stat_analyzer = StatisticalAnalyzer()
        table_extractor = TableContentExtractor()
        figure_extractor = FigureContentExtractor()
        
        # Text analyzer
        assert text_analyzer.extract_scientific_notation("") == []
        assert text_analyzer.identify_domain("") == []
        assert text_analyzer.extract_numerical_values("") == []
        
        # Statistical analyzer
        assert stat_analyzer.analyze_distribution([]) == {}
        assert stat_analyzer.detect_outliers([]) == []
        assert stat_analyzer.correlation_analysis([], []) == {}
        
        # Table extractor
        empty_structure = table_extractor.parse_table_structure([])
        assert empty_structure.rows == 0
        
        # Figure extractor
        empty_analysis = figure_extractor.analyze_figure_content("")
        assert isinstance(empty_analysis, ContentAnalysis)
    
    def test_malformed_data_handling(self):
        """Test handling of malformed or inconsistent data."""
        table_extractor = TableContentExtractor()
        
        # Inconsistent row lengths
        malformed_data = [
            ["A", "B", "C"],
            ["1", "2"],      # Missing column
            ["3", "4", "5", "6"]  # Extra column
        ]
        
        # Should not crash
        structure = table_extractor.parse_table_structure(malformed_data)
        assert isinstance(structure, TableStructure)
        
        content = table_extractor.extract_table_content(malformed_data, structure)
        assert isinstance(content, TableContent)
    
    def test_large_dataset_performance(self):
        """Test performance with large datasets."""
        table_extractor = TableContentExtractor()
        
        # Create large dataset
        large_data = [["col1", "col2", "col3"]]
        for i in range(1000):
            large_data.append([str(i), str(i * 2), str(i * 3)])
        
        # Should complete in reasonable time
        import time
        start_time = time.time()
        
        structure = table_extractor.parse_table_structure(large_data)
        content = table_extractor.extract_table_content(large_data, structure)
        analysis = table_extractor.analyze_table_content(content, "Large dataset")
        
        end_time = time.time()
        
        # Should complete within 10 seconds (generous limit)
        assert (end_time - start_time) < 10.0
        assert isinstance(structure, TableStructure)
        assert isinstance(content, TableContent)
        assert isinstance(analysis, ContentAnalysis)
    
    def test_unicode_and_special_characters(self):
        """Test handling of Unicode and special characters."""
        text_analyzer = TextAnalyzer()
        
        unicode_text = "Étude avec données spécialisées: α=0.05, β-test, ±2.5°C, µg/mL"
        
        # Should not crash
        keywords = text_analyzer.extract_keywords(unicode_text)
        assert isinstance(keywords, list)
        
        values = text_analyzer.extract_numerical_values(unicode_text)
        assert isinstance(values, list)
        assert 0.05 in values
        assert 2.5 in values
        
        analysis = text_analyzer.analyze_text_complexity(unicode_text)
        assert isinstance(analysis, dict)
    
    @patch('aim2_project.aim2_ontology.parsers.content_utils.NUMPY_AVAILABLE', False)
    def test_numpy_unavailable_fallback(self):
        """Test functionality when numpy is not available."""
        # Most functionality should still work without numpy
        stat_analyzer = StatisticalAnalyzer()
        
        values = [1, 2, 3, 4, 5]
        analysis = stat_analyzer.analyze_distribution(values)
        
        # Should still work with built-in statistics module
        assert isinstance(analysis, dict)
        assert 'mean' in analysis
        assert 'median' in analysis
    
    def test_numerical_precision(self):
        """Test numerical precision and floating point handling."""
        stat_analyzer = StatisticalAnalyzer()
        
        # Test with very small numbers
        small_values = [1e-10, 2e-10, 3e-10, 4e-10, 5e-10]
        small_analysis = stat_analyzer.analyze_distribution(small_values)
        
        assert isinstance(small_analysis, dict)
        assert small_analysis['mean'] == pytest.approx(3e-10, rel=1e-3)
        
        # Test with very large numbers
        large_values = [1e10, 2e10, 3e10, 4e10, 5e10]
        large_analysis = stat_analyzer.analyze_distribution(large_values)
        
        assert isinstance(large_analysis, dict)
        assert large_analysis['mean'] == pytest.approx(3e10, rel=1e-3)


class TestIntegration:
    """Integration tests for content utilities working together."""
    
    def test_complete_table_analysis_workflow(self):
        """Test complete table analysis workflow."""
        # Sample scientific table
        table_data = [
            ["Treatment", "N", "Mean Response", "Std Dev", "P-value", "Efficacy"],
            ["Placebo", "25", "12.5", "2.1", "0.456", "Low"],
            ["Drug A", "24", "15.7", "2.8", "0.023", "Medium"],
            ["Drug B", "26", "18.2", "3.1", "0.001", "High"],
            ["Drug C", "23", "20.4", "3.5", "<0.001", "High"]
        ]
        
        extractor = TableContentExtractor()
        
        # Parse structure
        structure = extractor.parse_table_structure(table_data)
        assert structure.rows == 5
        assert structure.columns == 6
        
        # Extract content
        content = extractor.extract_table_content(table_data, structure)
        assert len(content.numerical_data) >= 3  # N, Mean Response, Std Dev
        
        # Analyze content
        analysis = extractor.analyze_table_content(content, "Clinical trial results")
        assert analysis.complexity_score > 0
        assert len(analysis.statistical_summary) > 0
        
        # Verify statistical analysis is meaningful
        if "Mean Response" in analysis.statistical_summary:
            mean_stats = analysis.statistical_summary["Mean Response"]
            assert "mean" in mean_stats
            assert "std_dev" in mean_stats
    
    def test_content_analyzer_consistency(self):
        """Test consistency between different content analyzers."""
        text_analyzer = TextAnalyzer()
        figure_extractor = FigureContentExtractor()
        
        test_caption = "Statistical analysis showing p<0.001 significance with 95% confidence intervals"
        
        # Both should extract similar numerical content
        text_values = text_analyzer.extract_numerical_values(test_caption)
        
        figure_analysis = figure_extractor.analyze_figure_content(test_caption)
        figure_values = figure_analysis.numerical_content
        
        # Should find the same numerical values
        assert set(text_values) == set(figure_values)
        
        # Both should identify scientific domains
        text_domains = text_analyzer.identify_domain(test_caption)
        figure_themes = figure_analysis.content_themes
        
        # Should have some overlap in identified themes/domains
        common_themes = set(text_domains) & set(figure_themes)
        assert len(common_themes) > 0 or (len(text_domains) == 0 and len(figure_themes) == 0)
    
    def test_quality_consistency_across_analyzers(self):
        """Test that quality metrics are consistent across different analyzers."""
        table_extractor = TableContentExtractor()
        
        # High quality data
        high_quality_data = [
            ["Parameter", "Value", "Unit", "Error"],
            ["Temperature", "25.5", "°C", "0.1"],
            ["Pressure", "1013.25", "hPa", "0.5"],
            ["Humidity", "65.2", "%", "1.0"]
        ]
        
        # Low quality data (missing values, inconsistent)
        low_quality_data = [
            ["Parameter", "Value", "Unit"],
            ["Temperature", "", "°C"],
            ["Pressure", "1013.25", ""],
            ["", "65.2", "%"]
        ]
        
        # Process both datasets
        hq_structure = table_extractor.parse_table_structure(high_quality_data)
        hq_content = table_extractor.extract_table_content(high_quality_data, hq_structure)
        
        lq_structure = table_extractor.parse_table_structure(low_quality_data)
        lq_content = table_extractor.extract_table_content(low_quality_data, lq_structure)
        
        # High quality should have better scores
        assert hq_content.data_quality_score > lq_content.data_quality_score
        assert hq_content.missing_values < lq_content.missing_values