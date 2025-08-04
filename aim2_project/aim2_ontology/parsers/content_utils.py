"""
Advanced Content Extraction Utilities

This module provides specialized utilities for extracting and analyzing content
from figures and tables in both PDF and XML documents.

Classes:
    - TableContentExtractor: Advanced table content parsing
    - FigureContentExtractor: Figure content analysis
    - TextAnalyzer: Text content analysis utilities
    - StatisticalAnalyzer: Statistical analysis for numerical data
"""

import logging
import re
import statistics
from typing import Any, Dict, List, Optional, Tuple, Union
from collections import Counter
import csv
import io

try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False

from .metadata_framework import (
    DataType, TableType, FigureType, TableStructure, TableContent,
    ContentAnalysis, TechnicalDetails
)


class TextAnalyzer:
    """Advanced text analysis utilities."""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
        # Scientific terminology patterns
        self.scientific_patterns = {
            'p_values': r'\bp\s*[<>=]\s*\d+\.?\d*\b',
            'confidence_intervals': r'\b\d+\.?\d*%?\s*CI\b',
            'measurements': r'\b\d+\.?\d*\s*[µμ]?[gmkMGT]?[lLgGmMsShHzZ]?\b',
            'percentages': r'\b\d+\.?\d*\s*%\b',
            'ranges': r'\b\d+\.?\d*\s*[-–—]\s*\d+\.?\d*\b',
            'scientific_notation': r'\b\d+\.?\d*\s*[×x]\s*10\s*[\^]?\s*[-]?\d+\b',
            'units': r'\b\d+\.?\d*\s*[µμnpfakMGT]?[gmlsAVWJKNPa]/?[0-9]*\b'
        }
        
        # Medical/biological terms (simplified)
        self.domain_keywords = {
            'medical': ['patient', 'treatment', 'therapy', 'clinical', 'diagnosis', 'symptom', 'disease'],
            'biological': ['cell', 'protein', 'gene', 'dna', 'rna', 'enzyme', 'organism'],
            'statistical': ['mean', 'median', 'deviation', 'correlation', 'regression', 'significance'],
            'experimental': ['control', 'variable', 'hypothesis', 'trial', 'sample', 'group']
        }
    
    def extract_scientific_notation(self, text: str) -> List[str]:
        """Extract scientific notation expressions from text."""
        notations = []
        for pattern_name, pattern in self.scientific_patterns.items():
            matches = re.findall(pattern, text, re.IGNORECASE)
            notations.extend(matches)
        return notations
    
    def identify_domain(self, text: str) -> List[str]:
        """Identify scientific domains based on keywords."""
        text_lower = text.lower()
        domains = []
        
        for domain, keywords in self.domain_keywords.items():
            keyword_count = sum(1 for keyword in keywords if keyword in text_lower)
            if keyword_count >= 2:  # At least 2 keywords from domain
                domains.append(domain)
        
        return domains
    
    def extract_numerical_values(self, text: str) -> List[float]:
        """Extract numerical values from text."""
        # Pattern for various number formats
        number_pattern = r'[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?'
        matches = re.findall(number_pattern, text)
        
        values = []
        for match in matches:
            try:
                values.append(float(match))
            except ValueError:
                continue
        
        return values
    
    def analyze_text_complexity(self, text: str) -> Dict[str, Any]:
        """Analyze text complexity metrics."""
        words = text.split()
        sentences = re.split(r'[.!?]+', text)
        
        analysis = {
            'word_count': len(words),
            'sentence_count': len([s for s in sentences if s.strip()]),
            'average_word_length': statistics.mean([len(word) for word in words]) if words else 0,
            'average_sentence_length': len(words) / len(sentences) if sentences else 0,
            'unique_words': len(set(word.lower() for word in words)),
            'lexical_diversity': len(set(word.lower() for word in words)) / len(words) if words else 0
        }
        
        # Scientific terminology count
        scientific_count = 0
        for pattern in self.scientific_patterns.values():
            scientific_count += len(re.findall(pattern, text, re.IGNORECASE))
        
        analysis['scientific_terms'] = scientific_count
        analysis['scientific_density'] = scientific_count / len(words) if words else 0
        
        return analysis


class StatisticalAnalyzer:
    """Statistical analysis utilities for numerical data."""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
    
    def analyze_distribution(self, values: List[float]) -> Dict[str, Any]:
        """Analyze statistical distribution of values."""
        if not values or len(values) < 2:
            return {}
        
        analysis = {
            'count': len(values),
            'mean': statistics.mean(values),
            'median': statistics.median(values),
            'mode': statistics.mode(values) if len(set(values)) < len(values) else None,
            'min': min(values),
            'max': max(values),
            'range': max(values) - min(values),
            'std_dev': statistics.stdev(values) if len(values) > 1 else 0,
            'variance': statistics.variance(values) if len(values) > 1 else 0
        }
        
        # Quartiles
        sorted_values = sorted(values)
        n = len(sorted_values)
        analysis['q1'] = sorted_values[n // 4] if n >= 4 else None
        analysis['q3'] = sorted_values[3 * n // 4] if n >= 4 else None
        analysis['iqr'] = analysis['q3'] - analysis['q1'] if analysis['q1'] and analysis['q3'] else None
        
        # Skewness (simplified)
        if analysis['std_dev'] > 0:
            skew_sum = sum((x - analysis['mean']) ** 3 for x in values)
            analysis['skewness'] = skew_sum / (len(values) * analysis['std_dev'] ** 3)
        else:
            analysis['skewness'] = 0
        
        return analysis
    
    def detect_outliers(self, values: List[float], method: str = 'iqr') -> List[int]:
        """Detect outliers in numerical data."""
        if len(values) < 4:
            return []
        
        outlier_indices = []
        
        if method == 'iqr':
            sorted_values = sorted(values)
            n = len(sorted_values)
            q1 = sorted_values[n // 4]
            q3 = sorted_values[3 * n // 4]
            iqr = q3 - q1
            
            lower_bound = q1 - 1.5 * iqr
            upper_bound = q3 + 1.5 * iqr
            
            for i, value in enumerate(values):
                if value < lower_bound or value > upper_bound:
                    outlier_indices.append(i)
        
        elif method == 'zscore':
            mean_val = statistics.mean(values)
            std_val = statistics.stdev(values) if len(values) > 1 else 0
            
            if std_val > 0:
                for i, value in enumerate(values):
                    z_score = abs((value - mean_val) / std_val)
                    if z_score > 3:  # 3-sigma rule
                        outlier_indices.append(i)
        
        return outlier_indices
    
    def correlation_analysis(self, x_values: List[float], y_values: List[float]) -> Dict[str, Any]:
        """Analyze correlation between two numerical series."""
        if len(x_values) != len(y_values) or len(x_values) < 2:
            return {}
        
        # Pearson correlation coefficient (simplified)
        n = len(x_values)
        mean_x = statistics.mean(x_values)
        mean_y = statistics.mean(y_values)
        
        numerator = sum((x_values[i] - mean_x) * (y_values[i] - mean_y) for i in range(n))
        
        sum_sq_x = sum((x - mean_x) ** 2 for x in x_values)
        sum_sq_y = sum((y - mean_y) ** 2 for y in y_values)
        
        if sum_sq_x == 0 or sum_sq_y == 0:
            correlation = 0
        else:
            correlation = numerator / (sum_sq_x * sum_sq_y) ** 0.5
        
        return {
            'correlation_coefficient': correlation,
            'correlation_strength': self._interpret_correlation(correlation),
            'sample_size': n
        }
    
    def _interpret_correlation(self, r: float) -> str:
        """Interpret correlation coefficient strength."""
        abs_r = abs(r)
        if abs_r >= 0.9:
            return 'very_strong'
        elif abs_r >= 0.7:
            return 'strong'
        elif abs_r >= 0.5:
            return 'moderate'
        elif abs_r >= 0.3:
            return 'weak'
        else:
            return 'very_weak'


class TableContentExtractor:
    """Advanced table content extraction and analysis."""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.text_analyzer = TextAnalyzer()
        self.stat_analyzer = StatisticalAnalyzer()
    
    def parse_table_structure(self, raw_data: List[List[str]]) -> TableStructure:
        """Parse table structure from raw data."""
        structure = TableStructure()
        
        if not raw_data:
            return structure
        
        structure.rows = len(raw_data)
        structure.columns = len(raw_data[0]) if raw_data else 0
        
        # Detect header rows (simplified heuristic)
        if structure.rows > 1:
            # First row is likely header if it contains mostly text
            first_row = raw_data[0]
            text_cells = sum(1 for cell in first_row if not self._is_numeric(str(cell)))
            if text_cells / len(first_row) > 0.7:  # 70% text cells
                structure.header_rows = 1
                structure.column_headers = [str(cell).strip() for cell in first_row]
        
        # Detect data types for each column
        if structure.rows > structure.header_rows:
            data_rows = raw_data[structure.header_rows:]
            structure.data_types = self._detect_column_types(data_rows)
        
        # Calculate structure complexity
        structure.structure_complexity = self._calculate_structure_complexity(structure)
        
        return structure
    
    def extract_table_content(self, raw_data: List[List[str]], structure: TableStructure) -> TableContent:
        """Extract and analyze table content."""
        content = TableContent()
        content.raw_data = raw_data
        
        if not raw_data or structure.rows <= structure.header_rows:
            return content
        
        # Get data rows (excluding headers)
        data_rows = raw_data[structure.header_rows:]
        
        # Structure data by columns
        content.structured_data = {}
        content.numerical_data = {}
        content.categorical_data = {}
        
        for col_idx in range(structure.columns):
            col_name = (structure.column_headers[col_idx] 
                       if col_idx < len(structure.column_headers) 
                       else f"column_{col_idx}")
            
            # Extract column values
            col_values = []
            for row in data_rows:
                if col_idx < len(row):
                    value = str(row[col_idx]).strip()
                    col_values.append(value if value else None)
                else:
                    col_values.append(None)
            
            content.structured_data[col_name] = col_values
            
            # Extract numerical data if column is numeric
            if (col_idx < len(structure.data_types) and 
                structure.data_types[col_idx] in [DataType.NUMERIC, DataType.PERCENTAGE]):
                
                numeric_values = []
                for value in col_values:
                    if value and self._is_numeric(value):
                        try:
                            # Clean value (remove %, $, commas)
                            cleaned = re.sub(r'[%$,]', '', value)
                            numeric_values.append(float(cleaned))
                        except ValueError:
                            continue
                
                if numeric_values:
                    content.numerical_data[col_name] = numeric_values
            
            # Extract categorical data
            elif (col_idx < len(structure.data_types) and 
                  structure.data_types[col_idx] == DataType.TEXT):
                
                categorical_values = [v for v in col_values if v]
                if categorical_values:
                    content.categorical_data[col_name] = categorical_values
        
        # Count missing values
        total_cells = structure.rows * structure.columns
        content.missing_values = sum(
            1 for row in raw_data for cell in row 
            if not cell or str(cell).strip() == ''
        )
        
        # Generate data summary
        content.data_summary = self._generate_data_summary(content, structure)
        
        # Calculate data quality score
        content.data_quality_score = self._calculate_data_quality(content, structure)
        
        return content
    
    def analyze_table_content(self, content: TableContent, caption: str = "") -> ContentAnalysis:
        """Perform comprehensive content analysis."""
        analysis = ContentAnalysis()
        
        # Analyze text content (caption + all text data)
        all_text = caption + " "
        for col_name, values in content.categorical_data.items():
            all_text += f"{col_name} " + " ".join(str(v) for v in values if v) + " "
        
        analysis.keywords = self.text_analyzer.extract_numerical_values(all_text)[:10]
        analysis.content_themes = self.text_analyzer.identify_domain(all_text)
        
        # Statistical analysis
        if content.numerical_data:
            analysis.statistical_summary = {}
            for col_name, values in content.numerical_data.items():
                col_analysis = self.stat_analyzer.analyze_distribution(values)
                analysis.statistical_summary[col_name] = col_analysis
        
        # Complexity scoring
        complexity_factors = []
        
        # Data complexity
        if len(content.structured_data) > 5:  # Many columns
            complexity_factors.append(0.3)
        
        if any(len(values) > 50 for values in content.structured_data.values()):  # Many rows
            complexity_factors.append(0.2)
        
        # Statistical complexity
        if analysis.statistical_summary:
            has_outliers = any(
                'outliers' in summary and summary['outliers']
                for summary in analysis.statistical_summary.values()
            )
            if has_outliers:
                complexity_factors.append(0.2)
        
        # Text complexity
        text_complexity = self.text_analyzer.analyze_text_complexity(all_text)
        if text_complexity.get('scientific_density', 0) > 0.1:
            complexity_factors.append(0.3)
        
        analysis.complexity_score = min(sum(complexity_factors), 1.0)
        
        return analysis
    
    def _detect_column_types(self, data_rows: List[List[str]]) -> List[DataType]:
        """Detect data types for table columns."""
        if not data_rows or not data_rows[0]:
            return []
        
        column_count = len(data_rows[0])
        data_types = []
        
        for col_idx in range(column_count):
            col_values = []
            for row in data_rows:
                if col_idx < len(row) and row[col_idx]:
                    col_values.append(str(row[col_idx]).strip())
            
            if not col_values:
                data_types.append(DataType.UNKNOWN)
                continue
            
            # Analyze sample values
            numeric_count = sum(1 for v in col_values[:10] if self._is_numeric(v))
            percentage_count = sum(1 for v in col_values[:10] if '%' in v)
            boolean_count = sum(1 for v in col_values[:10] 
                              if v.lower() in ['true', 'false', 'yes', 'no', '1', '0'])
            
            sample_size = min(10, len(col_values))
            
            if percentage_count / sample_size > 0.6:
                data_types.append(DataType.PERCENTAGE)
            elif boolean_count / sample_size > 0.6:
                data_types.append(DataType.BOOLEAN)
            elif numeric_count / sample_size > 0.6:
                data_types.append(DataType.NUMERIC)
            else:
                data_types.append(DataType.TEXT)
        
        return data_types
    
    def _is_numeric(self, value: str) -> bool:
        """Check if a string value represents a number."""
        if not value:
            return False
        
        # Remove common formatting
        cleaned = re.sub(r'[,$%\s]', '', value.strip())
        
        try:
            float(cleaned)
            return True
        except ValueError:
            return False
    
    def _calculate_structure_complexity(self, structure: TableStructure) -> float:
        """Calculate structural complexity score."""
        complexity = 0.0
        
        # Size complexity
        cell_count = structure.rows * structure.columns
        complexity += min(cell_count / 100.0, 0.4)  # Max 0.4 for size
        
        # Header complexity
        if structure.header_rows > 1:
            complexity += 0.2
        
        # Merged cells complexity
        if structure.merged_cells:
            complexity += min(len(structure.merged_cells) / 10.0, 0.2)
        
        # Data type diversity
        if structure.data_types:
            unique_types = len(set(structure.data_types))
            complexity += min(unique_types / 5.0, 0.2)  # Max 0.2 for type diversity
        
        return min(complexity, 1.0)
    
    def _generate_data_summary(self, content: TableContent, structure: TableStructure) -> Dict[str, Any]:
        """Generate comprehensive data summary."""
        summary = {
            'total_cells': structure.rows * structure.columns,
            'data_cells': (structure.rows - structure.header_rows) * structure.columns,
            'missing_cells': content.missing_values,
            'completeness_ratio': 1.0 - (content.missing_values / (structure.rows * structure.columns)),
            'column_count': structure.columns,
            'data_row_count': structure.rows - structure.header_rows,
            'numerical_columns': len(content.numerical_data),
            'categorical_columns': len(content.categorical_data)
        }
        
        # Numerical summary
        if content.numerical_data:
            all_numeric_values = []
            for values in content.numerical_data.values():
                all_numeric_values.extend(values)
            
            if all_numeric_values:
                summary['numerical_summary'] = self.stat_analyzer.analyze_distribution(all_numeric_values)
        
        # Categorical summary
        if content.categorical_data:
            summary['categorical_summary'] = {}
            for col_name, values in content.categorical_data.items():
                unique_values = list(set(values))
                summary['categorical_summary'][col_name] = {
                    'unique_count': len(unique_values),
                    'most_common': Counter(values).most_common(3)
                }
        
        return summary
    
    def _calculate_data_quality(self, content: TableContent, structure: TableStructure) -> float:
        """Calculate data quality score."""
        quality_factors = []
        
        # Completeness
        if structure.rows * structure.columns > 0:
            completeness = 1.0 - (content.missing_values / (structure.rows * structure.columns))
            quality_factors.append(completeness * 0.4)  # 40% weight
        
        # Data consistency (simplified)
        consistency_score = 1.0
        for col_idx, data_type in enumerate(structure.data_types):
            col_name = (structure.column_headers[col_idx] 
                       if col_idx < len(structure.column_headers) 
                       else f"column_{col_idx}")
            
            if col_name in content.structured_data:
                values = content.structured_data[col_name]
                non_null_values = [v for v in values if v]
                
                if data_type == DataType.NUMERIC:
                    numeric_count = sum(1 for v in non_null_values if self._is_numeric(str(v)))
                    if non_null_values and numeric_count / len(non_null_values) < 0.8:
                        consistency_score -= 0.1
        
        quality_factors.append(consistency_score * 0.3)  # 30% weight
        
        # Structure quality
        structure_quality = 1.0
        if structure.header_rows == 0:
            structure_quality -= 0.2  # No headers detected
        if not structure.column_headers:
            structure_quality -= 0.1
        
        quality_factors.append(structure_quality * 0.3)  # 30% weight
        
        return min(sum(quality_factors), 1.0)


class FigureContentExtractor:
    """Advanced figure content analysis (for figures with extractable content)."""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.text_analyzer = TextAnalyzer()
    
    def analyze_figure_content(self, caption: str, extracted_text: List[str] = None) -> ContentAnalysis:
        """Analyze figure content based on available information."""
        analysis = ContentAnalysis()
        
        # Combine all text content
        all_text = caption
        if extracted_text:
            all_text += " " + " ".join(extracted_text)
        
        # Extract keywords and themes
        analysis.keywords = self.text_analyzer.extract_numerical_values(all_text)[:10]
        analysis.content_themes = self.text_analyzer.identify_domain(all_text)
        
        # Classify content type based on caption analysis
        analysis.content_type = self._classify_figure_content(caption)
        
        # Extract text content
        analysis.text_content = extracted_text or []
        
        # Extract numerical content from text
        analysis.numerical_content = self.text_analyzer.extract_numerical_values(all_text)
        
        # Calculate complexity
        analysis.complexity_score = self._calculate_figure_complexity(caption, extracted_text)
        
        # Identify visual elements (heuristic based on caption)
        analysis.visual_elements = self._identify_visual_elements(caption)
        
        return analysis
    
    def _classify_figure_content(self, caption: str) -> str:
        """Classify figure content type based on caption."""
        caption_lower = caption.lower()
        
        chart_keywords = ['chart', 'graph', 'plot', 'histogram', 'scatter', 'bar chart', 'pie chart']
        diagram_keywords = ['diagram', 'schematic', 'flowchart', 'workflow', 'process']
        image_keywords = ['image', 'photo', 'photograph', 'microscopy', 'scan']
        map_keywords = ['map', 'geographic', 'spatial', 'location']
        
        if any(keyword in caption_lower for keyword in chart_keywords):
            return 'chart_visualization'
        elif any(keyword in caption_lower for keyword in diagram_keywords):
            return 'technical_diagram'
        elif any(keyword in caption_lower for keyword in image_keywords):
            return 'photographic_image'
        elif any(keyword in caption_lower for keyword in map_keywords):
            return 'geographic_map'
        else:
            return 'mixed_content'
    
    def _calculate_figure_complexity(self, caption: str, extracted_text: List[str] = None) -> float:
        """Calculate figure complexity score."""
        complexity = 0.0
        
        # Caption complexity
        if caption:
            text_complexity = self.text_analyzer.analyze_text_complexity(caption)
            complexity += min(text_complexity.get('scientific_density', 0) * 2, 0.3)
            complexity += min(len(caption.split()) / 50.0, 0.2)
        
        # Extracted text complexity
        if extracted_text:
            total_text = " ".join(extracted_text)
            complexity += min(len(total_text.split()) / 100.0, 0.3)
            
            # Technical notation
            scientific_notation = self.text_analyzer.extract_scientific_notation(total_text)
            complexity += min(len(scientific_notation) / 10.0, 0.2)
        
        return min(complexity, 1.0)
    
    def _identify_visual_elements(self, caption: str) -> List[str]:
        """Identify visual elements based on caption analysis."""
        caption_lower = caption.lower()
        elements = []
        
        visual_indicators = {
            'axes': ['x-axis', 'y-axis', 'axis', 'axes'],
            'legend': ['legend', 'key', 'color code'],
            'grid': ['grid', 'gridlines'],
            'labels': ['label', 'annotation', 'text'],
            'symbols': ['symbol', 'marker', 'icon'],
            'colors': ['color', 'colored', 'red', 'blue', 'green', 'yellow'],
            'lines': ['line', 'curve', 'trend'],
            'bars': ['bar', 'column', 'histogram'],
            'points': ['point', 'dot', 'scatter'],
            'areas': ['area', 'region', 'zone', 'shaded']
        }
        
        for element_type, keywords in visual_indicators.items():
            if any(keyword in caption_lower for keyword in keywords):
                elements.append(element_type)
        
        return elements