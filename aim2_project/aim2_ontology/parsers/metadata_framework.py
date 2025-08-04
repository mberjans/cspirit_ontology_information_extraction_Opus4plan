"""
Comprehensive Metadata Framework for Figure and Table Extraction

This module provides standardized metadata structures and content extraction utilities
for both PDF and XML parsers, enabling rich data extraction for ontology information
extraction applications.

Classes:
    - FigureMetadata: Standardized figure metadata structure
    - TableMetadata: Standardized table metadata structure
    - ContentExtractor: Unified content extraction utilities
    - QualityAssessment: Quality scoring and validation
    - ExtractionSummary: Overall extraction statistics

Usage:
    from aim2_project.aim2_ontology.parsers.metadata_framework import (
        FigureMetadata, TableMetadata, ContentExtractor, QualityAssessment
    )
"""

import logging
import re
import statistics
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple, Union
from enum import Enum


class FigureType(Enum):
    """Enumeration of figure types for classification."""
    CHART = "chart"
    DIAGRAM = "diagram"
    PHOTO = "photo"
    SCHEMATIC = "schematic"
    FLOWCHART = "flowchart"
    GRAPH = "graph"
    MAP = "map"
    ILLUSTRATION = "illustration"
    COMPOSITE = "composite"
    UNKNOWN = "unknown"


class TableType(Enum):
    """Enumeration of table types for classification."""
    DATA_SUMMARY = "data_summary"
    STATISTICAL = "statistical"
    DEMOGRAPHIC = "demographic"
    EXPERIMENTAL = "experimental"
    COMPARISON = "comparison"
    REFERENCE = "reference"
    METADATA = "metadata"
    RESULTS = "results"
    UNKNOWN = "unknown"


class DataType(Enum):
    """Enumeration of data types found in table columns."""
    NUMERIC = "numeric"
    TEXT = "text"
    CATEGORICAL = "categorical"
    DATETIME = "datetime"
    BOOLEAN = "boolean"
    PERCENTAGE = "percentage"
    CURRENCY = "currency"
    MIXED = "mixed"
    UNKNOWN = "unknown"


@dataclass
class QualityMetrics:
    """Quality assessment metrics for extracted content."""
    extraction_confidence: float = 0.0  # 0.0 to 1.0
    completeness_score: float = 0.0     # 0.0 to 1.0
    parsing_accuracy: float = 0.0       # 0.0 to 1.0
    validation_status: str = "pending"  # pending, passed, failed
    quality_issues: List[str] = field(default_factory=list)
    processing_time: float = 0.0        # seconds
    
    def overall_quality(self) -> float:
        """Calculate overall quality score."""
        return (self.extraction_confidence + self.completeness_score + self.parsing_accuracy) / 3.0


@dataclass
class ContextInfo:
    """Context information for figures and tables."""
    section_context: str = ""
    page_number: Optional[int] = None
    section_number: Optional[str] = None
    cross_references: List[str] = field(default_factory=list)
    related_text: str = ""
    preceding_paragraphs: List[str] = field(default_factory=list)
    following_paragraphs: List[str] = field(default_factory=list)
    document_position: float = 0.0  # Relative position in document (0.0 to 1.0)


@dataclass
class TechnicalDetails:
    """Technical details for figures and tables."""
    file_format: Optional[str] = None
    dimensions: Optional[Tuple[int, int]] = None  # (width, height) in pixels
    resolution: Optional[int] = None  # DPI
    color_info: Optional[str] = None  # color, grayscale, bw
    file_size: Optional[int] = None  # bytes
    encoding: Optional[str] = None
    compression: Optional[str] = None


@dataclass
class ContentAnalysis:
    """Content analysis results for figures and tables."""
    content_type: str = ""
    complexity_score: float = 0.0  # 0.0 to 1.0
    visual_elements: List[str] = field(default_factory=list)
    text_content: List[str] = field(default_factory=list)
    numerical_content: List[float] = field(default_factory=list)
    statistical_summary: Dict[str, Any] = field(default_factory=dict)
    content_themes: List[str] = field(default_factory=list)
    keywords: List[str] = field(default_factory=list)


@dataclass
class FigureMetadata:
    """Comprehensive metadata structure for figures."""
    # Basic Information
    id: str = ""
    label: str = ""
    caption: str = ""
    figure_type: FigureType = FigureType.UNKNOWN
    position: int = 0
    
    # Context Information
    context: ContextInfo = field(default_factory=ContextInfo)
    
    # Technical Details
    technical: TechnicalDetails = field(default_factory=TechnicalDetails)
    
    # Content Analysis
    analysis: ContentAnalysis = field(default_factory=ContentAnalysis)
    
    # Quality Metrics
    quality: QualityMetrics = field(default_factory=QualityMetrics)
    
    # Additional Metadata
    source_parser: str = ""
    extraction_method: str = ""
    timestamp: datetime = field(default_factory=datetime.now)
    raw_data: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "id": self.id,
            "label": self.label,
            "caption": self.caption,
            "type": self.figure_type.value,
            "position": self.position,
            "metadata": {
                "context": {
                    "section_context": self.context.section_context,
                    "page_number": self.context.page_number,
                    "section_number": self.context.section_number,
                    "cross_references": self.context.cross_references,
                    "related_text": self.context.related_text,
                    "document_position": self.context.document_position
                },
                "technical": {
                    "file_format": self.technical.file_format,
                    "dimensions": self.technical.dimensions,
                    "resolution": self.technical.resolution,
                    "color_info": self.technical.color_info,
                    "file_size": self.technical.file_size
                }
            },
            "content": {
                "content_type": self.analysis.content_type,
                "complexity_score": self.analysis.complexity_score,
                "visual_elements": self.analysis.visual_elements,
                "text_content": self.analysis.text_content,
                "keywords": self.analysis.keywords
            },
            "analysis": {
                "statistical_summary": self.analysis.statistical_summary,
                "content_themes": self.analysis.content_themes
            },
            "quality": {
                "extraction_confidence": self.quality.extraction_confidence,
                "completeness_score": self.quality.completeness_score,
                "parsing_accuracy": self.quality.parsing_accuracy,
                "validation_status": self.quality.validation_status,
                "overall_quality": self.quality.overall_quality(),
                "quality_issues": self.quality.quality_issues
            }
        }


@dataclass
class TableStructure:
    """Table structure information."""
    rows: int = 0
    columns: int = 0
    header_rows: int = 0
    header_columns: int = 0
    merged_cells: List[Tuple[int, int, int, int]] = field(default_factory=list)  # (start_row, start_col, end_row, end_col)
    column_headers: List[str] = field(default_factory=list)
    row_headers: List[str] = field(default_factory=list)
    data_types: List[DataType] = field(default_factory=list)
    structure_complexity: float = 0.0  # 0.0 to 1.0


@dataclass
class TableContent:
    """Table content data."""
    raw_data: List[List[str]] = field(default_factory=list)
    structured_data: Dict[str, List[Any]] = field(default_factory=dict)
    numerical_data: Dict[str, List[float]] = field(default_factory=dict)
    categorical_data: Dict[str, List[str]] = field(default_factory=dict)
    data_summary: Dict[str, Any] = field(default_factory=dict)
    missing_values: int = 0
    data_quality_score: float = 0.0  # 0.0 to 1.0


@dataclass
class TableMetadata:
    """Comprehensive metadata structure for tables."""
    # Basic Information
    id: str = ""
    label: str = ""
    caption: str = ""
    table_type: TableType = TableType.UNKNOWN
    position: int = 0
    
    # Context Information
    context: ContextInfo = field(default_factory=ContextInfo)
    
    # Table Structure
    structure: TableStructure = field(default_factory=TableStructure)
    
    # Table Content
    content: TableContent = field(default_factory=TableContent)
    
    # Technical Details
    technical: TechnicalDetails = field(default_factory=TechnicalDetails)
    
    # Content Analysis
    analysis: ContentAnalysis = field(default_factory=ContentAnalysis)
    
    # Quality Metrics
    quality: QualityMetrics = field(default_factory=QualityMetrics)
    
    # Additional Metadata
    source_parser: str = ""
    extraction_method: str = ""
    timestamp: datetime = field(default_factory=datetime.now)
    raw_data: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "id": self.id,
            "label": self.label,
            "caption": self.caption,
            "type": self.table_type.value,
            "position": self.position,
            "metadata": {
                "context": {
                    "section_context": self.context.section_context,
                    "page_number": self.context.page_number,
                    "section_number": self.context.section_number,
                    "cross_references": self.context.cross_references,
                    "related_text": self.context.related_text,
                    "document_position": self.context.document_position
                },
                "structure": {
                    "rows": self.structure.rows,
                    "columns": self.structure.columns,
                    "header_rows": self.structure.header_rows,
                    "column_headers": self.structure.column_headers,
                    "data_types": [dt.value for dt in self.structure.data_types],
                    "structure_complexity": self.structure.structure_complexity
                }
            },
            "content": {
                "structured_data": self.content.structured_data,
                "data_summary": self.content.data_summary,
                "missing_values": self.content.missing_values,
                "data_quality_score": self.content.data_quality_score
            },
            "analysis": {
                "statistical_summary": self.analysis.statistical_summary,
                "content_themes": self.analysis.content_themes,
                "keywords": self.analysis.keywords
            },
            "quality": {
                "extraction_confidence": self.quality.extraction_confidence,
                "completeness_score": self.quality.completeness_score,
                "parsing_accuracy": self.quality.parsing_accuracy,
                "validation_status": self.quality.validation_status,
                "overall_quality": self.quality.overall_quality(),
                "quality_issues": self.quality.quality_issues
            }
        }


@dataclass
class ExtractionSummary:
    """Overall extraction statistics and summary."""
    total_figures: int = 0
    total_tables: int = 0
    figures_by_type: Dict[str, int] = field(default_factory=dict)
    tables_by_type: Dict[str, int] = field(default_factory=dict)
    average_quality_score: float = 0.0
    extraction_time: float = 0.0
    parsing_method: str = ""
    document_characteristics: Dict[str, Any] = field(default_factory=dict)
    processing_notes: List[str] = field(default_factory=list)
    extraction_timestamp: datetime = field(default_factory=datetime.now)


class ContentExtractor:
    """Unified content extraction utilities for figures and tables."""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
        # Figure type patterns
        self.figure_type_patterns = {
            FigureType.CHART: [r'\b(chart|bar\s+chart|pie\s+chart|histogram)\b', r'\bgraph\b'],
            FigureType.DIAGRAM: [r'\b(diagram|schematic|flowchart|flow\s+chart)\b'],
            FigureType.PHOTO: [r'\b(photo|photograph|image|picture)\b'],
            FigureType.GRAPH: [r'\b(plot|scatter\s+plot|line\s+graph|xy\s+plot)\b'],
            FigureType.MAP: [r'\b(map|geographic|spatial)\b'],
            FigureType.ILLUSTRATION: [r'\b(illustration|drawing|sketch)\b']
        }
        
        # Table type patterns
        self.table_type_patterns = {
            TableType.STATISTICAL: [r'\b(mean|median|std|p-value|confidence|statistical)\b'],
            TableType.DEMOGRAPHIC: [r'\b(age|gender|demographics|population|baseline)\b'],
            TableType.EXPERIMENTAL: [r'\b(trial|experiment|treatment|control|intervention)\b'],
            TableType.COMPARISON: [r'\b(comparison|compare|versus|vs\.?|before|after)\b'],
            TableType.RESULTS: [r'\b(results|outcomes|findings|measurements)\b']
        }
    
    def classify_figure_type(self, caption: str, content: str = "") -> FigureType:
        """Classify figure type based on caption and content."""
        text = f"{caption} {content}".lower()
        
        for fig_type, patterns in self.figure_type_patterns.items():
            for pattern in patterns:
                if re.search(pattern, text, re.IGNORECASE):
                    return fig_type
        
        return FigureType.UNKNOWN
    
    def classify_table_type(self, caption: str, headers: List[str] = None, content: str = "") -> TableType:
        """Classify table type based on caption, headers, and content."""
        text = f"{caption} {content}".lower()
        if headers:
            text += " " + " ".join(headers).lower()
        
        for table_type, patterns in self.table_type_patterns.items():
            for pattern in patterns:
                if re.search(pattern, text, re.IGNORECASE):
                    return table_type
        
        return TableType.UNKNOWN
    
    def extract_keywords(self, text: str) -> List[str]:
        """Extract relevant keywords from text."""
        # Simple keyword extraction - can be enhanced with NLP libraries
        words = re.findall(r'\b[a-zA-Z]{3,}\b', text.lower())
        
        # Filter common words
        common_words = {'the', 'and', 'for', 'are', 'but', 'not', 'you', 'all', 'can', 'had', 'her', 'was', 'one', 'our', 'out', 'day', 'get', 'has', 'him', 'his', 'how', 'man', 'new', 'now', 'old', 'see', 'two', 'way', 'who', 'boy', 'did', 'its', 'let', 'put', 'say', 'she', 'too', 'use'}
        
        keywords = [word for word in set(words) if word not in common_words and len(word) > 3]
        return keywords[:10]  # Return top 10 keywords
    
    def analyze_numerical_data(self, data: List[List[Any]]) -> Dict[str, Any]:
        """Analyze numerical data in table content."""
        numerical_cols = []
        analysis = {}
        
        if not data or len(data) < 2:  # Need at least header + one data row
            return analysis
        
        # Skip header row and analyze columns
        for col_idx in range(len(data[0])):
            col_data = []
            for row_idx in range(1, len(data)):  # Skip header
                try:
                    cell_value = data[row_idx][col_idx]
                    # Try to convert to float
                    if isinstance(cell_value, (int, float)):
                        col_data.append(float(cell_value))
                    elif isinstance(cell_value, str):
                        # Remove common formatting
                        cleaned = re.sub(r'[,$%]', '', cell_value.strip())
                        if cleaned and cleaned.replace('.', '').replace('-', '').isdigit():
                            col_data.append(float(cleaned))
                except (ValueError, TypeError):
                    continue
            
            if col_data and len(col_data) >= 2:  # Need at least 2 values for statistics
                numerical_cols.append(col_idx)
                col_name = f"column_{col_idx}"
                if len(data) > 0 and col_idx < len(data[0]):
                    col_name = str(data[0][col_idx])  # Use header as column name
                
                analysis[col_name] = {
                    'count': len(col_data),
                    'mean': statistics.mean(col_data),
                    'median': statistics.median(col_data),
                    'min': min(col_data),
                    'max': max(col_data),
                    'std': statistics.stdev(col_data) if len(col_data) > 1 else 0.0
                }
        
        return analysis
    
    def detect_data_types(self, data: List[List[Any]]) -> List[DataType]:
        """Detect data types for each column in table data."""
        if not data or len(data) < 2:
            return []
        
        data_types = []
        
        for col_idx in range(len(data[0])):
            col_values = []
            for row_idx in range(1, len(data)):  # Skip header
                if row_idx < len(data) and col_idx < len(data[row_idx]):
                    col_values.append(data[row_idx][col_idx])
            
            if not col_values:
                data_types.append(DataType.UNKNOWN)
                continue
            
            # Analyze column values to determine type
            numeric_count = 0
            percentage_count = 0
            datetime_count = 0
            boolean_count = 0
            
            for value in col_values[:10]:  # Sample first 10 values
                if not value or str(value).strip() == '':
                    continue
                    
                value_str = str(value).strip()
                
                # Check for percentage
                if '%' in value_str:
                    percentage_count += 1
                    continue
                
                # Check for boolean
                if value_str.lower() in ['true', 'false', 'yes', 'no', '1', '0']:
                    boolean_count += 1
                    continue
                
                # Check for datetime patterns
                if re.match(r'\d{4}-\d{2}-\d{2}', value_str) or re.match(r'\d{1,2}/\d{1,2}/\d{4}', value_str):
                    datetime_count += 1
                    continue
                
                # Check for numeric
                try:
                    float(re.sub(r'[,$]', '', value_str))
                    numeric_count += 1
                except ValueError:
                    pass
            
            total_values = len([v for v in col_values[:10] if v and str(v).strip()])
            if total_values == 0:
                data_types.append(DataType.UNKNOWN)
            elif percentage_count / total_values > 0.5:
                data_types.append(DataType.PERCENTAGE)
            elif boolean_count / total_values > 0.5:
                data_types.append(DataType.BOOLEAN)
            elif datetime_count / total_values > 0.5:
                data_types.append(DataType.DATETIME)
            elif numeric_count / total_values > 0.5:
                data_types.append(DataType.NUMERIC)
            else:
                data_types.append(DataType.TEXT)
        
        return data_types
    
    def calculate_complexity_score(self, content: str, structure_info: Dict[str, Any] = None) -> float:
        """Calculate complexity score for content."""
        if not content:
            return 0.0
        
        score = 0.0
        
        # Text complexity factors
        word_count = len(content.split())
        score += min(word_count / 100.0, 0.3)  # Max 0.3 for word count
        
        # Technical terms
        technical_patterns = [
            r'\b\d+\.?\d*\s*[%°±∆]\b',  # Numbers with units/symbols
            r'\bp\s*[<>=]\s*\d+\.?\d*\b',  # p-values
            r'\b\d+\.?\d*\s*×\s*\d+\.?\d*\b',  # Multiplication notation
            r'\b[A-Z]{2,}\b',  # Acronyms
        ]
        
        for pattern in technical_patterns:
            matches = len(re.findall(pattern, content))
            score += min(matches / 10.0, 0.2)  # Max 0.2 for technical terms
        
        # Structure complexity (if provided)
        if structure_info:
            if 'rows' in structure_info and 'columns' in structure_info:
                cells = structure_info['rows'] * structure_info['columns']
                score += min(cells / 100.0, 0.3)  # Max 0.3 for table size
            
            if 'merged_cells' in structure_info:
                score += min(len(structure_info['merged_cells']) / 10.0, 0.2)  # Max 0.2 for merged cells
        
        return min(score, 1.0)


class QualityAssessment:
    """Quality scoring and validation for extracted content."""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
    
    def assess_figure_quality(self, figure_metadata: FigureMetadata) -> QualityMetrics:
        """Assess quality of figure extraction."""
        metrics = QualityMetrics()
        issues = []
        
        # Extraction confidence based on available data
        confidence_factors = []
        
        if figure_metadata.id:
            confidence_factors.append(0.2)
        else:
            issues.append("Missing figure ID")
        
        if figure_metadata.caption:
            confidence_factors.append(0.3)
            if len(figure_metadata.caption) > 20:  # Substantial caption
                confidence_factors.append(0.1)
        else:
            issues.append("Missing or empty caption")
        
        if figure_metadata.label:
            confidence_factors.append(0.2)
        
        if figure_metadata.figure_type != FigureType.UNKNOWN:
            confidence_factors.append(0.2)
        else:
            issues.append("Unknown figure type")
        
        metrics.extraction_confidence = sum(confidence_factors)
        
        # Completeness score
        completeness_factors = []
        total_fields = 8  # Key fields to check
        
        if figure_metadata.context.section_context:
            completeness_factors.append(1)
        if figure_metadata.context.cross_references:
            completeness_factors.append(1)
        if figure_metadata.analysis.content_type:
            completeness_factors.append(1)
        if figure_metadata.analysis.keywords:
            completeness_factors.append(1)
        if figure_metadata.technical.file_format:
            completeness_factors.append(1)
        if figure_metadata.technical.dimensions:
            completeness_factors.append(1)
        if figure_metadata.analysis.visual_elements:
            completeness_factors.append(1)
        if figure_metadata.analysis.complexity_score > 0:
            completeness_factors.append(1)
        
        metrics.completeness_score = len(completeness_factors) / total_fields
        
        # Parsing accuracy (simplified - based on data consistency)
        accuracy_score = 1.0
        if not figure_metadata.caption and figure_metadata.id:
            accuracy_score -= 0.2  # Should have caption if has ID
        if figure_metadata.position < 0:
            accuracy_score -= 0.1
            issues.append("Invalid position value")
        
        metrics.parsing_accuracy = max(0.0, accuracy_score)
        
        # Validation status
        if metrics.overall_quality() > 0.7:
            metrics.validation_status = "passed"
        elif metrics.overall_quality() > 0.4:
            metrics.validation_status = "partial"
        else:
            metrics.validation_status = "failed"
        
        metrics.quality_issues = issues
        return metrics
    
    def assess_table_quality(self, table_metadata: TableMetadata) -> QualityMetrics:
        """Assess quality of table extraction."""
        metrics = QualityMetrics()
        issues = []
        
        # Extraction confidence based on available data
        confidence_factors = []
        
        if table_metadata.id:
            confidence_factors.append(0.2)
        else:
            issues.append("Missing table ID")
        
        if table_metadata.caption:
            confidence_factors.append(0.3)
            if len(table_metadata.caption) > 20:  # Substantial caption
                confidence_factors.append(0.1)
        else:
            issues.append("Missing or empty caption")
        
        if table_metadata.structure.rows > 0 and table_metadata.structure.columns > 0:
            confidence_factors.append(0.3)
        else:
            issues.append("Invalid table structure")
        
        if table_metadata.content.raw_data:
            confidence_factors.append(0.1)
        
        metrics.extraction_confidence = sum(confidence_factors)
        
        # Completeness score
        completeness_factors = []
        total_fields = 10  # Key fields to check
        
        if table_metadata.structure.column_headers:
            completeness_factors.append(1)
        if table_metadata.structure.data_types:
            completeness_factors.append(1)
        if table_metadata.content.structured_data:
            completeness_factors.append(1)
        if table_metadata.content.data_summary:
            completeness_factors.append(1)
        if table_metadata.context.section_context:
            completeness_factors.append(1)
        if table_metadata.context.cross_references:
            completeness_factors.append(1)
        if table_metadata.analysis.statistical_summary:
            completeness_factors.append(1)
        if table_metadata.analysis.keywords:
            completeness_factors.append(1)
        if table_metadata.table_type != TableType.UNKNOWN:
            completeness_factors.append(1)
        if table_metadata.content.data_quality_score > 0:
            completeness_factors.append(1)
        
        metrics.completeness_score = len(completeness_factors) / total_fields
        
        # Parsing accuracy based on data consistency
        accuracy_score = 1.0
        
        # Check if structure matches content
        if table_metadata.content.raw_data:
            actual_rows = len(table_metadata.content.raw_data)
            actual_cols = len(table_metadata.content.raw_data[0]) if table_metadata.content.raw_data else 0
            
            if table_metadata.structure.rows != actual_rows:
                accuracy_score -= 0.2
                issues.append("Row count mismatch between structure and content")
            
            if table_metadata.structure.columns != actual_cols:
                accuracy_score -= 0.2
                issues.append("Column count mismatch between structure and content")
        
        if table_metadata.content.missing_values > (table_metadata.structure.rows * table_metadata.structure.columns * 0.5):
            accuracy_score -= 0.3
            issues.append("High proportion of missing values")
        
        metrics.parsing_accuracy = max(0.0, accuracy_score)
        
        # Validation status
        if metrics.overall_quality() > 0.7:
            metrics.validation_status = "passed"
        elif metrics.overall_quality() > 0.4:
            metrics.validation_status = "partial"
        else:
            metrics.validation_status = "failed"
        
        metrics.quality_issues = issues
        return metrics
    
    def generate_extraction_summary(
        self, 
        figures: List[FigureMetadata], 
        tables: List[TableMetadata],
        extraction_time: float = 0.0,
        parsing_method: str = ""
    ) -> ExtractionSummary:
        """Generate comprehensive extraction summary."""
        summary = ExtractionSummary()
        
        summary.total_figures = len(figures)
        summary.total_tables = len(tables)
        summary.extraction_time = extraction_time
        summary.parsing_method = parsing_method
        
        # Figures by type
        for figure in figures:
            fig_type = figure.figure_type.value
            summary.figures_by_type[fig_type] = summary.figures_by_type.get(fig_type, 0) + 1
        
        # Tables by type
        for table in tables:
            table_type = table.table_type.value
            summary.tables_by_type[table_type] = summary.tables_by_type.get(table_type, 0) + 1
        
        # Average quality score
        all_quality_scores = []
        for figure in figures:
            all_quality_scores.append(figure.quality.overall_quality())
        for table in tables:
            all_quality_scores.append(table.quality.overall_quality())
        
        if all_quality_scores:
            summary.average_quality_score = statistics.mean(all_quality_scores)
        
        # Processing notes
        summary.processing_notes = []
        if summary.total_figures == 0 and summary.total_tables == 0:
            summary.processing_notes.append("No figures or tables found")
        
        low_quality_count = sum(1 for score in all_quality_scores if score < 0.5)
        if low_quality_count > 0:
            summary.processing_notes.append(f"{low_quality_count} items with low quality scores")
        
        return summary