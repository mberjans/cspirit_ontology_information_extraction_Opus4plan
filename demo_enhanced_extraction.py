#!/usr/bin/env python3
"""
Demonstration of Enhanced Figure and Table Extraction

This script demonstrates the comprehensive metadata extraction capabilities
of the enhanced PDF and XML parsers, showcasing the unified data structures
and rich metadata information available for ontology extraction applications.
"""

import json
import sys
from pathlib import Path

# Add the project to path
sys.path.insert(0, str(Path(__file__).parent))

from aim2_project.aim2_ontology.parsers.metadata_framework import (
    FigureMetadata, TableMetadata, ExtractionSummary
)
from aim2_project.aim2_ontology.parsers.content_utils import (
    TableContentExtractor, FigureContentExtractor
)


def demonstrate_unified_return_format():
    """Demonstrate the unified return format for both parsers."""
    print("=" * 80)
    print("ENHANCED FIGURE AND TABLE EXTRACTION DEMONSTRATION")
    print("=" * 80)
    print()
    
    print("📋 UNIFIED RETURN FORMAT SPECIFICATION")
    print("-" * 40)
    print("""
The enhanced parsers now return a standardized format:

{
    "figures": [
        {
            "id": str,                    # Unique identifier
            "label": str,                 # Display label (e.g., "Figure 1")
            "caption": str,               # Full caption text
            "type": str,                  # Classified type (chart, diagram, etc.)
            "position": int,              # Position in document
            
            "metadata": {
                "context": {              # Contextual information
                    "section_context": str,
                    "page_number": int,
                    "cross_references": [str],
                    "document_position": float
                },
                "technical": {            # Technical details
                    "file_format": str,
                    "dimensions": tuple,
                    "resolution": int,
                    "color_info": str
                }
            },
            
            "content": {                  # Extracted content
                "content_type": str,
                "complexity_score": float,
                "visual_elements": [str],
                "text_content": [str],
                "keywords": [str]
            },
            
            "analysis": {                 # Content analysis
                "statistical_summary": dict,
                "content_themes": [str]
            },
            
            "quality": {                  # Quality metrics
                "extraction_confidence": float,
                "completeness_score": float,
                "parsing_accuracy": float,
                "overall_quality": float,
                "validation_status": str,
                "quality_issues": [str]
            }
        }
    ],
    
    "tables": [
        {
            "id": str,
            "label": str,
            "caption": str,
            "type": str,                  # Classified type (statistical, demographic, etc.)
            "position": int,
            
            "metadata": {
                "context": { ... },       # Same as figures
                "structure": {            # Table structure
                    "rows": int,
                    "columns": int,
                    "header_rows": int,
                    "column_headers": [str],
                    "data_types": [str],
                    "structure_complexity": float
                }
            },
            
            "content": {                  # Table data
                "structured_data": dict,  # Column-wise data
                "data_summary": dict,     # Statistical summaries
                "missing_values": int,
                "data_quality_score": float
            },
            
            "analysis": {                 # Content analysis
                "statistical_summary": dict,
                "content_themes": [str],
                "keywords": [str]
            },
            
            "quality": { ... }            # Same as figures
        }
    ],
    
    "extraction_summary": {              # Overall extraction statistics
        "total_figures": int,
        "total_tables": int,
        "figures_by_type": dict,
        "tables_by_type": dict,
        "average_quality_score": float,
        "extraction_time": float,
        "parsing_method": str,
        "processing_notes": [str]
    }
}
    """)
    
    print("\n🎯 KEY ENHANCEMENTS")
    print("-" * 40)
    enhancements = [
        "✅ Comprehensive metadata extraction with standardized structures",
        "✅ Advanced content analysis and quality assessment",
        "✅ Statistical analysis for numerical table data",
        "✅ Cross-parser compatibility with unified return format",
        "✅ Quality metrics and validation scoring",
        "✅ Content classification and complexity scoring",
        "✅ Cross-reference detection and context extraction",
        "✅ Technical details extraction (dimensions, format, etc.)",
        "✅ Flexible return formats (dictionaries or metadata objects)",
        "✅ Comprehensive error handling with fallback mechanisms"
    ]
    
    for enhancement in enhancements:
        print(f"  {enhancement}")
    
    print("\n📊 EXTRACTION OPTIONS")
    print("-" * 40)
    options = [
        "extract_table_data: bool = True        # Extract actual table content",
        "analyze_content: bool = True           # Perform content analysis",
        "include_quality_metrics: bool = True   # Include quality assessment",
        "confidence_threshold: float = 0.0     # Minimum confidence for inclusion",
        "return_metadata_objects: bool = False # Return objects instead of dicts"
    ]
    
    for option in options:
        print(f"  {option}")


def demonstrate_table_analysis():
    """Demonstrate comprehensive table analysis capabilities."""
    print("\n\n🔬 TABLE ANALYSIS DEMONSTRATION")
    print("=" * 50)
    
    # Create sample table data
    sample_table_data = [
        ["Treatment Group", "Sample Size", "Mean Response", "Std Deviation", "P-value", "95% CI"],
        ["Control", "25", "12.5", "2.1", "-", "11.2-13.8"],
        ["Treatment A", "24", "15.7", "2.8", "0.023", "14.5-16.9"],
        ["Treatment B", "26", "18.2", "3.1", "0.001", "16.8-19.6"],
        ["Treatment C", "23", "20.4", "3.5", "<0.001", "18.9-21.9"],
        ["High Dose", "22", "22.1", "4.2", "<0.001", "20.2-24.0"]
    ]
    
    # Analyze with enhanced extractors
    extractor = TableContentExtractor()
    
    print("Raw Table Data:")
    for i, row in enumerate(sample_table_data):
        print(f"  Row {i}: {row}")
    
    print("\n📐 STRUCTURE ANALYSIS")
    print("-" * 30)
    structure = extractor.parse_table_structure(sample_table_data)
    print(f"Dimensions: {structure.rows} rows × {structure.columns} columns")
    print(f"Header rows: {structure.header_rows}")
    print(f"Column headers: {structure.column_headers}")
    print(f"Data types: {[dt.value for dt in structure.data_types]}")
    print(f"Structure complexity: {structure.structure_complexity:.2f}")
    
    print("\n📊 CONTENT EXTRACTION")
    print("-" * 30)
    content = extractor.extract_table_content(sample_table_data, structure)
    print(f"Structured data columns: {len(content.structured_data)}")
    print(f"Numerical columns: {len(content.numerical_data)}")
    print(f"Missing values: {content.missing_values}")
    print(f"Data quality score: {content.data_quality_score:.2f}")
    
    # Show numerical data analysis
    if content.numerical_data:
        print("\nNumerical Data Analysis:")
        for col_name, values in content.numerical_data.items():
            print(f"  {col_name}: {values} (mean: {sum(values)/len(values):.2f})")
    
    print("\n🧮 STATISTICAL SUMMARY")
    print("-" * 30)
    analysis = extractor.analyze_table_content(content, "Clinical trial results table")
    print(f"Complexity score: {analysis.complexity_score:.2f}")
    print(f"Content themes: {analysis.content_themes}")
    
    if analysis.statistical_summary:
        print("Statistical Analysis:")
        for col, stats in analysis.statistical_summary.items():
            print(f"  {col}:")
            print(f"    Mean: {stats.get('mean', 'N/A'):.2f}")
            print(f"    Std Dev: {stats.get('std', 'N/A'):.2f}")
            print(f"    Range: {stats.get('min', 'N/A'):.1f} - {stats.get('max', 'N/A'):.1f}")
    
    print("\n💎 DATA SUMMARY")
    print("-" * 30)
    summary = content.data_summary
    print(f"Total cells: {summary['total_cells']}")
    print(f"Data cells: {summary['data_cells']}")
    print(f"Completeness ratio: {summary['completeness_ratio']:.2f}")
    print(f"Numerical columns: {summary['numerical_columns']}")
    print(f"Categorical columns: {summary['categorical_columns']}")


def demonstrate_quality_assessment():
    """Demonstrate quality assessment capabilities."""
    print("\n\n⭐ QUALITY ASSESSMENT DEMONSTRATION")
    print("=" * 50)
    
    # Create sample metadata for assessment
    from aim2_project.aim2_ontology.parsers.metadata_framework import (
        QualityAssessment, FigureType, TableType
    )
    
    assessor = QualityAssessment()
    
    # High-quality figure example
    high_quality_figure = FigureMetadata()
    high_quality_figure.id = "fig1"
    high_quality_figure.label = "Figure 1"
    high_quality_figure.caption = "Comprehensive analysis of treatment outcomes showing statistical significance across multiple patient groups with detailed error bars and confidence intervals."
    high_quality_figure.figure_type = FigureType.CHART
    high_quality_figure.context.section_context = "Results"
    high_quality_figure.context.cross_references = ["see Figure 1", "Figure 1 demonstrates", "as shown in Figure 1"]
    high_quality_figure.analysis.content_type = "statistical_visualization"
    high_quality_figure.analysis.keywords = ["treatment", "outcomes", "statistical", "significance", "confidence"]
    high_quality_figure.technical.file_format = "PNG"
    high_quality_figure.technical.dimensions = (800, 600)
    high_quality_figure.analysis.visual_elements = ["bars", "error_bars", "legend", "axes"]
    
    # Low-quality figure example
    low_quality_figure = FigureMetadata()
    low_quality_figure.id = "fig2"
    low_quality_figure.caption = "Chart"  # Minimal caption
    low_quality_figure.figure_type = FigureType.UNKNOWN
    
    print("HIGH-QUALITY FIGURE ASSESSMENT:")
    print("-" * 35)
    hq_quality = assessor.assess_figure_quality(high_quality_figure)
    print(f"Extraction confidence: {hq_quality.extraction_confidence:.2f}")
    print(f"Completeness score: {hq_quality.completeness_score:.2f}")
    print(f"Parsing accuracy: {hq_quality.parsing_accuracy:.2f}")
    print(f"Overall quality: {hq_quality.overall_quality():.2f}")
    print(f"Validation status: {hq_quality.validation_status}")
    if hq_quality.quality_issues:
        print(f"Quality issues: {hq_quality.quality_issues}")
    
    print("\nLOW-QUALITY FIGURE ASSESSMENT:")
    print("-" * 34)
    lq_quality = assessor.assess_figure_quality(low_quality_figure)
    print(f"Extraction confidence: {lq_quality.extraction_confidence:.2f}")
    print(f"Completeness score: {lq_quality.completeness_score:.2f}")
    print(f"Parsing accuracy: {lq_quality.parsing_accuracy:.2f}")
    print(f"Overall quality: {lq_quality.overall_quality():.2f}")
    print(f"Validation status: {lq_quality.validation_status}")
    print(f"Quality issues: {lq_quality.quality_issues}")
    
    # Generate extraction summary
    print("\nEXTRACTION SUMMARY:")
    print("-" * 20)
    figures = [high_quality_figure, low_quality_figure]
    tables = []
    summary = assessor.generate_extraction_summary(figures, tables, 0.25, "demonstration")
    
    print(f"Total items: {summary.total_figures + summary.total_tables}")
    print(f"Average quality: {summary.average_quality_score:.2f}")
    print(f"Extraction time: {summary.extraction_time:.3f}s")
    print(f"Figures by type: {summary.figures_by_type}")
    if summary.processing_notes:
        print(f"Processing notes: {summary.processing_notes}")


def demonstrate_usage_examples():
    """Show practical usage examples."""
    print("\n\n💻 USAGE EXAMPLES")
    print("=" * 30)
    
    examples = [
        {
            "title": "Basic Usage",
            "code": """
# Initialize parser
from aim2_project.aim2_ontology.parsers.pdf_parser import PDFParser
parser = PDFParser()

# Extract with default settings
results = parser.extract_figures_tables(pdf_content)

# Access results
for figure in results['figures']:
    print(f"Figure {figure['id']}: {figure['caption']}")
    print(f"  Type: {figure['type']}")
    print(f"  Quality: {figure['quality']['overall_quality']:.2f}")

for table in results['tables']:
    print(f"Table {table['id']}: {table['caption']}")
    if table['content']['structured_data']:
        print(f"  Columns: {list(table['content']['structured_data'].keys())}")
"""
        },
        {
            "title": "Advanced Usage with Options",
            "code": """
# Extract with comprehensive analysis
results = parser.extract_figures_tables(
    content,
    extract_table_data=True,           # Extract actual table content
    analyze_content=True,              # Perform content analysis
    include_quality_metrics=True,      # Include quality assessment
    confidence_threshold=0.7,          # Only high-confidence items
    return_metadata_objects=False      # Return dictionaries
)

# Filter high-quality items
high_quality_figures = [
    fig for fig in results['figures'] 
    if fig['quality']['overall_quality'] > 0.8
]

high_quality_tables = [
    table for table in results['tables']
    if table['quality']['overall_quality'] > 0.8
]
"""
        },
        {
            "title": "Working with Table Data",
            "code": """
# Access structured table data
for table in results['tables']:
    if table['content']['structured_data']:
        # Get column data
        for col_name, values in table['content']['structured_data'].items():
            print(f"Column '{col_name}': {values}")
        
        # Access statistical analysis
        stats = table['analysis']['statistical_summary']
        for col_name, analysis in stats.items():
            mean = analysis.get('mean', 0)
            std = analysis.get('std', 0)
            print(f"{col_name}: mean={mean:.2f}, std={std:.2f}")
"""
        },
        {
            "title": "Quality Filtering and Assessment",
            "code": """
# Filter by quality metrics
summary = results['extraction_summary']
print(f"Overall extraction quality: {summary['average_quality_score']:.2f}")

# Filter items by specific quality criteria
reliable_figures = [
    fig for fig in results['figures']
    if (fig['quality']['extraction_confidence'] > 0.8 and
        fig['quality']['completeness_score'] > 0.7)
]

# Check for quality issues
for table in results['tables']:
    if table['quality']['quality_issues']:
        print(f"Table {table['id']} has issues: {table['quality']['quality_issues']}")
"""
        },
        {
            "title": "Cross-Parser Compatibility",
            "code": """
# Both parsers return the same format
from aim2_project.aim2_ontology.parsers.xml_parser import XMLParser

pdf_parser = PDFParser()
xml_parser = XMLParser()

pdf_results = pdf_parser.extract_figures_tables(pdf_content)
xml_results = xml_parser.extract_figures_tables(xml_content)

# Same data structure for both
def process_results(results, source):
    print(f"Results from {source}:")
    print(f"  Figures: {len(results['figures'])}")
    print(f"  Tables: {len(results['tables'])}")
    print(f"  Avg Quality: {results['extraction_summary']['average_quality_score']:.2f}")

process_results(pdf_results, "PDF")
process_results(xml_results, "XML")
"""
        }
    ]
    
    for i, example in enumerate(examples, 1):
        print(f"\n{i}. {example['title']}")
        print("-" * (len(example['title']) + 3))
        print(example['code'])


def main():
    """Run the complete demonstration."""
    try:
        demonstrate_unified_return_format()
        demonstrate_table_analysis()
        demonstrate_quality_assessment()
        demonstrate_usage_examples()
        
        print("\n\n🎉 DEMONSTRATION COMPLETE")
        print("=" * 40)
        print("The enhanced figure and table extraction system is now ready for use!")
        print("Key benefits for ontology information extraction:")
        print("  • Rich metadata provides context for better entity extraction")
        print("  • Quality metrics help filter reliable information")
        print("  • Statistical analysis enables quantitative data extraction")
        print("  • Unified format simplifies downstream processing")
        print("  • Cross-reference detection links related information")
        
        return 0
        
    except Exception as e:
        print(f"Demonstration failed: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())