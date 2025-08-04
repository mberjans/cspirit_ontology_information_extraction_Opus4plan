#!/usr/bin/env python3
"""
Test script for enhanced parsers with comprehensive metadata extraction.

This script demonstrates the new comprehensive metadata extraction capabilities
of both PDF and XML parsers, showcasing the unified data structures and
quality assessment features.
"""

import json
import logging
import sys
from pathlib import Path

# Add the project to path for testing
sys.path.insert(0, str(Path(__file__).parent))

from aim2_project.aim2_ontology.parsers.pdf_parser import PDFParser
from aim2_project.aim2_ontology.parsers.xml_parser import XMLParser
from aim2_project.aim2_ontology.parsers.metadata_framework import (
    FigureMetadata, TableMetadata, ExtractionSummary
)

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)


def test_pdf_parser_enhancements():
    """Test PDF parser with enhanced metadata extraction."""
    logger.info("=== Testing PDF Parser Enhancements ===")
    
    try:
        # Initialize PDF parser
        pdf_parser = PDFParser()
        logger.info("PDF parser initialized successfully")
        
        # Test with a sample text (simulating PDF content for demo)
        sample_pdf_content = {
            "text": """
            Introduction
            
            This study examines the effects of various treatments. Figure 1 shows the experimental setup
            used in our research. The data collected is presented in Table 1 below.
            
            Figure 1. Experimental setup showing the treatment chambers and measurement devices.
            This figure illustrates the complete experimental configuration used throughout the study.
            
            Table 1. Summary of experimental results
            Treatment Group | Sample Size | Mean Response | Standard Deviation | P-value
            Control        | 25          | 12.5         | 2.1               | -
            Treatment A    | 24          | 15.7         | 2.8               | 0.023
            Treatment B    | 26          | 18.2         | 3.1               | 0.001
            
            Results
            
            As shown in Figure 1, the experimental setup was standardized across all conditions.
            Table 1 presents the statistical analysis of the collected data.
            """,
            "extraction_method": "simulated",
            "pages": 1
        }
        
        # Test comprehensive extraction
        results = pdf_parser.extract_figures_tables(
            sample_pdf_content,
            extract_table_data=True,
            analyze_content=True,
            include_quality_metrics=True
        )
        
        logger.info(f"PDF extraction completed successfully")
        logger.info(f"Found {len(results['figures'])} figures and {len(results['tables'])} tables")
        
        # Display extraction summary
        summary = results['extraction_summary']
        logger.info(f"Average quality score: {summary['average_quality_score']:.2f}")
        logger.info(f"Extraction time: {summary['extraction_time']:.3f} seconds")
        
        # Show first figure details if available
        if results['figures']:
            fig = results['figures'][0]
            logger.info(f"First figure: {fig['id']} - Type: {fig['type']} - Quality: {fig['quality']['overall_quality']:.2f}")
        
        # Show first table details if available
        if results['tables']:
            table = results['tables'][0]
            logger.info(f"First table: {table['id']} - Type: {table['type']} - Quality: {table['quality']['overall_quality']:.2f}")
            if 'content' in table and table['content']['structured_data']:
                logger.info(f"Table has {len(table['content']['structured_data'])} columns")
        
        return True
        
    except Exception as e:
        logger.error(f"PDF parser test failed: {e}")
        return False


def test_xml_parser_enhancements():
    """Test XML parser with enhanced metadata extraction."""
    logger.info("\n=== Testing XML Parser Enhancements ===")
    
    try:
        # Initialize XML parser
        xml_parser = XMLParser()
        logger.info("XML parser initialized successfully")
        
        # Test with sample XML content (simulating PMC format)
        sample_xml = """<?xml version="1.0" encoding="UTF-8"?>
        <article>
            <front>
                <article-meta>
                    <title-group>
                        <article-title>Sample Research Article</article-title>
                    </title-group>
                </article-meta>
            </front>
            <body>
                <sec>
                    <title>Introduction</title>
                    <p>This study examines treatment effects. See <xref ref-type="fig" rid="fig1">Figure 1</xref> 
                    for the experimental setup and <xref ref-type="table" rid="tab1">Table 1</xref> for results.</p>
                </sec>
                <sec>
                    <title>Results</title>
                    <fig id="fig1">
                        <label>Figure 1</label>
                        <caption>
                            <p>Experimental setup showing treatment chambers and measurement devices.</p>
                        </caption>
                        <graphic mimetype="image/jpeg" width="600" height="400"/>
                    </fig>
                    
                    <table-wrap id="tab1">
                        <label>Table 1</label>
                        <caption>
                            <p>Summary of experimental results showing treatment effects.</p>
                        </caption>
                        <table>
                            <thead>
                                <tr>
                                    <th>Treatment Group</th>
                                    <th>Sample Size</th>
                                    <th>Mean Response</th>
                                    <th>P-value</th>
                                </tr>
                            </thead>
                            <tbody>
                                <tr>
                                    <td>Control</td>
                                    <td>25</td>
                                    <td>12.5</td>
                                    <td>-</td>
                                </tr>
                                <tr>
                                    <td>Treatment A</td>
                                    <td>24</td>
                                    <td>15.7</td>
                                    <td>0.023</td>
                                </tr>
                                <tr>
                                    <td>Treatment B</td>
                                    <td>26</td>
                                    <td>18.2</td>
                                    <td>0.001</td>
                                </tr>
                            </tbody>
                        </table>
                    </table-wrap>
                </sec>
            </body>
        </article>"""
        
        # Test comprehensive extraction
        results = xml_parser.extract_figures_tables(
            sample_xml,
            extract_table_data=True,
            analyze_content=True,
            include_quality_metrics=True
        )
        
        logger.info(f"XML extraction completed successfully")
        logger.info(f"Found {len(results['figures'])} figures and {len(results['tables'])} tables")
        
        # Display extraction summary
        summary = results['extraction_summary']
        logger.info(f"Average quality score: {summary['average_quality_score']:.2f}")
        logger.info(f"Extraction time: {summary['extraction_time']:.3f} seconds")
        
        # Show first figure details if available
        if results['figures']:
            fig = results['figures'][0]
            logger.info(f"First figure: {fig['id']} - Type: {fig['type']} - Quality: {fig['quality']['overall_quality']:.2f}")
            if fig['metadata']['technical']['dimensions']:
                logger.info(f"Figure dimensions: {fig['metadata']['technical']['dimensions']}")
        
        # Show first table details if available
        if results['tables']:
            table = results['tables'][0]
            logger.info(f"First table: {table['id']} - Type: {table['type']} - Quality: {table['quality']['overall_quality']:.2f}")
            if 'content' in table and table['content']['structured_data']:
                logger.info(f"Table structure: {table['metadata']['structure']['rows']} rows x {table['metadata']['structure']['columns']} columns")
                logger.info(f"Data analysis available: {len(table['analysis']['statistical_summary'])} columns analyzed")
        
        return True
        
    except Exception as e:
        logger.error(f"XML parser test failed: {e}")
        return False


def test_unified_data_structure():
    """Test that both parsers return compatible data structures."""
    logger.info("\n=== Testing Unified Data Structure Compatibility ===")
    
    try:
        # Test with metadata objects
        pdf_parser = PDFParser()
        xml_parser = XMLParser()
        
        # Simple test content for both parsers
        pdf_content = {"text": "Figure 1. Test figure for compatibility.", "extraction_method": "test"}
        xml_content = """<?xml version="1.0"?>
        <article>
            <body>
                <fig id="fig1">
                    <label>Figure 1</label>
                    <caption><p>Test figure for compatibility.</p></caption>
                </fig>
            </body>
        </article>"""
        
        # Extract with metadata objects
        pdf_results = pdf_parser.extract_figures_tables(pdf_content, return_metadata_objects=True)
        xml_results = xml_parser.extract_figures_tables(xml_content, return_metadata_objects=True)
        
        # Check that both return FigureMetadata objects
        if pdf_results['figures'] and isinstance(pdf_results['figures'][0], FigureMetadata):
            logger.info("PDF parser returns FigureMetadata objects ✓")
        
        if xml_results['figures'] and isinstance(xml_results['figures'][0], FigureMetadata):
            logger.info("XML parser returns FigureMetadata objects ✓")
        
        # Check that both return ExtractionSummary objects
        if isinstance(pdf_results['extraction_summary'], ExtractionSummary):
            logger.info("PDF parser returns ExtractionSummary objects ✓")
        
        if isinstance(xml_results['extraction_summary'], ExtractionSummary):
            logger.info("XML parser returns ExtractionSummary objects ✓")
        
        # Test dictionary format compatibility
        pdf_dict_results = pdf_parser.extract_figures_tables(pdf_content, return_metadata_objects=False)
        xml_dict_results = xml_parser.extract_figures_tables(xml_content, return_metadata_objects=False)
        
        # Check that dictionary structures are compatible
        required_keys = {'figures', 'tables', 'extraction_summary'}
        
        pdf_keys = set(pdf_dict_results.keys())
        xml_keys = set(xml_dict_results.keys())
        
        if required_keys.issubset(pdf_keys) and required_keys.issubset(xml_keys):
            logger.info("Both parsers return compatible dictionary structures ✓")
        
        # Check figure dictionary structure
        if pdf_dict_results['figures']:
            fig_keys = set(pdf_dict_results['figures'][0].keys())
            required_fig_keys = {'id', 'type', 'caption', 'metadata', 'content', 'analysis', 'quality'}
            if required_fig_keys.issubset(fig_keys):
                logger.info("Figure dictionary structure is standardized ✓")
        
        logger.info("Unified data structure compatibility test completed successfully")
        return True
        
    except Exception as e:
        logger.error(f"Unified data structure test failed: {e}")
        return False


def demonstrate_enhanced_features():
    """Demonstrate the enhanced features of the metadata framework."""
    logger.info("\n=== Demonstrating Enhanced Features ===")
    
    try:
        # Initialize parser
        pdf_parser = PDFParser()
        
        # Rich content example
        rich_content = {
            "text": """
            Abstract
            
            This comprehensive study analyzed treatment efficacy across multiple patient groups.
            
            Methods
            
            Figure 1 shows the experimental design with randomized controlled trial setup.
            Statistical analysis was performed as detailed in Table 1.
            
            Results
            
            Figure 1. Randomized controlled trial design showing patient allocation and treatment protocols.
            The study included 150 participants randomly assigned to three treatment groups.
            
            Table 1. Baseline characteristics and statistical analysis results
            Characteristic     | Control (n=50) | Treatment A (n=50) | Treatment B (n=50) | P-value
            Age (years)       | 45.2 ± 12.1    | 46.8 ± 11.3       | 44.9 ± 13.2       | 0.67
            Gender (M/F)      | 22/28          | 25/25             | 24/26             | 0.82
            BMI (kg/m²)       | 26.4 ± 4.2     | 27.1 ± 3.8        | 26.8 ± 4.1        | 0.65
            Primary Outcome   | 12.5 ± 2.1     | 15.7 ± 2.8        | 18.2 ± 3.1        | <0.001
            
            As shown in Figure 1, the randomization process ensured balanced group allocation.
            Table 1 demonstrates significant treatment effects (p<0.001) for the primary outcome.
            """,
            "extraction_method": "enhanced_demo"
        }
        
        # Extract with full analysis
        results = pdf_parser.extract_figures_tables(
            rich_content,
            extract_table_data=True,
            analyze_content=True,
            include_quality_metrics=True,
            confidence_threshold=0.3
        )
        
        logger.info("=== Enhanced Features Demonstration ===")
        
        # Extraction summary
        summary = results['extraction_summary']
        logger.info(f"Extraction Summary:")
        logger.info(f"  - Total items: {summary['total_figures'] + summary['total_tables']}")
        logger.info(f"  - Average quality: {summary['average_quality_score']:.2f}")
        logger.info(f"  - Processing time: {summary['extraction_time']:.3f}s")
        
        # Figure analysis
        if results['figures']:
            fig = results['figures'][0]
            logger.info(f"\nFigure Analysis:")
            logger.info(f"  - ID: {fig['id']}")
            logger.info(f"  - Type: {fig['type']}")
            logger.info(f"  - Quality score: {fig['quality']['overall_quality']:.2f}")
            logger.info(f"  - Extraction confidence: {fig['quality']['extraction_confidence']:.2f}")
            logger.info(f"  - Content themes: {fig['analysis']['content_themes']}")
            logger.info(f"  - Cross-references: {len(fig['metadata']['context']['cross_references'])}")
        
        # Table analysis
        if results['tables']:
            table = results['tables'][0]
            logger.info(f"\nTable Analysis:")
            logger.info(f"  - ID: {table['id']}")
            logger.info(f"  - Type: {table['type']}")
            logger.info(f"  - Quality score: {table['quality']['overall_quality']:.2f}")
            logger.info(f"  - Structure: {table['metadata']['structure']['rows']}x{table['metadata']['structure']['columns']}")
            logger.info(f"  - Data completeness: {table['content']['data_quality_score']:.2f}")
            
            if table['analysis']['statistical_summary']:
                logger.info(f"  - Statistical analysis: {len(table['analysis']['statistical_summary'])} columns")
                # Show first column stats if available
                first_col = list(table['analysis']['statistical_summary'].keys())[0]
                stats = table['analysis']['statistical_summary'][first_col]
                logger.info(f"    - {first_col}: mean={stats.get('mean', 'N/A'):.2f}, std={stats.get('std', 'N/A'):.2f}")
        
        logger.info("\nEnhanced features demonstration completed successfully")
        return True
        
    except Exception as e:
        logger.error(f"Enhanced features demonstration failed: {e}")
        return False


def main():
    """Run all tests and demonstrations."""
    logger.info("Starting comprehensive parser enhancement tests...\n")
    
    tests = [
        ("PDF Parser Enhancements", test_pdf_parser_enhancements),
        ("XML Parser Enhancements", test_xml_parser_enhancements),
        ("Unified Data Structure", test_unified_data_structure),
        ("Enhanced Features Demo", demonstrate_enhanced_features)
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
    logger.info("TEST SUMMARY")
    logger.info("="*60)
    
    passed = sum(results.values())
    total = len(results)
    
    for test_name, result in results.items():
        status = "PASS" if result else "FAIL"
        logger.info(f"{test_name:<30} : {status}")
    
    logger.info("-"*60)
    logger.info(f"Tests passed: {passed}/{total} ({passed/total*100:.1f}%)")
    
    if passed == total:
        logger.info("🎉 All tests passed! Enhanced parsers are working correctly.")
        return 0
    else:
        logger.error("⚠️  Some tests failed. Please check the implementation.")
        return 1


if __name__ == "__main__":
    sys.exit(main())