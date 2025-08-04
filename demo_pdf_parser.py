#!/usr/bin/env python3
"""
PDF Parser Demo Script

This script demonstrates the functionality of the PDFParser implementation
for the AIM2 ontology information extraction system.

Usage:
    python demo_pdf_parser.py
"""

import io
import logging
from aim2_project.aim2_ontology.parsers import PDFParser, auto_detect_parser, get_parser_for_format

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

def demonstrate_pdf_parser():
    """Demonstrate PDFParser functionality."""
    print("=== AIM2 PDF Parser Demo ===\n")
    
    # 1. Test parser creation and basic properties
    print("1. Creating PDFParser instance...")
    try:
        parser = PDFParser()
        print(f"   ✓ Parser created: {parser}")
        print(f"   ✓ Supported formats: {parser.get_supported_formats()}")
        print(f"   ✓ Available extraction methods: {parser._supported_methods}")
    except Exception as e:
        print(f"   ✗ Error creating parser: {e}")
        return
    
    # 2. Test format detection and auto-parser selection
    print("\n2. Testing format detection...")
    try:
        parser_class = get_parser_for_format("pdf")
        print(f"   ✓ Parser class for 'pdf': {parser_class}")
        
        # Test auto-detection (would work with actual file)
        auto_parser = auto_detect_parser(content="sample.pdf")
        print(f"   ✓ Auto-detected parser: {auto_parser}")
    except Exception as e:
        print(f"   ✗ Error in format detection: {e}")
    
    # 3. Test validation with sample PDF content
    print("\n3. Testing PDF validation...")
    try:
        # Simple PDF header for testing
        sample_pdf = b"%PDF-1.4\n1 0 obj\n<<\n/Type /Catalog\n>>\nendobj\n%%EOF"
        
        is_valid = parser.validate(sample_pdf)
        print(f"   ✓ PDF validation result: {is_valid}")
        
        # Test invalid content
        invalid_content = b"This is not a PDF"
        is_invalid = parser.validate(invalid_content)
        print(f"   ✓ Invalid content validation: {is_invalid}")
    except Exception as e:
        print(f"   ✗ Error in validation: {e}")
    
    # 4. Test parsing capabilities
    print("\n4. Testing parsing capabilities...")
    try:
        # Create a more complete mock PDF for testing
        mock_pdf_content = b"""%PDF-1.4
1 0 obj
<<
/Type /Catalog
/Pages 2 0 R
>>
endobj

2 0 obj
<<
/Type /Pages
/Kids [3 0 R]
/Count 1
>>
endobj

3 0 obj
<<
/Type /Page
/Parent 2 0 R
/MediaBox [0 0 612 792]
/Contents 4 0 R
>>
endobj

4 0 obj
<<
/Length 44
>>
stream
BT
/F1 12 Tf
72 720 Td
(Sample PDF Content) Tj
ET
endstream
endobj

xref
0 5
0000000000 65535 f 
0000000009 00000 n 
0000000074 00000 n 
0000000120 00000 n 
0000000179 00000 n 
trailer
<<
/Size 5
/Root 1 0 R
>>
startxref
274
%%EOF"""
        
        print("   ✓ Testing with mock PDF content...")
        
        # Test validation first
        if parser.validate(mock_pdf_content):
            print("   ✓ Mock PDF content validated successfully")
            
            # Test parsing (this would work with actual PDF libraries)
            print("   ℹ Parsing would require actual PDF libraries (pypdf/pdfplumber/PyMuPDF)")
            print("   ℹ The parser is designed to work with these libraries when available")
        else:
            print("   ℹ Mock PDF content validation failed (expected without full PDF structure)")
    except Exception as e:
        print(f"   ✗ Error in parsing test: {e}")
    
    # 5. Test method availability
    print("\n5. Testing method availability...")
    try:
        required_methods = [
            'parse', 'validate', 'get_supported_formats',
            'extract_text', 'extract_metadata', 'identify_sections',
            'extract_figures_tables', 'extract_references'
        ]
        
        for method in required_methods:
            if hasattr(parser, method):
                print(f"   ✓ Method '{method}' available")
            else:
                print(f"   ✗ Method '{method}' missing")
    except Exception as e:
        print(f"   ✗ Error checking methods: {e}")
    
    # 6. Test configuration options
    print("\n6. Testing configuration capabilities...")
    try:
        # Test option setting
        parser.set_options({
            'extraction_method': 'auto',
            'validate_on_parse': True
        })
        print("   ✓ Options set successfully")
        
        # Get current options
        current_options = parser.get_options()
        print(f"   ✓ Current options retrieved: extraction method = {current_options.get('extraction_method', 'not set')}")
    except Exception as e:
        print(f"   ✗ Error in configuration test: {e}")
    
    print("\n=== Demo completed ===")
    print("\nNotes:")
    print("- This PDF parser supports pypdf, pdfplumber, and PyMuPDF libraries")
    print("- Install these libraries for full functionality:")
    print("  pip install pypdf pdfplumber PyMuPDF")
    print("- The parser includes fallback mechanisms and intelligent method selection")
    print("- Scientific paper processing includes section identification and reference extraction")


if __name__ == "__main__":
    demonstrate_pdf_parser()