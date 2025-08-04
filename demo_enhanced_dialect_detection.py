#!/usr/bin/env python3
"""
Demonstration of Enhanced CSV Dialect Detection

This script demonstrates the enhanced CSV dialect detection functionality
implemented for AIM2-012-08, including confidence scoring and robust fallback mechanisms.

Author: Claude Code
Date: 2025-08-04
Task: AIM2-012-08: Add CSV dialect detection
"""

import logging

from aim2_project.aim2_ontology.parsers import CSVParser

# Set up logging to see dialect detection details
logging.basicConfig(level=logging.DEBUG, format="%(levelname)s: %(message)s")


def demo_dialect_detection():
    """Demonstrate enhanced CSV dialect detection capabilities."""

    print("=" * 80)
    print("Enhanced CSV Dialect Detection Demonstration")
    print("=" * 80)

    # Initialize CSV parser
    parser = CSVParser()

    # Test cases with different CSV formats
    test_cases = [
        {
            "name": "Standard CSV with commas",
            "content": """id,name,definition,category
CHEM:001,Chemical,A chemical compound,Chemical
CHEM:002,Glucose,Simple sugar,Carbohydrate
CHEM:003,Protein,Large biomolecule,Macromolecule""",
        },
        {
            "name": "Tab-separated values (TSV)",
            "content": """id	name	definition	category
CHEM:001	Chemical	A chemical compound	Chemical
CHEM:002	Glucose	Simple sugar	Carbohydrate
CHEM:003	Protein	Large biomolecule	Macromolecule""",
        },
        {
            "name": "Pipe-separated values",
            "content": """id|name|definition|category
CHEM:001|Chemical|A chemical compound|Chemical
CHEM:002|Glucose|Simple sugar|Carbohydrate
CHEM:003|Protein|Large biomolecule|Macromolecule""",
        },
        {
            "name": "Semicolon-separated (European format)",
            "content": """id;name;definition;category
CHEM:001;Chemical;A chemical compound;Chemical
CHEM:002;Glucose;Simple sugar;Carbohydrate
CHEM:003;Protein;Large biomolecule;Macromolecule""",
        },
        {
            "name": "CSV with single quotes",
            "content": """id,name,definition,category
CHEM:001,'Chemical Compound','A basic unit of matter','Chemical'
CHEM:002,'D-Glucose','Simple sugar C6H12O6','Carbohydrate'
CHEM:003,'Protein Molecule','Large biomolecule','Macromolecule'""",
        },
        {
            "name": "CSV with complex quoted fields",
            "content": '''id,name,definition,properties
CHEM:001,"Chemical Compound","A pure substance, basic unit","molecular_weight:100.5;state:solid"
CHEM:002,"D-Glucose","Simple sugar, C6H12O6","formula:C6H12O6;sweet:true"
CHEM:003,"Protein","Large biomolecule, complex","amino_acids:20;folding:3D"''',
        },
        {
            "name": "Inconsistent CSV (mixed delimiters)",
            "content": """id,name;definition|category
CHEM:001,Chemical;A compound|Chemical
CHEM:002,Glucose;Sugar|Carbohydrate""",
        },
        {"name": "Single line CSV", "content": "id,name,definition,category"},
        {"name": "Empty content", "content": ""},
    ]

    for i, test_case in enumerate(test_cases, 1):
        print(f"\n{i}. Testing: {test_case['name']}")
        print("-" * 60)

        # Show first few lines of content
        lines = test_case["content"].split("\n")[:3]
        for line in lines:
            if line.strip():
                print(f"   {line}")
        if len(test_case["content"].split("\n")) > 3:
            print("   ...")

        try:
            # Detect dialect with enhanced method
            result = parser.detect_dialect(test_case["content"])

            print(f"\nDetection Results:")
            print(f"   Method: {result['method']}")
            print(f"   Confidence: {result['confidence']:.2f}")
            print(f"   Details: {result['details']}")

            if result["dialect"]:
                print(
                    f"   Delimiter: '{result['dialect'].delimiter}' (repr: {repr(result['dialect'].delimiter)})"
                )
                print(f"   Quote char: '{result['dialect'].quotechar}'")
                print(f"   Escape char: {result['dialect'].escapechar}")
            else:
                print(f"   Dialect: None (empty or invalid content)")

            # Confidence level interpretation
            confidence = result["confidence"]
            if confidence >= 0.8:
                level = "HIGH"
            elif confidence >= 0.5:
                level = "MEDIUM"
            elif confidence >= 0.3:
                level = "LOW"
            else:
                level = "VERY LOW"

            print(f"   Confidence Level: {level}")

        except Exception as e:
            print(f"   Error: {str(e)}")

    print("\n" + "=" * 80)
    print("Enhanced Features Summary:")
    print("=" * 80)
    print("✓ Extended delimiter support: comma, tab, pipe, semicolon, colon, space")
    print(
        "✓ Enhanced quote character detection: double quotes, single quotes, backticks"
    )
    print("✓ Escape character detection: backslash and other escape sequences")
    print("✓ Confidence scoring: 0.0 (no confidence) to 1.0 (high confidence)")
    print("✓ Multiple detection methods: csv.Sniffer, manual analysis, fallback")
    print("✓ Robust fallback mechanisms for difficult cases")
    print("✓ User preference support with higher confidence scoring")
    print("✓ Comprehensive logging for debugging and analysis")

    print("\nConfidence Level Guide:")
    print("  0.8-1.0: HIGH - Very reliable detection, proceed with confidence")
    print("  0.5-0.79: MEDIUM - Good detection, minor verification recommended")
    print("  0.3-0.49: LOW - Uncertain detection, manual review suggested")
    print(
        "  0.0-0.29: VERY LOW - Unreliable detection, fallback or manual input needed"
    )


def demo_user_specified_dialect():
    """Demonstrate user-specified dialect with higher confidence."""

    print("\n" + "=" * 80)
    print("User-Specified Dialect Detection")
    print("=" * 80)

    # Create parser with user-specified options
    user_options = {"delimiter": "|", "quotechar": "'", "has_headers": True}

    parser = CSVParser(options=user_options)

    custom_csv = """id|name|description
CHEM:001|Chemical|'A basic compound'
CHEM:002|Glucose|'Simple sugar molecule'"""

    print("User-specified options:")
    print(f"  delimiter: '{user_options['delimiter']}'")
    print(f"  quotechar: '{user_options['quotechar']}'")
    print(f"  has_headers: {user_options['has_headers']}")

    print(f"\nCSV content:")
    for line in custom_csv.split("\n"):
        print(f"  {line}")

    result = parser.detect_dialect(custom_csv)

    print(f"\nDetection Results:")
    print(f"  Method: {result['method']}")
    print(
        f"  Confidence: {result['confidence']:.2f} (higher due to user specification)"
    )
    print(f"  Details: {result['details']}")
    print(f"  Detected delimiter: '{result['dialect'].delimiter}'")
    print(f"  Detected quote char: '{result['dialect'].quotechar}'")


if __name__ == "__main__":
    demo_dialect_detection()
    demo_user_specified_dialect()

    print("\n" + "=" * 80)
    print("Demo completed successfully!")
    print("Enhanced CSV dialect detection is now available in AIM2 project.")
    print("=" * 80)
