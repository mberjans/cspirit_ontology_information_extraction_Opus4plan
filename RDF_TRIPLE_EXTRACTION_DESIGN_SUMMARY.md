# RDF Triple Extraction Design Summary

## Overview

This document provides a comprehensive design for adding RDF triple extraction functionality to the AIM2 ontology project's OWL parser. The design follows existing architectural patterns and integrates seamlessly with the current codebase.

## Design Components

### 1. RDFTriple Class

The `RDFTriple` class represents an RDF triple (subject-predicate-object) with comprehensive validation and metadata support.

#### Key Features:
- **Comprehensive Validation**: URI format validation, object type consistency, language/datatype validation
- **Multiple Object Types**: Support for URIs, literals, and blank nodes
- **Metadata Support**: Confidence scores, extraction methods, namespace prefixes, source context
- **Serialization**: JSON, dictionary, compact form, and N-Triples format support
- **Standard Python Methods**: Equality, hashing, string representation

#### Core Attributes:
```python
@dataclass
class RDFTriple:
    subject: str                                    # Subject URI or identifier
    predicate: str                                  # Predicate URI (relationship type)
    object: str                                     # Object URI, literal, or identifier
    object_type: str = "uri"                       # 'uri', 'literal', 'blank_node'
    language: Optional[str] = None                  # Language tag for literals
    datatype: Optional[str] = None                  # Datatype URI for typed literals
    source_graph: Optional[str] = None              # Named graph context
    confidence: float = 1.0                        # Confidence score (0.0-1.0)
    extraction_method: str = "unknown"             # Extraction method used
    namespace_prefixes: Dict[str, str] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
```

#### Validation Rules:
- **Subject/Predicate**: Must be valid URIs or namespace:identifier format
- **Object**: Validation depends on object_type (URI/literal/blank_node)
- **Language Tags**: RFC 5646 format (e.g., 'en', 'en-US')
- **Datatypes**: Must be valid URI format
- **Confidence**: Must be between 0.0 and 1.0
- **Literal Constraints**: Cannot have both language tag and datatype

### 2. Triple Extraction Architecture

The design uses a strategy pattern with multiple extractors to handle different parsing libraries.

#### Base Extractor Interface:
```python
class TripleExtractor(ABC):
    @abstractmethod
    def extract_triples(self, parsed_result: Dict[str, Any]) -> List[RDFTriple]:
        """Extract RDF triples from parsed ontology data."""
        pass

    @abstractmethod
    def get_namespace_prefixes(self, parsed_result: Dict[str, Any]) -> Dict[str, str]:
        """Extract namespace prefixes from parsed data."""
        pass
```

#### Concrete Extractors:

1. **RDFlibTripleExtractor**:
   - Extracts triples directly from `rdflib.Graph` objects
   - Handles all RDF object types with proper type detection
   - Extracts namespace bindings from graph

2. **OwlReady2TripleExtractor**:
   - Extracts triples from `owlready2.Ontology` objects
   - Provides OWL-specific knowledge and type inference
   - Handles class hierarchies, properties, and individuals

### 3. OWL Parser Integration

#### New `extract_triples()` Method:
```python
def extract_triples(self, parsed_result: Any, **kwargs) -> List[RDFTriple]:
    """
    Extract RDF triples from parsed OWL data.

    Args:
        parsed_result: Result from parse method
        **kwargs: Extraction options
            - extractor_preference: 'rdflib', 'owlready2', 'both'
            - filter_predicates: Set of predicate URIs to include
            - exclude_predicates: Set of predicate URIs to exclude
            - max_triples: Maximum number of triples to extract
            - confidence_threshold: Minimum confidence score

    Returns:
        List[RDFTriple]: Extracted RDF triples
    """
```

#### Method Features:
- **Multiple Extraction Strategies**: Can use rdflib, owlready2, or both
- **Flexible Filtering**: Include/exclude specific predicates
- **Confidence-based Filtering**: Filter by minimum confidence score
- **Duplicate Removal**: Intelligent deduplication across extraction methods
- **Error Recovery**: Graceful handling of extractor failures
- **Comprehensive Logging**: Detailed extraction statistics and error reporting

### 4. Parse Result Integration

The existing parse result structure is enhanced to include triple extraction data:

#### Enhanced Parse Result:
```python
{
    # Existing fields
    "rdf_graph": rdflib.Graph,
    "owl_ontology": owlready2.Ontology,
    "format": str,
    "parsed_at": float,
    "content_size": int,
    "options_used": dict,
    "validation": dict,

    # New triple extraction fields
    "triples": List[RDFTriple],           # Extracted triples
    "triple_extraction": {               # Extraction metadata
        "total_triples": int,
        "extraction_methods": List[str],
        "extraction_time": float,
        "extraction_errors": List[str],
        "namespace_prefixes": Dict[str, str],
        "extraction_options": dict
    }
}
```

### 5. Error Handling Strategy

The design uses the existing AIM2Exception hierarchy for consistent error handling:

#### Error Scenarios:
1. **Library Unavailable**: Warning logged, graceful degradation
2. **Invalid Parse Result**: Warning logged, return empty list
3. **Extraction Failures**: `OntologyException` with detailed context
4. **Validation Failures**: Invalid triples filtered out with logging
5. **Resource Constraints**: Configurable limits and streaming support

#### Error Code Usage:
- `AIM2_ONTO_PARS_E_002`: "RDF triple extraction failure"
- Context includes extraction method, error details, and available libraries

### 6. Configuration Options

#### Parser-Level Options:
```python
{
    "extract_triples_on_parse": False,      # Auto-extract during parsing
    "triple_extraction_method": "both",     # 'rdflib', 'owlready2', 'both'
    "max_triples_per_parse": None,         # Limit extraction size
    "triple_confidence_threshold": 0.0,     # Minimum confidence
    "include_inferred_triples": False,      # Include reasoning results
    "triple_batch_size": 1000,             # Batch processing size
}
```

#### Extraction-Specific Options:
```python
{
    "filter_predicates": None,              # Include only specific predicates
    "exclude_predicates": set(),           # Exclude specific predicates
    "preserve_blank_nodes": True,          # Keep blank node identifiers
    "normalize_literals": False,           # Normalize literal values
    "extract_metadata": True,              # Include extraction metadata
}
```

### 7. Performance Considerations

#### Memory Management:
- **Streaming Processing**: For large ontologies
- **Batch Extraction**: Configurable batch sizes
- **Lazy Loading**: Extract triples on demand
- **Memory Monitoring**: Track memory usage during extraction

#### Optimization Strategies:
- **Duplicate Detection**: Efficient hash-based deduplication
- **Selective Extraction**: Filter during extraction, not after
- **Caching**: Cache namespace prefixes and common URIs
- **Parallel Processing**: Multi-threaded extraction for large datasets

### 8. Unit Test Structure

#### Test Modules:

1. **`test_rdf_triple.py`**: Core RDFTriple class tests
   - Creation and validation scenarios
   - Serialization/deserialization
   - Equality and hashing
   - Format conversions (compact, N-Triples)
   - Edge cases and error conditions

2. **`test_triple_extractors.py`**: Extractor implementation tests
   - RDFlib extractor functionality
   - OwlReady2 extractor functionality
   - Namespace prefix extraction
   - Error handling and recovery

3. **`test_owl_parser_triple_integration.py`**: Integration tests
   - `extract_triples()` method behavior
   - Parse result integration
   - Configuration option handling
   - Filtering and limits

4. **`test_triple_extraction_performance.py`**: Performance tests
   - Large ontology handling
   - Memory usage patterns
   - Extraction speed benchmarks
   - Batch processing efficiency

#### Test Data Requirements:
- **Small Test Ontologies**: Basic RDF/OWL files for unit tests
- **Large Test Ontologies**: Performance testing datasets
- **Malformed Data**: Error handling validation
- **Multi-format Data**: Different RDF serializations

### 9. Implementation Roadmap

#### Phase 1: Core Implementation
1. Implement `RDFTriple` class with validation
2. Create basic extractor interfaces
3. Implement `RDFlibTripleExtractor`
4. Add `extract_triples()` method to OWLParser

#### Phase 2: Enhanced Features
1. Implement `OwlReady2TripleExtractor`
2. Add parse result integration
3. Implement filtering and configuration options
4. Add comprehensive error handling

#### Phase 3: Optimization & Testing
1. Performance optimization
2. Memory management improvements
3. Comprehensive test suite
4. Documentation and examples

### 10. Usage Examples

#### Basic Triple Extraction:
```python
# Create and use OWL parser
parser = OWLParser()
parsed_data = parser.parse(owl_content)
triples = parser.extract_triples(parsed_data)

print(f"Extracted {len(triples)} triples")
for triple in triples[:5]:  # Show first 5
    print(f"  {triple.to_compact_form()}")
```

#### Filtered Extraction:
```python
# Extract only class hierarchy triples
subclass_triples = parser.extract_triples(
    parsed_data,
    filter_predicates={
        "http://www.w3.org/2000/01/rdf-schema#subClassOf"
    }
)

# Extract high-confidence triples only
reliable_triples = parser.extract_triples(
    parsed_data,
    confidence_threshold=0.8,
    max_triples=1000
)
```

#### Serialization and Storage:
```python
# Convert triples to various formats
for triple in triples:
    print("Compact:", triple.to_compact_form())
    print("N-Triples:", triple.to_ntriples())
    print("JSON:", triple.to_json())

# Bulk serialization
triples_data = [t.to_dict() for t in triples]
with open("extracted_triples.json", "w") as f:
    json.dump(triples_data, f, indent=2)
```

### 11. Integration Benefits

#### For Existing Codebase:
- **Consistent Architecture**: Follows existing patterns and conventions
- **Backward Compatibility**: No breaking changes to existing functionality
- **Error Handling**: Uses established exception hierarchy
- **Configuration**: Integrates with existing configuration system

#### For Users:
- **Flexible Extraction**: Multiple strategies and filtering options
- **Rich Metadata**: Comprehensive triple metadata and provenance
- **Multiple Formats**: Various serialization options
- **Performance**: Optimized for large ontologies

#### For Development:
- **Extensible Design**: Easy to add new extractors
- **Testable**: Comprehensive test coverage
- **Maintainable**: Clear separation of concerns
- **Documented**: Extensive documentation and examples

## Conclusion

This design provides a robust, flexible, and well-integrated solution for RDF triple extraction in the AIM2 ontology project. It leverages existing architectural patterns, provides comprehensive functionality, and maintains high code quality standards. The modular design allows for incremental implementation and future enhancements while ensuring backward compatibility and performance optimization.
