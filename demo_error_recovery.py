#!/usr/bin/env python3
"""
Error Recovery System Demonstration

This script demonstrates the error recovery functionality implemented in the parser system.
It shows how different types of errors are classified, how recovery strategies are selected,
and how the system handles various error scenarios gracefully.

The demonstration covers:
- Error severity classification (WARNING, RECOVERABLE, FATAL)
- Recovery strategy selection (SKIP, DEFAULT, RETRY, REPLACE, ABORT, CONTINUE)
- Parser-specific recovery methods (OWL, CSV, JSON-LD)
- Error statistics tracking and reporting
- Configuration-driven recovery behavior

Usage:
    python demo_error_recovery.py
"""

import json
import tempfile
import logging
from datetime import datetime
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field
from enum import Enum
import sys
import os

# Add the project root to Python path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'aim2_project'))

# Configure logging for the demonstration
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('ErrorRecoveryDemo')


class ErrorSeverity(Enum):
    """Classification of error severity levels for recovery decisions."""
    WARNING = "warning"          # Minor issues that don't prevent processing
    RECOVERABLE = "recoverable"  # Errors that can be handled with fallback strategies  
    FATAL = "fatal"             # Critical errors that prevent further processing


class RecoveryStrategy(Enum):
    """Available error recovery strategies."""
    SKIP = "skip"               # Skip the problematic data/section
    DEFAULT = "default"         # Use default/fallback values
    RETRY = "retry"             # Retry with different parameters
    REPLACE = "replace"         # Replace with corrected data
    ABORT = "abort"             # Abort processing due to unrecoverable error
    CONTINUE = "continue"       # Continue processing despite the error


@dataclass
class ErrorContext:
    """Context information for error recovery decisions."""
    error: Exception
    severity: ErrorSeverity
    location: str
    recovery_strategy: Optional[RecoveryStrategy] = None
    attempted_recoveries: List[RecoveryStrategy] = field(default_factory=list)
    recovery_data: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class ErrorRecoveryStats:
    """Statistics tracking for error recovery operations."""
    total_errors: int = 0
    warnings: int = 0
    recoverable_errors: int = 0
    fatal_errors: int = 0
    successful_recoveries: int = 0
    failed_recoveries: int = 0
    recovery_strategies_used: Dict[str, int] = field(default_factory=dict)


class DemoParser:
    """Demonstration parser with error recovery capabilities."""
    
    def __init__(self, parser_type: str):
        self.parser_type = parser_type
        self.error_stats = ErrorRecoveryStats()
        self.config = {
            "error_recovery": {
                "max_retries": 3,
                "skip_malformed_rows": True,
                "use_default_values": True,
                "abort_on_fatal": True,
                "collect_statistics": True
            }
        }
        self.logger = logging.getLogger(f'DemoParser-{parser_type}')
    
    def _classify_error_severity(self, error: Exception, context: str = "") -> ErrorSeverity:
        """Classify error severity based on error type and context."""
        error_type = type(error).__name__
        
        # Fatal errors that prevent further processing
        if error_type in ["MemoryError", "SystemError", "KeyboardInterrupt"]:
            return ErrorSeverity.FATAL
        
        # Warning-level errors for minor issues
        if error_type in ["KeyError", "AttributeError"] and "optional" in str(error).lower():
            return ErrorSeverity.WARNING
        
        # Most parsing errors are recoverable
        if error_type in ["SyntaxError", "ValueError", "TypeError", "UnicodeDecodeError", 
                         "TimeoutError", "ConnectionError", "JSONDecodeError"]:
            return ErrorSeverity.RECOVERABLE
        
        # Default to recoverable for unknown errors
        return ErrorSeverity.RECOVERABLE
    
    def _select_recovery_strategy(self, error_context: ErrorContext) -> RecoveryStrategy:
        """Select appropriate recovery strategy based on error context."""
        error_type = type(error_context.error).__name__
        severity = error_context.severity
        attempted = error_context.attempted_recoveries
        
        # Fatal errors should abort (unless configured otherwise)
        if severity == ErrorSeverity.FATAL and self.config["error_recovery"]["abort_on_fatal"]:
            return RecoveryStrategy.ABORT
        
        # Warnings should continue processing
        if severity == ErrorSeverity.WARNING:
            return RecoveryStrategy.CONTINUE
        
        # Strategy selection for recoverable errors based on error type
        if error_type in ["SyntaxError", "ValueError"]:
            if RecoveryStrategy.SKIP not in attempted and self.config["error_recovery"]["skip_malformed_rows"]:
                return RecoveryStrategy.SKIP
            elif RecoveryStrategy.DEFAULT not in attempted and self.config["error_recovery"]["use_default_values"]:
                return RecoveryStrategy.DEFAULT
            else:
                return RecoveryStrategy.ABORT
        
        if error_type in ["KeyError", "AttributeError"]:
            if RecoveryStrategy.DEFAULT not in attempted and self.config["error_recovery"]["use_default_values"]:
                return RecoveryStrategy.DEFAULT
            elif RecoveryStrategy.SKIP not in attempted:
                return RecoveryStrategy.SKIP
            else:
                return RecoveryStrategy.ABORT
        
        if error_type in ["TimeoutError", "ConnectionError"]:
            retry_count = len([s for s in attempted if s == RecoveryStrategy.RETRY])
            if retry_count < self.config["error_recovery"]["max_retries"]:
                return RecoveryStrategy.RETRY
            else:
                return RecoveryStrategy.ABORT
        
        # Default strategy progression
        if RecoveryStrategy.SKIP not in attempted:
            return RecoveryStrategy.SKIP
        elif RecoveryStrategy.DEFAULT not in attempted:
            return RecoveryStrategy.DEFAULT
        else:
            return RecoveryStrategy.ABORT
    
    def _apply_recovery_strategy(self, error_context: ErrorContext, **kwargs) -> Any:
        """Apply the selected recovery strategy."""
        strategy = error_context.recovery_strategy
        self.logger.info(f"Applying recovery strategy '{strategy.value}' for {type(error_context.error).__name__}")
        
        if strategy == RecoveryStrategy.SKIP:
            self.logger.info(f"Skipping problematic content at {error_context.location}")
            return None
        
        elif strategy == RecoveryStrategy.DEFAULT:
            default_value = self._get_default_value(error_context)
            self.logger.info(f"Using default value: {default_value}")
            return default_value
        
        elif strategy == RecoveryStrategy.RETRY:
            retry_count = len([s for s in error_context.attempted_recoveries if s == RecoveryStrategy.RETRY])
            self.logger.info(f"Retrying operation (attempt {retry_count + 1})")
            return self._retry_operation(error_context, **kwargs)
        
        elif strategy == RecoveryStrategy.REPLACE:
            replacement = kwargs.get("replacement", {})
            self.logger.info(f"Replacing problematic content with: {replacement}")
            return replacement
        
        elif strategy == RecoveryStrategy.CONTINUE:
            self.logger.info("Continuing processing despite warning")
            return {"warning_handled": True}
        
        elif strategy == RecoveryStrategy.ABORT:
            self.logger.error(f"Aborting due to unrecoverable error: {error_context.error}")
            raise error_context.error
        
        return None
    
    def _get_default_value(self, error_context: ErrorContext) -> Any:
        """Get appropriate default value based on parser type and error context."""
        if self.parser_type == "OWL":
            return {
                "@context": {"owl": "http://www.w3.org/2002/07/owl#"},
                "@type": "owl:Ontology",
                "@id": "http://example.org/minimal"
            }
        elif self.parser_type == "CSV":
            return {"field": "", "value": ""}
        elif self.parser_type == "JSON-LD":
            return {
                "@context": {"@vocab": "http://schema.org/"},
                "@type": "Thing"
            }
        else:
            return {}
    
    def _retry_operation(self, error_context: ErrorContext, **kwargs) -> Any:
        """Simulate retry operation with different parameters."""
        original_content = kwargs.get("content", "")
        
        if self.parser_type == "OWL":
            # Try sanitizing XML content
            sanitized = original_content.replace("&", "&amp;").replace("<>", "")
            return {"status": "retried", "content": sanitized}
        
        elif self.parser_type == "CSV":
            # Try different encoding or delimiter
            return {"status": "retried", "encoding": "latin-1"}
        
        elif self.parser_type == "JSON-LD":
            # Try fixing common JSON issues
            fixed_content = original_content.replace(",}", "}").replace(",]", "]")
            return {"status": "retried", "content": fixed_content}
        
        return {"status": "retry_failed"}
    
    def _handle_parse_error(self, error: Exception, location: str = "", **kwargs) -> ErrorContext:
        """Comprehensive error handler that creates error context and applies recovery."""
        # Create error context
        severity = self._classify_error_severity(error, location)
        error_context = ErrorContext(
            error=error,
            severity=severity,
            location=location or "unknown location"
        )
        
        # Select recovery strategy
        error_context.recovery_strategy = self._select_recovery_strategy(error_context)
        
        # Apply recovery strategy
        try:
            recovery_result = self._apply_recovery_strategy(error_context, **kwargs)
            
            # Mark recovery attempt
            error_context.attempted_recoveries.append(error_context.recovery_strategy)
            
            # Update statistics
            self.error_stats.total_errors += 1
            if severity == ErrorSeverity.WARNING:
                self.error_stats.warnings += 1
            elif severity == ErrorSeverity.RECOVERABLE:
                self.error_stats.recoverable_errors += 1
            elif severity == ErrorSeverity.FATAL:
                self.error_stats.fatal_errors += 1
            
            # Track recovery strategy usage
            strategy_name = error_context.recovery_strategy.value
            self.error_stats.recovery_strategies_used[strategy_name] = \
                self.error_stats.recovery_strategies_used.get(strategy_name, 0) + 1
            
            if recovery_result is not None or error_context.recovery_strategy == RecoveryStrategy.SKIP:
                self.error_stats.successful_recoveries += 1
                self.logger.info(f"Successfully recovered from {type(error).__name__} using {error_context.recovery_strategy.value}")
            else:
                self.error_stats.failed_recoveries += 1
                self.logger.warning(f"Failed to recover from {type(error).__name__}")
            
            error_context.recovery_data["result"] = recovery_result
            
        except Exception as recovery_error:
            self.error_stats.failed_recoveries += 1
            self.logger.error(f"Recovery failed: {recovery_error}")
            error_context.recovery_data["recovery_error"] = str(recovery_error)
        
        return error_context
    
    def parse(self, content: str, **kwargs) -> Dict[str, Any]:
        """Main parse method that demonstrates error recovery."""
        results = {"status": "success", "data": [], "errors": []}
        
        try:
            # Simulate parser-specific processing
            if self.parser_type == "OWL":
                results["data"] = self._parse_owl(content)
            elif self.parser_type == "CSV":
                results["data"] = self._parse_csv(content)
            elif self.parser_type == "JSON-LD":
                results["data"] = self._parse_jsonld(content)
            
        except Exception as e:
            # Handle error through recovery system
            error_context = self._handle_parse_error(e, location="main parsing", content=content)
            results["errors"].append({
                "error": str(e),
                "severity": error_context.severity.value,
                "recovery_strategy": error_context.recovery_strategy.value,
                "recovery_successful": error_context.recovery_data.get("result") is not None
            })
            
            # Use recovered data if available
            if error_context.recovery_data.get("result"):
                results["data"] = [error_context.recovery_data["result"]]
                results["status"] = "recovered"
            else:
                results["status"] = "failed"
        
        return results
    
    def _parse_owl(self, content: str) -> List[Dict]:
        """Simulate OWL parsing with potential errors."""
        if "malformed" in content:
            raise SyntaxError("Malformed XML structure in OWL content")
        if "missing_namespace" in content:
            raise KeyError("Missing required namespace declaration")
        if "memory_intensive" in content:
            raise MemoryError("File too large to process")
        
        return [{"type": "owl:Class", "id": "http://example.org/Thing"}]
    
    def _parse_csv(self, content: str) -> List[Dict]:
        """Simulate CSV parsing with potential errors."""
        lines = content.strip().split('\n')
        
        if len(lines) < 2:
            raise ValueError("CSV must have at least header and one data row")
        
        if "field_mismatch" in content:
            raise ValueError("Field count mismatch in CSV row")
        
        if "encoding_error" in content:
            raise UnicodeDecodeError('utf-8', b'', 0, 1, 'invalid start byte')
        
        if "optional_field_missing" in content:
            raise KeyError("Optional field 'description' missing")
        
        # Parse normally
        header = lines[0].split(',')
        data = []
        for line in lines[1:]:
            if line.strip():
                fields = line.split(',')
                data.append(dict(zip(header, fields)))
        
        return data
    
    def _parse_jsonld(self, content: str) -> List[Dict]:
        """Simulate JSON-LD parsing with potential errors."""
        if "malformed_json" in content:
            raise json.JSONDecodeError("Expecting ',' delimiter", content, 10)
        
        if "missing_context" in content:
            raise KeyError("@context not found in JSON-LD document")
        
        if "timeout" in content:
            raise TimeoutError("Timeout while resolving remote @context")
        
        # Try to parse as JSON
        try:
            data = json.loads(content)
            return [data] if isinstance(data, dict) else data
        except json.JSONDecodeError:
            raise json.JSONDecodeError("Invalid JSON syntax", content, 0)
    
    def get_error_report(self) -> Dict[str, Any]:
        """Generate comprehensive error recovery report."""
        stats = self.error_stats
        return {
            "parser_type": self.parser_type,
            "summary": {
                "total_errors": stats.total_errors,
                "warnings": stats.warnings,
                "recoverable_errors": stats.recoverable_errors,
                "fatal_errors": stats.fatal_errors,
                "successful_recoveries": stats.successful_recoveries,
                "failed_recoveries": stats.failed_recoveries,
                "success_rate": (stats.successful_recoveries / stats.total_errors * 100) 
                              if stats.total_errors > 0 else 100
            },
            "recovery_strategies": stats.recovery_strategies_used
        }


def demonstrate_error_classification():
    """Demonstrate error severity classification."""
    print("\n" + "="*80)
    print("ERROR SEVERITY CLASSIFICATION DEMONSTRATION")
    print("="*80)
    
    parser = DemoParser("Demo")
    
    test_errors = [
        (SyntaxError("Invalid XML syntax"), "XML parsing"),
        (ValueError("Invalid field value"), "data validation"),
        (KeyError("Optional field missing"), "optional field processing"),
        (MemoryError("Out of memory"), "large file processing"),
        (TimeoutError("Connection timeout"), "remote resource"),
        (UnicodeDecodeError('utf-8', b'', 0, 1, 'invalid start byte'), "encoding detection")
    ]
    
    for error, location in test_errors:
        severity = parser._classify_error_severity(error, location)
        print(f"Error: {type(error).__name__:<20} | Severity: {severity.value:<12} | Location: {location}")


def demonstrate_recovery_strategy_selection():
    """Demonstrate recovery strategy selection."""
    print("\n" + "="*80)
    print("RECOVERY STRATEGY SELECTION DEMONSTRATION")
    print("="*80)
    
    parser = DemoParser("Demo")
    
    test_scenarios = [
        (SyntaxError("Parse error"), ErrorSeverity.RECOVERABLE, [], "parsing"),
        (ValueError("Invalid value"), ErrorSeverity.RECOVERABLE, [RecoveryStrategy.SKIP], "validation"),
        (KeyError("Missing optional field"), ErrorSeverity.WARNING, [], "field processing"),
        (TimeoutError("Connection timeout"), ErrorSeverity.RECOVERABLE, [], "network operation"),
        (MemoryError("Out of memory"), ErrorSeverity.FATAL, [], "memory allocation"),
    ]
    
    for error, severity, attempted, location in test_scenarios:
        context = ErrorContext(error=error, severity=severity, location=location, attempted_recoveries=attempted)
        strategy = parser._select_recovery_strategy(context)
        
        print(f"Error: {type(error).__name__:<20} | Severity: {severity.value:<12} | "
              f"Attempted: {[s.value for s in attempted]:<20} | Strategy: {strategy.value}")


def demonstrate_owl_parser_recovery():
    """Demonstrate OWL parser error recovery."""
    print("\n" + "="*80)
    print("OWL PARSER ERROR RECOVERY DEMONSTRATION")
    print("="*80)
    
    parser = DemoParser("OWL")
    
    test_cases = [
        ("malformed OWL content", "Testing malformed XML recovery"),
        ("missing_namespace in OWL", "Testing missing namespace recovery"),
        ("memory_intensive OWL file", "Testing memory error handling"),
        ("normal OWL content", "Testing normal processing")
    ]
    
    for content, description in test_cases:
        print(f"\n{description}:")
        print(f"Input: {content}")
        
        result = parser.parse(content)
        print(f"Status: {result['status']}")
        
        if result['errors']:
            for error_info in result['errors']:
                print(f"  Error: {error_info['error']}")
                print(f"  Severity: {error_info['severity']}")
                print(f"  Recovery: {error_info['recovery_strategy']}")
                print(f"  Success: {error_info['recovery_successful']}")
        
        if result['data']:
            print(f"  Data: {result['data']}")


def demonstrate_csv_parser_recovery():
    """Demonstrate CSV parser error recovery."""
    print("\n" + "="*80)
    print("CSV PARSER ERROR RECOVERY DEMONSTRATION")
    print("="*80)
    
    parser = DemoParser("CSV")
    
    test_cases = [
        ("name,age\nJohn,25", "Testing normal CSV processing"),
        ("field_mismatch in CSV", "Testing field count mismatch"),
        ("encoding_error in CSV", "Testing encoding error recovery"),
        ("optional_field_missing in CSV", "Testing missing optional field"),
    ]
    
    for content, description in test_cases:
        print(f"\n{description}:")
        print(f"Input: {content}")
        
        result = parser.parse(content)
        print(f"Status: {result['status']}")
        
        if result['errors']:
            for error_info in result['errors']:
                print(f"  Error: {error_info['error']}")
                print(f"  Severity: {error_info['severity']}")
                print(f"  Recovery: {error_info['recovery_strategy']}")
                print(f"  Success: {error_info['recovery_successful']}")
        
        if result['data']:
            print(f"  Data: {result['data']}")


def demonstrate_jsonld_parser_recovery():
    """Demonstrate JSON-LD parser error recovery."""
    print("\n" + "="*80)
    print("JSON-LD PARSER ERROR RECOVERY DEMONSTRATION")
    print("="*80)
    
    parser = DemoParser("JSON-LD")
    
    test_cases = [
        ('{"@type": "Person", "name": "John"}', "Testing normal JSON-LD processing"),
        ("malformed_json content", "Testing malformed JSON recovery"),
        ("missing_context in JSON-LD", "Testing missing @context recovery"),
        ("timeout while processing", "Testing timeout recovery"),
    ]
    
    for content, description in test_cases:
        print(f"\n{description}:")
        print(f"Input: {content}")
        
        result = parser.parse(content)
        print(f"Status: {result['status']}")
        
        if result['errors']:
            for error_info in result['errors']:
                print(f"  Error: {error_info['error']}")
                print(f"  Severity: {error_info['severity']}")
                print(f"  Recovery: {error_info['recovery_strategy']}")
                print(f"  Success: {error_info['recovery_successful']}")
        
        if result['data']:
            print(f"  Data: {result['data']}")


def demonstrate_error_statistics():
    """Demonstrate error statistics and reporting."""
    print("\n" + "="*80)
    print("ERROR STATISTICS AND REPORTING DEMONSTRATION")
    print("="*80)
    
    # Create parsers for different types
    parsers = {
        "OWL": DemoParser("OWL"),
        "CSV": DemoParser("CSV"), 
        "JSON-LD": DemoParser("JSON-LD")
    }
    
    # Process various content with errors
    test_data = {
        "OWL": ["malformed OWL", "missing_namespace", "normal OWL"],
        "CSV": ["field_mismatch", "encoding_error", "name,age\nJohn,25"],
        "JSON-LD": ["malformed_json", "missing_context", '{"@type": "Person"}']
    }
    
    for parser_type, parser in parsers.items():
        print(f"\nProcessing {parser_type} content:")
        
        for content in test_data[parser_type]:
            result = parser.parse(content)
            print(f"  {content[:20]:<20} -> {result['status']}")
        
        # Generate and display error report
        report = parser.get_error_report()
        print(f"\n{parser_type} Parser Error Report:")
        print(f"  Total Errors: {report['summary']['total_errors']}")
        print(f"  Warnings: {report['summary']['warnings']}")
        print(f"  Recoverable: {report['summary']['recoverable_errors']}")
        print(f"  Fatal: {report['summary']['fatal_errors']}")
        print(f"  Success Rate: {report['summary']['success_rate']:.1f}%")
        
        if report['recovery_strategies']:
            print(f"  Recovery Strategies Used:")
            for strategy, count in report['recovery_strategies'].items():
                print(f"    {strategy}: {count}")


def demonstrate_configuration_impact():
    """Demonstrate how configuration affects recovery behavior."""
    print("\n" + "="*80)
    print("CONFIGURATION IMPACT DEMONSTRATION")
    print("="*80)
    
    # Test with default configuration
    parser_default = DemoParser("CSV")
    print("Default Configuration:")
    print(f"  Max Retries: {parser_default.config['error_recovery']['max_retries']}")
    print(f"  Skip Malformed: {parser_default.config['error_recovery']['skip_malformed_rows']}")
    print(f"  Use Defaults: {parser_default.config['error_recovery']['use_default_values']}")
    
    result_default = parser_default.parse("field_mismatch in CSV")
    print(f"  Result: {result_default['status']}")
    if result_default['errors']:
        print(f"  Recovery Strategy: {result_default['errors'][0]['recovery_strategy']}")
    
    # Test with modified configuration
    parser_modified = DemoParser("CSV")
    parser_modified.config["error_recovery"]["skip_malformed_rows"] = False
    parser_modified.config["error_recovery"]["use_default_values"] = True
    
    print("\nModified Configuration (skip_malformed_rows=False):")
    result_modified = parser_modified.parse("field_mismatch in CSV")
    print(f"  Result: {result_modified['status']}") 
    if result_modified['errors']:
        print(f"  Recovery Strategy: {result_modified['errors'][0]['recovery_strategy']}")


def main():
    """Run all error recovery demonstrations."""
    print("ERROR RECOVERY SYSTEM DEMONSTRATION")
    print("="*80)
    print("This demonstration shows the comprehensive error recovery system")
    print("implemented in the parser classes. It covers error classification,")
    print("recovery strategy selection, parser-specific recovery methods,")
    print("statistics tracking, and configuration options.")
    
    try:
        demonstrate_error_classification()
        demonstrate_recovery_strategy_selection()
        demonstrate_owl_parser_recovery()
        demonstrate_csv_parser_recovery()
        demonstrate_jsonld_parser_recovery()
        demonstrate_error_statistics()
        demonstrate_configuration_impact()
        
        print("\n" + "="*80)
        print("DEMONSTRATION COMPLETED SUCCESSFULLY")
        print("="*80)
        print("The error recovery system provides comprehensive error handling")
        print("with configurable recovery strategies, detailed statistics,")
        print("and parser-specific recovery methods for robust data processing.")
        
    except Exception as e:
        logger.error(f"Demonstration failed: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())