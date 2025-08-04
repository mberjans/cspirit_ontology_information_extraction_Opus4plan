"""
Reference Parser Module

This module provides the main ReferenceParser class for parsing bibliographic references
and extracting structured metadata. The parser supports multiple citation formats
(APA, MLA, IEEE) with automatic format detection and confidence scoring.

The ReferenceParser is designed as a standalone class (not inheriting from AbstractDocumentParser)
and provides both single reference parsing and batch processing capabilities.

Key Features:
- Auto-detection of citation formats with confidence scores
- Format-specific parsing using specialized handlers
- Structured metadata extraction with confidence scores
- Batch processing of multiple references
- Graceful handling of malformed references
- Comprehensive logging and error handling
- Integration with the existing codebase patterns

Classes:
    ReferenceParser: Main parser class for bibliographic references
    ReferenceParseResult: Data class for parsing results
    BatchParseResult: Data class for batch parsing results

Usage:
    from aim2_project.aim2_ontology.parsers.reference_parser import ReferenceParser

    parser = ReferenceParser()
    result = parser.parse_reference("Smith, J. (2023). Title. Journal, 15(3), 123-145.")
    metadata = result.metadata

    # Batch processing
    references = ["citation1", "citation2", "citation3"]
    batch_result = parser.parse_references(references)

Dependencies:
    - typing: Type hints and annotations
    - dataclasses: Structured data classes
    - logging: Logging functionality
    - .reference_patterns: Pattern matching utilities
    - .citation_formats: Citation format handlers
    - ..exceptions: Project-specific exceptions
"""

import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

from .citation_formats import (
    BaseCitationHandler,
    CitationMetadata,
    get_handler_for_format,
    get_supported_formats,
)
from .reference_patterns import CitationDetector, detect_citation_format

logger = logging.getLogger(__name__)


class ParsingStrategy(Enum):
    """Enumeration of parsing strategies."""

    AUTO_DETECT = "auto_detect"
    FORCE_FORMAT = "force_format"
    TRY_ALL_FORMATS = "try_all_formats"


@dataclass
class ReferenceParseResult:
    """
    Result of parsing a single reference.

    Contains the extracted metadata, format detection information,
    and parsing details including errors and warnings.
    """

    metadata: CitationMetadata
    detected_format: str
    format_confidence: float
    parsing_strategy: ParsingStrategy
    parsing_errors: List[str] = field(default_factory=list)
    parsing_warnings: List[str] = field(default_factory=list)
    processing_time: Optional[float] = None

    @property
    def success(self) -> bool:
        """Check if parsing was successful (no errors)."""
        return len(self.parsing_errors) == 0

    @property
    def overall_confidence(self) -> float:
        """Get overall confidence combining format detection and metadata extraction."""
        if not self.metadata:
            return 0.0
        return (self.format_confidence + self.metadata.get_overall_confidence()) / 2.0

    def to_dict(self) -> Dict[str, Any]:
        """Convert result to dictionary representation."""
        return {
            "metadata": self.metadata.to_dict() if self.metadata else {},
            "detected_format": self.detected_format,
            "format_confidence": self.format_confidence,
            "parsing_strategy": self.parsing_strategy.value,
            "parsing_errors": self.parsing_errors,
            "parsing_warnings": self.parsing_warnings,
            "processing_time": self.processing_time,
            "success": self.success,
            "overall_confidence": self.overall_confidence,
        }


@dataclass
class BatchParseResult:
    """
    Result of parsing multiple references.

    Contains individual results, summary statistics, and batch-level information.
    """

    results: List[ReferenceParseResult]
    total_processed: int
    successful_parses: int
    failed_parses: int
    total_processing_time: Optional[float] = None

    @property
    def success_rate(self) -> float:
        """Calculate success rate as percentage."""
        if self.total_processed == 0:
            return 0.0
        return (self.successful_parses / self.total_processed) * 100.0

    @property
    def average_confidence(self) -> float:
        """Calculate average confidence across all successful parses."""
        successful_results = [r for r in self.results if r.success]
        if not successful_results:
            return 0.0
        return sum(r.overall_confidence for r in successful_results) / len(
            successful_results
        )

    def get_format_distribution(self) -> Dict[str, int]:
        """Get distribution of detected formats."""
        format_counts = {}
        for result in self.results:
            if result.success:
                format_name = result.detected_format
                format_counts[format_name] = format_counts.get(format_name, 0) + 1
        return format_counts

    def to_dict(self) -> Dict[str, Any]:
        """Convert batch result to dictionary representation."""
        return {
            "results": [r.to_dict() for r in self.results],
            "total_processed": self.total_processed,
            "successful_parses": self.successful_parses,
            "failed_parses": self.failed_parses,
            "success_rate": self.success_rate,
            "average_confidence": self.average_confidence,
            "format_distribution": self.get_format_distribution(),
            "total_processing_time": self.total_processing_time,
        }


class ReferenceParser:
    """
    Main parser class for bibliographic references.

    This standalone parser provides comprehensive functionality for parsing
    bibliographic references in multiple formats with automatic format
    detection and structured metadata extraction.

    Features:
    - Auto-detection of APA, MLA, IEEE, and other citation formats
    - Format-specific parsing with specialized handlers
    - Confidence scoring for all extracted fields
    - Batch processing capabilities
    - Graceful error handling and recovery
    - Comprehensive logging and debugging support
    """

    def __init__(
        self,
        default_strategy: ParsingStrategy = ParsingStrategy.AUTO_DETECT,
        confidence_threshold: float = 0.5,
        enable_fallback: bool = True,
    ):
        """
        Initialize the reference parser.

        Args:
            default_strategy: Default parsing strategy to use
            confidence_threshold: Minimum confidence threshold for format detection
            enable_fallback: Whether to try fallback parsing methods
        """
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        self.citation_detector = CitationDetector()
        self.default_strategy = default_strategy
        self.confidence_threshold = confidence_threshold
        self.enable_fallback = enable_fallback

        # Initialize format handlers
        self.handlers: Dict[str, BaseCitationHandler] = {}
        for format_name in get_supported_formats():
            try:
                self.handlers[format_name] = get_handler_for_format(format_name)
                self.logger.debug(f"Initialized {format_name} handler")
            except Exception as e:
                self.logger.error(f"Failed to initialize {format_name} handler: {e}")

        self.logger.info(
            f"ReferenceParser initialized with {len(self.handlers)} format handlers"
        )

    def parse_reference(
        self,
        reference_text: str,
        strategy: Optional[ParsingStrategy] = None,
        force_format: Optional[str] = None,
    ) -> ReferenceParseResult:
        """
        Parse a single bibliographic reference.

        Args:
            reference_text: The reference text to parse
            strategy: Parsing strategy to use (overrides default)
            force_format: Force specific format (e.g., 'APA', 'MLA', 'IEEE')

        Returns:
            ReferenceParseResult containing extracted metadata and parsing information
        """
        import time

        start_time = time.time()

        if not reference_text or not reference_text.strip():
            return ReferenceParseResult(
                metadata=CitationMetadata(),
                detected_format="Unknown",
                format_confidence=0.0,
                parsing_strategy=strategy or self.default_strategy,
                parsing_errors=["Empty or invalid reference text"],
                processing_time=time.time() - start_time,
            )

        strategy = strategy or self.default_strategy

        try:
            if strategy == ParsingStrategy.FORCE_FORMAT and force_format:
                result = self._parse_with_forced_format(reference_text, force_format)
            elif strategy == ParsingStrategy.TRY_ALL_FORMATS:
                result = self._parse_with_all_formats(reference_text)
            else:  # AUTO_DETECT
                result = self._parse_with_auto_detection(reference_text)

            result.parsing_strategy = strategy
            result.processing_time = time.time() - start_time

            self.logger.debug(
                f"Parsed reference with {result.detected_format} format "
                f"(confidence: {result.format_confidence:.2f})"
            )

            return result

        except Exception as e:
            self.logger.error(f"Error parsing reference: {e}")
            return ReferenceParseResult(
                metadata=CitationMetadata(),
                detected_format="Unknown",
                format_confidence=0.0,
                parsing_strategy=strategy,
                parsing_errors=[f"Parsing error: {str(e)}"],
                processing_time=time.time() - start_time,
            )

    def parse_references(
        self, reference_texts: List[str], strategy: Optional[ParsingStrategy] = None
    ) -> BatchParseResult:
        """
        Parse multiple bibliographic references.

        Args:
            reference_texts: List of reference texts to parse
            strategy: Parsing strategy to use for all references

        Returns:
            BatchParseResult containing individual results and batch statistics
        """
        import time

        start_time = time.time()

        if not reference_texts:
            return BatchParseResult(
                results=[],
                total_processed=0,
                successful_parses=0,
                failed_parses=0,
                total_processing_time=0.0,
            )

        results = []
        successful_count = 0

        self.logger.info(f"Starting batch parsing of {len(reference_texts)} references")

        for i, reference_text in enumerate(reference_texts):
            try:
                result = self.parse_reference(reference_text, strategy=strategy)
                results.append(result)

                if result.success:
                    successful_count += 1

                # Log progress for large batches
                if (i + 1) % 100 == 0:
                    self.logger.info(
                        f"Processed {i + 1}/{len(reference_texts)} references"
                    )

            except Exception as e:
                self.logger.error(f"Error processing reference {i + 1}: {e}")
                # Create error result
                error_result = ReferenceParseResult(
                    metadata=CitationMetadata(),
                    detected_format="Unknown",
                    format_confidence=0.0,
                    parsing_strategy=strategy or self.default_strategy,
                    parsing_errors=[f"Batch processing error: {str(e)}"],
                )
                results.append(error_result)

        total_time = time.time() - start_time
        failed_count = len(reference_texts) - successful_count

        batch_result = BatchParseResult(
            results=results,
            total_processed=len(reference_texts),
            successful_parses=successful_count,
            failed_parses=failed_count,
            total_processing_time=total_time,
        )

        self.logger.info(
            f"Batch parsing completed: {successful_count}/{len(reference_texts)} successful "
            f"({batch_result.success_rate:.1f}% success rate) in {total_time:.2f}s"
        )

        return batch_result

    def detect_format(self, reference_text: str) -> Tuple[str, float]:
        """
        Detect the citation format of a reference.

        Args:
            reference_text: The reference text to analyze

        Returns:
            Tuple of (format_name, confidence_score)
        """
        if not reference_text or not reference_text.strip():
            return "Unknown", 0.0

        # Use the general detector first
        general_format, general_confidence = detect_citation_format(reference_text)

        # Then try each specific handler's detection
        handler_scores = []
        for format_name, handler in self.handlers.items():
            try:
                confidence = handler.detect_format(reference_text)
                handler_scores.append((format_name, confidence))
            except Exception as e:
                self.logger.warning(f"Error detecting {format_name} format: {e}")
                handler_scores.append((format_name, 0.0))

        # Find the best scoring handler
        if handler_scores:
            best_format, best_confidence = max(handler_scores, key=lambda x: x[1])

            # Use handler result if it's significantly better than general detection
            if best_confidence > general_confidence + 0.1:
                return best_format, best_confidence

        return general_format, general_confidence

    def get_supported_formats(self) -> List[str]:
        """
        Get list of supported citation formats.

        Returns:
            List of supported format names
        """
        return list(self.handlers.keys())

    def get_parser_statistics(self) -> Dict[str, Any]:
        """
        Get parser statistics and configuration.

        Returns:
            Dictionary containing parser statistics
        """
        return {
            "supported_formats": self.get_supported_formats(),
            "active_handlers": len(self.handlers),
            "default_strategy": self.default_strategy.value,
            "confidence_threshold": self.confidence_threshold,
            "enable_fallback": self.enable_fallback,
        }

    def _parse_with_auto_detection(self, reference_text: str) -> ReferenceParseResult:
        """Parse reference with automatic format detection."""
        # Detect format
        detected_format, format_confidence = self.detect_format(reference_text)

        # If confidence is too low, try fallback strategies
        if format_confidence < self.confidence_threshold and self.enable_fallback:
            self.logger.debug(
                f"Low confidence ({format_confidence:.2f}) for {detected_format}, "
                "trying fallback strategies"
            )
            return self._parse_with_fallback(
                reference_text, detected_format, format_confidence
            )

        # Parse with detected format
        if detected_format in self.handlers:
            handler = self.handlers[detected_format]
            parse_result = handler.parse_citation(reference_text)

            return ReferenceParseResult(
                metadata=parse_result.metadata,
                detected_format=detected_format,
                format_confidence=format_confidence,
                parsing_strategy=ParsingStrategy.AUTO_DETECT,
                parsing_errors=parse_result.parsing_errors,
                parsing_warnings=parse_result.parsing_warnings,
            )
        else:
            return ReferenceParseResult(
                metadata=CitationMetadata(),
                detected_format=detected_format,
                format_confidence=format_confidence,
                parsing_strategy=ParsingStrategy.AUTO_DETECT,
                parsing_errors=[f"No handler available for format: {detected_format}"],
            )

    def _parse_with_forced_format(
        self, reference_text: str, format_name: str
    ) -> ReferenceParseResult:
        """Parse reference with a specific forced format."""
        if format_name not in self.handlers:
            return ReferenceParseResult(
                metadata=CitationMetadata(),
                detected_format=format_name,
                format_confidence=0.0,
                parsing_strategy=ParsingStrategy.FORCE_FORMAT,
                parsing_errors=[f"Unsupported format: {format_name}"],
            )

        handler = self.handlers[format_name]

        # Get format confidence but proceed regardless
        format_confidence = handler.detect_format(reference_text)

        parse_result = handler.parse_citation(reference_text)

        return ReferenceParseResult(
            metadata=parse_result.metadata,
            detected_format=format_name,
            format_confidence=format_confidence,
            parsing_strategy=ParsingStrategy.FORCE_FORMAT,
            parsing_errors=parse_result.parsing_errors,
            parsing_warnings=parse_result.parsing_warnings,
        )

    def _parse_with_all_formats(self, reference_text: str) -> ReferenceParseResult:
        """Try parsing with all available formats and return the best result."""
        results = []

        for format_name, handler in self.handlers.items():
            try:
                format_confidence = handler.detect_format(reference_text)
                parse_result = handler.parse_citation(reference_text)

                result = ReferenceParseResult(
                    metadata=parse_result.metadata,
                    detected_format=format_name,
                    format_confidence=format_confidence,
                    parsing_strategy=ParsingStrategy.TRY_ALL_FORMATS,
                    parsing_errors=parse_result.parsing_errors,
                    parsing_warnings=parse_result.parsing_warnings,
                )

                results.append(result)

            except Exception as e:
                self.logger.warning(f"Error trying {format_name} format: {e}")

        if not results:
            return ReferenceParseResult(
                metadata=CitationMetadata(),
                detected_format="Unknown",
                format_confidence=0.0,
                parsing_strategy=ParsingStrategy.TRY_ALL_FORMATS,
                parsing_errors=["All format parsers failed"],
            )

        # Return the result with highest overall confidence
        best_result = max(results, key=lambda r: r.overall_confidence)
        return best_result

    def _parse_with_fallback(
        self, reference_text: str, detected_format: str, format_confidence: float
    ) -> ReferenceParseResult:
        """Parse with fallback strategies when confidence is low."""
        # Try the detected format first
        if detected_format in self.handlers:
            handler = self.handlers[detected_format]
            parse_result = handler.parse_citation(reference_text)

            # If parsing was successful, use it despite low confidence
            if (
                parse_result.success
                and parse_result.metadata.get_overall_confidence() > 0.3
            ):
                return ReferenceParseResult(
                    metadata=parse_result.metadata,
                    detected_format=detected_format,
                    format_confidence=format_confidence,
                    parsing_strategy=ParsingStrategy.AUTO_DETECT,
                    parsing_errors=parse_result.parsing_errors,
                    parsing_warnings=parse_result.parsing_warnings
                    + ["Low format detection confidence, but parsing succeeded"],
                )

        # Fallback: try all formats and pick the best
        self.logger.debug("Using fallback: trying all formats")
        return self._parse_with_all_formats(reference_text)
