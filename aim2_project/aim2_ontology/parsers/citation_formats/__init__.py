"""
Citation Formats Module

This module provides format-specific handlers for parsing different citation styles.
Each handler implements specialized logic for extracting metadata from citations
following specific formatting conventions (APA, MLA, IEEE, etc.).

Available Handlers:
    APAHandler: American Psychological Association citation format
    MLAHandler: Modern Language Association citation format
    IEEEHandler: Institute of Electrical and Electronics Engineers citation format

Base Classes:
    BaseCitationHandler: Abstract base class for all citation format handlers

The handlers are designed to work with the reference pattern matching system
to provide accurate, format-specific parsing of bibliographic references.

Usage:
    from aim2_project.aim2_ontology.parsers.citation_formats import APAHandler

    handler = APAHandler()
    result = handler.parse_citation("Smith, J. (2023). Title of paper. Journal Name, 15(3), 123-145.")
"""

from typing import List

from .apa_handler import APAHandler
from .base_handler import (
    BaseCitationHandler,
    CitationMetadata,
    CitationType,
    ParseResult,
)
from .ieee_handler import IEEEHandler
from .mla_handler import MLAHandler

__all__ = [
    "BaseCitationHandler",
    "CitationMetadata",
    "ParseResult",
    "CitationType",
    "APAHandler",
    "MLAHandler",
    "IEEEHandler",
]

# Format registry for easy lookup
CITATION_HANDLERS = {"APA": APAHandler, "MLA": MLAHandler, "IEEE": IEEEHandler}


def get_handler_for_format(format_name: str) -> BaseCitationHandler:
    """
    Get appropriate citation handler for the specified format.

    Args:
        format_name: Name of the citation format ('APA', 'MLA', 'IEEE')

    Returns:
        Instance of the appropriate citation handler

    Raises:
        ValueError: If format_name is not supported
    """
    if format_name not in CITATION_HANDLERS:
        raise ValueError(
            f"Unsupported citation format: {format_name}. "
            f"Supported formats: {list(CITATION_HANDLERS.keys())}"
        )

    handler_class = CITATION_HANDLERS[format_name]
    return handler_class()


def get_supported_formats() -> List[str]:
    """
    Get list of supported citation formats.

    Returns:
        List of supported format names
    """
    return list(CITATION_HANDLERS.keys())
