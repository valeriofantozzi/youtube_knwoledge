"""
Document Parsers Package

Contains format-specific document parsers that implement the DocumentParser interface.
"""

from .srt_parser import SRTDocumentParser, SRTParser, SubtitleEntry
from .text_parser import TextDocumentParser
from .markdown_parser import MarkdownDocumentParser

__all__ = [
    "SRTDocumentParser",
    "SRTParser",  # Backward compatibility alias
    "SubtitleEntry",
    "TextDocumentParser",
    "MarkdownDocumentParser",
]
