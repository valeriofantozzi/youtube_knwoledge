"""
Preprocessing module for document files.

Handles document parsing, text cleaning, chunking, and metadata extraction.
Supports multiple file formats through pluggable parsers.
"""

# Core parser infrastructure
from .parser_base import (
    DocumentParser,
    ParserRegistry,
    TextEntry,
    SourceMetadata,
    find_document_files,
)

# Format-specific parsers (auto-register on import)
from .parsers import (
    SRTDocumentParser,
    SRTParser,  # Alias for SRTDocumentParser
    SubtitleEntry,
    TextDocumentParser,
    MarkdownDocumentParser,
)

# Legacy SRT parser
from .srt_parser import SRTParser as LegacySRTParser, SubtitleEntry as LegacySubtitleEntry

# Other preprocessing components
from .text_cleaner import TextCleaner
from .chunker import SemanticChunker, Chunk
from .metadata_extractor import MetadataExtractor
from .pipeline import PreprocessingPipeline, ProcessedDocument

__all__ = [
    # Parser infrastructure
    "DocumentParser",
    "ParserRegistry",
    "TextEntry",
    "SourceMetadata",
    "find_document_files",
    
    # Parsers
    "SRTDocumentParser",
    "SRTParser",
    "SubtitleEntry",
    "TextDocumentParser",
    "MarkdownDocumentParser",
    
    # Processing components
    "TextCleaner",
    "SemanticChunker",
    "Chunk",
    "MetadataExtractor",
    "PreprocessingPipeline",
    "ProcessedDocument",
]
