"""
Document Parser Base Module

Defines abstract base class for document parsers and parser registry.
Enables pluggable parser architecture for supporting multiple file formats.
"""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import List, Dict, Optional, Tuple, Any, Type
from dataclasses import dataclass, field


@dataclass
class TextEntry:
    """
    Represents a single text entry from a parsed document.

    This is a generic data structure used by all parsers.
    For SRT files, this maps to individual subtitle entries with timestamps.
    For plain text, this maps to paragraphs or sections.
    For PDFs, this maps to text blocks or pages.
    """

    sequence: int
    text: str
    start_position: Optional[int] = None  # Character position in original document
    end_position: Optional[int] = None
    # Optional timing info (for SRT, audio, etc.)
    start_time: Optional[str] = None
    end_time: Optional[str] = None
    # Additional metadata
    extra: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        result = {
            "sequence": self.sequence,
            "text": self.text,
        }
        if self.start_position is not None:
            result["start_position"] = self.start_position
        if self.end_position is not None:
            result["end_position"] = self.end_position
        if self.start_time is not None:
            result["start_time"] = self.start_time
        if self.end_time is not None:
            result["end_time"] = self.end_time
        if self.extra:
            result["extra"] = self.extra
        return result


@dataclass
class SourceMetadata:
    """
    Represents metadata extracted from a source document.

    This is a generic container that works for any document type.
    """

    source_id: str  # Unique identifier (filename hash, etc.)
    title: str  # Human-readable title
    date: str  # Date string (format: YYYY/MM/DD)
    source_type: str  # File type: "srt", "txt", "md", "pdf", etc.
    original_filename: str  # Original filename
    file_path: str  # Full file path
    content_hash: str = ""  # SHA-256 hash of content
    extra: Dict[str, Any] = field(default_factory=dict)  # Format-specific metadata

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        result: Dict[str, Any] = {
            "source_id": self.source_id,
            "title": self.title,
            "date": self.date,
            "source_type": self.source_type,
            "original_filename": self.original_filename,
            "file_path": self.file_path,
            "content_hash": self.content_hash,
        }
        if self.extra:
            result["extra"] = self.extra
        return result

    @property
    def filename(self) -> str:
        """Alias for original_filename."""
        return self.original_filename

    @property
    def content_type(self) -> str:
        """Alias for source_type."""
        return self.source_type


class DocumentParser(ABC):
    """
    Abstract base class for document parsers.

    All document parsers must implement this interface to be registered
    with the ParserRegistry.
    """

    @property
    @abstractmethod
    def supported_extensions(self) -> List[str]:
        """
        Return list of supported file extensions.

        Returns:
            List of extensions with leading dot (e.g., ['.srt', '.sub'])
        """
        pass

    @property
    def parser_name(self) -> str:
        """Return human-readable parser name."""
        return self.__class__.__name__

    @abstractmethod
    def parse(self, file_path: Path) -> Tuple[List[TextEntry], SourceMetadata]:
        """
        Parse document and return text entries with metadata.

        Args:
            file_path: Path to the document file

        Returns:
            Tuple of (list of TextEntry objects, SourceMetadata)

        Raises:
            FileNotFoundError: If file doesn't exist
            ValueError: If file cannot be parsed
        """
        pass

    @abstractmethod
    def extract_metadata(self, file_path: Path) -> SourceMetadata:
        """
        Extract metadata from file without full parsing.

        Useful for quick metadata inspection without processing content.

        Args:
            file_path: Path to the document file

        Returns:
            SourceMetadata object
        """
        pass

    def can_parse(self, file_path: Path) -> bool:
        """
        Check if this parser can handle the given file.

        Default implementation checks file extension.
        Override for more sophisticated checks (e.g., magic bytes).

        Args:
            file_path: Path to check

        Returns:
            True if parser can handle this file
        """
        ext = file_path.suffix.lower()
        return ext in [e.lower() for e in self.supported_extensions]


class ParserRegistry:
    """
    Registry for document parsers.

    Manages parser registration and retrieval based on file extensions.
    Implemented as a singleton pattern.
    """

    _instance = None
    _parsers: Dict[str, DocumentParser] = {}

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._parsers = {}
        return cls._instance

    @classmethod
    def register(cls, parser: DocumentParser) -> None:
        """
        Register a parser for its supported extensions.

        Args:
            parser: DocumentParser instance to register
        """
        for ext in parser.supported_extensions:
            ext_lower = ext.lower()
            if not ext_lower.startswith("."):
                ext_lower = "." + ext_lower
            cls._parsers[ext_lower] = parser

    @classmethod
    def unregister(cls, extension: str) -> None:
        """
        Unregister parser for an extension.

        Args:
            extension: File extension to unregister
        """
        ext_lower = extension.lower()
        if not ext_lower.startswith("."):
            ext_lower = "." + ext_lower
        cls._parsers.pop(ext_lower, None)

    @classmethod
    def get_parser(cls, file_path: Path) -> Optional[DocumentParser]:
        """
        Get parser for a given file.

        Args:
            file_path: Path to file

        Returns:
            DocumentParser instance or None if no parser found
        """
        ext = file_path.suffix.lower()
        return cls._parsers.get(ext)

    @classmethod
    def get_parser_by_extension(cls, extension: str) -> Optional[DocumentParser]:
        """
        Get parser by file extension.

        Args:
            extension: File extension (with or without leading dot)

        Returns:
            DocumentParser instance or None if no parser found
        """
        ext_lower = extension.lower()
        if not ext_lower.startswith("."):
            ext_lower = "." + ext_lower
        return cls._parsers.get(ext_lower)

    @classmethod
    def supported_extensions(cls) -> List[str]:
        """
        Get list of all supported file extensions.

        Returns:
            List of extensions (e.g., ['.srt', '.txt', '.md'])
        """
        return sorted(cls._parsers.keys())

    @classmethod
    def get_all_parsers(cls) -> Dict[str, DocumentParser]:
        """
        Get all registered parsers.

        Returns:
            Dictionary mapping extensions to parsers
        """
        return cls._parsers.copy()

    @classmethod
    def clear(cls) -> None:
        """Clear all registered parsers (mainly for testing)."""
        cls._parsers.clear()

    @classmethod
    def is_supported(cls, file_path: Path) -> bool:
        """
        Check if a file type is supported.

        Args:
            file_path: Path to file

        Returns:
            True if file type is supported
        """
        return cls.get_parser(file_path) is not None


def find_document_files(
    directory: Path, extensions: Optional[List[str]] = None, recursive: bool = True
) -> List[Path]:
    """
    Find all document files in a directory.

    Args:
        directory: Directory to search
        extensions: List of extensions to include (default: all registered)
        recursive: If True, search subdirectories

    Returns:
        List of file paths
    """
    if not directory.is_dir():
        raise ValueError(f"Not a directory: {directory}")

    if extensions is None:
        extensions = ParserRegistry.supported_extensions()

    # Normalize extensions
    extensions = [
        e.lower() if e.startswith(".") else f".{e.lower()}" for e in extensions
    ]

    files = []
    if recursive:
        for ext in extensions:
            files.extend(directory.rglob(f"*{ext}"))
    else:
        for ext in extensions:
            files.extend(directory.glob(f"*{ext}"))

    return sorted(files)


# Backward compatibility alias
find_srt_files = find_document_files
