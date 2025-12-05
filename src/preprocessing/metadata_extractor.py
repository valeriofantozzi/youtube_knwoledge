"""
Metadata Extractor Module

Extracts metadata from document filenames and content.
Supports various document types including SRT files, PDFs, and text files.
"""

import re
import hashlib
import logging
from pathlib import Path
from typing import Dict, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime

from .parser_base import SourceMetadata

logger = logging.getLogger(__name__)


class MetadataExtractor:
    """Extracts metadata from document filenames."""

    # Pattern for YouTube video ID (11 characters)
    YOUTUBE_ID_PATTERN = re.compile(r"[a-zA-Z0-9_-]{11}")

    # Pattern for date in YYYYMMDD format
    DATE_PATTERN = re.compile(r"(\d{4})(\d{2})(\d{2})")

    def __init__(self):
        """Initialize metadata extractor."""
        pass

    def _generate_source_id_from_content(self, file_path: Path) -> Tuple[str, str]:
        """
        Generate unique source identifier and content hash.
        
        This ensures that identical files get identical source_ids,
        preventing duplicate documents from being indexed.
        
        Args:
            file_path: Path to the source file
        
        Returns:
            Tuple of (source_id, content_hash)
            source_id: {filename}_{content_hash_16chars}
            content_hash: SHA-256 hash of content
        """
        filename = file_path.stem
        
        try:
            # Read file content and compute SHA-256 hash
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Generate SHA-256 hash of content
            content_hash = hashlib.sha256(content.encode('utf-8')).hexdigest()
            
            # Use first 16 characters of hash (still 64 bits of entropy)
            hash_suffix = content_hash[:16]
            
            source_id = f"{filename}_{hash_suffix}"
            
            logger.debug(f"Generated source_id for {file_path.name}: {source_id}")
            return source_id, content_hash
            
        except Exception as e:
            logger.warning(f"Could not hash file {file_path}, using filename only: {e}")
            # Fallback: use filename only (may not prevent duplicates)
            return filename, ""

    def extract_from_filename(self, file_path: Path) -> SourceMetadata:
        """
        Extract metadata from filename and file content.

        Expected format: YYYYMMDD_<channel>_<title>.en.srt
        or variations thereof.
        
        The source_id is generated from file content hash to ensure
        identical files are deduplicated.

        Args:
            file_path: Path to subtitle file

        Returns:
            SourceMetadata object

        Raises:
            ValueError: If metadata cannot be extracted
        """
        filename = file_path.stem  # Without extension
        full_filename = file_path.name
        
        # Generate source_id and content_hash from content (for deduplication)
        source_id, content_hash = self._generate_source_id_from_content(file_path)

        # Remove language suffix if present (e.g., ".en")
        if "." in filename:
            parts = filename.rsplit(".", 1)
            if len(parts[1]) <= 3:  # Likely language code
                filename = parts[0]

        # Extract date (YYYYMMDD format)
        date_match = self.DATE_PATTERN.search(filename)
        if not date_match:
            raise ValueError(f"Cannot extract date from filename: {full_filename}")

        year, month, day = date_match.groups()
        date_str = f"{year}/{month}/{day}"

        # Validate date
        try:
            datetime(int(year), int(month), int(day))
        except ValueError:
            raise ValueError(f"Invalid date in filename: {full_filename}")

        # Extract video ID (11 characters, typically after date and channel)
        # Extract title from filename
        # Format: YYYYMMDD_<channel>_<title>.en.srt
        # Remove date part to get the rest
        parts = filename.split("_")
        
        # Skip date part (first part)
        if len(parts) > 1 and date_match and len(parts[0]) == 8:
            # Skip date and first part (channel/id), use rest as title
            title_parts = parts[2:] if len(parts) > 2 else []
            title = "_".join(title_parts).strip() if title_parts else ""
        else:
            # No clear date/channel separation, use the whole filename
            title = filename.strip()

        # Clean up title
        title = self._clean_title(title)

        # If title is still empty, use filename
        if not title:
            title = filename.replace(date_match.group(), "").strip("_") if date_match else filename

        return SourceMetadata(
            source_id=source_id,
            title=title,
            date=date_str,
            source_type="srt",
            original_filename=full_filename,
            file_path=str(file_path.absolute()),
            content_hash=content_hash,
        )

    def _clean_title(self, title: str) -> str:
        """
        Clean title string.

        Args:
            title: Raw title string

        Returns:
            Cleaned title
        """
        # Replace underscores with spaces
        title = title.replace("_", " ")

        # Remove multiple spaces
        title = re.sub(r"\s+", " ", title)

        # Remove leading/trailing dashes and spaces
        title = title.strip(" -_")

        return title

    def extract_from_path(self, file_path: Path) -> SourceMetadata:
        """
        Extract metadata from file path (same as extract_from_filename).

        Args:
            file_path: Path to subtitle file

        Returns:
            SourceMetadata object
        """
        return self.extract_from_filename(file_path)

    def validate_metadata(self, metadata: SourceMetadata) -> bool:
        """
        Validate extracted metadata.

        Args:
            metadata: SourceMetadata object

        Returns:
            True if valid, False otherwise
        """
        # Check video ID length (should be 11 characters)
        if len(metadata.source_id) != 11:
            return False

        # Check date format
        try:
            year, month, day = metadata.date.split("/")
            datetime(int(year), int(month), int(day))
        except (ValueError, AttributeError):
            return False

        # Check title is not empty
        if not metadata.title or not metadata.title.strip():
            return False

        return True

    def extract_batch(self, file_paths: list[Path]) -> Dict[Path, SourceMetadata]:
        """
        Extract metadata from multiple files.

        Args:
            file_paths: List of file paths

        Returns:
            Dictionary mapping paths to SourceMetadata objects
        """
        results = {}
        for file_path in file_paths:
            try:
                metadata = self.extract_from_filename(file_path)
                if self.validate_metadata(metadata):
                    results[file_path] = metadata
            except Exception:
                # Skip files that cannot be parsed
                continue

        return results
