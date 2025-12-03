"""
Metadata Extractor Module

Extracts metadata from document filenames (primarily SRT subtitle files).
"""

import re
from pathlib import Path
from typing import Dict, Optional
from dataclasses import dataclass
from datetime import datetime

from .parser_base import SourceMetadata


class MetadataExtractor:
    """Extracts metadata from document filenames."""
    
    # Pattern for YouTube video ID (11 characters)
    YOUTUBE_ID_PATTERN = re.compile(r'[a-zA-Z0-9_-]{11}')
    
    # Pattern for date in YYYYMMDD format
    DATE_PATTERN = re.compile(r'(\d{4})(\d{2})(\d{2})')
    
    def __init__(self):
        """Initialize metadata extractor."""
        pass
    
    def extract_from_filename(self, file_path: Path) -> SourceMetadata:
        """
        Extract metadata from filename.
        
        Expected format: YYYYMMDD_<channel>_<source_id>_<title>.en.srt
        or variations thereof.
        
        Args:
            file_path: Path to subtitle file
        
        Returns:
            SourceMetadata object
        
        Raises:
            ValueError: If metadata cannot be extracted
        """
        filename = file_path.stem  # Without extension
        full_filename = file_path.name
        
        # Remove language suffix if present (e.g., ".en")
        if '.' in filename:
            parts = filename.rsplit('.', 1)
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
        # Format: YYYYMMDD_<channel>_<11_char_source_id>_<title>
        # Find all potential 11-character sequences
        parts = filename.split('_')
        source_id = None
        
        # Look for 11-character alphanumeric sequence (YouTube ID format)
        for i, part in enumerate(parts):
            # Skip date part (first part)
            if i == 0:
                continue
            # Check if this part is exactly 11 characters and alphanumeric
            clean_part = part.replace('-', '').replace('_', '')
            if len(part) == 11 and clean_part.isalnum() and len(clean_part) == 11:
                source_id = part
                break
        
        # If not found, try regex pattern
        if not source_id:
            source_id_match = self.YOUTUBE_ID_PATTERN.search(filename)
            if source_id_match:
                # Make sure it's not part of the date
                match_pos = source_id_match.start()
                date_end = date_match.end()
                if match_pos > date_end:
                    source_id = source_id_match.group()
        
        # If video ID not found, try to extract from parts after date
        # Some files might have format: YYYYMMDD_<channel>_<partial_id>_<title>
        if not source_id:
            # Try to find any alphanumeric sequence that could be an ID
            parts_after_date = parts[1:]  # Skip date
            if len(parts_after_date) > 0:
                # Use first non-empty part as potential ID (might be channel or partial ID)
                potential_id = parts_after_date[0]
                if potential_id and len(potential_id) >= 5:  # At least 5 chars
                    source_id = potential_id[:11] if len(potential_id) >= 11 else potential_id
                else:
                    # Fallback: use a placeholder
                    source_id = "unknown"
        
        if not source_id:
            raise ValueError(f"Cannot extract source ID from filename: {full_filename}")
        
        # Extract title (everything after source ID)
        # Find position of source ID in filename (handle case where ID was truncated)
        source_id_pos = filename.find(source_id)
        if source_id_pos == -1:
            # Try to find the original part that was used to create source_id
            # Look for the part in filename that contains source_id
            parts_after_date = parts[1:]
            if parts_after_date:
                # Find which part contains or matches source_id
                for part in parts_after_date:
                    if source_id in part or part.startswith(source_id):
                        source_id_pos = filename.find(part)
                        if source_id_pos != -1:
                            # Use the full part for position calculation
                            title_start = source_id_pos + len(part)
                            break
                else:
                    # Fallback: skip first part after date
                    if len(parts) > 2:
                        # Skip date and first part (channel/id)
                        title_start = len(parts[0]) + len(parts[1]) + 2  # +2 for underscores
                    else:
                        raise ValueError(f"Cannot find source ID position in filename: {full_filename}")
            else:
                raise ValueError(f"Cannot find source ID position in filename: {full_filename}")
        else:
            # Title starts after source ID and underscore
            title_start = source_id_pos + len(source_id)
            if title_start < len(filename) and filename[title_start] == '_':
                title_start += 1
        
        title = filename[title_start:].strip()
        
        # Clean up title
        title = self._clean_title(title)
        
        # If title is empty, use filename without extension and date
        if not title:
            # Remove date and source ID parts
            title_parts = filename.split('_')
            title_parts = [p for p in title_parts if p != date_match.group() and p != source_id]
            title = ' '.join(title_parts).strip()
            if not title:
                title = filename.replace(date_match.group(), '').replace(source_id, '').strip('_')
        
        return SourceMetadata(
            source_id=source_id,
            title=title,
            date=date_str,
            source_type="srt",
            original_filename=full_filename,
            file_path=str(file_path.absolute())
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
        title = title.replace('_', ' ')
        
        # Remove multiple spaces
        title = re.sub(r'\s+', ' ', title)
        
        # Remove leading/trailing dashes and spaces
        title = title.strip(' -_')
        
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
            year, month, day = metadata.date.split('/')
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
