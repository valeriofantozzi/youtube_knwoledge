"""
Metadata Extractor Module

Extracts metadata from subtitle filenames.
"""

import re
from pathlib import Path
from typing import Dict, Optional
from dataclasses import dataclass
from datetime import datetime


@dataclass
class VideoMetadata:
    """Represents video metadata extracted from filename."""
    video_id: str
    date: str  # Format: YYYY/MM/DD
    title: str
    filename: str
    file_path: str
    
    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            "video_id": self.video_id,
            "date": self.date,
            "title": self.title,
            "filename": self.filename,
            "file_path": str(self.file_path),
        }


class MetadataExtractor:
    """Extracts metadata from subtitle filenames."""
    
    # Pattern for YouTube video ID (11 characters)
    YOUTUBE_ID_PATTERN = re.compile(r'[a-zA-Z0-9_-]{11}')
    
    # Pattern for date in YYYYMMDD format
    DATE_PATTERN = re.compile(r'(\d{4})(\d{2})(\d{2})')
    
    def __init__(self):
        """Initialize metadata extractor."""
        pass
    
    def extract_from_filename(self, file_path: Path) -> VideoMetadata:
        """
        Extract metadata from filename.
        
        Expected format: YYYYMMDD_<channel>_<video_id>_<title>.en.srt
        or variations thereof.
        
        Args:
            file_path: Path to subtitle file
        
        Returns:
            VideoMetadata object
        
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
        # Format: YYYYMMDD_<channel>_<11_char_video_id>_<title>
        # Find all potential 11-character sequences
        parts = filename.split('_')
        video_id = None
        
        # Look for 11-character alphanumeric sequence (YouTube ID format)
        for i, part in enumerate(parts):
            # Skip date part (first part)
            if i == 0:
                continue
            # Check if this part is exactly 11 characters and alphanumeric
            clean_part = part.replace('-', '').replace('_', '')
            if len(part) == 11 and clean_part.isalnum() and len(clean_part) == 11:
                video_id = part
                break
        
        # If not found, try regex pattern
        if not video_id:
            video_id_match = self.YOUTUBE_ID_PATTERN.search(filename)
            if video_id_match:
                # Make sure it's not part of the date
                match_pos = video_id_match.start()
                date_end = date_match.end()
                if match_pos > date_end:
                    video_id = video_id_match.group()
        
        # If video ID not found, try to extract from parts after date
        # Some files might have format: YYYYMMDD_<channel>_<partial_id>_<title>
        if not video_id:
            # Try to find any alphanumeric sequence that could be an ID
            parts_after_date = parts[1:]  # Skip date
            if len(parts_after_date) > 0:
                # Use first non-empty part as potential ID (might be channel or partial ID)
                potential_id = parts_after_date[0]
                if potential_id and len(potential_id) >= 5:  # At least 5 chars
                    video_id = potential_id[:11] if len(potential_id) >= 11 else potential_id
                else:
                    # Fallback: use a placeholder
                    video_id = "unknown"
        
        if not video_id:
            raise ValueError(f"Cannot extract video ID from filename: {full_filename}")
        
        # Extract title (everything after video ID)
        # Find position of video ID in filename (handle case where ID was truncated)
        video_id_pos = filename.find(video_id)
        if video_id_pos == -1:
            # Try to find the original part that was used to create video_id
            # Look for the part in filename that contains video_id
            parts_after_date = parts[1:]
            if parts_after_date:
                # Find which part contains or matches video_id
                for part in parts_after_date:
                    if video_id in part or part.startswith(video_id):
                        video_id_pos = filename.find(part)
                        if video_id_pos != -1:
                            # Use the full part for position calculation
                            title_start = video_id_pos + len(part)
                            break
                else:
                    # Fallback: skip first part after date
                    if len(parts) > 2:
                        # Skip date and first part (channel/id)
                        title_start = len(parts[0]) + len(parts[1]) + 2  # +2 for underscores
                    else:
                        raise ValueError(f"Cannot find video ID position in filename: {full_filename}")
            else:
                raise ValueError(f"Cannot find video ID position in filename: {full_filename}")
        else:
            # Title starts after video ID and underscore
            title_start = video_id_pos + len(video_id)
            if title_start < len(filename) and filename[title_start] == '_':
                title_start += 1
        
        title = filename[title_start:].strip()
        
        # Clean up title
        title = self._clean_title(title)
        
        # If title is empty, use filename without extension and date
        if not title:
            # Remove date and video ID parts
            title_parts = filename.split('_')
            title_parts = [p for p in title_parts if p != date_match.group() and p != video_id]
            title = ' '.join(title_parts).strip()
            if not title:
                title = filename.replace(date_match.group(), '').replace(video_id, '').strip('_')
        
        return VideoMetadata(
            video_id=video_id,
            date=date_str,
            title=title,
            filename=full_filename,
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
    
    def extract_from_path(self, file_path: Path) -> VideoMetadata:
        """
        Extract metadata from file path (same as extract_from_filename).
        
        Args:
            file_path: Path to subtitle file
        
        Returns:
            VideoMetadata object
        """
        return self.extract_from_filename(file_path)
    
    def validate_metadata(self, metadata: VideoMetadata) -> bool:
        """
        Validate extracted metadata.
        
        Args:
            metadata: VideoMetadata object
        
        Returns:
            True if valid, False otherwise
        """
        # Check video ID length (should be 11 characters)
        if len(metadata.video_id) != 11:
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
    
    def extract_batch(self, file_paths: list[Path]) -> Dict[Path, VideoMetadata]:
        """
        Extract metadata from multiple files.
        
        Args:
            file_paths: List of file paths
        
        Returns:
            Dictionary mapping paths to VideoMetadata objects
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
