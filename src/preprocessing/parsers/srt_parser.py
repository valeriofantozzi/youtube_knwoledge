"""
SRT Document Parser

Parser for SRT subtitle files, implementing the DocumentParser interface.
"""

import re
import hashlib
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime
import chardet

from ..parser_base import (
    DocumentParser, 
    TextEntry, 
    SourceMetadata, 
    ParserRegistry
)


@dataclass
class SubtitleEntry:
    """
    Represents a single subtitle entry.
    
    Kept for backward compatibility with existing code.
    """
    sequence: int
    start_time: str  # Format: HH:MM:SS,mmm
    end_time: str    # Format: HH:MM:SS,mmm
    text: str        # Subtitle text (may contain multiple lines)
    
    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            "sequence": self.sequence,
            "start_time": self.start_time,
            "end_time": self.end_time,
            "text": self.text,
        }
    
    def get_duration_seconds(self) -> float:
        """Calculate duration in seconds."""
        start_sec = self._time_to_seconds(self.start_time)
        end_sec = self._time_to_seconds(self.end_time)
        return end_sec - start_sec
    
    @staticmethod
    def _time_to_seconds(time_str: str) -> float:
        """Convert SRT time format to seconds."""
        # Format: HH:MM:SS,mmm or HH:MM:SS.mmm
        time_str = time_str.replace(',', '.')
        parts = time_str.split(':')
        hours = int(parts[0])
        minutes = int(parts[1])
        seconds = float(parts[2])
        return hours * 3600 + minutes * 60 + seconds
    
    def to_text_entry(self) -> TextEntry:
        """Convert to generic TextEntry."""
        return TextEntry(
            sequence=self.sequence,
            text=self.text,
            start_time=self.start_time,
            end_time=self.end_time,
            extra={"duration_seconds": self.get_duration_seconds()}
        )


class SRTDocumentParser(DocumentParser):
    """
    Parser for SRT subtitle files.
    
    Implements the DocumentParser interface for .srt files.
    """
    
    # Pattern to match SRT timestamp line: HH:MM:SS,mmm --> HH:MM:SS,mmm
    TIMESTAMP_PATTERN = re.compile(
        r'(\d{2}:\d{2}:\d{2}[,.]\d{3})\s*-->\s*(\d{2}:\d{2}:\d{2}[,.]\d{3})'
    )
    
    # Pattern for date in YYYYMMDD format
    DATE_PATTERN = re.compile(r'(\d{4})(\d{2})(\d{2})')
    
    # Pattern for YouTube video ID (11 characters)
    YOUTUBE_ID_PATTERN = re.compile(r'[a-zA-Z0-9_-]{11}')
    
    def __init__(self, encoding: Optional[str] = None):
        """
        Initialize SRT parser.
        
        Args:
            encoding: File encoding. If None, will auto-detect.
        """
        self.encoding = encoding
    
    @property
    def supported_extensions(self) -> List[str]:
        """Return list of supported file extensions."""
        return ['.srt', '.sub']
    
    @property
    def parser_name(self) -> str:
        """Return human-readable parser name."""
        return "SRT Subtitle Parser"
    
    def parse(self, file_path: Path) -> Tuple[List[TextEntry], SourceMetadata]:
        """
        Parse SRT file and return text entries with metadata.
        
        Args:
            file_path: Path to SRT file
            
        Returns:
            Tuple of (list of TextEntry objects, SourceMetadata)
        """
        # Parse subtitle entries
        subtitle_entries = self.parse_file(file_path)
        
        # Convert to TextEntry objects
        text_entries = [entry.to_text_entry() for entry in subtitle_entries]
        
        # Extract metadata
        metadata = self.extract_metadata(file_path)
        
        return text_entries, metadata
    
    def extract_metadata(self, file_path: Path) -> SourceMetadata:
        """
        Extract metadata from SRT filename.
        
        Expected format: YYYYMMDD_<channel>_<video_id>_<title>.en.srt
        or variations thereof.
        
        Args:
            file_path: Path to SRT file
            
        Returns:
            SourceMetadata object
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
        if date_match:
            year, month, day = date_match.groups()
            date_str = f"{year}/{month}/{day}"
            # Validate date
            try:
                datetime(int(year), int(month), int(day))
            except ValueError:
                date_str = "0000/00/00"
        else:
            date_str = "0000/00/00"
        
        # Extract video ID
        source_id = self._extract_video_id(filename, date_match)
        
        # Extract title
        title = self._extract_title(filename, source_id)
        
        return SourceMetadata(
            source_id=source_id,
            title=title,
            date=date_str,
            source_type="srt",
            original_filename=full_filename,
            file_path=str(file_path.absolute()),
            extra={"language": self._extract_language(file_path.name)}
        )
    
    def _extract_video_id(self, filename: str, date_match: Optional[re.Match]) -> str:
        """Extract video ID from filename."""
        parts = filename.split('_')
        
        # Look for 11-character alphanumeric sequence (YouTube ID format)
        for i, part in enumerate(parts):
            if i == 0:  # Skip date part
                continue
            clean_part = part.replace('-', '').replace('_', '')
            if len(part) == 11 and clean_part.isalnum() and len(clean_part) == 11:
                return part
        
        # If not found, try regex pattern
        video_id_match = self.YOUTUBE_ID_PATTERN.search(filename)
        if video_id_match and date_match:
            match_pos = video_id_match.start()
            date_end = date_match.end()
            if match_pos > date_end:
                return video_id_match.group()
        
        # Try to find any alphanumeric sequence that could be an ID
        parts_after_date = parts[1:]
        if parts_after_date:
            potential_id = parts_after_date[0]
            if potential_id and len(potential_id) >= 5:
                return potential_id[:11] if len(potential_id) >= 11 else potential_id
        
        # Fallback: generate hash from filename
        return hashlib.md5(filename.encode()).hexdigest()[:11]
    
    def _extract_title(self, filename: str, source_id: str) -> str:
        """Extract title from filename."""
        parts = filename.split('_')
        
        # Find position of source_id and get everything after
        source_id_pos = filename.find(source_id)
        if source_id_pos != -1:
            title_start = source_id_pos + len(source_id)
            if title_start < len(filename):
                title = filename[title_start:].lstrip('_')
                if title:
                    return title.replace('_', ' ')
        
        # Fallback: use last part of filename
        if len(parts) > 2:
            return '_'.join(parts[2:]).replace('_', ' ')
        
        return filename.replace('_', ' ')
    
    def _extract_language(self, filename: str) -> Optional[str]:
        """Extract language code from filename."""
        # Look for pattern like .en.srt
        match = re.search(r'\.([a-z]{2})\.srt$', filename, re.IGNORECASE)
        if match:
            return match.group(1).lower()
        return None
    
    def parse_file(self, file_path: Path) -> List[SubtitleEntry]:
        """
        Parse an SRT file and return list of subtitle entries.
        
        Args:
            file_path: Path to SRT file
        
        Returns:
            List of SubtitleEntry objects
        
        Raises:
            FileNotFoundError: If file doesn't exist
            ValueError: If file cannot be parsed
        """
        if not file_path.exists():
            raise FileNotFoundError(f"SRT file not found: {file_path}")
        
        # Detect encoding if not specified
        encoding = self.encoding
        if encoding is None:
            encoding = self._detect_encoding(file_path)
        
        # Read file content
        try:
            with open(file_path, 'r', encoding=encoding) as f:
                content = f.read()
        except UnicodeDecodeError:
            # Try UTF-8-BOM
            try:
                with open(file_path, 'r', encoding='utf-8-sig') as f:
                    content = f.read()
            except UnicodeDecodeError:
                # Try latin-1 as fallback
                with open(file_path, 'r', encoding='latin-1') as f:
                    content = f.read()
        
        return self._parse_content(content)
    
    def _detect_encoding(self, file_path: Path) -> str:
        """Detect file encoding."""
        with open(file_path, 'rb') as f:
            raw_data = f.read(10000)
        
        result = chardet.detect(raw_data)
        encoding = result.get('encoding', 'utf-8')
        
        if encoding is None:
            return 'utf-8'
        
        if encoding.lower() in ['utf-8', 'ascii']:
            if raw_data.startswith(b'\xef\xbb\xbf'):
                return 'utf-8-sig'
            return 'utf-8'
        
        return encoding
    
    def _parse_content(self, content: str) -> List[SubtitleEntry]:
        """Parse SRT content string."""
        entries = []
        blocks = re.split(r'\n\s*\n', content.strip())
        
        for block in blocks:
            if not block.strip():
                continue
            entry = self._parse_block(block)
            if entry:
                entries.append(entry)
        
        return entries
    
    def _parse_block(self, block: str) -> Optional[SubtitleEntry]:
        """Parse a single subtitle block."""
        lines = [line.strip() for line in block.split('\n') if line.strip()]
        
        if not lines:
            return None
        
        # First line should be sequence number
        try:
            sequence = int(lines[0])
        except ValueError:
            seq_match = re.search(r'^\d+', lines[0])
            if seq_match:
                sequence = int(seq_match.group())
            else:
                return None
        
        # Find timestamp line
        timestamp_line = None
        timestamp_idx = 1
        
        for i, line in enumerate(lines[1:], start=1):
            if self.TIMESTAMP_PATTERN.match(line):
                timestamp_line = line
                timestamp_idx = i
                break
        
        if not timestamp_line:
            for line in lines:
                match = self.TIMESTAMP_PATTERN.match(line)
                if match:
                    timestamp_line = line
                    break
        
        if not timestamp_line:
            return None
        
        # Extract timestamps
        match = self.TIMESTAMP_PATTERN.match(timestamp_line)
        if not match:
            return None
        
        start_time = match.group(1).replace('.', ',')
        end_time = match.group(2).replace('.', ',')
        
        # Remaining lines are subtitle text
        text_lines = lines[timestamp_idx + 1:]
        text = '\n'.join(text_lines).strip()
        
        if not text:
            return None
        
        return SubtitleEntry(
            sequence=sequence,
            start_time=start_time,
            end_time=end_time,
            text=text
        )
    
    def get_all_text(self, entries: List[SubtitleEntry]) -> str:
        """Extract all text from subtitle entries, joined with spaces."""
        return ' '.join(entry.text for entry in entries)
    
    def get_text_by_time_range(
        self,
        entries: List[SubtitleEntry],
        start_seconds: float,
        end_seconds: float
    ) -> str:
        """Get subtitle text within a time range."""
        matching_entries = []
        for entry in entries:
            entry_start = entry._time_to_seconds(entry.start_time)
            entry_end = entry._time_to_seconds(entry.end_time)
            
            if not (entry_end < start_seconds or entry_start > end_seconds):
                matching_entries.append(entry)
        
        return ' '.join(entry.text for entry in matching_entries)


# Create singleton instance and register with registry
_srt_parser = SRTDocumentParser()
ParserRegistry.register(_srt_parser)


# Backward compatibility alias
SRTParser = SRTDocumentParser
