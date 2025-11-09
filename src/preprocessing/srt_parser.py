"""
SRT Parser Module

Parses SRT subtitle files and extracts text with timestamps.
"""

import re
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
import chardet


@dataclass
class SubtitleEntry:
    """Represents a single subtitle entry."""
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


class SRTParser:
    """Parser for SRT subtitle files."""
    
    # Pattern to match SRT timestamp line: HH:MM:SS,mmm --> HH:MM:SS,mmm
    TIMESTAMP_PATTERN = re.compile(
        r'(\d{2}:\d{2}:\d{2}[,.]\d{3})\s*-->\s*(\d{2}:\d{2}:\d{2}[,.]\d{3})'
    )
    
    def __init__(self, encoding: Optional[str] = None):
        """
        Initialize SRT parser.
        
        Args:
            encoding: File encoding. If None, will auto-detect.
        """
        self.encoding = encoding
    
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
        if self.encoding is None:
            self.encoding = self._detect_encoding(file_path)
        
        # Read file content
        try:
            with open(file_path, 'r', encoding=self.encoding) as f:
                content = f.read()
        except UnicodeDecodeError:
            # Try UTF-8-BOM
            try:
                with open(file_path, 'r', encoding='utf-8-sig') as f:
                    content = f.read()
                self.encoding = 'utf-8-sig'
            except UnicodeDecodeError:
                # Try latin-1 as fallback
                with open(file_path, 'r', encoding='latin-1') as f:
                    content = f.read()
                self.encoding = 'latin-1'
        
        return self._parse_content(content)
    
    def _detect_encoding(self, file_path: Path) -> str:
        """
        Detect file encoding.
        
        Args:
            file_path: Path to file
        
        Returns:
            Detected encoding string
        """
        with open(file_path, 'rb') as f:
            raw_data = f.read(10000)  # Read first 10KB for detection
        
        result = chardet.detect(raw_data)
        encoding = result.get('encoding', 'utf-8')
        
        # Handle None encoding (empty file)
        if encoding is None:
            return 'utf-8'
        
        # Normalize common encodings
        if encoding.lower() in ['utf-8', 'ascii']:
            # Check for BOM
            if raw_data.startswith(b'\xef\xbb\xbf'):
                return 'utf-8-sig'
            return 'utf-8'
        
        return encoding
    
    def _parse_content(self, content: str) -> List[SubtitleEntry]:
        """
        Parse SRT content string.
        
        Args:
            content: SRT file content
        
        Returns:
            List of SubtitleEntry objects
        """
        entries = []
        
        # Split content by double newlines (separates subtitle blocks)
        blocks = re.split(r'\n\s*\n', content.strip())
        
        for block in blocks:
            if not block.strip():
                continue
            
            entry = self._parse_block(block)
            if entry:
                entries.append(entry)
        
        return entries
    
    def _parse_block(self, block: str) -> Optional[SubtitleEntry]:
        """
        Parse a single subtitle block.
        
        Args:
            block: Single subtitle block text
        
        Returns:
            SubtitleEntry or None if parsing fails
        """
        lines = [line.strip() for line in block.split('\n') if line.strip()]
        
        if not lines:
            return None
        
        # First line should be sequence number
        try:
            sequence = int(lines[0])
        except ValueError:
            # Try to find sequence number in the block
            seq_match = re.search(r'^\d+', lines[0])
            if seq_match:
                sequence = int(seq_match.group())
            else:
                # Skip this block if we can't find sequence number
                return None
        
        # Second line should be timestamp
        timestamp_line = None
        timestamp_idx = 1
        
        # Look for timestamp line (might not be exactly second line)
        for i, line in enumerate(lines[1:], start=1):
            if self.TIMESTAMP_PATTERN.match(line):
                timestamp_line = line
                timestamp_idx = i
                break
        
        if not timestamp_line:
            # Try to find timestamp anywhere in the block
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
        
        # Skip empty subtitles
        if not text:
            return None
        
        return SubtitleEntry(
            sequence=sequence,
            start_time=start_time,
            end_time=end_time,
            text=text
        )
    
    def parse_multiple_files(
        self,
        file_paths: List[Path],
        skip_errors: bool = True
    ) -> Dict[Path, List[SubtitleEntry]]:
        """
        Parse multiple SRT files.
        
        Args:
            file_paths: List of paths to SRT files
            skip_errors: If True, skip files that fail to parse
        
        Returns:
            Dictionary mapping file paths to lists of SubtitleEntry objects
        """
        results = {}
        
        for file_path in file_paths:
            try:
                entries = self.parse_file(file_path)
                results[file_path] = entries
            except Exception as e:
                if skip_errors:
                    continue
                else:
                    raise
        
        return results
    
    def get_all_text(self, entries: List[SubtitleEntry]) -> str:
        """
        Extract all text from subtitle entries, joined with spaces.
        
        Args:
            entries: List of SubtitleEntry objects
        
        Returns:
            Combined text string
        """
        return ' '.join(entry.text for entry in entries)
    
    def get_text_by_time_range(
        self,
        entries: List[SubtitleEntry],
        start_seconds: float,
        end_seconds: float
    ) -> str:
        """
        Get subtitle text within a time range.
        
        Args:
            entries: List of SubtitleEntry objects
            start_seconds: Start time in seconds
            end_seconds: End time in seconds
        
        Returns:
            Combined text from entries within time range
        """
        matching_entries = []
        for entry in entries:
            entry_start = entry._time_to_seconds(entry.start_time)
            entry_end = entry._time_to_seconds(entry.end_time)
            
            # Check if entry overlaps with time range
            if not (entry_end < start_seconds or entry_start > end_seconds):
                matching_entries.append(entry)
        
        return ' '.join(entry.text for entry in matching_entries)
