"""
Plain Text Document Parser

Parser for plain text files (.txt), implementing the DocumentParser interface.
"""

import hashlib
from pathlib import Path
from typing import List, Optional, Tuple
from datetime import datetime
import re

from ..parser_base import (
    DocumentParser, 
    TextEntry, 
    SourceMetadata, 
    ParserRegistry
)


class TextDocumentParser(DocumentParser):
    """
    Parser for plain text files.
    
    Splits text into paragraphs or by sentence count for chunking.
    """
    
    def __init__(
        self,
        split_by: str = "paragraph",
        min_paragraph_length: int = 50,
        encoding: Optional[str] = None
    ):
        """
        Initialize text parser.
        
        Args:
            split_by: How to split text - "paragraph", "sentence", or "line"
            min_paragraph_length: Minimum characters for a paragraph to be kept
            encoding: File encoding. If None, will auto-detect.
        """
        self.split_by = split_by
        self.min_paragraph_length = min_paragraph_length
        self.encoding = encoding
    
    @property
    def supported_extensions(self) -> List[str]:
        """Return list of supported file extensions."""
        return ['.txt', '.text']
    
    @property
    def parser_name(self) -> str:
        """Return human-readable parser name."""
        return "Plain Text Parser"
    
    def parse(self, file_path: Path) -> Tuple[List[TextEntry], SourceMetadata]:
        """
        Parse text file and return text entries with metadata.
        
        Args:
            file_path: Path to text file
            
        Returns:
            Tuple of (list of TextEntry objects, SourceMetadata)
        """
        if not file_path.exists():
            raise FileNotFoundError(f"Text file not found: {file_path}")
        
        # Read file content
        content = self._read_file(file_path)
        
        # Split content into entries
        text_entries = self._split_content(content)
        
        # Extract metadata
        metadata = self.extract_metadata(file_path)
        
        return text_entries, metadata
    
    def extract_metadata(self, file_path: Path) -> SourceMetadata:
        """
        Extract metadata from text file.
        
        Args:
            file_path: Path to text file
            
        Returns:
            SourceMetadata object
        """
        filename = file_path.stem
        full_filename = file_path.name
        
        # Try to extract date from filename (YYYYMMDD or YYYY-MM-DD patterns)
        date_str = self._extract_date(filename)
        
        # Generate source ID from filename hash
        source_id = hashlib.md5(full_filename.encode()).hexdigest()[:11]
        
        # Use filename as title
        title = self._clean_title(filename)
        
        # Get file modification date if no date in filename
        if date_str == "0000/00/00":
            try:
                mtime = file_path.stat().st_mtime
                dt = datetime.fromtimestamp(mtime)
                date_str = dt.strftime("%Y/%m/%d")
            except Exception:
                pass
        
        return SourceMetadata(
            source_id=source_id,
            title=title,
            date=date_str,
            source_type="txt",
            original_filename=full_filename,
            file_path=str(file_path.absolute()),
            extra={"encoding": self.encoding or "utf-8"}
        )
    
    def _read_file(self, file_path: Path) -> str:
        """Read file content with encoding detection."""
        encodings = [self.encoding] if self.encoding else ['utf-8', 'utf-8-sig', 'latin-1', 'cp1252']
        
        for enc in encodings:
            if enc is None:
                continue
            try:
                with open(file_path, 'r', encoding=enc) as f:
                    content = f.read()
                self.encoding = enc
                return content
            except UnicodeDecodeError:
                continue
        
        # Last resort: read as binary and decode with errors ignored
        with open(file_path, 'rb') as f:
            content = f.read().decode('utf-8', errors='ignore')
        return content
    
    def _split_content(self, content: str) -> List[TextEntry]:
        """Split content into text entries based on split_by setting."""
        if self.split_by == "paragraph":
            return self._split_by_paragraph(content)
        elif self.split_by == "sentence":
            return self._split_by_sentence(content)
        elif self.split_by == "line":
            return self._split_by_line(content)
        else:
            return self._split_by_paragraph(content)
    
    def _split_by_paragraph(self, content: str) -> List[TextEntry]:
        """Split content by paragraphs (double newlines)."""
        # Split by double newlines
        paragraphs = re.split(r'\n\s*\n', content.strip())
        
        entries = []
        position = 0
        
        for i, para in enumerate(paragraphs):
            para = para.strip()
            if len(para) < self.min_paragraph_length:
                position += len(para) + 2
                continue
            
            entries.append(TextEntry(
                sequence=len(entries) + 1,
                text=para,
                start_position=position,
                end_position=position + len(para)
            ))
            position += len(para) + 2  # +2 for paragraph break
        
        return entries
    
    def _split_by_sentence(self, content: str) -> List[TextEntry]:
        """Split content by sentences."""
        # Simple sentence splitting - can be improved with NLP
        sentences = re.split(r'(?<=[.!?])\s+', content.strip())
        
        entries = []
        position = 0
        
        for i, sentence in enumerate(sentences):
            sentence = sentence.strip()
            if not sentence:
                continue
            
            entries.append(TextEntry(
                sequence=len(entries) + 1,
                text=sentence,
                start_position=position,
                end_position=position + len(sentence)
            ))
            position += len(sentence) + 1
        
        return entries
    
    def _split_by_line(self, content: str) -> List[TextEntry]:
        """Split content by lines."""
        lines = content.strip().split('\n')
        
        entries = []
        position = 0
        
        for i, line in enumerate(lines):
            line = line.strip()
            if not line:
                position += 1
                continue
            
            entries.append(TextEntry(
                sequence=len(entries) + 1,
                text=line,
                start_position=position,
                end_position=position + len(line)
            ))
            position += len(line) + 1
        
        return entries
    
    def _extract_date(self, filename: str) -> str:
        """Extract date from filename."""
        # Try YYYYMMDD pattern
        match = re.search(r'(\d{4})(\d{2})(\d{2})', filename)
        if match:
            year, month, day = match.groups()
            try:
                datetime(int(year), int(month), int(day))
                return f"{year}/{month}/{day}"
            except ValueError:
                pass
        
        # Try YYYY-MM-DD pattern
        match = re.search(r'(\d{4})-(\d{2})-(\d{2})', filename)
        if match:
            year, month, day = match.groups()
            try:
                datetime(int(year), int(month), int(day))
                return f"{year}/{month}/{day}"
            except ValueError:
                pass
        
        return "0000/00/00"
    
    def _clean_title(self, filename: str) -> str:
        """Clean filename to create title."""
        # Remove date patterns
        title = re.sub(r'\d{4}[-_]?\d{2}[-_]?\d{2}[-_]?', '', filename)
        # Replace underscores and hyphens with spaces
        title = re.sub(r'[-_]+', ' ', title)
        # Clean up extra spaces
        title = ' '.join(title.split())
        return title if title else filename


# Create singleton instance and register with registry
_text_parser = TextDocumentParser()
ParserRegistry.register(_text_parser)
