"""
Markdown Document Parser

Parser for Markdown files (.md), implementing the DocumentParser interface.
"""

import hashlib
import re
from pathlib import Path
from typing import List, Optional, Tuple
from datetime import datetime

from ..parser_base import (
    DocumentParser, 
    TextEntry, 
    SourceMetadata, 
    ParserRegistry
)


class MarkdownDocumentParser(DocumentParser):
    """
    Parser for Markdown files.
    
    Preserves document structure by splitting on headers.
    Extracts title from first H1 header.
    """
    
    # Pattern for Markdown headers
    HEADER_PATTERN = re.compile(r'^(#{1,6})\s+(.+)$', re.MULTILINE)
    
    def __init__(
        self,
        split_by: str = "header",
        include_header_in_content: bool = True,
        encoding: Optional[str] = None
    ):
        """
        Initialize Markdown parser.
        
        Args:
            split_by: How to split - "header" (by sections) or "paragraph"
            include_header_in_content: Include header text in section content
            encoding: File encoding. If None, will auto-detect.
        """
        self.split_by = split_by
        self.include_header_in_content = include_header_in_content
        self.encoding = encoding
    
    @property
    def supported_extensions(self) -> List[str]:
        """Return list of supported file extensions."""
        return ['.md', '.markdown', '.mdown']
    
    @property
    def parser_name(self) -> str:
        """Return human-readable parser name."""
        return "Markdown Parser"
    
    def parse(self, file_path: Path) -> Tuple[List[TextEntry], SourceMetadata]:
        """
        Parse Markdown file and return text entries with metadata.
        
        Args:
            file_path: Path to Markdown file
            
        Returns:
            Tuple of (list of TextEntry objects, SourceMetadata)
        """
        if not file_path.exists():
            raise FileNotFoundError(f"Markdown file not found: {file_path}")
        
        # Read file content
        content = self._read_file(file_path)
        
        # Split content into entries
        text_entries = self._split_content(content)
        
        # Extract metadata (including title from content)
        metadata = self.extract_metadata(file_path, content)
        
        return text_entries, metadata
    
    def extract_metadata(
        self, 
        file_path: Path, 
        content: Optional[str] = None
    ) -> SourceMetadata:
        """
        Extract metadata from Markdown file.
        
        Args:
            file_path: Path to Markdown file
            content: Optional pre-read content
            
        Returns:
            SourceMetadata object
        """
        filename = file_path.stem
        full_filename = file_path.name
        
        # Read content if not provided
        if content is None:
            content = self._read_file(file_path)
        
        # Extract title from first H1 header
        title = self._extract_title(content, filename)
        
        # Try to extract date from filename or frontmatter
        date_str = self._extract_date(filename, content)
        
        # Generate source ID from filename hash
        source_id = hashlib.md5(full_filename.encode()).hexdigest()[:11]
        
        # Get file modification date if no date found
        if date_str == "0000/00/00":
            try:
                mtime = file_path.stat().st_mtime
                dt = datetime.fromtimestamp(mtime)
                date_str = dt.strftime("%Y/%m/%d")
            except Exception:
                pass
        
        # Extract any frontmatter metadata
        frontmatter = self._extract_frontmatter(content)
        
        return SourceMetadata(
            source_id=source_id,
            title=title,
            date=date_str,
            source_type="md",
            original_filename=full_filename,
            file_path=str(file_path.absolute()),
            extra={"frontmatter": frontmatter} if frontmatter else {}
        )
    
    def _read_file(self, file_path: Path) -> str:
        """Read file content with encoding detection."""
        encodings = [self.encoding] if self.encoding else ['utf-8', 'utf-8-sig', 'latin-1']
        
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
        
        # Last resort
        with open(file_path, 'rb') as f:
            content = f.read().decode('utf-8', errors='ignore')
        return content
    
    def _split_content(self, content: str) -> List[TextEntry]:
        """Split content into text entries."""
        if self.split_by == "header":
            return self._split_by_header(content)
        else:
            return self._split_by_paragraph(content)
    
    def _split_by_header(self, content: str) -> List[TextEntry]:
        """Split content by headers into sections."""
        # Remove frontmatter first
        content = self._remove_frontmatter(content)
        
        entries = []
        
        # Find all headers
        headers = list(self.HEADER_PATTERN.finditer(content))
        
        if not headers:
            # No headers, treat entire content as one entry
            text = content.strip()
            if text:
                entries.append(TextEntry(
                    sequence=1,
                    text=text,
                    start_position=0,
                    end_position=len(text)
                ))
            return entries
        
        # Handle content before first header
        first_header_pos = headers[0].start()
        if first_header_pos > 0:
            pre_content = content[:first_header_pos].strip()
            if pre_content:
                entries.append(TextEntry(
                    sequence=1,
                    text=pre_content,
                    start_position=0,
                    end_position=first_header_pos,
                    extra={"section": "preamble"}
                ))
        
        # Process each section
        for i, header_match in enumerate(headers):
            header_level = len(header_match.group(1))
            header_text = header_match.group(2).strip()
            
            # Find end of section (next header of same or higher level, or end)
            section_start = header_match.start()
            section_end = len(content)
            
            for next_header in headers[i+1:]:
                next_level = len(next_header.group(1))
                if next_level <= header_level:
                    section_end = next_header.start()
                    break
            
            # Extract section content
            section_content = content[section_start:section_end].strip()
            
            if not self.include_header_in_content:
                # Remove header line from content
                section_content = content[header_match.end():section_end].strip()
            
            if section_content:
                entries.append(TextEntry(
                    sequence=len(entries) + 1,
                    text=section_content,
                    start_position=section_start,
                    end_position=section_end,
                    extra={
                        "section": header_text,
                        "header_level": header_level
                    }
                ))
        
        return entries
    
    def _split_by_paragraph(self, content: str) -> List[TextEntry]:
        """Split content by paragraphs."""
        content = self._remove_frontmatter(content)
        paragraphs = re.split(r'\n\s*\n', content.strip())
        
        entries = []
        position = 0
        
        for para in paragraphs:
            para = para.strip()
            if not para:
                position += 2
                continue
            
            entries.append(TextEntry(
                sequence=len(entries) + 1,
                text=para,
                start_position=position,
                end_position=position + len(para)
            ))
            position += len(para) + 2
        
        return entries
    
    def _extract_title(self, content: str, filename: str) -> str:
        """Extract title from first H1 or filename."""
        # Look for first H1 header
        match = re.search(r'^#\s+(.+)$', content, re.MULTILINE)
        if match:
            return match.group(1).strip()
        
        # Check frontmatter for title
        frontmatter = self._extract_frontmatter(content)
        if frontmatter and 'title' in frontmatter:
            return frontmatter['title']
        
        # Fall back to cleaned filename
        title = re.sub(r'[-_]+', ' ', filename)
        return title
    
    def _extract_date(self, filename: str, content: str) -> str:
        """Extract date from filename or frontmatter."""
        # Check frontmatter first
        frontmatter = self._extract_frontmatter(content)
        if frontmatter and 'date' in frontmatter:
            date_val = frontmatter['date']
            if isinstance(date_val, str):
                # Try to parse common date formats
                for fmt in ['%Y-%m-%d', '%Y/%m/%d', '%d-%m-%Y', '%d/%m/%Y']:
                    try:
                        dt = datetime.strptime(date_val[:10], fmt)
                        return dt.strftime("%Y/%m/%d")
                    except ValueError:
                        continue
        
        # Try YYYYMMDD pattern in filename
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
    
    def _extract_frontmatter(self, content: str) -> dict:
        """Extract YAML frontmatter if present."""
        # Check for YAML frontmatter (--- ... ---)
        match = re.match(r'^---\s*\n(.*?)\n---\s*\n', content, re.DOTALL)
        if not match:
            return {}
        
        frontmatter_text = match.group(1)
        result = {}
        
        # Simple YAML parsing (key: value pairs)
        for line in frontmatter_text.split('\n'):
            line = line.strip()
            if ':' in line:
                key, value = line.split(':', 1)
                key = key.strip().lower()
                value = value.strip().strip('"\'')
                result[key] = value
        
        return result
    
    def _remove_frontmatter(self, content: str) -> str:
        """Remove YAML frontmatter from content."""
        return re.sub(r'^---\s*\n.*?\n---\s*\n', '', content, flags=re.DOTALL)


# Create singleton instance and register with registry
_md_parser = MarkdownDocumentParser()
ParserRegistry.register(_md_parser)
