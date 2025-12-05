"""
Docling Parser Module

This module implements the DoclingParser class, which uses the IBM docling library
to parse various document formats into a unified TextEntry format.
"""

from pathlib import Path
from typing import List, Tuple, Dict, Any, Optional
import logging

try:
    from docling.document_converter import DocumentConverter
    from docling.datamodel.base_models import InputFormat
    from docling.datamodel.document import DoclingDocument
except ImportError:
    # Fallback for when docling is not installed (e.g. in CI or before install)
    DocumentConverter = None
    InputFormat = None
    DoclingDocument = None

from ..parser_base import DocumentParser, TextEntry, SourceMetadata, ParserRegistry

logger = logging.getLogger(__name__)

class DoclingParser(DocumentParser):
    """
    Universal document parser using IBM's Docling library.
    
    Supports PDF, DOCX, PPTX, HTML, Markdown, and more.
    """
    
    def __init__(self):
        if DocumentConverter is None:
            raise ImportError("docling library is not installed. Please install it with `pip install docling`.")
        self.converter = DocumentConverter()
        
    @property
    def supported_extensions(self) -> List[str]:
        """
        Return list of supported file extensions.
        Docling supports a wide range of formats.
        """
        return [
            '.pdf',
            '.docx',
            '.pptx',
            '.html',
            '.htm',
            '.md',
            '.markdown',
            '.txt',
            '.asciidoc',
            '.adoc',
            '.srt',
            '.sub'
        ]

    def parse(self, file_path: Path) -> Tuple[List[TextEntry], SourceMetadata]:
        """
        Parse document using Docling and return text entries with metadata.
        
        Args:
            file_path: Path to the document file
            
        Returns:
            Tuple of (list of TextEntry objects, SourceMetadata)
        """
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
            
        try:
            # Check for SRT/SUB files first to avoid Docling error logs
            if file_path.suffix.lower() in ['.srt', '.sub']:
                return self._parse_srt_manual(file_path), self.extract_metadata(file_path)

            # Convert document
            logger.info(f"Parsing file with Docling: {file_path}")
            conversion_result = self.converter.convert(file_path)
            doc: DoclingDocument = conversion_result.document
            
            entries = []
            
            # Iterate over texts in the document
            # Docling provides a structured view of the document
            for i, text_item in enumerate(doc.texts()):
                # Extract page number if available
                page_no = None
                if text_item.prov and len(text_item.prov) > 0:
                    page_no = text_item.prov[0].page_no
                
                # Create extra metadata
                extra = {
                    "type": text_item.label if hasattr(text_item, "label") else "text",
                    "path": text_item.self_ref if hasattr(text_item, "self_ref") else None
                }
                
                if page_no:
                    extra["page"] = page_no
                
                # Create TextEntry
                entry = TextEntry(
                    sequence=i,
                    text=text_item.text,
                    extra=extra
                )
                entries.append(entry)
                
            metadata = self.extract_metadata(file_path)
            # Enhance metadata with Docling info if available
            metadata.extra["docling_name"] = doc.name
            metadata.extra["page_count"] = len(doc.pages) if hasattr(doc, "pages") else None
            
            return entries, metadata
            
        except Exception as e:
            # Fallback for SRT/SUB files if Docling fails or rejects them
            if file_path.suffix.lower() in ['.srt', '.sub']:
                logger.info(f"Docling failed on {file_path.suffix}, attempting manual parsing.")
                try:
                    return self._parse_srt_manual(file_path), self.extract_metadata(file_path)
                except Exception as manual_e:
                    logger.error(f"Manual SRT parsing also failed: {manual_e}")
                    raise ValueError(f"Failed to parse {file_path}: {str(e)} -> {str(manual_e)}")
            
            logger.error(f"Error parsing {file_path} with Docling: {str(e)}")
            raise ValueError(f"Failed to parse {file_path}: {str(e)}")

    def _parse_srt_manual(self, file_path: Path) -> List[TextEntry]:
        """
        Manually parse SRT files since Docling might not support them directly.
        """
        entries = []
        with open(file_path, 'r', encoding='utf-8', errors='replace') as f:
            content = f.read()
        
        # Split by double newlines to get blocks
        blocks = content.strip().split('\n\n')
        
        for i, block in enumerate(blocks):
            lines = [line.strip() for line in block.split('\n') if line.strip()]
            if not lines:
                continue
                
            # Try to identify parts
            # Standard SRT:
            # 1
            # 00:00:01,000 --> 00:00:04,000
            # Text line 1
            # Text line 2
            
            text_start_idx = 0
            timestamps = None
            
            # Check if first line is a number (sequence)
            if lines[0].isdigit():
                text_start_idx = 1
            
            # Check if next line is timestamp
            if text_start_idx < len(lines) and '-->' in lines[text_start_idx]:
                timestamps = lines[text_start_idx]
                text_start_idx += 1
            
            # The rest is text
            text_lines = lines[text_start_idx:]
            text = ' '.join(text_lines)
            
            if text:
                entries.append(TextEntry(
                    sequence=i,
                    text=text,
                    extra={
                        "type": "subtitle",
                        "timestamps": timestamps
                    }
                ))
        
        return entries

    def extract_metadata(self, file_path: Path) -> SourceMetadata:
        """
        Extract metadata from file.
        
        Generates source_id from file content hash to prevent duplicates.
        """
        import datetime
        import hashlib
        import logging
        
        logger = logging.getLogger(__name__)
        
        stats = file_path.stat()
        modified_time = datetime.datetime.fromtimestamp(stats.st_mtime).strftime("%Y/%m/%d")
        
        # Generate source_id from content hash (for deduplication)
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            content_hash = hashlib.sha256(content.encode('utf-8')).hexdigest()[:16]
            source_id = f"{file_path.stem}_{content_hash}"
            logger.debug(f"Generated content-based source_id: {source_id}")
        except Exception as e:
            logger.warning(f"Could not hash file {file_path}, using filename only: {e}")
            source_id = file_path.stem
        
        return SourceMetadata(
            source_id=source_id,
            title=file_path.stem,
            date=modified_time,
            source_type=file_path.suffix.lower().lstrip('.'),
            original_filename=file_path.name,
            file_path=str(file_path.absolute())
        )

# Register the parser
# Note: This registration happens when the module is imported.
# To ensure it overrides other parsers, import this module AFTER others or explicitly.
try:
    parser = DoclingParser()
    ParserRegistry.register(parser)
except ImportError:
    pass
