"""
Schema Module

Defines metadata schema for vector store.

This module provides data structures for chunk metadata in the vector store.
"""

from typing import Dict, Any, Optional
from dataclasses import dataclass, asdict, field


@dataclass
class ChunkMetadata:
    """
    Metadata schema for a chunk in the vector store.
    
    This class represents the metadata associated with each indexed chunk,
    supporting any document type (SRT, text, markdown, etc.).
    
    Attributes:
        source_id: Unique identifier for the source document
        date: Document date in YYYY/MM/DD format
        title: Document title
        chunk_index: Position of chunk within the document
        chunk_id: Unique identifier for this chunk
        token_count: Number of tokens in the chunk
        filename: Original filename of the source document
        content_type: Type of content (srt, text, markdown, etc.)
    """
    source_id: str = ""
    date: str = ""  # Format: YYYY/MM/DD
    title: str = ""
    chunk_index: int = 0
    chunk_id: str = ""
    token_count: int = 0
    filename: str = ""
    content_type: str = "unknown"  # srt, text, markdown, etc.
    content_hash: str = ""  # SHA-256 hash of content
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ChunkMetadata':
        """
        Create from dictionary.
        """
        # Handle missing content_type for legacy data
        if 'content_type' not in data:
            data = data.copy()
            data['content_type'] = 'srt'  # Legacy data is SRT
        
        # Remove any unknown fields
        valid_fields = {'source_id', 'date', 'title', 'chunk_index', 'chunk_id', 'token_count', 'filename', 'content_type', 'content_hash'}
        data = {k: v for k, v in data.items() if k in valid_fields}
        
        return cls(**data)
    
    def validate(self) -> bool:
        """
        Validate metadata.
        
        Returns:
            True if valid, False otherwise
        """
        # Check required fields
        if not self.source_id:
            return False
        
        if not self.chunk_id:
            return False
        
        # Check date format (basic validation)
        if len(self.date.split('/')) != 3:
            return False
        
        # Check chunk_index is non-negative
        if self.chunk_index < 0:
            return False
        
        # Check token_count is positive
        if self.token_count <= 0:
            return False
        
        return True


def create_metadata_from_chunk(
    chunk: Any,
    source_metadata: Dict[str, Any]
) -> ChunkMetadata:
    """
    Create ChunkMetadata from chunk and source metadata.
    
    Args:
        chunk: Chunk object with chunk_id, chunk_index, token_count, metadata
        source_metadata: Source document metadata dictionary
    
    Returns:
        ChunkMetadata object
    """
    # Extract source metadata
    source_id = source_metadata.get('source_id', 'unknown')
    date = source_metadata.get('date', '0000/00/00')
    title = source_metadata.get('title', '')
    filename = source_metadata.get('filename', '')
    content_type = source_metadata.get('content_type', 'unknown')
    content_hash = source_metadata.get('content_hash', '')
    
    # Extract chunk metadata
    chunk_id = getattr(chunk, 'chunk_id', '')
    chunk_index = getattr(chunk, 'chunk_index', 0)
    token_count = getattr(chunk, 'token_count', 0)
    
    # Get additional metadata from chunk.metadata if available
    chunk_meta = getattr(chunk, 'metadata', {})
    if isinstance(chunk_meta, dict):
        source_id = chunk_meta.get('source_id', source_id)
        date = chunk_meta.get('date', date)
        title = chunk_meta.get('title', title)
        filename = chunk_meta.get('filename', filename)
        content_type = chunk_meta.get('content_type', content_type)
        content_hash = chunk_meta.get('content_hash', content_hash)
    
    return ChunkMetadata(
        source_id=source_id,
        date=date,
        title=title,
        chunk_index=chunk_index,
        chunk_id=chunk_id,
        token_count=token_count,
        filename=filename,
        content_type=content_type,
        content_hash=content_hash,
    )


def metadata_to_chromadb_format(metadata: ChunkMetadata) -> Dict[str, Any]:
    """
    Convert metadata to ChromaDB-compatible format.
    
    ChromaDB metadata must have string, int, or float values.
    
    Args:
        metadata: ChunkMetadata object
    
    Returns:
        Dictionary compatible with ChromaDB metadata format
    """
    return {
        "source_id": str(metadata.source_id),
        "date": str(metadata.date),
        "title": str(metadata.title),
        "chunk_index": int(metadata.chunk_index),
        "chunk_id": str(metadata.chunk_id),
        "token_count": int(metadata.token_count),
        "filename": str(metadata.filename),
        "content_type": str(metadata.content_type),
        "content_hash": str(metadata.content_hash),
    }


def chromadb_metadata_to_schema(data: Dict[str, Any]) -> ChunkMetadata:
    """
    Convert ChromaDB metadata format to ChunkMetadata.
    
    Args:
        data: ChromaDB metadata dictionary
    
    Returns:
        ChunkMetadata object
    """
    source_id = data.get('source_id', 'unknown')
    
    return ChunkMetadata(
        source_id=str(source_id),
        date=str(data.get('date', '0000/00/00')),
        title=str(data.get('title', '')),
        chunk_index=int(data.get('chunk_index', 0)),
        chunk_id=str(data.get('chunk_id', '')),
        token_count=int(data.get('token_count', 0)),
        filename=str(data.get('filename', '')),
        content_type=str(data.get('content_type', 'srt')),  # Default to srt for legacy data
        content_hash=str(data.get('content_hash', '')),
    )
