"""
Schema Module

Defines metadata schema for vector store.
"""

from typing import Dict, Any, Optional
from dataclasses import dataclass, asdict


@dataclass
class ChunkMetadata:
    """Metadata schema for a chunk in the vector store."""
    video_id: str
    date: str  # Format: YYYY/MM/DD
    title: str
    chunk_index: int
    chunk_id: str
    token_count: int
    filename: str
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ChunkMetadata':
        """Create from dictionary."""
        return cls(**data)
    
    def validate(self) -> bool:
        """
        Validate metadata.
        
        Returns:
            True if valid, False otherwise
        """
        # Check required fields
        if not self.video_id:
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
    video_metadata: Dict[str, Any]
) -> ChunkMetadata:
    """
    Create ChunkMetadata from chunk and video metadata.
    
    Args:
        chunk: Chunk object with chunk_id, chunk_index, token_count, metadata
        video_metadata: Video metadata dictionary
    
    Returns:
        ChunkMetadata object
    """
    # Extract video metadata
    video_id = video_metadata.get('video_id', 'unknown')
    date = video_metadata.get('date', '0000/00/00')
    title = video_metadata.get('title', '')
    filename = video_metadata.get('filename', '')
    
    # Extract chunk metadata
    chunk_id = getattr(chunk, 'chunk_id', '')
    chunk_index = getattr(chunk, 'chunk_index', 0)
    token_count = getattr(chunk, 'token_count', 0)
    
    # Get additional metadata from chunk.metadata if available
    chunk_meta = getattr(chunk, 'metadata', {})
    if isinstance(chunk_meta, dict):
        video_id = chunk_meta.get('video_id', video_id)
        date = chunk_meta.get('date', date)
        title = chunk_meta.get('title', title)
        filename = chunk_meta.get('filename', filename)
    
    return ChunkMetadata(
        video_id=video_id,
        date=date,
        title=title,
        chunk_index=chunk_index,
        chunk_id=chunk_id,
        token_count=token_count,
        filename=filename
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
        "video_id": str(metadata.video_id),
        "date": str(metadata.date),
        "title": str(metadata.title),
        "chunk_index": int(metadata.chunk_index),
        "chunk_id": str(metadata.chunk_id),
        "token_count": int(metadata.token_count),
        "filename": str(metadata.filename),
    }


def chromadb_metadata_to_schema(data: Dict[str, Any]) -> ChunkMetadata:
    """
    Convert ChromaDB metadata format to ChunkMetadata.
    
    Args:
        data: ChromaDB metadata dictionary
    
    Returns:
        ChunkMetadata object
    """
    return ChunkMetadata(
        video_id=str(data.get('video_id', 'unknown')),
        date=str(data.get('date', '0000/00/00')),
        title=str(data.get('title', '')),
        chunk_index=int(data.get('chunk_index', 0)),
        chunk_id=str(data.get('chunk_id', '')),
        token_count=int(data.get('token_count', 0)),
        filename=str(data.get('filename', '')),
    )
