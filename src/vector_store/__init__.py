"""
Vector store module for managing embeddings in ChromaDB.

Handles database setup, indexing, and metadata management.
"""

from .chroma_manager import ChromaDBManager
from .indexer import Indexer
from .schema import ChunkMetadata, create_metadata_from_chunk, metadata_to_chromadb_format
from .pipeline import VectorStorePipeline
from .migrations import SchemaMigrator

__all__ = [
    "ChromaDBManager",
    "Indexer",
    "ChunkMetadata",
    "create_metadata_from_chunk",
    "metadata_to_chromadb_format",
    "VectorStorePipeline",
    "SchemaMigrator",
]
