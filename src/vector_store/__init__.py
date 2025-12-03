"""
Vector store module for managing embeddings in ChromaDB.

Handles database setup, indexing, and metadata management.
Supports any document type (SRT, text, markdown, etc.).
"""

from .chroma_manager import ChromaDBManager
from .indexer import Indexer
from .schema import (
    ChunkMetadata,
    create_metadata_from_chunk,
    metadata_to_chromadb_format,
    chromadb_metadata_to_schema,
)
from .pipeline import VectorStorePipeline
from .migrations import SchemaMigrator

__all__ = [
    "ChromaDBManager",
    "Indexer",
    "ChunkMetadata",
    "create_metadata_from_chunk",
    "metadata_to_chromadb_format",
    "chromadb_metadata_to_schema",
    "VectorStorePipeline",
    "SchemaMigrator",
]
