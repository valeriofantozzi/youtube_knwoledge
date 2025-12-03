"""
Indexer Module

Indexes embeddings with metadata in ChromaDB.
"""

import numpy as np
from typing import List, Dict, Optional, Tuple, Any
from tqdm import tqdm

from .chroma_manager import ChromaDBManager
from .schema import (
    ChunkMetadata,
    create_metadata_from_chunk,
    metadata_to_chromadb_format
)
from ..utils.config import get_config
from ..utils.logger import get_default_logger


class Indexer:
    """Indexes embeddings in ChromaDB."""
    
    def __init__(
        self,
        chroma_manager: Optional[ChromaDBManager] = None,
        collection_name: Optional[str] = None
    ):
        """
        Initialize indexer.
        
        Args:
            chroma_manager: ChromaDBManager instance (creates new if None)
            collection_name: Collection name (default from config)
        """
        self.chroma_manager = chroma_manager or ChromaDBManager(
            collection_name=collection_name
        )
        self.config = get_config()
        self.logger = get_default_logger()
        self._collection = None
    
    @property
    def collection(self):
        """Get collection (lazy loading)."""
        if self._collection is None:
            self._collection = self.chroma_manager.get_or_create_collection()
        return self._collection
    
    def index_chunks(
        self,
        chunks: List,
        embeddings: np.ndarray,
        source_metadata: Dict[str, Any],
        batch_size: int = 1000,
        show_progress: bool = True,
        collection: Optional[Any] = None
    ) -> int:
        """
        Index chunks with their embeddings.

        Args:
            chunks: List of chunk objects
            embeddings: Embeddings array (shape: [num_chunks, embedding_dim])
            source_metadata: Source document metadata dictionary
            batch_size: Batch size for indexing
            show_progress: Show progress bar
            collection: Optional collection to use (uses default if None)

        Returns:
            Number of chunks indexed
        """
        if len(chunks) != len(embeddings):
            raise ValueError(
                f"Chunks and embeddings count mismatch: "
                f"{len(chunks)} chunks vs {len(embeddings)} embeddings"
            )
        
        if len(chunks) == 0:
            self.logger.warning("No chunks to index")
            return 0
        
        self.logger.info(f"Indexing {len(chunks)} chunks")
        
        # Prepare data for indexing
        ids = []
        metadatas = []
        documents = []
        
        for chunk in chunks:
            # Create metadata
            chunk_metadata = create_metadata_from_chunk(chunk, source_metadata)
            
            # Validate metadata
            if not chunk_metadata.validate():
                self.logger.warning(
                    f"Invalid metadata for chunk {chunk_metadata.chunk_id}, skipping"
                )
                continue
            
            # Prepare data
            ids.append(chunk_metadata.chunk_id)
            metadatas.append(metadata_to_chromadb_format(chunk_metadata))
            documents.append(chunk.text)
        
        # Index in batches
        indexed_count = self._index_batch(
            ids=ids,
            embeddings=embeddings,
            metadatas=metadatas,
            documents=documents,
            batch_size=batch_size,
            show_progress=show_progress,
            collection=collection
        )
        
        self.logger.info(f"Indexed {indexed_count} chunks")
        return indexed_count
    
    def _index_batch(
        self,
        ids: List[str],
        embeddings: np.ndarray,
        metadatas: List[Dict],
        documents: List[str],
        batch_size: int = 1000,
        show_progress: bool = True,
        collection: Optional[Any] = None
    ) -> int:
        """
        Index data in batches.

        Args:
            ids: List of document IDs
            embeddings: Embeddings array
            metadatas: List of metadata dictionaries
            documents: List of document texts
            batch_size: Batch size
            show_progress: Show progress bar
            collection: Optional collection to use (uses default if None)

        Returns:
            Number of documents indexed
        """
        collection = collection or self.collection
        total = len(ids)
        indexed = 0
        
        if show_progress:
            pbar = tqdm(total=total, desc="Indexing", unit="chunks")
        
        for i in range(0, total, batch_size):
            end_idx = min(i + batch_size, total)
            
            batch_ids = ids[i:end_idx]
            batch_embeddings = embeddings[i:end_idx]
            batch_metadatas = metadatas[i:end_idx]
            batch_documents = documents[i:end_idx]
            
            try:
                # Add to collection
                collection.add(
                    ids=batch_ids,
                    embeddings=batch_embeddings.tolist(),
                    metadatas=batch_metadatas,
                    documents=batch_documents
                )
                
                indexed += len(batch_ids)
                
                if show_progress:
                    pbar.update(len(batch_ids))
            
            except Exception as e:
                self.logger.error(
                    f"Error indexing batch {i//batch_size + 1}: {e}",
                    exc_info=True
                )
                # Continue with next batch
                continue
        
        if show_progress:
            pbar.close()
        
        return indexed
    
    def update_chunks(
        self,
        chunk_ids: List[str],
        embeddings: Optional[np.ndarray] = None,
        metadatas: Optional[List[Dict]] = None,
        documents: Optional[List[str]] = None
    ) -> int:
        """
        Update existing chunks.
        
        Args:
            chunk_ids: List of chunk IDs to update
            embeddings: Optional new embeddings
            metadatas: Optional new metadata
            documents: Optional new documents
        
        Returns:
            Number of chunks updated
        """
        collection = self.collection
        
        try:
            # ChromaDB uses update method
            collection.update(
                ids=chunk_ids,
                embeddings=embeddings.tolist() if embeddings is not None else None,
                metadatas=metadatas,
                documents=documents
            )
            
            self.logger.info(f"Updated {len(chunk_ids)} chunks")
            return len(chunk_ids)
        
        except Exception as e:
            self.logger.error(f"Error updating chunks: {e}", exc_info=True)
            raise
    
    def delete_chunks(self, chunk_ids: List[str]) -> int:
        """
        Delete chunks by IDs.
        
        Args:
            chunk_ids: List of chunk IDs to delete
        
        Returns:
            Number of chunks deleted
        """
        collection = self.collection
        
        try:
            collection.delete(ids=chunk_ids)
            self.logger.info(f"Deleted {len(chunk_ids)} chunks")
            return len(chunk_ids)
        
        except Exception as e:
            self.logger.error(f"Error deleting chunks: {e}", exc_info=True)
            raise
    
    def get_index_stats(self) -> Dict:
        """
        Get indexing statistics.
        
        Returns:
            Dictionary with statistics
        """
        try:
            stats = self.chroma_manager.get_collection_stats()
            return stats
        except Exception as e:
            self.logger.error(f"Error getting index stats: {e}")
            return {"error": str(e)}
    
    def check_duplicates(self, chunk_ids: List[str]) -> List[str]:
        """
        Check which chunk IDs already exist in the index.
        
        Args:
            chunk_ids: List of chunk IDs to check
        
        Returns:
            List of IDs that already exist
        """
        collection = self.collection
        
        try:
            # Get existing IDs
            existing = collection.get(ids=chunk_ids)
            existing_ids = set(existing['ids']) if existing['ids'] else set()
            
            # Find duplicates
            duplicates = [id for id in chunk_ids if id in existing_ids]
            
            return duplicates
        
        except Exception as e:
            self.logger.warning(f"Error checking duplicates: {e}")
            return []
