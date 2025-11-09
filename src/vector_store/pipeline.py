"""
Vector Store Pipeline Module

Orchestrates the complete vector store indexing pipeline.
"""

import numpy as np
from pathlib import Path
from typing import List, Dict, Optional, Tuple, Any

from .chroma_manager import ChromaDBManager
from .indexer import Indexer
from .schema import ChunkMetadata, chromadb_metadata_to_schema
from ..preprocessing.pipeline import ProcessedVideo
from ..embeddings.pipeline import EmbeddingPipeline
from ..embeddings.embedder import Embedder
from ..utils.logger import get_default_logger
from ..utils.config import get_config


class VectorStorePipeline:
    """Main vector store indexing pipeline."""
    
    def __init__(
        self,
        db_path: Optional[str] = None,
        collection_name: Optional[str] = None
    ):
        """
        Initialize vector store pipeline.
        
        Args:
            db_path: Path to ChromaDB database (default from config)
            collection_name: Collection name (default from config)
        """
        self.config = get_config()
        self.logger = get_default_logger()
        
        # Initialize components
        self.chroma_manager = ChromaDBManager(
            db_path=db_path,
            collection_name=collection_name
        )
        self.indexer = Indexer(chroma_manager=self.chroma_manager)
    
    def index_processed_video(
        self,
        processed_video: ProcessedVideo,
        embeddings: np.ndarray,
        skip_duplicates: bool = True,
        show_progress: bool = True
    ) -> int:
        """
        Index a processed video with embeddings.
        
        Args:
            processed_video: ProcessedVideo object
            embeddings: Embeddings array for chunks
            skip_duplicates: Skip chunks that already exist
            show_progress: Show progress bar
        
        Returns:
            Number of chunks indexed
        """
        if not processed_video.chunks:
            self.logger.warning("No chunks to index")
            return 0
        
        if len(processed_video.chunks) != len(embeddings):
            raise ValueError(
                f"Chunks and embeddings count mismatch: "
                f"{len(processed_video.chunks)} chunks vs {len(embeddings)} embeddings"
            )
        
        # Prepare video metadata
        video_metadata = {
            "video_id": processed_video.metadata.video_id,
            "date": processed_video.metadata.date,
            "title": processed_video.metadata.title,
            "filename": processed_video.metadata.filename,
        }
        
        # Check for duplicates if requested
        if skip_duplicates:
            chunk_ids = [chunk.chunk_id for chunk in processed_video.chunks]
            duplicates = self.indexer.check_duplicates(chunk_ids)
            
            if duplicates:
                self.logger.info(
                    f"Found {len(duplicates)} duplicate chunks, skipping them"
                )
                
                # Filter out duplicates
                filtered_chunks = []
                filtered_embeddings = []
                filtered_indices = []
                
                for i, chunk in enumerate(processed_video.chunks):
                    if chunk.chunk_id not in duplicates:
                        filtered_chunks.append(chunk)
                        filtered_embeddings.append(embeddings[i])
                        filtered_indices.append(i)
                
                if not filtered_chunks:
                    self.logger.info("All chunks are duplicates, nothing to index")
                    return 0
                
                processed_video.chunks = filtered_chunks
                embeddings = np.array(filtered_embeddings)
        
        # Index chunks
        indexed_count = self.indexer.index_chunks(
            chunks=processed_video.chunks,
            embeddings=embeddings,
            video_metadata=video_metadata,
            show_progress=show_progress
        )
        
        self.logger.info(
            f"Indexed {indexed_count} chunks from {processed_video.metadata.filename}"
        )
        
        return indexed_count
    
    def index_multiple_videos(
        self,
        processed_videos: List[ProcessedVideo],
        embeddings_list: List[np.ndarray],
        skip_duplicates: bool = True,
        show_progress: bool = True
    ) -> Dict[str, int]:
        """
        Index multiple processed videos.
        
        Args:
            processed_videos: List of ProcessedVideo objects
            embeddings_list: List of embeddings arrays
            skip_duplicates: Skip chunks that already exist
            show_progress: Show progress bar
        
        Returns:
            Dictionary mapping video IDs to indexed chunk counts
        """
        if len(processed_videos) != len(embeddings_list):
            raise ValueError(
                f"Videos and embeddings count mismatch: "
                f"{len(processed_videos)} videos vs {len(embeddings_list)} embedding arrays"
            )
        
        results = {}
        
        for i, (processed_video, embeddings) in enumerate(
            zip(processed_videos, embeddings_list)
        ):
            self.logger.info(
                f"Indexing video {i+1}/{len(processed_videos)}: "
                f"{processed_video.metadata.filename}"
            )
            
            try:
                indexed_count = self.index_processed_video(
                    processed_video,
                    embeddings,
                    skip_duplicates=skip_duplicates,
                    show_progress=show_progress
                )
                
                results[processed_video.metadata.video_id] = indexed_count
            
            except Exception as e:
                self.logger.error(
                    f"Error indexing video {processed_video.metadata.filename}: {e}",
                    exc_info=True
                )
                results[processed_video.metadata.video_id] = 0
        
        return results
    
    def get_index_statistics(self) -> Dict:
        """
        Get index statistics.
        
        Returns:
            Dictionary with index statistics
        """
        return self.indexer.get_index_stats()
    
    def rebuild_index(self) -> None:
        """
        Rebuild the entire index (delete and recreate).
        
        WARNING: This will delete all existing data!
        """
        self.logger.warning("Rebuilding index - all existing data will be deleted!")
        
        try:
            # Delete collection
            self.chroma_manager.delete_collection()
            
            # Recreate collection
            self.indexer._collection = None  # Reset collection reference
            self.chroma_manager.get_or_create_collection()
            
            self.logger.info("Index rebuilt successfully")
        
        except Exception as e:
            self.logger.error(f"Error rebuilding index: {e}", exc_info=True)
            raise
    
    def health_check(self) -> bool:
        """
        Perform health check on vector store.
        
        Returns:
            True if healthy, False otherwise
        """
        try:
            # Check ChromaDB health
            if not self.chroma_manager.health_check():
                return False
            
            # Check collection exists
            stats = self.get_index_statistics()
            if "error" in stats:
                return False
            
            return True
        
        except Exception as e:
            self.logger.error(f"Health check failed: {e}")
            return False
    
    def verify_index(
        self,
        test_samples: Optional[List[str]] = None,
        num_test_queries: int = 5,
        top_k: int = 10
    ) -> Dict[str, Any]:
        """
        Verify index integrity by querying test samples.
        
        Args:
            test_samples: Optional list of test query strings
            num_test_queries: Number of random test queries if test_samples not provided
            top_k: Number of results to retrieve per query
        
        Returns:
            Dictionary with verification results
        """
        self.logger.info("Starting index verification...")
        
        collection = self.chroma_manager.get_or_create_collection()
        total_docs = collection.count()
        
        if total_docs == 0:
            return {
                "status": "empty",
                "message": "Index is empty, nothing to verify",
                "total_documents": 0
            }
        
        results = {
            "status": "success",
            "total_documents": total_docs,
            "test_queries": [],
            "metadata_retrieval": {},
            "index_integrity": {}
        }
        
        # Generate test queries if not provided
        if not test_samples:
            # Get random samples from the collection
            try:
                sample_data = collection.get(limit=min(num_test_queries, total_docs))
                if sample_data.get("documents"):
                    test_samples = sample_data["documents"][:num_test_queries]
                else:
                    test_samples = ["test query"] * num_test_queries
            except Exception as e:
                self.logger.warning(f"Failed to get sample documents: {e}")
                test_samples = ["test query"] * num_test_queries
        
        # Test queries
        embedder = Embedder()
        query_results = []
        
        for i, query_text in enumerate(test_samples[:num_test_queries]):
            try:
                # Generate query embedding
                query_embedding = embedder.encode([query_text], is_query=True)[0]
                
                # Perform similarity search
                search_results = collection.query(
                    query_embeddings=[query_embedding.tolist()],
                    n_results=top_k
                )
                
                # Verify results
                num_results = len(search_results.get("ids", [])[0] if search_results.get("ids") else [])
                has_metadata = bool(search_results.get("metadatas") and search_results["metadatas"][0])
                has_documents = bool(search_results.get("documents") and search_results["documents"][0])
                
                query_results.append({
                    "query": query_text[:50] + "..." if len(query_text) > 50 else query_text,
                    "num_results": num_results,
                    "has_metadata": has_metadata,
                    "has_documents": has_documents,
                    "status": "success" if num_results > 0 else "no_results"
                })
                
            except Exception as e:
                self.logger.error(f"Error testing query {i+1}: {e}")
                query_results.append({
                    "query": query_text[:50] + "..." if len(query_text) > 50 else query_text,
                    "status": "error",
                    "error": str(e)
                })
        
        results["test_queries"] = query_results
        
        # Test metadata retrieval
        try:
            sample_ids = collection.get(limit=min(10, total_docs))["ids"]
            metadata_results = []
            
            for doc_id in sample_ids[:5]:
                try:
                    doc_data = collection.get(ids=[doc_id])
                    if doc_data.get("metadatas") and doc_data["metadatas"][0]:
                        metadata = doc_data["metadatas"][0]
                        # Validate metadata schema
                        is_valid = chromadb_metadata_to_schema(metadata).validate()
                        metadata_results.append({
                            "id": doc_id,
                            "has_metadata": True,
                            "is_valid": is_valid,
                            "fields": list(metadata.keys())
                        })
                    else:
                        metadata_results.append({
                            "id": doc_id,
                            "has_metadata": False,
                            "is_valid": False
                        })
                except Exception as e:
                    metadata_results.append({
                        "id": doc_id,
                        "has_metadata": False,
                        "error": str(e)
                    })
            
            results["metadata_retrieval"] = {
                "tested": len(metadata_results),
                "valid": sum(1 for r in metadata_results if r.get("is_valid", False)),
                "results": metadata_results
            }
        
        except Exception as e:
            self.logger.error(f"Error testing metadata retrieval: {e}")
            results["metadata_retrieval"] = {"error": str(e)}
        
        # Check index integrity
        try:
            # Check for duplicate IDs
            all_ids = collection.get()["ids"]
            unique_ids = set(all_ids)
            duplicate_count = len(all_ids) - len(unique_ids)
            
            # Check embedding dimensions
            sample_embedding = collection.get(limit=1, include=["embeddings"])
            embedding_dim = None
            if sample_embedding.get("embeddings") and sample_embedding["embeddings"][0]:
                embedding_dim = len(sample_embedding["embeddings"][0])
            
            results["index_integrity"] = {
                "total_documents": total_docs,
                "unique_ids": len(unique_ids),
                "duplicate_ids": duplicate_count,
                "embedding_dimension": embedding_dim,
                "is_valid": duplicate_count == 0 and embedding_dim is not None
            }
        
        except Exception as e:
            self.logger.error(f"Error checking index integrity: {e}")
            results["index_integrity"] = {"error": str(e)}
        
        # Overall status
        all_queries_ok = all(q.get("status") == "success" for q in query_results)
        metadata_ok = results["metadata_retrieval"].get("valid", 0) > 0
        integrity_ok = results["index_integrity"].get("is_valid", False)
        
        if all_queries_ok and metadata_ok and integrity_ok:
            results["status"] = "success"
            self.logger.info("Index verification passed")
        else:
            results["status"] = "warning"
            self.logger.warning("Index verification completed with warnings")
        
        return results

