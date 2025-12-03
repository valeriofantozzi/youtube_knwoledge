"""
Vector Store Pipeline Module

Orchestrates the complete vector store indexing pipeline.
Supports any document type (SRT, text, markdown, etc.).
"""

import numpy as np
from pathlib import Path
from typing import List, Dict, Optional, Tuple, Any

from .chroma_manager import ChromaDBManager
from .indexer import Indexer
from .schema import ChunkMetadata, chromadb_metadata_to_schema
from ..preprocessing.pipeline import ProcessedDocument
from ..embeddings.pipeline import EmbeddingPipeline
from ..embeddings.embedder import Embedder
from ..embeddings.model_registry import get_model_registry
from ..utils.logger import get_default_logger
from ..utils.config import get_config


class VectorStorePipeline:
    """Main vector store indexing pipeline."""
    
    def __init__(
        self,
        db_path: Optional[str] = None,
        collection_name: Optional[str] = None,
        model_name: Optional[str] = None
    ):
        """
        Initialize vector store pipeline.

        Args:
            db_path: Path to ChromaDB database (default from config)
            collection_name: Collection name (default from config)
            model_name: Embedding model name (default from config)
        """
        self.config = get_config()
        self.logger = get_default_logger()
        self.model_name = model_name

        # Initialize components
        self.chroma_manager = ChromaDBManager(
            db_path=db_path,
            collection_name=collection_name
        )
        self.indexer = Indexer(chroma_manager=self.chroma_manager)
    
    def index_processed_document(
        self,
        processed_document: ProcessedDocument,
        embeddings: np.ndarray,
        skip_duplicates: bool = True,
        show_progress: bool = True,
        model_name: Optional[str] = None
    ) -> int:
        """
        Index a processed document with embeddings.

        Args:
            processed_document: ProcessedDocument object
            embeddings: Embeddings array for chunks
            skip_duplicates: Skip chunks that already exist
            show_progress: Show progress bar
            model_name: Actual model name used to generate embeddings (for validation)

        Returns:
            Number of chunks indexed
        """
        if not processed_document.chunks:
            self.logger.warning("No chunks to index")
            return 0
        
        if len(processed_document.chunks) != len(embeddings):
            raise ValueError(
                f"Chunks and embeddings count mismatch: "
                f"{len(processed_document.chunks)} chunks vs {len(embeddings)} embeddings"
            )
        
        # Prepare source metadata
        source_metadata = {
            "source_id": processed_document.metadata.source_id,
            "date": processed_document.metadata.date,
            "title": processed_document.metadata.title,
            "filename": processed_document.metadata.filename,
            "content_type": processed_document.metadata.content_type,
        }
        
        # Check for duplicates if requested
        if skip_duplicates:
            chunk_ids = [chunk.chunk_id for chunk in processed_document.chunks]
            duplicates = self.indexer.check_duplicates(chunk_ids)
            
            if duplicates:
                self.logger.info(
                    f"Found {len(duplicates)} duplicate chunks, skipping them"
                )
                
                # Filter out duplicates
                filtered_chunks = []
                filtered_embeddings = []
                filtered_indices = []
                
                for i, chunk in enumerate(processed_document.chunks):
                    if chunk.chunk_id not in duplicates:
                        filtered_chunks.append(chunk)
                        filtered_embeddings.append(embeddings[i])
                        filtered_indices.append(i)
                
                if not filtered_chunks:
                    self.logger.info("All chunks are duplicates, nothing to index")
                    return 0
                
                processed_document.chunks = filtered_chunks
                embeddings = np.array(filtered_embeddings)
        
        # Validate embedding dimensions before indexing
        validation_model_name = model_name or self.model_name
        self._validate_embedding_dimensions(embeddings, validation_model_name)

        # Get or create collection with model-specific name
        collection = self.chroma_manager.get_or_create_collection(
            model_name=validation_model_name
        )

        # Index chunks
        indexed_count = self.indexer.index_chunks(
            chunks=processed_document.chunks,
            embeddings=embeddings,
            source_metadata=source_metadata,
            show_progress=show_progress,
            collection=collection
        )
        
        self.logger.info(
            f"Indexed {indexed_count} chunks from {processed_document.metadata.filename}"
        )
        
        return indexed_count
    
    def index_multiple_documents(
        self,
        processed_documents: List[ProcessedDocument],
        embeddings_list: List[np.ndarray],
        skip_duplicates: bool = True,
        show_progress: bool = True
    ) -> Dict[str, int]:
        """
        Index multiple processed documents.
        
        Args:
            processed_documents: List of ProcessedDocument objects
            embeddings_list: List of embeddings arrays
            skip_duplicates: Skip chunks that already exist
            show_progress: Show progress bar
        
        Returns:
            Dictionary mapping source IDs to indexed chunk counts
        """
        if len(processed_documents) != len(embeddings_list):
            raise ValueError(
                f"Documents and embeddings count mismatch: "
                f"{len(processed_documents)} documents vs {len(embeddings_list)} embedding arrays"
            )
        
        results = {}
        
        for i, (processed_document, embeddings) in enumerate(
            zip(processed_documents, embeddings_list)
        ):
            self.logger.info(
                f"Indexing document {i+1}/{len(processed_documents)}: "
                f"{processed_document.metadata.filename}"
            )
            
            try:
                indexed_count = self.index_processed_document(
                    processed_document,
                    embeddings,
                    skip_duplicates=skip_duplicates,
                    show_progress=show_progress
                )
                
                results[processed_document.metadata.source_id] = indexed_count
            
            except Exception as e:
                self.logger.error(
                    f"Error indexing document {processed_document.metadata.filename}: {e}",
                    exc_info=True
                )
                results[processed_document.metadata.source_id] = 0
        
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

    def _validate_embedding_dimensions(self, embeddings: np.ndarray, model_name: str) -> None:
        """
        Validate that embedding dimensions match the expected dimensions for the model.

        Args:
            embeddings: Embedding array to validate
            model_name: Name of the model used to generate embeddings

        Raises:
            ValueError: If embedding dimensions don't match model expectations
        """
        try:
            # Get expected dimension from model registry
            registry = get_model_registry()
            metadata = registry.get_model_metadata(model_name)

            if metadata:
                expected_dim = metadata.embedding_dimension
            else:
                # Fallback to config
                config = get_config()
                expected_dim = config.get_embedding_dimension()

            # Check embedding shape
            if embeddings.ndim != 2:
                raise ValueError(
                    f"Embeddings must be 2D array, got {embeddings.ndim}D with shape {embeddings.shape}"
                )

            actual_dim = embeddings.shape[1]  # Second dimension is embedding size

            if actual_dim != expected_dim:
                raise ValueError(
                    f"Embedding dimension mismatch for model '{model_name}': "
                    f"expected {expected_dim}, got {actual_dim}. "
                    f"This usually indicates the embeddings were generated with a different model. "
                    f"Please ensure you're using the correct model for generating embeddings."
                )

            # Check for reasonable embedding values
            if np.isnan(embeddings).any():
                raise ValueError("Embeddings contain NaN values")

            if not np.isfinite(embeddings).all():
                raise ValueError("Embeddings contain infinite values")

            self.logger.debug(
                f"Embedding validation passed: {embeddings.shape[0]} embeddings, "
                f"{actual_dim} dimensions for model '{model_name}'"
            )

        except Exception as e:
            self.logger.error(f"Embedding dimension validation failed: {e}")
            raise

