"""
Similarity Search Module

Implements similarity search in ChromaDB with filtering and ranking capabilities.
"""

from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime

import numpy as np

from ..embeddings.embedder import Embedder
from ..vector_store.chroma_manager import ChromaDBManager
from ..vector_store.schema import ChunkMetadata, chromadb_metadata_to_schema
from ..utils.logger import get_default_logger


@dataclass
class SearchResult:
    """Represents a single search result."""
    id: str
    text: str
    similarity_score: float
    distance: float
    metadata: ChunkMetadata
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "id": self.id,
            "text": self.text,
            "similarity_score": self.similarity_score,
            "distance": self.distance,
            "metadata": self.metadata.to_dict()
        }


@dataclass
class SearchFilters:
    """
    Search filters for narrowing down results.
    """
    source_id: Optional[str] = None
    date_start: Optional[str] = None  # Format: YYYY/MM/DD
    date_end: Optional[str] = None    # Format: YYYY/MM/DD
    title_keywords: Optional[List[str]] = None
    content_type: Optional[str] = None  # Filter by document type (srt, text, markdown, etc.)
    
    def to_chromadb_where(self) -> Optional[Dict[str, Any]]:
        """
        Convert filters to ChromaDB where clause.
        
        Returns:
            ChromaDB where clause dictionary or None
        """
        conditions = []
        
        if self.source_id:
            conditions.append({"source_id": {"$eq": self.source_id}})
        
        if self.date_start:
            conditions.append({"date": {"$gte": self.date_start}})
        
        if self.date_end:
            conditions.append({"date": {"$lte": self.date_end}})
        
        if self.content_type:
            conditions.append({"content_type": {"$eq": self.content_type}})
        
        if self.title_keywords:
            # ChromaDB doesn't support text search directly, so we'll filter post-query
            # Store keywords for post-filtering
            pass
        
        if not conditions:
            return None
        
        if len(conditions) == 1:
            return conditions[0]
        
        # Combine conditions with AND
        return {"$and": conditions}


class SimilaritySearch:
    """Implements similarity search in ChromaDB."""
    
    def __init__(
        self,
        chroma_manager: Optional[ChromaDBManager] = None,
        embedder: Optional[Embedder] = None,
        model_name: Optional[str] = None
    ):
        """
        Initialize similarity search.

        Args:
            chroma_manager: ChromaDBManager instance (creates new if None)
            embedder: Embedder instance (creates new if None)
            model_name: Embedding model name (uses config default if None)
        """
        self.chroma_manager = chroma_manager or ChromaDBManager()
        self.embedder = embedder or Embedder()
        self.model_name = model_name
        self.logger = get_default_logger()

        # Ensure collection is initialized
        self._collection = None
    
    @property
    def collection(self):
        """Get ChromaDB collection (lazy initialization)."""
        if self._collection is None:
            self._collection = self.chroma_manager.get_or_create_collection(model_name=self.model_name)
        return self._collection
    
    def search(
        self,
        query: str,
        top_k: int = 10,
        score_threshold: Optional[float] = None,
        include_metadata: bool = True,
        filters: Optional[SearchFilters] = None
    ) -> List[SearchResult]:
        """
        Perform similarity search.
        
        Args:
            query: Query text
            top_k: Number of results to return (default: 10)
            score_threshold: Minimum similarity score (0.0-1.0). Results below this are filtered out.
            include_metadata: Whether to include metadata in results (default: True)
            filters: SearchFilters object for filtering results
        
        Returns:
            List of SearchResult objects, sorted by similarity score (descending)
        """
        if not query or not query.strip():
            raise ValueError("Query cannot be empty")
        
        if top_k < 1:
            raise ValueError(f"top_k must be >= 1, got {top_k}")
        
        if score_threshold is not None and not (0.0 <= score_threshold <= 1.0):
            raise ValueError(f"score_threshold must be between 0.0 and 1.0, got {score_threshold}")
        
        self.logger.debug(f"Searching for: '{query}' (top_k={top_k}, threshold={score_threshold})")
        
        try:
            # Generate query embedding
            query_embedding = self.embedder.encode_single(query, is_query=True)

            # Validate query embedding dimensions
            self._validate_query_embedding(query_embedding)

            # Build ChromaDB query parameters
            query_params = {
                "query_embeddings": [query_embedding.tolist()],
                "n_results": top_k * 2 if filters and filters.title_keywords else top_k,  # Get more if we need to filter
                "include": ["documents", "metadatas", "distances"]
            }
            
            # Add where clause if filters provided
            if filters:
                where_clause = filters.to_chromadb_where()
                if where_clause:
                    query_params["where"] = where_clause
            
            # Perform search
            results = self.collection.query(**query_params)
            
            # Process results
            search_results = self._process_results(
                results,
                score_threshold=score_threshold,
                include_metadata=include_metadata,
                filters=filters
            )
            
            # Sort by similarity score (descending) and limit to top_k
            search_results = sorted(
                search_results,
                key=lambda x: x.similarity_score,
                reverse=True
            )[:top_k]
            
            self.logger.debug(f"Found {len(search_results)} results after filtering")
            return search_results
        
        except Exception as e:
            self.logger.error(f"Error during search: {e}", exc_info=True)
            raise
    
    def _process_results(
        self,
        results: Dict[str, Any],
        score_threshold: Optional[float] = None,
        include_metadata: bool = True,
        filters: Optional[SearchFilters] = None
    ) -> List[SearchResult]:
        """
        Process ChromaDB query results into SearchResult objects.
        
        Args:
            results: ChromaDB query results dictionary
            score_threshold: Minimum similarity score
            include_metadata: Whether to include metadata
            filters: SearchFilters for post-filtering
        
        Returns:
            List of SearchResult objects
        """
        if not results.get('ids') or not results['ids'][0]:
            return []
        
        search_results = []
        
        ids = results['ids'][0]
        documents = results.get('documents', [[]])[0]
        metadatas = results.get('metadatas', [[]])[0]
        distances = results.get('distances', [[]])[0]
        
        for i, (doc_id, doc_text, metadata, distance) in enumerate(zip(
            ids,
            documents,
            metadatas,
            distances
        )):
            # Convert distance to similarity score
            # ChromaDB default is Squared L2: distance = 2(1 - similarity) -> similarity = 1 - distance/2
            similarity_score = 1.0 - (distance / 2)
            
            # Apply score threshold
            if score_threshold is not None and similarity_score < score_threshold:
                continue
            
            # Apply title keyword filter (post-query filtering)
            if filters and filters.title_keywords:
                if metadata:
                    title = metadata.get('title', '').lower()
                    if not any(keyword.lower() in title for keyword in filters.title_keywords):
                        continue
            
            # Convert metadata to ChunkMetadata schema
            chunk_metadata = None
            if include_metadata and metadata:
                try:
                    chunk_metadata = chromadb_metadata_to_schema(metadata)
                except Exception as e:
                    self.logger.warning(f"Failed to parse metadata for result {doc_id}: {e}")
                    # Create minimal metadata
                    source_id = metadata.get('source_id', 'unknown')
                    chunk_metadata = ChunkMetadata(
                        source_id=source_id,
                        date=metadata.get('date', '0000/00/00'),
                        title=metadata.get('title', ''),
                        chunk_index=int(metadata.get('chunk_index', 0)),
                        chunk_id=metadata.get('chunk_id', doc_id),
                        token_count=int(metadata.get('token_count', 0)),
                        filename=metadata.get('filename', ''),
                        content_type=metadata.get('content_type', 'srt')
                    )
            
            search_result = SearchResult(
                id=doc_id,
                text=doc_text or "",
                similarity_score=similarity_score,
                distance=distance,
                metadata=chunk_metadata or ChunkMetadata(
                    source_id='unknown',
                    date='0000/00/00',
                    title='',
                    chunk_index=0,
                    chunk_id=doc_id,
                    token_count=0,
                    filename='',
                    content_type='unknown'
                )
            )
            
            search_results.append(search_result)
        
        return search_results
    
    def search_by_embedding(
        self,
        query_embedding: np.ndarray,
        top_k: int = 10,
        score_threshold: Optional[float] = None,
        include_metadata: bool = True,
        filters: Optional[SearchFilters] = None
    ) -> List[SearchResult]:
        """
        Perform similarity search using a pre-computed embedding.
        
        Args:
            query_embedding: Pre-computed query embedding (numpy array)
            top_k: Number of results to return
            score_threshold: Minimum similarity score
            include_metadata: Whether to include metadata
            filters: SearchFilters object for filtering results
        
        Returns:
            List of SearchResult objects, sorted by similarity score (descending)
        """
        if query_embedding is None or len(query_embedding) == 0:
            raise ValueError("Query embedding cannot be empty")
        
        if top_k < 1:
            raise ValueError(f"top_k must be >= 1, got {top_k}")

        # Validate query embedding
        self._validate_query_embedding(query_embedding)

        self.logger.debug(f"Searching by embedding (top_k={top_k}, threshold={score_threshold})")

        try:
            # Build ChromaDB query parameters
            query_params = {
                "query_embeddings": [query_embedding.tolist()],
                "n_results": top_k * 2 if filters and filters.title_keywords else top_k,
                "include": ["documents", "metadatas", "distances"]
            }
            
            # Add where clause if filters provided
            if filters:
                where_clause = filters.to_chromadb_where()
                if where_clause:
                    query_params["where"] = where_clause
            
            # Perform search
            results = self.collection.query(**query_params)
            
            # Process results
            search_results = self._process_results(
                results,
                score_threshold=score_threshold,
                include_metadata=include_metadata,
                filters=filters
            )
            
            # Sort by similarity score (descending) and limit to top_k
            search_results = sorted(
                search_results,
                key=lambda x: x.similarity_score,
                reverse=True
            )[:top_k]
            
            return search_results
        
        except Exception as e:
            self.logger.error(f"Error during search by embedding: {e}", exc_info=True)
            raise
    
    def batch_search(
        self,
        queries: List[str],
        top_k: int = 10,
        score_threshold: Optional[float] = None,
        include_metadata: bool = True,
        filters: Optional[SearchFilters] = None
    ) -> List[List[SearchResult]]:
        """
        Perform batch similarity search for multiple queries.
        
        Args:
            queries: List of query texts
            top_k: Number of results per query
            score_threshold: Minimum similarity score
            include_metadata: Whether to include metadata
            filters: SearchFilters object for filtering results
        
        Returns:
            List of SearchResult lists, one per query
        """
        if not queries:
            return []
        
        # Generate embeddings for all queries
        query_embeddings = self.embedder.encode(queries, is_query=True)
        
        # Perform batch search
        results_list = []
        for query_embedding in query_embeddings:
            search_results = self.search_by_embedding(
                query_embedding=query_embedding,
                top_k=top_k,
                score_threshold=score_threshold,
                include_metadata=include_metadata,
                filters=filters
            )
            results_list.append(search_results)
        
        return results_list
    
    def get_collection_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the collection.

        Returns:
            Dictionary with collection statistics
        """
        return self.chroma_manager.get_collection_stats()

    def set_model(self, model_name: str) -> None:
        """
        Set the embedding model to use for search.

        Args:
            model_name: Name of the embedding model to use
        """
        if self.model_name != model_name:
            self.model_name = model_name

            # Update embedder with new model
            self.embedder.set_model(model_name)

            # Reset collection so it will be reloaded with new model
            self._collection = None

            self.logger.info(f"Switched similarity search to model: {model_name}")

    def _validate_query_embedding(self, embedding: np.ndarray) -> None:
        """
        Validate query embedding dimensions and values.

        Args:
            embedding: Query embedding array

        Raises:
            ValueError: If embedding validation fails
        """
        # Get expected dimension from model
        model_name = self.model_name or self.embedder.model_name
        try:
            registry = get_model_registry()
            metadata = registry.get_model_metadata(model_name)

            if metadata:
                expected_dim = metadata.embedding_dimension
            else:
                # Fallback to embedder dimension
                expected_dim = self.embedder.get_embedding_dimension()

        except Exception:
            # Final fallback - just check if it's a reasonable vector
            expected_dim = None

        # Basic validation
        if embedding.ndim != 1:
            raise ValueError(
                f"Query embedding must be 1D array, got {embedding.ndim}D with shape {embedding.shape}"
            )

        if len(embedding) == 0:
            raise ValueError("Query embedding is empty")

        # Check dimension if we know what to expect
        if expected_dim is not None and len(embedding) != expected_dim:
            raise ValueError(
                f"Query embedding dimension mismatch: expected {expected_dim} for model '{model_name}', "
                f"got {len(embedding)}. "
                f"This suggests you're using the wrong model for querying. "
                f"Please ensure the query model matches the indexing model."
            )

        # Check for invalid values
        if np.isnan(embedding).any():
            raise ValueError("Query embedding contains NaN values")

        if not np.isfinite(embedding).all():
            raise ValueError("Query embedding contains infinite values")

        self.logger.debug(
            f"Query embedding validation passed: {len(embedding)} dimensions"
        )
