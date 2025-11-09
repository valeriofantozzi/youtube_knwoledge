"""
Cluster Integrator Module

Integrates clustering with the retrieval system for cluster-aware search.
"""

from typing import List, Dict, Any, Optional
import numpy as np

from .cluster_manager import ClusterManager
from ..retrieval.similarity_search import SimilaritySearch, SearchFilters, SearchResult
from ..utils.logger import get_logger


class ClusterIntegrator:
    """
    Integrates clustering with retrieval system.
    
    Provides:
    - Cluster-based filtering in queries
    - Cluster-aware reranking
    - Query expansion using cluster representatives
    - Related content discovery
    """
    
    def __init__(
        self,
        cluster_manager: Optional[ClusterManager] = None,
        similarity_search: Optional[SimilaritySearch] = None
    ):
        """
        Initialize cluster integrator.
        
        Args:
            cluster_manager: ClusterManager instance
            similarity_search: SimilaritySearch instance
        """
        self.logger = get_logger(__name__)
        self.cluster_manager = cluster_manager or ClusterManager()
        self.similarity_search = similarity_search or SimilaritySearch()
    
    def search_by_cluster(
        self,
        cluster_id: int,
        top_k: int = 10,
        score_threshold: Optional[float] = None
    ) -> List[SearchResult]:
        """
        Search for chunks within a specific cluster.
        
        Args:
            cluster_id: Cluster ID to search in
            top_k: Number of results to return
            score_threshold: Minimum similarity score
        
        Returns:
            List of SearchResult objects from the cluster
        """
        # Get chunks from cluster
        cluster_data = self.cluster_manager.get_chunks_by_cluster(
            cluster_id=cluster_id,
            limit=top_k
        )
        
        if not cluster_data.get("ids"):
            return []
        
        # Convert to SearchResult format
        results = []
        ids = cluster_data.get("ids", [])
        documents = cluster_data.get("documents", [])
        metadatas = cluster_data.get("metadatas", [])
        
        for doc_id, doc_text, metadata in zip(ids, documents, metadatas):
            # Create SearchResult (without similarity score since we're not doing search)
            from ..retrieval.similarity_search import SearchResult
            from ..vector_store.schema import chromadb_metadata_to_schema
            
            chunk_metadata = None
            if metadata:
                try:
                    chunk_metadata = chromadb_metadata_to_schema(metadata)
                except Exception:
                    pass
            
            result = SearchResult(
                id=doc_id,
                text=doc_text or "",
                similarity_score=1.0,  # All results from same cluster
                distance=0.0,
                metadata=chunk_metadata
            )
            results.append(result)
        
        return results[:top_k]
    
    def rerank_by_cluster(
        self,
        results: List[SearchResult],
        query_cluster_id: Optional[int] = None,
        boost_same_cluster: float = 1.2
    ) -> List[SearchResult]:
        """
        Rerank search results based on cluster membership.
        
        Args:
            results: List of SearchResult objects
            query_cluster_id: Cluster ID of the query (if known)
            boost_same_cluster: Multiplier for same-cluster results
        
        Returns:
            Reranked list of SearchResult objects
        """
        if query_cluster_id is None:
            # Try to infer query cluster from results
            # Use the most common cluster in top results
            cluster_ids = []
            for result in results[:10]:  # Check top 10
                if result.metadata:
                    # Check if cluster_id is in metadata
                    # Note: This assumes cluster_id is stored in metadata
                    pass  # Would need to access cluster_id from metadata
            
            if cluster_ids:
                from collections import Counter
                query_cluster_id = Counter(cluster_ids).most_common(1)[0][0]
        
        if query_cluster_id is None:
            return results  # No reranking possible
        
        # Boost results from same cluster
        reranked = []
        for result in results:
            # Check if result belongs to query cluster
            # This would require accessing cluster_id from metadata
            # For now, just return original results
            reranked.append(result)
        
        return reranked
    
    def expand_query_with_cluster(
        self,
        query: str,
        cluster_id: int,
        n_expansions: int = 3
    ) -> List[str]:
        """
        Expand query using cluster representatives.
        
        Args:
            query: Original query string
            cluster_id: Cluster ID to use for expansion
            n_expansions: Number of expansion terms to add
        
        Returns:
            List of expanded query strings
        """
        # Get cluster chunks
        cluster_data = self.cluster_manager.get_chunks_by_cluster(
            cluster_id=cluster_id,
            limit=20  # Get more chunks to extract keywords
        )
        
        documents = cluster_data.get("documents", [])
        
        if not documents:
            return [query]
        
        # Extract keywords from cluster documents
        # Simple approach: use most common words
        from collections import Counter
        all_words = []
        for doc in documents:
            words = doc.lower().split()
            # Filter out common stop words
            stop_words = {"the", "a", "an", "and", "or", "but", "in", "on", "at", "to", "for", "of", "with", "by"}
            words = [w for w in words if w not in stop_words and len(w) > 3]
            all_words.extend(words)
        
        word_counts = Counter(all_words)
        top_keywords = [word for word, count in word_counts.most_common(n_expansions)]
        
        # Create expanded queries
        expanded_queries = [query]
        for keyword in top_keywords:
            expanded_queries.append(f"{query} {keyword}")
        
        return expanded_queries
    
    def discover_related_clusters(
        self,
        cluster_id: int,
        top_k: int = 5
    ) -> List[Dict[str, Any]]:
        """
        Discover clusters related to a given cluster.
        
        Args:
            cluster_id: Source cluster ID
            top_k: Number of related clusters to return
        
        Returns:
            List of dictionaries with cluster_id and similarity score
        """
        # Get cluster statistics
        stats = self.cluster_manager.get_cluster_statistics()
        
        if cluster_id not in stats:
            return []
        
        source_cluster = stats[cluster_id]
        source_centroid = source_cluster.centroid
        
        if source_centroid is None:
            return []
        
        # Calculate similarity to other clusters
        similarities = []
        for other_id, other_cluster in stats.items():
            if other_id == cluster_id or other_cluster.centroid is None:
                continue
            
            # Calculate cosine similarity between centroids
            dot_product = np.dot(source_centroid, other_cluster.centroid)
            norm_product = np.linalg.norm(source_centroid) * np.linalg.norm(other_cluster.centroid)
            
            if norm_product > 0:
                similarity = dot_product / norm_product
                similarities.append({
                    "cluster_id": other_id,
                    "similarity": float(similarity),
                    "size": other_cluster.size
                })
        
        # Sort by similarity
        similarities.sort(key=lambda x: x["similarity"], reverse=True)
        
        return similarities[:top_k]
    
    def get_cluster_context(
        self,
        cluster_id: int
    ) -> Dict[str, Any]:
        """
        Get context information for a cluster.
        
        Args:
            cluster_id: Cluster ID
        
        Returns:
            Dictionary with cluster context (keywords, theme, related clusters, etc.)
        """
        stats = self.cluster_manager.get_cluster_statistics()
        
        if cluster_id not in stats:
            return {}
        
        cluster_meta = stats[cluster_id]
        
        # Get related clusters
        related = self.discover_related_clusters(cluster_id, top_k=3)
        
        return {
            "cluster_id": cluster_id,
            "size": cluster_meta.size,
            "video_ids": list(cluster_meta.video_ids) if cluster_meta.video_ids else [],
            "keywords": cluster_meta.keywords or [],
            "theme": cluster_meta.theme,
            "related_clusters": related
        }

