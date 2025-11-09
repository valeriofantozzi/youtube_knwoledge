"""
Cluster Manager Module

Manages cluster storage, updates, and queries in ChromaDB.
"""

from typing import List, Dict, Any, Optional, Set
from dataclasses import dataclass
import numpy as np

from ..vector_store.chroma_manager import ChromaDBManager
from ..utils.logger import get_logger


@dataclass
class ClusterMetadata:
    """Metadata for a cluster."""
    cluster_id: int
    size: int
    centroid: Optional[np.ndarray] = None
    keywords: Optional[List[str]] = None
    theme: Optional[str] = None
    video_ids: Optional[Set[str]] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "cluster_id": self.cluster_id,
            "size": self.size,
            "centroid": self.centroid.tolist() if self.centroid is not None else None,
            "keywords": self.keywords or [],
            "theme": self.theme,
            "video_ids": list(self.video_ids) if self.video_ids else []
        }


class ClusterManager:
    """
    Manages cluster storage and operations in ChromaDB.
    
    Handles:
    - Storing cluster labels in ChromaDB metadata
    - Updating cluster assignments
    - Querying chunks by cluster
    - Cluster statistics
    """
    
    def __init__(self, chroma_manager: Optional[ChromaDBManager] = None):
        """
        Initialize cluster manager.
        
        Args:
            chroma_manager: ChromaDBManager instance (creates default if None)
        """
        self.logger = get_logger(__name__)
        self.chroma_manager = chroma_manager or ChromaDBManager()
        self.collection = self.chroma_manager.get_or_create_collection()
    
    def store_cluster_labels(
        self,
        chunk_ids: List[str],
        labels: np.ndarray,
        probabilities: Optional[np.ndarray] = None
    ) -> int:
        """
        Store cluster labels in ChromaDB metadata.
        
        Args:
            chunk_ids: List of chunk IDs
            labels: Array of cluster labels (-1 for outliers)
            probabilities: Optional array of cluster membership probabilities
        
        Returns:
            Number of chunks updated
        """
        if len(chunk_ids) != len(labels):
            raise ValueError("chunk_ids and labels must have same length")
        
        if probabilities is not None and len(probabilities) != len(labels):
            raise ValueError("probabilities must have same length as labels")
        
        self.logger.info(f"Storing cluster labels for {len(chunk_ids)} chunks")
        
        # Get existing metadata
        existing_data = self.collection.get(ids=chunk_ids, include=["metadatas"])
        existing_metadatas = existing_data.get("metadatas", [])
        
        # Update metadata with cluster information
        updated_metadatas = []
        for i, (chunk_id, label, existing_meta) in enumerate(zip(chunk_ids, labels, existing_metadatas)):
            if existing_meta is None:
                existing_meta = {}
            
            # Update with cluster info
            updated_meta = existing_meta.copy()
            updated_meta["cluster_id"] = int(label)
            
            if probabilities is not None:
                updated_meta["cluster_probability"] = float(probabilities[i])
            else:
                # Default probability: 1.0 for assigned clusters, 0.0 for outliers
                updated_meta["cluster_probability"] = 1.0 if label != -1 else 0.0
            
            updated_metadatas.append(updated_meta)
        
        # Update in ChromaDB
        self.collection.update(
            ids=chunk_ids,
            metadatas=updated_metadatas
        )
        
        self.logger.info(f"Stored cluster labels for {len(chunk_ids)} chunks")
        return len(chunk_ids)
    
    def get_chunks_by_cluster(
        self,
        cluster_id: int,
        limit: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Get chunks belonging to a specific cluster.
        
        Args:
            cluster_id: Cluster ID to query
            limit: Maximum number of results
        
        Returns:
            Dictionary with ids, documents, metadatas, embeddings
        """
        where_clause = {"cluster_id": cluster_id}
        
        results = self.collection.get(
            where=where_clause,
            limit=limit,
            include=["documents", "metadatas", "embeddings"]
        )
        
        return results
    
    def get_cluster_statistics(self) -> Dict[int, ClusterMetadata]:
        """
        Get statistics for all clusters.
        
        Returns:
            Dictionary mapping cluster_id -> ClusterMetadata
        """
        # Get all chunks with cluster information
        all_data = self.collection.get(
            include=["metadatas", "embeddings"]
        )
        
        metadatas = all_data.get("metadatas", [])
        embeddings = all_data.get("embeddings", [])
        ids = all_data.get("ids", [])
        
        # Group by cluster_id
        clusters: Dict[int, List[int]] = {}  # cluster_id -> list of indices
        
        for i, metadata in enumerate(metadatas):
            if metadata and "cluster_id" in metadata:
                cluster_id = int(metadata["cluster_id"])
                if cluster_id != -1:  # Exclude outliers
                    if cluster_id not in clusters:
                        clusters[cluster_id] = []
                    clusters[cluster_id].append(i)
        
        # Calculate statistics for each cluster
        cluster_stats = {}
        
        for cluster_id, indices in clusters.items():
            cluster_embeddings = None
            if embeddings:
                cluster_embeddings = np.array([embeddings[i] for i in indices])
            
            # Calculate centroid
            centroid = None
            if cluster_embeddings is not None and len(cluster_embeddings) > 0:
                centroid = np.mean(cluster_embeddings, axis=0)
            
            # Get video IDs
            video_ids = set()
            for i in indices:
                if metadatas[i] and "video_id" in metadatas[i]:
                    video_ids.add(metadatas[i]["video_id"])
            
            cluster_stats[cluster_id] = ClusterMetadata(
                cluster_id=cluster_id,
                size=len(indices),
                centroid=centroid,
                video_ids=video_ids
            )
        
        return cluster_stats
    
    def get_cluster_sizes(self) -> Dict[int, int]:
        """
        Get sizes of all clusters.
        
        Returns:
            Dictionary mapping cluster_id -> size
        """
        stats = self.get_cluster_statistics()
        return {cluster_id: metadata.size for cluster_id, metadata in stats.items()}
    
    def get_outlier_count(self) -> int:
        """
        Get count of outliers (chunks with cluster_id == -1).
        
        Returns:
            Number of outliers
        """
        where_clause = {"cluster_id": -1}
        results = self.collection.get(where=where_clause)
        return len(results.get("ids", []))
    
    def update_cluster_metadata(
        self,
        cluster_id: int,
        keywords: Optional[List[str]] = None,
        theme: Optional[str] = None
    ):
        """
        Update cluster metadata (keywords, theme).
        
        Note: This stores metadata separately, not in ChromaDB.
        For persistent storage, consider using a separate JSON file or database.
        
        Args:
            cluster_id: Cluster ID
            keywords: List of keywords for the cluster
            theme: Theme description
        """
        # This is a placeholder - in a full implementation, you might store
        # cluster metadata in a separate collection or JSON file
        self.logger.info(
            f"Updating metadata for cluster {cluster_id}: "
            f"keywords={keywords}, theme={theme}"
        )
    
    def clear_cluster_labels(self, chunk_ids: Optional[List[str]] = None) -> int:
        """
        Clear cluster labels from metadata.
        
        Args:
            chunk_ids: List of chunk IDs to clear (None = all chunks)
        
        Returns:
            Number of chunks updated
        """
        if chunk_ids is None:
            # Get all chunk IDs
            all_data = self.collection.get(include=["metadatas"])
            chunk_ids = all_data.get("ids", [])
        
        if not chunk_ids:
            return 0
        
        # Get existing metadata
        existing_data = self.collection.get(ids=chunk_ids, include=["metadatas"])
        existing_metadatas = existing_data.get("metadatas", [])
        
        # Remove cluster fields
        updated_metadatas = []
        for existing_meta in existing_metadatas:
            if existing_meta is None:
                existing_meta = {}
            
            updated_meta = existing_meta.copy()
            updated_meta.pop("cluster_id", None)
            updated_meta.pop("cluster_probability", None)
            updated_metadatas.append(updated_meta)
        
        # Update in ChromaDB
        self.collection.update(
            ids=chunk_ids,
            metadatas=updated_metadatas
        )
        
        self.logger.info(f"Cleared cluster labels for {len(chunk_ids)} chunks")
        return len(chunk_ids)

