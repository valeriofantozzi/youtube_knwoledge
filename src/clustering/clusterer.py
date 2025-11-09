"""
Clusterer Base Class

Defines the interface for clustering algorithms.
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional, Tuple
import numpy as np

from ..utils.logger import get_logger


class Clusterer(ABC):
    """
    Base class for clustering algorithms.
    
    Provides a unified interface for different clustering implementations.
    """
    
    def __init__(self, **kwargs):
        """Initialize clusterer with algorithm-specific parameters."""
        self.logger = get_logger(__name__)
        self.parameters = kwargs
    
    @abstractmethod
    def fit(
        self,
        embeddings: np.ndarray,
        metadata: Optional[List[Dict[str, Any]]] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Fit clustering model to embeddings.
        
        Args:
            embeddings: Array of shape (n_samples, n_features) with embeddings
            metadata: Optional list of metadata dictionaries for each embedding
        
        Returns:
            Tuple of (labels, probabilities):
            - labels: Array of cluster labels (-1 for noise/outliers)
            - probabilities: Array of cluster membership probabilities (0.0-1.0)
        """
        pass
    
    @abstractmethod
    def predict(
        self,
        embeddings: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Predict cluster assignments for new embeddings.
        
        Args:
            embeddings: Array of shape (n_samples, n_features) with embeddings
        
        Returns:
            Tuple of (labels, probabilities):
            - labels: Array of cluster labels (-1 for noise/outliers)
            - probabilities: Array of cluster membership probabilities (0.0-1.0)
        """
        pass
    
    @abstractmethod
    def get_cluster_centroids(self) -> np.ndarray:
        """
        Get cluster centroids.
        
        Returns:
            Array of shape (n_clusters, n_features) with cluster centroids
        """
        pass
    
    @abstractmethod
    def get_n_clusters(self) -> int:
        """
        Get number of clusters found.
        
        Returns:
            Number of clusters (excluding noise/outliers)
        """
        pass
    
    def get_cluster_sizes(self, labels: np.ndarray) -> Dict[int, int]:
        """
        Get sizes of each cluster.
        
        Args:
            labels: Array of cluster labels
        
        Returns:
            Dictionary mapping cluster_id -> size
        """
        unique, counts = np.unique(labels, return_counts=True)
        return dict(zip(unique.tolist(), counts.tolist()))
    
    def get_outlier_count(self, labels: np.ndarray) -> int:
        """
        Get count of outliers (label == -1).
        
        Args:
            labels: Array of cluster labels
        
        Returns:
            Number of outliers
        """
        return int(np.sum(labels == -1))

