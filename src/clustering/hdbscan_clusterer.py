"""
HDBSCAN Clusterer Implementation

Implements HDBSCAN clustering algorithm optimized for high-dimensional embeddings.
"""

from typing import List, Dict, Any, Optional, Tuple
import numpy as np

try:
    import hdbscan
    HDBSCAN_AVAILABLE = True
except ImportError:
    HDBSCAN_AVAILABLE = False
    hdbscan = None

from .clusterer import Clusterer
from ..utils.logger import get_logger


class HDBSCANClusterer(Clusterer):
    """
    HDBSCAN clustering implementation.
    
    HDBSCAN (Hierarchical Density-Based Spatial Clustering of Applications with Noise)
    is optimal for high-dimensional normalized embeddings with cosine distance.
    
    Features:
    - Handles noise/outliers automatically
    - Supports soft clustering (membership probabilities)
    - Works well with cosine distance metric
    - No need to specify number of clusters a priori
    """
    
    def __init__(
        self,
        min_cluster_size: int = 10,
        min_samples: int = 5,
        metric: str = "cosine",
        cluster_selection_method: str = "eom",
        cluster_selection_epsilon: float = 0.0,
        **kwargs
    ):
        """
        Initialize HDBSCAN clusterer.
        
        Args:
            min_cluster_size: Minimum size of clusters
            min_samples: Number of samples in neighborhood for core point
            metric: Distance metric ('cosine', 'euclidean', etc.)
            cluster_selection_method: 'eom' (Excess of Mass) or 'leaf'
            cluster_selection_epsilon: Distance threshold for cluster selection
            **kwargs: Additional HDBSCAN parameters
        """
        if not HDBSCAN_AVAILABLE:
            raise ImportError(
                "hdbscan is not installed. Install it with: pip install hdbscan>=0.8.33"
            )
        
        super().__init__(**kwargs)
        
        self.min_cluster_size = min_cluster_size
        self.min_samples = min_samples
        self.metric = metric
        self.cluster_selection_method = cluster_selection_method
        self.cluster_selection_epsilon = cluster_selection_epsilon
        
        self.clusterer = None
        self.embeddings_ = None
        self.labels_ = None
        self.probabilities_ = None
        self.centroids_ = None
    
    def fit(
        self,
        embeddings: np.ndarray,
        metadata: Optional[List[Dict[str, Any]]] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Fit HDBSCAN model to embeddings.
        
        Args:
            embeddings: Array of shape (n_samples, n_features) with embeddings
            metadata: Optional list of metadata dictionaries (not used by HDBSCAN)
        
        Returns:
            Tuple of (labels, probabilities):
            - labels: Array of cluster labels (-1 for noise/outliers)
            - probabilities: Array of cluster membership probabilities (0.0-1.0)
        """
        self.logger.info(
            f"Fitting HDBSCAN with {len(embeddings)} embeddings "
            f"(min_cluster_size={self.min_cluster_size}, min_samples={self.min_samples})"
        )
        
        # Normalize embeddings if using cosine distance
        if self.metric == "cosine":
            norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
            norms[norms == 0] = 1.0  # Avoid division by zero
            embeddings_normalized = embeddings / norms
        else:
            embeddings_normalized = embeddings
        
        # Create HDBSCAN clusterer
        self.clusterer = hdbscan.HDBSCAN(
            min_cluster_size=self.min_cluster_size,
            min_samples=self.min_samples,
            metric=self.metric,
            cluster_selection_method=self.cluster_selection_method,
            cluster_selection_epsilon=self.cluster_selection_epsilon,
            **self.parameters
        )
        
        # Fit the model
        self.clusterer.fit(embeddings_normalized)
        
        # Store results
        self.embeddings_ = embeddings_normalized
        self.labels_ = self.clusterer.labels_
        self.probabilities_ = self.clusterer.probabilities_
        
        # Calculate centroids
        self._calculate_centroids()
        
        n_clusters = self.get_n_clusters()
        n_outliers = self.get_outlier_count(self.labels_)
        
        self.logger.info(
            f"HDBSCAN clustering complete: {n_clusters} clusters, "
            f"{n_outliers} outliers"
        )
        
        return self.labels_, self.probabilities_
    
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
        if self.clusterer is None:
            raise ValueError("Model must be fitted before prediction")
        
        # Normalize embeddings if using cosine distance
        if self.metric == "cosine":
            norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
            norms[norms == 0] = 1.0
            embeddings_normalized = embeddings / norms
        else:
            embeddings_normalized = embeddings
        
        # Use approximate_predict for new points
        labels, probabilities = hdbscan.approximate_predict(
            self.clusterer,
            embeddings_normalized
        )
        
        return labels, probabilities
    
    def _calculate_centroids(self):
        """Calculate cluster centroids."""
        if self.labels_ is None or self.embeddings_ is None:
            return
        
        unique_labels = np.unique(self.labels_)
        unique_labels = unique_labels[unique_labels != -1]  # Exclude noise
        
        if len(unique_labels) == 0:
            self.centroids_ = np.array([]).reshape(0, self.embeddings_.shape[1])
            return
        
        centroids = []
        for label in unique_labels:
            mask = self.labels_ == label
            cluster_embeddings = self.embeddings_[mask]
            centroid = np.mean(cluster_embeddings, axis=0)
            centroids.append(centroid)
        
        self.centroids_ = np.array(centroids)
    
    def get_cluster_centroids(self) -> np.ndarray:
        """
        Get cluster centroids.
        
        Returns:
            Array of shape (n_clusters, n_features) with cluster centroids
        """
        if self.centroids_ is None:
            self._calculate_centroids()
        return self.centroids_
    
    def get_n_clusters(self) -> int:
        """
        Get number of clusters found.
        
        Returns:
            Number of clusters (excluding noise/outliers)
        """
        if self.labels_ is None:
            return 0
        
        unique_labels = np.unique(self.labels_)
        unique_labels = unique_labels[unique_labels != -1]  # Exclude noise
        return len(unique_labels)
    
    def get_cluster_tree(self):
        """
        Get HDBSCAN cluster tree (condensed tree).
        
        Returns:
            Condensed tree object
        """
        if self.clusterer is None:
            raise ValueError("Model must be fitted before getting cluster tree")
        return self.clusterer.condensed_tree_

