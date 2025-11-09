"""
Cluster Evaluator Module

Evaluates cluster quality using various metrics.
"""

from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
import numpy as np

try:
    from sklearn.metrics import (
        silhouette_score,
        davies_bouldin_score,
        calinski_harabasz_score
    )
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

from ..utils.logger import get_logger


@dataclass
class ClusterMetrics:
    """Cluster quality metrics."""
    silhouette_score: float = 0.0
    davies_bouldin_index: float = 0.0
    calinski_harabasz_index: float = 0.0
    n_clusters: int = 0
    n_outliers: int = 0
    avg_cluster_size: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "silhouette_score": self.silhouette_score,
            "davies_bouldin_index": self.davies_bouldin_index,
            "calinski_harabasz_index": self.calinski_harabasz_index,
            "n_clusters": self.n_clusters,
            "n_outliers": self.n_outliers,
            "avg_cluster_size": self.avg_cluster_size
        }


class ClusterEvaluator:
    """
    Evaluates cluster quality using various metrics.
    
    Provides metrics for:
    - Silhouette Score (higher is better, -1 to 1)
    - Davies-Bouldin Index (lower is better)
    - Calinski-Harabasz Index (higher is better)
    """
    
    def __init__(self):
        """Initialize cluster evaluator."""
        self.logger = get_logger(__name__)
        
        if not SKLEARN_AVAILABLE:
            self.logger.warning(
                "scikit-learn not available. Some metrics will not be calculated. "
                "Install with: pip install scikit-learn"
            )
    
    def evaluate(
        self,
        embeddings: np.ndarray,
        labels: np.ndarray,
        metric: str = "cosine"
    ) -> ClusterMetrics:
        """
        Evaluate cluster quality.
        
        Args:
            embeddings: Array of shape (n_samples, n_features) with embeddings
            labels: Array of cluster labels (-1 for outliers)
            metric: Distance metric for silhouette score ('cosine', 'euclidean')
        
        Returns:
            ClusterMetrics object with quality scores
        """
        # Filter out outliers for metric calculation
        mask = labels != -1
        filtered_embeddings = embeddings[mask]
        filtered_labels = labels[mask]
        
        if len(np.unique(filtered_labels)) < 2:
            self.logger.warning(
                "Need at least 2 clusters to calculate metrics. "
                "Returning default metrics."
            )
            return ClusterMetrics(
                n_clusters=len(np.unique(filtered_labels)),
                n_outliers=int(np.sum(labels == -1))
            )
        
        # Calculate metrics
        silhouette = 0.0
        davies_bouldin = 0.0
        calinski_harabasz = 0.0
        
        if SKLEARN_AVAILABLE:
            try:
                # Silhouette Score
                if metric == "cosine":
                    # Use precomputed distance matrix for cosine
                    from sklearn.metrics.pairwise import cosine_distances
                    distances = cosine_distances(filtered_embeddings)
                    silhouette = silhouette_score(
                        distances,
                        filtered_labels,
                        metric="precomputed"
                    )
                else:
                    silhouette = silhouette_score(
                        filtered_embeddings,
                        filtered_labels,
                        metric=metric
                    )
                
                # Davies-Bouldin Index
                davies_bouldin = davies_bouldin_score(
                    filtered_embeddings,
                    filtered_labels
                )
                
                # Calinski-Harabasz Index
                calinski_harabasz = calinski_harabasz_score(
                    filtered_embeddings,
                    filtered_labels
                )
            except Exception as e:
                self.logger.warning(f"Error calculating metrics: {e}")
        
        # Calculate cluster sizes
        unique_labels, counts = np.unique(filtered_labels, return_counts=True)
        avg_cluster_size = float(np.mean(counts)) if len(counts) > 0 else 0.0
        
        return ClusterMetrics(
            silhouette_score=silhouette,
            davies_bouldin_index=davies_bouldin,
            calinski_harabasz_index=calinski_harabasz,
            n_clusters=len(unique_labels),
            n_outliers=int(np.sum(labels == -1)),
            avg_cluster_size=avg_cluster_size
        )
    
    def analyze_cluster_coherence(
        self,
        cluster_id: int,
        texts: List[str],
        labels: np.ndarray
    ) -> Dict[str, Any]:
        """
        Analyze semantic coherence of a cluster.
        
        This is a placeholder for keyword extraction and theme identification.
        In a full implementation, you might use:
        - TF-IDF for keyword extraction
        - Topic modeling (LDA, BERTopic)
        - Text summarization
        
        Args:
            cluster_id: Cluster ID to analyze
            texts: List of text documents
            labels: Array of cluster labels
        
        Returns:
            Dictionary with coherence metrics and keywords
        """
        # Filter texts for this cluster
        cluster_texts = [
            text for text, label in zip(texts, labels)
            if label == cluster_id
        ]
        
        if not cluster_texts:
            return {
                "cluster_id": cluster_id,
                "size": 0,
                "keywords": [],
                "theme": None
            }
        
        # Simple keyword extraction (placeholder)
        # In production, use proper NLP techniques
        all_words = []
        for text in cluster_texts:
            words = text.lower().split()
            all_words.extend(words)
        
        # Count word frequencies
        from collections import Counter
        word_counts = Counter(all_words)
        top_keywords = [word for word, count in word_counts.most_common(10)]
        
        return {
            "cluster_id": cluster_id,
            "size": len(cluster_texts),
            "keywords": top_keywords,
            "theme": None  # Would be filled by topic modeling
        }
    
    def find_optimal_parameters(
        self,
        embeddings: np.ndarray,
        min_cluster_size_range: Tuple[int, int] = (5, 50),
        min_samples_range: Tuple[int, int] = (3, 20),
        n_trials: int = 10
    ) -> Dict[str, Any]:
        """
        Find optimal clustering parameters using grid search.
        
        Args:
            embeddings: Array of shape (n_samples, n_features) with embeddings
            min_cluster_size_range: Range for min_cluster_size parameter
            min_samples_range: Range for min_samples parameter
            n_trials: Number of parameter combinations to try
        
        Returns:
            Dictionary with optimal parameters and scores
        """
        self.logger.info("Finding optimal clustering parameters...")
        
        best_score = -np.inf
        best_params = None
        results = []
        
        # Generate parameter combinations
        min_cluster_sizes = np.linspace(
            min_cluster_size_range[0],
            min_cluster_size_range[1],
            n_trials // 2,
            dtype=int
        )
        min_samples_list = np.linspace(
            min_samples_range[0],
            min_samples_range[1],
            n_trials // 2,
            dtype=int
        )
        
        from ..clustering.hdbscan_clusterer import HDBSCANClusterer
        
        for min_cluster_size in min_cluster_sizes:
            for min_samples in min_samples_list:
                try:
                    # Create clusterer with these parameters
                    clusterer = HDBSCANClusterer(
                        min_cluster_size=int(min_cluster_size),
                        min_samples=int(min_samples)
                    )
                    
                    # Fit clustering
                    labels, probabilities = clusterer.fit(embeddings)
                    
                    # Evaluate
                    metrics = self.evaluate(embeddings, labels)
                    
                    # Use silhouette score as optimization target
                    score = metrics.silhouette_score
                    
                    results.append({
                        "min_cluster_size": int(min_cluster_size),
                        "min_samples": int(min_samples),
                        "silhouette_score": score,
                        "n_clusters": metrics.n_clusters,
                        "n_outliers": metrics.n_outliers
                    })
                    
                    if score > best_score:
                        best_score = score
                        best_params = {
                            "min_cluster_size": int(min_cluster_size),
                            "min_samples": int(min_samples),
                            "silhouette_score": score,
                            "n_clusters": metrics.n_clusters,
                            "n_outliers": metrics.n_outliers
                        }
                
                except Exception as e:
                    self.logger.warning(f"Error with parameters ({min_cluster_size}, {min_samples}): {e}")
                    continue
        
        self.logger.info(f"Found optimal parameters: {best_params}")
        
        return {
            "best_params": best_params,
            "all_results": results
        }

