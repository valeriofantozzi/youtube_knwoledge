"""
Clustering module for semantic grouping of embeddings.

Provides clustering algorithms, cluster management, evaluation, and integration
with the retrieval system.
"""

from .clusterer import Clusterer
from .hdbscan_clusterer import HDBSCANClusterer
from .cluster_manager import ClusterManager
from .cluster_evaluator import ClusterEvaluator, ClusterMetrics
from .cluster_integrator import ClusterIntegrator

__all__ = [
    "Clusterer",
    "HDBSCANClusterer",
    "ClusterManager",
    "ClusterEvaluator",
    "ClusterMetrics",
    "ClusterIntegrator",
]

