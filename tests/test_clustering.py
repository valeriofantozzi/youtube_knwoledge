"""
Unit tests for clustering module.
"""

import pytest
import numpy as np
from unittest.mock import Mock, MagicMock, patch

from src.clustering.clusterer import Clusterer
from src.clustering.hdbscan_clusterer import HDBSCANClusterer, HDBSCAN_AVAILABLE
from src.clustering.cluster_manager import ClusterManager, ClusterMetadata
from src.clustering.cluster_evaluator import ClusterEvaluator, ClusterMetrics
from src.clustering.cluster_integrator import ClusterIntegrator


class TestHDBSCANClusterer:
    """Test cases for HDBSCANClusterer."""
    
    @pytest.fixture
    def sample_embeddings(self):
        """Create sample embeddings."""
        np.random.seed(42)
        # Create 3 distinct clusters
        cluster1 = np.random.randn(20, 1024).astype(np.float32)
        cluster2 = np.random.randn(20, 1024).astype(np.float32) + 5
        cluster3 = np.random.randn(20, 1024).astype(np.float32) + 10
        
        embeddings = np.vstack([cluster1, cluster2, cluster3])
        # Normalize
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        embeddings = embeddings / norms
        return embeddings
    
    @pytest.mark.skipif(not HDBSCAN_AVAILABLE, reason="hdbscan not installed")
    def test_fit_basic(self, sample_embeddings):
        """Test basic fitting."""
        clusterer = HDBSCANClusterer(min_cluster_size=5, min_samples=3)
        labels, probabilities = clusterer.fit(sample_embeddings)
        
        assert len(labels) == len(sample_embeddings)
        assert len(probabilities) == len(sample_embeddings)
        assert all(0.0 <= p <= 1.0 for p in probabilities)
        
        n_clusters = clusterer.get_n_clusters()
        assert n_clusters > 0
    
    @pytest.mark.skipif(not HDBSCAN_AVAILABLE, reason="hdbscan not installed")
    def test_predict(self, sample_embeddings):
        """Test prediction on new embeddings."""
        clusterer = HDBSCANClusterer(min_cluster_size=5, min_samples=3)
        clusterer.fit(sample_embeddings)
        
        # Predict on new embeddings
        new_embeddings = np.random.randn(5, 1024).astype(np.float32)
        norms = np.linalg.norm(new_embeddings, axis=1, keepdims=True)
        new_embeddings = new_embeddings / norms
        
        labels, probabilities = clusterer.predict(new_embeddings)
        
        assert len(labels) == 5
        assert len(probabilities) == 5
    
    @pytest.mark.skipif(not HDBSCAN_AVAILABLE, reason="hdbscan not installed")
    def test_get_cluster_centroids(self, sample_embeddings):
        """Test getting cluster centroids."""
        clusterer = HDBSCANClusterer(min_cluster_size=5, min_samples=3)
        clusterer.fit(sample_embeddings)
        
        centroids = clusterer.get_cluster_centroids()
        n_clusters = clusterer.get_n_clusters()
        
        if n_clusters > 0:
            assert centroids.shape[0] == n_clusters
            assert centroids.shape[1] == sample_embeddings.shape[1]
    
    @pytest.mark.skipif(not HDBSCAN_AVAILABLE, reason="hdbscan not installed")
    def test_get_cluster_sizes(self, sample_embeddings):
        """Test getting cluster sizes."""
        clusterer = HDBSCANClusterer(min_cluster_size=5, min_samples=3)
        labels, _ = clusterer.fit(sample_embeddings)
        
        sizes = clusterer.get_cluster_sizes(labels)
        assert isinstance(sizes, dict)
        assert all(isinstance(k, (int, np.integer)) for k in sizes.keys())
        assert all(isinstance(v, (int, np.integer)) for v in sizes.values())
    
    def test_hdbscan_not_available(self):
        """Test error when hdbscan is not available."""
        if HDBSCAN_AVAILABLE:
            pytest.skip("hdbscan is available")
        
        with patch('src.clustering.hdbscan_clusterer.HDBSCAN_AVAILABLE', False):
            with pytest.raises(ImportError):
                HDBSCANClusterer()


class TestClusterManager:
    """Test cases for ClusterManager."""
    
    @pytest.fixture
    def mock_chroma_manager(self):
        """Create mock ChromaDBManager."""
        manager = Mock()
        collection = Mock()
        
        # Mock collection.get
        collection.get.return_value = {
            "ids": ["chunk_1", "chunk_2"],
            "metadatas": [
                {"source_id": "video_1", "title": "Test"},
                {"source_id": "video_2", "title": "Test2"}
            ],
            "embeddings": None
        }
        
        # Mock collection.update
        collection.update.return_value = None
        
        manager.get_or_create_collection.return_value = collection
        return manager
    
    @pytest.fixture
    def cluster_manager(self, mock_chroma_manager):
        """Create ClusterManager with mocked ChromaDB."""
        return ClusterManager(chroma_manager=mock_chroma_manager)
    
    def test_store_cluster_labels(self, cluster_manager):
        """Test storing cluster labels."""
        chunk_ids = ["chunk_1", "chunk_2"]
        labels = np.array([0, 1])
        probabilities = np.array([0.9, 0.8])
        
        # Mock get to return existing metadata
        cluster_manager.collection.get.return_value = {
            "metadatas": [
                {"source_id": "video_1"},
                {"source_id": "video_2"}
            ]
        }
        
        result = cluster_manager.store_cluster_labels(chunk_ids, labels, probabilities)
        
        assert result == 2
        cluster_manager.collection.update.assert_called_once()
    
    def test_get_chunks_by_cluster(self, cluster_manager):
        """Test getting chunks by cluster."""
        cluster_manager.collection.get.return_value = {
            "ids": ["chunk_1"],
            "documents": ["Test document"],
            "metadatas": [{"cluster_id": 0}],
            "embeddings": None
        }
        
        result = cluster_manager.get_chunks_by_cluster(cluster_id=0)
        
        assert "ids" in result
        assert "documents" in result
        cluster_manager.collection.get.assert_called_once()
    
    def test_get_cluster_statistics(self, cluster_manager):
        """Test getting cluster statistics."""
        cluster_manager.collection.get.return_value = {
            "ids": ["chunk_1", "chunk_2"],
            "metadatas": [
                {"cluster_id": 0, "source_id": "video_1"},
                {"cluster_id": 0, "source_id": "video_2"}
            ],
            "embeddings": [
                np.random.rand(1024).tolist(),
                np.random.rand(1024).tolist()
            ]
        }
        
        stats = cluster_manager.get_cluster_statistics()
        
        assert isinstance(stats, dict)
        assert 0 in stats
        assert isinstance(stats[0], ClusterMetadata)
    
    def test_get_outlier_count(self, cluster_manager):
        """Test getting outlier count."""
        cluster_manager.collection.get.return_value = {
            "ids": ["chunk_1", "chunk_2"]
        }
        
        count = cluster_manager.get_outlier_count()
        assert count == 2


class TestClusterEvaluator:
    """Test cases for ClusterEvaluator."""
    
    @pytest.fixture
    def evaluator(self):
        """Create ClusterEvaluator."""
        return ClusterEvaluator()
    
    @pytest.fixture
    def sample_clustered_data(self):
        """Create sample clustered embeddings."""
        np.random.seed(42)
        # Create 2 distinct clusters
        cluster1 = np.random.randn(20, 1024).astype(np.float32)
        cluster2 = np.random.randn(20, 1024).astype(np.float32) + 5
        
        embeddings = np.vstack([cluster1, cluster2])
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        embeddings = embeddings / norms
        
        labels = np.array([0] * 20 + [1] * 20)
        
        return embeddings, labels
    
    def test_evaluate_basic(self, evaluator, sample_clustered_data):
        """Test basic evaluation."""
        embeddings, labels = sample_clustered_data
        
        metrics = evaluator.evaluate(embeddings, labels)
        
        assert isinstance(metrics, ClusterMetrics)
        assert metrics.n_clusters == 2
        assert metrics.n_outliers == 0
    
    def test_evaluate_with_outliers(self, evaluator, sample_clustered_data):
        """Test evaluation with outliers."""
        embeddings, labels = sample_clustered_data
        
        # Add some outliers
        labels_with_outliers = labels.copy()
        labels_with_outliers[0] = -1
        labels_with_outliers[10] = -1
        
        metrics = evaluator.evaluate(embeddings, labels_with_outliers)
        
        assert metrics.n_outliers == 2
    
    def test_analyze_cluster_coherence(self, evaluator):
        """Test cluster coherence analysis."""
        texts = [
            "orchid care watering",
            "orchid fertilizer nutrients",
            "gardening tips plants",
            "gardening soil compost"
        ]
        labels = np.array([0, 0, 1, 1])
        
        result = evaluator.analyze_cluster_coherence(0, texts, labels)
        
        assert result["cluster_id"] == 0
        assert result["size"] == 2
        assert "keywords" in result


class TestClusterIntegrator:
    """Test cases for ClusterIntegrator."""
    
    @pytest.fixture
    def mock_cluster_manager(self):
        """Create mock ClusterManager."""
        manager = Mock()
        manager.get_chunks_by_cluster.return_value = {
            "ids": ["chunk_1"],
            "documents": ["Test document"],
            "metadatas": [{"cluster_id": 0}]
        }
        manager.get_cluster_statistics.return_value = {
            0: ClusterMetadata(cluster_id=0, size=10, centroid=np.random.rand(1024)),
            1: ClusterMetadata(cluster_id=1, size=5, centroid=np.random.rand(1024))
        }
        return manager
    
    @pytest.fixture
    def mock_similarity_search(self):
        """Create mock SimilaritySearch."""
        return Mock()
    
    @pytest.fixture
    def integrator(self, mock_cluster_manager, mock_similarity_search):
        """Create ClusterIntegrator."""
        return ClusterIntegrator(
            cluster_manager=mock_cluster_manager,
            similarity_search=mock_similarity_search
        )
    
    def test_search_by_cluster(self, integrator):
        """Test searching by cluster."""
        results = integrator.search_by_cluster(cluster_id=0, top_k=5)
        
        assert len(results) > 0
        assert all(isinstance(r, type(results[0])) for r in results)
    
    def test_discover_related_clusters(self, integrator):
        """Test discovering related clusters."""
        related = integrator.discover_related_clusters(cluster_id=0, top_k=3)
        
        assert isinstance(related, list)
        assert all("cluster_id" in r for r in related)
        assert all("similarity" in r for r in related)
    
    def test_get_cluster_context(self, integrator):
        """Test getting cluster context."""
        context = integrator.get_cluster_context(cluster_id=0)
        
        assert "cluster_id" in context
        assert "size" in context
        assert "related_clusters" in context

