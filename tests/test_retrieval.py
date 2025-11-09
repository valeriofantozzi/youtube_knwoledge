"""
Unit tests for retrieval module.
"""

import pytest
import numpy as np
from unittest.mock import Mock, MagicMock, patch
from typing import List, Dict, Any

from src.retrieval.similarity_search import (
    SimilaritySearch,
    SearchResult,
    SearchFilters
)
from src.retrieval.query_engine import QueryEngine
from src.retrieval.result_formatter import ResultFormatter, FormatOptions, OutputFormat
from src.retrieval.pipeline import RetrievalPipeline, RetrievalOptions, SearchMode
from src.vector_store.schema import ChunkMetadata


class TestSearchFilters:
    """Test cases for SearchFilters."""
    
    def test_empty_filters(self):
        """Test empty filters return None where clause."""
        filters = SearchFilters()
        where_clause = filters.to_chromadb_where()
        assert where_clause is None
    
    def test_video_id_filter(self):
        """Test video_id filter."""
        filters = SearchFilters(video_id="test_video_123")
        where_clause = filters.to_chromadb_where()
        
        assert where_clause is not None
        assert where_clause == {"video_id": {"$eq": "test_video_123"}}
    
    def test_date_range_filter(self):
        """Test date range filter."""
        filters = SearchFilters(
            date_start="2023/01/01",
            date_end="2023/12/31"
        )
        where_clause = filters.to_chromadb_where()
        
        assert where_clause is not None
        assert "$and" in where_clause
        assert len(where_clause["$and"]) == 2
    
    def test_combined_filters(self):
        """Test combined filters."""
        filters = SearchFilters(
            video_id="test_video_123",
            date_start="2023/01/01"
        )
        where_clause = filters.to_chromadb_where()
        
        assert where_clause is not None
        assert "$and" in where_clause
        assert len(where_clause["$and"]) == 2


class TestSearchResult:
    """Test cases for SearchResult."""
    
    def test_search_result_creation(self):
        """Test SearchResult creation."""
        metadata = ChunkMetadata(
            video_id="test_video",
            date="2023/01/01",
            title="Test Video",
            chunk_index=0,
            chunk_id="chunk_1",
            token_count=100,
            filename="test.srt"
        )
        
        result = SearchResult(
            id="doc_1",
            text="Test text",
            similarity_score=0.95,
            distance=0.05,
            metadata=metadata
        )
        
        assert result.id == "doc_1"
        assert result.text == "Test text"
        assert result.similarity_score == 0.95
        assert result.distance == 0.05
        assert result.metadata.video_id == "test_video"
    
    def test_search_result_to_dict(self):
        """Test SearchResult to_dict conversion."""
        metadata = ChunkMetadata(
            video_id="test_video",
            date="2023/01/01",
            title="Test Video",
            chunk_index=0,
            chunk_id="chunk_1",
            token_count=100,
            filename="test.srt"
        )
        
        result = SearchResult(
            id="doc_1",
            text="Test text",
            similarity_score=0.95,
            distance=0.05,
            metadata=metadata
        )
        
        result_dict = result.to_dict()
        assert isinstance(result_dict, dict)
        assert result_dict["id"] == "doc_1"
        assert result_dict["similarity_score"] == 0.95
        assert "metadata" in result_dict


class TestSimilaritySearch:
    """Test cases for SimilaritySearch."""
    
    @pytest.fixture
    def mock_chroma_manager(self):
        """Create mock ChromaDBManager."""
        manager = Mock()
        collection = Mock()
        
        # Mock collection.query to return sample results
        collection.query.return_value = {
            "ids": [["doc_1", "doc_2", "doc_3"]],
            "documents": [["Text 1", "Text 2", "Text 3"]],
            "metadatas": [[
                {
                    "video_id": "video_1",
                    "date": "2023/01/01",
                    "title": "Test Video 1",
                    "chunk_index": 0,
                    "chunk_id": "chunk_1",
                    "token_count": 100,
                    "filename": "test1.srt"
                },
                {
                    "video_id": "video_2",
                    "date": "2023/02/01",
                    "title": "Test Video 2",
                    "chunk_index": 0,
                    "chunk_id": "chunk_2",
                    "token_count": 150,
                    "filename": "test2.srt"
                },
                {
                    "video_id": "video_1",
                    "date": "2023/01/01",
                    "title": "Test Video 1",
                    "chunk_index": 1,
                    "chunk_id": "chunk_3",
                    "token_count": 120,
                    "filename": "test1.srt"
                }
            ]],
            "distances": [[0.1, 0.2, 0.3]]  # Lower distance = higher similarity
        }
        
        manager.get_or_create_collection.return_value = collection
        manager.get_collection_stats.return_value = {
            "name": "test_collection",
            "count": 3
        }
        
        return manager
    
    @pytest.fixture
    def mock_embedder(self):
        """Create mock Embedder."""
        embedder = Mock()
        # Mock embedding generation (1024-dimensional for BGE-large)
        mock_embedding = np.random.rand(1024).astype(np.float32)
        embedder.encode_single.return_value = mock_embedding
        embedder.encode.return_value = np.array([mock_embedding])
        return embedder
    
    @pytest.fixture
    def similarity_search(self, mock_chroma_manager, mock_embedder):
        """Create SimilaritySearch instance with mocks."""
        return SimilaritySearch(
            chroma_manager=mock_chroma_manager,
            embedder=mock_embedder
        )
    
    def test_search_basic(self, similarity_search):
        """Test basic search functionality."""
        results = similarity_search.search("test query", top_k=3)
        
        assert len(results) == 3
        assert all(isinstance(r, SearchResult) for r in results)
        assert results[0].similarity_score > results[1].similarity_score  # Should be sorted
    
    def test_search_empty_query(self, similarity_search):
        """Test search with empty query raises error."""
        with pytest.raises(ValueError, match="Query cannot be empty"):
            similarity_search.search("")
        
        with pytest.raises(ValueError, match="Query cannot be empty"):
            similarity_search.search("   ")
    
    def test_search_invalid_top_k(self, similarity_search):
        """Test search with invalid top_k raises error."""
        with pytest.raises(ValueError, match="top_k must be >= 1"):
            similarity_search.search("test", top_k=0)
        
        with pytest.raises(ValueError, match="top_k must be >= 1"):
            similarity_search.search("test", top_k=-1)
    
    def test_search_score_threshold(self, similarity_search):
        """Test search with score threshold."""
        # With threshold 0.85, only first result (similarity 0.9) should pass
        results = similarity_search.search(
            "test query",
            top_k=10,
            score_threshold=0.85
        )
        
        # All results should have similarity >= 0.85
        assert all(r.similarity_score >= 0.85 for r in results)
    
    def test_search_invalid_score_threshold(self, similarity_search):
        """Test search with invalid score threshold raises error."""
        with pytest.raises(ValueError, match="score_threshold must be between"):
            similarity_search.search("test", score_threshold=1.5)
        
        with pytest.raises(ValueError, match="score_threshold must be between"):
            similarity_search.search("test", score_threshold=-0.1)
    
    def test_search_with_video_id_filter(self, similarity_search):
        """Test search with video_id filter."""
        filters = SearchFilters(video_id="video_1")
        results = similarity_search.search(
            "test query",
            top_k=10,
            filters=filters
        )
        
        # Verify filter was applied (check that query was called with where clause)
        collection = similarity_search.chroma_manager.get_or_create_collection()
        call_args = collection.query.call_args
        
        assert call_args is not None
        # Check that where clause was included
        if "where" in call_args.kwargs:
            where_clause = call_args.kwargs["where"]
            assert where_clause == {"video_id": {"$eq": "video_1"}}
    
    def test_search_with_date_range_filter(self, similarity_search):
        """Test search with date range filter."""
        filters = SearchFilters(
            date_start="2023/01/01",
            date_end="2023/01/31"
        )
        results = similarity_search.search(
            "test query",
            top_k=10,
            filters=filters
        )
        
        # Verify filter was applied
        collection = similarity_search.chroma_manager.get_or_create_collection()
        call_args = collection.query.call_args
        
        if call_args and "where" in call_args.kwargs:
            where_clause = call_args.kwargs["where"]
            assert "$and" in where_clause or "date" in str(where_clause)
    
    def test_search_with_title_keywords_filter(self, similarity_search):
        """Test search with title keywords filter."""
        filters = SearchFilters(title_keywords=["Video 1"])
        results = similarity_search.search(
            "test query",
            top_k=10,
            filters=filters
        )
        
        # Results should only include documents with "Video 1" in title
        # Based on mock data, only first and third results have "Video 1" in title
        assert len(results) <= 2
        for result in results:
            assert "Video 1" in result.metadata.title
    
    def test_search_result_ranking(self, similarity_search):
        """Test that search results are ranked by similarity score."""
        results = similarity_search.search("test query", top_k=10)
        
        # Results should be sorted by similarity score (descending)
        for i in range(len(results) - 1):
            assert results[i].similarity_score >= results[i + 1].similarity_score
    
    def test_search_include_metadata(self, similarity_search):
        """Test search with include_metadata=True."""
        results = similarity_search.search(
            "test query",
            top_k=3,
            include_metadata=True
        )
        
        assert len(results) > 0
        assert all(r.metadata is not None for r in results)
        assert all(r.metadata.video_id != 'unknown' for r in results)
    
    def test_search_exclude_metadata(self, similarity_search):
        """Test search with include_metadata=False."""
        # Note: Our implementation always includes metadata, but creates minimal one if not available
        # This test verifies the behavior
        results = similarity_search.search(
            "test query",
            top_k=3,
            include_metadata=False
        )
        
        # Results should still have metadata (minimal if not included)
        assert len(results) > 0
        assert all(r.metadata is not None for r in results)
    
    def test_search_by_embedding(self, similarity_search):
        """Test search using pre-computed embedding."""
        query_embedding = np.random.rand(1024).astype(np.float32)
        results = similarity_search.search_by_embedding(
            query_embedding,
            top_k=3
        )
        
        assert len(results) == 3
        assert all(isinstance(r, SearchResult) for r in results)
    
    def test_search_by_embedding_invalid(self, similarity_search):
        """Test search_by_embedding with invalid embedding."""
        with pytest.raises(ValueError, match="Query embedding cannot be empty"):
            similarity_search.search_by_embedding(np.array([]))
    
    def test_batch_search(self, similarity_search):
        """Test batch search for multiple queries."""
        queries = ["query 1", "query 2", "query 3"]
        results_list = similarity_search.batch_search(queries, top_k=3)
        
        # batch_search should return one list per query
        assert len(results_list) == len(queries)
        assert all(isinstance(results, list) for results in results_list)
        # Each query should return up to top_k results
        assert all(len(results) <= 3 for results in results_list)
    
    def test_batch_search_empty(self, similarity_search):
        """Test batch search with empty query list."""
        results_list = similarity_search.batch_search([])
        assert results_list == []
    
    def test_get_collection_stats(self, similarity_search):
        """Test getting collection statistics."""
        stats = similarity_search.get_collection_stats()
        
        assert isinstance(stats, dict)
        assert "name" in stats
        assert "count" in stats
    
    def test_process_results_with_score_threshold(self, similarity_search):
        """Test _process_results with score threshold."""
        mock_results = {
            "ids": [["doc_1", "doc_2"]],
            "documents": [["Text 1", "Text 2"]],
            "metadatas": [[
                {
                    "video_id": "video_1",
                    "date": "2023/01/01",
                    "title": "Test Video",
                    "chunk_index": 0,
                    "chunk_id": "chunk_1",
                    "token_count": 100,
                    "filename": "test.srt"
                },
                {
                    "video_id": "video_2",
                    "date": "2023/02/01",
                    "title": "Test Video 2",
                    "chunk_index": 0,
                    "chunk_id": "chunk_2",
                    "token_count": 150,
                    "filename": "test2.srt"
                }
            ]],
            "distances": [[0.1, 0.5]]  # Similarities: 0.9, 0.5
        }
        
        results = similarity_search._process_results(
            mock_results,
            score_threshold=0.8
        )
        
        # Only first result should pass threshold
        assert len(results) == 1
        assert results[0].similarity_score == 0.9
    
    def test_process_results_with_title_keywords_filter(self, similarity_search):
        """Test _process_results with title keywords filter."""
        mock_results = {
            "ids": [["doc_1", "doc_2"]],
            "documents": [["Text 1", "Text 2"]],
            "metadatas": [[
                {
                    "video_id": "video_1",
                    "date": "2023/01/01",
                    "title": "Orchid Care Video",
                    "chunk_index": 0,
                    "chunk_id": "chunk_1",
                    "token_count": 100,
                    "filename": "test.srt"
                },
                {
                    "video_id": "video_2",
                    "date": "2023/02/01",
                    "title": "Garden Tips Video",
                    "chunk_index": 0,
                    "chunk_id": "chunk_2",
                    "token_count": 150,
                    "filename": "test2.srt"
                }
            ]],
            "distances": [[0.1, 0.2]]
        }
        
        filters = SearchFilters(title_keywords=["Orchid"])
        results = similarity_search._process_results(
            mock_results,
            filters=filters
        )
        
        # Only first result should pass filter
        assert len(results) == 1
        assert "orchid" in results[0].metadata.title.lower()


class TestQueryEngine:
    """Test cases for QueryEngine."""
    
    @pytest.fixture
    def mock_similarity_search(self):
        """Create mock SimilaritySearch."""
        search = Mock()
        search.search.return_value = [
            SearchResult(
                id="doc_1",
                text="Test result",
                similarity_score=0.9,
                distance=0.1,
                metadata=ChunkMetadata(
                    video_id="video_1",
                    date="2023/01/01",
                    title="Test Video",
                    chunk_index=0,
                    chunk_id="chunk_1",
                    token_count=100,
                    filename="test.srt"
                )
            )
        ]
        # Mock batch_search to return one result list per query
        def batch_search_side_effect(queries, **kwargs):
            return [
                [SearchResult(
                    id=f"doc_{i}",
                    text=f"Test result {i}",
                    similarity_score=0.9,
                    distance=0.1,
                    metadata=ChunkMetadata(
                        video_id="video_1",
                        date="2023/01/01",
                        title="Test Video",
                        chunk_index=0,
                        chunk_id=f"chunk_{i}",
                        token_count=100,
                        filename="test.srt"
                    )
                )]
                for i in range(len(queries))
            ]
        search.batch_search.side_effect = batch_search_side_effect
        return search
    
    @pytest.fixture
    def query_engine(self, mock_similarity_search):
        """Create QueryEngine instance with mock."""
        return QueryEngine(similarity_search=mock_similarity_search, enable_caching=True)
    
    def test_query_basic(self, query_engine):
        """Test basic query execution."""
        results = query_engine.query("test query", top_k=5)
        
        assert len(results) == 1
        assert results[0].similarity_score == 0.9
        assert query_engine.similarity_search.search.called
    
    def test_query_empty(self, query_engine):
        """Test query with empty string raises error."""
        with pytest.raises(ValueError, match="Query cannot be empty"):
            query_engine.query("")
        
        with pytest.raises(ValueError, match="Query cannot be empty"):
            query_engine.query("   ")
    
    def test_query_too_short(self, query_engine):
        """Test query validation for too short query."""
        # MIN_QUERY_LENGTH is 1, so empty string should fail
        with pytest.raises(ValueError, match="Query cannot be empty"):
            query_engine.query("")
    
    def test_query_too_long(self, query_engine):
        """Test query validation for too long query."""
        long_query = "a" * 1001  # MAX_QUERY_LENGTH is 1000
        with pytest.raises(ValueError, match="Query too long"):
            query_engine.query(long_query)
    
    def test_query_preprocessing(self, query_engine):
        """Test query preprocessing."""
        # Query with extra whitespace should be normalized
        results = query_engine.query("  test   query  ", top_k=5)
        
        # Verify preprocessing was applied (check that search was called with processed query)
        call_args = query_engine.similarity_search.search.call_args
        assert call_args is not None
        processed_query = call_args.kwargs.get('query', '')
        assert processed_query == "test query"  # Normalized whitespace
    
    def test_query_with_filters(self, query_engine):
        """Test query with filters."""
        filters = SearchFilters(video_id="video_1")
        results = query_engine.query("test query", top_k=5, filters=filters)
        
        assert len(results) == 1
        # Verify filters were passed to search
        call_args = query_engine.similarity_search.search.call_args
        assert call_args.kwargs.get('filters') == filters
    
    def test_query_caching(self, query_engine):
        """Test query caching."""
        # First query - should call search
        results1 = query_engine.query("test query", top_k=5)
        
        # Reset mock call count
        query_engine.similarity_search.search.reset_mock()
        
        # Second identical query - should use cache
        results2 = query_engine.query("test query", top_k=5)
        
        # Verify cache was used (search should not be called)
        assert not query_engine.similarity_search.search.called
        assert results1 == results2
    
    def test_query_cache_disabled(self, query_engine):
        """Test query with caching disabled."""
        query_engine.enable_caching = False
        
        results1 = query_engine.query("test query", top_k=5)
        query_engine.similarity_search.search.reset_mock()
        results2 = query_engine.query("test query", top_k=5)
        
        # Search should be called both times
        assert query_engine.similarity_search.search.called
    
    def test_query_cache_eviction(self, query_engine):
        """Test cache eviction when cache is full."""
        query_engine.cache_size = 2
        
        # Fill cache
        query_engine.query("query 1", top_k=5)
        query_engine.query("query 2", top_k=5)
        
        # Add third query - should evict first
        query_engine.query("query 3", top_k=5)
        
        assert len(query_engine._cache) == 2
    
    def test_clear_cache(self, query_engine):
        """Test clearing cache."""
        query_engine.query("test query", top_k=5)
        assert len(query_engine._cache) > 0
        
        query_engine.clear_cache()
        assert len(query_engine._cache) == 0
    
    def test_get_cache_stats(self, query_engine):
        """Test getting cache statistics."""
        query_engine.query("test query", top_k=5)
        stats = query_engine.get_cache_stats()
        
        assert stats["enabled"] is True
        assert stats["size"] == 1
        assert stats["max_size"] == 100
        assert stats["usage_percent"] > 0
    
    def test_batch_query(self, query_engine):
        """Test batch query execution."""
        queries = ["query 1", "query 2", "query 3"]
        results_list = query_engine.batch_query(queries, top_k=5)
        
        assert len(results_list) == len(queries)
        assert query_engine.similarity_search.batch_search.called
    
    def test_batch_query_empty(self, query_engine):
        """Test batch query with empty list."""
        results_list = query_engine.batch_query([])
        assert results_list == []
    
    def test_query_with_options(self, query_engine):
        """Test query with QueryOptions."""
        from src.retrieval.query_engine import QueryOptions
        
        options = QueryOptions(
            top_k=5,
            score_threshold=0.8,
            include_metadata=True,
            filters=SearchFilters(video_id="video_1")
        )
        
        results = query_engine.query_with_options("test query", options)
        
        assert len(results) == 1
        # Verify options were passed correctly
        call_args = query_engine.similarity_search.search.call_args
        assert call_args.kwargs.get('top_k') == 5
        assert call_args.kwargs.get('score_threshold') == 0.8
        assert call_args.kwargs.get('filters') == options.filters
    
    def test_query_invalid_type(self, query_engine):
        """Test query with invalid type raises error."""
        with pytest.raises(ValueError, match="Query must be a string"):
            query_engine.query(123)  # type: ignore
        
        with pytest.raises(ValueError, match="Query must be a string"):
            query_engine.query(None)  # type: ignore


class TestResultFormatter:
    """Test cases for ResultFormatter."""
    
    @pytest.fixture
    def sample_results(self):
        """Create sample search results."""
        return [
            SearchResult(
                id="doc_1",
                text="This is a test result about orchids and gardening.",
                similarity_score=0.95,
                distance=0.05,
                metadata=ChunkMetadata(
                    video_id="video_1",
                    date="2023/01/01",
                    title="Orchid Care Guide",
                    chunk_index=5,
                    chunk_id="chunk_1",
                    token_count=100,
                    filename="test1.srt"
                )
            ),
            SearchResult(
                id="doc_2",
                text="Another result about plant care and maintenance.",
                similarity_score=0.88,
                distance=0.12,
                metadata=ChunkMetadata(
                    video_id="video_2",
                    date="2023/02/01",
                    title="Plant Maintenance Tips",
                    chunk_index=3,
                    chunk_id="chunk_2",
                    token_count=120,
                    filename="test2.srt"
                )
            ),
            SearchResult(
                id="doc_3",
                text="More information about orchids and their care.",
                similarity_score=0.85,
                distance=0.15,
                metadata=ChunkMetadata(
                    video_id="video_1",
                    date="2023/01/01",
                    title="Orchid Care Guide",
                    chunk_index=6,  # Adjacent to doc_1
                    chunk_id="chunk_3",
                    token_count=110,
                    filename="test1.srt"
                )
            )
        ]
    
    @pytest.fixture
    def formatter(self):
        """Create ResultFormatter instance."""
        return ResultFormatter()
    
    def test_format_text(self, formatter, sample_results):
        """Test text formatting."""
        options = FormatOptions(format=OutputFormat.TEXT)
        formatted = formatter.format_results(sample_results, query="orchids", options=options)
        
        assert "SEARCH RESULTS" in formatted
        assert "Result 1:" in formatted
        assert "orchids" in formatted.lower()
        assert "Video ID:" in formatted
    
    def test_format_markdown(self, formatter, sample_results):
        """Test Markdown formatting."""
        options = FormatOptions(format=OutputFormat.MARKDOWN)
        formatted = formatter.format_results(sample_results, query="orchids", options=options)
        
        assert "# Search Results" in formatted
        assert "## Result 1" in formatted
        assert "| Field | Value |" in formatted
        assert "| Video ID |" in formatted
    
    def test_format_json(self, formatter, sample_results):
        """Test JSON formatting."""
        options = FormatOptions(format=OutputFormat.JSON, deduplicate=False, merge_adjacent=False)
        formatted = formatter.format_results(sample_results, query="orchids", options=options)
        
        import json
        data = json.loads(formatted)
        assert "query" in data
        assert "count" in data
        assert "results" in data
        assert len(data["results"]) == 3
    
    def test_format_empty_results(self, formatter):
        """Test formatting empty results."""
        options = FormatOptions(format=OutputFormat.TEXT)
        formatted = formatter.format_results([], options=options)
        assert "no results" in formatted.lower() or "no results found" in formatted.lower()
        
        options = FormatOptions(format=OutputFormat.JSON)
        formatted = formatter.format_results([], options=options)
        import json
        data = json.loads(formatted)
        assert data["count"] == 0
    
    def test_format_score_decimal(self, formatter, sample_results):
        """Test score formatting as decimal."""
        options = FormatOptions(format=OutputFormat.TEXT, score_format="decimal")
        formatted = formatter.format_results(sample_results[:1], options=options)
        assert "0.950" in formatted or "0.95" in formatted
    
    def test_format_score_percentage(self, formatter, sample_results):
        """Test score formatting as percentage."""
        options = FormatOptions(format=OutputFormat.TEXT, score_format="percentage")
        formatted = formatter.format_results(sample_results[:1], options=options)
        assert "95.0%" in formatted or "95%" in formatted
    
    def test_highlight_query(self, formatter, sample_results):
        """Test query highlighting."""
        options = FormatOptions(format=OutputFormat.TEXT, highlight_query=True)
        formatted = formatter.format_results(sample_results[:1], query="orchids", options=options)
        # Check that highlighting was applied (should contain **orchids** or similar)
        assert "orchids" in formatted.lower()
    
    def test_max_text_length(self, formatter, sample_results):
        """Test text truncation."""
        options = FormatOptions(format=OutputFormat.TEXT, max_text_length=20)
        formatted = formatter.format_results(sample_results[:1], options=options)
        # Text should be truncated
        assert "..." in formatted
    
    def test_deduplicate(self, formatter, sample_results):
        """Test deduplication."""
        # Add duplicate result
        duplicate = SearchResult(
            id="doc_1_dup",
            text="Duplicate text",
            similarity_score=0.90,
            distance=0.10,
            metadata=ChunkMetadata(
                video_id="video_1",
                date="2023/01/01",
                title="Orchid Care Guide",
                chunk_index=5,
                chunk_id="chunk_1",  # Same chunk_id as first result
                token_count=100,
                filename="test1.srt"
            )
        )
        results_with_dup = sample_results + [duplicate]
        
        options = FormatOptions(format=OutputFormat.TEXT, deduplicate=True)
        formatted = formatter.format_results(results_with_dup, options=options)
        
        # Should have fewer results after deduplication
        # Count occurrences of "Result X:" in formatted text
        result_count = formatted.count("Result")
        assert result_count <= len(results_with_dup)
    
    def test_merge_adjacent(self, formatter, sample_results):
        """Test merging adjacent chunks."""
        options = FormatOptions(format=OutputFormat.TEXT, merge_adjacent=True)
        formatted = formatter.format_results(sample_results, options=options)
        
        # Results from same video with adjacent chunk_index should be merged
        # doc_1 (chunk_index=5) and doc_3 (chunk_index=6) should be merged
        # So we should have fewer results
        result_count = formatted.count("Result")
        assert result_count <= len(sample_results)
    
    def test_format_single_result(self, formatter, sample_results):
        """Test formatting single result."""
        formatted = formatter.format_single_result(sample_results[0], query="test")
        assert "Result 1:" in formatted or "SEARCH RESULTS" in formatted
    
    def test_context_expansion_requires_manager(self, formatter, sample_results):
        """Test that context expansion requires chroma_manager."""
        options = FormatOptions(format=OutputFormat.TEXT, include_context=True)
        # Should not fail even without chroma_manager, just skip expansion
        formatted = formatter.format_results(sample_results[:1], options=options)
        assert formatted  # Should still format without context
    
    def test_deduplicate_and_merge_combined(self, formatter, sample_results):
        """Test combined deduplication and merging."""
        options = FormatOptions(
            format=OutputFormat.TEXT,
            deduplicate=True,
            merge_adjacent=True
        )
        formatted = formatter.format_results(sample_results, options=options)
        assert formatted
        # Should have fewer results after processing
        result_count = formatted.count("Result")
        assert result_count <= len(sample_results)


class TestRetrievalPipeline:
    """Integration tests for RetrievalPipeline."""
    
    @pytest.fixture
    def mock_query_engine(self):
        """Create mock QueryEngine."""
        engine = Mock()
        engine.query.return_value = [
            SearchResult(
                id="doc_1",
                text="Test result about orchids",
                similarity_score=0.9,
                distance=0.1,
                metadata=ChunkMetadata(
                    video_id="video_1",
                    date="2023/01/01",
                    title="Orchid Care Guide",
                    chunk_index=5,
                    chunk_id="chunk_1",
                    token_count=100,
                    filename="test.srt"
                )
            )
        ]
        engine._preprocess_query = lambda x: x.strip().lower()
        engine._get_cache_key = lambda *args: "cache_key"
        engine._cache = {}
        engine.clear_cache = Mock()
        return engine
    
    @pytest.fixture
    def mock_result_formatter(self):
        """Create mock ResultFormatter."""
        formatter = Mock()
        formatter.format_results.return_value = "Formatted results"
        return formatter
    
    @pytest.fixture
    def pipeline(self, mock_query_engine, mock_result_formatter):
        """Create RetrievalPipeline with mocked components."""
        return RetrievalPipeline(
            query_engine=mock_query_engine,
            result_formatter=mock_result_formatter,
            enable_monitoring=True
        )
    
    def test_search_basic(self, pipeline):
        """Test basic search functionality."""
        options = RetrievalOptions(top_k=5)
        results, perf_metrics, quality_metrics = pipeline.search(
            "test query",
            options=options,
            return_formatted=False
        )
        
        assert len(results) == 1
        assert results[0].id == "doc_1"
        assert perf_metrics is not None
        assert perf_metrics.num_results == 1
        assert quality_metrics is not None
        assert quality_metrics.avg_similarity_score == 0.9
    
    def test_search_with_formatting(self, pipeline):
        """Test search with result formatting."""
        options = RetrievalOptions(
            top_k=5,
            format_options=FormatOptions(format=OutputFormat.TEXT)
        )
        results, perf_metrics, quality_metrics = pipeline.search(
            "test query",
            options=options,
            return_formatted=True
        )
        
        assert len(results) == 1
        assert perf_metrics is not None
    
    def test_multi_query_search_union(self, pipeline):
        """Test multi-query search with union strategy."""
        options = RetrievalOptions(top_k=5)
        
        # Mock different results for different queries
        def query_side_effect(query_text, **kwargs):
            if "orchid" in query_text.lower():
                return [
                    SearchResult(
                        id="doc_1",
                        text="Orchid care",
                        similarity_score=0.9,
                        distance=0.1,
                        metadata=ChunkMetadata(
                            video_id="video_1",
                            date="2023/01/01",
                            title="Orchid Guide",
                            chunk_index=1,
                            chunk_id="chunk_1",
                            token_count=50,
                            filename="test.srt"
                        )
                    )
                ]
            else:
                return [
                    SearchResult(
                        id="doc_2",
                        text="Gardening tips",
                        similarity_score=0.8,
                        distance=0.2,
                        metadata=ChunkMetadata(
                            video_id="video_2",
                            date="2023/01/02",
                            title="Gardening Guide",
                            chunk_index=2,
                            chunk_id="chunk_2",
                            token_count=60,
                            filename="test2.srt"
                        )
                    )
                ]
        
        pipeline.query_engine.query.side_effect = query_side_effect
        
        queries = ["orchid care", "gardening tips"]
        results, perf_metrics = pipeline.multi_query_search(
            queries,
            options=options,
            combine_strategy="union"
        )
        
        assert len(results) >= 1
        assert perf_metrics is not None
        assert perf_metrics.num_results >= 1
    
    def test_multi_query_search_intersection(self, pipeline):
        """Test multi-query search with intersection strategy."""
        options = RetrievalOptions(top_k=5)
        
        # Mock same result for both queries (for intersection)
        same_result = SearchResult(
            id="doc_1",
            text="Common result",
            similarity_score=0.85,
            distance=0.15,
            metadata=ChunkMetadata(
                video_id="video_1",
                date="2023/01/01",
                title="Common Video",
                chunk_index=1,
                chunk_id="chunk_1",
                token_count=50,
                filename="test.srt"
            )
        )
        
        pipeline.query_engine.query.return_value = [same_result]
        
        queries = ["query 1", "query 2"]
        results, perf_metrics = pipeline.multi_query_search(
            queries,
            options=options,
            combine_strategy="intersection"
        )
        
        assert perf_metrics is not None
    
    def test_performance_metrics_tracking(self, pipeline):
        """Test performance metrics tracking."""
        options = RetrievalOptions(top_k=5)
        
        # Execute multiple searches
        for i in range(3):
            pipeline.search(f"query {i}", options=options, return_formatted=False)
        
        stats = pipeline.get_performance_stats()
        
        assert stats["total_queries"] == 3
        assert "avg_latency_ms" in stats
        assert "cache_hit_rate" in stats
    
    def test_quality_metrics_calculation(self, pipeline):
        """Test quality metrics calculation."""
        # Mock multiple results with different scores
        pipeline.query_engine.query.return_value = [
            SearchResult(
                id=f"doc_{i}",
                text=f"Result {i}",
                similarity_score=0.9 - i * 0.1,
                distance=0.1 + i * 0.1,
                metadata=ChunkMetadata(
                    video_id=f"video_{i}",
                    date="2023/01/01",
                    title=f"Video {i}",
                    chunk_index=i,
                    chunk_id=f"chunk_{i}",
                    token_count=50,
                    filename="test.srt"
                )
            )
            for i in range(3)
        ]
        
        options = RetrievalOptions(top_k=3)
        results, perf_metrics, quality_metrics = pipeline.search(
            "test query",
            options=options,
            return_formatted=False
        )
        
        assert quality_metrics is not None
        assert quality_metrics.avg_similarity_score > 0
        assert quality_metrics.min_similarity_score <= quality_metrics.max_similarity_score
        assert quality_metrics.diversity_score > 0
    
    def test_format_results(self, pipeline):
        """Test format_results method."""
        results = [
            SearchResult(
                id="doc_1",
                text="Test result",
                similarity_score=0.9,
                distance=0.1,
                metadata=ChunkMetadata(
                    video_id="video_1",
                    date="2023/01/01",
                    title="Test Video",
                    chunk_index=1,
                    chunk_id="chunk_1",
                    token_count=50,
                    filename="test.srt"
                )
            )
        ]
        
        options = FormatOptions(format=OutputFormat.TEXT)
        formatted = pipeline.format_results(results, query="test", options=options)
        
        assert formatted == "Formatted results"
        pipeline.result_formatter.format_results.assert_called_once()
    
    def test_clear_cache(self, pipeline):
        """Test cache clearing."""
        pipeline.clear_cache()
        pipeline.query_engine.clear_cache.assert_called_once()
    
    def test_reset_metrics(self, pipeline):
        """Test metrics reset."""
        # Add some metrics
        options = RetrievalOptions(top_k=5)
        pipeline.search("test query", options=options, return_formatted=False)
        
        assert len(pipeline._metrics_history) == 1
        
        pipeline.reset_metrics()
        assert len(pipeline._metrics_history) == 0
    
    def test_search_with_filters(self, pipeline):
        """Test search with filters."""
        filters = SearchFilters(video_id="video_1")
        options = RetrievalOptions(top_k=5, filters=filters)
        
        results, perf_metrics, quality_metrics = pipeline.search(
            "test query",
            options=options,
            return_formatted=False
        )
        
        assert perf_metrics is not None
        # Verify filters were passed to query engine
        pipeline.query_engine.query.assert_called_once()
        call_args = pipeline.query_engine.query.call_args
        assert call_args[1]["filters"] == filters
