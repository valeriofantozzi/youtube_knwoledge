"""
Query Engine Module

Main retrieval interface for semantic search with query preprocessing, validation, and caching.
"""

from typing import List, Dict, Any, Optional
from dataclasses import dataclass
import hashlib
import json

from .similarity_search import SimilaritySearch, SearchFilters, SearchResult
from ..utils.logger import get_default_logger


@dataclass
class QueryOptions:
    """Options for query execution."""
    top_k: int = 10
    score_threshold: Optional[float] = None
    include_metadata: bool = True
    filters: Optional[SearchFilters] = None


class QueryEngine:
    """Main query interface for semantic search."""
    
    # Minimum and maximum query length (in characters)
    MIN_QUERY_LENGTH = 1
    MAX_QUERY_LENGTH = 1000
    
    def __init__(
        self,
        similarity_search: Optional[SimilaritySearch] = None,
        enable_caching: bool = True,
        cache_size: int = 100
    ):
        """
        Initialize query engine.
        
        Args:
            similarity_search: SimilaritySearch instance (creates new if None)
            enable_caching: Whether to enable query caching
            cache_size: Maximum number of cached queries
        """
        self.similarity_search = similarity_search or SimilaritySearch()
        self.enable_caching = enable_caching
        self.cache_size = cache_size
        self.logger = get_default_logger()
        
        # Simple in-memory cache (query_hash -> results)
        self._cache: Dict[str, List[SearchResult]] = {}
    
    def query(
        self,
        query_text: str,
        top_k: int = 10,
        score_threshold: Optional[float] = None,
        include_metadata: bool = True,
        filters: Optional[SearchFilters] = None,
        use_cache: bool = True
    ) -> List[SearchResult]:
        """
        Execute a natural language query.
        
        Args:
            query_text: Natural language query string
            top_k: Number of results to return (default: 10)
            score_threshold: Minimum similarity score (0.0-1.0)
            include_metadata: Whether to include metadata in results
            filters: SearchFilters object for filtering results
            use_cache: Whether to use cached results if available
        
        Returns:
            List of SearchResult objects, sorted by similarity score (descending)
        """
        # Validate query
        self._validate_query(query_text)
        
        # Preprocess query
        processed_query = self._preprocess_query(query_text)
        
        # Check cache
        if self.enable_caching and use_cache:
            cache_key = self._get_cache_key(
                processed_query,
                top_k,
                score_threshold,
                filters
            )
            
            if cache_key in self._cache:
                self.logger.debug(f"Cache hit for query: '{query_text[:50]}...'")
                return self._cache[cache_key]
        
        # Execute search
        self.logger.debug(f"Executing query: '{query_text[:50]}...'")
        results = self.similarity_search.search(
            query=processed_query,
            top_k=top_k,
            score_threshold=score_threshold,
            include_metadata=include_metadata,
            filters=filters
        )
        
        # Cache results
        if self.enable_caching and use_cache:
            cache_key = self._get_cache_key(
                processed_query,
                top_k,
                score_threshold,
                filters
            )
            self._cache_result(cache_key, results)
        
        return results
    
    def batch_query(
        self,
        queries: List[str],
        top_k: int = 10,
        score_threshold: Optional[float] = None,
        include_metadata: bool = True,
        filters: Optional[SearchFilters] = None
    ) -> List[List[SearchResult]]:
        """
        Execute multiple queries in batch.
        
        Args:
            queries: List of query strings
            top_k: Number of results per query
            score_threshold: Minimum similarity score
            include_metadata: Whether to include metadata
            filters: SearchFilters object for filtering results
        
        Returns:
            List of SearchResult lists, one per query
        """
        if not queries:
            return []
        
        # Validate all queries
        for query in queries:
            self._validate_query(query)
        
        # Preprocess all queries
        processed_queries = [self._preprocess_query(q) for q in queries]
        
        # Execute batch search
        results_list = self.similarity_search.batch_search(
            queries=processed_queries,
            top_k=top_k,
            score_threshold=score_threshold,
            include_metadata=include_metadata,
            filters=filters
        )
        
        return results_list
    
    def _preprocess_query(self, query: str) -> str:
        """
        Preprocess query text.
        
        Args:
            query: Raw query string
        
        Returns:
            Preprocessed query string
        """
        # Trim whitespace
        query = query.strip()
        
        # Normalize whitespace (replace multiple spaces with single space)
        import re
        query = re.sub(r'\s+', ' ', query)
        
        # Remove leading/trailing punctuation (optional, can be customized)
        # query = query.strip('.,!?;:')
        
        return query
    
    def _validate_query(self, query: str) -> None:
        """
        Validate query input.
        
        Args:
            query: Query string to validate
        
        Raises:
            ValueError: If query is invalid
        """
        if not isinstance(query, str):
            raise ValueError(f"Query must be a string, got {type(query)}")
        
        if not query or not query.strip():
            raise ValueError("Query cannot be empty")
        
        query_length = len(query.strip())
        
        if query_length < self.MIN_QUERY_LENGTH:
            raise ValueError(
                f"Query too short: minimum length is {self.MIN_QUERY_LENGTH} characters"
            )
        
        if query_length > self.MAX_QUERY_LENGTH:
            raise ValueError(
                f"Query too long: maximum length is {self.MAX_QUERY_LENGTH} characters"
            )
    
    def _get_cache_key(
        self,
        query: str,
        top_k: int,
        score_threshold: Optional[float],
        filters: Optional[SearchFilters]
    ) -> str:
        """
        Generate cache key for query.
        
        Args:
            query: Processed query string
            top_k: Number of results
            score_threshold: Minimum similarity score
            filters: SearchFilters object
        
        Returns:
            Cache key string (hash)
        """
        # Create dictionary of cache parameters
        cache_params = {
            "query": query.lower(),  # Case-insensitive
            "top_k": top_k,
            "score_threshold": score_threshold,
        }
        
        # Add filter parameters if present
        if filters:
            cache_params["filters"] = {
                "video_id": filters.video_id,
                "date_start": filters.date_start,
                "date_end": filters.date_end,
                "title_keywords": filters.title_keywords,
            }
        
        # Generate hash from JSON representation
        cache_str = json.dumps(cache_params, sort_keys=True)
        cache_hash = hashlib.md5(cache_str.encode()).hexdigest()
        
        return cache_hash
    
    def _cache_result(self, cache_key: str, results: List[SearchResult]) -> None:
        """
        Cache query results.
        
        Args:
            cache_key: Cache key
            results: Search results to cache
        """
        # Simple LRU-like eviction: remove oldest entries if cache is full
        if len(self._cache) >= self.cache_size:
            # Remove first (oldest) entry
            oldest_key = next(iter(self._cache))
            del self._cache[oldest_key]
            self.logger.debug(f"Evicted cache entry: {oldest_key[:8]}...")
        
        self._cache[cache_key] = results
        self.logger.debug(f"Cached query result: {cache_key[:8]}... ({len(results)} results)")
    
    def clear_cache(self) -> None:
        """Clear query cache."""
        cache_size = len(self._cache)
        self._cache.clear()
        self.logger.info(f"Cleared query cache ({cache_size} entries)")
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """
        Get cache statistics.
        
        Returns:
            Dictionary with cache statistics
        """
        return {
            "enabled": self.enable_caching,
            "size": len(self._cache),
            "max_size": self.cache_size,
            "usage_percent": (len(self._cache) / self.cache_size * 100) if self.cache_size > 0 else 0
        }
    
    def query_with_options(
        self,
        query_text: str,
        options: QueryOptions
    ) -> List[SearchResult]:
        """
        Execute query with QueryOptions object.
        
        Args:
            query_text: Natural language query string
            options: QueryOptions object with search parameters
        
        Returns:
            List of SearchResult objects
        """
        return self.query(
            query_text=query_text,
            top_k=options.top_k,
            score_threshold=options.score_threshold,
            include_metadata=options.include_metadata,
            filters=options.filters
        )
