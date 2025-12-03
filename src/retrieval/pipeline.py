"""
Retrieval Pipeline Module

Main interface for the complete retrieval system integrating all components:
- Query Engine
- Similarity Search
- Result Formatter
- Performance Monitoring
- Multi-query Search
"""

from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum
import time
from collections import defaultdict

from .query_engine import QueryEngine
from .similarity_search import SimilaritySearch, SearchFilters, SearchResult
from .result_formatter import ResultFormatter, FormatOptions, OutputFormat
from ..utils.logger import get_logger


class SearchMode(Enum):
    """Search mode options."""
    SINGLE = "single"  # Single query
    MULTI = "multi"    # Multiple queries combined
    HYBRID = "hybrid"  # Semantic + keyword (future)


@dataclass
class PerformanceMetrics:
    """Performance metrics for search operations."""
    query_latency_ms: float = 0.0
    embedding_time_ms: float = 0.0
    search_time_ms: float = 0.0
    formatting_time_ms: float = 0.0
    num_results: int = 0
    cache_hit: bool = False
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "query_latency_ms": self.query_latency_ms,
            "embedding_time_ms": self.embedding_time_ms,
            "search_time_ms": self.search_time_ms,
            "formatting_time_ms": self.formatting_time_ms,
            "num_results": self.num_results,
            "cache_hit": self.cache_hit,
            "total_time_ms": self.query_latency_ms
        }


@dataclass
class SearchQualityMetrics:
    """Search quality metrics."""
    avg_similarity_score: float = 0.0
    min_similarity_score: float = 0.0
    max_similarity_score: float = 0.0
    score_variance: float = 0.0
    diversity_score: float = 0.0  # How diverse are the results (different source documents)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "avg_similarity_score": self.avg_similarity_score,
            "min_similarity_score": self.min_similarity_score,
            "max_similarity_score": self.max_similarity_score,
            "score_variance": self.score_variance,
            "diversity_score": self.diversity_score
        }


@dataclass
class RetrievalOptions:
    """Options for retrieval pipeline."""
    top_k: int = 10
    score_threshold: Optional[float] = None
    include_metadata: bool = True
    filters: Optional[SearchFilters] = None
    format_options: Optional[FormatOptions] = None
    return_metrics: bool = False
    mode: SearchMode = SearchMode.SINGLE


class RetrievalPipeline:
    """
    Main retrieval pipeline integrating all components.
    
    Provides a unified interface for semantic search with:
    - Query processing and validation
    - Similarity search
    - Result formatting
    - Performance monitoring
    - Multi-query search
    """
    
    def __init__(
        self,
        query_engine: Optional[QueryEngine] = None,
        result_formatter: Optional[ResultFormatter] = None,
        enable_monitoring: bool = True
    ):
        """
        Initialize retrieval pipeline.
        
        Args:
            query_engine: QueryEngine instance (creates default if None)
            result_formatter: ResultFormatter instance (creates default if None)
            enable_monitoring: Whether to track performance metrics
        """
        self.logger = get_logger(__name__)
        
        # Initialize components
        self.query_engine = query_engine or QueryEngine()
        self.result_formatter = result_formatter or ResultFormatter()
        self.enable_monitoring = enable_monitoring
        
        # Performance tracking
        self._metrics_history: List[PerformanceMetrics] = []
        self._max_history_size = 1000
    
    def search(
        self,
        query: str,
        options: Optional[RetrievalOptions] = None,
        return_formatted: bool = True
    ) -> Tuple[List[SearchResult], Optional[PerformanceMetrics], Optional[SearchQualityMetrics]]:
        """
        Execute a search query.
        
        Args:
            query: Natural language query string
            options: RetrievalOptions for search configuration
            return_formatted: Whether to format results (if False, returns raw results)
        
        Returns:
            Tuple of (results, performance_metrics, quality_metrics)
            - results: List of SearchResult objects
            - performance_metrics: PerformanceMetrics if monitoring enabled
            - quality_metrics: SearchQualityMetrics if monitoring enabled
        """
        if options is None:
            options = RetrievalOptions()
        
        start_time = time.time()
        perf_metrics = PerformanceMetrics() if self.enable_monitoring else None
        
        try:
            # Execute query
            results = self.query_engine.query(
                query_text=query,
                top_k=options.top_k,
                score_threshold=options.score_threshold,
                include_metadata=options.include_metadata,
                filters=options.filters,
                use_cache=True
            )
            
            # Track cache hit
            if perf_metrics:
                # Check if result was cached (simplified check)
                cache_key = self.query_engine._get_cache_key(
                    self.query_engine._preprocess_query(query),
                    options.top_k,
                    options.score_threshold,
                    options.filters
                )
                perf_metrics.cache_hit = cache_key in self.query_engine._cache
            
            # Calculate quality metrics
            quality_metrics = None
            if self.enable_monitoring and results:
                quality_metrics = self._calculate_quality_metrics(results)
            
            # Format results if requested
            if return_formatted and options.format_options:
                format_start = time.time()
                formatted_results = self.result_formatter.format_results(
                    results,
                    query=query,
                    options=options.format_options
                )
                if perf_metrics:
                    perf_metrics.formatting_time_ms = (time.time() - format_start) * 1000
                # Note: formatted_results is a string, but we return raw results
                # Caller can use format_results separately if needed
            
            # Calculate performance metrics
            if perf_metrics:
                perf_metrics.query_latency_ms = (time.time() - start_time) * 1000
                perf_metrics.num_results = len(results)
                self._record_metrics(perf_metrics)
            
            return results, perf_metrics, quality_metrics
        
        except Exception as e:
            self.logger.error(f"Error in search pipeline: {e}", exc_info=True)
            raise
    
    def multi_query_search(
        self,
        queries: List[str],
        options: Optional[RetrievalOptions] = None,
        combine_strategy: str = "union"
    ) -> Tuple[List[SearchResult], Optional[PerformanceMetrics]]:
        """
        Execute multiple queries and combine results.
        
        Args:
            queries: List of query strings
            options: RetrievalOptions for search configuration
            combine_strategy: How to combine results ("union", "intersection", "weighted")
        
        Returns:
            Tuple of (combined_results, performance_metrics)
        """
        if options is None:
            options = RetrievalOptions()
        
        start_time = time.time()
        perf_metrics = PerformanceMetrics() if self.enable_monitoring else None
        
        try:
            # Execute all queries
            all_results: Dict[str, SearchResult] = {}  # id -> result
            result_scores: Dict[str, List[float]] = defaultdict(list)  # id -> scores
            
            for query in queries:
                query_results = self.query_engine.query(
                    query_text=query,
                    top_k=options.top_k,
                    score_threshold=options.score_threshold,
                    include_metadata=options.include_metadata,
                    filters=options.filters,
                    use_cache=True
                )
                
                # Collect results with scores
                for result in query_results:
                    result_id = result.id
                    if result_id not in all_results:
                        all_results[result_id] = result
                    result_scores[result_id].append(result.similarity_score)
            
            # Combine results based on strategy
            combined_results = self._combine_results(
                list(all_results.values()),
                result_scores,
                combine_strategy,
                options.top_k
            )
            
            # Calculate performance metrics
            if perf_metrics:
                perf_metrics.query_latency_ms = (time.time() - start_time) * 1000
                perf_metrics.num_results = len(combined_results)
                self._record_metrics(perf_metrics)
            
            return combined_results, perf_metrics
        
        except Exception as e:
            self.logger.error(f"Error in multi-query search: {e}", exc_info=True)
            raise
    
    def _combine_results(
        self,
        results: List[SearchResult],
        result_scores: Dict[str, List[float]],
        strategy: str,
        top_k: int
    ) -> List[SearchResult]:
        """
        Combine results from multiple queries.
        
        Args:
            results: List of unique SearchResult objects
            result_scores: Dictionary mapping result IDs to lists of scores from each query
            strategy: Combination strategy ("union", "intersection", "weighted")
            top_k: Maximum number of results to return
        
        Returns:
            Combined and ranked list of SearchResult objects
        """
        if strategy == "union":
            # Return all unique results, sorted by max score
            for result in results:
                scores = result_scores[result.id]
                result.similarity_score = max(scores)
            
            return sorted(results, key=lambda x: x.similarity_score, reverse=True)[:top_k]
        
        elif strategy == "intersection":
            # Only return results that appear in all queries
            num_queries = len(result_scores[next(iter(result_scores))]) if result_scores else 0
            intersection_results = [
                result for result in results
                if len(result_scores[result.id]) == num_queries
            ]
            
            # Average scores for intersection
            for result in intersection_results:
                scores = result_scores[result.id]
                result.similarity_score = sum(scores) / len(scores)
            
            return sorted(intersection_results, key=lambda x: x.similarity_score, reverse=True)[:top_k]
        
        elif strategy == "weighted":
            # Weight results by average score and frequency
            # Calculate number of queries from result_scores
            num_queries = len(result_scores[next(iter(result_scores))]) if result_scores else 1
            
            for result in results:
                scores = result_scores[result.id]
                avg_score = sum(scores) / len(scores) if scores else 0.0
                frequency_weight = len(scores) / num_queries if num_queries > 0 else 1.0
                result.similarity_score = avg_score * (1 + frequency_weight * 0.2)  # Boost frequent results
            
            return sorted(results, key=lambda x: x.similarity_score, reverse=True)[:top_k]
        
        else:
            self.logger.warning(f"Unknown combine strategy: {strategy}, using union")
            return self._combine_results(results, result_scores, "union", top_k)
    
    def _calculate_quality_metrics(self, results: List[SearchResult]) -> SearchQualityMetrics:
        """
        Calculate search quality metrics.
        
        Args:
            results: List of SearchResult objects
        
        Returns:
            SearchQualityMetrics object
        """
        if not results:
            return SearchQualityMetrics()
        
        scores = [r.similarity_score for r in results]
        avg_score = sum(scores) / len(scores)
        min_score = min(scores)
        max_score = max(scores)
        
        # Calculate variance
        variance = sum((s - avg_score) ** 2 for s in scores) / len(scores)
        
        # Calculate diversity (fraction of unique sources)
        unique_sources = set(r.metadata.source_id for r in results)
        diversity_score = len(unique_sources) / len(results) if results else 0.0
        
        return SearchQualityMetrics(
            avg_similarity_score=avg_score,
            min_similarity_score=min_score,
            max_similarity_score=max_score,
            score_variance=variance,
            diversity_score=diversity_score
        )
    
    def _record_metrics(self, metrics: PerformanceMetrics):
        """Record performance metrics in history."""
        self._metrics_history.append(metrics)
        if len(self._metrics_history) > self._max_history_size:
            self._metrics_history.pop(0)
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """
        Get aggregated performance statistics.
        
        Returns:
            Dictionary with aggregated performance metrics
        """
        if not self._metrics_history:
            return {}
        
        latencies = [m.query_latency_ms for m in self._metrics_history]
        num_results = [m.num_results for m in self._metrics_history]
        cache_hits = sum(1 for m in self._metrics_history if m.cache_hit)
        
        return {
            "total_queries": len(self._metrics_history),
            "avg_latency_ms": sum(latencies) / len(latencies),
            "min_latency_ms": min(latencies),
            "max_latency_ms": max(latencies),
            "avg_num_results": sum(num_results) / len(num_results),
            "cache_hit_rate": cache_hits / len(self._metrics_history),
            "total_cache_hits": cache_hits
        }
    
    def format_results(
        self,
        results: List[SearchResult],
        query: Optional[str] = None,
        options: Optional[FormatOptions] = None
    ) -> str:
        """
        Format search results.
        
        Args:
            results: List of SearchResult objects
            query: Original query string (for highlighting)
            options: FormatOptions for formatting configuration
        
        Returns:
            Formatted string (text, JSON, or Markdown)
        """
        if options is None:
            options = FormatOptions()
        
        if query and options.highlight_query is None:
            options.highlight_query = query
        
        return self.result_formatter.format_results(results, query=query, options=options)
    
    def clear_cache(self):
        """Clear query cache."""
        self.query_engine.clear_cache()
    
    def reset_metrics(self):
        """Reset performance metrics history."""
        self._metrics_history.clear()

