"""
Retrieval module for semantic search over embeddings.

Handles query processing, similarity search, and result formatting.
"""

from .similarity_search import SimilaritySearch, SearchFilters, SearchResult
from .query_engine import QueryEngine, QueryOptions
from .result_formatter import ResultFormatter, FormatOptions, OutputFormat
from .pipeline import (
    RetrievalPipeline,
    RetrievalOptions,
    PerformanceMetrics,
    SearchQualityMetrics,
    SearchMode
)

__all__ = [
    # Core components
    "SimilaritySearch",
    "SearchFilters",
    "SearchResult",
    "QueryEngine",
    "QueryOptions",
    "ResultFormatter",
    "FormatOptions",
    "OutputFormat",
    # Pipeline
    "RetrievalPipeline",
    "RetrievalOptions",
    "PerformanceMetrics",
    "SearchQualityMetrics",
    "SearchMode",
]

