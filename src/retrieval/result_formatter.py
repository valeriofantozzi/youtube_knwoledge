"""
Result Formatter Module

Formats search results for display in multiple formats with context expansion and deduplication.
"""

from typing import List, Dict, Any, Optional, Set
from dataclasses import dataclass
from enum import Enum
import json
from datetime import datetime

from .similarity_search import SearchResult
from ..vector_store.chroma_manager import ChromaDBManager
from ..utils.logger import get_default_logger


class OutputFormat(Enum):
    """Output format options."""
    TEXT = "text"
    JSON = "json"
    MARKDOWN = "markdown"


@dataclass
class FormatOptions:
    """Options for result formatting."""
    format: OutputFormat = OutputFormat.TEXT
    include_context: bool = False
    context_window: int = 2  # Number of surrounding chunks to include
    deduplicate: bool = True
    merge_adjacent: bool = True
    max_text_length: Optional[int] = None  # Truncate text if longer
    highlight_query: Optional[str] = None  # Query text for highlighting
    score_format: str = "decimal"  # "decimal" or "percentage"


class ResultFormatter:
    """Formats search results for display."""
    
    def __init__(
        self,
        chroma_manager: Optional[ChromaDBManager] = None,
        options: Optional[FormatOptions] = None
    ):
        """
        Initialize result formatter.
        
        Args:
            chroma_manager: ChromaDBManager instance (needed for context expansion)
            options: FormatOptions instance (uses defaults if None)
        """
        self.chroma_manager = chroma_manager
        self.options = options or FormatOptions()
        self.logger = get_default_logger()
    
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
            query: Original query text (for highlighting)
            options: FormatOptions (uses instance default if None)
        
        Returns:
            Formatted string in requested format
        """
        if not results:
            return self._format_empty_results(options or self.options)
        
        opts = options or self.options
        
        # Apply deduplication and merging if requested
        if opts.deduplicate or opts.merge_adjacent:
            results = self._deduplicate_and_merge(results, opts)
        
        # Expand context if requested
        if opts.include_context:
            results = self._expand_context(results, opts.context_window)
        
        # Format based on output format
        if opts.format == OutputFormat.JSON:
            return self._format_json(results, query)
        elif opts.format == OutputFormat.MARKDOWN:
            return self._format_markdown(results, query, opts)
        else:  # TEXT
            return self._format_text(results, query, opts)
    
    def _format_text(
        self,
        results: List[SearchResult],
        query: Optional[str],
        options: FormatOptions
    ) -> str:
        """Format results as human-readable text."""
        lines = []
        lines.append("=" * 80)
        lines.append(f"SEARCH RESULTS ({len(results)} results)")
        lines.append("=" * 80)
        lines.append("")
        
        for i, result in enumerate(results, 1):
            lines.append(f"Result {i}:")
            lines.append("-" * 80)
            
            # Format metadata
            metadata_lines = []
            if result.metadata:
                metadata_lines.append(f"Source ID: {result.metadata.source_id}")
                metadata_lines.append(f"Title: {result.metadata.title}")
                metadata_lines.append(f"Date: {result.metadata.date}")
                metadata_lines.append(f"Chunk Index: {result.metadata.chunk_index}")
                if hasattr(result.metadata, 'content_type') and result.metadata.content_type:
                    metadata_lines.append(f"Content Type: {result.metadata.content_type}")
            
            # Format similarity score
            score_str = self._format_score(result.similarity_score, options.score_format)
            metadata_lines.append(f"Similarity: {score_str}")
            
            lines.extend(metadata_lines)
            lines.append("")
            
            # Format text
            text = result.text
            if options.max_text_length and len(text) > options.max_text_length:
                text = text[:options.max_text_length] + "..."
            
            # Highlight query terms if provided
            if options.highlight_query and query:
                text = self._highlight_text(text, query)
            
            lines.append("Text:")
            lines.append(text)
            lines.append("")
        
        return "\n".join(lines)
    
    def _format_markdown(
        self,
        results: List[SearchResult],
        query: Optional[str],
        options: FormatOptions
    ) -> str:
        """Format results as Markdown."""
        lines = []
        lines.append(f"# Search Results ({len(results)} results)")
        lines.append("")
        
        for i, result in enumerate(results, 1):
            lines.append(f"## Result {i}")
            lines.append("")
            
            # Format metadata as table
            lines.append("| Field | Value |")
            lines.append("|-------|-------|")
            if result.metadata:
                lines.append(f"| Source ID | `{result.metadata.source_id}` |")
                lines.append(f"| Title | {result.metadata.title} |")
                lines.append(f"| Date | {result.metadata.date} |")
                lines.append(f"| Chunk Index | {result.metadata.chunk_index} |")
                if hasattr(result.metadata, 'content_type') and result.metadata.content_type:
                    lines.append(f"| Content Type | {result.metadata.content_type} |")
            
            score_str = self._format_score(result.similarity_score, options.score_format)
            lines.append(f"| Similarity | **{score_str}** |")
            lines.append("")
            
            # Format text
            text = result.text
            if options.max_text_length and len(text) > options.max_text_length:
                text = text[:options.max_text_length] + "..."
            
            # Highlight query terms
            if options.highlight_query and query:
                text = self._highlight_text_markdown(text, query)
            
            lines.append("### Text")
            lines.append("")
            lines.append(text)
            lines.append("")
        
        return "\n".join(lines)
    
    def _format_json(
        self,
        results: List[SearchResult],
        query: Optional[str]
    ) -> str:
        """Format results as JSON."""
        output = {
            "query": query,
            "count": len(results),
            "results": [result.to_dict() for result in results]
        }
        return json.dumps(output, indent=2, ensure_ascii=False)
    
    def _format_score(self, score: float, format_type: str) -> str:
        """Format similarity score."""
        if format_type == "percentage":
            return f"{score * 100:.1f}%"
        else:  # decimal
            return f"{score:.3f}"
    
    def _highlight_text(self, text: str, query: str) -> str:
        """Highlight query terms in text (simple case-insensitive matching)."""
        # Simple word-based highlighting
        words = query.lower().split()
        highlighted_text = text
        
        for word in words:
            if len(word) < 3:  # Skip very short words
                continue
            # Case-insensitive replacement
            import re
            pattern = re.compile(re.escape(word), re.IGNORECASE)
            highlighted_text = pattern.sub(
                lambda m: f"**{m.group()}**",
                highlighted_text
            )
        
        return highlighted_text
    
    def _highlight_text_markdown(self, text: str, query: str) -> str:
        """Highlight query terms in text for Markdown (already uses **)."""
        return self._highlight_text(text, query)
    
    def _format_empty_results(self, options: FormatOptions) -> str:
        """Format message for empty results."""
        if options.format == OutputFormat.JSON:
            return json.dumps({"count": 0, "results": []}, indent=2)
        elif options.format == OutputFormat.MARKDOWN:
            return "## No Results Found\n\nNo results match your query."
        else:
            return "No results found."
    
    def _deduplicate_and_merge(
        self,
        results: List[SearchResult],
        options: FormatOptions
    ) -> List[SearchResult]:
        """
        Deduplicate and merge adjacent results from the same source document.
        
        Args:
            results: List of SearchResult objects
            options: FormatOptions
        
        Returns:
            Deduplicated and merged list of results
        """
        if not results:
            return results
        
        # Group by source_id
        source_groups: Dict[str, List[SearchResult]] = {}
        for result in results:
            source_id = result.metadata.source_id if result.metadata else "unknown"
            if source_id not in source_groups:
                source_groups[source_id] = []
            source_groups[source_id].append(result)
        
        deduplicated = []
        
        for source_id, source_results in source_groups.items():
            if options.deduplicate:
                # Remove exact duplicates (same chunk_id)
                seen_chunks: Set[str] = set()
                unique_results = []
                for result in source_results:
                    chunk_id = result.metadata.chunk_id if result.metadata else result.id
                    if chunk_id not in seen_chunks:
                        seen_chunks.add(chunk_id)
                        unique_results.append(result)
                source_results = unique_results
            
            if options.merge_adjacent and len(source_results) > 1:
                # Sort by chunk_index
                source_results.sort(key=lambda r: r.metadata.chunk_index if r.metadata else 0)
                
                # Merge adjacent chunks
                merged_results = []
                current_group = [source_results[0]]
                
                for result in source_results[1:]:
                    prev_result = current_group[-1]
                    prev_index = prev_result.metadata.chunk_index if prev_result.metadata else 0
                    curr_index = result.metadata.chunk_index if result.metadata else 0
                    
                    # Check if adjacent (within 2 chunks)
                    if curr_index - prev_index <= 2:
                        current_group.append(result)
                    else:
                        # Merge current group and start new one
                        merged_results.append(self._merge_result_group(current_group))
                        current_group = [result]
                
                # Merge last group
                if current_group:
                    merged_results.append(self._merge_result_group(current_group))
                
                deduplicated.extend(merged_results)
            else:
                deduplicated.extend(source_results)
        
        # Re-sort by similarity score
        deduplicated.sort(key=lambda r: r.similarity_score, reverse=True)
        
        return deduplicated
    
    def _merge_result_group(self, group: List[SearchResult]) -> SearchResult:
        """
        Merge a group of adjacent results into a single result.
        
        Args:
            group: List of SearchResult objects from same source document
        
        Returns:
            Merged SearchResult
        """
        if len(group) == 1:
            return group[0]
        
        # Use the result with highest similarity as base
        base_result = max(group, key=lambda r: r.similarity_score)
        
        # Merge texts
        texts = [r.text for r in group]
        merged_text = " ... ".join(texts)
        
        # Create merged metadata
        merged_metadata = base_result.metadata
        if merged_metadata:
            # Update chunk_index range
            indices = [r.metadata.chunk_index for r in group if r.metadata]
            if indices:
                min_index = min(indices)
                max_index = max(indices)
                # Create a new metadata with range info
                from copy import copy
                merged_metadata = copy(merged_metadata)
                merged_metadata.chunk_index = min_index
        
        # Create merged result
        merged_result = SearchResult(
            id=base_result.id,
            text=merged_text,
            similarity_score=base_result.similarity_score,  # Use highest score
            distance=base_result.distance,
            metadata=merged_metadata
        )
        
        return merged_result
    
    def _expand_context(
        self,
        results: List[SearchResult],
        context_window: int
    ) -> List[SearchResult]:
        """
        Expand results with surrounding chunks for context.
        
        Args:
            results: List of SearchResult objects
            context_window: Number of surrounding chunks to include
        
        Returns:
            List of results with expanded context
        """
        if not self.chroma_manager or context_window <= 0:
            return results
        
        expanded_results = []
        
        try:
            collection = self.chroma_manager.get_or_create_collection()
            
            for result in results:
                if not result.metadata:
                    expanded_results.append(result)
                    continue
                
                source_id = result.metadata.source_id
                chunk_index = result.metadata.chunk_index
                
                # Get surrounding chunks from same source document
                context_chunks = []
                
                # Query for chunks in range [chunk_index - context_window, chunk_index + context_window]
                # Filter by source_id and chunk_index range
                try:
                    # Get all chunks from this source document
                    source_chunks = collection.get(
                        where={"source_id": {"$eq": source_id}},
                        include=['documents', 'metadatas']
                    )
                    
                    # Find chunks in context window
                    for i, meta in enumerate(source_chunks.get('metadatas', [])):
                        if meta and 'chunk_index' in meta:
                            idx = int(meta['chunk_index'])
                            if abs(idx - chunk_index) <= context_window and idx != chunk_index:
                                doc_text = source_chunks.get('documents', [])[i]
                                if doc_text:
                                    context_chunks.append((idx, doc_text))
                    
                    # Sort by chunk_index
                    context_chunks.sort(key=lambda x: x[0])
                    
                    # Build expanded text
                    expanded_text = result.text
                    if context_chunks:
                        before_texts = [t for idx, t in context_chunks if idx < chunk_index]
                        after_texts = [t for idx, t in context_chunks if idx > chunk_index]
                        
                        if before_texts:
                            expanded_text = " ... ".join(before_texts[-context_window:]) + " ... " + expanded_text
                        if after_texts:
                            expanded_text = expanded_text + " ... " + " ... ".join(after_texts[:context_window])
                    
                    # Create expanded result
                    expanded_result = SearchResult(
                        id=result.id,
                        text=expanded_text,
                        similarity_score=result.similarity_score,
                        distance=result.distance,
                        metadata=result.metadata
                    )
                    expanded_results.append(expanded_result)
                
                except Exception as e:
                    self.logger.warning(f"Failed to expand context for result {result.id}: {e}")
                    expanded_results.append(result)
        
        except Exception as e:
            self.logger.error(f"Error expanding context: {e}", exc_info=True)
            return results
        
        return expanded_results
    
    def format_single_result(
        self,
        result: SearchResult,
        query: Optional[str] = None,
        options: Optional[FormatOptions] = None
    ) -> str:
        """
        Format a single search result.
        
        Args:
            result: SearchResult object
            query: Original query text
            options: FormatOptions
        
        Returns:
            Formatted string
        """
        return self.format_results([result], query, options)
