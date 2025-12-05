"""
Search Page Module

Implements semantic search interface:
- Natural language query input
- Search filters (source ID, date range, title keywords, content type)
- Results display with similarity scores
- Result grouping and context expansion
"""

import streamlit as st
import pandas as pd
from typing import List, Dict, Any, Optional
from datetime import datetime

from ..state import reset_search_state
from ..theme import ICONS, COLORS, get_score_color, get_score_label
from ..components.feedback import (
    show_empty_state,
    show_info_callout,
)
from ..components.result_card import (
    render_result_card,
    score_bar,
    highlight_query_terms,
    get_full_document,
)


def get_unique_source_ids() -> List[str]:
    """
    Get list of unique source IDs from the collection.

    Returns:
        List of source IDs
    """
    if "collection" not in st.session_state or st.session_state.collection is None:
        return []

    try:
        # Get sample of metadata
        sample = st.session_state.collection.get(
            limit=min(5000, st.session_state.get("total_docs", 1000)),
            include=["metadatas"],
        )

        source_ids = set()
        for meta in sample.get("metadatas", []):
            if meta:
                sid = meta.get("source_id")
                if sid:
                    source_ids.add(sid)

        return sorted(list(source_ids))
    except Exception:
        return []


def perform_search(
    query: str,
    top_k: int,
    min_score: float,
    source_id_filter: Optional[str],
    date_start: Optional[str],
    date_end: Optional[str],
    title_keywords: Optional[str],
    content_type_filter: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """
    Perform semantic search.

    Args:
        query: Search query text
        top_k: Number of results
        min_score: Minimum similarity score
        source_id_filter: Filter by source document ID
        date_start: Filter by start date
        date_end: Filter by end date
        title_keywords: Filter by title keywords
        content_type_filter: Filter by content type (srt, text, markdown)

    Returns:
        List of search result dictionaries
    """
    try:
        from ...embeddings.embedder import Embedder
        from ...retrieval.similarity_search import SearchFilters

        # Initialize embedder
        embedder = Embedder()

        # Generate query embedding
        query_embedding = embedder.encode([query], is_query=True)[0]

        # Build filters
        where_clause = None
        conditions = []

        if source_id_filter and source_id_filter != "All Documents":
            conditions.append({"source_id": {"$eq": source_id_filter}})

        if date_start:
            conditions.append({"date": {"$gte": date_start}})

        if date_end:
            conditions.append({"date": {"$lte": date_end}})

        if content_type_filter and content_type_filter != "All Types":
            conditions.append({"content_type": {"$eq": content_type_filter}})

        if conditions:
            if len(conditions) == 1:
                where_clause = conditions[0]
            else:
                where_clause = {"$and": conditions}

        # Execute search
        results = st.session_state.collection.query(
            query_embeddings=[query_embedding.tolist()],
            n_results=top_k * 2,  # Get more to filter by score
            include=["documents", "metadatas", "distances"],
            where=where_clause,
        )

        if not results.get("ids") or not results["ids"][0]:
            return []

        # Process results
        processed_results = []
        for i, (doc_id, doc_text, metadata, distance) in enumerate(
            zip(
                results["ids"][0],
                results["documents"][0],
                results["metadatas"][0],
                results["distances"][0],
            )
        ):
            # Convert distance to similarity
            # ChromaDB default is Squared L2: distance = 2(1 - similarity) -> similarity = 1 - distance/2
            similarity = 1 - (distance / 2)

            # Filter by minimum score
            if similarity < min_score:
                continue

            # Filter by title keywords if provided
            if title_keywords:
                keywords = [k.strip().lower() for k in title_keywords.split(",")]
                title = metadata.get("title", "").lower() if metadata else ""
                if not any(kw in title for kw in keywords):
                    continue

            processed_results.append(
                {
                    "id": doc_id,
                    "text": doc_text,
                    "score": similarity,
                    "metadata": metadata or {},
                }
            )

            # Limit to requested top_k
            if len(processed_results) >= top_k:
                break

        return processed_results

    except Exception as e:
        st.error(f"Search error: {e}")
        return []


def render_search_input() -> str:
    """
    Render the search input section.

    Returns:
        Search query string
    """
    st.subheader(f"{ICONS['search']} Search Query")

    query = st.text_input(
        "Enter your search query",
        placeholder="e.g., How to care for orchids in winter?",
        help="Use natural language questions for best results.",
        key="search_query_input",
    )

    show_info_callout(
        "Tip: Use natural language questions or descriptive phrases for better results.",
        icon="💡",
    )

    return query


def render_search_options() -> Dict[str, Any]:
    """
    Render search options.

    Returns:
        Dictionary of search options
    """
    col1, col2 = st.columns(2)

    with col1:
        top_k = st.slider(
            "Number of results",
            min_value=1,
            max_value=50,
            value=10,
            help="Maximum number of results to return",
        )

    with col2:
        min_score = st.slider(
            "Minimum similarity score",
            min_value=0.0,
            max_value=1.0,
            value=0.3,
            step=0.05,
            help="Filter out results below this similarity score",
        )

    return {"top_k": top_k, "min_score": min_score}


def render_advanced_filters() -> Dict[str, Any]:
    """
    Render advanced filter section.

    Returns:
        Dictionary of filter values
    """
    with st.expander(f"{ICONS['settings']} Advanced Filters", expanded=False):
        col1, col2 = st.columns(2)

        with col1:
            # Source document filter
            source_ids = get_unique_source_ids()
            source_options = ["All Documents"] + source_ids

            source_id_filter = st.selectbox(
                "Filter by Document", options=source_options, index=0
            )

        with col2:
            # Content type filter
            content_types = ["All Types", "srt", "text", "markdown"]
            content_type_filter = st.selectbox(
                "Filter by Content Type",
                options=content_types,
                index=0,
                help="Filter by document format",
            )

        # Title keywords
        title_keywords = st.text_input(
            "Title keywords (comma-separated)",
            placeholder="keyword1, keyword2, keyword3",
            help="Filter results by keywords in document title",
        )

        # Date range
        st.write("**Date Range:**")
        date_col1, date_col2 = st.columns(2)

        with date_col1:
            date_start = st.text_input(
                "Start date",
                placeholder="YYYY/MM/DD",
                help="Filter results from this date onwards",
            )

        with date_col2:
            date_end = st.text_input(
                "End date",
                placeholder="YYYY/MM/DD",
                help="Filter results up to this date",
            )

    return {
        "source_id": source_id_filter if source_id_filter != "All Documents" else None,
        "content_type": content_type_filter
        if content_type_filter != "All Types"
        else None,
        "date_start": date_start if date_start else None,
        "date_end": date_end if date_end else None,
        "title_keywords": title_keywords if title_keywords else None,
    }


def render_search_results(results: List[Dict], query: str) -> None:
    """
    Render search results.

    Args:
        results: List of search result dictionaries
        query: Original search query for highlighting
    """
    if not results:
        show_empty_state(
            title="No Results Found",
            message="Try adjusting your search query or filters.",
            icon="🔍",
        )
        return

    # Results header
    st.markdown("---")
    col1, col2 = st.columns([3, 1])

    with col1:
        st.subheader(f"📋 Results ({len(results)} found)")

    with col2:
        # Export button
        if st.button(f"{ICONS['download']} Export", key="export_search_results"):
            export_results(results, query)

    # Results list
    for i, result in enumerate(results, 1):
        render_search_result_card(rank=i, result=result, query=query)


def render_search_result_card(rank: int, result: Dict, query: str) -> None:
    """
    Render a single search result card.

    Args:
        rank: Result rank
        result: Result dictionary
        query: Search query for highlighting
    """
    score = result.get("score", 0.0)
    text = result.get("text", "")
    metadata = result.get("metadata", {})

    title = metadata.get("title", "Untitled")
    source_id = metadata.get("source_id", "N/A")
    date = metadata.get("date", "N/A")
    chunk_index = metadata.get("chunk_index", "N/A")
    content_type = metadata.get("content_type", "unknown")

    # Color based on score
    score_color = get_score_color(score)
    quality = get_score_label(score)

    # Icon based on content type
    content_icons = {"srt": "🎬", "text": "📄", "markdown": "📝", "unknown": "📁"}
    content_icon = content_icons.get(content_type, "📁")

    with st.container(border=True):
        # Score bar
        filled = int(score * 20)
        bar = "█" * filled + "░" * (20 - filled)
        st.markdown(
            f"""
        <div style="font-family: monospace;">
            <span style="color: {score_color};">{bar}</span>
            <strong>{score:.1%}</strong> match
            <span style="color: gray; font-size: 0.8em;">({quality})</span>
        </div>
        """,
            unsafe_allow_html=True,
        )

        # Title and metadata
        col1, col2 = st.columns([3, 1])
        with col1:
            st.markdown(
                f"**{content_icon} {title[:60]}{'...' if len(title) > 60 else ''}**"
            )
        with col2:
            st.caption(f"📅 {date}")

        st.caption(
            f"Chunk {chunk_index} • Source: `{source_id}` • Type: {content_type}"
        )

        # Text preview with highlighting
        preview_length = 300
        preview_text = text[:preview_length]
        if len(text) > preview_length:
            preview_text += "..."

        # Simple highlighting
        highlighted = highlight_query_terms(preview_text, query)
        st.markdown(highlighted)

        # Action buttons
        btn_col1, btn_col2, btn_col3 = st.columns(3)

        with btn_col1:
            expand_key = f"expand_result_{rank}"
            if st.button("📖 Full Text", key=f"full_text_{rank}"):
                st.session_state[expand_key] = not st.session_state.get(
                    expand_key, False
                )

        with btn_col2:
            # Context button (future feature)
            st.button("🔗 Context", key=f"context_{rank}", disabled=True)

        with btn_col3:
            # Show link button only for SRT content with YouTube IDs (11 characters)
            if (
                content_type == "srt"
                and source_id
                and source_id != "N/A"
                and len(source_id) == 11
            ):
                st.link_button(
                    "▶️ YouTube",
                    f"https://youtube.com/watch?v={source_id}",
                    use_container_width=True,
                )

        # Expanded full text
        if st.session_state.get(f"expand_result_{rank}", False):
            st.markdown("---")
            st.markdown("**📄 Full Original Document:**")
            
            # Try to get the full document
            if source_id and source_id != "N/A" and chunk_index != "N/A":
                original_doc, filename, chunks = get_full_document(source_id, int(chunk_index), text)
                
                if original_doc:
                    # Try to find and highlight the current chunk in the original document
                    # Clean the chunk text for better matching
                    clean_chunk = text.strip()
                    
                    # Try to find the chunk in the original document
                    if clean_chunk in original_doc:
                        # Split the document around the chunk and highlight it
                        parts = original_doc.split(clean_chunk, 1)
                        if len(parts) == 2:
                            before, after = parts
                            st.markdown(before)
                            st.markdown(f"**{clean_chunk}**")
                            st.markdown(after)
                        else:
                            st.markdown(original_doc)
                    else:
                        # If exact match not found, show the full document
                        st.markdown(original_doc)
                        st.info("💡 Current chunk highlighted separately below:")
                        st.markdown(f"**Current chunk:** {text}")
                    
                    # Show document stats
                    st.caption(f"📊 Document reconstructed from {len(chunks)} chunks • Original file: `{filename}`")
                else:
                    # Fallback to just the current chunk
                    st.write(text)
            else:
                # Fallback to just the current chunk
                st.write(text)


def export_results(results: List[Dict], query: str) -> None:
    """
    Export search results.

    Args:
        results: List of search result dictionaries
        query: Search query
    """
    # Create DataFrame
    export_data = []
    for i, result in enumerate(results, 1):
        metadata = result.get("metadata", {})
        export_data.append(
            {
                "Rank": i,
                "Score": result.get("score", 0.0),
                "Source ID": metadata.get("source_id", "N/A"),
                "Title": metadata.get("title", "N/A"),
                "Date": metadata.get("date", "N/A"),
                "Content Type": metadata.get("content_type", "unknown"),
                "Chunk Index": metadata.get("chunk_index", "N/A"),
                "Text": result.get("text", ""),
            }
        )

    df = pd.DataFrame(export_data)

    st.download_button(
        label=f"{ICONS['download']} Download Results (CSV)",
        data=df.to_csv(index=False),
        file_name=f"search_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
        mime="text/csv",
    )


def render_search_page() -> None:
    """
    Render the Search page.

    This is the main entry point for the page module.
    """
    st.header(f"{ICONS['search']} Semantic Search")
    st.markdown("Find relevant content using natural language queries.")

    # Check if collection is loaded
    total_docs = st.session_state.get("total_docs", 0)
    if total_docs == 0:
        show_empty_state(
            title="No Documents Loaded",
            message="Load documents first to enable search.",
            icon="📭",
            action_label="Go to Load Documents",
            action_page="📥 Load Documents",
        )
        return

    st.markdown("---")

    # Search input
    query = render_search_input()

    # Search options
    options = render_search_options()

    # Advanced filters
    filters = render_advanced_filters()

    st.markdown("---")

    # Search button
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        search_button = st.button(
            f"{ICONS['search']} Search",
            type="primary",
            use_container_width=True,
            disabled=not query,
        )

    # Execute search
    if search_button and query:
        with st.spinner("Searching..."):
            results = perform_search(
                query=query,
                top_k=options["top_k"],
                min_score=options["min_score"],
                source_id_filter=filters.get("source_id"),
                date_start=filters.get("date_start"),
                date_end=filters.get("date_end"),
                title_keywords=filters.get("title_keywords"),
                content_type_filter=filters.get("content_type"),
            )

        # Store results in session state
        st.session_state["last_search_query"] = query
        st.session_state["last_search_results"] = results

        # Render results
        render_search_results(results, query)

    # Show cached results if available
    elif st.session_state.get("last_search_results"):
        cached_query = st.session_state.get("last_search_query", "")
        if cached_query:
            st.info(f'Showing results for: "{cached_query}"')
            render_search_results(st.session_state["last_search_results"], cached_query)
