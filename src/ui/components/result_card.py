"""
Result Card Component

Provides styled search result cards with similarity scores and highlighting.
"""

import streamlit as st
import re
from typing import Optional, Dict, Any
from ..theme import get_score_color, get_score_label


def score_bar(
    score: float,
    label: str = "Match",
    show_quality: bool = True
) -> None:
    """
    Display a visual similarity score indicator with color coding.
    
    Args:
        score: Similarity score (0.0 to 1.0)
        label: Score label
        show_quality: Whether to show quality label
    """
    color = get_score_color(score)
    quality = get_score_label(score) if show_quality else ""
    
    filled = int(score * 20)
    bar = "█" * filled + "░" * (20 - filled)
    
    quality_text = f' <span style="color: gray; font-size: 0.8em;">({quality})</span>' if quality else ""
    
    st.markdown(f"""
    <div style="font-family: monospace;">
        <span style="color: {color};">{bar}</span>
        <strong>{score:.1%}</strong> {label}
        {quality_text}
    </div>
    """, unsafe_allow_html=True)


def highlight_query_terms(text: str, query: str) -> str:
    """
    Highlight query terms in text.
    
    Args:
        text: Text to highlight
        query: Query string
    
    Returns:
        Text with highlighted terms
    """
    if not query:
        return text
    
    # Split query into terms
    terms = query.lower().split()
    
    # Highlight each term
    highlighted = text
    for term in terms:
        if len(term) > 2:  # Skip very short terms
            pattern = re.compile(re.escape(term), re.IGNORECASE)
            highlighted = pattern.sub(f"**{term}**", highlighted)
    
    return highlighted


def render_result_card(
    rank: int,
    text: str,
    score: float,
    metadata: Dict[str, Any],
    query: str = "",
    show_full: bool = False,
    max_preview_length: int = 200
) -> None:
    """
    Render a search result card.
    
    Args:
        rank: Result rank (1-based)
        text: Document text
        score: Similarity score
        metadata: Document metadata
        query: Search query for highlighting
        show_full: Whether to show full text
        max_preview_length: Max characters for preview
    """
    title = metadata.get("title", "Untitled")
    source_id = metadata.get("source_id", "N/A")
    date = metadata.get("date", "N/A")
    chunk_index = metadata.get("chunk_index", "N/A")
    
    with st.container(border=True):
        # Score bar
        score_bar(score)
        
        # Metadata row
        col1, col2 = st.columns([3, 1])
        with col1:
            st.markdown(f"**🎬 {title[:60]}{'...' if len(title) > 60 else ''}**")
        with col2:
            st.caption(f"📅 {date}")
        
        # Additional metadata
        st.caption(f"Chunk {chunk_index} • Source ID: `{source_id}`")
        
        # Text content
        if show_full:
            display_text = text
        else:
            display_text = text[:max_preview_length]
            if len(text) > max_preview_length:
                display_text += "..."
        
        # Highlight query terms if provided
        if query:
            display_text = highlight_query_terms(display_text, query)
        
        st.markdown(display_text)
        
        # Action buttons
        btn_col1, btn_col2, btn_col3 = st.columns(3)
        with btn_col1:
            expand_key = f"expand_{rank}"
            if st.button("📖 Full Text", key=f"full_{rank}"):
                st.session_state[expand_key] = not st.session_state.get(expand_key, False)
        with btn_col2:
            st.button("🔗 Context", key=f"ctx_{rank}", disabled=True)  # Future feature
        with btn_col3:
            if source_id and source_id != "N/A":
                st.link_button(
                    "▶️ Source",
                    f"#source-{source_id}",
                    use_container_width=True
                )
        
        # Show full text if expanded
        if st.session_state.get(f"expand_{rank}", False):
            st.markdown("---")
            st.markdown("**Full Text:**")
            st.write(text)


def render_result_list(
    results: list,
    query: str = "",
    show_export: bool = True
) -> None:
    """
    Render a list of search results.
    
    Args:
        results: List of result dictionaries
        query: Search query for highlighting
        show_export: Whether to show export button
    """
    if not results:
        st.info("🔍 No results found. Try a different query.")
        return
    
    # Results header
    col1, col2 = st.columns([3, 1])
    with col1:
        st.markdown(f"**📋 {len(results)} Results Found**")
    with col2:
        if show_export:
            # Export functionality would go here
            st.button("⬇️ Export", key="export_results", disabled=True)
    
    # Render each result
    for i, result in enumerate(results, 1):
        render_result_card(
            rank=i,
            text=result.get("text", ""),
            score=result.get("score", 0.0),
            metadata=result.get("metadata", {}),
            query=query
        )
