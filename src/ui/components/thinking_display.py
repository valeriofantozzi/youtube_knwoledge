"""
Thinking Display Component

Streamlit component for displaying agent thinking process in real-time
with dynamic status updates from multiple agents.
"""

import streamlit as st
from typing import List, Optional, Dict, Any
from src.ai_search.thinking import ThinkingUpdate, ThinkingStatus


def get_status_icon(status: ThinkingStatus) -> str:
    """Get emoji icon for status type."""
    icons = {
        ThinkingStatus.ANALYZING: "🔍",
        ThinkingStatus.PROCESSING: "⚙️",
        ThinkingStatus.RETRIEVING: "📚",
        ThinkingStatus.GENERATING: "✍️",
        ThinkingStatus.REASONING: "🧠",
        ThinkingStatus.COMPLETE: "✅",
        ThinkingStatus.ERROR: "❌",
    }
    return icons.get(status, "⏳")


def get_status_color(status: ThinkingStatus) -> str:
    """Get color for status type (for markdown)."""
    colors = {
        ThinkingStatus.ANALYZING: "#3498db",      # Blue
        ThinkingStatus.PROCESSING: "#9b59b6",     # Purple
        ThinkingStatus.RETRIEVING: "#e74c3c",     # Red
        ThinkingStatus.GENERATING: "#2ecc71",     # Green
        ThinkingStatus.REASONING: "#f39c12",      # Orange
        ThinkingStatus.COMPLETE: "#27ae60",       # Dark Green
        ThinkingStatus.ERROR: "#c0392b",          # Dark Red
    }
    return colors.get(status, "#95a5a6")


def render_thinking_update(update: ThinkingUpdate) -> None:
    """Render a single thinking update."""
    icon = get_status_icon(update.status)
    color = get_status_color(update.status)
    
    # Format timestamp
    timestamp = update.timestamp.split("T")[1].split(".")[0] if "T" in update.timestamp else ""
    
    # Create the phase title with agent name and icon
    title_html = f'<span style="color: {color}; font-weight: bold;">{icon} {update.phase_title}</span>'
    st.markdown(title_html, unsafe_allow_html=True)
    
    # Show agent name and details
    if update.details:
        st.caption(f"**{update.agent_name}**: {update.details}")
    else:
        st.caption(f"**{update.agent_name}**")
    
    # Show progress bar if applicable
    if update.status not in [ThinkingStatus.COMPLETE, ThinkingStatus.ERROR]:
        if update.progress > 0:
            st.progress(update.progress, text=f"{int(update.progress * 100)}%")
    
    # Show metadata if available
    if update.metadata:
        with st.expander("📊 Details", expanded=False):
            st.json(update.metadata)


def render_thinking_session(updates: List[ThinkingUpdate]) -> None:
    """
    Render a complete thinking session with all updates from different agents.
    
    Args:
        updates: List of ThinkingUpdate objects from agents
    """
    if not updates:
        return
    
    # Group updates by agent
    agent_updates: Dict[str, List[ThinkingUpdate]] = {}
    for update in updates:
        if update.agent_name not in agent_updates:
            agent_updates[update.agent_name] = []
        agent_updates[update.agent_name].append(update)
    
    # Create tabs for each agent
    if len(agent_updates) > 1:
        tabs = st.tabs([f"🤖 {agent}" for agent in agent_updates.keys()])
        
        for tab, (agent_name, agent_thinking) in zip(tabs, agent_updates.items()):
            with tab:
                for update in agent_thinking:
                    render_thinking_update(update)
                    st.divider()
    else:
        # Single agent - show all updates in sequence
        for update in updates:
            render_thinking_update(update)
            st.divider()


def render_thinking_stream(
    container,
    thinking_updates: List[ThinkingUpdate],
    show_details: bool = False,
) -> None:
    """
    Render thinking updates in a container with optional details.
    
    This is useful for real-time streaming of thinking updates.
    
    Args:
        container: Streamlit container (st.container, st.expander, etc.)
        thinking_updates: List of ThinkingUpdate objects
        show_details: Whether to show expanded details
    """
    with container:
        if not thinking_updates:
            st.markdown("*No thinking updates captured*")
            return
        
        # Show most recent first
        for update in reversed(thinking_updates):
            col1, col2 = st.columns([0.1, 0.9])
            
            with col1:
                icon = get_status_icon(update.status)
                color = get_status_color(update.status)
                st.markdown(f'<span style="color: {color}; font-size: 20px;">{icon}</span>', unsafe_allow_html=True)
            
            with col2:
                st.markdown(f"**{update.phase_title}**")
                if update.details:
                    st.caption(update.details)
                st.caption(f"*{update.agent_name}*")
                
                if show_details and update.metadata:
                    with st.expander("Show details"):
                        st.json(update.metadata)


def render_thinking_status_simple(updates: List[ThinkingUpdate]) -> None:
    """
    Render the latest thinking status as a simple single line.
    Shows only the current phase without details or progress bars.
    
    Args:
        updates: List of ThinkingUpdate objects
    """
    if not updates:
        st.markdown("⏳ *Thinking...*")
        return
    
    # Get the latest update
    latest = updates[-1]
    
    # The phase_title already includes emoji from the LLM, so display it directly
    status_text = latest.phase_title
    
    # Format as markdown for better rendering
    st.markdown(status_text)


def render_thinking_expandable(updates: List[ThinkingUpdate]) -> None:
    """
    Render thinking updates in an expandable section.
    
    Args:
        updates: List of ThinkingUpdate objects
    """
    if not updates:
        return
    
    with st.expander(f"🧠 Agent Thinking Process ({len(updates)} updates)", expanded=False):
        render_thinking_session(updates)


def render_thinking_inline(updates: List[ThinkingUpdate]) -> None:
    """
    Render thinking updates inline (not in expander).
    
    Args:
        updates: List of ThinkingUpdate objects
    """
    if not updates:
        return
    
    st.markdown("### 🧠 Agent Thinking Process")
    render_thinking_session(updates)
