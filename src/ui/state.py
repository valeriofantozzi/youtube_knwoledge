"""
Session State Management Module

Provides centralized session state initialization and management for the
Streamlit application.
"""

import streamlit as st
from typing import Any, Optional, Dict
from dataclasses import dataclass
from enum import Enum


class ProcessingStatus(Enum):
    """Processing status states."""
    IDLE = "idle"
    PREPROCESSING = "preprocessing"
    EMBEDDING = "embedding"
    INDEXING = "indexing"
    COMPLETE = "complete"
    ERROR = "error"


@dataclass
class ProcessingStats:
    """Statistics for processing operations."""
    files_total: int = 0
    files_processed: int = 0
    chunks_total: int = 0
    chunks_processed: int = 0
    embeddings_generated: int = 0
    documents_indexed: int = 0
    errors: int = 0
    processing_time_seconds: float = 0.0


# Default session state schema
SESSION_STATE_DEFAULTS: Dict[str, Any] = {
    # Collection management
    "collection": None,
    "total_docs": 0,
    "model_name": None,
    "collection_initialized": False,
    
    # Processing state
    "processing_status": ProcessingStatus.IDLE.value,
    "processing_progress": 0.0,
    "processing_stats": None,
    "processing_errors": [],
    "processing_current_phase": "",
    "processing_current_file": "",
    
    # Clustering state
    "cluster_labels": None,
    "cluster_metrics": None,
    "reduced_embeddings_2d": None,
    "reduced_embeddings_3d": None,
    "clustering_params": {
        "min_cluster_size": 15,
        "min_samples": 5,
        "metric": "cosine",
    },
    
    # Search state
    "last_search_query": "",
    "last_search_results": [],
    "search_filters": {},
    
    # Navigation state
    "current_page": "📥 Load Documents",
    
    # UI state
    "sidebar_expanded": True,
    "show_advanced_options": False,
}


def initialize_session_state() -> None:
    """
    Initialize all session state variables with defaults.
    
    This function should be called at the beginning of the app to ensure
    all required state variables exist.
    """
    for key, default_value in SESSION_STATE_DEFAULTS.items():
        if key not in st.session_state:
            st.session_state[key] = default_value


def get_session_state(key: str, default: Any = None) -> Any:
    """
    Get a session state value with optional default.
    
    Args:
        key: Session state key
        default: Default value if key not found
    
    Returns:
        Session state value or default
    """
    return st.session_state.get(key, default)


def set_session_state(key: str, value: Any) -> None:
    """
    Set a session state value.
    
    Args:
        key: Session state key
        value: Value to set
    """
    st.session_state[key] = value


def reset_processing_state() -> None:
    """Reset all processing-related state to defaults."""
    processing_keys = [
        "processing_status",
        "processing_progress",
        "processing_stats",
        "processing_errors",
        "processing_current_phase",
        "processing_current_file",
    ]
    for key in processing_keys:
        st.session_state[key] = SESSION_STATE_DEFAULTS[key]


def reset_clustering_state() -> None:
    """Reset all clustering-related state to defaults."""
    clustering_keys = [
        "cluster_labels",
        "cluster_metrics",
        "reduced_embeddings_2d",
        "reduced_embeddings_3d",
    ]
    for key in clustering_keys:
        st.session_state[key] = SESSION_STATE_DEFAULTS[key]


def reset_search_state() -> None:
    """Reset all search-related state to defaults."""
    search_keys = [
        "last_search_query",
        "last_search_results",
        "search_filters",
    ]
    for key in search_keys:
        st.session_state[key] = SESSION_STATE_DEFAULTS[key]


def update_processing_progress(
    phase: str,
    progress: float,
    current_file: str = "",
    stats: Optional[ProcessingStats] = None
) -> None:
    """
    Update processing progress state.
    
    Args:
        phase: Current processing phase name
        progress: Progress value (0.0 to 1.0)
        current_file: Currently processing file
        stats: Optional processing statistics
    """
    st.session_state["processing_current_phase"] = phase
    st.session_state["processing_progress"] = progress
    st.session_state["processing_current_file"] = current_file
    if stats:
        st.session_state["processing_stats"] = stats


def add_processing_error(error: str) -> None:
    """
    Add an error to the processing errors list.
    
    Args:
        error: Error message to add
    """
    if "processing_errors" not in st.session_state:
        st.session_state["processing_errors"] = []
    st.session_state["processing_errors"].append(error)


def get_collection_info() -> Dict[str, Any]:
    """
    Get current collection information.
    
    Returns:
        Dictionary with collection info
    """
    return {
        "collection": st.session_state.get("collection"),
        "total_docs": st.session_state.get("total_docs", 0),
        "model_name": st.session_state.get("model_name"),
        "initialized": st.session_state.get("collection_initialized", False),
    }
