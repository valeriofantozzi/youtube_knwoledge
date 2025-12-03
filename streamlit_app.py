"""
Vector Database Explorer - Streamlit Web Application

A comprehensive web interface for managing document embeddings:
- Load Documents: Upload documents (SRT, text, markdown) and generate embeddings
- PostProcessing: Visualize and analyze the embedding space
- Search: Perform semantic search with filters

Run with: streamlit run streamlit_app.py
"""

import streamlit as st
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

# Import UI modules
from src.ui.state import initialize_session_state
from src.ui.theme import inject_custom_css, ICONS
from src.ui.pages import (
    render_load_documents_page,
    render_postprocessing_page,
    render_search_page,
    render_ai_search_page,
)
from src.vector_store.chroma_manager import ChromaDBManager


# Page configuration
st.set_page_config(
    page_title="Vector DB Explorer",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)


def initialize_collection() -> None:
    """Initialize ChromaDB collection on first load."""
    if not st.session_state.get("collection_initialized", False):
        with st.spinner("Loading vector database..."):
            try:
                from src.utils.config import get_config
                config = get_config()
                model_name = config.MODEL_NAME
                
                manager = ChromaDBManager()
                # Load model-specific collection (same as used during processing)
                st.session_state.collection = manager.get_or_create_collection(
                    model_name=model_name
                )
                st.session_state.total_docs = st.session_state.collection.count()
                st.session_state.model_name = model_name
                st.session_state.collection_initialized = True
            except Exception as e:
                st.error(f"Failed to initialize database: {e}")
                st.session_state.collection = None
                st.session_state.total_docs = 0


def render_sidebar() -> str:
    """
    Render the sidebar with navigation and info.
    
    Returns:
        Selected page name
    """
    st.sidebar.title(f"{ICONS['search']} Vector DB Explorer")
    st.sidebar.markdown("---")
    
    # Determine default index from query params
    default_index = 0
    try:
        # Try new API first
        params = st.query_params
        page_param = params.get("page")
    except:
        # Fallback to old API
        try:
            params = st.experimental_get_query_params()
            page_param = params.get("page", [None])[0]
        except:
            page_param = None
            
    if page_param == "ai_search":
        default_index = 3
    
    # Navigation
    page = st.sidebar.radio(
        "Navigation",
        [
            f"{ICONS['load']} Load Documents",
            f"{ICONS['analysis']} PostProcessing",
            f"{ICONS['search']} Search",
            "🤖 AI Search",
        ],
        index=default_index,
        label_visibility="collapsed",
        key="nav_radio"
    )
    
    st.sidebar.markdown("---")
    
    # Quick stats
    st.sidebar.markdown(f"**{ICONS['chart']} Quick Stats**")
    total_docs = st.session_state.get("total_docs", 0)
    st.sidebar.markdown(f"📄 Documents: **{total_docs:,}**")
    
    # Model info
    model_name = st.session_state.get("model_name")
    if model_name:
        st.sidebar.markdown(f"🤖 Model: `{model_name}`")
    
    st.sidebar.markdown("---")
    
    # Settings
    with st.sidebar.expander(f"{ICONS['settings']} Settings"):
        if st.button("🔄 Refresh Collection", use_container_width=True):
            st.session_state.collection_initialized = False
            st.rerun()
        
        if st.button("⚙️ Reload Config (.env)", use_container_width=True):
            from src.utils.config import reset_config
            reset_config()
            st.session_state.collection_initialized = False
            st.success("Configuration reloaded from .env")
            st.rerun()
        
        st.caption("Version 2.0.0")
    
    return page


def main():
    """Main application entry point."""
    # Initialize session state
    initialize_session_state()
    
    # Inject custom CSS
    inject_custom_css()
    
    # Initialize collection
    initialize_collection()
    
    # Render sidebar and get selected page
    selected_page = render_sidebar()
    
    # Route to appropriate page
    if f"{ICONS['load']} Load Documents" in selected_page:
        render_load_documents_page()
    elif f"{ICONS['analysis']} PostProcessing" in selected_page:
        render_postprocessing_page()
    elif f"{ICONS['search']} Search" in selected_page:
        render_search_page()
    elif "🤖 AI Search" in selected_page:
        render_ai_search_page()
    
    # Footer
    # st.markdown("---")
    # st.markdown(
    #     f"**{ICONS['database']} Vector Database Explorer** | "
    #     "Powered by ChromaDB & Sentence Transformers"
    # )


if __name__ == "__main__":
    main()
