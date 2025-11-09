"""
Streamlit Web App for Vector Database Visualization

Run with: streamlit run streamlit_app.py
"""

import streamlit as st
import sys
from pathlib import Path
from collections import Counter
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.vector_store.chroma_manager import ChromaDBManager
from src.embeddings.embedder import Embedder


# Page config
st.set_page_config(
    page_title="Vector DB Viewer",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state
if 'collection' not in st.session_state:
    with st.spinner("Loading vector database..."):
        manager = ChromaDBManager()
        st.session_state.collection = manager.get_or_create_collection()
        st.session_state.total_docs = st.session_state.collection.count()

# Sidebar
st.sidebar.title("🔍 Vector DB Explorer")
st.sidebar.markdown("---")

# Main content
st.title("📊 Vector Database Visualization")
st.markdown("Explore your subtitle embeddings database interactively")

# Statistics section
st.header("📈 Statistics")

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric("Total Documents", st.session_state.total_docs)

# Get sample for analysis
sample_size = min(1000, st.session_state.total_docs)
sample = st.session_state.collection.get(limit=sample_size)

# Analyze metadata
video_ids = []
dates = []
titles = []

if sample.get('metadatas'):
    for meta in sample['metadatas']:
        if meta:
            if 'video_id' in meta:
                video_ids.append(meta['video_id'])
            if 'date' in meta:
                dates.append(meta['date'])
            if 'title' in meta:
                titles.append(meta['title'])

unique_videos = len(set(video_ids)) if video_ids else 0
avg_chunks = st.session_state.total_docs / unique_videos if unique_videos > 0 else 0

with col2:
    st.metric("Unique Videos", unique_videos)
with col3:
    st.metric("Avg Chunks/Video", f"{avg_chunks:.1f}")
with col4:
    if sample.get('embeddings') and sample['embeddings']:
        emb_dim = len(sample['embeddings'][0])
        st.metric("Embedding Dimension", emb_dim)

# Date distribution chart
if dates:
    st.subheader("📅 Date Distribution")
    date_counter = Counter(dates)
    date_df = pd.DataFrame([
        {'Date': date, 'Count': count}
        for date, count in date_counter.most_common(20)
    ])
    
    fig = px.bar(
        date_df,
        x='Date',
        y='Count',
        title="Chunks per Date (Top 20)",
        labels={'Count': 'Number of Chunks', 'Date': 'Date'}
    )
    fig.update_layout(height=400)
    st.plotly_chart(fig, use_container_width=True)

# Search section
st.header("🔍 Semantic Search")

search_query = st.text_input(
    "Enter your search query:",
    placeholder="e.g., 'how to care for orchids'",
    key="search_input"
)

col_search1, col_search2 = st.columns([3, 1])
with col_search1:
    top_k = st.slider("Number of results:", min_value=1, max_value=20, value=5)
with col_search2:
    st.write("")  # Spacing
    search_button = st.button("🔍 Search", type="primary", use_container_width=True)

if search_button and search_query:
    with st.spinner("Searching..."):
        try:
            # Generate query embedding
            embedder = Embedder()
            query_embedding = embedder.encode([search_query], is_query=True)[0]
            
            # Search in collection
            results = st.session_state.collection.query(
                query_embeddings=[query_embedding.tolist()],
                n_results=top_k,
                include=['documents', 'metadatas', 'distances']
            )
            
            if results.get('ids') and results['ids'][0]:
                st.success(f"Found {len(results['ids'][0])} results")
                
                # Display results
                for i, (doc_id, doc_text, metadata, distance) in enumerate(zip(
                    results['ids'][0],
                    results['documents'][0],
                    results['metadatas'][0],
                    results['distances'][0]
                ), 1):
                    similarity = 1 - distance
                    
                    with st.expander(
                        f"Result {i}: {metadata.get('title', 'Untitled')[:60]}... "
                        f"(Similarity: {similarity:.3f})",
                        expanded=(i == 1)
                    ):
                        col_meta1, col_meta2 = st.columns(2)
                        with col_meta1:
                            st.write(f"**Video ID:** {metadata.get('video_id', 'N/A')}")
                            st.write(f"**Date:** {metadata.get('date', 'N/A')}")
                        with col_meta2:
                            st.write(f"**Similarity Score:** {similarity:.3f}")
                            st.write(f"**Chunk Index:** {metadata.get('chunk_index', 'N/A')}")
                        
                        st.markdown("---")
                        st.write("**Text:**")
                        st.write(doc_text)
                        
                        # Show similarity bar
                        st.progress(similarity)
            else:
                st.warning("No results found")
        
        except Exception as e:
            st.error(f"Error during search: {e}")
            st.exception(e)

# Document browser section
st.header("📄 Document Browser")

# Filter options
col_filter1, col_filter2, col_filter3 = st.columns(3)

with col_filter1:
    browse_limit = st.number_input("Documents to show:", min_value=1, max_value=100, value=10)
with col_filter2:
    filter_video_id = st.text_input("Filter by Video ID (optional):", "")
with col_filter3:
    filter_date = st.text_input("Filter by Date (optional, format: YYYY/MM/DD):", "")

if st.button("🔄 Load Documents"):
    # Get documents with filters
    all_data = st.session_state.collection.get()
    
    # Apply filters
    filtered_indices = list(range(len(all_data.get('ids', []))))
    
    if filter_video_id:
        filtered_indices = [
            i for i in filtered_indices
            if all_data.get('metadatas', [{}])[i].get('video_id') == filter_video_id
        ]
    
    if filter_date:
        filtered_indices = [
            i for i in filtered_indices
            if all_data.get('metadatas', [{}])[i].get('date') == filter_date
        ]
    
    # Limit results
    filtered_indices = filtered_indices[:browse_limit]
    
    if filtered_indices:
        st.write(f"Showing {len(filtered_indices)} documents")
        
        for idx in filtered_indices:
            doc_id = all_data['ids'][idx]
            doc_text = all_data.get('documents', [''])[idx]
            metadata = all_data.get('metadatas', [{}])[idx]
            
            with st.expander(
                f"Document {idx + 1}: {metadata.get('title', 'Untitled')[:50]}...",
                expanded=False
            ):
                col_doc1, col_doc2 = st.columns(2)
                with col_doc1:
                    st.write(f"**ID:** {doc_id}")
                    st.write(f"**Video ID:** {metadata.get('video_id', 'N/A')}")
                with col_doc2:
                    st.write(f"**Date:** {metadata.get('date', 'N/A')}")
                    st.write(f"**Chunk:** {metadata.get('chunk_index', 'N/A')}")
                
                st.markdown("---")
                st.write("**Text:**")
                st.write(doc_text)
    else:
        st.warning("No documents match the filters")

# Video list section
st.header("📹 Video List")

if st.button("📋 Show Video List"):
    all_data = st.session_state.collection.get()
    
    # Group by video
    videos = {}
    if all_data.get('metadatas'):
        for meta in all_data['metadatas']:
            if meta and 'video_id' in meta:
                vid_id = meta['video_id']
                if vid_id not in videos:
                    videos[vid_id] = {
                        'title': meta.get('title', 'N/A'),
                        'date': meta.get('date', 'N/A'),
                        'chunks': 0
                    }
                videos[vid_id]['chunks'] += 1
    
    # Create DataFrame
    video_df = pd.DataFrame([
        {
            'Video ID': vid_id,
            'Title': info['title'],
            'Date': info['date'],
            'Chunks': info['chunks']
        }
        for vid_id, info in sorted(videos.items(), key=lambda x: x[1]['chunks'], reverse=True)
    ])
    
    st.dataframe(
        video_df,
        use_container_width=True,
        height=400
    )
    
    # Download button
    csv = video_df.to_csv(index=False)
    st.download_button(
        label="📥 Download as CSV",
        data=csv,
        file_name="video_list.csv",
        mime="text/csv"
    )

# Footer
st.markdown("---")
st.markdown("**Vector Database Explorer** | Powered by ChromaDB & BGE Embeddings")

