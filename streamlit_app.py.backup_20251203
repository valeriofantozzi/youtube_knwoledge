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

# Sidebar navigation
page = st.sidebar.radio(
    "Navigation",
    ["📈 Statistics", "🔍 Search", "📄 Documents", "📹 Videos", "🌐 3D View"],
    label_visibility="collapsed"
)

st.sidebar.markdown("---")
st.sidebar.markdown(f"**Total Documents:** {st.session_state.total_docs}")

# Main content
st.title("📊 Vector Database Visualization")
st.markdown("Explore your subtitle embeddings database interactively")

# Get sample for analysis (used by multiple pages)
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

# =============================================================================
# PAGE: Statistics
# =============================================================================
if page == "📈 Statistics":
    st.header("📈 Statistics")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Total Documents", st.session_state.total_docs)

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

# =============================================================================
# PAGE: Search
# =============================================================================
elif page == "🔍 Search":
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

# =============================================================================
# PAGE: Documents
# =============================================================================
elif page == "📄 Documents":
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

# =============================================================================
# PAGE: Videos
# =============================================================================
elif page == "📹 Videos":
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

# =============================================================================
# PAGE: 3D View
# =============================================================================
elif page == "🌐 3D View":
    st.header("🌐 3D Embedding Visualization")

    st.markdown("""
    Visualizza gli embeddings nello spazio 3D usando riduzione dimensionale (t-SNE/UMAP).
    Ogni punto rappresenta un chunk di testo, e i punti vicini hanno contenuti simili.
    """)

    # Check if we have enough documents for 3D visualization
    if st.session_state.total_docs < 10:
        st.warning(f"Not enough documents for 3D visualization. Need at least 10 documents, but only {st.session_state.total_docs} found.")
        generate_3d = False
        num_points_3d = 0
        reduction_method = "UMAP"
        color_option = "Video ID"
    else:
        col_3d1, col_3d2, col_3d3 = st.columns(3)

        with col_3d1:
            # Ensure min_value <= max_value by capping min_value at total_docs
            slider_min = min(100, st.session_state.total_docs)
            slider_max = min(2000, st.session_state.total_docs)
            slider_default = min(500, st.session_state.total_docs)
            num_points_3d = st.slider(
                "Number of points to visualize:",
                min_value=slider_min,
                max_value=slider_max,
                value=slider_default,
                step=100
            )
        with col_3d2:
            reduction_method = st.selectbox(
                "Reduction method:",
                ["UMAP", "t-SNE"],
                index=0
            )
        with col_3d3:
            color_option = st.selectbox(
                "Color by:",
                ["Video ID", "Date", "None"],
                index=0
            )

        st.write("")  # Spacing
        generate_3d = st.button("🎨 Generate 3D Visualization", type="primary", use_container_width=True)

    if generate_3d:
        with st.spinner(f"Generating 3D visualization with {num_points_3d} points..."):
            try:
                # Get sample embeddings
                # Note: 'ids' are always returned, don't include them in include parameter
                sample_data = st.session_state.collection.get(
                    limit=num_points_3d,
                    include=['embeddings', 'metadatas', 'documents']
                )
                
                # Check if embeddings exist and are not empty
                embeddings_list = sample_data.get('embeddings')
                # Handle both list and numpy array cases
                if embeddings_list is None:
                    st.error("No embeddings found in database")
                elif hasattr(embeddings_list, '__len__'):
                    if len(embeddings_list) == 0:
                        st.error("No embeddings found in database")
                    else:
                        embeddings = sample_data['embeddings']
                else:
                    st.error("Invalid embeddings format")
                    embeddings = None
                
                if embeddings is not None:
                    metadatas = sample_data.get('metadatas', [{}] * len(embeddings))
                    documents = sample_data.get('documents', [''] * len(embeddings))
                    ids = sample_data.get('ids', [''] * len(embeddings))
                    
                    import numpy as np
                    embeddings_array = np.array(embeddings)
                    
                    # Progress bar
                    progress_bar = st.progress(0)
                    status_text = st.empty()
                    
                    # Dimensionality reduction
                    status_text.text("Reducing dimensions...")
                    progress_bar.progress(0.3)
                    
                    if reduction_method == "UMAP":
                        try:
                            import umap
                            reducer = umap.UMAP(n_components=3, random_state=42, n_neighbors=15, min_dist=0.1)
                            reduced_embeddings = reducer.fit_transform(embeddings_array)
                        except ImportError:
                            st.error("UMAP not installed. Install with: pip install umap-learn")
                            st.stop()
                    else:  # t-SNE
                        from sklearn.manifold import TSNE
                        status_text.text("Running t-SNE (this may take a while)...")
                        progress_bar.progress(0.5)
                        reducer = TSNE(n_components=3, random_state=42, perplexity=30, n_iter=1000)
                        reduced_embeddings = reducer.fit_transform(embeddings_array)
                    
                    progress_bar.progress(0.8)
                    status_text.text("Preparing visualization...")
                    
                    # Prepare data for visualization
                    x_coords = reduced_embeddings[:, 0]
                    y_coords = reduced_embeddings[:, 1]
                    z_coords = reduced_embeddings[:, 2]
                    
                    # Create hover text
                    hover_texts = []
                    for i, (meta, doc, doc_id) in enumerate(zip(metadatas, documents, ids)):
                        hover_parts = []
                        if meta:
                            if 'title' in meta:
                                hover_parts.append(f"Title: {meta['title'][:50]}")
                            if 'video_id' in meta:
                                hover_parts.append(f"Video: {meta['video_id']}")
                            if 'date' in meta:
                                hover_parts.append(f"Date: {meta['date']}")
                        hover_parts.append(f"Text: {doc[:100]}...")
                        hover_texts.append("<br>".join(hover_parts))
                    
                    # Color by video_id or date (using the option selected above)
                    if color_option == "Video ID":
                        colors = [meta.get('video_id', 'unknown') if meta else 'unknown' for meta in metadatas]
                        color_title = "Video ID"
                    elif color_option == "Date":
                        colors = [meta.get('date', 'unknown') if meta else 'unknown' for meta in metadatas]
                        color_title = "Date"
                    else:
                        colors = None
                        color_title = None
                    
                    # Create color mapping for categorical data
                    if colors and color_option != "None":
                        # Convert categorical to numeric for colorscale
                        unique_colors = list(set(colors))
                        color_map = {color: i for i, color in enumerate(unique_colors)}
                        numeric_colors = [color_map[c] for c in colors]
                        colorscale = 'Viridis'
                        colorbar_title = color_option
                    else:
                        numeric_colors = None
                        colorscale = None
                        colorbar_title = None
                    
                    # Create 3D scatter plot
                    fig = go.Figure(data=go.Scatter3d(
                        x=x_coords,
                        y=y_coords,
                        z=z_coords,
                        mode='markers',
                        marker=dict(
                            size=5,
                            color=numeric_colors if numeric_colors is not None else 'blue',
                            colorscale=colorscale if colorscale else None,
                            opacity=0.7,
                            line=dict(width=0.5, color='white'),
                            colorbar=dict(title=colorbar_title) if colorbar_title else None,
                            showscale=(colorbar_title is not None)
                        ),
                        text=hover_texts,
                        hovertemplate='<b>%{text}</b><extra></extra>',
                        name='Embeddings'
                    ))
                    
                    fig.update_layout(
                        title=f"3D Embedding Space ({reduction_method}) - {num_points_3d} points",
                        scene=dict(
                            xaxis_title="Dimension 1",
                            yaxis_title="Dimension 2",
                            zaxis_title="Dimension 3",
                            bgcolor='rgba(0,0,0,0)',
                            xaxis=dict(backgroundcolor='rgba(0,0,0,0)'),
                            yaxis=dict(backgroundcolor='rgba(0,0,0,0)'),
                            zaxis=dict(backgroundcolor='rgba(0,0,0,0)'),
                        ),
                        width=1000,
                        height=800,
                        margin=dict(l=0, r=0, b=0, t=50)
                    )
                    
                    progress_bar.progress(1.0)
                    status_text.text("Visualization ready!")
                    
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Statistics
                    st.subheader("📊 3D Visualization Statistics")
                    col_stat1, col_stat2, col_stat3 = st.columns(3)
                    with col_stat1:
                        st.metric("Points visualized", num_points_3d)
                    with col_stat2:
                        st.metric("Original dimensions", embeddings_array.shape[1])
                    with col_stat3:
                        st.metric("Reduced dimensions", 3)
                    
                    # Download coordinates
                    coords_df = pd.DataFrame({
                        'ID': ids,
                        'X': x_coords,
                        'Y': y_coords,
                        'Z': z_coords,
                        'Video ID': [m.get('video_id', '') if m else '' for m in metadatas],
                        'Title': [m.get('title', '') if m else '' for m in metadatas],
                        'Date': [m.get('date', '') if m else '' for m in metadatas]
                    })
                    
                    csv_coords = coords_df.to_csv(index=False)
                    st.download_button(
                        label="📥 Download 3D Coordinates (CSV)",
                        data=csv_coords,
                        file_name="embedding_3d_coordinates.csv",
                        mime="text/csv"
                    )
            
            except Exception as e:
                st.error(f"Error generating 3D visualization: {e}")
                st.exception(e)

# Footer
st.markdown("---")
st.markdown("**Vector Database Explorer** | Powered by ChromaDB & BGE Embeddings")