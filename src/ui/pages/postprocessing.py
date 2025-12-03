"""
PostProcessing Page Module

Implements vector database visualization and statistical analysis:
- Overview tab: Database statistics and distributions
- Visualization tab: 2D/3D embedding space visualization
- Clustering tab: HDBSCAN clustering with evaluation metrics
- Export tab: Data export capabilities
"""

import streamlit as st
import pandas as pd
import numpy as np
from collections import Counter
from typing import List, Dict, Any, Optional, Tuple
import plotly.express as px
import plotly.graph_objects as go

from ..state import reset_clustering_state
from ..theme import ICONS, COLORS, CLUSTERING_PRESETS, format_number
from ..components.feedback import (
    show_empty_state,
    show_info_callout,
    show_success_summary,
)
from ..components.metric_card import metric_card, metric_row


def get_collection_data(limit: int = 1000) -> Dict[str, Any]:
    """
    Get sample data from ChromaDB collection.
    
    Args:
        limit: Maximum number of documents to retrieve
    
    Returns:
        Dictionary with collection data
    """
    if "collection" not in st.session_state or st.session_state.collection is None:
        return {"error": "No collection loaded"}
    
    try:
        sample = st.session_state.collection.get(
            limit=limit,
            include=["embeddings", "metadatas", "documents"]
        )
        return sample
    except Exception as e:
        return {"error": str(e)}


def analyze_metadata(metadatas: List[Dict]) -> Dict[str, Any]:
    """
    Analyze metadata for statistics.
    
    Args:
        metadatas: List of metadata dictionaries
    
    Returns:
        Analysis results
    """
    source_ids = []
    dates = []
    titles = []
    
    for meta in metadatas:
        if meta:
            if "source_id" in meta:
                source_ids.append(meta["source_id"])
            if "date" in meta:
                dates.append(meta["date"])
            if "title" in meta:
                titles.append(meta["title"])
    
    return {
        "source_ids": source_ids,
        "dates": dates,
        "titles": titles,
        "unique_sources": len(set(source_ids)),
        "date_counter": Counter(dates),
        "source_counter": Counter(source_ids),
    }


def render_overview_tab() -> None:
    """Render the Overview tab with database statistics."""
    total_docs = st.session_state.get("total_docs", 0)
    
    if total_docs == 0:
        show_empty_state(
            title="No Documents Loaded",
            message="Upload subtitle files to get started with analysis.",
            icon="📭",
            action_label="Go to Load Documents",
            action_page="📥 Load Documents"
        )
        return
    
    # Get sample data for analysis
    sample_size = min(1000, total_docs)
    sample = get_collection_data(limit=sample_size)
    
    if "error" in sample:
        st.error(f"Error loading data: {sample['error']}")
        return
    
    # Analyze metadata
    analysis = analyze_metadata(sample.get("metadatas", []))
    
    # Key metrics row
    st.subheader(f"{ICONS['metrics']} Key Metrics")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        metric_card("Documents", format_number(total_docs), icon="📄")
    with col2:
        metric_card("Unique Sources", analysis["unique_sources"], icon="🎬")
    with col3:
        avg_chunks = total_docs / analysis["unique_sources"] if analysis["unique_sources"] > 0 else 0
        metric_card("Avg Chunks/Source", f"{avg_chunks:.1f}", icon="📏")
    with col4:
        if sample.get("embeddings") and len(sample["embeddings"]) > 0:
            emb_dim = len(sample["embeddings"][0])
            metric_card("Embedding Dim", emb_dim, icon="🔢")
    
    st.markdown("---")
    
    # Date distribution chart
    if analysis["dates"]:
        st.subheader(f"{ICONS['calendar']} Date Distribution")
        
        date_df = pd.DataFrame([
            {"Date": date, "Count": count}
            for date, count in analysis["date_counter"].most_common(20)
        ])
        
        fig = px.bar(
            date_df,
            x="Date",
            y="Count",
            title="Chunks per Date (Top 20)",
            labels={"Count": "Number of Chunks", "Date": "Date"},
            color_discrete_sequence=[COLORS["primary"]]
        )
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)
    
    # Source distribution
    if analysis["source_ids"]:
        st.subheader(f"{ICONS['video']} Top Sources by Chunks")
        
        source_df = pd.DataFrame([
            {"Source ID": sid, "Chunks": count}
            for sid, count in analysis["source_counter"].most_common(15)
        ])
        
        fig = px.bar(
            source_df,
            y="Source ID",
            x="Chunks",
            orientation="h",
            title="Top 15 Sources by Chunk Count",
            color_discrete_sequence=[COLORS["info"]]
        )
        fig.update_layout(height=500, yaxis={"categoryorder": "total ascending"})
        st.plotly_chart(fig, use_container_width=True)


def render_visualization_tab() -> None:
    """Render the Visualization tab with 2D/3D plots."""
    total_docs = st.session_state.get("total_docs", 0)
    
    if total_docs < 10:
        st.warning(
            f"⚠️ Not enough documents for visualization. "
            f"Need at least 10 documents, but only {total_docs} found."
        )
        return
    
    # Controls
    st.subheader(f"{ICONS['settings']} Visualization Settings")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        slider_min = min(100, total_docs)
        slider_max = min(2000, total_docs)
        slider_default = min(500, total_docs)
        num_points = st.slider(
            "Number of points",
            min_value=slider_min,
            max_value=slider_max,
            value=slider_default,
            step=100
        )
    
    with col2:
        reduction_method = st.selectbox(
            "Algorithm",
            ["UMAP", "t-SNE", "PCA"],
            index=0,
            help="UMAP is faster, t-SNE gives better cluster separation"
        )
    
    with col3:
        dimensions = st.radio(
            "Dimensions",
            ["2D", "3D"],
            horizontal=True
        )
    
    color_option = st.selectbox(
        "Color by",
        ["Source ID", "Date", "Cluster", "None"],
        index=0
    )
    
    # Generate visualization button
    if st.button(f"{ICONS['chart']} Generate Visualization", type="primary", use_container_width=True):
        generate_visualization(
            num_points=num_points,
            reduction_method=reduction_method,
            dimensions=dimensions,
            color_option=color_option
        )


def generate_visualization(
    num_points: int,
    reduction_method: str,
    dimensions: str,
    color_option: str
) -> None:
    """
    Generate and display visualization.
    
    Args:
        num_points: Number of points to visualize
        reduction_method: Dimensionality reduction method
        dimensions: "2D" or "3D"
        color_option: How to color points
    """
    with st.spinner(f"Generating {dimensions} visualization with {num_points} points..."):
        try:
            # Get data
            sample = st.session_state.collection.get(
                limit=num_points,
                include=["embeddings", "metadatas", "documents"]
            )
            
            embeddings = sample.get("embeddings")
            if not embeddings or len(embeddings) == 0:
                st.error("No embeddings found in database")
                return
            
            embeddings_array = np.array(embeddings)
            metadatas = sample.get("metadatas", [{}] * len(embeddings))
            documents = sample.get("documents", [""] * len(embeddings))
            ids = sample.get("ids", [""] * len(embeddings))
            
            # Progress tracking
            progress = st.progress(0, text="Reducing dimensions...")
            
            # Dimensionality reduction
            n_components = 3 if dimensions == "3D" else 2
            
            if reduction_method == "UMAP":
                try:
                    import umap
                    reducer = umap.UMAP(
                        n_components=n_components,
                        random_state=42,
                        n_neighbors=15,
                        min_dist=0.1
                    )
                    reduced = reducer.fit_transform(embeddings_array)
                except ImportError:
                    st.error("UMAP not installed. Install with: pip install umap-learn")
                    return
            elif reduction_method == "t-SNE":
                from sklearn.manifold import TSNE
                progress.progress(0.3, text="Running t-SNE (this may take a while)...")
                reducer = TSNE(
                    n_components=n_components,
                    random_state=42,
                    perplexity=30,
                    n_iter=1000
                )
                reduced = reducer.fit_transform(embeddings_array)
            else:  # PCA
                from sklearn.decomposition import PCA
                reducer = PCA(n_components=n_components, random_state=42)
                reduced = reducer.fit_transform(embeddings_array)
            
            progress.progress(0.8, text="Preparing visualization...")
            
            # Prepare colors
            if color_option == "Source ID":
                colors = [m.get("source_id", "unknown") if m else "unknown" for m in metadatas]
            elif color_option == "Date":
                colors = [m.get("date", "unknown") if m else "unknown" for m in metadatas]
            elif color_option == "Cluster":
                colors = st.session_state.get("cluster_labels")
                if colors is None:
                    colors = ["No clusters"] * len(embeddings)
                else:
                    colors = [f"Cluster {c}" if c >= 0 else "Outlier" for c in colors[:len(embeddings)]]
            else:
                colors = None
            
            # Create hover text
            hover_texts = []
            for meta, doc in zip(metadatas, documents):
                parts = []
                if meta:
                    if "title" in meta:
                        parts.append(f"Title: {meta['title'][:50]}")
                    if "source_id" in meta:
                        parts.append(f"Source: {meta['source_id']}")
                    if "date" in meta:
                        parts.append(f"Date: {meta['date']}")
                parts.append(f"Text: {doc[:100]}...")
                hover_texts.append("<br>".join(parts))
            
            # Create plot
            if dimensions == "3D":
                fig = go.Figure(data=go.Scatter3d(
                    x=reduced[:, 0],
                    y=reduced[:, 1],
                    z=reduced[:, 2],
                    mode="markers",
                    marker=dict(
                        size=5,
                        color=[hash(c) % 256 for c in colors] if colors else "blue",
                        colorscale="Viridis" if colors else None,
                        opacity=0.7,
                        line=dict(width=0.5, color="white")
                    ),
                    text=hover_texts,
                    hovertemplate="<b>%{text}</b><extra></extra>",
                    name="Embeddings"
                ))
                
                fig.update_layout(
                    title=f"3D Embedding Space ({reduction_method}) - {num_points} points",
                    scene=dict(
                        xaxis_title="Dimension 1",
                        yaxis_title="Dimension 2",
                        zaxis_title="Dimension 3"
                    ),
                    height=700
                )
            else:
                # 2D plot
                df = pd.DataFrame({
                    "x": reduced[:, 0],
                    "y": reduced[:, 1],
                    "color": colors if colors else "All",
                    "hover": hover_texts
                })
                
                fig = px.scatter(
                    df,
                    x="x",
                    y="y",
                    color="color" if colors else None,
                    hover_data={"hover": True, "x": False, "y": False, "color": False},
                    title=f"2D Embedding Space ({reduction_method}) - {num_points} points"
                )
                fig.update_layout(height=600)
                fig.update_traces(marker=dict(size=6, opacity=0.7))
            
            progress.progress(1.0, text="✅ Visualization complete")
            st.plotly_chart(fig, use_container_width=True)
            
            # Store reduced embeddings for potential clustering
            if dimensions == "2D":
                st.session_state["reduced_embeddings_2d"] = reduced
            else:
                st.session_state["reduced_embeddings_3d"] = reduced
            
            # Statistics
            st.subheader(f"{ICONS['chart']} Visualization Statistics")
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Points visualized", num_points)
            with col2:
                st.metric("Original dimensions", embeddings_array.shape[1])
            with col3:
                st.metric("Reduced dimensions", n_components)
            
            # Download coordinates
            coords_df = pd.DataFrame({
                "ID": ids,
                "X": reduced[:, 0],
                "Y": reduced[:, 1],
            })
            if dimensions == "3D":
                coords_df["Z"] = reduced[:, 2]
            coords_df["Source_ID"] = [m.get("source_id", "") if m else "" for m in metadatas]
            coords_df["Date"] = [m.get("date", "") if m else "" for m in metadatas]
            
            st.download_button(
                label=f"{ICONS['download']} Download Coordinates (CSV)",
                data=coords_df.to_csv(index=False),
                file_name=f"embedding_{dimensions.lower()}_coordinates.csv",
                mime="text/csv"
            )
            
        except Exception as e:
            st.error(f"Error generating visualization: {e}")
            st.exception(e)


def render_clustering_tab() -> None:
    """Render the Clustering tab with HDBSCAN analysis."""
    total_docs = st.session_state.get("total_docs", 0)
    
    if total_docs < 50:
        st.warning(
            f"⚠️ Need at least 50 documents for meaningful clustering. "
            f"Currently have {total_docs} documents."
        )
        return
    
    # Clustering parameters
    st.subheader(f"{ICONS['settings']} Clustering Parameters")
    
    # Presets
    st.write("**Quick Presets:**")
    preset_cols = st.columns(3)
    for i, (name, params) in enumerate(CLUSTERING_PRESETS.items()):
        with preset_cols[i]:
            if st.button(name, help=params["description"], use_container_width=True):
                st.session_state["clustering_params"] = {
                    "min_cluster_size": params["min_cluster_size"],
                    "min_samples": params["min_samples"],
                    "metric": "cosine"
                }
                st.rerun()
    
    st.markdown("---")
    
    # Manual parameters
    col1, col2, col3 = st.columns(3)
    
    clustering_params = st.session_state.get("clustering_params", {
        "min_cluster_size": 15,
        "min_samples": 5,
        "metric": "cosine"
    })
    
    with col1:
        min_cluster_size = st.slider(
            "min_cluster_size",
            min_value=5,
            max_value=100,
            value=clustering_params.get("min_cluster_size", 15),
            help="Minimum number of points to form a cluster"
        )
    
    with col2:
        min_samples = st.slider(
            "min_samples",
            min_value=1,
            max_value=50,
            value=clustering_params.get("min_samples", 5),
            help="Number of samples in neighborhood for core point"
        )
    
    with col3:
        metric = st.selectbox(
            "Distance metric",
            ["cosine", "euclidean"],
            index=0 if clustering_params.get("metric", "cosine") == "cosine" else 1
        )
    
    # Action buttons
    col1, col2 = st.columns(2)
    with col1:
        run_clustering = st.button(
            f"{ICONS['cluster']} Run Clustering",
            type="primary",
            use_container_width=True
        )
    with col2:
        save_clustering = st.button(
            f"{ICONS['download']} Save to DB",
            use_container_width=True,
            disabled=st.session_state.get("cluster_labels") is None
        )
    
    # Run clustering
    if run_clustering:
        perform_clustering(min_cluster_size, min_samples, metric)
    
    # Display results if available
    if st.session_state.get("cluster_labels") is not None:
        render_clustering_results()


def perform_clustering(
    min_cluster_size: int,
    min_samples: int,
    metric: str
) -> None:
    """
    Perform HDBSCAN clustering.
    
    Args:
        min_cluster_size: Minimum cluster size
        min_samples: Min samples for core point
        metric: Distance metric
    """
    with st.spinner("Running HDBSCAN clustering..."):
        try:
            from ...clustering.hdbscan_clusterer import HDBSCANClusterer
            from ...clustering.cluster_evaluator import ClusterEvaluator
            
            # Get embeddings
            sample = st.session_state.collection.get(
                limit=min(5000, st.session_state.total_docs),
                include=["embeddings", "metadatas"]
            )
            
            embeddings = np.array(sample["embeddings"])
            
            # Run HDBSCAN
            clusterer = HDBSCANClusterer(
                min_cluster_size=min_cluster_size,
                min_samples=min_samples,
                metric=metric
            )
            
            labels, probabilities = clusterer.fit(embeddings)
            
            # Store results
            st.session_state["cluster_labels"] = labels
            st.session_state["cluster_probabilities"] = probabilities
            st.session_state["cluster_metadatas"] = sample["metadatas"]
            
            # Evaluate clustering
            evaluator = ClusterEvaluator()
            metrics = evaluator.evaluate(embeddings, labels, metric)
            st.session_state["cluster_metrics"] = metrics
            
            # Update params
            st.session_state["clustering_params"] = {
                "min_cluster_size": min_cluster_size,
                "min_samples": min_samples,
                "metric": metric
            }
            
            st.success(f"✅ Clustering complete! Found {metrics.n_clusters} clusters.")
            st.rerun()
            
        except Exception as e:
            st.error(f"Clustering failed: {e}")
            st.exception(e)


def render_clustering_results() -> None:
    """Render clustering results."""
    labels = st.session_state.get("cluster_labels")
    metrics = st.session_state.get("cluster_metrics")
    metadatas = st.session_state.get("cluster_metadatas", [])
    
    if labels is None:
        return
    
    st.markdown("---")
    st.subheader(f"{ICONS['success']} Clustering Results")
    
    # Metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        metric_card("Clusters", n_clusters, icon="🔬")
    with col2:
        n_outliers = int(np.sum(labels == -1))
        metric_card("Outliers", n_outliers, icon="📍")
    with col3:
        if metrics:
            metric_card("Silhouette", f"{metrics.silhouette_score:.3f}", icon="📊")
    with col4:
        if metrics:
            metric_card("Davies-Bouldin", f"{metrics.davies_bouldin_index:.3f}", icon="📈")
    
    # Cluster size distribution
    st.subheader("📊 Cluster Size Distribution")
    
    cluster_sizes = Counter(labels)
    # Remove outliers from count display
    if -1 in cluster_sizes:
        del cluster_sizes[-1]
    
    if cluster_sizes:
        size_df = pd.DataFrame([
            {"Cluster": f"Cluster {k}", "Size": v}
            for k, v in sorted(cluster_sizes.items(), key=lambda x: -x[1])
        ])
        
        fig = px.bar(
            size_df,
            x="Cluster",
            y="Size",
            title="Documents per Cluster",
            color_discrete_sequence=[COLORS["info"]]
        )
        st.plotly_chart(fig, use_container_width=True)
    
    # Cluster exploration
    with st.expander(f"{ICONS['search']} Explore Cluster Contents", expanded=False):
        cluster_options = [f"Cluster {i}" for i in sorted(set(labels)) if i >= 0]
        if cluster_options:
            selected_cluster = st.selectbox(
                "Select Cluster",
                options=cluster_options
            )
            
            cluster_idx = int(selected_cluster.split()[-1])
            cluster_mask = labels == cluster_idx
            cluster_docs_idx = np.where(cluster_mask)[0][:10]  # First 10
            
            st.write(f"**Sample documents from {selected_cluster}:**")
            for idx in cluster_docs_idx:
                if idx < len(metadatas):
                    meta = metadatas[idx]
                    title = meta.get("title", "Untitled") if meta else "Untitled"
                    st.markdown(f"- {title[:80]}...")


def render_export_tab() -> None:
    """Render the Export tab."""
    st.subheader(f"{ICONS['download']} Export Data")
    
    total_docs = st.session_state.get("total_docs", 0)
    
    if total_docs == 0:
        st.info("No data to export. Load documents first.")
        return
    
    # Source list export
    st.write("**📹 Source List**")
    if st.button("Generate Source List"):
        with st.spinner("Generating source list..."):
            all_data = st.session_state.collection.get()
            
            sources = {}
            for meta in all_data.get("metadatas", []):
                if meta and "source_id" in meta:
                    source_id = meta["source_id"]
                    if source_id not in sources:
                        sources[source_id] = {
                            "title": meta.get("title", "N/A"),
                            "date": meta.get("date", "N/A"),
                            "chunks": 0
                        }
                    sources[source_id]["chunks"] += 1
            
            source_df = pd.DataFrame([
                {"Source ID": sid, "Title": info["title"], "Date": info["date"], "Chunks": info["chunks"]}
                for sid, info in sorted(sources.items(), key=lambda x: -x[1]["chunks"])
            ])
            
            st.dataframe(source_df, use_container_width=True, height=400)
            
            st.download_button(
                label=f"{ICONS['download']} Download Source List (CSV)",
                data=source_df.to_csv(index=False),
                file_name="source_list.csv",
                mime="text/csv"
            )
    
    st.markdown("---")
    
    # Clustering results export
    if st.session_state.get("cluster_labels") is not None:
        st.write("**🔬 Clustering Results**")
        
        labels = st.session_state["cluster_labels"]
        cluster_df = pd.DataFrame({
            "Index": range(len(labels)),
            "Cluster": labels
        })
        
        st.download_button(
            label=f"{ICONS['download']} Download Cluster Labels (CSV)",
            data=cluster_df.to_csv(index=False),
            file_name="cluster_labels.csv",
            mime="text/csv"
        )


def render_postprocessing_page() -> None:
    """
    Render the PostProcessing page.
    
    This is the main entry point for the page module.
    """
    st.header(f"{ICONS['analysis']} PostProcessing")
    st.markdown("Analyze and visualize your embedding space.")
    
    # Tab navigation
    tab1, tab2, tab3, tab4 = st.tabs([
        f"{ICONS['chart']} Overview",
        "🌐 Visualization",
        f"{ICONS['cluster']} Clustering",
        f"{ICONS['download']} Export"
    ])
    
    with tab1:
        render_overview_tab()
    
    with tab2:
        render_visualization_tab()
    
    with tab3:
        render_clustering_tab()
    
    with tab4:
        render_export_tab()
