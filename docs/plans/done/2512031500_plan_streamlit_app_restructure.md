# Implementation Plan: Streamlit App Restructure

**Plan ID:** `2512031500_plan_streamlit_app_restructure`  
**Created:** 2025-12-03  
**Status:** ✅ Implemented (Phase 1-5 Complete)
**Last Updated:** 2025-12-03

---

## Technical & Engineering Description

### Overview

**What is being built:**  
Complete restructuring of the Streamlit web application (`streamlit_app.py`) into three main functional sections:

1. **Load Documents** – Upload/select subtitle files (SRT) and trigger the complete processing pipeline (preprocessing → embedding generation → vector store indexing)
2. **PostProcessing** – Visualize the vector database, apply statistical analysis (clustering, dimensionality reduction, metrics visualization)
3. **Search** – Perform semantic search with filters and result visualization

**Business/Functional Goals:**

- Provide a unified web interface for the entire subtitle embedding workflow
- Enable non-technical users to load, analyze, and search subtitle embeddings
- Consolidate existing CLI functionality into an intuitive GUI
- Leverage existing backend modules without reimplementation

---

### Architecture

**High-Level Design:**

```
┌─────────────────────────────────────────────────────────────────────┐
│                        Streamlit Web App                            │
├─────────────────────────────────────────────────────────────────────┤
│  ┌───────────────┐  ┌──────────────────┐  ┌───────────────────┐    │
│  │ Load Documents│  │  PostProcessing  │  │     Search        │    │
│  │   Section     │  │     Section      │  │    Section        │    │
│  └───────┬───────┘  └────────┬─────────┘  └─────────┬─────────┘    │
│          │                   │                      │               │
├──────────┼───────────────────┼──────────────────────┼───────────────┤
│          │         Session State Management         │               │
├──────────┼───────────────────┼──────────────────────┼───────────────┤
│          ▼                   ▼                      ▼               │
│  ┌──────────────────────────────────────────────────────────────┐  │
│  │                    Backend Modules (src/)                     │  │
│  │  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────┐   │  │
│  │  │preprocessing│  │ embeddings  │  │   vector_store      │   │  │
│  │  │  pipeline   │  │  pipeline   │  │   pipeline          │   │  │
│  │  └─────────────┘  └─────────────┘  └─────────────────────┘   │  │
│  │  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────┐   │  │
│  │  │  clustering │  │  retrieval  │  │   utils/config      │   │  │
│  │  │   module    │  │query_engine │  │                     │   │  │
│  │  └─────────────┘  └─────────────┘  └─────────────────────┘   │  │
│  └──────────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────────┘
                                  │
                                  ▼
                    ┌─────────────────────────┐
                    │   ChromaDB (persistent) │
                    │   data/vector_db/       │
                    └─────────────────────────┘
```

**Design Patterns Used:**

- **Session State Pattern**: Streamlit session state for managing app state across reruns
- **Pipeline Pattern**: Reuse existing preprocessing, embedding, and vector store pipelines
- **Strategy Pattern**: Clustering algorithms via `Clusterer` interface
- **Adapter Pattern**: Model-specific embedding adapters (already implemented)

**Key Architectural Decisions:**

1. **Reuse existing pipelines** – No reimplementation; call `PreprocessingPipeline`, `EmbeddingPipeline`, `VectorStorePipeline` directly
2. **Background processing** – Long-running tasks (embedding generation) should show progress and not block UI
3. **Modular page structure** – Each section as a separate function for maintainability
4. **Lazy loading** – Load expensive resources (models, collections) only when needed

---

### Components & Modules

#### 1. Load Documents Section (`render_load_documents_page()`)

**Responsibility:**

- File upload (single/multiple SRT files) or directory selection
- Model selection (dropdown from model registry)
- Processing options (batch size, skip duplicates, parallel processing)
- Progress visualization during processing
- Processing statistics display

**Interactions:**

- `PreprocessingPipeline.process_file()` / `process_multiple_files()`
- `EmbeddingPipeline.generate_embeddings_with_checkpointing()`
- `VectorStorePipeline.index_processed_video()`
- Session state for tracking progress and results

#### 2. PostProcessing Section (`render_postprocessing_page()`)

**Responsibility:**

- Vector database statistics and visualization
- Dimensionality reduction (UMAP, t-SNE, PCA) for 2D/3D visualization
- Clustering analysis (HDBSCAN with configurable parameters)
- Cluster evaluation metrics (Silhouette, Davies-Bouldin, Calinski-Harabasz)
- Export capabilities (CSV, JSON)

**Interactions:**

- `ChromaDBManager.get_or_create_collection()` for data access
- `HDBSCANClusterer` for clustering
- `ClusterEvaluator` for metrics
- `ClusterManager` for storing/retrieving cluster labels
- `umap`, `sklearn.manifold.TSNE`, `sklearn.decomposition.PCA` for dimensionality reduction

#### 3. Search Section (`render_search_page()`)

**Responsibility:**

- Natural language query input
- Search filters (video ID, date range, title keywords)
- Results display with similarity scores
- Result grouping by video
- Context expansion (show surrounding chunks)

**Interactions:**

- `QueryEngine.query()` for semantic search
- `SimilaritySearch` for low-level search operations
- `SearchFilters` for filter construction
- `Embedder` for query embedding generation

---

### Technology Stack

**Existing Dependencies (no changes):**

- `streamlit` – Web framework
- `plotly` – Interactive visualizations
- `pandas` – Data manipulation
- `numpy` – Numerical operations
- `chromadb` – Vector database
- `umap-learn` – UMAP dimensionality reduction
- `scikit-learn` – t-SNE, PCA, clustering metrics
- `hdbscan` – Clustering algorithm

**No new dependencies required** – All functionality is covered by existing packages.

---

### Integration Points

| Component      | Integrates With         | Data Flow                            |
| -------------- | ----------------------- | ------------------------------------ |
| Load Documents | `PreprocessingPipeline` | SRT files → `ProcessedVideo` objects |
| Load Documents | `EmbeddingPipeline`     | `ProcessedVideo` → embeddings array  |
| Load Documents | `VectorStorePipeline`   | embeddings → ChromaDB                |
| PostProcessing | `ChromaDBManager`       | Read embeddings + metadata           |
| PostProcessing | `HDBSCANClusterer`      | embeddings → cluster labels          |
| PostProcessing | `ClusterManager`        | Store/retrieve cluster labels        |
| PostProcessing | `ClusterEvaluator`      | labels → quality metrics             |
| Search         | `QueryEngine`           | query text → `SearchResult` list     |
| Search         | `Embedder`              | query text → query embedding         |

---

### Data Models & Schemas

**Existing schemas to reuse:**

- `ProcessedVideo` (dataclass) – preprocessing output
- `Chunk` (dataclass) – text chunk with metadata
- `ChunkMetadata` (dataclass) – metadata schema for ChromaDB
- `SearchResult` (dataclass) – search result with score
- `SearchFilters` (dataclass) – filter parameters
- `ClusterMetrics` (dataclass) – clustering quality metrics
- `ClusterMetadata` (dataclass) – cluster information

**Session State Schema:**

```python
st.session_state = {
    # Collection management
    'collection': ChromaDB collection object,
    'total_docs': int,
    'model_name': str,

    # Processing state
    'processing_status': 'idle' | 'preprocessing' | 'embedding' | 'indexing' | 'complete' | 'error',
    'processing_progress': float (0.0 - 1.0),
    'processing_stats': dict,
    'processing_errors': list[str],

    # Clustering state
    'cluster_labels': np.ndarray | None,
    'cluster_metrics': ClusterMetrics | None,
    'reduced_embeddings_2d': np.ndarray | None,
    'reduced_embeddings_3d': np.ndarray | None,

    # Search state
    'last_search_query': str,
    'last_search_results': list[SearchResult],
}
```

---

### Security Considerations

- **File upload validation**: Only accept `.srt` files, limit file size
- **Path traversal prevention**: Sanitize uploaded filenames
- **No sensitive data exposure**: ChromaDB is local; no external API keys required for core functionality
- **Resource limits**: Limit number of files per upload, max embedding batch size

---

### Performance Requirements

| Metric                   | Target                   | Mitigation                     |
| ------------------------ | ------------------------ | ------------------------------ |
| File upload              | < 10MB per file          | Streamlit default limits       |
| Preprocessing            | ~100 files/min           | Parallel processing (existing) |
| Embedding generation     | ~50 chunks/sec (GPU)     | Batch processing, progress bar |
| Clustering (1000 points) | < 5 sec                  | HDBSCAN is efficient           |
| Search latency           | < 500ms                  | Cached query embeddings        |
| 3D visualization         | < 10 sec for 2000 points | UMAP is faster than t-SNE      |

**Caching Strategy:**

- Cache collection reference in session state
- Cache dimensionality reduction results
- QueryEngine already has query caching

---

### Other Relevant Specs

**Observability:**

- Streamlit `st.spinner()` and `st.progress()` for user feedback
- Logging to existing `logs/app.log` via `get_default_logger()`
- Error display via `st.error()` and `st.exception()`

**Backward Compatibility:**

- Existing CLI scripts (`scripts/process_subtitles.py`, `scripts/query_subtitles.py`) remain unchanged
- Existing model-specific collections in ChromaDB are preserved
- No database migrations required

---

## UI/UX Design Specifications

### Design Principles

1. **Progressive Disclosure** – Show essential controls first; advanced options in collapsible sections
2. **Immediate Feedback** – Every action must provide visual feedback within 100ms
3. **Error Prevention** – Validate inputs before submission; disable invalid actions
4. **Recognition over Recall** – Use visual cues, icons, and contextual help
5. **Consistency** – Unified visual language across all sections

---

### Visual Design System

#### Color Palette (Streamlit-compatible)

```python
COLORS = {
    # Primary actions
    "primary": "#FF4B4B",        # Streamlit red - main CTAs
    "primary_hover": "#FF6B6B",

    # Status colors
    "success": "#28A745",        # Green - completed, positive
    "warning": "#FFC107",        # Yellow - caution, in-progress
    "error": "#DC3545",          # Red - errors, failures
    "info": "#17A2B8",           # Cyan - informational

    # Semantic colors for data visualization
    "cluster_palette": [
        "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728",
        "#9467bd", "#8c564b", "#e377c2", "#7f7f7f"
    ],

    # Similarity score gradient
    "score_high": "#28A745",     # > 0.8
    "score_medium": "#FFC107",   # 0.5 - 0.8
    "score_low": "#DC3545",      # < 0.5
}
```

#### Typography Hierarchy

| Element        | Streamlit Component            | Usage                      |
| -------------- | ------------------------------ | -------------------------- |
| Page Title     | `st.title()`                   | Main section headers       |
| Section Header | `st.header()`                  | Major content divisions    |
| Subsection     | `st.subheader()`               | Feature groups             |
| Body Text      | `st.write()` / `st.markdown()` | Descriptions, help text    |
| Caption        | `st.caption()`                 | Secondary info, timestamps |
| Code/Data      | `st.code()` / `st.dataframe()` | Technical details          |

#### Iconography (Emoji-based for Streamlit)

| Category   | Icons                                                       |
| ---------- | ----------------------------------------------------------- |
| Navigation | 📥 Load, 🔬 Analysis, 🔍 Search                             |
| Status     | ✅ Success, ⚠️ Warning, ❌ Error, ⏳ Loading, 🔄 Processing |
| Actions    | ▶️ Run, ⬇️ Download, 🗑️ Delete, ➕ Add, ✏️ Edit             |
| Data       | 📊 Chart, 📈 Metrics, 📁 Files, 🎬 Video, 📄 Document       |
| Clustering | 🔬 Cluster, 🎯 Centroid, 📍 Outlier                         |

---

### Component Design Specifications

#### 1. Global Layout

```
┌──────────────────────────────────────────────────────────────────┐
│  SIDEBAR (280px)              │  MAIN CONTENT AREA               │
├───────────────────────────────┼──────────────────────────────────┤
│  ┌─────────────────────────┐  │  ┌────────────────────────────┐  │
│  │ 🔍 Vector DB Explorer   │  │  │ Page Title                 │  │
│  │ ═══════════════════════ │  │  │ Brief description          │  │
│  └─────────────────────────┘  │  └────────────────────────────┘  │
│                               │                                  │
│  ┌─────────────────────────┐  │  ┌────────────────────────────┐  │
│  │ Navigation              │  │  │ PRIMARY CONTENT            │  │
│  │ ○ 📥 Load Documents     │  │  │                            │  │
│  │ ● 🔬 PostProcessing     │  │  │ Cards, forms, charts...    │  │
│  │ ○ 🔍 Search             │  │  │                            │  │
│  └─────────────────────────┘  │  └────────────────────────────┘  │
│                               │                                  │
│  ┌─────────────────────────┐  │  ┌────────────────────────────┐  │
│  │ 📊 Quick Stats          │  │  │ SECONDARY CONTENT          │  │
│  │ Documents: 1,234        │  │  │                            │  │
│  │ Videos: 56              │  │  │ Details, exports...        │  │
│  │ Model: BGE-large        │  │  └────────────────────────────┘  │
│  └─────────────────────────┘  │                                  │
│                               │                                  │
│  ┌─────────────────────────┐  │                                  │
│  │ ⚙️ Settings (expander)  │  │                                  │
│  └─────────────────────────┘  │                                  │
└───────────────────────────────┴──────────────────────────────────┘
```

#### 2. Load Documents Section - UI Wireframe

```
┌────────────────────────────────────────────────────────────────────┐
│  📥 Load Documents                                                  │
│  Import subtitle files and generate embeddings                      │
├────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  ┌─── Input Method ─────────────────────────────────────────────┐  │
│  │  [📁 Upload Files]  [📂 Select Directory]                    │  │
│  │  ════════════════════════════════════════                    │  │
│  │                                                               │  │
│  │  ┌─────────────────────────────────────────────────────────┐ │  │
│  │  │  🔼 Drag & drop SRT files here                          │ │  │
│  │  │     or click to browse                                   │ │  │
│  │  │                                                          │ │  │
│  │  │  Accepted: .srt files up to 10MB each                   │ │  │
│  │  └─────────────────────────────────────────────────────────┘ │  │
│  │                                                               │  │
│  │  📄 3 files selected (2.4 MB total)                          │  │
│  │  ├── video1_subtitles.srt (800 KB)                          │  │
│  │  ├── video2_subtitles.srt (900 KB)                          │  │
│  │  └── video3_subtitles.srt (700 KB)                          │  │
│  └───────────────────────────────────────────────────────────────┘  │
│                                                                     │
│  ┌─── Model Selection ──────────────────────────────────────────┐  │
│  │  🤖 Embedding Model                                          │  │
│  │  ┌─────────────────────────────────────────────────────────┐ │  │
│  │  │ BAAI/bge-large-en-v1.5                              ▼ │ │  │
│  │  └─────────────────────────────────────────────────────────┘ │  │
│  │  ℹ️ 1024 dimensions • Best quality • ~2GB VRAM              │  │
│  └───────────────────────────────────────────────────────────────┘  │
│                                                                     │
│  ▶ Advanced Options                                                │
│  ┌───────────────────────────────────────────────────────────────┐  │
│  │  Batch Size: [====○====] 64                                  │  │
│  │  ☑ Skip already indexed videos                               │  │
│  │  ☑ Enable parallel preprocessing                             │  │
│  └───────────────────────────────────────────────────────────────┘  │
│                                                                     │
│  ┌─────────────────────────────────────────────────────────────────┐│
│  │  [▶️ Start Processing]                                         ││
│  └─────────────────────────────────────────────────────────────────┘│
│                                                                     │
│  ┌─── Processing Status ────────────────────────────────────────┐  │
│  │  ⏳ Phase 2/3: Generating Embeddings                         │  │
│  │  ████████████████░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░  45%        │  │
│  │                                                               │  │
│  │  📊 Progress Details                                          │  │
│  │  ├── Files: 2/3 complete                                     │  │
│  │  ├── Chunks: 156/350 processed                               │  │
│  │  ├── Speed: 48 chunks/sec                                    │  │
│  │  └── ETA: ~4 seconds                                         │  │
│  └───────────────────────────────────────────────────────────────┘  │
│                                                                     │
└────────────────────────────────────────────────────────────────────┘
```

#### 3. PostProcessing Section - UI Wireframe

```
┌────────────────────────────────────────────────────────────────────┐
│  🔬 PostProcessing                                                  │
│  Analyze and visualize your embedding space                         │
├────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │ [📊 Overview] [🌐 Visualization] [🔬 Clustering] [📈 Export] │   │
│  └─────────────────────────────────────────────────────────────┘   │
│                                                                     │
│  ═══════════════════════════════════════════════════════════════   │
│  📊 OVERVIEW TAB                                                    │
│  ═══════════════════════════════════════════════════════════════   │
│                                                                     │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐           │
│  │📄 1,234  │  │🎬 56     │  │📏 22.0   │  │🔢 1024   │           │
│  │Documents │  │Videos    │  │Avg/Video │  │Dimensions│           │
│  └──────────┘  └──────────┘  └──────────┘  └──────────┘           │
│                                                                     │
│  ┌─────────────────────────────┐  ┌────────────────────────────┐   │
│  │ 📅 Documents by Date        │  │ 🎬 Top Videos by Chunks    │   │
│  │ ▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓ │  │ video_abc123  ████████ 45  │   │
│  │ ▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓    │  │ video_def456  ██████ 38    │   │
│  │ ▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓  │  │ video_ghi789  █████ 32     │   │
│  │ Jan  Feb  Mar  Apr  May    │  │ ...                         │   │
│  └─────────────────────────────┘  └────────────────────────────┘   │
│                                                                     │
│  ═══════════════════════════════════════════════════════════════   │
│  🌐 VISUALIZATION TAB                                               │
│  ═══════════════════════════════════════════════════════════════   │
│                                                                     │
│  ┌─── Controls ─────────────────────────────────────────────────┐  │
│  │  Algorithm: [UMAP ▼]  Points: [===○===] 500  Dim: [2D][3D]  │  │
│  │  Color by: [● Cluster  ○ Video  ○ Date  ○ None]              │  │
│  │  [🎨 Generate Visualization]                                  │  │
│  └───────────────────────────────────────────────────────────────┘  │
│                                                                     │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │                                                              │   │
│  │                    ●  ●●                                     │   │
│  │               ●● ●●●●●● ●●                                   │   │
│  │            ●●●●●●●●●●●●●●●●●                                 │   │
│  │         ●●●●●●●●●●●●●●●●●●●●●●     ○○                        │   │
│  │        ●●●●●●●●●●●●●●●●●●●●●●●   ○○○○○                       │   │
│  │         ●●●●●●●●●●●●●●●●●●●●●  ○○○○○○○○                      │   │
│  │            ●●●●●●●●●●●●●●●●  ○○○○○○○○○○○                     │   │
│  │               ●●●●●●●●●●      ○○○○○○○                        │   │
│  │                  ●●●●           ○○○                          │   │
│  │                                                              │   │
│  │  [Hover for details]         Legend: ● Cluster 1  ○ Cluster 2│   │
│  └─────────────────────────────────────────────────────────────┘   │
│                                                                     │
│  ═══════════════════════════════════════════════════════════════   │
│  🔬 CLUSTERING TAB                                                  │
│  ═══════════════════════════════════════════════════════════════   │
│                                                                     │
│  ┌─── Parameters ───────────────────────────────────────────────┐  │
│  │  min_cluster_size: [===○=====] 15                            │  │
│  │  min_samples:      [==○======] 5                             │  │
│  │  metric:           [cosine ▼]                                │  │
│  │                                                               │  │
│  │  [🔬 Run Clustering]  [💾 Save to DB]                        │  │
│  └───────────────────────────────────────────────────────────────┘  │
│                                                                     │
│  ┌─── Results ──────────────────────────────────────────────────┐  │
│  │  ✅ Clustering Complete                                       │  │
│  │                                                               │  │
│  │  ┌────────┐  ┌────────┐  ┌────────┐                          │  │
│  │  │🔬 8    │  │📍 45   │  │📊 0.72 │                          │  │
│  │  │Clusters│  │Outliers│  │Silhouet│                          │  │
│  │  └────────┘  └────────┘  └────────┘                          │  │
│  │                                                               │  │
│  │  📊 Cluster Size Distribution                                 │  │
│  │  Cluster 0: ████████████████████ 234                         │  │
│  │  Cluster 1: ██████████████ 178                               │  │
│  │  Cluster 2: ███████████ 145                                  │  │
│  │  ...                                                          │  │
│  │                                                               │  │
│  │  ▶ Explore Cluster Contents                                   │  │
│  │  ┌─────────────────────────────────────────────────────────┐ │  │
│  │  │ Select Cluster: [Cluster 0 (234 docs) ▼]                │ │  │
│  │  │                                                          │ │  │
│  │  │ Sample Documents:                                        │ │  │
│  │  │ • "In this tutorial we'll learn about orchid care..."   │ │  │
│  │  │ • "The best time to water your orchids is when..."      │ │  │
│  │  │ • "Repotting orchids should be done every..."           │ │  │
│  │  └─────────────────────────────────────────────────────────┘ │  │
│  └───────────────────────────────────────────────────────────────┘  │
│                                                                     │
└────────────────────────────────────────────────────────────────────┘
```

#### 4. Search Section - UI Wireframe

```
┌────────────────────────────────────────────────────────────────────┐
│  🔍 Semantic Search                                                 │
│  Find relevant content using natural language                       │
├────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  ┌─── Search Input ─────────────────────────────────────────────┐  │
│  │  ┌─────────────────────────────────────────────────────────┐ │  │
│  │  │ How to care for orchids in winter?                    🔍│ │  │
│  │  └─────────────────────────────────────────────────────────┘ │  │
│  │  💡 Tip: Use natural questions for best results              │  │
│  └───────────────────────────────────────────────────────────────┘  │
│                                                                     │
│  ┌─── Search Options ───────────────────────────────────────────┐  │
│  │  Results: [===○=====] 10    Min Score: [======○==] 0.5      │  │
│  └───────────────────────────────────────────────────────────────┘  │
│                                                                     │
│  ▶ Advanced Filters                                                │
│  ┌───────────────────────────────────────────────────────────────┐  │
│  │  Video: [All Videos ▼]                                       │  │
│  │  Date Range: [Start ▼] to [End ▼]                            │  │
│  │  Keywords: [                                               ]  │  │
│  └───────────────────────────────────────────────────────────────┘  │
│                                                                     │
│  ═══════════════════════════════════════════════════════════════   │
│  📋 Results (8 found in 0.3s)                    [⬇️ Export]       │
│  ═══════════════════════════════════════════════════════════════   │
│                                                                     │
│  ┌─── Result 1 ─────────────────────────────────────────────────┐  │
│  │  ████████████████████████████░░░░░░  92.3% match             │  │
│  │                                                               │  │
│  │  🎬 How to Care for Orchids - Complete Guide                  │  │
│  │  📅 2024/01/15  •  Chunk 12/45                               │  │
│  │                                                               │  │
│  │  "During **winter** months, **orchid** **care** requires     │  │
│  │  special attention. Reduce watering frequency and ensure     │  │
│  │  adequate humidity levels..."                                 │  │
│  │                                                               │  │
│  │  [📖 Show Full Text]  [🔗 Show Context]  [▶️ Go to Video]    │  │
│  └───────────────────────────────────────────────────────────────┘  │
│                                                                     │
│  ┌─── Result 2 ─────────────────────────────────────────────────┐  │
│  │  ████████████████████████░░░░░░░░░░  78.5% match             │  │
│  │                                                               │  │
│  │  🎬 Winter Plant Care Tips                                    │  │
│  │  📅 2024/02/20  •  Chunk 8/32                                │  │
│  │                                                               │  │
│  │  "When temperatures drop, your **orchids** need extra        │  │
│  │  protection from cold drafts. Position them away from..."    │  │
│  │                                                               │  │
│  │  [📖 Show Full Text]  [🔗 Show Context]  [▶️ Go to Video]    │  │
│  └───────────────────────────────────────────────────────────────┘  │
│                                                                     │
│  ... more results ...                                               │
│                                                                     │
└────────────────────────────────────────────────────────────────────┘
```

---

### Interaction Design Patterns

#### 1. Loading States

```python
# Pattern: Multi-phase progress with detailed status
def show_processing_progress():
    """
    Three-phase progress indicator:
    1. Preprocessing (parsing, chunking)
    2. Embedding (model inference)
    3. Indexing (ChromaDB storage)
    """
    with st.status("Processing files...", expanded=True) as status:
        # Phase 1
        st.write("📄 Preprocessing files...")
        progress_bar = st.progress(0)
        # Update progress_bar.progress(value)

        # Phase 2
        st.write("🧠 Generating embeddings...")
        # Show sub-progress with chunks/second

        # Phase 3
        st.write("💾 Indexing in database...")

        status.update(label="✅ Processing complete!", state="complete")
```

#### 2. Empty States

| State                           | Message                                        | Action                                     |
| ------------------------------- | ---------------------------------------------- | ------------------------------------------ |
| No documents loaded             | "📭 No documents in database yet"              | "📥 Go to Load Documents" button           |
| No search results               | "🔍 No matches found for your query"           | Suggestions: "Try broader terms"           |
| No clusters (insufficient data) | "📊 Need at least 50 documents for clustering" | Show current count                         |
| Processing error                | "❌ Error: [message]"                          | "🔄 Retry" button + error details expander |

#### 3. Feedback Patterns

```python
# Success toast (auto-dismiss)
st.toast("✅ 3 files processed successfully!", icon="✅")

# Warning banner (persistent)
st.warning("⚠️ 2 files were skipped (already indexed)")

# Error with details
st.error("❌ Failed to process video_xyz.srt")
with st.expander("View error details"):
    st.code(traceback_string)

# Informational callout
st.info("💡 Tip: Use UMAP for faster visualization, t-SNE for better cluster separation")
```

#### 4. Confirmation Dialogs

```python
# For destructive actions
if st.button("🗑️ Clear All Data"):
    # Show confirmation modal
    st.warning("⚠️ This will delete all embeddings and cannot be undone!")
    col1, col2 = st.columns(2)
    with col1:
        if st.button("❌ Cancel"):
            st.rerun()
    with col2:
        if st.button("✅ Confirm Delete", type="primary"):
            # Perform deletion
            pass
```

---

### Responsive Design Considerations

#### Layout Breakpoints

| Screen Size         | Sidebar              | Columns   | Chart Size |
| ------------------- | -------------------- | --------- | ---------- |
| Desktop (>1200px)   | Expanded (280px)     | 4 columns | Large      |
| Tablet (768-1200px) | Collapsed by default | 2 columns | Medium     |
| Mobile (<768px)     | Hidden               | 1 column  | Full width |

#### Streamlit Column Usage

```python
# Metric cards - responsive grid
cols = st.columns([1, 1, 1, 1])  # Desktop: 4 columns

# Form layout - 2/3 + 1/3 split
col_main, col_side = st.columns([2, 1])

# Action buttons - equal width
btn_col1, btn_col2 = st.columns(2)
```

---

### Accessibility Guidelines

1. **Color Contrast**: All text meets WCAG AA (4.5:1 ratio)
2. **Interactive Elements**: Minimum 44x44px touch targets
3. **Keyboard Navigation**: All features accessible via keyboard
4. **Screen Reader**: Use `st.markdown()` with proper heading hierarchy
5. **Alt Text**: All charts have descriptive captions via `st.caption()`

---

### Microinteractions

| Action          | Feedback                 | Duration         |
| --------------- | ------------------------ | ---------------- |
| Button click    | Visual press state       | Instant          |
| File drag over  | Border highlight         | Instant          |
| Search submit   | Input disabled + spinner | Until complete   |
| Chart hover     | Tooltip appears          | 100ms delay      |
| Progress update | Smooth animation         | 200ms transition |
| Tab switch      | Content fade             | 150ms            |

---

### Error State Designs

```
┌─────────────────────────────────────────────────────────────────┐
│  ❌ Processing Failed                                            │
│  ═══════════════════════════════════════════════════════════════ │
│                                                                  │
│  Could not process 2 of 5 files:                                │
│                                                                  │
│  ┌────────────────────────────────────────────────────────────┐ │
│  │ ⚠️ video_corrupt.srt                                       │ │
│  │   Error: Invalid SRT format at line 45                     │ │
│  │   [View Details]                                           │ │
│  ├────────────────────────────────────────────────────────────┤ │
│  │ ⚠️ video_empty.srt                                         │ │
│  │   Error: File contains no subtitle entries                 │ │
│  │   [View Details]                                           │ │
│  └────────────────────────────────────────────────────────────┘ │
│                                                                  │
│  ✅ 3 files processed successfully                              │
│                                                                  │
│  [🔄 Retry Failed]  [➡️ Continue to PostProcessing]            │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

---

## Implementation Plan

### 1. [x] Phase 1: Project Setup & Refactoring Preparation

_Description: Analyze existing code, establish structure, and create helper modules._

#### 1.1. [x] Task: Backup existing streamlit_app.py

_Description: Create backup of current implementation before major changes._

- Create `streamlit_app.py.backup_20251203` (manual backup)
- Document current functionality for reference

#### 1.2. [x] Task: Create page module structure

_Description: Create separate Python files for each page's rendering logic._

##### 1.2.1. [x] Sub-Task: Create `src/ui/__init__.py`

_Description: Initialize UI module package._

##### 1.2.2. [x] Sub-Task: Create `src/ui/pages/__init__.py`

_Description: Initialize pages subpackage._

##### 1.2.3. [x] Sub-Task: Create `src/ui/components/__init__.py`

_Description: Initialize reusable UI components subpackage._

#### 1.3. [x] Task: Define session state schema

_Description: Create centralized session state initialization and management._

##### 1.3.1. [x] Sub-Task: Create `src/ui/state.py`

_Description: Implement `initialize_session_state()` function with all required keys and defaults._

#### 1.4. [x] Task: Implement Design System Foundation

_Description: Create reusable styling and component utilities._

##### 1.4.1. [x] Sub-Task: Create `src/ui/theme.py`

_Description: Define color palette, typography constants, and styling helpers._

```python
# Contents to implement:
COLORS = {...}  # As defined in UI/UX specs
ICONS = {...}   # Emoji icon mapping
def style_metric_card(...): ...
def style_result_card(...): ...
```

##### 1.4.2. [x] Sub-Task: Create `src/ui/components/feedback.py`

_Description: Implement feedback components:_

- `show_success_toast(message)`
- `show_error_with_details(message, details)`
- `show_empty_state(title, message, action_label, action_callback)`
- `show_loading_status(phases, current_phase, progress)`

##### 1.4.3. [x] Sub-Task: Create custom CSS injection

_Description: Use `st.markdown()` with `unsafe_allow_html=True` for custom styles:_

- Metric card hover effects
- Result card styling
- Progress bar customization
- Responsive adjustments

---

### 2. [x] Phase 2: Load Documents Section Implementation

_Description: Implement file upload and complete processing pipeline in Streamlit._

#### 2.1. [x] Task: Create Load Documents page structure

_Description: Implement `src/ui/pages/load_documents.py` with main rendering function._

##### 2.1.1. [x] Sub-Task: Implement file upload widget

_Description: Use `st.file_uploader()` for multiple SRT file upload with validation._

- Accept only `.srt` files
- Show file count and total size
- Validate file encoding (UTF-8)
- **UX Enhancement**: Drag-and-drop visual feedback with highlighted border
- **UX Enhancement**: File list preview with remove option per file

##### 2.1.2. [x] Sub-Task: Implement directory path input

_Description: Alternative to upload: text input for local directory path (for large batches)._

- Validate path exists
- Count SRT files in directory
- Show preview of files to process
- **UX Enhancement**: Use tabs to switch between Upload/Directory modes

##### 2.1.3. [x] Sub-Task: Implement model selection dropdown

_Description: Populate from `model_registry.get_available_models()` with model info display._

- Show model name, dimensions, adapter type
- Default to config model
- Store selection in session state
- **UX Enhancement**: Info tooltip with model capabilities and resource requirements

#### 2.2. [x] Task: Implement processing options UI

_Description: Add configuration controls for processing pipeline._

##### 2.2.1. [x] Sub-Task: Add batch size slider

_Description: Slider for embedding batch size (16-256, default from config)._

##### 2.2.2. [x] Sub-Task: Add "Skip duplicates" checkbox

_Description: Toggle for skipping already-indexed videos._

##### 2.2.3. [x] Sub-Task: Add parallel processing toggle

_Description: Enable/disable parallel preprocessing._

#### 2.3. [x] Task: Implement processing pipeline integration

_Description: Connect UI to backend processing pipelines with progress tracking._

##### 2.3.1. [x] Sub-Task: Create `process_uploaded_files()` helper function

_Description: Function that:_

1. Saves uploaded files to temp directory
2. Calls `PreprocessingPipeline.process_multiple_files()`
3. Updates session state progress
4. Returns list of `ProcessedVideo` objects

##### 2.3.2. [x] Sub-Task: Create `generate_and_index_embeddings()` helper function

_Description: Function that:_

1. Iterates over `ProcessedVideo` objects
2. Calls `EmbeddingPipeline.generate_embeddings_with_checkpointing()`
3. Calls `VectorStorePipeline.index_processed_video()`
4. Updates progress bar
5. Handles errors gracefully (continue on error, log failures)

##### 2.3.3. [x] Sub-Task: Implement progress visualization

_Description: Use `st.progress()` and `st.status()` to show:_

- Current phase (Preprocessing / Embedding / Indexing)
- Files processed / total
- Chunks generated / indexed
- Estimated time remaining

**UX Implementation Details:**

```python
# Use st.status() for multi-phase progress
with st.status("🔄 Processing files...", expanded=True) as status:
    st.write("📄 Phase 1: Preprocessing...")
    preprocess_progress = st.progress(0, text="Parsing SRT files...")

    # Update during processing
    preprocess_progress.progress(50, text="2/4 files parsed")

    st.write("🧠 Phase 2: Generating embeddings...")
    embed_progress = st.progress(0)
    speed_metric = st.empty()  # For real-time speed display

    # Show speed: "48 chunks/sec • ETA: 12s"
    speed_metric.caption("⚡ 48 chunks/sec • ⏱️ ETA: 12s")

    st.write("💾 Phase 3: Indexing...")
    status.update(label="✅ Complete!", state="complete", expanded=False)
```

#### 2.4. [ ] Task: Implement processing results display

_Description: Show statistics after processing completes._

##### 2.4.1. [ ] Sub-Task: Create processing summary component

_Description: Display:_

- Total files processed
- Total chunks created
- Total embeddings generated
- Processing time
- Any errors encountered

##### 2.4.2. [ ] Sub-Task: Add "View in PostProcessing" button

_Description: Navigation button to PostProcessing section after successful load._

---

### 3. [ ] Phase 3: PostProcessing Section Implementation

_Description: Implement vector database visualization and statistical analysis._

#### 3.1. [x] Task: Create PostProcessing page structure

_Description: Implement `src/ui/pages/postprocessing.py` with sub-tabs for different analyses._

##### 3.1.1. [x] Sub-Task: Create main page layout with tabs

_Description: Use `st.tabs()` to create:_

- "📊 Overview" – Database statistics
- "🌐 Visualization" – 2D/3D embedding plots
- "🔬 Clustering" – Clustering analysis
- "📈 Metrics" – Detailed statistics and export

#### 3.2. [x] Task: Implement Overview tab

_Description: Database statistics and metadata distribution._

##### 3.2.1. [x] Sub-Task: Display key metrics

_Description: Show:_

- Total documents
- Unique videos
- Average chunks per video
- Embedding dimensions
- Collection size on disk

##### 3.2.2. [x] Sub-Task: Add date distribution chart

_Description: Bar chart of chunks per date (existing code, refactor)._

##### 3.2.3. [x] Sub-Task: Add video distribution chart

_Description: Bar chart of chunks per video (top N)._

#### 3.3. [x] Task: Implement Visualization tab

_Description: 2D and 3D embedding space visualization._

##### 3.3.1. [x] Sub-Task: Add dimensionality reduction options

_Description: Controls for:_

- Algorithm selection (UMAP, t-SNE, PCA)
- Number of points to visualize
- Dimension (2D / 3D toggle)
- Color coding (by Video ID, Date, Cluster, None)

##### 3.3.2. [x] Sub-Task: Implement 2D scatter plot

_Description: Add 2D visualization option using Plotly `scatter`._

##### 3.3.3. [x] Sub-Task: Refactor 3D visualization

_Description: Move existing 3D visualization code to this section with improvements:_

- Better hover information
- Cluster coloring support
- Animation controls

##### 3.3.4. [x] Sub-Task: Add download button for coordinates

_Description: Export reduced coordinates as CSV._

#### 3.4. [x] Task: Implement Clustering tab

_Description: Interactive clustering analysis using HDBSCAN._

##### 3.4.1. [x] Sub-Task: Add clustering parameter controls

_Description: UI controls for HDBSCAN parameters:_

- `min_cluster_size` (slider, 5-100)
- `min_samples` (slider, 1-50)
- `cluster_selection_epsilon` (slider, 0.0-1.0)
- `metric` (selectbox: cosine, euclidean)

##### 3.4.2. [x] Sub-Task: Implement "Run Clustering" action

_Description: Button that:_

1. Fetches embeddings from ChromaDB
2. Instantiates `HDBSCANClusterer` with UI parameters
3. Calls `clusterer.fit(embeddings)`
4. Stores labels in session state
5. Optionally persists labels via `ClusterManager.store_cluster_labels()`

##### 3.4.3. [x] Sub-Task: Display clustering results

_Description: Show after clustering:_

- Number of clusters found
- Number of outliers
- Cluster size distribution (bar chart)
- Option to view cluster contents

##### 3.4.4. [x] Sub-Task: Integrate cluster evaluation

_Description: Use `ClusterEvaluator` to display:_

- Silhouette score
- Davies-Bouldin index
- Calinski-Harabasz index
- Interpretation hints (e.g., "Silhouette > 0.5 indicates good clustering")

##### 3.4.5. [x] Sub-Task: Add cluster exploration UI

_Description: Expandable sections to view documents in each cluster:_

- Select cluster from dropdown
- Show sample documents
- Show cluster keywords/themes (if available)

#### 3.5. [x] Task: Implement Metrics tab

_Description: Detailed statistics and data export._

##### 3.5.1. [x] Sub-Task: Add video list with statistics

_Description: Table showing all videos with chunk counts (refactor existing code)._

##### 3.5.2. [x] Sub-Task: Add export functionality

_Description: Download buttons for:_

- Video list (CSV)
- Full metadata (JSON)
- Clustering results (CSV)
- Embeddings sample (NPZ)

---

### 4. [x] Phase 4: Search Section Implementation

_Description: Implement semantic search interface with filters and result display._

#### 4.1. [x] Task: Create Search page structure

_Description: Implement `src/ui/pages/search.py` with search interface._

##### 4.1.1. [x] Sub-Task: Implement search query input

_Description: Text input with:_

- Placeholder text
- Character count
- Clear button

##### 4.1.2. [x] Sub-Task: Implement search controls

_Description: Add controls:_

- Number of results slider (1-50)
- Minimum similarity threshold slider (0.0-1.0)
- "Include metadata" checkbox

#### 4.2. [x] Task: Implement search filters UI

_Description: Collapsible filter section using `st.expander()`._

##### 4.2.1. [x] Sub-Task: Add video ID filter

_Description: Text input or selectbox populated from collection._

##### 4.2.2. [x] Sub-Task: Add date range filter

_Description: Date input widgets for start/end date._

##### 4.2.3. [x] Sub-Task: Add title keyword filter

_Description: Text input for title keywords (comma-separated)._

#### 4.3. [x] Task: Implement search execution

_Description: Connect to `QueryEngine` for search._

##### 4.3.1. [x] Sub-Task: Create search handler function

_Description: Function that:_

1. Validates query (non-empty, within length limits)
2. Constructs `SearchFilters` from UI inputs
3. Calls `QueryEngine.query()`
4. Stores results in session state
5. Handles errors gracefully

#### 4.4. [x] Task: Implement search results display

_Description: Rich result visualization._

##### 4.4.1. [x] Sub-Task: Create result card component

_Description: For each result, show:_

- Similarity score (with progress bar)
- Video title
- Date
- Text snippet with query highlighting
- Expand button for full text

**UX Implementation Details:**

```python
def render_result_card(result: SearchResult, query: str, rank: int):
    """Render a search result with visual similarity score."""
    score = result.similarity_score

    # Color-coded score bar
    if score > 0.8:
        color = "green"
    elif score > 0.5:
        color = "orange"
    else:
        color = "red"

    with st.container(border=True):
        # Score bar with percentage
        st.progress(score, text=f"{'█' * int(score*20)}{'░' * (20-int(score*20))} {score:.1%} match")

        # Metadata row
        col1, col2 = st.columns([3, 1])
        with col1:
            st.markdown(f"**🎬 {result.metadata.title}**")
        with col2:
            st.caption(f"📅 {result.metadata.date}")

        # Highlighted text snippet
        highlighted_text = highlight_query_terms(result.text[:200], query)
        st.markdown(highlighted_text + "...")

        # Action buttons
        btn_col1, btn_col2, btn_col3 = st.columns(3)
        with btn_col1:
            if st.button("📖 Full Text", key=f"full_{rank}"):
                st.session_state[f"expand_{rank}"] = True
        with btn_col2:
            st.button("🔗 Context", key=f"ctx_{rank}")
        with btn_col3:
            st.link_button("▶️ Video", f"https://youtube.com/watch?v={result.metadata.video_id}")
```

##### 4.4.2. [x] Sub-Task: Add result grouping option

_Description: Toggle to group results by video ID._

##### 4.4.3. [x] Sub-Task: Add context expansion

_Description: Button to fetch and display surrounding chunks for context._

##### 4.4.4. [x] Sub-Task: Add result export

_Description: Download button for search results as JSON/CSV._

---

### 5. [x] Phase 5: Integration & Main App Restructure

_Description: Integrate all sections into main app with navigation._

#### 5.1. [x] Task: Restructure main `streamlit_app.py`

_Description: Refactor main file to use new modular structure._

##### 5.1.1. [x] Sub-Task: Update imports

_Description: Import page rendering functions from `src.ui.pages`._

##### 5.1.2. [x] Sub-Task: Implement sidebar navigation

_Description: Use `st.sidebar.radio()` with three sections:_

- 📥 Load Documents
- 🔬 PostProcessing
- 🔍 Search

##### 5.1.3. [x] Sub-Task: Add sidebar info panel

_Description: Display:_

- Current model name
- Total documents in collection
- Last processing timestamp

##### 5.1.4. [x] Sub-Task: Implement page routing

_Description: Call appropriate render function based on selection._

#### 5.2. [x] Task: Implement global session state management

_Description: Ensure consistent state across page navigation._

##### 5.2.1. [x] Sub-Task: Initialize state on app load

_Description: Call `initialize_session_state()` at app start._

##### 5.2.2. [x] Sub-Task: Add collection refresh mechanism

_Description: Button or automatic refresh when collection changes._

---

### 6. [ ] Phase 6: Testing & Quality Assurance

_Description: Test all functionality and fix issues._

#### 6.1. [ ] Task: Manual UI testing

_Description: Test each section manually._

##### 6.1.1. [ ] Sub-Task: Test Load Documents section

_Description: Test with:_

- Single file upload
- Multiple file upload
- Directory path input
- Different model selections
- Error handling (invalid files, processing failures)

##### 6.1.2. [ ] Sub-Task: Test PostProcessing section

_Description: Test:_

- Statistics display
- 2D/3D visualization with different settings
- Clustering with various parameters
- Export functionality

##### 6.1.3. [ ] Sub-Task: Test Search section

_Description: Test:_

- Basic search
- Search with filters
- Empty results handling
- Result display and expansion

#### 6.2. [ ] Task: Performance testing

_Description: Ensure acceptable performance._

##### 6.2.1. [ ] Sub-Task: Test with large file batches

_Description: Process 50+ files, verify progress tracking works._

##### 6.2.2. [ ] Sub-Task: Test visualization with many points

_Description: Visualize 2000+ points, verify responsiveness._

#### 6.3. [ ] Task: Error handling review

_Description: Ensure graceful error handling throughout._

##### 6.3.1. [ ] Sub-Task: Review and improve error messages

_Description: Ensure user-friendly error messages for all failure modes._

---

### 7. [ ] Phase 7: Documentation & Cleanup

_Description: Update documentation and clean up code._

#### 7.1. [ ] Task: Update README_STREAMLIT.md

_Description: Document new three-section structure and usage._

##### 7.1.1. [ ] Sub-Task: Document Load Documents section

_Description: Usage instructions, supported file formats, options._

##### 7.1.2. [ ] Sub-Task: Document PostProcessing section

_Description: Explain visualization options, clustering parameters, metrics interpretation._

##### 7.1.3. [ ] Sub-Task: Document Search section

_Description: Search tips, filter usage, result interpretation._

#### 7.2. [ ] Task: Code cleanup

_Description: Remove dead code, add docstrings._

##### 7.2.1. [ ] Sub-Task: Remove backup files

_Description: Remove old backup files after confirming new implementation works._

##### 7.2.2. [ ] Sub-Task: Add docstrings to all new functions

_Description: Comprehensive docstrings for maintainability._

#### 7.3. [ ] Task: Update requirements.txt if needed

_Description: Verify all dependencies are listed (likely no changes needed)._

---

## Appendix: File Structure After Implementation

```
streamlit_app.py                    # Main entry point (simplified)
src/
├── ui/
│   ├── __init__.py
│   ├── state.py                    # Session state management
│   ├── theme.py                    # Design system (colors, icons, styles)
│   ├── components/
│   │   ├── __init__.py
│   │   ├── feedback.py             # Toast, errors, empty states
│   │   ├── progress_tracker.py     # Multi-phase progress
│   │   ├── result_card.py          # Search result display
│   │   ├── metric_card.py          # Statistics cards
│   │   ├── file_preview.py         # File list with actions
│   │   └── cluster_explorer.py     # Cluster contents viewer
│   └── pages/
│       ├── __init__.py
│       ├── load_documents.py       # Load Documents section
│       ├── postprocessing.py       # PostProcessing section
│       └── search.py               # Search section
```

---

## Appendix: Reusable UI Component Catalog

### 1. MetricCard Component

```python
def metric_card(
    label: str,
    value: Union[int, float, str],
    icon: str = "📊",
    delta: Optional[str] = None,
    help_text: Optional[str] = None
):
    """
    Styled metric display card.

    Usage:
        metric_card("Documents", 1234, icon="📄", delta="+56 today")
    """
    with st.container(border=True):
        st.metric(
            label=f"{icon} {label}",
            value=value,
            delta=delta,
            help=help_text
        )
```

### 2. EmptyState Component

```python
def empty_state(
    title: str,
    message: str,
    icon: str = "📭",
    action_label: Optional[str] = None,
    action_page: Optional[str] = None
):
    """
    Display when no data is available.

    Usage:
        empty_state(
            "No documents yet",
            "Upload subtitle files to get started",
            icon="📥",
            action_label="Go to Load Documents",
            action_page="📥 Load Documents"
        )
    """
    st.markdown(f"""
    <div style="text-align: center; padding: 40px;">
        <div style="font-size: 48px;">{icon}</div>
        <h3>{title}</h3>
        <p style="color: gray;">{message}</p>
    </div>
    """, unsafe_allow_html=True)

    if action_label and action_page:
        if st.button(action_label, use_container_width=True):
            st.session_state.page = action_page
            st.rerun()
```

### 3. ProcessingStatus Component

```python
def processing_status(
    phases: List[str],
    current_phase: int,
    phase_progress: float,
    details: Dict[str, Any]
):
    """
    Multi-phase processing indicator.

    Usage:
        processing_status(
            phases=["Preprocessing", "Embedding", "Indexing"],
            current_phase=1,
            phase_progress=0.45,
            details={"files": "2/5", "speed": "48 chunks/s", "eta": "12s"}
        )
    """
    phase_icons = ["📄", "🧠", "💾"]

    # Phase indicators
    cols = st.columns(len(phases))
    for i, (phase, icon) in enumerate(zip(phases, phase_icons)):
        with cols[i]:
            if i < current_phase:
                st.markdown(f"✅ {phase}")
            elif i == current_phase:
                st.markdown(f"**⏳ {phase}**")
            else:
                st.markdown(f"⬜ {phase}")

    # Progress bar
    st.progress(phase_progress)

    # Details row
    detail_cols = st.columns(3)
    with detail_cols[0]:
        st.caption(f"📁 {details.get('files', '-')}")
    with detail_cols[1]:
        st.caption(f"⚡ {details.get('speed', '-')}")
    with detail_cols[2]:
        st.caption(f"⏱️ ETA: {details.get('eta', '-')}")
```

### 4. ScoreBar Component

```python
def score_bar(score: float, label: str = "Match"):
    """
    Visual similarity score indicator with color coding.

    Usage:
        score_bar(0.85, label="Similarity")
    """
    if score >= 0.8:
        color = "#28A745"  # Green
        quality = "Excellent"
    elif score >= 0.5:
        color = "#FFC107"  # Yellow
        quality = "Good"
    else:
        color = "#DC3545"  # Red
        quality = "Low"

    filled = int(score * 20)
    bar = "█" * filled + "░" * (20 - filled)

    st.markdown(f"""
    <div style="font-family: monospace;">
        <span style="color: {color};">{bar}</span>
        <strong>{score:.1%}</strong> {label}
        <span style="color: gray; font-size: 0.8em;">({quality})</span>
    </div>
    """, unsafe_allow_html=True)
```

### 5. ClusteringPresets

```python
CLUSTERING_PRESETS = {
    "🚀 Quick": {
        "min_cluster_size": 5,
        "min_samples": 3,
        "description": "Fast results, may have more noise"
    },
    "⚖️ Balanced": {
        "min_cluster_size": 15,
        "min_samples": 5,
        "description": "Good balance of speed and quality"
    },
    "🔬 Detailed": {
        "min_cluster_size": 30,
        "min_samples": 10,
        "description": "Fewer, more coherent clusters"
    }
}

def clustering_preset_selector():
    """Preset buttons for clustering parameters."""
    st.write("**Quick Presets:**")
    cols = st.columns(3)

    for i, (name, params) in enumerate(CLUSTERING_PRESETS.items()):
        with cols[i]:
            if st.button(name, help=params["description"], use_container_width=True):
                st.session_state.min_cluster_size = params["min_cluster_size"]
                st.session_state.min_samples = params["min_samples"]
                st.rerun()
```

---

## Estimated Effort

| Phase                          | Estimated Hours | UX Focus Areas                                     |
| ------------------------------ | --------------- | -------------------------------------------------- |
| Phase 1: Setup + Design System | 4               | Theme, components, feedback patterns               |
| Phase 2: Load Documents        | 7               | Progress visualization, error states, file preview |
| Phase 3: PostProcessing        | 10              | Interactive charts, clustering UX, tooltips        |
| Phase 4: Search                | 6               | Result cards, highlighting, empty states           |
| Phase 5: Integration           | 3               | Navigation, state management                       |
| Phase 6: Testing               | 4               | Usability testing, edge cases                      |
| Phase 7: Documentation         | 2               | User guides with screenshots                       |
| **Total**                      | **36 hours**    |                                                    |

---

## UX Success Metrics

| Metric                             | Target       | Measurement                                   |
| ---------------------------------- | ------------ | --------------------------------------------- |
| Time to first search               | < 30 seconds | From app load to search result                |
| Processing feedback clarity        | 100%         | Users always know current status              |
| Error recovery rate                | > 90%        | Users can recover from errors without refresh |
| Feature discoverability            | > 80%        | Users find clustering without help text       |
| Search result relevance perception | > 4/5        | User satisfaction rating                      |

---

## Risk Assessment

| Risk                                     | Impact | Mitigation                                                |
| ---------------------------------------- | ------ | --------------------------------------------------------- |
| Long processing times block UI           | High   | Use `st.status()` with expandable progress, show ETA      |
| Large file uploads exceed memory         | Medium | Process files in batches, stream processing               |
| Clustering fails on small datasets       | Low    | Add minimum data checks, informative empty states         |
| Model loading fails                      | Medium | Graceful fallback, clear error messages with retry button |
| Users don't understand clustering params | Medium | Add presets ("Quick", "Balanced", "Detailed"), tooltips   |
| Search returns irrelevant results        | Medium | Show score interpretation, filtering tips                 |
