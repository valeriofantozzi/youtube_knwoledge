# Vector Database Web Explorer

Interactive web interface for managing subtitle embeddings with a complete processing pipeline.

## Overview

The application is organized into three main sections:

1. **📥 Load Documents** - Upload SRT files and generate embeddings
2. **🔬 PostProcessing** - Visualize and analyze the embedding space
3. **🔍 Search** - Perform semantic search with filters

## Installation

Dependencies are already installed if you ran `pip install -r requirements.txt`.

If needed:

```bash
source .venv/bin/activate
pip install streamlit plotly umap-learn hdbscan
```

## Quick Start

```bash
./start_viewer.sh
```

Or manually:

```bash
source .venv/bin/activate
streamlit run streamlit_app.py --server.port 8601
```

The application will automatically open in your browser at `http://localhost:8601`

---

## Section Details

### 📥 Load Documents

Upload subtitle files and process them through the complete pipeline.

**Features:**

- Upload single or multiple SRT files (drag & drop supported)
- Alternatively, specify a directory path containing SRT files
- Select embedding model from the registry
- Configure batch size and processing options
- Real-time progress tracking with phase indicators
- Skip already indexed videos to avoid duplicates

**Processing Pipeline:**

1. **Preprocessing**: Parse SRT files and create semantic chunks
2. **Embedding**: Generate vector embeddings using selected model
3. **Indexing**: Store embeddings in ChromaDB vector database

---

### 🔬 PostProcessing

Analyze and visualize your embedding space.

**Tabs:**

#### Overview

- Key metrics: total documents, unique videos, average chunks per video
- Date distribution chart
- Top videos by chunk count

#### Visualization

- 2D and 3D scatter plots of the embedding space
- Dimensionality reduction methods: UMAP (fast), t-SNE (better separation), PCA
- Color by: Video ID, Date, Cluster, or None
- Download coordinates as CSV

#### Clustering

- HDBSCAN clustering with configurable parameters
- Quick presets: Quick, Balanced, Detailed
- Manual parameter tuning: min_cluster_size, min_samples, metric
- Cluster quality metrics: Silhouette Score, Davies-Bouldin Index
- Explore cluster contents with sample documents

#### Export

- Export video list as CSV
- Export clustering results
- Download visualization coordinates

---

### 🔍 Search

Perform semantic search with natural language queries.

**Features:**

- Natural language query input
- Configurable result count and minimum similarity score
- Advanced filters:
  - Filter by video ID
  - Date range filters
  - Title keywords (comma-separated)
- Results display with:
  - Similarity score bar (color-coded)
  - Video title and metadata
  - Text preview with query highlighting
  - Full text expansion
  - Direct link to YouTube video
- Export search results as CSV

**Tips:**

- Use natural language questions for best results
- Try descriptive phrases like "how to care for orchids in winter"
- Use filters to narrow down results to specific videos or date ranges

---

## Architecture

```
streamlit_app.py              # Main entry point
src/
└── ui/
    ├── __init__.py           # UI module exports
    ├── state.py              # Session state management
    ├── theme.py              # Design system (colors, icons)
    ├── components/           # Reusable UI components
    │   ├── feedback.py       # Toast, errors, empty states
    │   ├── metric_card.py    # Statistics display
    │   ├── progress_tracker.py # Multi-phase progress
    │   └── result_card.py    # Search result display
    └── pages/                # Page modules
        ├── load_documents.py # Load Documents section
        ├── postprocessing.py # PostProcessing section
        └── search.py         # Search section
```

---

## Supported Models

### BAAI/bge-large-en-v1.5

- **Dimensions**: 1024
- **Max Length**: 512 tokens
- **Best for**: High-quality general search

### Google/embeddinggemma-300m

- **Dimensions**: 768 (with MRL support for flexible dimensions)
- **Max Length**: 2048 tokens
- **Best for**: Long content and contextual search

---

## Important Notes

- **Separate Collections**: Each model uses its own collection in the database
- **Model-Specific Search**: Search results are specific to the selected model
- **Persistent Storage**: All data is stored in `data/vector_db/`
- **Backup**: Original app backed up as `streamlit_app_old.py`
