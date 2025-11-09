# Subtitle Embedding & Retrieval System - Implementation Plan

## Progress Summary

**Last Updated**: 2025-11-09

### Completed Phases
- ✅ **Phase 1**: Project Setup & Infrastructure (100% complete)
- ✅ **Phase 2**: Preprocessing Pipeline (100% complete)
- ✅ **Phase 3**: Embedding Generation (100% complete)
- ✅ **Phase 4**: Vector Store Setup (100% complete - fully functional with Python 3.12)
- ✅ **Phase 4.5**: Performance Optimization & Hardware Detection (100% complete - core optimizations implemented)
- ✅ **Phase 4.6**: Vector Database Visualization (100% complete - CLI and Web UI implemented)

### In Progress
- ⏳ **Phase 5**: Retrieval System (in progress - clustering system added)

### Overall Progress
- **Tasks Completed**: 137/210 (~65%)
- **Core Infrastructure**: ✅ Complete
- **Data Processing**: ✅ Complete (including parallel processing)
- **Embedding Generation**: ✅ Complete (including checkpointing and memory management)
- **Vector Storage**: ✅ Complete (including schema migrations and index verification) - ChromaDB fully functional with Python 3.12
- **Performance Optimization**: ✅ Complete - Hardware-aware optimization system implemented with ~4x speedup
- **Visualization Tools**: ✅ Complete - CLI viewer and interactive Streamlit web UI with 3D visualization

---

## Model Selection

**Selected Model: `BGE-large-en-v1.5` (BAAI General Embedding)**

**Rationale:**
- Optimized specifically for retrieval tasks
- 1024-dimensional embeddings (high quality)
- Excellent performance on semantic search benchmarks
- Supports instruction-based queries
- ~1.2GB model size (manageable for powerful machines)
- Good balance between quality and performance

**Alternative Models Considered:**
- `e5-large-v2`: Similar quality, slightly different architecture
- `multilingual-e5-large`: If multilingual support needed later

---

## Project Architecture

### System Components

1. **Preprocessing Pipeline**: Parse SRT files, extract text, create semantic chunks
2. **Embedding Pipeline**: Generate embeddings using BGE-large model
3. **Vector Store**: Store embeddings with metadata in ChromaDB
4. **Retrieval System**: Query interface with similarity search
5. **CLI Interface**: Command-line tools for processing and querying

### Technology Stack

- **Language**: Python 3.9+
- **Embedding Model**: `sentence-transformers` with `BAAI/bge-large-en-v1.5`
- **Vector Database**: ChromaDB (local, persistent)
- **Deep Learning**: PyTorch (with CUDA support if GPU available)
- **Data Processing**: pandas, numpy
- **CLI**: argparse, rich (for better terminal output)

---

## Directory Structure

```
project_root/
├── README.md
├── requirements.txt
├── .env.example
├── .gitignore
│
├── src/
│   ├── __init__.py
│   │
│   ├── preprocessing/
│   │   ├── __init__.py
│   │   ├── srt_parser.py          # Parse SRT files, extract text
│   │   ├── text_cleaner.py        # Clean and normalize text
│   │   ├── chunker.py             # Semantic chunking with overlap
│   │   └── metadata_extractor.py  # Extract metadata from filenames
│   │
│   ├── embeddings/
│   │   ├── __init__.py
│   │   ├── model_loader.py        # Load and initialize BGE model
│   │   ├── embedder.py            # Generate embeddings in batches
│   │   └── batch_processor.py     # Handle batch processing logic
│   │
│   ├── vector_store/
│   │   ├── __init__.py
│   │   ├── chroma_manager.py      # ChromaDB setup and management
│   │   ├── indexer.py             # Index embeddings with metadata
│   │   └── schema.py              # Define metadata schema
│   │
│   ├── retrieval/
│   │   ├── __init__.py
│   │   ├── query_engine.py        # Main retrieval interface
│   │   ├── similarity_search.py  # Similarity search implementation
│   │   ├── reranker.py            # Optional reranking (future)
│   │   └── result_formatter.py    # Format search results
│   │
│   ├── clustering/
│   │   ├── __init__.py
│   │   ├── clusterer.py           # Main clustering interface
│   │   ├── hdbscan_clusterer.py   # HDBSCAN implementation
│   │   ├── cluster_manager.py     # Cluster storage and management
│   │   ├── cluster_evaluator.py   # Cluster quality evaluation
│   │   ├── cluster_visualizer.py   # Visualization integration
│   │   └── cluster_integrator.py  # Integration with retrieval
│   │
│   └── utils/
│       ├── __init__.py
│       ├── config.py              # Configuration management
│       ├── logger.py              # Logging setup
│       └── progress.py            # Progress tracking utilities
│
├── scripts/
│   ├── process_subtitles.py      # Main script: process all subtitles
│   ├── view_vector_db.py          # CLI viewer for vector database
│   ├── query_subtitles.py         # Query interface script
│   └── rebuild_index.py           # Rebuild index from scratch
│
├── streamlit_app.py               # Streamlit web UI for vector database visualization
├── README_STREAMLIT.md            # Documentation for Streamlit app
│
├── data/
│   ├── raw/
│   │   └── subtitles/             # Original SRT files (symlink or copy)
│   ├── processed/
│   │   ├── chunks/                # Processed chunks (JSON/Parquet)
│   │   └── metadata/              # Extracted metadata
│   └── vector_db/                 # ChromaDB persistent storage
│
├── tests/
│   ├── __init__.py
│   ├── test_preprocessing.py
│   ├── test_embeddings.py
│   ├── test_retrieval.py
│   └── fixtures/
│       └── sample.srt
│
├── notebooks/
│   └── exploration.ipynb          # Jupyter notebook for exploration
│
└── docs/
    ├── architecture.md
    ├── api_reference.md
    └── usage_guide.md
```

---

## Implementation Phases

### Phase 1: Project Setup & Infrastructure

#### Task 1.1: Initialize Project Structure
- [x] Create directory structure as defined above
- [x] Initialize Python package structure (`src/` with `__init__.py` files)
- [x] Create `.gitignore` file (exclude `data/vector_db/`, `__pycache__/`, `.venv/`, etc.)
- [x] Create `README.md` with project overview
- [x] Create `.env.example` for configuration template

#### Task 1.2: Setup Python Environment
- [x] Create virtual environment (`.venv`)
- [x] Create `requirements.txt` with dependencies:
  - `sentence-transformers>=2.2.0`
  - `torch>=2.0.0` (with CUDA if GPU available)
  - `chromadb>=0.4.0`
  - `pandas>=2.0.0`
  - `numpy>=1.24.0`
  - `rich>=13.0.0` (for CLI output)
  - `python-dotenv>=1.0.0`
  - `tqdm>=4.65.0` (progress bars)
- [x] Install dependencies in virtual environment
- [x] Verify installation and imports

#### Task 1.3: Configuration System
- [x] Create `src/utils/config.py` with configuration management
- [x] Define configuration schema:
  - Model name and path
  - Batch size for embeddings
  - Chunk size and overlap
  - Vector database path
  - Device (CPU/GPU)
  - Logging level
- [x] Implement environment variable support
- [x] Create default configuration file
- [x] Add configuration validation

#### Task 1.4: Logging Infrastructure
- [x] Create `src/utils/logger.py`
- [x] Setup structured logging with levels (DEBUG, INFO, WARNING, ERROR)
- [x] Configure log file rotation
- [x] Add colored console output for better UX
- [x] Integrate logging across all modules

---

### Phase 2: Preprocessing Pipeline

#### Task 2.1: SRT Parser Implementation
- [x] Create `src/preprocessing/srt_parser.py`
- [x] Implement parser to read SRT files:
  - Parse sequence numbers
  - Extract timestamps (start/end)
  - Extract subtitle text
  - Handle multi-line subtitles
- [x] Handle edge cases:
  - Empty subtitles
  - Malformed timestamps
  - Encoding issues (UTF-8, UTF-8-BOM)
- [x] Add unit tests with sample SRT files
- [x] Implement batch processing for multiple files

#### Task 2.2: Metadata Extractor
- [x] Create `src/preprocessing/metadata_extractor.py`
- [x] Implement filename parser:
  - Extract date (YYYYMMDD format)
  - Extract video ID (11-character YouTube ID)
  - Extract title (remaining text)
  - Handle edge cases (missing parts, special characters)
- [x] Create metadata structure:
  ```python
  {
    "video_id": str,
    "date": str (YYYY/MM/DD),
    "title": str,
    "filename": str,
    "file_path": str
  }
  ```
- [x] Add validation for extracted metadata
- [x] Write unit tests

#### Task 2.3: Text Cleaner
- [x] Create `src/preprocessing/text_cleaner.py`
- [x] Implement text cleaning functions:
  - Remove SRT formatting artifacts
  - Remove `[Music]`, `[Applause]` tags
  - Normalize whitespace
  - Remove HTML entities if present
  - Handle special characters
- [x] Preserve sentence boundaries
- [ ] Add language detection (optional, for future multilingual support)
- [x] Write unit tests

#### Task 2.4: Semantic Chunker
- [x] Create `src/preprocessing/chunker.py`
- [x] Implement chunking strategy:
  - **Method**: Sentence-based with semantic grouping
  - **Chunk size**: 200-500 tokens (configurable)
  - **Overlap**: 20% between chunks (configurable)
  - **Min chunk size**: 50 tokens (avoid tiny chunks)
- [x] Group related sentences together:
  - Use sentence boundaries
  - Consider paragraph breaks
  - Maintain context continuity
- [x] Create chunk structure:
  ```python
  {
    "chunk_id": str (unique),
    "text": str,
    "chunk_index": int (position in video),
    "token_count": int,
    "metadata": dict (video metadata)
  }
  ```
- [x] Implement batch chunking for multiple videos
- [x] Add progress tracking
- [x] Write unit tests

#### Task 2.5: Preprocessing Pipeline Integration
- [x] Create main preprocessing orchestrator
- [x] Integrate all preprocessing components:
  - SRT parser → Text cleaner → Chunker → Metadata attachment
- [x] Implement parallel processing for multiple files
- [x] Add error handling and recovery
- [x] Save processed chunks to disk (JSON or Parquet format)
- [x] Create preprocessing statistics report:
  - Total chunks created
  - Average chunk size
  - Processing time
- [x] Write integration tests

---

### Phase 3: Embedding Generation

#### Task 3.1: Model Loader
- [x] Create `src/embeddings/model_loader.py`
- [x] Implement model loading:
  - Download `BAAI/bge-large-en-v1.5` from HuggingFace
  - Cache model locally
  - Load with appropriate device (CPU/GPU)
- [x] Handle model initialization:
  - Set model to eval mode
  - Configure normalization (BGE models use normalized embeddings)
  - Set max sequence length
- [x] Add model info logging:
  - Model name and version
  - Device used
  - Model parameters count
- [x] Implement model health check
- [x] Write unit tests

#### Task 3.2: Embedding Generator
- [x] Create `src/embeddings/embedder.py`
- [x] Implement embedding generation:
  - Accept list of text chunks
  - Generate embeddings using BGE model
  - Return numpy arrays or torch tensors
- [x] Handle instruction-based queries (BGE supports this):
  - For queries: prepend "Represent this sentence for searching relevant passages: "
  - For documents: use as-is or prepend "Represent this sentence: "
- [x] Implement normalization (BGE embeddings should be normalized)
- [x] Add input validation:
  - Check text length
  - Handle empty strings
  - Validate input types
- [x] Write unit tests

#### Task 3.3: Batch Processor
- [x] Create `src/embeddings/batch_processor.py`
- [x] Implement efficient batch processing:
  - **Batch size**: 64-128 for CPU, 256-512 for GPU (configurable)
  - Dynamic batching based on text length
  - Memory-efficient processing for large datasets
- [x] Add progress tracking:
  - Use tqdm for progress bars
  - Log processing speed (chunks/sec)
  - Estimate remaining time
- [x] Implement error handling:
  - Retry failed batches
  - Skip problematic chunks with logging
  - Continue processing on errors
- [x] Add memory management:
  - Clear cache between batches if needed
  - Monitor memory usage
- [x] Write unit tests

#### Task 3.4: Embedding Pipeline Integration
- [x] Create main embedding orchestrator
- [x] Integrate components:
  - Load model → Process chunks in batches → Generate embeddings
- [x] Implement checkpointing:
  - Save embeddings periodically
  - Resume from checkpoint if interrupted
- [x] Add embedding validation:
  - Check embedding dimensions (should be 1024)
  - Verify no NaN or Inf values
  - Validate embedding norms
- [x] Save embeddings to disk (numpy format or Parquet)
- [x] Create processing statistics:
  - Total embeddings generated
  - Processing time
  - Throughput (embeddings/sec)
- [x] Write integration tests

---

### Phase 4: Vector Store Setup

#### Task 4.1: ChromaDB Manager
- [x] Create `src/vector_store/chroma_manager.py`
- [x] Implement ChromaDB initialization:
  - Create or connect to persistent database
  - Configure storage path (`data/vector_db/`)
  - Set up collection with correct embedding dimensions (1024)
- [x] Define collection settings:
  - Distance metric: cosine similarity (default for normalized embeddings)
  - Embedding function: custom (we provide embeddings)
- [x] Implement collection management:
  - Create collection
  - Delete collection (for rebuilds)
  - List collections
  - Get collection stats
- [x] Add error handling for database operations
- [x] Write unit tests

#### Task 4.2: Metadata Schema
- [x] Create `src/vector_store/schema.py`
- [x] Define metadata schema:
  ```python
  {
    "video_id": str,
    "date": str,
    "title": str,
    "chunk_index": int,
    "chunk_id": str,
    "token_count": int,
    "filename": str
  }
  ```
- [x] Implement schema validation
- [x] Create schema migration utilities (for future changes)
- [x] Document schema fields

#### Task 4.3: Indexer Implementation
- [x] Create `src/vector_store/indexer.py`
- [x] Implement indexing logic:
  - Accept embeddings and metadata
  - Batch insert into ChromaDB
  - Handle large datasets efficiently
- [x] Implement batch insertion:
  - Optimal batch size for ChromaDB (1000-10000 items)
  - Progress tracking
  - Error handling and retry logic
- [x] Add deduplication:
  - Check for existing chunks
  - Skip or update duplicates
- [x] Implement incremental indexing:
  - Add new embeddings without rebuilding entire index
  - Update existing embeddings if needed
- [x] Add indexing statistics:
  - Total documents indexed
  - Indexing time
  - Index size on disk
- [x] Write unit tests

#### Task 4.4: Vector Store Integration
- [x] Integrate all vector store components
- [x] Create main indexing pipeline:
  - Load embeddings → Validate → Index in ChromaDB
- [x] Implement index verification:
  - Query test samples
  - Verify metadata retrieval
  - Check index integrity
- [x] Add index maintenance utilities:
  - Rebuild index
  - Compact index
  - Backup index
- [x] Write integration tests

#### Task 4.5: Performance Optimization & Hardware Detection
- [x] Create `src/utils/hardware_detector.py`
- [x] Implement hardware detection:
  - Detect CPU cores and threads
  - Detect available RAM
  - Detect GPU availability (CUDA, MPS for Apple Silicon)
  - Detect Apple Silicon chip type (M1/M2/M3, Pro/Max/Ultra)
  - Detect system architecture (x86_64, arm64)
- [x] Create hardware profile system:
  - Generate hardware profile on first run
  - Cache hardware profile
  - Provide hardware info API
- [x] Implement device auto-detection:
  - Support MPS (Metal Performance Shaders) for Apple Silicon
  - Fallback: CUDA → MPS → CPU
  - Log device selection with rationale
- [x] Create `src/utils/performance_optimizer.py`
- [x] Implement dynamic batch size optimization:
  - Auto-tune batch size based on:
    - Available device (CPU/MPS/CUDA)
    - Available RAM
    - Model size and memory requirements
    - Text length distribution
  - Test different batch sizes and select optimal
  - Provide batch size recommendations API
- [x] Implement parallel processing optimization:
  - Auto-detect optimal worker count
  - Balance between CPU cores and memory
  - Support for multi-video parallel processing
  - Thread pool vs Process pool selection
- [x] Add model compilation optimization:
  - Use `torch.compile()` for PyTorch 2.0+
  - Compile model once and cache compiled version
  - Fallback gracefully if compilation fails
- [x] Implement memory management optimization:
  - Dynamic memory threshold detection
  - Adaptive cache clearing frequency
  - Memory-efficient batch processing
  - Monitor memory usage and adjust strategies
- [ ] Add prefetching and async processing (optional, future enhancement):
  - Prefetch next batch while processing current
  - Async I/O for file operations
  - Pipeline parallelism for preprocessing → embedding → indexing
- [x] Create performance benchmarking:
  - Benchmark different configurations
  - Measure throughput (chunks/sec, embeddings/sec)
  - Measure memory usage
  - Generate performance report
- [x] Integrate optimizations into existing pipelines:
  - Update `EmbeddingPipeline` to use hardware detection
  - Update `BatchProcessor` to use dynamic batch sizing
  - Update `ModelLoader` to support MPS and compilation
  - Update `Config` to use hardware-aware defaults
- [x] Add performance monitoring:
  - Track processing times per stage
  - Monitor resource utilization
  - Log performance metrics
  - Provide performance statistics API
- [x] Create optimization configuration:
  - Allow manual override of auto-detected settings
  - Provide optimization presets (fast, balanced, memory-efficient)
  - Environment variables for fine-tuning
- [x] Write unit tests for hardware detection
- [x] Write unit tests for performance optimization
- [ ] Write integration tests with different hardware profiles (optional)
- [ ] Document optimization strategies and recommendations (optional)

#### Task 4.6: Vector Database Visualization Tools
- [x] Create `scripts/view_vector_db.py` CLI viewer:
  - Display database statistics (total documents, unique videos, avg chunks per video)
  - Show date distribution
  - Display sample documents with metadata
  - Implement semantic search functionality
  - Add command-line arguments for search queries and result count
- [x] Create `streamlit_app.py` web interface:
  - Interactive dashboard for exploring vector database
  - Real-time statistics display
  - Date distribution visualization (bar charts)
  - Semantic search interface with results display
  - Document browser with filtering
  - Video list with metadata
- [x] Implement 3D embedding visualization:
  - Add dimensionality reduction (UMAP/t-SNE) for 2D/3D projection
  - Create interactive 3D scatter plot using Plotly
  - Add color coding by video ID or date
  - Implement hover tooltips with document metadata
  - Add controls for number of points, reduction method, and color scheme
  - Export 3D coordinates to CSV
- [x] Add dependencies for visualization:
  - `streamlit>=1.28.0` for web UI
  - `plotly>=5.17.0` for interactive charts
  - `umap-learn>=0.5.5` for UMAP dimensionality reduction
  - `scikit-learn>=1.3.0` for t-SNE dimensionality reduction
- [x] Create `README_STREAMLIT.md` documentation:
  - Installation instructions
  - Launch instructions
  - Feature descriptions
  - Usage examples
- [x] Integrate visualization tools into project structure
- [x] Test visualization with real data from processed subtitles

---

### Phase 5: Retrieval System

> **📚 Documentazione di Riferimento**: Per una analisi tecnica approfondita del sistema di clustering, vedere [`docs/clustering_report.md`](docs/clustering_report.md). Il report include analisi comparativa degli algoritmi, architettura dettagliata, considerazioni di performance, e roadmap completa.

#### Task 5.1: Similarity Search Implementation
- [x] Create `src/retrieval/similarity_search.py`
- [x] Implement core search functionality:
  - Accept query text
  - Generate query embedding using same model
  - Perform similarity search in ChromaDB
  - Return top-K results
- [x] Add search parameters:
  - `top_k`: Number of results (default: 10)
  - `score_threshold`: Minimum similarity score
  - `include_metadata`: Return metadata with results
- [x] Implement filtering:
  - Filter by video_id
  - Filter by date range
  - Filter by title keywords
- [x] Add search result ranking:
  - Sort by similarity score
  - Handle ties appropriately
- [x] Write unit tests

#### Task 5.2: Query Engine
- [x] Create `src/retrieval/query_engine.py`
- [x] Implement main query interface:
  - Accept natural language queries
  - Handle query preprocessing
  - Generate query embedding with instruction prefix
  - Execute similarity search
  - Format results
- [x] Add query expansion (optional):
  - Synonym expansion
  - Related terms
- [x] Implement query validation:
  - Check query length
  - Validate input format
- [x] Add query caching (optional):
  - Cache frequent queries
  - Improve response time
- [x] Write unit tests

#### Task 5.3: Result Formatter
- [x] Create `src/retrieval/result_formatter.py`
- [x] Implement result formatting:
  - Format similarity scores (as percentages or decimals)
  - Format metadata (date, title, video info)
  - Highlight relevant text snippets
  - Create readable output
- [x] Add multiple output formats:
  - Human-readable text
  - JSON format
  - Markdown format
- [x] Implement context expansion:
  - Include surrounding chunks for context
  - Show video timeline information
- [x] Add result deduplication:
  - Remove duplicate chunks from same video
  - Merge adjacent chunks
- [x] Write unit tests

#### Task 5.4: Retrieval Pipeline Integration
- [x] Integrate all retrieval components
- [x] Create main retrieval interface
- [x] Add advanced features:
  - Hybrid search (semantic + keyword, future)
  - Reranking (optional, future)
  - Multi-query search (combine multiple queries)
- [x] Implement performance monitoring:
  - Query latency
  - Search quality metrics
- [x] Write integration tests

#### Task 5.5: Clustering System

> **📖 Riferimento Completo**: Questo task implementa il sistema di clustering descritto in dettaglio in [`docs/clustering_report.md`](docs/clustering_report.md). Il report contiene:
> - Analisi comparativa approfondita degli algoritmi di clustering (HDBSCAN, K-Means, DBSCAN, etc.)
> - Architettura completa del sistema con diagrammi
> - Analisi di performance e scalabilità
> - Casi d'uso dettagliati
> - Roadmap di implementazione in 6 fasi
> - Considerazioni su rischi e mitigazioni

- [x] Create `src/clustering/` module structure:
  - `__init__.py`
  - `clusterer.py` - Main clustering interface
  - `hdbscan_clusterer.py` - HDBSCAN implementation
  - `cluster_manager.py` - Cluster storage and management
  - `cluster_evaluator.py` - Cluster quality evaluation
  - `cluster_integrator.py` - Integration with retrieval
- [x] Implement core clustering functionality:
  - Create `Clusterer` base class with unified interface
  - Implement HDBSCAN clusterer with cosine distance support
  - Handle high-dimensional embeddings (1024D)
  - Support soft clustering (probability assignments)
- [x] Configure HDBSCAN parameters:
  - `min_cluster_size`: Configurable (default: 10-20)
  - `min_samples`: Configurable (default: 5-10)
  - `metric`: Cosine distance for normalized embeddings
  - `cluster_selection_method`: 'eom' or 'leaf'
  - Auto-tuning based on dataset characteristics (parameter optimization)
- [x] Implement cluster management:
  - Store cluster labels in ChromaDB metadata (`cluster_id`, `cluster_probability`)
  - Create cluster metadata storage (centroids, keywords, themes)
  - Cluster query and filtering capabilities
  - Cluster statistics and reporting
- [x] Implement cluster evaluation:
  - Silhouette Score calculation
  - Davies-Bouldin Index
  - Calinski-Harabasz Index
  - Semantic coherence analysis (keyword extraction, theme identification)
  - Parameter optimization utilities
- [x] Integrate with retrieval system:
  - Filter queries by cluster_id
  - Cluster-based reranking
  - Query expansion using cluster representatives
  - Cluster-aware search context
- [x] Add dependencies:
  - `hdbscan>=0.8.33` - Primary clustering algorithm
  - Update `requirements.txt`
- [x] Write unit tests:
  - Test clustering algorithms
  - Test cluster management operations
  - Test evaluation metrics
  - Test integration with retrieval

**Clustering Architecture Notes**:
- **Primary Algorithm**: HDBSCAN (optimal for high-dimensional normalized embeddings)
- **Alternative Algorithms**: K-Means (for baseline), DBSCAN (for exploration)
- **Distance Metric**: Cosine distance (appropriate for normalized embeddings)
- **Storage**: Extend ChromaDB metadata schema with cluster information
- **Use Cases**: Thematic discovery, improved retrieval, content exploration, pattern analysis
- **Performance Target**: Clustering complete dataset (< 50K chunks) in < 5 minutes
- **📚 Detailed Analysis**: See [`docs/clustering_report.md`](docs/clustering_report.md) for:
  - Complete algorithm comparison and selection rationale
  - Detailed architecture diagrams and component specifications
  - Performance analysis and optimization strategies
  - Integration patterns with existing system
  - Risk analysis and mitigation strategies
  - Complete implementation roadmap (7-13 days estimated)

---

### Phase 6: CLI Interface

#### Task 6.1: Processing Script
- [x] Create `scripts/process_subtitles.py`
- [x] Implement CLI interface:
  - Accept input directory (subtitles folder)
  - Accept output directory (optional)
  - Configuration options (batch size, chunk size, etc.)
  - Verbose/debug flags
- [x] Add processing workflow:
  - Preprocess all SRT files
  - Generate embeddings
  - Index in vector store
  - Generate processing report
- [x] Implement resume capability:
  - Check for existing processed chunks
  - Skip already processed files
  - Resume from last checkpoint
- [x] Add progress visualization:
  - Rich progress bars
  - Real-time statistics
  - ETA calculations
- [ ] Write CLI tests (optional)

#### Task 6.2: Query Script
- [ ] Create `scripts/query_subtitles.py`
- [ ] Implement query interface:
  - Accept query string (command-line or interactive)
  - Accept search parameters (top_k, filters)
  - Output format options
- [ ] Add interactive mode:
  - REPL-like interface
  - Query history
  - Command shortcuts
- [ ] Implement batch query mode:
  - Process multiple queries from file
  - Generate query results report
- [ ] Add result export:
  - Save results to file (JSON, CSV, Markdown)
  - Export for further analysis
- [ ] Write CLI tests

#### Task 6.3: Index Management Script
- [ ] Create `scripts/rebuild_index.py`
- [ ] Implement index management:
  - Rebuild entire index
  - Update index with new files
  - Delete and recreate index
- [ ] Add index statistics:
  - Show index size
  - Show document count
  - Show index health
- [ ] Implement backup/restore:
  - Backup index to archive
  - Restore from backup
- [ ] Write CLI tests

---

### Phase 7: Testing & Quality Assurance

#### Task 7.1: Unit Tests
- [ ] Write comprehensive unit tests for preprocessing:
  - SRT parser tests
  - Text cleaner tests
  - Chunker tests
  - Metadata extractor tests
- [ ] Write unit tests for embeddings:
  - Model loader tests
  - Embedder tests
  - Batch processor tests
- [ ] Write unit tests for vector store:
  - ChromaDB manager tests
  - Indexer tests
- [ ] Write unit tests for retrieval:
  - Similarity search tests
  - Query engine tests
  - Result formatter tests
- [ ] Achieve >80% code coverage

#### Task 7.2: Integration Tests
- [ ] Write end-to-end integration tests:
  - Full pipeline: SRT → Chunks → Embeddings → Index → Query
  - Test with sample data
  - Verify results correctness
- [ ] Test error handling:
  - Invalid input files
  - Corrupted data
  - Database errors
  - Model loading failures
- [ ] Test performance:
  - Processing speed benchmarks
  - Memory usage tests
  - Query latency tests

#### Task 7.3: Quality Validation
- [ ] Create test queries and expected results
- [ ] Validate retrieval quality:
  - Test semantic similarity
  - Verify relevant results
  - Check false positives/negatives
- [ ] Benchmark against baseline:
  - Compare with simple keyword search
  - Measure improvement
- [ ] Document test results

---

### Phase 8: Documentation & Deployment

#### Task 8.1: Code Documentation
- [ ] Add docstrings to all modules and functions
- [ ] Document function parameters and return values
- [ ] Add usage examples in docstrings
- [ ] Generate API documentation (Sphinx or similar)

#### Task 8.2: User Documentation
- [ ] Create `README.md` with:
  - Project overview
  - Installation instructions
  - Quick start guide
  - Usage examples
- [ ] Create `docs/usage_guide.md`:
  - Detailed usage instructions
  - Configuration options
  - Troubleshooting guide
- [ ] Create `docs/architecture.md`:
  - System architecture diagram
  - Component descriptions
  - Data flow diagrams

#### Task 8.3: Performance Documentation
- [ ] Document performance characteristics:
  - Processing times for different dataset sizes
  - Query latency benchmarks
  - Memory requirements
  - Storage requirements
- [ ] Create optimization guide:
  - Tips for faster processing
  - GPU vs CPU comparison
  - Batch size recommendations

#### Task 8.4: Deployment Preparation
- [ ] Create deployment checklist
- [ ] Document system requirements
- [ ] Create example configuration files
- [ ] Add startup scripts if needed
- [ ] Create troubleshooting guide

---

## Implementation Notes

### Additional Implementations

1. **Vector Database Visualization Tools (Phase 4.6)**: 
   - Implemented CLI viewer (`scripts/view_vector_db.py`) for command-line exploration of the vector database
   - Created interactive Streamlit web application (`streamlit_app.py`) with:
     - Real-time statistics and charts
     - Semantic search interface
     - Document and video browser
     - **3D embedding visualization** using UMAP/t-SNE dimensionality reduction and Plotly
     - Export capabilities (CSV download)
   - Added dependencies: `streamlit`, `plotly`, `umap-learn`, `scikit-learn`
   - Created documentation (`README_STREAMLIT.md`)

2. **Processing Script (Task 6.1)**: 
   - Implemented `scripts/process_subtitles.py` ahead of schedule as part of Phase 4 integration
   - Full CLI interface with all required features including checkpointing and progress visualization

3. **Clustering System (Phase 5.5)**: 
   - Planned comprehensive clustering system for thematic discovery and improved retrieval
   - Primary algorithm: HDBSCAN with cosine distance (optimal for high-dimensional normalized embeddings)
   - Features: Incremental clustering, cluster-aware retrieval, visualization integration
   - **📚 Complete Technical Analysis**: See [`docs/clustering_report.md`](docs/clustering_report.md) for:
     - Deep brainstorming on clustering technologies and algorithms
     - Comparative analysis of 7+ clustering algorithms
     - Detailed architecture with 7 core components
     - Performance considerations and optimization strategies
     - Integration patterns with ChromaDB, retrieval, and visualization
     - Complete use cases and application scenarios
     - Risk analysis and mitigation strategies
     - Full implementation roadmap (6 phases, 7-13 days)
   - Will extend ChromaDB metadata schema with cluster information
   - Target: Clustering < 50K chunks in < 5 minutes

### Known Issues

1. **ChromaDB Python Compatibility**: ChromaDB requires Python 3.11 or 3.12. The project is configured to use Python 3.12.7 via pyenv (see `.python-version` file). ChromaDB is fully functional with this setup.

2. ~~**Apple Silicon GPU Not Utilized**: Currently, the system defaults to CPU on Mac even when Apple Silicon GPU (MPS) is available. This will be addressed in Phase 4.5 with hardware auto-detection and MPS support.~~ ✅ **RESOLVED** - Phase 4.5 implemented MPS support with automatic detection.

3. ~~**Batch Size Not Optimized**: Default batch size (128) may be suboptimal for powerful hardware. Phase 4.5 will implement dynamic batch size optimization based on hardware capabilities.~~ ✅ **RESOLVED** - Phase 4.5 implemented dynamic batch size optimization.

### Performance Optimizations

#### Current Optimizations (Implemented)
1. **Batch Processing**: Use large batch sizes (128-512) for embedding generation
2. **Parallel Processing**: Use multiprocessing for preprocessing multiple files
3. **Memory Management**: Process in chunks to avoid memory overflow
4. **Caching**: Cache model and frequently accessed data
5. **Checkpointing**: Save progress periodically to allow resume

#### Implemented Optimizations (Phase 4.5) ✅
1. **Hardware Auto-Detection**: ✅ Automatically detect and utilize available hardware resources
   - Apple Silicon (MPS) support for 5-10x speedup on Mac
   - CUDA GPU detection and utilization
   - CPU core and RAM detection for optimal parallelization
2. **Dynamic Batch Size Optimization**: ✅ Auto-tune batch size based on hardware
   - Apple Silicon: 512-1024 batch size
   - CUDA GPU: 256-512 batch size
   - CPU: 128-256 batch size (scaled by cores)
3. **Model Compilation**: ✅ Use `torch.compile()` for PyTorch 2.0+ (10-30% speedup)
4. **Intelligent Parallelization**: ✅ 
   - Multi-video parallel processing
   - Optimal worker count based on hardware
   - Pipeline parallelism (preprocessing → embedding → indexing)
5. **Prefetching & Async Processing**: ⏳ (Optional - future enhancement)
   - Prefetch next batch while processing current
   - Async I/O operations
   - Reduce idle time between batches
6. **Adaptive Memory Management**: ✅ 
   - Dynamic memory threshold detection
   - Adaptive cache clearing based on available RAM
   - Memory-efficient batch processing strategies
7. **Performance Benchmarking**: ✅ 
   - Auto-benchmark different configurations
   - Select optimal settings per hardware profile
   - Performance monitoring and reporting

#### Visualization Tools (Phase 4.6) ✅
1. **CLI Viewer**: ✅ Command-line tool for exploring vector database
   - Statistics display
   - Semantic search functionality
   - Sample document browsing
2. **Streamlit Web UI**: ✅ Interactive web interface
   - Real-time statistics and charts
   - Semantic search with results display
   - Document and video browser
   - 3D embedding visualization with UMAP/t-SNE
   - Export capabilities (CSV download)

#### Expected Performance Improvements
- **Apple Silicon Mac**: 20-180x improvement (with MPS support)
- **Mac Intel with GPU**: 5-15x improvement (with CUDA)
- **High-end CPU-only**: 2-5x improvement (with optimizations)

### Error Handling Strategy

1. **Graceful Degradation**: Continue processing even if individual files fail
2. **Checkpointing**: Save progress periodically to allow resume
3. **Logging**: Comprehensive logging for debugging
4. **Validation**: Validate inputs at each stage

### Future Enhancements

1. **Reranking**: Add cross-encoder reranking for better results
2. **Hybrid Search**: Combine semantic and keyword search
3. **Fine-tuning**: Fine-tune BGE model on domain-specific data
4. **Multilingual Support**: Add support for multiple languages
5. ~~**Web Interface**: Create web UI for querying~~ ✅ **COMPLETED** - Streamlit web UI implemented with 3D visualization
6. **API Server**: Expose REST API for programmatic access
7. ~~**Advanced Visualization**: Add more visualization options (2D projections, clustering, etc.)~~ ⏳ **IN PROGRESS** - Clustering system being implemented in Phase 5.5
8. **Prefetching & Async Processing**: Implement async I/O and pipeline parallelism for further performance gains

---

## Success Criteria

- [ ] All 602 subtitle files processed successfully
- [ ] Embeddings generated with 1024 dimensions
- [ ] Vector index created and queryable
- [ ] Query latency < 100ms for typical queries
- [ ] Retrieval quality validated with test queries
- [ ] Documentation complete and clear
- [ ] Code coverage > 80%
- [ ] System runs entirely locally with zero external API costs

---

## Timeline Estimate

- **Phase 1**: 1-2 days (Setup & Infrastructure)
- **Phase 2**: 2-3 days (Preprocessing)
- **Phase 3**: 2-3 days (Embeddings)
- **Phase 4**: 1-2 days (Vector Store)
- **Phase 5**: 2-3 days (Retrieval)
- **Phase 6**: 1-2 days (CLI)
- **Phase 7**: 2-3 days (Testing)
- **Phase 8**: 1-2 days (Documentation)

**Total Estimated Time**: 12-20 days (depending on experience and testing depth)

---

## Getting Started

1. Check off tasks as you complete them
2. Update this document with any deviations or learnings
3. Add notes or issues encountered in each phase
4. Mark phases as complete when all tasks are done

