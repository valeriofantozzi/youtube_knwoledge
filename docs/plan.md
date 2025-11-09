# Subtitle Embedding & Retrieval System - Implementation Plan

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
│   └── utils/
│       ├── __init__.py
│       ├── config.py              # Configuration management
│       ├── logger.py              # Logging setup
│       └── progress.py            # Progress tracking utilities
│
├── scripts/
│   ├── process_subtitles.py      # Main script: process all subtitles
│   ├── query_subtitles.py         # Query interface script
│   └── rebuild_index.py           # Rebuild index from scratch
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
- [ ] Create directory structure as defined above
- [ ] Initialize Python package structure (`src/` with `__init__.py` files)
- [ ] Create `.gitignore` file (exclude `data/vector_db/`, `__pycache__/`, `.venv/`, etc.)
- [ ] Create `README.md` with project overview
- [ ] Create `.env.example` for configuration template

#### Task 1.2: Setup Python Environment
- [ ] Create virtual environment (`.venv`)
- [ ] Create `requirements.txt` with dependencies:
  - `sentence-transformers>=2.2.0`
  - `torch>=2.0.0` (with CUDA if GPU available)
  - `chromadb>=0.4.0`
  - `pandas>=2.0.0`
  - `numpy>=1.24.0`
  - `rich>=13.0.0` (for CLI output)
  - `python-dotenv>=1.0.0`
  - `tqdm>=4.65.0` (progress bars)
- [ ] Install dependencies in virtual environment
- [ ] Verify installation and imports

#### Task 1.3: Configuration System
- [ ] Create `src/utils/config.py` with configuration management
- [ ] Define configuration schema:
  - Model name and path
  - Batch size for embeddings
  - Chunk size and overlap
  - Vector database path
  - Device (CPU/GPU)
  - Logging level
- [ ] Implement environment variable support
- [ ] Create default configuration file
- [ ] Add configuration validation

#### Task 1.4: Logging Infrastructure
- [ ] Create `src/utils/logger.py`
- [ ] Setup structured logging with levels (DEBUG, INFO, WARNING, ERROR)
- [ ] Configure log file rotation
- [ ] Add colored console output for better UX
- [ ] Integrate logging across all modules

---

### Phase 2: Preprocessing Pipeline

#### Task 2.1: SRT Parser Implementation
- [ ] Create `src/preprocessing/srt_parser.py`
- [ ] Implement parser to read SRT files:
  - Parse sequence numbers
  - Extract timestamps (start/end)
  - Extract subtitle text
  - Handle multi-line subtitles
- [ ] Handle edge cases:
  - Empty subtitles
  - Malformed timestamps
  - Encoding issues (UTF-8, UTF-8-BOM)
- [ ] Add unit tests with sample SRT files
- [ ] Implement batch processing for multiple files

#### Task 2.2: Metadata Extractor
- [ ] Create `src/preprocessing/metadata_extractor.py`
- [ ] Implement filename parser:
  - Extract date (YYYYMMDD format)
  - Extract video ID (11-character YouTube ID)
  - Extract title (remaining text)
  - Handle edge cases (missing parts, special characters)
- [ ] Create metadata structure:
  ```python
  {
    "video_id": str,
    "date": str (YYYY/MM/DD),
    "title": str,
    "filename": str,
    "file_path": str
  }
  ```
- [ ] Add validation for extracted metadata
- [ ] Write unit tests

#### Task 2.3: Text Cleaner
- [ ] Create `src/preprocessing/text_cleaner.py`
- [ ] Implement text cleaning functions:
  - Remove SRT formatting artifacts
  - Remove `[Music]`, `[Applause]` tags
  - Normalize whitespace
  - Remove HTML entities if present
  - Handle special characters
- [ ] Preserve sentence boundaries
- [ ] Add language detection (optional, for future multilingual support)
- [ ] Write unit tests

#### Task 2.4: Semantic Chunker
- [ ] Create `src/preprocessing/chunker.py`
- [ ] Implement chunking strategy:
  - **Method**: Sentence-based with semantic grouping
  - **Chunk size**: 200-500 tokens (configurable)
  - **Overlap**: 20% between chunks (configurable)
  - **Min chunk size**: 50 tokens (avoid tiny chunks)
- [ ] Group related sentences together:
  - Use sentence boundaries
  - Consider paragraph breaks
  - Maintain context continuity
- [ ] Create chunk structure:
  ```python
  {
    "chunk_id": str (unique),
    "text": str,
    "chunk_index": int (position in video),
    "token_count": int,
    "metadata": dict (video metadata)
  }
  ```
- [ ] Implement batch chunking for multiple videos
- [ ] Add progress tracking
- [ ] Write unit tests

#### Task 2.5: Preprocessing Pipeline Integration
- [ ] Create main preprocessing orchestrator
- [ ] Integrate all preprocessing components:
  - SRT parser → Text cleaner → Chunker → Metadata attachment
- [ ] Implement parallel processing for multiple files
- [ ] Add error handling and recovery
- [ ] Save processed chunks to disk (JSON or Parquet format)
- [ ] Create preprocessing statistics report:
  - Total chunks created
  - Average chunk size
  - Processing time
- [ ] Write integration tests

---

### Phase 3: Embedding Generation

#### Task 3.1: Model Loader
- [ ] Create `src/embeddings/model_loader.py`
- [ ] Implement model loading:
  - Download `BAAI/bge-large-en-v1.5` from HuggingFace
  - Cache model locally
  - Load with appropriate device (CPU/GPU)
- [ ] Handle model initialization:
  - Set model to eval mode
  - Configure normalization (BGE models use normalized embeddings)
  - Set max sequence length
- [ ] Add model info logging:
  - Model name and version
  - Device used
  - Model parameters count
- [ ] Implement model health check
- [ ] Write unit tests

#### Task 3.2: Embedding Generator
- [ ] Create `src/embeddings/embedder.py`
- [ ] Implement embedding generation:
  - Accept list of text chunks
  - Generate embeddings using BGE model
  - Return numpy arrays or torch tensors
- [ ] Handle instruction-based queries (BGE supports this):
  - For queries: prepend "Represent this sentence for searching relevant passages: "
  - For documents: use as-is or prepend "Represent this sentence: "
- [ ] Implement normalization (BGE embeddings should be normalized)
- [ ] Add input validation:
  - Check text length
  - Handle empty strings
  - Validate input types
- [ ] Write unit tests

#### Task 3.3: Batch Processor
- [ ] Create `src/embeddings/batch_processor.py`
- [ ] Implement efficient batch processing:
  - **Batch size**: 64-128 for CPU, 256-512 for GPU (configurable)
  - Dynamic batching based on text length
  - Memory-efficient processing for large datasets
- [ ] Add progress tracking:
  - Use tqdm for progress bars
  - Log processing speed (chunks/sec)
  - Estimate remaining time
- [ ] Implement error handling:
  - Retry failed batches
  - Skip problematic chunks with logging
  - Continue processing on errors
- [ ] Add memory management:
  - Clear cache between batches if needed
  - Monitor memory usage
- [ ] Write unit tests

#### Task 3.4: Embedding Pipeline Integration
- [ ] Create main embedding orchestrator
- [ ] Integrate components:
  - Load model → Process chunks in batches → Generate embeddings
- [ ] Implement checkpointing:
  - Save embeddings periodically
  - Resume from checkpoint if interrupted
- [ ] Add embedding validation:
  - Check embedding dimensions (should be 1024)
  - Verify no NaN or Inf values
  - Validate embedding norms
- [ ] Save embeddings to disk (numpy format or Parquet)
- [ ] Create processing statistics:
  - Total embeddings generated
  - Processing time
  - Throughput (embeddings/sec)
- [ ] Write integration tests

---

### Phase 4: Vector Store Setup

#### Task 4.1: ChromaDB Manager
- [ ] Create `src/vector_store/chroma_manager.py`
- [ ] Implement ChromaDB initialization:
  - Create or connect to persistent database
  - Configure storage path (`data/vector_db/`)
  - Set up collection with correct embedding dimensions (1024)
- [ ] Define collection settings:
  - Distance metric: cosine similarity (default for normalized embeddings)
  - Embedding function: custom (we provide embeddings)
- [ ] Implement collection management:
  - Create collection
  - Delete collection (for rebuilds)
  - List collections
  - Get collection stats
- [ ] Add error handling for database operations
- [ ] Write unit tests

#### Task 4.2: Metadata Schema
- [ ] Create `src/vector_store/schema.py`
- [ ] Define metadata schema:
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
- [ ] Implement schema validation
- [ ] Create schema migration utilities (for future changes)
- [ ] Document schema fields

#### Task 4.3: Indexer Implementation
- [ ] Create `src/vector_store/indexer.py`
- [ ] Implement indexing logic:
  - Accept embeddings and metadata
  - Batch insert into ChromaDB
  - Handle large datasets efficiently
- [ ] Implement batch insertion:
  - Optimal batch size for ChromaDB (1000-10000 items)
  - Progress tracking
  - Error handling and retry logic
- [ ] Add deduplication:
  - Check for existing chunks
  - Skip or update duplicates
- [ ] Implement incremental indexing:
  - Add new embeddings without rebuilding entire index
  - Update existing embeddings if needed
- [ ] Add indexing statistics:
  - Total documents indexed
  - Indexing time
  - Index size on disk
- [ ] Write unit tests

#### Task 4.4: Vector Store Integration
- [ ] Integrate all vector store components
- [ ] Create main indexing pipeline:
  - Load embeddings → Validate → Index in ChromaDB
- [ ] Implement index verification:
  - Query test samples
  - Verify metadata retrieval
  - Check index integrity
- [ ] Add index maintenance utilities:
  - Rebuild index
  - Compact index
  - Backup index
- [ ] Write integration tests

---

### Phase 5: Retrieval System

#### Task 5.1: Similarity Search Implementation
- [ ] Create `src/retrieval/similarity_search.py`
- [ ] Implement core search functionality:
  - Accept query text
  - Generate query embedding using same model
  - Perform similarity search in ChromaDB
  - Return top-K results
- [ ] Add search parameters:
  - `top_k`: Number of results (default: 10)
  - `score_threshold`: Minimum similarity score
  - `include_metadata`: Return metadata with results
- [ ] Implement filtering:
  - Filter by video_id
  - Filter by date range
  - Filter by title keywords
- [ ] Add search result ranking:
  - Sort by similarity score
  - Handle ties appropriately
- [ ] Write unit tests

#### Task 5.2: Query Engine
- [ ] Create `src/retrieval/query_engine.py`
- [ ] Implement main query interface:
  - Accept natural language queries
  - Handle query preprocessing
  - Generate query embedding with instruction prefix
  - Execute similarity search
  - Format results
- [ ] Add query expansion (optional):
  - Synonym expansion
  - Related terms
- [ ] Implement query validation:
  - Check query length
  - Validate input format
- [ ] Add query caching (optional):
  - Cache frequent queries
  - Improve response time
- [ ] Write unit tests

#### Task 5.3: Result Formatter
- [ ] Create `src/retrieval/result_formatter.py`
- [ ] Implement result formatting:
  - Format similarity scores (as percentages or decimals)
  - Format metadata (date, title, video info)
  - Highlight relevant text snippets
  - Create readable output
- [ ] Add multiple output formats:
  - Human-readable text
  - JSON format
  - Markdown format
- [ ] Implement context expansion:
  - Include surrounding chunks for context
  - Show video timeline information
- [ ] Add result deduplication:
  - Remove duplicate chunks from same video
  - Merge adjacent chunks
- [ ] Write unit tests

#### Task 5.4: Retrieval Pipeline Integration
- [ ] Integrate all retrieval components
- [ ] Create main retrieval interface
- [ ] Add advanced features:
  - Hybrid search (semantic + keyword, future)
  - Reranking (optional, future)
  - Multi-query search (combine multiple queries)
- [ ] Implement performance monitoring:
  - Query latency
  - Search quality metrics
- [ ] Write integration tests

---

### Phase 6: CLI Interface

#### Task 6.1: Processing Script
- [ ] Create `scripts/process_subtitles.py`
- [ ] Implement CLI interface:
  - Accept input directory (subtitles folder)
  - Accept output directory (optional)
  - Configuration options (batch size, chunk size, etc.)
  - Verbose/debug flags
- [ ] Add processing workflow:
  - Preprocess all SRT files
  - Generate embeddings
  - Index in vector store
  - Generate processing report
- [ ] Implement resume capability:
  - Check for existing processed chunks
  - Skip already processed files
  - Resume from last checkpoint
- [ ] Add progress visualization:
  - Rich progress bars
  - Real-time statistics
  - ETA calculations
- [ ] Write CLI tests

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

### Performance Optimizations

1. **Batch Processing**: Use large batch sizes (128-512) for embedding generation
2. **Parallel Processing**: Use multiprocessing for preprocessing multiple files
3. **GPU Acceleration**: Utilize GPU if available for embedding generation
4. **Memory Management**: Process in chunks to avoid memory overflow
5. **Caching**: Cache model and frequently accessed data

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
5. **Web Interface**: Create web UI for querying
6. **API Server**: Expose REST API for programmatic access

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

