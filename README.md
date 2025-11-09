# Subtitle Embedding & Retrieval System

A semantic search system for YouTube subtitle files that enables natural language queries over video content.

## Overview

This system processes YouTube subtitle files (SRT format), generates high-quality embeddings using the BGE-large-en-v1.5 model, and stores them in a local ChromaDB vector database for fast semantic search.

## Features

- **Preprocessing Pipeline**: Parse SRT files, clean text, and create semantic chunks
- **Embedding Generation**: Generate 1024-dimensional embeddings using BGE-large-en-v1.5
- **Vector Storage**: Store embeddings with metadata in ChromaDB
- **Semantic Search**: Query video content using natural language
- **CLI Interface**: Command-line tools for processing and querying

## Technology Stack

- **Language**: Python 3.9+
- **Embedding Model**: BAAI/bge-large-en-v1.5 (via sentence-transformers)
- **Vector Database**: ChromaDB (local, persistent)
- **Deep Learning**: PyTorch (with CUDA support if GPU available)

## Quick Start

### Installation

1. Clone the repository
2. Create a virtual environment:
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   ```
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

### Usage

1. Process subtitle files:
   ```bash
   python scripts/process_subtitles.py --input subtitles/
   ```

2. Query the indexed content:
   ```bash
   python scripts/query_subtitles.py "your search query"
   ```

## Project Structure

```
project_root/
├── src/                    # Source code
│   ├── preprocessing/      # SRT parsing, text cleaning, chunking
│   ├── embeddings/         # Model loading, embedding generation
│   ├── vector_store/       # ChromaDB management, indexing
│   ├── retrieval/          # Query engine, similarity search
│   └── utils/              # Configuration, logging, utilities
├── scripts/                # CLI scripts
├── data/                   # Data directories
│   ├── raw/               # Original subtitle files
│   ├── processed/         # Processed chunks
│   └── vector_db/        # ChromaDB storage
└── tests/                 # Test suite
```

## Configuration

Copy `.env.example` to `.env` and configure as needed:

```bash
cp .env.example .env
```

## License

[Add your license here]

## Contributing

[Add contribution guidelines here]

