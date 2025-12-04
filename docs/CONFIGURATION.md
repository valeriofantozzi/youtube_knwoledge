# KnowBase Configuration System

## Overview

The KnowBase configuration system enables:

- **Modular pipeline execution** – Run only the parts you need (preprocessing, embeddings, search, RAG)
- **Configuration-as-code** – Save/load configurations from YAML/JSON files
- **CLI + GUI compatibility** – Use the same config from terminal or web interface
- **Environment-aware** – Override config with environment variables
- **Strongly typed** – Pydantic validation for all configuration parameters

---

## Quick Start

### 1. Use a Preset Configuration

Load a preset configuration for your use case:

```python
from src.utils.config_manager import ConfigManager, get_preset_config

# Option A: Use a preset
config = get_preset_config("full_pipeline")  # Or: embeddings_only, search_only, rag_only
mgr = ConfigManager()
mgr.config = config

# Option B: Load from YAML file
mgr = ConfigManager(config_file=Path("config/presets/full_pipeline.yaml"))
```

### 2. Use from CLI (Examples)

```bash
# Full pipeline: preprocessing → embeddings → indexing
knowbase load \
  --config config/presets/full_pipeline.yaml \
  --input ./subtitles

# Only generate embeddings (assumes preprocessed docs)
knowbase load \
  --config config/presets/embeddings_only.yaml \
  --input ./processed_docs

# Only search (assumes indexed documents)
knowbase search \
  --config config/presets/search_only.yaml \
  --query "how to grow orchids"

# Only RAG/AI Search (requires indexed documents + API key)
knowbase ask \
  --config config/presets/rag_only.yaml \
  "What's the secret to fast-growing orchids?"
```

### 3. Override Configuration

```python
from src.utils.config_manager import ConfigManager
from pathlib import Path

# Load base config
mgr = ConfigManager(config_file=Path("config/presets/full_pipeline.yaml"))

# Override specific settings
mgr.merge_with_dict({
    "embedding": {
        "model_name": "google/embeddinggemma-300m",  # Different model
        "batch_size": 64  # Larger batch
    },
    "preprocessing": {
        "chunk_size": 256  # Smaller chunks
    }
})

# Save the modified config
mgr.save_to_file(Path("config/my_config.yaml"))
```

---

## Configuration Files

### Preset Configurations

Located in `config/presets/`:

#### 1. `full_pipeline.yaml`

**Use when:** You have raw documents and want to process everything

```yaml
pipeline:
  run_preprocessing: true
  run_embedding: true
  run_indexing: true
  run_retrieval: false
  run_ai_search: false
  run_clustering: false
```

**CLI Command:**

```bash
knowbase load --config config/presets/full_pipeline.yaml --input ./subtitles
```

#### 2. `embeddings_only.yaml`

**Use when:** Documents are already preprocessed, you only need embeddings

```yaml
pipeline:
  run_preprocessing: false # Skip
  run_embedding: true # Generate embeddings
  run_indexing: true # Index into DB
  run_retrieval: false
  run_ai_search: false
  run_clustering: false
```

**CLI Command:**

```bash
knowbase load --config config/presets/embeddings_only.yaml --input ./chunked_docs
```

#### 3. `search_only.yaml`

**Use when:** Documents are indexed, you only need to search

```yaml
pipeline:
  run_preprocessing: false
  run_embedding: false
  run_indexing: false
  run_retrieval: true # Only search
  run_ai_search: false
  run_clustering: false
```

**CLI Command:**

```bash
knowbase search --config config/presets/search_only.yaml --query "orchid care"
```

#### 4. `rag_only.yaml`

**Use when:** You want conversational RAG with LLM

```yaml
pipeline:
  run_preprocessing: false
  run_embedding: false
  run_indexing: false
  run_retrieval: false
  run_ai_search: true # Only RAG

ai_search:
  enabled: true
  llm_provider: "openai"
  llm_model: "gpt-4-mini"
  query_analyzer_enabled: true
  query_rewriter_enabled: true
  show_thinking: true
```

**CLI Command:**

```bash
knowbase ask --config config/presets/rag_only.yaml "How to grow orchids?"
```

#### 5. `custom_template.yaml`

**Use as a starting point for your custom configuration**

Fully documented template with all options explained.

---

## Configuration Structure

### 1. Preprocessing (`preprocessing`)

Controls document cleaning and chunking:

```yaml
preprocessing:
  chunk_size: 512 # Chunk size in tokens
  chunk_overlap: 50 # Overlap between chunks
  min_chunk_size: 50 # Min chunk to keep
  remove_html: true # Strip HTML
  normalize_whitespace: true # Clean spaces
  lowercase: false # Convert to lowercase
  language: "en" # Language for processing
```

### 2. Embedding (`embedding`)

Controls embedding model and hardware:

```yaml
embedding:
  model_name: "BAAI/bge-large-en-v1.5" # Model to use
  device: "auto" # auto|cpu|cuda|mps
  batch_size: 32 # Batch size
  cache_dir: "~/.cache/huggingface" # Model cache
  precision: "fp32" # fp32|fp16|bf16
```

Available models:

- `BAAI/bge-large-en-v1.5` – Best quality, slower
- `google/embeddinggemma-300m` – Fast, good quality
- Any HuggingFace model with sentence-transformers

### 3. Vector Store (`vector_store`)

Controls ChromaDB configuration:

```yaml
vector_store:
  db_path: "./data/vector_db" # Where to store
  collection_name: "documents" # Collection name
  distance_metric: "cosine" # cosine|l2|ip
  persist_directory: true # Save to disk
```

### 4. Retrieval (`retrieval`)

Controls semantic search:

```yaml
retrieval:
  top_k: 5 # Return top 5 results
  similarity_threshold: 0.0 # Min similarity score
  rerank_enabled: false # Re-rank results
  filter_by_metadata: {} # Filter by metadata
```

### 5. AI Search / RAG (`ai_search`)

Controls LLM-powered question answering:

```yaml
ai_search:
  enabled: true
  llm_provider: "openai" # openai|anthropic|groq|azure|ollama
  llm_model: "gpt-4-mini" # Model name
  llm_temperature: 0.7 # Creativity (0-2)
  llm_api_key: null # From env if null
  query_analyzer_enabled: true # Analyze query
  query_rewriter_enabled: true # Rewrite query
  clarification_enabled: true # Ask if ambiguous
  show_thinking: true # Display reasoning
```

**LLM Provider Options:**

- **openai** (requires OPENAI_API_KEY)
- **anthropic** (requires ANTHROPIC_API_KEY)
- **groq** (requires GROQ_API_KEY)
- **azure** (requires AZURE_OPENAI_API_KEY + AZURE_OPENAI_ENDPOINT)
- **ollama** (local, requires running ollama service)

### 6. Clustering (`clustering`)

Controls embedding analysis and visualization:

```yaml
clustering:
  enabled: true
  min_cluster_size: 5 # Min documents per cluster
  clustering_metric: "cosine"
  use_umap: true # Dimensionality reduction
  umap_n_components: 3 # 2D or 3D visualization
```

### 7. Pipeline (`pipeline`)

Controls which pipelines run and execution settings:

```yaml
pipeline:
  run_preprocessing: true
  run_embedding: true
  run_indexing: true
  run_retrieval: false
  run_ai_search: false
  run_clustering: false

  skip_existing: false # Don't re-process
  parallel_processing: true # Use multiple cores
  verbose: false # Debug logging
  log_level: "INFO" # DEBUG|INFO|WARNING|ERROR
```

---

## Usage Examples

### Example 1: Process New Documents (Full Pipeline)

```python
from src.utils.config_manager import ConfigManager
from pathlib import Path

# Load full pipeline config
mgr = ConfigManager(config_file=Path("config/presets/full_pipeline.yaml"))

# View configuration
print(mgr.to_yaml())

# Get specific configs
emb_config = mgr.get_embedding_config()
prep_config = mgr.get_preprocessing_config()

# Use in pipeline
from src.preprocessing.pipeline import PreprocessingPipeline
from src.embeddings.pipeline import EmbeddingPipeline
from src.vector_store.pipeline import VectorStorePipeline

prep = PreprocessingPipeline(config=prep_config)
docs = prep.process_multiple_files("./subtitles")

emb = EmbeddingPipeline(config=emb_config)
embeddings = emb.generate_embeddings(docs)

vs = VectorStorePipeline(config=mgr.get_vector_store_config())
vs.index_processed_documents(docs, embeddings)
```

### Example 2: Search Indexed Documents

```python
from src.utils.config_manager import ConfigManager
from pathlib import Path

# Load search-only config
mgr = ConfigManager(config_file=Path("config/presets/search_only.yaml"))

# Override search parameters
mgr.merge_with_dict({
    "retrieval": {
        "top_k": 10,
        "similarity_threshold": 0.5
    }
})

# Use for searching
from src.retrieval.pipeline import RetrievalPipeline

retrieval = RetrievalPipeline(config=mgr.get_retrieval_config())
results = retrieval.search("how to grow orchids")

for result in results:
    print(f"Score: {result.score}")
    print(f"Text: {result.text}")
```

### Example 3: RAG with Multiple LLM Options

```python
from src.utils.config_manager import ConfigManager
from pathlib import Path

# Load RAG config
mgr = ConfigManager(config_file=Path("config/presets/rag_only.yaml"))

# Switch to different LLM
mgr.merge_with_dict({
    "ai_search": {
        "llm_provider": "anthropic",
        "llm_model": "claude-3-5-sonnet"
    }
})

# Or use local Ollama
mgr.merge_with_dict({
    "ai_search": {
        "llm_provider": "ollama",
        "llm_model": "mistral"
    }
})

# Run RAG
from src.ai_search.graph import build_graph

graph = build_graph(config=mgr.get_ai_search_config())
answer = graph.run("What's the secret to growing orchids?")
```

---

## Environment Variable Overrides

Any configuration value can be overridden via environment variables:

```bash
# Embedding config
export MODEL_NAME="google/embeddinggemma-300m"
export BATCH_SIZE=64
export DEVICE=cuda

# Preprocessing
export CHUNK_SIZE=256
export CHUNK_OVERLAP=100

# Vector store
export VECTOR_DB_PATH="/custom/path/vector_db"

# LLM/RAG
export LLM_PROVIDER=openai
export LLM_MODEL=gpt-4
export LLM_TEMPERATURE=0.5
export OPENAI_API_KEY=sk-...

# Pipeline
export VERBOSE=true
export LOG_LEVEL=DEBUG

# Load config (env vars will override file values)
mgr = ConfigManager(config_file=Path("config/presets/full_pipeline.yaml"))
```

---

## Saving Custom Configurations

```python
from src.utils.config_manager import ConfigManager
from pathlib import Path

# Create from scratch
mgr = ConfigManager()

# Customize
mgr.merge_with_dict({
    "embedding": {
        "model_name": "google/embeddinggemma-300m",
        "batch_size": 64
    },
    "ai_search": {
        "llm_provider": "anthropic",
        "llm_model": "claude-3-opus"
    }
})

# Save to file
mgr.save_to_file(Path("config/my_custom_config.yaml"))

# Later, reload
mgr2 = ConfigManager(config_file=Path("config/my_custom_config.yaml"))
```

---

## CLI Integration

The CLI uses the configuration system for all commands:

```bash
# Load with custom config
knowbase load \
  --config config/my_config.yaml \
  --input ./documents

# Override via CLI flags
knowbase load \
  --config config/my_config.yaml \
  --input ./documents \
  --model "google/embeddinggemma-300m" \
  --batch-size 64 \
  --device cuda

# Environment overrides
export BATCH_SIZE=128
knowbase load --config config/my_config.yaml --input ./documents
```

---

## Troubleshooting

### Config not loading

- Check file path is correct
- Use absolute paths: `Path("/full/path/config.yaml").absolute()`
- Verify YAML/JSON syntax

### Environment variables not working

- Use uppercase names: `MODEL_NAME`, not `model_name`
- For nested configs, use section prefix: `AI_SEARCH_TEMPERATURE`
- Check with: `printenv | grep MODEL`

### Validation errors

- Review Pydantic error messages carefully
- Check field types (int, str, bool, etc.)
- Review preset YAML files for correct format

---

## Next Steps

1. Copy `config/presets/custom_template.yaml` to create your config
2. Adjust values for your use case
3. Test with CLI: `knowbase load --config config/my_config.yaml --input ./docs`
4. Save successful configs for future use
