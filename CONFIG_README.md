# Configuration System - Quick Reference

## 🎯 What You Can Do

✅ **Run complete pipeline** – Preprocess → Embed → Index  
✅ **Run partial pipelines** – Only embeddings, only search, only RAG  
✅ **Save/load configurations** – YAML and JSON support  
✅ **Use from CLI or Python** – Same config system everywhere  
✅ **Override via environment** – ENV variables override config files  
✅ **Strongly validated** – Pydantic ensures all configs are correct

---

## 📁 Configuration Files

### Presets (in `config/presets/`)

```
config/presets/
├── full_pipeline.yaml       # All stages: prep → embed → index
├── embeddings_only.yaml     # Skip preprocessing, just embed
├── search_only.yaml         # Skip indexing, just search
├── rag_only.yaml            # Skip search, just RAG/LLM
└── custom_template.yaml     # Template for your config
```

---

## 🚀 Quick Start

### CLI Usage

```bash
# Full pipeline
knowbase load --config config/presets/full_pipeline.yaml --input ./subtitles

# Only embeddings
knowbase load --config config/presets/embeddings_only.yaml --input ./docs

# Only search
knowbase search --config config/presets/search_only.yaml --query "orchids"

# Only RAG
knowbase ask --config config/presets/rag_only.yaml "How to grow orchids?"
```

### Python Usage

```python
from src.utils.config_manager import ConfigManager, get_preset_config
from pathlib import Path

# Option 1: Use preset
config = get_preset_config("full_pipeline")

# Option 2: Load from file
mgr = ConfigManager(config_file=Path("config/presets/full_pipeline.yaml"))

# Option 3: Override specific values
mgr.merge_with_dict({
    "embedding": {"batch_size": 64},
    "ai_search": {"llm_temperature": 0.5}
})

# Get specific configs
embedding_cfg = mgr.get_embedding_config()
rag_cfg = mgr.get_ai_search_config()
```

---

## 🔧 Configuration Sections

| Section           | Purpose           | Key Options                               |
| ----------------- | ----------------- | ----------------------------------------- |
| **embedding**     | Model & hardware  | model_name, device, batch_size            |
| **preprocessing** | Document cleaning | chunk_size, remove_html, lowercase        |
| **vector_store**  | ChromaDB settings | db_path, collection_name, distance_metric |
| **retrieval**     | Search settings   | top_k, similarity_threshold, rerank       |
| **ai_search**     | LLM/RAG settings  | llm_provider, llm_model, temperature      |
| **clustering**    | Analysis settings | min_cluster_size, use_umap                |
| **pipeline**      | What to run       | run_preprocessing, run_embedding, etc     |

---

## 🎛️ Common Customizations

### Change Embedding Model

```yaml
embedding:
  model_name: "google/embeddinggemma-300m" # Faster than BAAI
  batch_size: 64
```

### Use Different LLM

```yaml
ai_search:
  llm_provider: "anthropic"
  llm_model: "claude-3-opus"
```

### Adjust Chunk Size

```yaml
preprocessing:
  chunk_size: 256 # Smaller chunks
  chunk_overlap: 50
```

### Filter Search Results

```yaml
retrieval:
  top_k: 10
  similarity_threshold: 0.5 # Only results > 0.5 similarity
```

---

## 🔐 Environment Variables

Override any config value via environment:

```bash
# Embedding
export MODEL_NAME="google/embeddinggemma-300m"
export BATCH_SIZE=64
export DEVICE=cuda

# LLM/RAG
export LLM_PROVIDER=openai
export LLM_TEMPERATURE=0.7
export OPENAI_API_KEY=sk-...

# Pipeline
export VERBOSE=true
export LOG_LEVEL=DEBUG
```

---

## 📊 Pipeline Execution Examples

### Example 1: Full Pipeline

```yaml
pipeline:
  run_preprocessing: true   ✓
  run_embedding: true       ✓
  run_indexing: true        ✓
  run_retrieval: false
  run_ai_search: false
```

**Use:** Raw documents → Indexed vectors

### Example 2: Search Only

```yaml
pipeline:
  run_preprocessing: false
  run_embedding: false
  run_indexing: false
  run_retrieval: true       ✓
  run_ai_search: false
```

**Use:** Already indexed, just search

### Example 3: RAG Only

```yaml
pipeline:
  run_preprocessing: false
  run_embedding: false
  run_indexing: false
  run_retrieval: false
  run_ai_search: true       ✓
```

**Use:** Ask questions with LLM

---

## 📝 Create Custom Config

1. Copy template:

```bash
cp config/presets/custom_template.yaml config/my_config.yaml
```

2. Edit `config/my_config.yaml`

3. Use it:

```bash
knowbase load --config config/my_config.yaml --input ./docs
```

Or in Python:

```python
mgr = ConfigManager(config_file=Path("config/my_config.yaml"))
```

---

## ✅ Testing Configuration

```bash
python scripts/test_config.py
```

This runs 8 tests to verify the configuration system works correctly.

---

## 📚 Full Documentation

See `docs/CONFIGURATION.md` for detailed documentation with:

- All configuration options explained
- Usage examples for every scenario
- Troubleshooting guide
- Environment variable reference

---

## 💡 Tips

- **Save successful configs** – Keep configs that work for your data
- **Use presets as templates** – Don't start from scratch
- **Version your configs** – `git` them like code
- **Document your choices** – Add comments in YAML
- **Test before production** – Run test with `test_config.py`
