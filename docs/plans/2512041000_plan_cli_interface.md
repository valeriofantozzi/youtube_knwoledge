# Implementation Plan: CLI Interface for KnowBase

**Plan ID:** `2512041000_plan_cli_interface`  
**Created:** 2025-12-04  
**Status:** Draft  
**Last Updated:** 2025-12-04

---

## Technical & Engineering Description

### Overview

**What is being built:**

A complete **Command-Line Interface (CLI)** for KnowBase that enables users to interact with the knowledge base platform from the terminal, complementing the existing Streamlit web GUI. The CLI will expose all major operations currently available only in the web interface (and scripts), making the platform accessible to:

- **Developers** and **data engineers** who prefer terminal workflows
- **Automation scripts** and **batch processing** pipelines
- **DevOps/MLOps** environments (Docker, CI/CD, cloud functions)
- **API consumers** programmatically (through Python imports)

**Key Operations Exposed:**

1. **`knowbase load`** – Load documents, preprocess, generate embeddings, index into vector DB
2. **`knowbase search`** – Semantic search with filters, output in multiple formats
3. **`knowbase ask`** – Conversational RAG queries with thinking display
4. **`knowbase cluster`** – Analyze embeddings with clustering and dimensionality reduction
5. **`knowbase export`** – Export collections to JSON/CSV
6. **`knowbase info`** – Show database statistics, available models, hardware info
7. **`knowbase reindex`** – Reindex existing documents with a different model

**Business/Functional Goals:**

- Provide **command-line accessibility** to all core features
- Enable **batch processing** and **automation** workflows
- Support **easy integration** with shell scripts, CI/CD pipelines, Docker
- Maintain **backward compatibility** with existing web UI and scripts
- Prepare project for **distribution** (PyPI pip install)

---

### Architecture

**Current State (Multi-Interface):**

```
User Input
   │
   ├─→ Streamlit Web UI (streamlit_app.py)
   │       └─→ Session state → Pages (Load, Search, PostProcessing, AI Search)
   │
   ├─→ CLI Scripts (scripts/)
   │       ├─→ process_subtitles.py
   │       ├─→ test_ai_search.py
   │       └─→ view_vector_db.py
   │
   └─→ Python API (direct imports)
           └─→ from src.embeddings.pipeline import EmbeddingPipeline


Backend Pipelines (src/)
   ├─→ preprocessing.pipeline
   ├─→ embeddings.pipeline
   ├─→ vector_store.pipeline
   ├─→ retrieval.pipeline
   └─→ ai_search.graph

ChromaDB (persistent)
   └─→ data/vector_db/
```

**Target State (Unified CLI + Web):**

```
User Input
   │
   ├─→ Streamlit Web UI (streamlit_app.py)
   │       └─→ Page renderers
   │
   ├─→ CLI Interface (NEW: src/cli/)
   │       ├─→ main.py (Click CLI group)
   │       ├─→ commands/
   │       │   ├─→ load.py
   │       │   ├─→ search.py
   │       │   ├─→ ask.py
   │       │   ├─→ cluster.py
   │       │   ├─→ export.py
   │       │   ├─→ info.py
   │       │   └─→ reindex.py
   │       └─→ utils/
   │           ├─→ formatters.py (output formatting: text, JSON, CSV)
   │           ├─→ progress.py (progress bars, spinners)
   │           └─→ validators.py (input validation)
   │
   └─→ Python API (same imports)


Shared Backend Pipelines (src/)
   └─→ [unchanged from current]
```

**Design Patterns:**

- **Command Pattern**: Each CLI command as a separate module (Click decorators)
- **Factory Pattern**: Output formatters (TextFormatter, JSONFormatter, CSVFormatter)
- **Strategy Pattern**: Progress reporters (RichProgress, TqdmProgress)
- **Validation Layer**: Pydantic models for input validation before business logic
- **Single Responsibility**: CLI layer handles parsing/output, pipelines handle business logic

**Integration Points:**

1. **Existing Pipelines**: Directly import and reuse (no duplication)
2. **Configuration**: Extend `utils/config.py` to support CLI flags + env vars + config files
3. **Logging**: Use existing logger from `utils/logger.py`
4. **Progress Tracking**: Integrate Rich/tqdm with existing progress callbacks
5. **Error Handling**: Consistent error messages and exit codes (0 success, 1+ failure)

---

### Components & Modules

**1. CLI Core (`src/cli/`)**

| Component               | Responsibility                                       | Dependencies                                      |
| ----------------------- | ---------------------------------------------------- | ------------------------------------------------- |
| **main.py**             | Click CLI group, command routing, global options     | click, utils.config                               |
| **commands/load.py**    | Document loading, preprocessing, embedding, indexing | preprocessing, embeddings, vector_store pipelines |
| **commands/search.py**  | Semantic search with filters, result formatting      | retrieval.pipeline, formatters                    |
| **commands/ask.py**     | Conversational RAG queries                           | ai_search.graph, thinking.py                      |
| **commands/cluster.py** | Clustering analysis, UMAP visualization export       | clustering, utils.embeddings_loader               |
| **commands/export.py**  | Export collections to JSON/CSV                       | vector_store.chroma_manager, formatters           |
| **commands/info.py**    | Database stats, model info, hardware profile         | vector_store, utils.hardware_detector             |
| **commands/reindex.py** | Reindex documents with different model               | preprocessing, embeddings, vector_store           |

**2. CLI Utilities (`src/cli/utils/`)**

| Module            | Purpose                                  | Exports                                                           |
| ----------------- | ---------------------------------------- | ----------------------------------------------------------------- |
| **formatters.py** | Output formatting (text, JSON, CSV)      | `BaseFormatter`, `TextFormatter`, `JSONFormatter`, `CSVFormatter` |
| **progress.py**   | Progress bars, spinners, status messages | `ProgressReporter`, `RichProgress`, `TqdmProgress`                |
| **validators.py** | Input validation + custom Click types    | `ValidatePath`, `ValidateModel`, `ValidateTopK`                   |
| **output.py**     | Rich console + styling                   | `console`, `print_success()`, `print_error()`, `print_table()`    |

**3. Configuration Extension (`utils/config.py`)**

Extend to support:

- CLI flag overrides (`--model`, `--device`, `--batch-size`)
- Environment variable fallbacks
- Config file loading (YAML/TOML)
- Validation via Pydantic

---

### Data Models & Schemas

**Input Validation Models (src/cli/validators.py):**

```python
class LoadCommand(BaseModel):
    input_path: Path = Field(..., description="Input file/directory")
    output_dir: Optional[Path] = Field(None, description="Output directory for embeddings")
    model: str = Field(default="BAAI/bge-large-en-v1.5", pattern="^[a-zA-Z0-9/-]+$")
    device: str = Field(default="auto", pattern="^(auto|cpu|cuda|mps)$")
    batch_size: int = Field(default=32, ge=1, le=256)
    chunk_size: int = Field(default=512, ge=64)
    chunk_overlap: int = Field(default=50, ge=0)

class SearchCommand(BaseModel):
    query: str = Field(..., min_length=1, max_length=2000)
    model: str = Field(default="BAAI/bge-large-en-v1.5")
    top_k: int = Field(default=5, ge=1, le=50)
    threshold: float = Field(default=0.0, ge=0.0, le=1.0)
    filters: Optional[Dict[str, str]] = None

class AskCommand(BaseModel):
    question: str = Field(..., min_length=1, max_length=2000)
    model: str = Field(default="BAAI/bge-large-en-v1.5")
    top_k: int = Field(default=5, ge=1, le=50)
    llm_provider: str = Field(default="openai", pattern="^(openai|anthropic|ollama)$")
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)
    show_thinking: bool = Field(default=True)

class ExportCommand(BaseModel):
    output_format: str = Field(default="json", pattern="^(json|csv)$")
    output_file: Path = Field(..., description="Output file path")
    model: str = Field(default="BAAI/bge-large-en-v1.5")
    include_embeddings: bool = Field(default=False)
```

---

### API Contracts (CLI Commands)

**1. Load Command**

```bash
$ knowbase load --input ./subtitles --model BAAI/bge-large-en-v1.5 --batch-size 32 --device auto

# Output:
# ✓ Parsing 15 files... [████████████████] 100%
# ✓ Cleaning text... [████████████████] 100%
# ✓ Chunking into 1,245 chunks... [████████████████] 100%
# ✓ Generating embeddings... [████████████████] 100%
# ✓ Indexing into ChromaDB... [████████████████] 100%
# ✓ Success! Indexed 1,245 documents in 125.5s
```

**2. Search Command**

```bash
$ knowbase search --query "how to grow orchids" --top-k 5 --output-format text

# Output:
# Query: "how to grow orchids"
# Model: BAAI/bge-large-en-v1.5
# Results:
#
# 1. [0.89] "Orchid care tips..." (source: 20231028_*.srt, chunk 45)
# 2. [0.86] "Growing orchids requires..." (source: 20231015_*.srt, chunk 12)
# ...
```

**3. Ask Command (RAG)**

```bash
$ knowbase ask "What's the secret to fast-growing orchids?" --llm-provider openai --show-thinking

# Output:
# 💭 Thinking...
# > Analyzing question: "What's the secret to fast-growing orchids?"
# > Retrieving relevant documents...
# > Generating answer...
#
# Answer:
# Based on the knowledge base, the secrets to fast-growing orchids are:
# 1. Proper watering techniques...
# 2. Nutrient-rich fertilizers...
# ...
#
# Sources:
# - 20231028_*.srt (chunk 45)
# - 20231015_*.srt (chunk 12)
```

**4. Info Command**

```bash
$ knowbase info

# Output:
# KnowBase System Information
# ═══════════════════════════════════════════════════════════════
#
# Database
#   Location: /Users/.../knowbase/data/vector_db
#   Size: 125 MB
#   Models: 2 (BAAI/bge-large-en-v1.5, google/embeddinggemma-300m)
#   Documents: 1,245 (BAAI), 1,200 (google)
#
# Hardware
#   Device: CUDA (GPU)
#   RAM: 16 GB / 64 GB
#   VRAM: 8 GB / 12 GB (NVIDIA RTX 3080)
#
# Models Available
#   ✓ BAAI/bge-large-en-v1.5
#   ✓ google/embeddinggemma-300m
#
# Configuration
#   API Key (OpenAI): [loaded from .env]
#   Device: auto (detected as cuda)
```

---

### Technology Stack

**Core CLI Framework:**

- **Click 8.1.0+** – CLI command routing, argument/option parsing
- **Pydantic 2.0.0+** – Input validation (already in project)
- **Rich 13.0.0+** – Colored output, tables, progress bars

**Output & Formatting:**

- **Rich Console** – Beautiful terminal output
- **Tabulate** – Table formatting
- **json** (stdlib) – JSON output
- **csv** (stdlib) – CSV export

**Progress Tracking:**

- **tqdm 4.65.0+** – Simple progress bars
- **alive-progress 3.0.0+** – Alternative (prettier)

**Utilities:**

- **python-dotenv** (already present) – Environment variables
- **pathlib** (stdlib) – Path handling

**Testing:**

- **pytest 7.0.0+** – Unit tests
- **Click's CliRunner** – CLI testing

---

### Security Considerations

1. **Input Validation**: All user inputs validated via Pydantic before business logic
2. **Path Traversal Prevention**: Use `pathlib.Path` with `.resolve()`, validate against base dirs
3. **Secrets Management**: API keys loaded from `.env` or secure env vars, not hardcoded
4. **Output Sanitization**: Escape special characters in output to prevent injection
5. **File Permissions**: Check read/write permissions before operations
6. **Rate Limiting**: For LLM calls (OpenAI API) via environment-based limits

---

### Performance Requirements

1. **CLI Startup Time**: < 2 seconds (lazy load models)
2. **Search Response**: < 5 seconds for 10k document queries
3. **Batch Loading**: Support 1000+ files with progress indication
4. **Memory**: Keep model in VRAM, stream results for large exports
5. **Progress Reporting**: Update every 1-2 seconds for user feedback

---

### Observability & Logging

1. **Log Levels**: DEBUG, INFO, WARNING, ERROR (configurable via `--verbose` flag)
2. **Log Output**: File (`logs/cli.log`) + console (filtered)
3. **Metrics**: Execution time, memory usage, document counts
4. **Error Messages**: Actionable, non-technical language with suggestions
5. **Audit Trail**: Log all DB modifications (load, delete, reindex)

---

### Backward Compatibility & Migration

1. **Keep existing scripts**: `scripts/process_subtitles.py`, etc. remain functional (deprecated)
2. **Python API unchanged**: `from src.preprocessing import ...` still works
3. **Streamlit UI unchanged**: No changes to `streamlit_app.py` (except optionally calling CLI)
4. **Configuration**: Extend `utils/config.py` without breaking existing code
5. **Data**: No database schema changes needed

---

## Implementation Plan

### Phase 1: Foundation & CLI Framework

1. [ ] **Create CLI package structure**
   - Create `src/cli/` directory
   - Create `src/cli/__init__.py`
   - Create `src/cli/main.py` (Click CLI group)
   - Create `src/cli/commands/` directory with `__init__.py`
   - Create `src/cli/utils/` directory with `__init__.py`

2. [ ] **Implement CLI utilities**
   - [ ] `src/cli/utils/output.py` – Rich console + helper functions
   - [ ] `src/cli/utils/formatters.py` – BaseFormatter, TextFormatter, JSONFormatter, CSVFormatter
   - [ ] `src/cli/utils/progress.py` – ProgressReporter classes
   - [ ] `src/cli/utils/validators.py` – Pydantic models for command inputs

3. [ ] **Extend configuration system**
   - [ ] Update `src/utils/config.py` to support CLI flag overrides
   - [ ] Add methods: `from_cli_args()`, `to_dict()`, `to_json()`
   - [ ] Document all configuration options with types and defaults

4. [ ] **Create main CLI entry point**
   - [ ] `src/cli/main.py` – Click group with global options (`--verbose`, `--config`, `--output-format`)
   - [ ] Help text and version info
   - [ ] Error handling and exit codes

5. [ ] **Write tests for CLI framework**
   - [ ] `tests/cli/test_main.py` – Test CLI routing
   - [ ] `tests/cli/test_formatters.py` – Test output formatters
   - [ ] `tests/cli/test_validators.py` – Test Pydantic input validation

   **Exit Criteria:**
   - CLI group callable, responds to `--help` and `--version`
   - All utilities tested
   - Configuration system extends cleanly

---

### Phase 2: Core Commands (Load, Search, Info)

6. [ ] **Implement `load` command**
   - [ ] `src/cli/commands/load.py` – Command handler
   - [ ] Call `PreprocessingPipeline`, `EmbeddingPipeline`, `VectorStorePipeline`
   - [ ] Stream progress with Rich progress bar
   - [ ] Output summary (documents indexed, time, embeddings generated)
   - [ ] Handle errors gracefully (file not found, OOM, etc.)

7. [ ] **Implement `search` command**
   - [ ] `src/cli/commands/search.py` – Command handler
   - [ ] Call `RetrievalPipeline`
   - [ ] Support output formats: text (pretty-printed), JSON, CSV
   - [ ] Display results with scores and metadata
   - [ ] Support filtering options (if available in retrieval pipeline)

8. [ ] **Implement `info` command**
   - [ ] `src/cli/commands/info.py` – System information
   - [ ] Show database statistics (size, document count, models)
   - [ ] Display hardware info (device, RAM, VRAM)
   - [ ] List available models and their states
   - [ ] Pretty-print with Rich tables

9. [ ] **Write tests for Phase 2 commands**
   - [ ] `tests/cli/test_load.py` – Test load command with mock data
   - [ ] `tests/cli/test_search.py` – Test search with various queries
   - [ ] `tests/cli/test_info.py` – Test info command

   **Exit Criteria:**
   - `knowbase load`, `knowbase search`, `knowbase info` fully functional
   - All commands tested with mock and real data
   - Help text complete and accurate

---

### Phase 3: Advanced Commands (Ask, Cluster, Export)

10. [ ] **Implement `ask` command (RAG)**
    - [ ] `src/cli/commands/ask.py` – Command handler
    - [ ] Call `ai_search.graph.build_graph()`
    - [ ] Stream thinking display (if `--show-thinking`)
    - [ ] Output final answer + sources
    - [ ] Support LLM provider selection (OpenAI, Anthropic, Ollama)

11. [ ] **Implement `cluster` command**
    - [ ] `src/cli/commands/cluster.py` – Command handler
    - [ ] Load embeddings from ChromaDB
    - [ ] Run HDBSCAN clustering
    - [ ] Compute statistics (cluster sizes, silhouette score)
    - [ ] Optionally export UMAP projection to JSON/image
    - [ ] Display cluster summary

12. [ ] **Implement `export` command**
    - [ ] `src/cli/commands/export.py` – Command handler
    - [ ] Export collection to JSON (with or without embeddings)
    - [ ] Export to CSV (documents + metadata, no embeddings for size)
    - [ ] Handle large datasets (stream to avoid OOM)
    - [ ] Show progress for large exports

13. [ ] **Write tests for Phase 3**
    - [ ] `tests/cli/test_ask.py` – Test RAG with mocked LLM
    - [ ] `tests/cli/test_cluster.py` – Test clustering
    - [ ] `tests/cli/test_export.py` – Test export formats

    **Exit Criteria:**
    - All advanced commands tested
    - Large dataset handling verified

---

### Phase 4: Packaging & Distribution

14. [ ] **Create setup entry point**
    - [ ] Update `pyproject.toml` with CLI entry point:
      ```toml
      [project.scripts]
      knowbase = "src.cli.main:cli"
      ```
    - [ ] Test: `pip install -e .` then `knowbase --help`

15. [ ] **Create wrapper script**
    - [ ] `bin/knowbase` – Simple shell wrapper (optional, for convenience)
    - [ ] Make executable

16. [ ] **Update documentation**
    - [ ] Add CLI section to `README.md`
    - [ ] Create `docs/CLI_GUIDE.md` with examples:
      - Basic usage
      - Advanced options
      - Output format examples
      - Troubleshooting
    - [ ] Add inline help to all commands

17. [ ] **Add reindex command** (bonus)
    - [ ] `src/cli/commands/reindex.py`
    - [ ] Reindex documents with a different embedding model

    **Exit Criteria:**
    - CLI installable via pip
    - Full documentation available
    - All commands working end-to-end

---

### Phase 5: Integration & Polish

18. [ ] **Integration with Streamlit UI** (optional)
    - [ ] Consider calling CLI commands from Streamlit for consistency
    - [ ] Or keep separate (simpler approach)

19. [ ] **End-to-end testing**
    - [ ] Create test scenarios covering all commands
    - [ ] Test with real documents
    - [ ] Verify output formats

20. [ ] **Performance optimization**
    - [ ] Lazy load models only when needed
    - [ ] Optimize progress bar performance for large datasets
    - [ ] Profile memory usage

21. [ ] **Error handling & edge cases**
    - [ ] Handle missing API keys gracefully
    - [ ] Handle corrupted ChromaDB
    - [ ] Handle out-of-memory conditions
    - [ ] Provide helpful error messages

    **Exit Criteria:**
    - All edge cases tested
    - Error messages clear and actionable
    - No crashes on invalid input

---

### Phase 6: Deprecation & Cleanup (Future)

22. [ ] **Deprecate old scripts** (after CLI is stable)
    - [ ] Add deprecation warnings to `scripts/process_subtitles.py`, etc.
    - [ ] Redirect users to `knowbase load` command
    - [ ] Plan removal in next major version

---

## Testing Strategy

### Unit Tests

- Test each command in isolation
- Mock pipelines and ChromaDB
- Test input validation (Pydantic models)
- Test output formatters

### Integration Tests

- Test end-to-end flows (load → search → export)
- Use small test dataset in `tests/fixtures/`
- Verify actual ChromaDB integration

### CLI Tests

- Use Click's `CliRunner` for testing commands
- Test CLI parsing and option handling
- Test help text generation

### Manual Testing

- Load real documents
- Search with various queries
- Export and verify output format
- Test error scenarios

---

## Success Criteria

1. ✅ All 7 main commands fully implemented and tested
2. ✅ CLI installable via `pip install -e .`
3. ✅ Help text complete and accurate (`knowbase --help`, `knowbase <command> --help`)
4. ✅ Output formats working (text, JSON, CSV)
5. ✅ Error handling graceful with helpful messages
6. ✅ Documentation complete with examples
7. ✅ No regression in existing functionality (Streamlit UI, Python API)
8. ✅ All edge cases handled (OOM, missing files, invalid input)

---

## Risks & Mitigation

| Risk                        | Impact | Mitigation                                                          |
| --------------------------- | ------ | ------------------------------------------------------------------- |
| Breaking existing API       | High   | Keep all existing imports/functions unchanged, add CLI layer on top |
| Performance degradation     | Medium | Profile memory/time before release, optimize if needed              |
| Complexity of CLI framework | Medium | Start with Click (proven, simple), extend gradually                 |
| Large file handling         | Medium | Stream results, implement pagination for large exports              |
| Dependency conflicts        | Low    | Pin versions in `requirements.txt`, test in isolation               |

---

## Dependencies to Add

```
click>=8.1.0              # CLI framework
rich>=13.0.0              # Colored output & progress
pydantic>=2.0.0           # Already present, extend for CLI
tabulate>=0.9.0           # Table formatting
tqdm>=4.65.0              # Progress bars

# Optional
alive-progress>=3.0.0     # Alternative progress bars
typer>=0.9.0              # Alternative to Click (modern, type-hint-based)
```

---

## Timeline Estimate

| Phase     | Tasks                    | Estimate          |
| --------- | ------------------------ | ----------------- |
| Phase 1   | Foundation & framework   | 3-4 days          |
| Phase 2   | Load, Search, Info       | 3-4 days          |
| Phase 3   | Ask, Cluster, Export     | 4-5 days          |
| Phase 4   | Packaging & distribution | 2-3 days          |
| Phase 5   | Integration & polish     | 2-3 days          |
| Phase 6   | Cleanup & deprecation    | 1-2 days (future) |
| **Total** |                          | **~20 days**      |

---

## Next Steps

1. **Review this plan** with team
2. **Approve technology stack** (Click + Rich recommended)
3. **Start Phase 1** – Create CLI package structure
4. **Iteratively implement** phases 2-5
5. **Gather feedback** on usability after Phase 2
6. **Polish and document** in Phase 5
