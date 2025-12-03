# Multi-Model Embedding Support - Implementation Plan

## Technical & Engineering Description

### Overview

This implementation plan addresses the requirement to enable users to select different embedding models for generating embeddings, with specific support for EmbeddingGemma-300m (`google/embeddinggemma-300m`) in addition to the current BGE-large-en-v1.5 model. The system currently hardcodes the BGE model throughout the codebase, making it difficult to switch models or support multiple models simultaneously.

The implementation will introduce a flexible, extensible architecture that abstracts model-specific behaviors (prompt formatting, precision requirements, embedding dimensions) while maintaining backward compatibility with existing BGE-based indexes. This will enable users to choose models based on their specific needs: BGE for high-quality 1024-dimensional embeddings, or EmbeddingGemma for smaller, faster 768-dimensional embeddings optimized for on-device deployment.

### Architecture

The solution follows an **Adapter Pattern** architecture where:

1. **Model Adapters** (`src/embeddings/adapters/`) encapsulate model-specific logic:

   - Prompt/instruction formatting (BGE uses simple prefixes, EmbeddingGemma uses structured prompts)
   - Precision handling (EmbeddingGemma requires float32/bfloat16, no float16)
   - Query vs document encoding methods (EmbeddingGemma has separate `encode_query()`/`encode_document()` methods)
   - Embedding dimension detection and validation

2. **Model Registry** (`src/embeddings/model_registry.py`) provides:

   - Centralized model metadata (dimensions, precision requirements, supported features)
   - Model detection and adapter instantiation
   - Validation of model compatibility

3. **Enhanced ModelLoader** maintains backward compatibility while supporting:

   - Dynamic model loading based on user selection
   - Model caching with per-model instances (replacing singleton pattern)
   - Automatic adapter selection based on model name

4. **Collection Management** extends ChromaDB to support:
   - Model-specific collections (separate collections per model to avoid dimension conflicts)
   - Collection naming convention: `{base_name}_{model_slug}` (e.g., `subtitle_embeddings_bge_large`, `subtitle_embeddings_gemma_300m`)
   - Metadata tracking of model name and embedding dimension per collection

### Components & Modules

**New Components:**

- `src/embeddings/adapters/` - Model adapter implementations
  - `base_adapter.py` - Abstract base class defining adapter interface
  - `bge_adapter.py` - BGE model adapter (wraps existing BGE logic)
  - `embeddinggemma_adapter.py` - EmbeddingGemma adapter with structured prompts
- `src/embeddings/model_registry.py` - Model metadata and adapter factory
- `src/embeddings/model_manager.py` - Enhanced model loader with multi-model support

**Modified Components:**

- `src/embeddings/model_loader.py` - Refactor to use adapters and support model switching
- `src/embeddings/embedder.py` - Update to use model adapters for prompt formatting
- `src/utils/config.py` - Add model selection configuration and validation
- `src/vector_store/chroma_manager.py` - Support model-specific collections
- `src/vector_store/pipeline.py` - Pass model information through indexing pipeline
- `scripts/process_subtitles.py` - Add `--model` CLI argument
- `streamlit_app.py` - Add model selection UI component

### Technology Stack

- **Existing**: Python 3.9+, sentence-transformers, PyTorch, ChromaDB
- **New Dependencies**: None (uses existing libraries)
- **Model Support**:
  - BGE-large-en-v1.5 (existing, 1024 dimensions)
  - EmbeddingGemma-300m (new, 768 dimensions with MRL support for 512/256/128)

### Integration Points

1. **Configuration Layer**: Model selection via environment variable `MODEL_NAME` or CLI argument
2. **Model Loading**: Adapter pattern integrates with existing `ModelLoader` class
3. **Embedding Generation**: Adapters handle prompt formatting transparently to `Embedder`
4. **Vector Store**: Collections are model-specific to prevent dimension conflicts
5. **CLI/UI**: Model selection exposed at user interface level

### Data Models

**Model Metadata Schema:**

```python
{
    "model_name": str,           # HuggingFace model identifier
    "embedding_dimension": int,   # Output dimension (768, 1024, etc.)
    "max_sequence_length": int,   # Max input tokens
    "precision_requirements": List[str],  # ["float32", "bfloat16"]
    "supports_query_method": bool,  # Has encode_query() method
    "supports_document_method": bool,  # Has encode_document() method
    "prompt_format": str,        # "prefix" or "structured"
    "mrl_supported": bool        # Matryoshka Representation Learning support
}
```

**Collection Metadata Extension:**

```python
{
    "embedding_dimension": int,
    "model_name": str,
    "model_version": str,        # Optional version tracking
    "created_at": str            # ISO timestamp
}
```

### API Contracts

**ModelAdapter Interface:**

```python
class ModelAdapter(ABC):
    def format_query_prompt(self, text: str) -> str
    def format_document_prompt(self, text: str, title: Optional[str] = None) -> str
    def get_embedding_dimension(self) -> int
    def get_max_sequence_length(self) -> int
    def get_precision_requirements(self) -> List[str]
    def validate_precision(self, precision: str) -> bool
    def encode_query(self, texts: List[str], **kwargs) -> np.ndarray
    def encode_document(self, texts: List[str], **kwargs) -> np.ndarray
```

**ModelRegistry API:**

```python
class ModelRegistry:
    def register_model(self, model_name: str, metadata: dict) -> None
    def get_adapter(self, model_name: str) -> ModelAdapter
    def get_model_metadata(self, model_name: str) -> dict
    def list_supported_models(self) -> List[str]
    def detect_model_type(self, model_name: str) -> str
```

### Security Considerations

- **Model Validation**: Validate model names to prevent arbitrary code execution via malicious model names
- **Collection Isolation**: Model-specific collections prevent cross-model contamination
- **Metadata Validation**: Validate embedding dimensions match collection expectations
- **Error Handling**: Graceful degradation when model switching fails

### Performance Requirements

- **Model Loading**: Cache loaded models per model name (avoid reloading same model)
- **Memory Management**: Unload previous model when switching (optional: keep N models in memory)
- **Collection Switching**: Fast collection retrieval (< 100ms)
- **Backward Compatibility**: No performance degradation for existing BGE workflows

### Key Design Decisions

1. **Separate Collections per Model**: Prevents dimension conflicts and allows parallel model usage
2. **Adapter Pattern**: Encapsulates model-specific logic without modifying core embedding pipeline
3. **Model Registry**: Centralized metadata enables easy addition of new models
4. **Backward Compatibility**: Existing BGE code continues to work without changes
5. **CLI/UI Integration**: Model selection at user interface level for maximum flexibility

---

## Implementation Plan

1. **[x] Phase 1: Model Adapter Infrastructure**

   - _Description: Create the foundational adapter pattern architecture that will abstract model-specific behaviors. This phase establishes the interfaces and base implementations that all model adapters will follow._

     1.1. **[x] Task: Create Adapter Base Class**

   - _Description: Create `src/embeddings/adapters/base_adapter.py` with abstract base class `ModelAdapter` that defines the interface all model adapters must implement. Include methods for prompt formatting, dimension detection, precision validation, and encoding methods._

     1.1.1. **[x] Sub-Task: Define Abstract Methods**

   - _Description: Define abstract methods in `ModelAdapter`: `format_query_prompt()`, `format_document_prompt()`, `get_embedding_dimension()`, `get_max_sequence_length()`, `get_precision_requirements()`, `validate_precision()`, `encode_query()`, `encode_document()`. Include comprehensive docstrings explaining expected behavior and return types._

     1.1.2. **[x] Sub-Task: Add Helper Methods**

   - _Description: Implement helper methods in base class: `_normalize_embeddings()`, `_validate_embeddings()`, `_get_model_info()`. These will be shared utilities used by all adapter implementations._

     1.2. **[x] Task: Create BGE Model Adapter**

   - _Description: Create `src/embeddings/adapters/bge_adapter.py` implementing `ModelAdapter` for BGE models. This adapter wraps the existing BGE-specific logic (instruction prefixes) into the adapter pattern._

     1.2.1. **[x] Sub-Task: Implement BGE Prompt Formatting**

   - _Description: Implement `format_query_prompt()` to prepend `"Represent this sentence for searching relevant passages: "` to query text. Implement `format_document_prompt()` to return document text as-is (BGE doesn't use document prefixes)._

     1.2.2. **[x] Sub-Task: Implement BGE Encoding Methods**

   - _Description: Implement `encode_query()` and `encode_document()` methods that use the underlying SentenceTransformer model with appropriate prompt formatting. Both methods should support batch processing and normalization._

     1.2.3. **[x] Sub-Task: Add BGE Metadata**

   - _Description: Implement methods to return BGE-specific metadata: embedding dimension (1024), max sequence length (512), precision requirements (["float32", "float16"]), and feature flags (no separate query/document methods)._

     1.3. **[x] Task: Create EmbeddingGemma Model Adapter**

   - _Description: Create `src/embeddings/adapters/embeddinggemma_adapter.py` implementing `ModelAdapter` for EmbeddingGemma models. This adapter handles structured prompts and separate query/document encoding methods._

     1.3.1. **[x] Sub-Task: Implement EmbeddingGemma Prompt Formatting**

   - _Description: Implement `format_query_prompt()` to create structured prompt: `"task: search result | query: {content}"`. Implement `format_document_prompt()` to create: `"title: {title | 'none'} | text: {content}"`. Handle optional title parameter._

     1.3.2. **[x] Sub-Task: Implement EmbeddingGemma Encoding Methods**

   - _Description: Implement `encode_query()` and `encode_document()` methods. First check if the SentenceTransformer model has `encode_query()` and `encode_document()` methods (sentence-transformers may add native support). If available, use them directly. Otherwise, manually format prompts and use standard `encode()` method. Support MRL dimension selection (768, 512, 256, 128)._

     1.3.3. **[x] Sub-Task: Add EmbeddingGemma Metadata**

   - _Description: Implement methods to return EmbeddingGemma-specific metadata: embedding dimension (768 default, configurable via MRL), max sequence length (2048), precision requirements (["float32", "bfloat16"] - no float16), and feature flags (supports separate query/document methods, supports MRL)._

     1.3.4. **[x] Sub-Task: Handle Precision Requirements**

   - _Description: Implement `validate_precision()` to ensure EmbeddingGemma models use only float32 or bfloat16. Add validation in model loading to prevent float16 usage. Log warnings if incompatible precision is detected._

     1.4. **[x] Task: Create Model Registry**

   - _Description: Create `src/embeddings/model_registry.py` to centralize model metadata and provide adapter factory functionality. This registry will map model names to their adapters and metadata._

     1.4.1. **[x] Sub-Task: Define Model Metadata Schema**

   - _Description: Create `ModelMetadata` dataclass or TypedDict with fields: `model_name`, `embedding_dimension`, `max_sequence_length`, `precision_requirements`, `supports_query_method`, `supports_document_method`, `prompt_format`, `mrl_supported`, `adapter_class`. Include validation methods._

     1.4.2. **[x] Sub-Task: Implement Model Registration**

   - _Description: Implement `register_model()` method to register model metadata. Create default registrations for BGE-large-en-v1.5 and EmbeddingGemma-300m. Store registrations in a dictionary keyed by model name._

     1.4.3. **[x] Sub-Task: Implement Adapter Factory**

   - _Description: Implement `get_adapter()` method that instantiates the appropriate adapter class for a given model name. Use model detection logic to determine adapter type (BGE vs EmbeddingGemma vs generic). Handle unknown models gracefully with a generic adapter fallback._

     1.4.4. **[x] Sub-Task: Add Model Detection Logic**

   - _Description: Implement `detect_model_type()` method that analyzes model name patterns to determine model family (e.g., "bge" in name → BGE adapter, "embeddinggemma" or "gemma" in name → EmbeddingGemma adapter). Return adapter type string._

     1.5. **[x] Task: Create Adapters Package Structure**

   - _Description: Create `src/embeddings/adapters/` directory with `__init__.py` that exports adapter classes and registry. Ensure proper package initialization._

     1.5.1. **[x] Sub-Task: Create Package Init File**

   - _Description: Create `src/embeddings/adapters/__init__.py` that imports and exports `ModelAdapter`, `BGEAdapter`, `EmbeddingGemmaAdapter`, and `ModelRegistry`. Make adapters easily importable from the package._

2. **[x] Phase 2: Enhanced Model Loader & Manager**

   - _Description: Refactor the existing ModelLoader to use adapters and support multiple models. Replace singleton pattern with model caching to allow switching between models._

     2.1. **[x] Task: Refactor ModelLoader to Use Adapters**

   - _Description: Modify `src/embeddings/model_loader.py` to use model adapters instead of hardcoded BGE logic. Integrate ModelRegistry for adapter retrieval._

     2.1.1. **[x] Sub-Task: Update ModelLoader Initialization**

   - _Description: Modify `__init__()` to accept optional `model_name` parameter and retrieve appropriate adapter from ModelRegistry. Store adapter instance as instance variable. Maintain backward compatibility with existing code that doesn't specify model name._

     2.1.2. **[x] Sub-Task: Update Model Loading Logic**

   - _Description: Modify `load_model()` to use adapter for model-specific initialization. Delegate prompt formatting and encoding to adapter. Maintain existing device and compilation logic. Add precision validation using adapter's `validate_precision()` method._

     2.1.3. **[x] Sub-Task: Update Model Info Logging**

   - _Description: Modify `_log_model_info()` to use adapter's metadata methods. Log embedding dimension, max sequence length, and precision requirements from adapter instead of hardcoded values._

     2.2. **[x] Task: Create Model Manager for Multi-Model Support**

   - _Description: Create `src/embeddings/model_manager.py` to manage multiple model instances, replacing the singleton pattern. This enables switching between models without conflicts._

     2.2.1. **[x] Sub-Task: Implement Model Cache**

   - _Description: Create `ModelManager` class with internal dictionary `_model_cache: Dict[str, ModelLoader]` to cache loaded models by model name. Implement `get_model_loader(model_name)` method that returns cached instance or creates new one._

     2.2.2. **[x] Sub-Task: Implement Model Switching**

   - _Description: Add `switch_model(model_name)` method that unloads current model (if different) and loads new model. Add `unload_model(model_name)` method to explicitly unload a model from cache. Add `list_loaded_models()` to show currently cached models._

     2.2.3. **[x] Sub-Task: Add Memory Management**

   - _Description: Implement `clear_cache()` method to unload all models. Add `unload_unused_models(max_models_to_keep)` to keep only N most recently used models. Add memory usage tracking per model._

     2.2.4. **[x] Sub-Task: Update Global Access Pattern**

   - _Description: Modify `get_model_loader()` function in `model_loader.py` to use ModelManager instead of singleton. Maintain backward compatibility by defaulting to config model name. Update all call sites if necessary._

     2.3. **[x] Task: Update Embedder to Use Adapters**

   - _Description: Modify `src/embeddings/embedder.py` to delegate prompt formatting and encoding to model adapter instead of using hardcoded BGE instructions._

     2.3.1. **[ ] Sub-Task: Update Embedder Initialization**

   - _Description: Modify `__init__()` to retrieve adapter from ModelLoader. Store adapter reference. Remove hardcoded `QUERY_INSTRUCTION` and `DOCUMENT_INSTRUCTION` class variables._

     2.3.2. **[ ] Sub-Task: Update Encoding Methods**

   - _Description: Modify `encode()` method to use adapter's `format_query_prompt()` or `format_document_prompt()` based on `is_query` parameter. Use adapter's `encode_query()` or `encode_document()` methods if available, otherwise fall back to standard `model.encode()` with formatted prompts._

     2.3.3. **[ ] Sub-Task: Update Dimension Methods**

   - _Description: Modify `get_embedding_dimension()` and `get_max_sequence_length()` to use adapter methods instead of directly accessing model properties._

3. **[x] Phase 3: Configuration & Model Selection**

   - _Description: Extend configuration system to support model selection and validation. Add model selection to CLI and UI interfaces._

     3.1. **[x] Task: Extend Configuration System**

   - _Description: Modify `src/utils/config.py` to add model selection support and validation._

     3.1.1. **[x] Sub-Task: Add Model Configuration**

   - _Description: Add `MODEL_NAME` configuration with default value from environment variable or "BAAI/bge-large-en-v1.5" for backward compatibility. Add `MODEL_REGISTRY` reference to access ModelRegistry._

     3.1.2. **[x] Sub-Task: Add Model Validation**

   - _Description: Add `_validate_model()` method that checks if selected model is registered in ModelRegistry. Validate model name format. Log warnings for unknown models but allow them (with generic adapter fallback)._

     3.1.3. **[x] Sub-Task: Add Model Metadata Access**

   - _Description: Add `get_model_metadata()` method to Config that returns model metadata from registry. Add `get_embedding_dimension()` convenience method._

     3.2. **[x] Task: Add CLI Model Selection**

   - _Description: Modify `scripts/process_subtitles.py` to accept `--model` argument for model selection._

     3.2.1. **[x] Sub-Task: Add CLI Argument**

   - _Description: Add `--model` argument to argument parser with help text explaining supported models. Accept HuggingFace model identifier string. Validate model name format._

     3.2.2. **[x] Sub-Task: Pass Model to Pipeline**

   - _Description: Modify `process_pipeline()` function to accept `model_name` parameter. Pass model name to `EmbeddingPipeline` initialization. Update function signature and all call sites._

     3.2.3. **[x] Sub-Task: Add Model Info Display**

   - _Description: Display selected model name and metadata (dimension, precision) in console output at start of processing. Use Rich library for formatted output._

     3.3. **[x] Task: Add Streamlit UI Model Selection**

   - _Description: Modify `streamlit_app.py` to add model selection UI component in sidebar._

     3.3.1. **[x] Sub-Task: Add Model Selector Widget**

   - _Description: Add `st.selectbox()` in sidebar for model selection. Populate with list of supported models from ModelRegistry. Store selected model in `st.session_state.selected_model`._

     3.3.2. **[x] Sub-Task: Update Embedder Initialization**

   - _Description: Modify Embedder instantiation in search functionality to use selected model from session state. Create new Embedder instance when model changes. Show model info (dimension, precision) in sidebar._

     3.3.3. **[x] Sub-Task: Handle Model Switching**

   - _Description: Detect when user changes model selection. Show warning if switching models with existing collection loaded. Optionally reload collection or show message about model-specific collections._

4. **[x] Phase 4: Vector Store Model Support**

   - _Description: Extend ChromaDB manager to support model-specific collections. Update collection naming and metadata to track model information._

     4.1. **[x] Task: Update Collection Naming Convention**

   - _Description: Modify `src/vector_store/chroma_manager.py` to generate model-specific collection names._

     4.1.1. **[x] Sub-Task: Create Collection Name Generator**

   - _Description: Add `_generate_collection_name(model_name, base_name)` method that creates collection name like `{base_name}_{model*slug}`. Extract model slug from model name (e.g., "BAAI/bge-large-en-v1.5" → "bge_large", "google/embeddinggemma-300m" → "gemma_300m"). Handle special characters and normalization.*

     4.1.2. **[x] Sub-Task: Update Collection Creation**

   - _Description: Modify `get_or_create_collection()` to accept optional `model_name` parameter. Use model name to generate collection name. Retrieve embedding dimension from model metadata instead of hardcoded 1024._

     4.1.3. **[x] Sub-Task: Update Collection Metadata**

   - _Description: Store model metadata in collection metadata: `model_name`, `embedding_dimension`, `model_version` (if available), `created_at` timestamp. Validate that existing collections match expected model and dimension._

     4.2. **[x] Task: Update Vector Store Pipeline**

   - _Description: Modify `src/vector_store/pipeline.py` to pass model information through indexing pipeline._

     4.2.1. **[x] Sub-Task: Update Pipeline Initialization**

   - _Description: Modify `VectorStorePipeline.__init__()` to accept optional `model_name` parameter. Pass model name to `ChromaDBManager` initialization._

     4.2.2. **[x] Sub-Task: Update Indexing Methods**

   - _Description: Modify `index_processed_video()` and `index_multiple_videos()` to retrieve model name from Embedder or EmbeddingPipeline. Pass model name to ChromaDBManager for collection selection. Validate embedding dimensions match collection expectations._

     4.3. **[x] Task: Add Collection Management Utilities**

   - _Description: Add utilities for managing multiple model-specific collections._

     4.3.1. **[x] Sub-Task: Add Collection Listing**

   - _Description: Add `list_collections_by_model()` method to ChromaDBManager that returns dictionary mapping model names to collection names. Add `get_collection_for_model(model_name)` convenience method._

     4.3.2. **[x] Sub-Task: Add Collection Validation**

   - _Description: Add `validate_collection_model(collection_name, expected_model_name)` method that checks if collection's metadata matches expected model. Return validation result with error details if mismatch._

     4.3.3. **[x] Sub-Task: Add Migration Utilities**

   - _Description: Add `migrate_collection(old_model_name, new_model_name)` method stub for future migration support. Document that migration requires re-embedding all documents (not implemented in this phase)._

5. **[x] Phase 5: Integration & Testing**

   - _Description: Integrate all components and ensure end-to-end functionality. Write comprehensive tests for new adapter system._

     5.1. **[x] Task: Update Embedding Pipeline Integration**

   - _Description: Ensure `src/embeddings/pipeline.py` properly integrates with new adapter system._

     5.1.1. **[x] Sub-Task: Update Pipeline Initialization**

   - _Description: Modify `EmbeddingPipeline.__init__()` to pass model_name to ModelLoader. Ensure adapter is properly initialized. Verify model metadata is accessible._

     5.1.2. **[x] Sub-Task: Update Embedding Generation**

   - _Description: Verify `generate_embeddings()` and `generate_embeddings_with_checkpointing()` work correctly with adapters. Test that embeddings have correct dimensions based on selected model. Validate normalization and precision requirements._

     5.2. **[x] Task: Update Retrieval System**

   - _Description: Ensure retrieval system works with model-specific collections._

     5.2.1. **[x] Sub-Task: Update Query Engine**

   - _Description: Modify `src/retrieval/query_engine.py` to accept model_name parameter. Ensure query embeddings use correct model adapter. Verify collection selection matches query model._

     5.2.2. **[x] Sub-Task: Update Similarity Search**

   - _Description: Modify `src/retrieval/similarity_search.py` to use model-specific collection. Ensure collection name matches model used for query embedding generation._

     5.3. **[x] Task: Write Unit Tests**

   - _Description: Create comprehensive unit tests for new adapter system and model selection functionality._

     5.3.1. **[x] Sub-Task: Test Model Adapters**

   - _Description: Create `tests/test_adapters.py` with tests for BGEAdapter and EmbeddingGemmaAdapter. Test prompt formatting, encoding methods, metadata retrieval, and precision validation. Use mock SentenceTransformer models where appropriate._

     5.3.2. **[x] Sub-Task: Test Model Registry**

   - _Description: Create `tests/test_model_registry.py` with tests for model registration, adapter retrieval, model detection, and metadata access. Test with known models (BGE, EmbeddingGemma) and unknown models (generic adapter fallback)._

     5.3.3. **[x] Sub-Task: Test Model Manager**

   - _Description: Create `tests/test_model_manager.py` with tests for model caching, model switching, memory management, and cache clearing. Verify models are properly loaded and unloaded._

     5.3.4. **[x] Sub-Task: Test Collection Management**

   - _Description: Create `tests/test_collection_model_support.py` with tests for collection naming, metadata storage, model validation, and collection listing. Test with multiple models and verify isolation._

     5.4. **[x] Task: Write Integration Tests**

   - _Description: Create end-to-end integration tests for complete workflows with different models._

     5.4.1. **[x] Sub-Task: Test BGE Workflow**

   - _Description: Create integration test that processes sample subtitles with BGE model, indexes in collection, and queries results. Verify end-to-end functionality matches existing behavior._

     5.4.2. **[x] Sub-Task: Test EmbeddingGemma Workflow**

   - _Description: Create integration test that processes sample subtitles with EmbeddingGemma model, indexes in separate collection, and queries results. Verify 768-dimensional embeddings are generated correctly._

     5.4.3. **[x] Sub-Task: Test Model Switching**

   - _Description: Create integration test that switches between models during processing. Verify collections are properly isolated and embeddings have correct dimensions for each model._

6. **[x] Phase 6: Documentation & User Guide**

   - _Description: Document new model selection features and provide usage examples for users._

     6.1. **[x] Task: Update Code Documentation**

   - _Description: Add comprehensive docstrings to all new classes and methods. Document adapter interface, model registry usage, and collection management._

     6.1.1. **[x] Sub-Task: Document Adapter Classes**

   - _Description: Add detailed docstrings to ModelAdapter, BGEAdapter, and EmbeddingGemmaAdapter classes. Include usage examples, method descriptions, and parameter documentation._

     6.1.2. **[x] Sub-Task: Document Model Registry**

   - _Description: Add docstrings to ModelRegistry class explaining registration process, adapter retrieval, and model detection. Include examples of registering custom models._

     6.2. **[x] Task: Create User Documentation**

   - _Description: Create user-facing documentation explaining how to select and use different embedding models._

     6.2.1. **[x] Sub-Task: Update README**

   - _Description: Add section to `README.md` explaining model selection feature. Document supported models, CLI usage with `--model` argument, and environment variable configuration. Include examples for both BGE and EmbeddingGemma._

     6.2.2. **[x] Sub-Task: Create Model Selection Guide**

   - _Description: Create `docs/model_selection_guide.md` with detailed guide on choosing between models, comparing BGE vs EmbeddingGemma, performance considerations, and use case recommendations. Include code examples for programmatic model selection._

     6.2.3. **[x] Sub-Task: Document Collection Management**

   - _Description: Add section to documentation explaining model-specific collections, collection naming conventions, and how to manage multiple collections. Explain that different models require separate collections due to dimension differences._

     6.3. **[x] Task: Update Streamlit Documentation**

   - _Description: Update `README_STREAMLIT.md` to document model selection feature in Streamlit UI._

     6.3.1. **[ ] Sub-Task: Document UI Model Selection**

   - _Description: Add section explaining model selector in sidebar, how to switch models, and implications for collections. Include screenshots or descriptions of UI changes._

7. **[x] Phase 7: Validation & Edge Cases**

   - _Description: Handle edge cases, error scenarios, and ensure robust error handling throughout the system._

     7.1. **[x] Task: Handle Model Loading Errors**

   - _Description: Implement robust error handling for model loading failures, invalid model names, and adapter instantiation errors._

     7.1.1. **[x] Sub-Task: Validate Model Names**

   - _Description: Add validation to prevent malicious model names. Sanitize model names before use. Log warnings for suspicious patterns. Provide clear error messages for invalid model names._

     7.1.2. **[x] Sub-Task: Handle Missing Models**

   - _Description: Implement graceful handling when HuggingFace model cannot be downloaded or loaded. Provide helpful error messages with troubleshooting suggestions. Fall back to generic adapter if model-specific adapter fails._

     7.2. **[x] Task: Handle Dimension Mismatches**

   - _Description: Implement validation and error handling for embedding dimension mismatches between models and collections._

     7.2.1. **[x] Sub-Task: Validate Collection Dimensions**

   - _Description: Add validation in `index_processed_video()` to check that embedding dimensions match collection's expected dimension. Raise clear error with model name and dimension details if mismatch detected._

     7.2.2. **[x] Sub-Task: Handle Query Dimension Mismatches**

   - _Description: Add validation in query methods to ensure query embeddings match collection dimension. Provide helpful error message suggesting correct model or collection selection._

     7.3. **[x] Task: Handle Precision Incompatibilities**

   - _Description: Implement validation and warnings for precision requirements, especially for EmbeddingGemma's float16 restriction._

     7.3.1. **[x] Sub-Task: Validate Precision at Load Time**

   - _Description: Check precision compatibility when loading model. Log warnings if device or PyTorch configuration doesn't support required precision. Provide fallback suggestions._

     7.3.2. **[x] Sub-Task: Handle Runtime Precision Errors**

   - _Description: Catch precision-related errors during embedding generation. Provide clear error messages explaining precision requirements and how to fix configuration._

     7.4. **[x] Task: Test Backward Compatibility**

   - _Description: Verify that existing code and workflows continue to work without modifications._

     7.4.1. **[x] Sub-Task: Test Default Model Behavior**

   - _Description: Verify that when no model is specified, system defaults to BGE-large-en-v1.5 and behaves identically to current implementation. Test with existing scripts and workflows._

     7.4.2. **[x] Sub-Task: Test Existing Collections**

   - _Description: Verify that existing BGE collections continue to work correctly. Test that collection detection and model assignment work for collections created before this feature. Ensure no breaking changes to collection access._

8. **[ ] Phase 8: Performance Optimization & Polish**

   - _Description: Optimize model loading and switching performance, add helpful features, and polish user experience._

     8.1. **[ ] Task: Optimize Model Loading**

   - _Description: Implement optimizations to reduce model loading time and memory usage when switching models._

     8.1.1. **[ ] Sub-Task: Implement Lazy Loading**

   - _Description: Ensure models are only loaded when first used, not at initialization. Implement lazy loading in ModelManager to defer model loading until `get_model()` is called._

     8.1.2. **[ ] Sub-Task: Optimize Model Caching**

   - _Description: Implement intelligent cache eviction (LRU) to keep most recently used models in memory. Add configuration option for maximum cached models. Monitor memory usage and evict models when memory threshold exceeded._

     8.2. **[ ] Task: Add Model Information Display**

   - _Description: Add helpful model information display in CLI and UI to help users understand current model selection._

     8.2.1. **[ ] Sub-Task: Add CLI Model Info**

   - _Description: Display model information (name, dimension, precision, features) at start of processing scripts. Use Rich library for formatted, colored output. Show model comparison if multiple models are available._

     8.2.2. **[ ] Sub-Task: Add UI Model Info**

   - _Description: Add model information panel in Streamlit sidebar showing current model, embedding dimension, precision requirements, and supported features. Update dynamically when model selection changes._

     8.3. **[ ] Task: Add Model Comparison Utilities**

   - _Description: Create utilities to help users compare models and make informed selection decisions._

     8.3.1. **[ ] Sub-Task: Create Model Comparison Function**

   - _Description: Add `compare_models(model_names)` function to ModelRegistry that returns comparison table with dimensions, precision, features, and performance characteristics. Format as table for CLI or dict for programmatic access._

     8.3.2. **[ ] Sub-Task: Add Model Recommendation**

   - _Description: Add `recommend_model(use_case, constraints)` function that suggests appropriate model based on use case (quality vs speed, memory constraints, precision requirements). Document use cases and recommendations._

     8.4. **[ ] Task: Add Migration Tooling**

   - _Description: Create utilities to help users migrate from one model to another (requires re-embedding)._

     8.4.1. **[ ] Sub-Task: Create Migration Script**

   - _Description: Create `scripts/migrate_model.py` script that helps users migrate collections from one model to another. Script should: 1) Validate source and target models, 2) Create new collection with target model, 3) Re-process all source documents with target model, 4) Index in new collection, 5) Provide comparison statistics. Include dry-run mode and confirmation prompts._

     8.4.2. **[ ] Sub-Task: Document Migration Process**

   - _Description: Add migration guide to documentation explaining when migration is needed, how to use migration script, and considerations for large collections. Include performance estimates and time requirements._
