"""
Configuration Manager - Modular Configuration System

Handles loading, validating, and managing configurations from:
- YAML/JSON files
- Environment variables
- CLI arguments
- Defaults

Supports partial pipeline execution:
- Only embeddings
- Only preprocessing
- Only retrieval
- Only RAG/AI Search
- Complete pipeline
"""

import os
import json
import yaml
from pathlib import Path
from typing import Optional, Dict, Any, List, Literal
from dataclasses import dataclass, field, asdict
from pydantic import BaseModel, Field, validator
from dotenv import load_dotenv
import logging

logger = logging.getLogger(__name__)


# ============================================================================
# PYDANTIC MODELS FOR VALIDATION
# ============================================================================


class EmbeddingConfig(BaseModel):
    """Configuration for embedding generation pipeline."""

    model_name: str = Field(
        default="google/embeddinggemma-300m",
        description="HuggingFace model name for embeddings",
    )
    device: Literal["auto", "cpu", "cuda", "mps"] = Field(
        default="auto", description="Device to use for embeddings (auto-detect)"
    )
    batch_size: int = Field(
        default=32, ge=1, le=512, description="Batch size for embedding generation"
    )
    cache_dir: str = Field(
        default="~/.cache/huggingface", description="HuggingFace cache directory"
    )
    model_cache_enabled: bool = Field(
        default=True, description="Enable model caching to disk"
    )
    precision: Literal["fp32", "fp16", "bf16"] = Field(
        default="fp32", description="Model precision (fp32, fp16, bf16)"
    )

    class Config:
        use_enum_values = True


class PreprocessingConfig(BaseModel):
    """Configuration for document preprocessing pipeline."""

    chunk_size: int = Field(
        default=512, ge=64, le=2048, description="Size of text chunks"
    )
    chunk_overlap: int = Field(
        default=50, ge=0, le=500, description="Overlap between chunks"
    )
    min_chunk_size: int = Field(
        default=50, ge=10, description="Minimum chunk size to keep"
    )
    remove_html: bool = Field(default=True, description="Remove HTML tags")
    normalize_whitespace: bool = Field(
        default=True, description="Normalize whitespace"
    )
    remove_special_chars: bool = Field(
        default=False, description="Remove special characters"
    )
    lowercase: bool = Field(default=False, description="Convert to lowercase")
    language: str = Field(default="en", description="Language for text processing")

    class Config:
        use_enum_values = True


class VectorStoreConfig(BaseModel):
    """Configuration for vector store (ChromaDB)."""

    db_path: str = Field(
        default="./data/vector_db", description="Path to ChromaDB directory"
    )
    collection_name: str = Field(
        default="documents", description="Default collection name"
    )
    distance_metric: Literal["cosine", "l2", "ip"] = Field(
        default="cosine", description="Distance metric for similarity search"
    )
    persist_directory: bool = Field(
        default=True, description="Persist vector store to disk"
    )

    class Config:
        use_enum_values = True


class RetrievalConfig(BaseModel):
    """Configuration for semantic search and retrieval."""

    top_k: int = Field(
        default=5, ge=1, le=100, description="Number of results to return"
    )
    similarity_threshold: float = Field(
        default=0.0, ge=0.0, le=1.0, description="Minimum similarity score"
    )
    rerank_enabled: bool = Field(
        default=False, description="Enable result reranking"
    )
    rerank_model: Optional[str] = Field(
        default=None, description="Model for reranking results"
    )
    filter_by_metadata: Dict[str, Any] = Field(
        default_factory=dict, description="Metadata filters for search"
    )

    class Config:
        use_enum_values = True


class AISearchConfig(BaseModel):
    """Configuration for RAG and AI Search."""

    enabled: bool = Field(default=True, description="Enable AI Search/RAG")
    llm_provider: Literal["openai", "anthropic", "groq", "azure", "ollama"] = Field(
        default="openai", description="LLM provider"
    )
    llm_model: str = Field(
        default="gpt-4-mini", description="LLM model name"
    )
    llm_temperature: float = Field(
        default=0.7, ge=0.0, le=2.0, description="LLM temperature"
    )
    llm_max_tokens: int = Field(
        default=2000, ge=100, le=8000, description="Max tokens in LLM response"
    )
    llm_api_key: Optional[str] = Field(
        default=None, description="LLM API key (from env if not set)"
    )

    # Query Analyzer
    query_analyzer_enabled: bool = Field(
        default=True, description="Analyze query clarity and intent"
    )
    query_clarity_threshold: float = Field(
        default=0.85, ge=0.0, le=1.0, description="Query clarity threshold"
    )

    # Query Rewriter
    query_rewriter_enabled: bool = Field(
        default=True, description="Rewrite query for better retrieval"
    )

    # Clarification
    clarification_enabled: bool = Field(
        default=True, description="Ask for clarification if needed"
    )

    # Conversation window
    conversation_window: int = Field(
        default=10, ge=1, le=100, description="Number of messages in conversation"
    )

    # Thinking/Reasoning display
    show_thinking: bool = Field(
        default=True, description="Show thinking/reasoning process"
    )

    class Config:
        use_enum_values = True


class ClusteringConfig(BaseModel):
    """Configuration for clustering analysis."""

    enabled: bool = Field(default=True, description="Enable clustering")
    min_cluster_size: int = Field(
        default=5, ge=2, description="Minimum cluster size for HDBSCAN"
    )
    min_samples: int = Field(
        default=5, ge=1, description="Min samples for HDBSCAN"
    )
    clustering_metric: Literal["euclidean", "cosine", "manhattan"] = Field(
        default="cosine", description="Distance metric for clustering"
    )
    use_umap: bool = Field(
        default=True, description="Use UMAP for dimensionality reduction"
    )
    umap_n_neighbors: int = Field(
        default=15, ge=2, description="Number of neighbors for UMAP"
    )
    umap_min_dist: float = Field(
        default=0.1, ge=0.0, description="Minimum distance for UMAP"
    )
    umap_n_components: int = Field(
        default=3, ge=2, le=3, description="Number of dimensions for UMAP (2D or 3D)"
    )

    class Config:
        use_enum_values = True


class PipelineConfig(BaseModel):
    """Configuration for pipeline execution."""

    # Which pipelines to run
    run_preprocessing: bool = Field(default=True, description="Run preprocessing")
    run_embedding: bool = Field(default=True, description="Run embedding generation")
    run_indexing: bool = Field(default=True, description="Run vector store indexing")
    run_retrieval: bool = Field(default=False, description="Run retrieval search")
    run_ai_search: bool = Field(default=False, description="Run RAG/AI search")
    run_clustering: bool = Field(default=False, description="Run clustering analysis")

    # Execution settings
    skip_existing: bool = Field(
        default=False, description="Skip already processed documents"
    )
    save_intermediate: bool = Field(
        default=False, description="Save intermediate processing results"
    )
    parallel_processing: bool = Field(
        default=True, description="Enable parallel processing"
    )
    num_workers: int = Field(
        default=-1, ge=-1, description="Number of workers (-1 = auto)"
    )

    # Logging
    verbose: bool = Field(default=False, description="Verbose logging")
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = Field(
        default="INFO", description="Logging level"
    )

    class Config:
        use_enum_values = True


class CompleteConfig(BaseModel):
    """Complete configuration encompassing all pipelines."""

    # Core sections
    embedding: EmbeddingConfig = Field(default_factory=EmbeddingConfig)
    preprocessing: PreprocessingConfig = Field(
        default_factory=PreprocessingConfig
    )
    vector_store: VectorStoreConfig = Field(default_factory=VectorStoreConfig)
    retrieval: RetrievalConfig = Field(default_factory=RetrievalConfig)
    ai_search: AISearchConfig = Field(default_factory=AISearchConfig)
    clustering: ClusteringConfig = Field(default_factory=ClusteringConfig)
    pipeline: PipelineConfig = Field(default_factory=PipelineConfig)

    # Metadata
    config_name: str = Field(default="default", description="Configuration name")
    description: str = Field(default="", description="Configuration description")
    created_at: Optional[str] = Field(default=None, description="Creation timestamp")

    class Config:
        use_enum_values = True

    @validator("config_name")
    def validate_config_name(cls, v):
        if not v or len(v) < 1:
            raise ValueError("config_name must not be empty")
        return v


# ============================================================================
# CONFIGURATION MANAGER
# ============================================================================


class ConfigManager:
    """Manager for loading, saving, and merging configurations."""

    def __init__(
        self, config_file: Optional[Path] = None, env_file: Optional[Path] = None
    ):
        """
        Initialize configuration manager.

        Args:
            config_file: Path to config YAML/JSON file
            env_file: Path to .env file for environment variables
        """
        self.config_file = config_file
        self.env_file = env_file

        # Load environment variables
        if env_file and env_file.exists():
            load_dotenv(env_file)
        else:
            # Try default location
            project_root = Path(__file__).parent.parent.parent
            default_env = project_root / ".env"
            if default_env.exists():
                load_dotenv(default_env)

        # Initialize with defaults
        self.config = CompleteConfig()

        # Load from file if provided
        if config_file and config_file.exists():
            self.load_from_file(config_file)

        # Override with environment variables
        self._load_from_env()

    def load_from_file(self, config_file: Path) -> None:
        """
        Load configuration from YAML or JSON file.

        Args:
            config_file: Path to config file

        Raises:
            FileNotFoundError: If file doesn't exist
            ValueError: If file format is invalid
        """
        if not config_file.exists():
            raise FileNotFoundError(f"Config file not found: {config_file}")

        try:
            with open(config_file, "r") as f:
                if config_file.suffix in [".yaml", ".yml"]:
                    data = yaml.safe_load(f) or {}
                elif config_file.suffix == ".json":
                    data = json.load(f)
                else:
                    raise ValueError(
                        f"Unsupported file format: {config_file.suffix}. "
                        "Use .yaml, .yml, or .json"
                    )

            # Validate and merge
            self.config = CompleteConfig(**data)
            logger.info(f"Loaded config from {config_file}")

        except yaml.YAMLError as e:
            raise ValueError(f"Invalid YAML format: {e}")
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON format: {e}")

    def save_to_file(self, output_file: Path, include_defaults: bool = False) -> None:
        """
        Save configuration to YAML or JSON file.

        Args:
            output_file: Path to output file
            include_defaults: Include default values in output
        """
        output_file.parent.mkdir(parents=True, exist_ok=True)

        data = self.config.dict(exclude_none=not include_defaults)

        try:
            with open(output_file, "w") as f:
                if output_file.suffix in [".yaml", ".yml"]:
                    yaml.dump(data, f, default_flow_style=False, sort_keys=False)
                elif output_file.suffix == ".json":
                    json.dump(data, f, indent=2)
                else:
                    raise ValueError(
                        f"Unsupported file format: {output_file.suffix}"
                    )

            logger.info(f"Saved config to {output_file}")

        except Exception as e:
            logger.error(f"Failed to save config: {e}")
            raise

    def _load_from_env(self) -> None:
        """Load configuration from environment variables, overriding file values."""
        # Embedding config
        if model := os.getenv("MODEL_NAME"):
            self.config.embedding.model_name = model
        if device := os.getenv("DEVICE"):
            if device in ["auto", "cpu", "cuda", "mps"]:
                self.config.embedding.device = device  # type: ignore
        if batch_size := os.getenv("BATCH_SIZE"):
            self.config.embedding.batch_size = int(batch_size)
        if cache_dir := os.getenv("MODEL_CACHE_DIR"):
            self.config.embedding.cache_dir = cache_dir

        # Preprocessing config
        if chunk_size := os.getenv("CHUNK_SIZE"):
            self.config.preprocessing.chunk_size = int(chunk_size)
        if chunk_overlap := os.getenv("CHUNK_OVERLAP"):
            self.config.preprocessing.chunk_overlap = int(chunk_overlap)

        # Vector store config
        if db_path := os.getenv("VECTOR_DB_PATH"):
            self.config.vector_store.db_path = db_path
        else:
            # Fallback to active database from DatabaseManager
            try:
                from .db_manager import get_db_manager
                db_manager = get_db_manager()
                self.config.vector_store.db_path = str(db_manager.get_db_path())
            except Exception:
                pass

        # Retrieval config
        if top_k := os.getenv("TOP_K"):
            self.config.retrieval.top_k = int(top_k)
        if threshold := os.getenv("SIMILARITY_THRESHOLD"):
            self.config.retrieval.similarity_threshold = float(threshold)

        # AI Search config
        if llm_provider := os.getenv("LLM_PROVIDER"):
            if llm_provider in ["openai", "anthropic", "groq", "azure", "ollama"]:
                self.config.ai_search.llm_provider = llm_provider  # type: ignore
        if llm_model := os.getenv("LLM_MODEL"):
            self.config.ai_search.llm_model = llm_model
        if llm_temp := os.getenv("LLM_TEMPERATURE"):
            self.config.ai_search.llm_temperature = float(llm_temp)
        if api_key := os.getenv("OPENAI_API_KEY"):
            self.config.ai_search.llm_api_key = api_key

        # Pipeline config
        if verbose := os.getenv("VERBOSE"):
            self.config.pipeline.verbose = verbose.lower() in ["true", "1", "yes"]
        if log_level := os.getenv("LOG_LEVEL"):
            if log_level in ["DEBUG", "INFO", "WARNING", "ERROR"]:
                self.config.pipeline.log_level = log_level  # type: ignore

    def merge_with_dict(self, overrides: Dict[str, Any]) -> None:
        """
        Merge configuration with dictionary overrides.

        Args:
            overrides: Dictionary with overrides (supports nested keys)

        Example:
            config_mgr.merge_with_dict({
                "embedding": {"batch_size": 64},
                "ai_search": {"llm_temperature": 0.5}
            })
        """
        config_dict = self.config.dict()
        self._deep_merge(config_dict, overrides)
        self.config = CompleteConfig(**config_dict)
        logger.debug(f"Merged with overrides: {overrides}")

    @staticmethod
    def _deep_merge(base: Dict, override: Dict) -> None:
        """Deep merge override dict into base dict."""
        for key, value in override.items():
            if isinstance(value, dict) and key in base and isinstance(base[key], dict):
                ConfigManager._deep_merge(base[key], value)
            else:
                base[key] = value

    def get_embedding_config(self) -> EmbeddingConfig:
        """Get embedding configuration."""
        return self.config.embedding

    def get_preprocessing_config(self) -> PreprocessingConfig:
        """Get preprocessing configuration."""
        return self.config.preprocessing

    def get_vector_store_config(self) -> VectorStoreConfig:
        """Get vector store configuration."""
        return self.config.vector_store

    def get_retrieval_config(self) -> RetrievalConfig:
        """Get retrieval configuration."""
        return self.config.retrieval

    def get_ai_search_config(self) -> AISearchConfig:
        """Get AI search configuration."""
        return self.config.ai_search

    def get_clustering_config(self) -> ClusteringConfig:
        """Get clustering configuration."""
        return self.config.clustering

    def get_pipeline_config(self) -> PipelineConfig:
        """Get pipeline configuration."""
        return self.config.pipeline

    def to_dict(self, exclude_none: bool = True) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        return self.config.dict(exclude_none=exclude_none)

    def to_json(self, indent: int = 2) -> str:
        """Convert configuration to JSON string."""
        return self.config.json(indent=indent)

    def to_yaml(self) -> str:
        """Convert configuration to YAML string."""
        return yaml.dump(self.config.dict(), default_flow_style=False, sort_keys=False)

    def __repr__(self) -> str:
        return f"ConfigManager(name={self.config.config_name})"


# ============================================================================
# PRESET CONFIGURATIONS
# ============================================================================


def get_preset_config(preset: Literal["full_pipeline", "embeddings_only", "search_only", "rag_only"]) -> CompleteConfig:
    """
    Get a preset configuration for common use cases.

    Args:
        preset: One of 'full_pipeline', 'embeddings_only', 'search_only', 'rag_only'

    Returns:
        CompleteConfig with appropriate pipeline settings
    """
    if preset == "full_pipeline":
        return CompleteConfig(
            config_name="full_pipeline",
            description="Complete pipeline: preprocessing → embeddings → indexing → search",
            pipeline=PipelineConfig(
                run_preprocessing=True,
                run_embedding=True,
                run_indexing=True,
                run_retrieval=False,
                run_ai_search=False,
                run_clustering=False,
            ),
        )

    elif preset == "embeddings_only":
        return CompleteConfig(
            config_name="embeddings_only",
            description="Only embedding generation (assumes preprocessed documents)",
            pipeline=PipelineConfig(
                run_preprocessing=False,
                run_embedding=True,
                run_indexing=True,
                run_retrieval=False,
                run_ai_search=False,
                run_clustering=False,
            ),
        )

    elif preset == "search_only":
        return CompleteConfig(
            config_name="search_only",
            description="Only semantic search retrieval (assumes indexed documents)",
            pipeline=PipelineConfig(
                run_preprocessing=False,
                run_embedding=False,
                run_indexing=False,
                run_retrieval=True,
                run_ai_search=False,
                run_clustering=False,
            ),
        )

    elif preset == "rag_only":
        return CompleteConfig(
            config_name="rag_only",
            description="Only RAG/AI search (assumes indexed documents)",
            pipeline=PipelineConfig(
                run_preprocessing=False,
                run_embedding=False,
                run_indexing=False,
                run_retrieval=False,
                run_ai_search=True,
                run_clustering=False,
            ),
        )

    else:
        raise ValueError(
            f"Unknown preset: {preset}. "
            "Use: full_pipeline, embeddings_only, search_only, rag_only"
        )


# ============================================================================
# GLOBAL SINGLETON (for backward compatibility with existing code)
# ============================================================================

_global_config_manager: Optional[ConfigManager] = None


def init_config_manager(
    config_file: Optional[Path] = None, env_file: Optional[Path] = None
) -> ConfigManager:
    """Initialize global configuration manager."""
    global _global_config_manager
    _global_config_manager = ConfigManager(config_file, env_file)
    return _global_config_manager


def get_config_manager() -> ConfigManager:
    """Get global configuration manager (lazy initialize if needed)."""
    global _global_config_manager
    if _global_config_manager is None:
        _global_config_manager = ConfigManager()
    return _global_config_manager
