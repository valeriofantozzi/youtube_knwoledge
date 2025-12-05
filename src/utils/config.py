"""
Configuration Module

Manages application configuration with environment variable support and validation.
Includes model selection and validation for multi-model embedding support.
"""

import os
from pathlib import Path
from typing import Optional, Literal, Dict, Any
from dotenv import load_dotenv


class Config:
    """Application configuration manager with validation."""

    def __init__(self, env_file: Optional[str] = None):
        """
        Initialize configuration.

        Args:
            env_file: Path to .env file. If None, looks for .env in project root.
        """
        # Load environment variables
        if env_file:
            load_dotenv(env_file)
        else:
            # Try to find .env in project root
            project_root = Path(__file__).parent.parent.parent
            env_path = project_root / ".env"
            if env_path.exists():
                load_dotenv(env_path)
            else:
                load_dotenv()  # Try default locations

        # Model Configuration
        self.MODEL_NAME = os.getenv("MODEL_NAME", "google/embeddinggemma-300m")
        self.MODEL_CACHE_DIR = os.path.expanduser(
            os.getenv("MODEL_CACHE_DIR", "~/.cache/huggingface")
        )

        # LLM Configuration
        self.OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

        # ==========================================
        # AI SEARCH AGENTS CONFIGURATION
        # ==========================================
        # Query Analyzer Agent
        self.AI_QUERY_ANALYZER_PROVIDER = os.getenv(
            "AI_QUERY_ANALYZER_PROVIDER", "openai"
        )
        self.AI_QUERY_ANALYZER_MODEL = os.getenv(
            "AI_QUERY_ANALYZER_MODEL", "gpt-4-mini"
        )
        self.AI_QUERY_ANALYZER_TEMPERATURE = float(
            os.getenv("AI_QUERY_ANALYZER_TEMPERATURE", "0.3")
        )
        self.AI_QUERY_ANALYZER_MAX_TOKENS = self._get_int(
            "AI_QUERY_ANALYZER_MAX_TOKENS", 500
        )
        self.AI_QUERY_ANALYZER_API_KEY = os.getenv(
            "AI_QUERY_ANALYZER_API_KEY", self.OPENAI_API_KEY
        )

        # Clarification Agent
        self.AI_CLARIFICATION_PROVIDER = os.getenv(
            "AI_CLARIFICATION_PROVIDER", "openai"
        )
        self.AI_CLARIFICATION_MODEL = os.getenv(
            "AI_CLARIFICATION_MODEL", "gpt-3.5-turbo"
        )
        self.AI_CLARIFICATION_TEMPERATURE = float(
            os.getenv("AI_CLARIFICATION_TEMPERATURE", "0.7")
        )
        self.AI_CLARIFICATION_MAX_TOKENS = self._get_int(
            "AI_CLARIFICATION_MAX_TOKENS", 800
        )
        self.AI_CLARIFICATION_API_KEY = os.getenv(
            "AI_CLARIFICATION_API_KEY", self.OPENAI_API_KEY
        )

        # Query Rewriter Agent
        self.AI_QUERY_REWRITER_PROVIDER = os.getenv(
            "AI_QUERY_REWRITER_PROVIDER", "openai"
        )
        self.AI_QUERY_REWRITER_MODEL = os.getenv(
            "AI_QUERY_REWRITER_MODEL", "gpt-3.5-turbo"
        )
        self.AI_QUERY_REWRITER_TEMPERATURE = float(
            os.getenv("AI_QUERY_REWRITER_TEMPERATURE", "0.3")
        )
        self.AI_QUERY_REWRITER_MAX_TOKENS = self._get_int(
            "AI_QUERY_REWRITER_MAX_TOKENS", 300
        )
        self.AI_QUERY_REWRITER_API_KEY = os.getenv(
            "AI_QUERY_REWRITER_API_KEY", self.OPENAI_API_KEY
        )

        # RAG Response Generator Agent
        self.AI_RAG_GENERATOR_PROVIDER = os.getenv(
            "AI_RAG_GENERATOR_PROVIDER", "openai"
        )
        self.AI_RAG_GENERATOR_MODEL = os.getenv("AI_RAG_GENERATOR_MODEL", "gpt-4")
        self.AI_RAG_GENERATOR_TEMPERATURE = float(
            os.getenv("AI_RAG_GENERATOR_TEMPERATURE", "0.2")
        )
        self.AI_RAG_GENERATOR_MAX_TOKENS = self._get_int(
            "AI_RAG_GENERATOR_MAX_TOKENS", 500
        )
        self.AI_RAG_GENERATOR_API_KEY = os.getenv(
            "AI_RAG_GENERATOR_API_KEY", self.OPENAI_API_KEY
        )

        # Alternative Provider Keys
        self.ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
        self.GROQ_API_KEY = os.getenv("GROQ_API_KEY")
        self.AZURE_OPENAI_API_KEY = os.getenv("AZURE_OPENAI_API_KEY")
        self.AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
        self.AZURE_OPENAI_API_VERSION = os.getenv(
            "AZURE_OPENAI_API_VERSION", "2024-02-15-preview"
        )
        self.OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
        self.OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "mistral")

        # Agent Behavior Configuration
        self.AI_QUERY_CLARITY_THRESHOLD = float(
            os.getenv("AI_QUERY_CLARITY_THRESHOLD", "0.85")
        )
        self.AI_MAX_SUGGESTED_QUERIES = self._get_int("AI_MAX_SUGGESTED_QUERIES", 5)
        self.AI_CONVERSATION_WINDOW = self._get_int("AI_CONVERSATION_WINDOW", 10)

        # Processing Configuration
        # Batch size will be auto-optimized if not explicitly set
        batch_size_env = os.getenv("BATCH_SIZE")
        if batch_size_env:
            self.BATCH_SIZE = self._get_int("BATCH_SIZE", 128)
        else:
            # Auto-optimize batch size based on hardware
            try:
                from .performance_optimizer import get_performance_optimizer

                optimizer = get_performance_optimizer()
                self.BATCH_SIZE = optimizer.get_optimal_batch_size()
            except Exception:
                # Fallback to default
                self.BATCH_SIZE = 128
        self.CHUNK_SIZE = self._get_int("CHUNK_SIZE", 300)
        self.CHUNK_OVERLAP = self._get_int("CHUNK_OVERLAP", 60)
        self.MIN_CHUNK_SIZE = self._get_int("MIN_CHUNK_SIZE", 50)

        # Vector Database Configuration
        # Use DatabaseManager to resolve path, but allow env var override
        try:
            from .db_manager import get_db_manager
            db_manager = get_db_manager()
            default_db_path = str(db_manager.get_db_path())
        except Exception as e:
            # Fallback if db_manager fails (e.g. during circular import or setup)
            default_db_path = "./data/vector_db"
            
        self.VECTOR_DB_PATH = os.getenv("VECTOR_DB_PATH", default_db_path)
        self.COLLECTION_NAME = os.getenv("COLLECTION_NAME", "document_embeddings")

        # Device Configuration
        device = os.getenv("DEVICE", "auto").lower()
        if device == "auto":
            # Use hardware detector for auto-detection (supports MPS)
            try:
                from .hardware_detector import get_hardware_detector

                hardware_detector = get_hardware_detector()
                self.DEVICE = hardware_detector.get_recommended_device()
            except Exception:
                # Fallback to basic detection
                import torch

                if torch.cuda.is_available():
                    self.DEVICE = "cuda"
                elif (
                    hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
                ):
                    self.DEVICE = "mps"
                else:
                    self.DEVICE = "cpu"
        elif device in ["cpu", "cuda", "mps"]:
            self.DEVICE = device
        else:
            raise ValueError(
                f"Invalid DEVICE value: {device}. Must be 'auto', 'cpu', 'cuda', or 'mps'"
            )

        self.CUDA_VISIBLE_DEVICES = os.getenv("CUDA_VISIBLE_DEVICES", "0")

        # Logging Configuration
        log_level = os.getenv("LOG_LEVEL", "INFO").upper()
        valid_log_levels = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
        if log_level not in valid_log_levels:
            raise ValueError(
                f"Invalid LOG_LEVEL: {log_level}. Must be one of {valid_log_levels}"
            )
        self.LOG_LEVEL = log_level

        log_file = os.getenv("LOG_FILE", "./logs/app.log")
        # Create logs directory if it doesn't exist
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        self.LOG_FILE = str(log_path)

        # Performance Configuration
        # Auto-optimize workers if not explicitly set
        workers_env = os.getenv("MAX_WORKERS")
        if workers_env:
            workers_value = self._get_int("MAX_WORKERS", -1)
            if workers_value == -1:
                # Auto-detect based on system resources with percentage limit
                self.MAX_WORKERS = self._calculate_optimal_workers()
            elif workers_value >= 0:
                self.MAX_WORKERS = workers_value
            else:
                raise ValueError(f"MAX_WORKERS must be -1 or >= 0, got {workers_value}")
        else:
            # Auto-optimize based on hardware
            try:
                from .performance_optimizer import get_performance_optimizer

                optimizer = get_performance_optimizer()
                self.MAX_WORKERS = optimizer.get_optimal_workers("cpu_bound")
            except Exception:
                # Fallback to CPU count
                self.MAX_WORKERS = os.cpu_count() or 1
        self.ENABLE_CHECKPOINTING = self._get_bool("ENABLE_CHECKPOINTING", True)
        self.CHECKPOINT_INTERVAL = self._get_int("CHECKPOINT_INTERVAL", 1000)

        # Validate configuration
        self._validate()

        # Validate model configuration
        self._validate_model()

    def _get_int(self, key: str, default: int) -> int:
        """Get integer environment variable."""
        value = os.getenv(key)
        if value is None:
            return default
        try:
            return int(value)
        except ValueError:
            raise ValueError(f"Invalid integer value for {key}: {value}")

    def _get_bool(self, key: str, default: bool) -> bool:
        """Get boolean environment variable."""
        value = os.getenv(key)
        if value is None:
            return default
        value_lower = value.lower()
        if value_lower in ("true", "1", "yes", "on"):
            return True
        elif value_lower in ("false", "0", "no", "off"):
            return False
        else:
            raise ValueError(f"Invalid boolean value for {key}: {value}")

    def _get_float(self, key: str, default: float) -> float:
        """Get float environment variable."""
        value = os.getenv(key)
        if value is None:
            return default
        try:
            return float(value)
        except ValueError:
            raise ValueError(f"Invalid float value for {key}: {value}")

    def _calculate_optimal_workers(self) -> int:
        """
        Calculate optimal number of workers based on system resources.

        Uses psutil to detect available CPU cores and applies the
        MAX_WORKERS_PERCENTAGE to limit resource usage.

        Returns:
            Optimal number of workers
        """
        try:
            import psutil

            # Get percentage limit (default 0.75 = 75%)
            percentage = self._get_float("MAX_WORKERS_PERCENTAGE", 0.75)

            # Validate percentage range
            if not 0.0 < percentage <= 1.0:
                raise ValueError(
                    f"MAX_WORKERS_PERCENTAGE must be between 0.0 and 1.0, got {percentage}"
                )

            # Get total CPU count (logical cores)
            cpu_count = psutil.cpu_count(logical=True) or os.cpu_count() or 1

            # Calculate workers based on percentage
            optimal_workers = max(1, int(cpu_count * percentage))

            # Log the calculation - skip logging here to avoid circular imports
            print(
                f"INFO: Auto-detected {optimal_workers} workers "
                f"({percentage * 100:.0f}% of {cpu_count} CPU cores)"
            )

            return optimal_workers

        except ImportError:
            # Fallback if psutil is not available
            cpu_count = os.cpu_count() or 1
            optimal_workers = max(1, int(cpu_count * 0.75))
            print(
                f"WARNING: psutil not available, using os.cpu_count() for worker calculation"
            )
            return os.cpu_count() or 1
        except Exception as e:
            # Fallback on any error
            # Fallback on error
            print("WARNING: Error calculating optimal workers. Using fallback.")
            return os.cpu_count() or 1

    def _validate(self) -> None:
        """Validate configuration values."""
        # Validate batch size
        if self.BATCH_SIZE < 1:
            raise ValueError(f"BATCH_SIZE must be >= 1, got {self.BATCH_SIZE}")

        # Validate chunk sizes
        if self.CHUNK_SIZE < self.MIN_CHUNK_SIZE:
            raise ValueError(
                f"CHUNK_SIZE ({self.CHUNK_SIZE}) must be >= MIN_CHUNK_SIZE "
                f"({self.MIN_CHUNK_SIZE})"
            )

        if self.CHUNK_OVERLAP < 0:
            raise ValueError(f"CHUNK_OVERLAP must be >= 0, got {self.CHUNK_OVERLAP}")

        if self.CHUNK_OVERLAP >= self.CHUNK_SIZE:
            raise ValueError(
                f"CHUNK_OVERLAP ({self.CHUNK_OVERLAP}) must be < CHUNK_SIZE "
                f"({self.CHUNK_SIZE})"
            )

        # Validate workers
        if self.MAX_WORKERS < 1:
            raise ValueError(f"MAX_WORKERS must be >= 1, got {self.MAX_WORKERS}")

        # Validate checkpoint interval
        if self.CHECKPOINT_INTERVAL < 1:
            raise ValueError(
                f"CHECKPOINT_INTERVAL must be >= 1, got {self.CHECKPOINT_INTERVAL}"
            )

    def _validate_model(self) -> None:
        """
        Validate model configuration against ModelRegistry.

        Logs warnings for unknown models but allows them to continue with generic adapter.
        """
        try:
            from ..embeddings.model_registry import get_model_registry

            registry = get_model_registry()

            # Check if model is registered
            if not registry.is_registered(self.MODEL_NAME):
                from ..utils.logger import get_default_logger

                logger = get_default_logger()
                logger.warning(
                    f"Model '{self.MODEL_NAME}' is not in the registered models list. "
                    "It will use a generic adapter fallback. "
                    f"Registered models: {registry.get_registered_models()}"
                )
            else:
                # Validate model name format (basic check)
                if not self.MODEL_NAME or not isinstance(self.MODEL_NAME, str):
                    raise ValueError(
                        f"Invalid MODEL_NAME: {self.MODEL_NAME}. Must be a non-empty string."
                    )

        except ImportError as e:
            # Graceful degradation if model registry is not available
            from ..utils.logger import get_default_logger

            logger = get_default_logger()
            logger.debug(f"Model registry not available for validation: {e}")
        except Exception as e:
            # Log validation errors but don't fail initialization
            from ..utils.logger import get_default_logger

            logger = get_default_logger()
            logger.warning(f"Model validation failed: {e}. Using model as-is.")

    def get_model_metadata(self) -> Optional[Dict[str, Any]]:
        """
        Get metadata for the configured model from ModelRegistry.

        Returns:
            Model metadata dictionary or None if not available
        """
        try:
            from ..embeddings.model_registry import get_model_registry

            registry = get_model_registry()
            return registry.get_model_metadata(self.MODEL_NAME)
        except Exception:
            return None

    def get_embedding_dimension(self) -> int:
        """
        Get embedding dimension for the configured model.

        Returns:
            Embedding dimension (default: 1024 for backward compatibility)
        """
        metadata = self.get_model_metadata()
        if metadata:
            return metadata.embedding_dimension

        # Fallback for backward compatibility
        return 1024

    def to_dict(self) -> dict:
        """Convert configuration to dictionary."""
        config_dict = {
            "MODEL_NAME": self.MODEL_NAME,
            "MODEL_CACHE_DIR": self.MODEL_CACHE_DIR,
            "BATCH_SIZE": self.BATCH_SIZE,
            "CHUNK_SIZE": self.CHUNK_SIZE,
            "CHUNK_OVERLAP": self.CHUNK_OVERLAP,
            "MIN_CHUNK_SIZE": self.MIN_CHUNK_SIZE,
            "VECTOR_DB_PATH": self.VECTOR_DB_PATH,
            "COLLECTION_NAME": self.COLLECTION_NAME,
            "DEVICE": self.DEVICE,
            "CUDA_VISIBLE_DEVICES": self.CUDA_VISIBLE_DEVICES,
            "LOG_LEVEL": self.LOG_LEVEL,
            "LOG_FILE": self.LOG_FILE,
            "MAX_WORKERS": self.MAX_WORKERS,
            "ENABLE_CHECKPOINTING": self.ENABLE_CHECKPOINTING,
            "CHECKPOINT_INTERVAL": self.CHECKPOINT_INTERVAL,
        }

        # Add model metadata if available
        metadata = self.get_model_metadata()
        if metadata:
            config_dict["MODEL_METADATA"] = {
                "embedding_dimension": metadata.embedding_dimension,
                "max_sequence_length": metadata.max_sequence_length,
                "precision_requirements": metadata.precision_requirements,
                "adapter_class": metadata.adapter_class.__name__,
            }

        return config_dict

    def __repr__(self) -> str:
        """String representation of configuration."""
        dim = self.get_embedding_dimension()
        return f"Config(device={self.DEVICE}, model={self.MODEL_NAME}, dim={dim}, batch_size={self.BATCH_SIZE})"


# Global configuration instance
_config: Optional[Config] = None


def get_config(env_file: Optional[str] = None) -> Config:
    """
    Get global configuration instance (singleton pattern).

    Args:
        env_file: Path to .env file. Only used on first call.

    Returns:
        Config instance
    """
    global _config
    if _config is None:
        _config = Config(env_file)
    return _config


def reset_config() -> None:
    """Reset global configuration (useful for testing)."""
    global _config
    _config = None
