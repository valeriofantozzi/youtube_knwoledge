"""
Configuration Module

Manages application configuration with environment variable support and validation.
"""

import os
from pathlib import Path
from typing import Optional, Literal
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
        self.MODEL_NAME = os.getenv("MODEL_NAME", "BAAI/bge-large-en-v1.5")
        self.MODEL_CACHE_DIR = os.path.expanduser(
            os.getenv("MODEL_CACHE_DIR", "~/.cache/huggingface")
        )
        
        # Processing Configuration
        self.BATCH_SIZE = self._get_int("BATCH_SIZE", 128)
        self.CHUNK_SIZE = self._get_int("CHUNK_SIZE", 300)
        self.CHUNK_OVERLAP = self._get_int("CHUNK_OVERLAP", 60)
        self.MIN_CHUNK_SIZE = self._get_int("MIN_CHUNK_SIZE", 50)
        
        # Vector Database Configuration
        self.VECTOR_DB_PATH = os.getenv("VECTOR_DB_PATH", "./data/vector_db")
        self.COLLECTION_NAME = os.getenv("COLLECTION_NAME", "subtitle_embeddings")
        
        # Device Configuration
        device = os.getenv("DEVICE", "auto").lower()
        if device == "auto":
            import torch
            self.DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
        elif device in ["cpu", "cuda"]:
            self.DEVICE = device
        else:
            raise ValueError(f"Invalid DEVICE value: {device}. Must be 'auto', 'cpu', or 'cuda'")
        
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
        self.MAX_WORKERS = self._get_int("MAX_WORKERS", 4)
        self.ENABLE_CHECKPOINTING = self._get_bool("ENABLE_CHECKPOINTING", True)
        self.CHECKPOINT_INTERVAL = self._get_int("CHECKPOINT_INTERVAL", 1000)
        
        # Validate configuration
        self._validate()
    
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
            raise ValueError(
                f"CHUNK_OVERLAP must be >= 0, got {self.CHUNK_OVERLAP}"
            )
        
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
    
    def to_dict(self) -> dict:
        """Convert configuration to dictionary."""
        return {
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
    
    def __repr__(self) -> str:
        """String representation of configuration."""
        return f"Config(device={self.DEVICE}, model={self.MODEL_NAME}, batch_size={self.BATCH_SIZE})"


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
