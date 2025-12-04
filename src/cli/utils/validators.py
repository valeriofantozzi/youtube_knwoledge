"""
Pydantic models for CLI input validation.

These models ensure all user inputs are valid before being passed to business logic.
"""

from pathlib import Path
from typing import Optional, Dict, Any
from pydantic import BaseModel, Field, field_validator


class LoadCommandInput(BaseModel):
    """Validation model for load command."""

    input_path: Path = Field(..., description="Input file or directory path")
    model: str = Field(
        default="BAAI/bge-large-en-v1.5",
        description="Embedding model name",
    )
    device: str = Field(
        default="auto",
        description="Device to use (auto, cpu, cuda, mps)",
    )
    batch_size: int = Field(
        default=32,
        ge=1,
        le=256,
        description="Batch size for embedding generation",
    )
    chunk_size: int = Field(
        default=512,
        ge=64,
        description="Text chunk size for preprocessing",
    )
    chunk_overlap: int = Field(
        default=50,
        ge=0,
        description="Overlap between chunks",
    )

    @field_validator("input_path")
    @classmethod
    def validate_path(cls, v: Path) -> Path:
        """Validate that input path exists."""
        p = Path(v)
        if not p.exists():
            raise ValueError(f"Path does not exist: {p}")
        return p

    @field_validator("device")
    @classmethod
    def validate_device(cls, v: str) -> str:
        """Validate device value."""
        allowed = {"auto", "cpu", "cuda", "mps"}
        if v not in allowed:
            raise ValueError(f"Device must be one of {allowed}, got {v}")
        return v

    @field_validator("model")
    @classmethod
    def validate_model(cls, v: str) -> str:
        """Validate model name format."""
        if not v or len(v) < 3:
            raise ValueError("Model name must be at least 3 characters")
        return v


class SearchCommandInput(BaseModel):
    """Validation model for search command."""

    query: str = Field(..., min_length=1, max_length=2000, description="Search query")
    model: str = Field(
        default="BAAI/bge-large-en-v1.5",
        description="Embedding model to use",
    )
    top_k: int = Field(
        default=5,
        ge=1,
        le=100,
        description="Number of results to return",
    )
    threshold: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Similarity score threshold",
    )
    output_format: str = Field(
        default="text",
        description="Output format (text, json, csv, table)",
    )
    filters: Optional[Dict[str, str]] = Field(
        default=None,
        description="Metadata filters as JSON",
    )

    @field_validator("output_format")
    @classmethod
    def validate_format(cls, v: str) -> str:
        """Validate output format."""
        allowed = {"text", "json", "csv", "table"}
        if v not in allowed:
            raise ValueError(f"Format must be one of {allowed}, got {v}")
        return v


class AskCommandInput(BaseModel):
    """Validation model for ask command (RAG)."""

    question: str = Field(
        ..., min_length=1, max_length=2000, description="Question to ask"
    )
    model: str = Field(
        default="BAAI/bge-large-en-v1.5",
        description="Embedding model to use",
    )
    top_k: int = Field(
        default=5,
        ge=1,
        le=50,
        description="Number of context documents",
    )
    llm_provider: str = Field(
        default="openai",
        description="LLM provider (openai, anthropic, ollama)",
    )
    temperature: float = Field(
        default=0.7,
        ge=0.0,
        le=2.0,
        description="LLM temperature for sampling",
    )
    show_thinking: bool = Field(
        default=True,
        description="Show thinking process",
    )

    @field_validator("llm_provider")
    @classmethod
    def validate_llm_provider(cls, v: str) -> str:
        """Validate LLM provider."""
        allowed = {"openai", "anthropic", "ollama"}
        if v not in allowed:
            raise ValueError(f"LLM provider must be one of {allowed}, got {v}")
        return v


class ExportCommandInput(BaseModel):
    """Validation model for export command."""

    output_format: str = Field(
        default="json",
        description="Export format (json, csv)",
    )
    output_file: Path = Field(..., description="Output file path")
    model: str = Field(
        default="BAAI/bge-large-en-v1.5",
        description="Embedding model to export",
    )
    include_embeddings: bool = Field(
        default=False,
        description="Include embedding vectors in export",
    )

    @field_validator("output_format")
    @classmethod
    def validate_format(cls, v: str) -> str:
        """Validate export format."""
        allowed = {"json", "csv"}
        if v not in allowed:
            raise ValueError(f"Format must be one of {allowed}, got {v}")
        return v

    @field_validator("output_file")
    @classmethod
    def validate_output_file(cls, v: Path) -> Path:
        """Validate output file path."""
        p = Path(v)
        # Check if parent directory exists
        if not p.parent.exists():
            raise ValueError(f"Parent directory does not exist: {p.parent}")
        return p


class ClusterCommandInput(BaseModel):
    """Validation model for cluster command."""

    model: str = Field(
        default="BAAI/bge-large-en-v1.5",
        description="Embedding model to cluster",
    )
    n_clusters: Optional[int] = Field(
        default=None,
        ge=2,
        description="Number of clusters (auto if None)",
    )
    min_cluster_size: int = Field(
        default=5,
        ge=2,
        description="Minimum cluster size for HDBSCAN",
    )
    output_file: Optional[Path] = Field(
        default=None,
        description="Optional output file for results",
    )


class InfoCommandInput(BaseModel):
    """Validation model for info command."""

    verbose: bool = Field(
        default=False,
        description="Show detailed information",
    )


__all__ = [
    "LoadCommandInput",
    "SearchCommandInput",
    "AskCommandInput",
    "ExportCommandInput",
    "ClusterCommandInput",
    "InfoCommandInput",
]
