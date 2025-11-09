"""
Embeddings module for generating vector representations of text.

Handles model loading, embedding generation, and batch processing.
"""

from .model_loader import ModelLoader, get_model_loader
from .embedder import Embedder
from .batch_processor import BatchProcessor
from .pipeline import EmbeddingPipeline

__all__ = [
    "ModelLoader",
    "get_model_loader",
    "Embedder",
    "BatchProcessor",
    "EmbeddingPipeline",
]
