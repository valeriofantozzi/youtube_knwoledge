"""
Embedder Module

Generates embeddings from text chunks.
"""

import numpy as np
from typing import List, Union, Optional
from sentence_transformers import SentenceTransformer

from .model_loader import ModelLoader, get_model_loader
from ..utils.logger import get_default_logger


class Embedder:
    """Generates embeddings from text."""
    
    # Instruction prefix for queries (BGE models support instruction-based queries)
    QUERY_INSTRUCTION = "Represent this sentence for searching relevant passages: "
    
    # Instruction prefix for documents (optional, can be empty)
    DOCUMENT_INSTRUCTION = ""
    
    def __init__(
        self,
        model_loader: Optional[ModelLoader] = None,
        normalize_embeddings: bool = True
    ):
        """
        Initialize embedder.
        
        Args:
            model_loader: ModelLoader instance (creates new if None)
            normalize_embeddings: Whether to normalize embeddings (BGE models use normalized embeddings)
        """
        self.model_loader = model_loader or get_model_loader()
        self.normalize_embeddings = normalize_embeddings
        self.logger = get_default_logger()
        self._model: Optional[SentenceTransformer] = None
    
    @property
    def model(self) -> SentenceTransformer:
        """Get model (lazy loading)."""
        if self._model is None:
            self._model = self.model_loader.get_model()
        return self._model
    
    def encode(
        self,
        texts: Union[str, List[str]],
        is_query: bool = False,
        show_progress: bool = False,
        batch_size: Optional[int] = None
    ) -> np.ndarray:
        """
        Generate embeddings for text(s).
        
        Args:
            texts: Single text string or list of texts
            is_query: If True, prepend query instruction prefix
            show_progress: Show progress bar
            batch_size: Batch size for encoding (default from config)
        
        Returns:
            Numpy array of embeddings (shape: [num_texts, embedding_dim])
        """
        # Convert single string to list
        if isinstance(texts, str):
            texts = [texts]
        
        if not texts:
            return np.array([])
        
        # Validate inputs
        self._validate_inputs(texts)
        
        # Prepare texts with instruction prefix if needed
        if is_query:
            texts = [self.QUERY_INSTRUCTION + text for text in texts]
        elif self.DOCUMENT_INSTRUCTION:
            texts = [self.DOCUMENT_INSTRUCTION + text for text in texts]
        
        # Get batch size from config if not provided
        if batch_size is None:
            from ..utils.config import get_config
            batch_size = get_config().BATCH_SIZE
        
        # Generate embeddings
        try:
            embeddings = self.model.encode(
                texts,
                batch_size=batch_size,
                show_progress_bar=show_progress,
                convert_to_numpy=True,
                normalize_embeddings=self.normalize_embeddings
            )
            
            # Validate embeddings
            self._validate_embeddings(embeddings)
            
            return embeddings
        
        except Exception as e:
            self.logger.error(f"Error generating embeddings: {e}", exc_info=True)
            raise
    
    def encode_single(self, text: str, is_query: bool = False) -> np.ndarray:
        """
        Generate embedding for a single text.
        
        Args:
            text: Text string
            is_query: If True, prepend query instruction prefix
        
        Returns:
            Numpy array of embedding (shape: [embedding_dim])
        """
        embedding = self.encode([text], is_query=is_query)
        return embedding[0]
    
    def _validate_inputs(self, texts: List[str]) -> None:
        """
        Validate input texts.
        
        Args:
            texts: List of texts
        
        Raises:
            ValueError: If validation fails
        """
        if not texts:
            raise ValueError("Empty text list")
        
        # Check for empty strings
        empty_texts = [i for i, text in enumerate(texts) if not text or not text.strip()]
        if empty_texts:
            self.logger.warning(f"Found {len(empty_texts)} empty texts at indices: {empty_texts[:10]}")
        
        # Check text length (warn if very long)
        max_length = self.model.max_seq_length
        long_texts = []
        for i, text in enumerate(texts):
            # Rough token estimate (words)
            word_count = len(text.split())
            if word_count > max_length * 0.8:  # Warn if >80% of max length
                long_texts.append((i, word_count))
        
        if long_texts:
            self.logger.warning(
                f"Found {len(long_texts)} texts that may exceed max length "
                f"({max_length} tokens)"
            )
    
    def _validate_embeddings(self, embeddings: np.ndarray) -> None:
        """
        Validate generated embeddings.
        
        Args:
            embeddings: Embedding array
        
        Raises:
            ValueError: If validation fails
        """
        if embeddings is None:
            raise ValueError("Embeddings are None")
        
        if len(embeddings) == 0:
            raise ValueError("Empty embeddings array")
        
        # Check for NaN or Inf values
        if not np.isfinite(embeddings).all():
            nan_count = np.isnan(embeddings).sum()
            inf_count = np.isinf(embeddings).sum()
            raise ValueError(
                f"Embeddings contain invalid values: {nan_count} NaN, {inf_count} Inf"
            )
        
        # Check embedding dimension
        expected_dim = self.model.get_sentence_embedding_dimension()
        if embeddings.shape[-1] != expected_dim:
            raise ValueError(
                f"Wrong embedding dimension: expected {expected_dim}, "
                f"got {embeddings.shape[-1]}"
            )
        
        # Check normalization if enabled
        if self.normalize_embeddings:
            norms = np.linalg.norm(embeddings, axis=-1)
            # Allow small tolerance for floating point errors
            if not np.allclose(norms, 1.0, atol=1e-5):
                self.logger.warning(
                    f"Some embeddings are not properly normalized. "
                    f"Norm range: [{norms.min():.6f}, {norms.max():.6f}]"
                )
    
    def get_embedding_dimension(self) -> int:
        """
        Get embedding dimension.
        
        Returns:
            Embedding dimension
        """
        return self.model.get_sentence_embedding_dimension()
    
    def get_max_sequence_length(self) -> int:
        """
        Get maximum sequence length.
        
        Returns:
            Maximum sequence length in tokens
        """
        return self.model.max_seq_length
