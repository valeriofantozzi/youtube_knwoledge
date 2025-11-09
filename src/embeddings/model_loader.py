"""
Model Loader Module

Loads and initializes the BGE embedding model.
"""

import torch
from typing import Optional
from sentence_transformers import SentenceTransformer
from ..utils.config import get_config
from ..utils.logger import get_default_logger


class ModelLoader:
    """Loads and manages the embedding model."""
    
    def __init__(
        self,
        model_name: Optional[str] = None,
        device: Optional[str] = None,
        cache_dir: Optional[str] = None
    ):
        """
        Initialize model loader.
        
        Args:
            model_name: Name of the model (default from config)
            device: Device to use ('cpu', 'cuda', or None for auto)
            cache_dir: Cache directory for model files
        """
        config = get_config()
        self.model_name = model_name or config.MODEL_NAME
        self.cache_dir = cache_dir or config.MODEL_CACHE_DIR
        self.device = device or config.DEVICE
        self.logger = get_default_logger()
        
        self.model: Optional[SentenceTransformer] = None
        self._model_info: Optional[dict] = None
    
    def load_model(self, enable_compilation: bool = True) -> SentenceTransformer:
        """
        Load the embedding model.
        
        Args:
            enable_compilation: Enable torch.compile() optimization if available
        
        Returns:
            Loaded SentenceTransformer model
        """
        if self.model is not None:
            return self.model
        
        self.logger.info(f"Loading model: {self.model_name}")
        self.logger.info(f"Device: {self.device}")
        
        try:
            # Load model with sentence-transformers
            self.model = SentenceTransformer(
                self.model_name,
                cache_folder=self.cache_dir,
                device=self.device
            )
            
            # Set model to eval mode
            self.model.eval()
            
            # Try to compile model for better performance
            if enable_compilation:
                try:
                    from ..utils.performance_optimizer import get_performance_optimizer
                    optimizer = get_performance_optimizer()
                    if optimizer.should_use_compilation():
                        self.model = optimizer.compile_model(self.model)
                except Exception as e:
                    self.logger.debug(f"Model compilation skipped: {e}")
            
            # Log model information
            self._log_model_info()
            
            self.logger.info("Model loaded successfully")
            
            return self.model
        
        except Exception as e:
            self.logger.error(f"Failed to load model: {e}", exc_info=True)
            raise
    
    def _log_model_info(self) -> None:
        """Log model information."""
        if self.model is None:
            return
        
        # Get model info
        max_seq_length = self.model.max_seq_length
        embedding_dimension = self.model.get_sentence_embedding_dimension()
        
        # Count parameters
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(
            p.numel() for p in self.model.parameters() if p.requires_grad
        )
        
        self._model_info = {
            "model_name": self.model_name,
            "device": self.device,
            "max_seq_length": max_seq_length,
            "embedding_dimension": embedding_dimension,
            "total_parameters": total_params,
            "trainable_parameters": trainable_params,
        }
        
        self.logger.info(f"Model info:")
        self.logger.info(f"  - Max sequence length: {max_seq_length}")
        self.logger.info(f"  - Embedding dimension: {embedding_dimension}")
        self.logger.info(f"  - Total parameters: {total_params:,}")
        self.logger.info(f"  - Trainable parameters: {trainable_params:,}")
    
    def get_model(self) -> SentenceTransformer:
        """
        Get the loaded model (loads if not already loaded).
        
        Returns:
            SentenceTransformer model
        """
        if self.model is None:
            return self.load_model()
        return self.model
    
    def get_model_info(self) -> dict:
        """
        Get model information.
        
        Returns:
            Dictionary with model information
        """
        if self._model_info is None:
            # Load model to get info
            self.load_model()
        return self._model_info.copy() if self._model_info else {}
    
    def health_check(self) -> bool:
        """
        Perform health check on the model.
        
        Returns:
            True if model is healthy, False otherwise
        """
        try:
            if self.model is None:
                self.load_model()
            
            # Test with a simple sentence
            test_sentence = "This is a test sentence."
            embedding = self.model.encode(test_sentence, convert_to_numpy=True)
            
            # Check embedding properties
            if embedding is None or len(embedding) == 0:
                self.logger.error("Health check failed: Empty embedding")
                return False
            
            expected_dim = self.model.get_sentence_embedding_dimension()
            if len(embedding) != expected_dim:
                self.logger.error(
                    f"Health check failed: Wrong embedding dimension. "
                    f"Expected {expected_dim}, got {len(embedding)}"
                )
                return False
            
            # Check for NaN or Inf values
            if not torch.isfinite(torch.tensor(embedding)).all():
                self.logger.error("Health check failed: NaN or Inf values in embedding")
                return False
            
            self.logger.debug("Model health check passed")
            return True
        
        except Exception as e:
            self.logger.error(f"Health check failed: {e}", exc_info=True)
            return False
    
    def unload_model(self) -> None:
        """Unload the model to free memory."""
        if self.model is not None:
            del self.model
            self.model = None
            
            # Clear GPU cache if available
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                torch.mps.empty_cache()
            
            self.logger.info("Model unloaded")


# Global model loader instance
_model_loader: Optional[ModelLoader] = None


def get_model_loader(
    model_name: Optional[str] = None,
    device: Optional[str] = None
) -> ModelLoader:
    """
    Get global model loader instance (singleton pattern).
    
    Args:
        model_name: Model name (only used on first call)
        device: Device (only used on first call)
    
    Returns:
        ModelLoader instance
    """
    global _model_loader
    if _model_loader is None:
        _model_loader = ModelLoader(model_name=model_name, device=device)
    return _model_loader
