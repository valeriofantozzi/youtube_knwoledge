"""
Model Loader Module

Loads and initializes embedding models using the adapter pattern.
Supports multiple model types through ModelAdapter interface.
"""

import torch
from typing import Optional, List, Dict, Any
import torch
from sentence_transformers import SentenceTransformer
from ..utils.config import get_config
from ..utils.logger import get_default_logger
from .model_registry import get_model_registry
from .adapters.base_adapter import ModelAdapter


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

        # Adapter will be created when model is loaded
        self.adapter: Optional[ModelAdapter] = None

        self.model: Optional[SentenceTransformer] = None
        self._model_info: Optional[dict] = None
    
    def _resolve_device(self, device: str) -> str:
        """
        Resolve device string to a valid PyTorch device.
        
        Converts 'auto' to the best available device on the system.
        
        Args:
            device: Device string ('auto', 'cpu', 'cuda', 'mps', etc.)
            
        Returns:
            Valid PyTorch device string
        """
        if device == 'auto':
            # Try to detect best device available
            if torch.cuda.is_available():
                return 'cuda'
            elif torch.backends.mps.is_available():
                return 'mps'
            else:
                return 'cpu'
        return device
    
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
        
        # Resolve device to a valid PyTorch device
        resolved_device = self._resolve_device(self.device)
        self.logger.info(f"Device: {resolved_device}")

        try:
            # Configure PyTorch threads for optimal CPU utilization
            # This is especially important when using CPU for embedding generation
            try:
                from ..utils.performance_optimizer import get_performance_optimizer
                optimizer = get_performance_optimizer()
                optimizer.configure_pytorch_threads()
            except Exception as e:
                self.logger.debug(f"PyTorch thread configuration skipped: {e}")
            
            # Load model with sentence-transformers
            self.model = SentenceTransformer(
                self.model_name,
                cache_folder=self.cache_dir,
                device=resolved_device
            )

            # Set model to eval mode
            self.model.eval()

            # Note: Precision validation happens during encoding, not model loading
            # The adapter will validate embeddings during encode operations

            # Try to compile model for better performance
            if enable_compilation:
                try:
                    from ..utils.performance_optimizer import get_performance_optimizer
                    optimizer = get_performance_optimizer()
                    if optimizer.should_use_compilation():
                        self.model = optimizer.compile_model(self.model)
                except Exception as e:
                    self.logger.debug(f"Model compilation skipped: {e}")

            # Create adapter for this model
            registry = get_model_registry()
            self.adapter = registry.get_adapter(self.model_name, self.model)

            # Check precision compatibility now that we have the adapter
            self._validate_and_warn_precision()

            # Log model information
            self._log_model_info()

            self.logger.info("Model loaded successfully")

            return self.model
        
        except Exception as e:
            self.logger.error(f"Failed to load model: {e}", exc_info=True)

            # Provide helpful error messages for common issues
            error_msg = f"Failed to load model '{self.model_name}': {str(e)}"

            # Check for common error patterns
            if "404" in str(e) or "not found" in str(e).lower():
                error_msg += (
                    f"\n\nModel '{self.model_name}' was not found on HuggingFace Hub. "
                    "Please check:\n"
                    "1. Model name spelling and case sensitivity\n"
                    "2. Model exists at https://huggingface.co/{model_name}\n"
                    "3. You have internet access\n"
                    "4. Your HuggingFace token is valid (if required)"
                )
            elif "connection" in str(e).lower() or "timeout" in str(e).lower():
                error_msg += (
                    f"\n\nNetwork connection issue while downloading '{self.model_name}'. "
                    "Please check:\n"
                    "1. Internet connection\n"
                    "2. Firewall/proxy settings\n"
                    "3. Model size (may require stable connection for large models)"
                )
            elif "disk" in str(e).lower() or "space" in str(e).lower():
                error_msg += (
                    f"\n\nDisk space issue while loading '{self.model_name}'. "
                    "Please check:\n"
                    "1. Available disk space (models can be several GB)\n"
                    "2. Cache directory permissions\n"
                    "3. Clear cache directory if needed"
                )
            elif "cuda" in str(e).lower() or "gpu" in str(e).lower():
                error_msg += (
                    f"\n\nGPU/CUDA issue with model '{self.model_name}'. "
                    "Please check:\n"
                    "1. CUDA installation and version compatibility\n"
                    "2. GPU memory availability\n"
                    "3. Try device='cpu' for CPU-only execution"
                )
            elif "gated repo" in str(e).lower() or "restricted" in str(e).lower() or "authorized list" in str(e).lower():
                # Handle gated model error with fallback
                self.logger.warning(
                    f"Model '{self.model_name}' is gated/restricted. "
                    "Attempting fallback to default model..."
                )

                # Try fallback to default model
                fallback_model = "BAAI/bge-large-en-v1.5"
                self.logger.info(f"Trying fallback model: {fallback_model}")

                try:
                    # Temporarily change model name and try again
                    original_model_name = self.model_name
                    self.model_name = fallback_model

                    # Reset model state
                    self.model = None
                    self._model_info = None

                    # Retry loading with fallback model
                    result = self.load_model()
                    self.logger.info(
                        f"Successfully fell back to model '{fallback_model}' "
                        f"from failed model '{original_model_name}'"
                    )
                    return result

                except Exception as fallback_e:
                    # Fallback also failed
                    error_msg = (
                        f"Failed to load model '{original_model_name}': {str(e)}\n"
                        f"Fallback to '{fallback_model}' also failed: {str(fallback_e)}\n\n"
                        f"Model '{original_model_name}' is a gated/restricted model. "
                        "To access it:\n"
                        "1. Visit: https://huggingface.co/google/embeddinggemma-300m\n"
                        "2. Click 'Request access' in the model description\n"
                        "3. Fill out the access request form explaining your use case\n"
                        "4. Wait for Google approval (may take several days)\n"
                        f"5. Alternative: Use public models like 'BAAI/bge-large-en-v1.5' or 'BAAI/bge-base-en'"
                    )
                    raise RuntimeError(error_msg) from fallback_e
            elif "memory" in str(e).lower() or "ram" in str(e).lower():
                error_msg += (
                    f"\n\nMemory issue while loading '{self.model_name}'. "
                    "Please check:\n"
                    "1. Available RAM (models need 2-8GB+ RAM)\n"
                    "2. Close other applications\n"
                    "3. Try a smaller model or CPU mode"
                )

            # Re-raise with enhanced error message
            raise RuntimeError(error_msg) from e
    
    def _log_model_info(self) -> None:
        """Log model information."""
        if self.model is None:
            return

        # Get model info from adapter (adapter knows model-specific details)
        max_seq_length = self.adapter.get_max_sequence_length()
        embedding_dimension = self.adapter.get_embedding_dimension()
        precision_requirements = self.adapter.get_precision_requirements()

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
            "precision_requirements": precision_requirements,
            "total_parameters": total_params,
            "trainable_parameters": trainable_params,
        }

        self.logger.info(f"Model info:")
        self.logger.info(f"  - Model type: {self.adapter.__class__.__name__}")
        self.logger.info(f"  - Max sequence length: {max_seq_length}")
        self.logger.info(f"  - Embedding dimension: {embedding_dimension}")
        self.logger.info(f"  - Precision requirements: {precision_requirements}")
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

    def get_actual_model_name(self) -> str:
        """
        Get the actual model name that was loaded (after any fallback).

        Returns:
            The model name that was actually loaded
        """
        return self.model_name
    
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
            
            expected_dim = self.adapter.get_embedding_dimension()
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

    def _check_precision_compatibility(self) -> Dict[str, Any]:
        """
        Check precision compatibility for the current model and device setup.

        Returns:
            Dictionary with compatibility information and warnings
        """
        result = {
            "compatible": True,
            "warnings": [],
            "device_info": {},
            "precision_info": {}
        }

        try:
            # Get model precision requirements
            precision_requirements = self.adapter.get_precision_requirements()
            result["precision_info"]["requirements"] = precision_requirements
            result["precision_info"]["device"] = self.device

            # Check device capabilities
            if self.device == "cuda" or (self.device == "auto" and torch.cuda.is_available()):
                if torch.cuda.is_available():
                    device_props = torch.cuda.get_device_properties(0)
                    result["device_info"]["cuda_available"] = True
                    result["device_info"]["cuda_version"] = torch.version.cuda
                    result["device_info"]["gpu_name"] = device_props.name

                    # Check for bfloat16 support (required for EmbeddingGemma)
                    if "bfloat16" in precision_requirements:
                        if not torch.cuda.is_bf16_supported():
                            result["warnings"].append(
                                f"GPU {device_props.name} does not support bfloat16, "
                                "but model requires it. Performance may be degraded."
                            )
                            result["compatible"] = False
                else:
                    result["warnings"].append("CUDA device requested but not available")
                    result["compatible"] = False

            elif self.device == "mps" or (self.device == "auto" and hasattr(torch.backends, "mps") and torch.backends.mps.is_available()):
                if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                    result["device_info"]["mps_available"] = True

                    # MPS has limited precision support
                    if "bfloat16" in precision_requirements:
                        result["warnings"].append(
                            "MPS (Apple Silicon) has limited bfloat16 support. "
                            "Consider using float32 or CPU."
                        )
                else:
                    result["warnings"].append("MPS device requested but not available")
                    result["compatible"] = False

            else:
                # CPU mode
                result["device_info"]["cpu_mode"] = True

                if len(precision_requirements) > 1:
                    result["warnings"].append(
                        f"Model supports multiple precisions {precision_requirements} "
                        "but CPU will use float32. This is normal and expected."
                    )

            # Check PyTorch version for precision support
            torch_version = torch.__version__
            if "bfloat16" in precision_requirements and torch_version.startswith("1."):
                result["warnings"].append(
                    f"PyTorch {torch_version} has limited bfloat16 support. "
                    "Consider upgrading to PyTorch 2.0+ for better performance."
                )

        except Exception as e:
            result["warnings"].append(f"Could not check precision compatibility: {e}")

        return result

    def _validate_and_warn_precision(self) -> None:
        """
        Validate precision compatibility and log warnings if needed.
        Called during model loading to provide early feedback.
        """
        try:
            compat_info = self._check_precision_compatibility()

            if not compat_info["compatible"]:
                self.logger.warning("Precision compatibility issues detected:")
                for warning in compat_info["warnings"]:
                    self.logger.warning(f"  - {warning}")

            elif compat_info["warnings"]:
                self.logger.info("Precision compatibility notes:")
                for warning in compat_info["warnings"]:
                    self.logger.info(f"  - {warning}")

            # Log device and precision info
            device_info = compat_info["device_info"]
            if device_info:
                self.logger.debug(f"Device info: {device_info}")

        except Exception as e:
            self.logger.debug(f"Precision validation failed: {e}")
            # Don't fail loading for precision check issues


def get_model_loader(
    model_name: Optional[str] = None,
    device: Optional[str] = None,
    cache_dir: Optional[str] = None
) -> ModelLoader:
    """
    Get model loader instance using the global model manager.

    This replaces the singleton pattern with a multi-model caching system.
    If no model_name is provided, uses the default from config.

    Args:
        model_name: Model name (defaults to config if None)
        device: Device to use for the model
        cache_dir: Cache directory for model files

    Returns:
        ModelLoader instance for the requested model
    """
    from ..utils.config import get_config
    from .model_manager import get_model_manager

    config = get_config()
    model_name = model_name or config.MODEL_NAME

    manager = get_model_manager()
    return manager.get_model_loader(
        model_name=model_name,
        device=device,
        cache_dir=cache_dir
    )
