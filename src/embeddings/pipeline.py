"""
Embedding Pipeline Module

Orchestrates the complete embedding generation pipeline.
"""

import numpy as np
from pathlib import Path
from typing import List, Dict, Optional, Tuple
import json
import pickle

from .model_loader import ModelLoader, get_model_loader
from .embedder import Embedder
from .batch_processor import BatchProcessor
from ..preprocessing.pipeline import ProcessedDocument
from ..utils.logger import get_default_logger
from ..utils.config import get_config


class EmbeddingPipeline:
    """Main embedding generation pipeline."""
    
    def __init__(
        self,
        model_name: Optional[str] = None,
        device: Optional[str] = None,
        batch_size: Optional[int] = None,
        enable_optimizations: bool = True
    ):
        """
        Initialize embedding pipeline.
        
        Args:
            model_name: Model name (default from config)
            device: Device to use (default from config or auto-detect)
            batch_size: Batch size (default from config or auto-optimize)
            enable_optimizations: Enable hardware-aware optimizations
        """
        self.config = get_config()
        self.logger = get_default_logger()
        self.enable_optimizations = enable_optimizations
        
        # Log hardware profile if optimizations enabled
        if enable_optimizations:
            try:
                from ..utils.hardware_detector import get_hardware_detector
                hardware_detector = get_hardware_detector()
                hardware_detector.print_profile()
            except Exception as e:
                self.logger.debug(f"Could not print hardware profile: {e}")
        
        # Initialize components
        self.model_loader = get_model_loader(
            model_name=model_name,
            device=device
        )
        self.embedder = Embedder(model_loader=self.model_loader)
        self.batch_processor = BatchProcessor(
            embedder=self.embedder,
            batch_size=batch_size,
            use_dynamic_batch_size=enable_optimizations
        )
    
    def generate_embeddings(
        self,
        processed_document: ProcessedDocument,
        show_progress: bool = True
    ) -> Tuple[np.ndarray, List[Dict]]:
        """
        Generate embeddings for a processed document.
        
        Args:
            processed_document: ProcessedDocument object with chunks
            show_progress: Show progress bar
        
        Returns:
            Tuple of (embeddings array, metadata list)
        """
        if not processed_document.chunks:
            self.logger.warning("No chunks to process")
            return np.array([]), []
        
        self.logger.info(
            f"Generating embeddings for {len(processed_document.chunks)} chunks "
            f"from {processed_document.metadata.filename}"
        )
        
        # Generate embeddings
        embeddings, metadata = self.batch_processor.process_chunks(
            processed_document.chunks,
            show_progress=show_progress
        )
        
        # Validate embeddings
        self._validate_embeddings(embeddings, len(processed_document.chunks))
        
        self.logger.info(
            f"Generated {len(embeddings)} embeddings "
            f"(shape: {embeddings.shape})"
        )

        # Get actual model name (after any fallback)
        actual_model_name = self.model_loader.get_actual_model_name()

        return embeddings, metadata, actual_model_name
    
    def generate_embeddings_batch(
        self,
        processed_documents: List[ProcessedDocument],
        show_progress: bool = True
    ) -> List[Tuple[np.ndarray, List[Dict]]]:
        """
        Generate embeddings for multiple processed documents.
        
        Args:
            processed_documents: List of ProcessedDocument objects
            show_progress: Show progress bar
        
        Returns:
            List of (embeddings, metadata) tuples
        """
        results = []
        
        for i, processed_document in enumerate(processed_documents):
            self.logger.info(
                f"Processing document {i+1}/{len(processed_documents)}: "
                f"{processed_document.metadata.filename}"
            )
            
            embeddings, metadata = self.generate_embeddings(
                processed_document,
                show_progress=show_progress
            )
            
            results.append((embeddings, metadata))
        
        return results
    
    def _validate_embeddings(
        self,
        embeddings: np.ndarray,
        expected_count: int
    ) -> None:
        """
        Validate generated embeddings.
        
        Args:
            embeddings: Embeddings array
            expected_count: Expected number of embeddings
        """
        if len(embeddings) != expected_count:
            raise ValueError(
                f"Embedding count mismatch: expected {expected_count}, "
                f"got {len(embeddings)}"
            )
        
        # Check for NaN or Inf
        if not np.isfinite(embeddings).all():
            nan_count = np.isnan(embeddings).sum()
            inf_count = np.isinf(embeddings).sum()
            raise ValueError(
                f"Invalid values in embeddings: {nan_count} NaN, {inf_count} Inf"
            )
        
        # Check embedding dimension
        expected_dim = self.embedder.get_embedding_dimension()
        if embeddings.shape[-1] != expected_dim:
            raise ValueError(
                f"Wrong embedding dimension: expected {expected_dim}, "
                f"got {embeddings.shape[-1]}"
            )
    
    def save_embeddings(
        self,
        embeddings: np.ndarray,
        metadata: List[Dict],
        output_path: Path,
        format: str = "numpy"
    ) -> Path:
        """
        Save embeddings to disk.
        
        Args:
            embeddings: Embeddings array
            metadata: Metadata list
            output_path: Output file path
            format: Format to save ("numpy" or "json")
        
        Returns:
            Path to saved file
        """
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        if format == "numpy":
            # Save as .npz file
            np.savez_compressed(
                output_path,
                embeddings=embeddings,
                metadata=metadata
            )
        elif format == "json":
            # Convert to JSON-serializable format
            data = {
                "embeddings": embeddings.tolist(),
                "metadata": metadata,
                "shape": list(embeddings.shape),
            }
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2)
        else:
            raise ValueError(f"Unsupported format: {format}")
        
        self.logger.info(f"Saved embeddings to {output_path}")
        return output_path
    
    def load_embeddings(
        self,
        input_path: Path,
        format: str = "numpy"
    ) -> Tuple[np.ndarray, List[Dict], str]:
        """
        Load embeddings from disk.

        Args:
            input_path: Input file path
            format: Format to load ("numpy" or "json")

        Returns:
            Tuple of (embeddings array, metadata list, model name)
        """
        if format == "numpy":
            data = np.load(input_path, allow_pickle=True)
            embeddings = data['embeddings']
            metadata = data['metadata'].tolist()
        elif format == "json":
            with open(input_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            embeddings = np.array(data['embeddings'])
            metadata = data['metadata']
        else:
            raise ValueError(f"Unsupported format: {format}")

        # Return current model name since we don't know what model created the cached embeddings
        actual_model_name = self.model_loader.get_actual_model_name()

        return embeddings, metadata, actual_model_name
    
    def process_and_save(
        self,
        processed_document: ProcessedDocument,
        output_dir: Path,
        save_format: str = "numpy",
        show_progress: bool = True
    ) -> Path:
        """
        Process video and save embeddings.
        
        Args:
            processed_document: ProcessedDocument object
            output_dir: Output directory
            save_format: Format to save ("numpy" or "json")
            show_progress: Show progress bar
        
        Returns:
            Path to saved file
        """
        # Generate embeddings
        embeddings, metadata = self.generate_embeddings(
            processed_document,
            show_progress=show_progress
        )
        
        # Save embeddings
        output_file = output_dir / f"{processed_document.metadata.source_id}_embeddings.{save_format if save_format == 'json' else 'npz'}"
        self.save_embeddings(embeddings, metadata, output_file, save_format)
        
        return output_file
    
    def generate_embeddings_with_checkpointing(
        self,
        processed_document: ProcessedDocument,
        checkpoint_dir: Optional[Path] = None,
        checkpoint_interval: Optional[int] = None,
        show_progress: bool = True,
        resume: bool = True
    ) -> Tuple[np.ndarray, List[Dict], str]:
        """
        Generate embeddings with checkpointing support.

        Args:
            processed_document: ProcessedDocument object with chunks
            checkpoint_dir: Directory to save checkpoints (default: data/checkpoints)
            checkpoint_interval: Save checkpoint every N chunks (default from config)
            show_progress: Show progress bar
            resume: If True, resume from checkpoint if available

        Returns:
            Tuple of (embeddings array, metadata list, actual model name used)
        """
        checkpoint_dir = checkpoint_dir or Path(getattr(self.config, "CHECKPOINT_DIR", "./data/checkpoints"))
        checkpoint_interval = checkpoint_interval or self.config.CHECKPOINT_INTERVAL
        
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        checkpoint_file = checkpoint_dir / f"{processed_document.metadata.source_id}_checkpoint.pkl"
        
        # Try to resume from checkpoint
        start_idx = 0
        accumulated_embeddings = []
        accumulated_metadata = []
        
        if resume and checkpoint_file.exists():
            try:
                self.logger.info(f"Resuming from checkpoint: {checkpoint_file}")
                checkpoint_data = self._load_checkpoint(checkpoint_file)
                checkpoint_processed_count = checkpoint_data['processed_count']
                checkpoint_embeddings = checkpoint_data['embeddings']
                checkpoint_metadata = checkpoint_data['metadata']
                checkpoint_source_id = checkpoint_data.get('source_id', '')
                
                # Validate checkpoint belongs to this source
                if checkpoint_source_id != processed_document.metadata.source_id:
                    self.logger.warning(
                        f"Checkpoint belongs to different source (checkpoint: {checkpoint_source_id}, "
                        f"current: {processed_document.metadata.source_id}). Starting from beginning."
                    )
                    checkpoint_file.unlink()
                    start_idx = 0
                    accumulated_embeddings = []
                    accumulated_metadata = []
                # Validate checkpoint data consistency (must match current chunk count)
                elif len(checkpoint_embeddings) != len(processed_document.chunks):
                    self.logger.warning(
                        f"Checkpoint chunk count mismatch (checkpoint: {len(checkpoint_embeddings)}, "
                        f"current: {len(processed_document.chunks)}). This suggests the SRT file or chunking "
                        f"logic changed. Discarding checkpoint and starting from beginning."
                    )
                    checkpoint_file.unlink()
                    start_idx = 0
                    accumulated_embeddings = []
                    accumulated_metadata = []
                # Validate checkpoint processed count doesn't exceed current chunk count
                elif checkpoint_processed_count > len(processed_document.chunks):
                    self.logger.warning(
                        f"Checkpoint processed_count ({checkpoint_processed_count}) exceeds "
                        f"current chunk count ({len(processed_document.chunks)}). Starting from beginning."
                    )
                    checkpoint_file.unlink()
                    start_idx = 0
                    accumulated_embeddings = []
                    accumulated_metadata = []
                # Validate checkpoint model compatibility
                elif 'model_name' in checkpoint_data:
                    checkpoint_model = checkpoint_data['model_name']
                    current_model = self.model_loader.get_actual_model_name()
                    if checkpoint_model != current_model:
                        self.logger.warning(
                            f"Checkpoint created with different model (checkpoint: {checkpoint_model}, "
                            f"current: {current_model}). Discarding checkpoint and starting from beginning."
                        )
                        checkpoint_file.unlink()
                        start_idx = 0
                        accumulated_embeddings = []
                        accumulated_metadata = []
                    # Also validate embedding dimensions
                    elif checkpoint_embeddings and len(checkpoint_embeddings) > 0:
                        checkpoint_dim = len(checkpoint_embeddings[0]) if isinstance(checkpoint_embeddings[0], (list, np.ndarray)) else 0
                        expected_dim = self.embedder.get_embedding_dimension()
                        if checkpoint_dim != expected_dim:
                            self.logger.warning(
                                f"Checkpoint embedding dimension mismatch (checkpoint: {checkpoint_dim}, "
                                f"expected: {expected_dim}). Discarding checkpoint and starting from beginning."
                            )
                            checkpoint_file.unlink()
                            start_idx = 0
                            accumulated_embeddings = []
                            accumulated_metadata = []
                        else:
                            start_idx = checkpoint_processed_count
                            accumulated_embeddings = checkpoint_embeddings
                            accumulated_metadata = checkpoint_metadata
                            self.logger.info(f"Resuming from chunk {start_idx}/{len(processed_document.chunks)}")
                    else:
                        start_idx = checkpoint_processed_count
                        accumulated_embeddings = checkpoint_embeddings
                        accumulated_metadata = checkpoint_metadata
                        self.logger.info(f"Resuming from chunk {start_idx}/{len(processed_document.chunks)}")
                # Validate internal checkpoint consistency (processed_count vs embeddings count)
                elif len(checkpoint_embeddings) != checkpoint_processed_count:
                    self.logger.warning(
                        f"Checkpoint data inconsistent: processed_count={checkpoint_processed_count}, "
                        f"but embeddings count={len(checkpoint_embeddings)}. Starting from beginning."
                    )
                    checkpoint_file.unlink()
                    start_idx = 0
                    accumulated_embeddings = []
                    accumulated_metadata = []
                # All validations passed - use checkpoint
                else:
                    start_idx = checkpoint_processed_count
                    accumulated_embeddings = checkpoint_embeddings
                    accumulated_metadata = checkpoint_metadata
                    self.logger.info(f"Resuming from chunk {start_idx}/{len(processed_document.chunks)}")
            except Exception as e:
                self.logger.warning(f"Failed to load checkpoint: {e}. Starting from beginning.")
                if checkpoint_file.exists():
                    checkpoint_file.unlink()
                start_idx = 0
                accumulated_embeddings = []
                accumulated_metadata = []
        
        # Process remaining chunks
        if start_idx < len(processed_document.chunks):
            remaining_chunks = processed_document.chunks[start_idx:]
            remaining_embeddings, remaining_metadata = self.batch_processor.process_chunks(
                remaining_chunks,
                show_progress=show_progress
            )
            
            # Combine with accumulated results
            if accumulated_embeddings:
                embeddings = np.vstack([np.array(accumulated_embeddings), remaining_embeddings])
                metadata = accumulated_metadata + remaining_metadata
            else:
                embeddings = remaining_embeddings
                metadata = remaining_metadata
            
            # Save checkpoint periodically (only save once per run, at the end of processing)
            # Intermediate checkpoints are saved after processing remaining chunks
            # This avoids saving multiple checkpoints in the same run
        else:
            # All chunks already processed
            embeddings = np.array(accumulated_embeddings) if accumulated_embeddings else np.array([])
            metadata = accumulated_metadata
        
        # Validate embeddings before saving final checkpoint
        if len(embeddings) != len(processed_document.chunks):
            self.logger.error(
                f"Embedding count mismatch before validation: expected {len(processed_document.chunks)}, "
                f"got {len(embeddings)}. start_idx={start_idx}, accumulated={len(accumulated_embeddings) if accumulated_embeddings else 0}, "
                f"remaining={len(remaining_embeddings) if 'remaining_embeddings' in locals() else 'N/A'}"
            )
            # Clear checkpoint and retry from beginning
            if checkpoint_file.exists():
                checkpoint_file.unlink()
                self.logger.info("Cleared corrupted checkpoint, retrying from beginning")
            raise ValueError(
                f"Embedding count mismatch: expected {len(processed_document.chunks)}, "
                f"got {len(embeddings)}"
            )
        
        # Final checkpoint
        if embeddings.size > 0:
            checkpoint_data = {
                'processed_count': len(processed_document.chunks),
                'embeddings': embeddings.tolist() if isinstance(embeddings, np.ndarray) else embeddings,
                'metadata': metadata,
                'source_id': processed_document.metadata.source_id,
                'model_name': self.model_loader.get_actual_model_name(),
                'embedding_dimension': self.embedder.get_embedding_dimension()
            }
            self._save_checkpoint(checkpoint_file, checkpoint_data)
            self.logger.info("Saved final checkpoint")
        
        # Validate embeddings
        self._validate_embeddings(embeddings, len(processed_document.chunks))

        # Get actual model name (after any fallback)
        actual_model_name = self.model_loader.get_actual_model_name()

        return embeddings, metadata, actual_model_name
    
    def _save_checkpoint(self, checkpoint_file: Path, data: Dict) -> None:
        """Save checkpoint data to file."""
        try:
            with open(checkpoint_file, 'wb') as f:
                pickle.dump(data, f)
        except Exception as e:
            self.logger.error(f"Failed to save checkpoint: {e}")
    
    def _load_checkpoint(self, checkpoint_file: Path) -> Dict:
        """Load checkpoint data from file."""
        with open(checkpoint_file, 'rb') as f:
            return pickle.load(f)
    
    def clear_checkpoint(self, source_id: str, checkpoint_dir: Optional[Path] = None) -> None:
        """
        Clear checkpoint for a source.
        
        Args:
            source_id: Source ID
            checkpoint_dir: Checkpoint directory (default from config)
        """
        checkpoint_dir = checkpoint_dir or Path(self.config.get("CHECKPOINT_DIR", "./data/checkpoints"))
        checkpoint_file = checkpoint_dir / f"{source_id}_checkpoint.pkl"
        
        if checkpoint_file.exists():
            checkpoint_file.unlink()
            self.logger.info(f"Cleared checkpoint for {source_id}")

