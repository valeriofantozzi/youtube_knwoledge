"""
Batch Processor Module

Handles batch processing of embeddings with progress tracking and error handling.
"""

import numpy as np
from typing import List, Optional, Callable, Tuple
from tqdm import tqdm
import time
import gc
import psutil
import os

from .embedder import Embedder
from ..utils.config import get_config
from ..utils.logger import get_default_logger


class BatchProcessor:
    """Processes embeddings in batches with progress tracking."""
    
    def __init__(
        self,
        embedder: Optional[Embedder] = None,
        batch_size: Optional[int] = None,
        use_dynamic_batch_size: bool = True
    ):
        """
        Initialize batch processor.
        
        Args:
            embedder: Embedder instance (creates new if None)
            batch_size: Batch size for processing (default from config or auto-optimized)
            use_dynamic_batch_size: If True, optimize batch size based on hardware
        """
        self.embedder = embedder or Embedder()
        self.config = get_config()
        
        # Use dynamic batch size optimization if enabled and batch_size not explicitly set
        if batch_size is None and use_dynamic_batch_size:
            try:
                from ..utils.performance_optimizer import get_performance_optimizer
                optimizer = get_performance_optimizer()
                self.batch_size = optimizer.get_optimal_batch_size()
            except Exception:
                self.batch_size = self.config.BATCH_SIZE
        else:
            self.batch_size = batch_size or self.config.BATCH_SIZE
        
        self.use_dynamic_batch_size = use_dynamic_batch_size
        self.logger = get_default_logger()
    
    def process_chunks(
        self,
        chunks: List,
        show_progress: bool = True,
        text_extractor: Optional[Callable] = None
    ) -> Tuple[np.ndarray, List[dict]]:
        """
        Process chunks and generate embeddings.
        
        Args:
            chunks: List of chunk objects with 'text' attribute
            show_progress: Show progress bar
            text_extractor: Optional function to extract text from chunk
        
        Returns:
            Tuple of (embeddings array, metadata list)
        """
        if not chunks:
            return np.array([]), []
        
        # Extract texts from chunks
        if text_extractor:
            texts = [text_extractor(chunk) for chunk in chunks]
        else:
            texts = [chunk.text for chunk in chunks]
        
        # Process in batches
        embeddings, metadata = self._process_batches(
            texts,
            chunks,
            show_progress=show_progress
        )
        
        return embeddings, metadata
    
    def _process_batches(
        self,
        texts: List[str],
        chunks: List,
        show_progress: bool = True
    ) -> Tuple[np.ndarray, List[dict]]:
        """
        Process texts in batches.
        
        Args:
            texts: List of text strings
            chunks: List of chunk objects (for metadata)
            show_progress: Show progress bar
        
        Returns:
            Tuple of (embeddings array, metadata list)
        """
        all_embeddings = []
        all_metadata = []
        
        num_batches = (len(texts) + self.batch_size - 1) // self.batch_size
        
        if show_progress:
            pbar = tqdm(
                total=len(texts),
                desc="Generating embeddings",
                unit="chunks"
            )
        
        start_time = time.time()
        processed_count = 0
        
        for batch_idx in range(num_batches):
            start_idx = batch_idx * self.batch_size
            end_idx = min(start_idx + self.batch_size, len(texts))
            
            batch_texts = texts[start_idx:end_idx]
            batch_chunks = chunks[start_idx:end_idx]
            
            try:
                # Generate embeddings for batch
                batch_embeddings = self.embedder.encode(
                    batch_texts,
                    is_query=False,
                    show_progress=False,
                    batch_size=self.batch_size
                )
                
                # Collect embeddings
                all_embeddings.append(batch_embeddings)
                
                # Collect metadata
                for chunk in batch_chunks:
                    metadata = {
                        "chunk_id": getattr(chunk, 'chunk_id', None),
                        "chunk_index": getattr(chunk, 'chunk_index', None),
                        "token_count": getattr(chunk, 'token_count', None),
                        "metadata": getattr(chunk, 'metadata', {}),
                    }
                    all_metadata.append(metadata)
                
                processed_count += len(batch_texts)
                
                # Update progress
                if show_progress:
                    pbar.update(len(batch_texts))
                
                # Memory management: clear cache periodically (adaptive interval)
                cache_interval = self._get_cache_clear_interval()
                if (batch_idx + 1) % cache_interval == 0:
                    self._clear_cache_if_needed()
                
                # Log progress periodically
                if (batch_idx + 1) % 10 == 0:
                    elapsed = time.time() - start_time
                    rate = processed_count / elapsed if elapsed > 0 else 0
                    memory_usage = self._get_memory_usage()
                    self.logger.debug(
                        f"Processed {processed_count}/{len(texts)} chunks "
                        f"({rate:.1f} chunks/sec, {memory_usage:.1f}MB memory)"
                    )
            
            except Exception as e:
                self.logger.error(
                    f"Error processing batch {batch_idx}: {e}",
                    exc_info=True
                )
                # Skip failed batch or re-raise based on configuration
                # For now, we'll skip and continue
                continue
        
        if show_progress:
            pbar.close()
        
        # Combine all embeddings
        if all_embeddings:
            embeddings = np.vstack(all_embeddings)
        else:
            embeddings = np.array([])
        
        # Log completion
        elapsed = time.time() - start_time
        rate = len(texts) / elapsed if elapsed > 0 else 0
        self.logger.info(
            f"Generated {len(texts)} embeddings in {elapsed:.2f}s "
            f"({rate:.1f} chunks/sec)"
        )
        
        return embeddings, all_metadata
    
    def process_texts(
        self,
        texts: List[str],
        show_progress: bool = True
    ) -> np.ndarray:
        """
        Process list of texts and return embeddings.
        
        Args:
            texts: List of text strings
            show_progress: Show progress bar
        
        Returns:
            Embeddings array
        """
        embeddings, _ = self._process_batches(
            texts,
            [None] * len(texts),  # No metadata
            show_progress=show_progress
        )
        return embeddings
    
    def process_with_checkpointing(
        self,
        chunks: List,
        checkpoint_dir: Optional[str] = None,
        checkpoint_interval: Optional[int] = None
    ) -> Tuple[np.ndarray, List[dict]]:
        """
        Process chunks with checkpointing support.
        
        Args:
            chunks: List of chunk objects
            checkpoint_dir: Directory to save checkpoints
            checkpoint_interval: Save checkpoint every N chunks
        
        Returns:
            Tuple of (embeddings array, metadata list)
        """
        # For now, use regular processing
        # Checkpointing can be implemented later if needed
        return self.process_chunks(chunks, show_progress=True)
    
    def get_optimal_batch_size(
        self,
        sample_texts: List[str],
        max_memory_mb: Optional[int] = None
    ) -> int:
        """
        Determine optimal batch size based on sample texts.
        
        Args:
            sample_texts: Sample texts to test with
            max_memory_mb: Maximum memory to use in MB
        
        Returns:
            Optimal batch size
        """
        # Simple heuristic: use configured batch size
        # More sophisticated version could test different batch sizes
        return self.batch_size
    
    def _get_cache_clear_interval(self) -> int:
        """Get optimal cache clearing interval based on hardware."""
        try:
            from ..utils.performance_optimizer import get_performance_optimizer
            optimizer = get_performance_optimizer()
            return optimizer.get_cache_clear_interval()
        except Exception:
            return 20  # Default fallback
    
    def _clear_cache_if_needed(self, memory_threshold_mb: Optional[float] = None):
        """
        Clear cache if memory usage exceeds threshold.
        
        Args:
            memory_threshold_mb: Memory threshold in MB (auto-detect if None)
        """
        try:
            # Get dynamic threshold if not provided
            if memory_threshold_mb is None:
                try:
                    from ..utils.performance_optimizer import get_performance_optimizer
                    optimizer = get_performance_optimizer()
                    memory_threshold_mb = optimizer.get_memory_threshold_mb()
                except Exception:
                    memory_threshold_mb = 80.0  # Default fallback
            
            process = psutil.Process(os.getpid())
            memory_mb = process.memory_info().rss / 1024 / 1024
            
            if memory_mb > memory_threshold_mb:
                self.logger.debug(f"Memory usage high ({memory_mb:.1f}MB > {memory_threshold_mb:.1f}MB), clearing cache")
                gc.collect()
                
                # Clear PyTorch cache if available
                try:
                    import torch
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                        torch.mps.empty_cache()
                except ImportError:
                    pass
        except Exception as e:
            self.logger.debug(f"Error checking memory: {e}")
    
    def _get_memory_usage(self) -> float:
        """
        Get current memory usage in MB.
        
        Returns:
            Memory usage in MB
        """
        try:
            process = psutil.Process(os.getpid())
            return process.memory_info().rss / 1024 / 1024
        except Exception:
            return 0.0
