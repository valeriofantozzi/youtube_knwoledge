"""
Performance Optimizer Module

Optimizes performance based on hardware capabilities and system resources.
"""

import os
import time
import torch
from typing import Dict, Optional, Tuple, Literal
import numpy as np
from .hardware_detector import get_hardware_detector, HardwareDetector
from .logger import get_default_logger


class PerformanceOptimizer:
    """Optimizes performance based on hardware capabilities."""
    
    # Batch size recommendations by device type
    BATCH_SIZE_RANGES = {
        "cuda": (256, 512),
        "mps": (512, 1024),
        "cpu": (64, 256),
    }
    
    # Memory thresholds for batch size adjustment (GB)
    MEMORY_THRESHOLDS = {
        "low": 8.0,      # < 8GB: conservative batch sizes
        "medium": 16.0,  # 8-16GB: moderate batch sizes
        "high": 32.0,    # 16-32GB: large batch sizes
        "very_high": 32.0,  # > 32GB: very large batch sizes
    }
    
    def __init__(
        self,
        hardware_detector: Optional[HardwareDetector] = None,
        enable_compilation: bool = True
    ):
        """
        Initialize performance optimizer.
        
        Args:
            hardware_detector: HardwareDetector instance (creates new if None)
            enable_compilation: Enable torch.compile() optimization if available
        """
        self.hardware_detector = hardware_detector or get_hardware_detector()
        self.enable_compilation = enable_compilation
        self.logger = get_default_logger()
        self._optimal_batch_size: Optional[int] = None
        self._benchmark_results: Optional[Dict] = None
        self._threads_configured: bool = False
    
    def configure_pytorch_threads(
        self,
        cpu_percentage: Optional[float] = None,
        force_reconfigure: bool = False
    ) -> int:
        """
        Configure PyTorch to use optimal number of CPU threads.
        
        This is essential for maximizing CPU utilization during embedding generation.
        PyTorch defaults to a conservative number of threads, which can result in
        low CPU usage even when more cores are available.
        
        Args:
            cpu_percentage: Percentage of CPU cores to use (0.0-1.0). 
                           If None, reads from MAX_WORKERS_PERCENTAGE env var or defaults to 0.75.
            force_reconfigure: Force reconfiguration even if already configured.
        
        Returns:
            Number of threads configured.
        """
        if self._threads_configured and not force_reconfigure:
            current_threads = torch.get_num_threads()
            self.logger.debug(f"PyTorch threads already configured: {current_threads}")
            return current_threads
        
        # Get CPU percentage from parameter, env var, or default
        if cpu_percentage is None:
            try:
                cpu_percentage = float(os.getenv("MAX_WORKERS_PERCENTAGE", "0.75"))
            except ValueError:
                cpu_percentage = 0.75
        
        # Validate percentage
        cpu_percentage = max(0.1, min(1.0, cpu_percentage))
        
        # Get hardware profile
        profile = self.hardware_detector.get_profile()
        cpu_cores = profile.get("cpu", {}).get("cores_logical", os.cpu_count() or 1)
        
        # Calculate optimal number of threads
        optimal_threads = max(1, int(cpu_cores * cpu_percentage))
        
        # Configure PyTorch threads
        # num_threads: for intra-op parallelism (within single operation)
        torch.set_num_threads(optimal_threads)
        
        # num_interop_threads: for inter-op parallelism (between operations)
        # Use a smaller number to avoid thread contention
        # Note: set_num_interop_threads can only be called once before parallel work starts
        interop_threads = max(1, optimal_threads // 2)
        try:
            torch.set_num_interop_threads(interop_threads)
        except RuntimeError:
            # Already set or parallel work has started, skip
            self.logger.debug(
                "Inter-op threads already configured or parallel work started, skipping"
            )
        
        # Set environment variables for other libraries that use them
        # (tokenizers, OpenMP, etc.)
        os.environ["OMP_NUM_THREADS"] = str(optimal_threads)
        os.environ["MKL_NUM_THREADS"] = str(optimal_threads)
        os.environ["TOKENIZERS_PARALLELISM"] = "true"
        
        self._threads_configured = True
        
        self.logger.info(
            f"Configured PyTorch threads: {optimal_threads} intra-op, {interop_threads} inter-op "
            f"({cpu_percentage*100:.0f}% of {cpu_cores} CPU cores)"
        )
        
        return optimal_threads
    
    def get_optimal_batch_size(
        self,
        device: Optional[str] = None,
        model_size_mb: Optional[float] = None,
        text_length_avg: Optional[int] = None,
        force_recompute: bool = False
    ) -> int:
        """
        Get optimal batch size based on hardware and model characteristics.
        
        Args:
            device: Device type ('cuda', 'mps', 'cpu') - auto-detect if None
            model_size_mb: Model size in MB (for memory estimation)
            text_length_avg: Average text length in tokens (for memory estimation)
            force_recompute: Force recomputation even if cached
        
        Returns:
            Optimal batch size
        """
        if self._optimal_batch_size is not None and not force_recompute:
            return self._optimal_batch_size
        
        profile = self.hardware_detector.get_profile()
        
        # Determine device
        if device is None:
            device = profile.get("gpu", {}).get("device", "cpu")
        
        # Get base batch size range for device
        batch_min, batch_max = self.BATCH_SIZE_RANGES.get(device, (64, 128))
        
        # Adjust based on available memory
        available_memory_gb = profile.get("memory", {}).get("available_gb", 8.0)
        batch_size = self._adjust_batch_size_for_memory(
            batch_min, batch_max, available_memory_gb, device
        )
        
        # Adjust based on model size if provided
        if model_size_mb:
            batch_size = self._adjust_batch_size_for_model(
                batch_size, model_size_mb, available_memory_gb
            )
        
        # Adjust based on text length if provided
        if text_length_avg:
            batch_size = self._adjust_batch_size_for_text_length(
                batch_size, text_length_avg, device
            )
        
        # Ensure reasonable bounds
        batch_size = max(8, min(batch_size, batch_max * 2))
        
        self._optimal_batch_size = batch_size
        
        self.logger.info(
            f"Optimal batch size: {batch_size} "
            f"(device: {device}, memory: {available_memory_gb:.1f}GB)"
        )
        
        return batch_size
    
    def _adjust_batch_size_for_memory(
        self,
        batch_min: int,
        batch_max: int,
        available_memory_gb: float,
        device: str
    ) -> int:
        """Adjust batch size based on available memory."""
        if available_memory_gb < self.MEMORY_THRESHOLDS["low"]:
            # Low memory: use lower end of range
            return batch_min
        elif available_memory_gb < self.MEMORY_THRESHOLDS["medium"]:
            # Medium memory: use middle of range
            return (batch_min + batch_max) // 2
        elif available_memory_gb < self.MEMORY_THRESHOLDS["high"]:
            # High memory: use upper end of range
            return batch_max
        else:
            # Very high memory: can exceed normal range
            multiplier = 1.5 if device in ["cuda", "mps"] else 1.2
            return int(batch_max * multiplier)
    
    def _adjust_batch_size_for_model(
        self,
        current_batch_size: int,
        model_size_mb: float,
        available_memory_gb: float
    ) -> int:
        """Adjust batch size based on model size."""
        # Estimate: model + batch data should use < 50% of available memory
        available_memory_mb = available_memory_gb * 1024
        usable_memory_mb = available_memory_mb * 0.5
        
        # Rough estimate: each batch item needs ~2x model size in memory
        # (model weights + activations + gradients)
        memory_per_batch_item = model_size_mb * 2.5
        
        max_batch_by_memory = int(usable_memory_mb / memory_per_batch_item)
        
        return min(current_batch_size, max_batch_by_memory)
    
    def _adjust_batch_size_for_text_length(
        self,
        current_batch_size: int,
        text_length_avg: int,
        device: str
    ) -> int:
        """Adjust batch size based on average text length."""
        # Longer texts need more memory per batch item
        # Rough heuristic: reduce batch size for very long texts
        if text_length_avg > 512:
            # Very long texts: reduce batch size
            reduction_factor = 512 / text_length_avg
            return max(8, int(current_batch_size * reduction_factor))
        elif text_length_avg < 100:
            # Short texts: can increase batch size slightly
            increase_factor = 1.2 if device in ["cuda", "mps"] else 1.1
            return int(current_batch_size * increase_factor)
        
        return current_batch_size
    
    def get_optimal_workers(self, task_type: Literal["cpu_bound", "io_bound"] = "cpu_bound") -> int:
        """
        Get optimal number of workers for parallel processing.
        
        Args:
            task_type: Type of task ('cpu_bound' or 'io_bound')
        
        Returns:
            Optimal number of workers
        """
        profile = self.hardware_detector.get_profile()
        cpu_cores = profile.get("cpu", {}).get("cores_logical", 1)
        available_memory_gb = profile.get("memory", {}).get("available_gb", 8.0)
        
        if task_type == "cpu_bound":
            # For CPU-bound tasks, use number of cores
            # But leave some cores free for system
            workers = max(1, cpu_cores - 1)
        else:
            # For I/O-bound tasks, can use more workers
            # But limit by available memory (rough estimate: 1GB per worker)
            workers = min(cpu_cores * 2, int(available_memory_gb))
        
        # Ensure reasonable bounds
        workers = max(1, min(workers, 16))  # Cap at 16 workers
        
        return workers
    
    def should_use_compilation(self) -> bool:
        """
        Check if model compilation should be used.
        
        Returns:
            True if compilation should be used
        """
        if not self.enable_compilation:
            return False
        
        # Check PyTorch version (torch.compile requires PyTorch 2.0+)
        try:
            torch_version = torch.__version__.split('.')
            major_version = int(torch_version[0])
            if major_version >= 2:
                return True
        except (ValueError, IndexError):
            pass
        
        return False
    
    def compile_model(self, model) -> any:
        """
        Compile model for better performance (if supported).
        
        Args:
            model: PyTorch model to compile
        
        Returns:
            Compiled model (or original if compilation fails)
        """
        if not self.should_use_compilation():
            return model
        
        try:
            self.logger.info("Compiling model with torch.compile() for optimization...")
            compiled_model = torch.compile(model, mode="reduce-overhead")
            self.logger.info("Model compilation successful")
            return compiled_model
        except Exception as e:
            self.logger.warning(f"Model compilation failed: {e}. Using uncompiled model.")
            return model
    
    def benchmark_batch_sizes(
        self,
        model,
        sample_texts: list,
        device: str,
        batch_sizes: Optional[list] = None,
        num_runs: int = 3
    ) -> Dict:
        """
        Benchmark different batch sizes to find optimal.
        
        Args:
            model: Model to benchmark
            sample_texts: Sample texts for benchmarking
            device: Device to use
            batch_sizes: List of batch sizes to test (auto-generate if None)
            num_runs: Number of runs per batch size
        
        Returns:
            Dictionary with benchmark results
        """
        if batch_sizes is None:
            profile = self.hardware_detector.get_profile()
            device_type = profile.get("gpu", {}).get("device", "cpu")
            batch_min, batch_max = self.BATCH_SIZE_RANGES.get(device_type, (64, 128))
            # Test a range of batch sizes
            batch_sizes = [
                batch_min,
                (batch_min + batch_max) // 2,
                batch_max,
                int(batch_max * 1.5),
            ]
        
        results = {}
        
        self.logger.info(f"Benchmarking batch sizes: {batch_sizes}")
        
        for batch_size in batch_sizes:
            times = []
            for run in range(num_runs):
                try:
                    start_time = time.time()
                    # Process in batches
                    for i in range(0, len(sample_texts), batch_size):
                        batch = sample_texts[i:i+batch_size]
                        _ = model.encode(batch, batch_size=batch_size, device=device)
                    elapsed = time.time() - start_time
                    times.append(elapsed)
                except Exception as e:
                    self.logger.warning(f"Batch size {batch_size} failed: {e}")
                    times.append(float('inf'))
            
            if times and not all(t == float('inf') for t in times):
                valid_times = [t for t in times if t != float('inf')]
                avg_time = np.mean(valid_times)
                throughput = len(sample_texts) / avg_time if avg_time > 0 else 0
                results[batch_size] = {
                    "avg_time": avg_time,
                    "throughput": throughput,
                    "success": True,
                }
            else:
                results[batch_size] = {
                    "avg_time": float('inf'),
                    "throughput": 0,
                    "success": False,
                }
        
        # Find optimal batch size (highest throughput)
        if results:
            valid_results = {
                k: v for k, v in results.items()
                if v.get("success", False)
            }
            if valid_results:
                optimal_batch = max(
                    valid_results.keys(),
                    key=lambda k: valid_results[k]["throughput"]
                )
                results["optimal"] = optimal_batch
                self._optimal_batch_size = optimal_batch
        
        self._benchmark_results = results
        return results
    
    def get_memory_threshold_mb(self) -> float:
        """
        Get memory threshold for cache clearing (in MB).
        
        Returns:
            Memory threshold in MB
        """
        profile = self.hardware_detector.get_profile()
        available_memory_gb = profile.get("memory", {}).get("available_gb", 8.0)
        
        # Use 80% of available memory as threshold
        threshold_gb = available_memory_gb * 0.8
        return threshold_gb * 1024  # Convert to MB
    
    def get_cache_clear_interval(self) -> int:
        """
        Get optimal cache clearing interval (number of batches).
        
        Returns:
            Number of batches between cache clears
        """
        profile = self.hardware_detector.get_profile()
        available_memory_gb = profile.get("memory", {}).get("available_gb", 8.0)
        
        # More memory = less frequent clearing
        if available_memory_gb >= 32:
            return 100  # Very high memory: clear every 100 batches
        elif available_memory_gb >= 16:
            return 50   # High memory: clear every 50 batches
        elif available_memory_gb >= 8:
            return 20   # Medium memory: clear every 20 batches
        else:
            return 10   # Low memory: clear every 10 batches
    
    def get_optimization_preset(self) -> Literal["fast", "balanced", "memory_efficient"]:
        """
        Get optimization preset based on hardware.
        
        Returns:
            Optimization preset name
        """
        profile = self.hardware_detector.get_profile()
        available_memory_gb = profile.get("memory", {}).get("available_gb", 8.0)
        device = profile.get("gpu", {}).get("device", "cpu")
        
        if device != "cpu" and available_memory_gb >= 16:
            return "fast"
        elif available_memory_gb >= 8:
            return "balanced"
        else:
            return "memory_efficient"
    
    def get_recommendations(self) -> Dict:
        """
        Get performance optimization recommendations.
        
        Returns:
            Dictionary with recommendations
        """
        profile = self.hardware_detector.get_profile()
        device = profile.get("gpu", {}).get("device", "cpu")
        
        # Get CPU percentage from env or default
        try:
            cpu_percentage = float(os.getenv("MAX_WORKERS_PERCENTAGE", "0.75"))
        except ValueError:
            cpu_percentage = 0.75
        
        cpu_cores = profile.get("cpu", {}).get("cores_logical", os.cpu_count() or 1)
        optimal_threads = max(1, int(cpu_cores * cpu_percentage))
        
        recommendations = {
            "device": device,
            "optimal_batch_size": self.get_optimal_batch_size(device=device),
            "optimal_workers_cpu": self.get_optimal_workers("cpu_bound"),
            "optimal_workers_io": self.get_optimal_workers("io_bound"),
            "use_compilation": self.should_use_compilation(),
            "memory_threshold_mb": self.get_memory_threshold_mb(),
            "cache_clear_interval": self.get_cache_clear_interval(),
            "optimization_preset": self.get_optimization_preset(),
            "pytorch_threads": optimal_threads,
            "cpu_cores": cpu_cores,
            "cpu_percentage": cpu_percentage,
        }
        
        return recommendations


# Global performance optimizer instance
_performance_optimizer: Optional[PerformanceOptimizer] = None


def get_performance_optimizer(
    hardware_detector: Optional[HardwareDetector] = None,
    enable_compilation: bool = True
) -> PerformanceOptimizer:
    """
    Get global performance optimizer instance (singleton pattern).
    
    Args:
        hardware_detector: HardwareDetector instance (only used on first call)
        enable_compilation: Enable compilation (only used on first call)
    
    Returns:
        PerformanceOptimizer instance
    """
    global _performance_optimizer
    if _performance_optimizer is None:
        _performance_optimizer = PerformanceOptimizer(
            hardware_detector=hardware_detector,
            enable_compilation=enable_compilation
        )
    return _performance_optimizer

