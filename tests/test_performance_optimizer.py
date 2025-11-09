"""
Unit tests for performance optimizer module.
"""

import pytest
from src.utils.performance_optimizer import (
    PerformanceOptimizer,
    get_performance_optimizer
)
from src.utils.hardware_detector import HardwareDetector


def test_performance_optimizer_initialization():
    """Test performance optimizer initialization."""
    optimizer = PerformanceOptimizer()
    assert optimizer is not None
    assert optimizer.hardware_detector is not None


def test_get_optimal_batch_size():
    """Test optimal batch size calculation."""
    optimizer = PerformanceOptimizer()
    
    # Test with different devices
    for device in ["cpu", "cuda", "mps"]:
        batch_size = optimizer.get_optimal_batch_size(device=device)
        assert batch_size >= 8
        assert batch_size <= 2048  # Reasonable upper bound


def test_get_optimal_workers():
    """Test optimal workers calculation."""
    optimizer = PerformanceOptimizer()
    
    workers_cpu = optimizer.get_optimal_workers("cpu_bound")
    workers_io = optimizer.get_optimal_workers("io_bound")
    
    assert workers_cpu >= 1
    assert workers_io >= 1
    assert workers_io >= workers_cpu  # I/O bound can use more workers


def test_should_use_compilation():
    """Test compilation check."""
    optimizer = PerformanceOptimizer(enable_compilation=True)
    result = optimizer.should_use_compilation()
    assert isinstance(result, bool)


def test_get_memory_threshold_mb():
    """Test memory threshold calculation."""
    optimizer = PerformanceOptimizer()
    threshold = optimizer.get_memory_threshold_mb()
    assert threshold > 0


def test_get_cache_clear_interval():
    """Test cache clear interval calculation."""
    optimizer = PerformanceOptimizer()
    interval = optimizer.get_cache_clear_interval()
    assert interval >= 10
    assert interval <= 100


def test_get_optimization_preset():
    """Test optimization preset selection."""
    optimizer = PerformanceOptimizer()
    preset = optimizer.get_optimization_preset()
    assert preset in ["fast", "balanced", "memory_efficient"]


def test_get_recommendations():
    """Test getting optimization recommendations."""
    optimizer = PerformanceOptimizer()
    recommendations = optimizer.get_recommendations()
    
    assert "device" in recommendations
    assert "optimal_batch_size" in recommendations
    assert "optimal_workers_cpu" in recommendations
    assert "optimal_workers_io" in recommendations
    assert "use_compilation" in recommendations
    assert "memory_threshold_mb" in recommendations
    assert "cache_clear_interval" in recommendations
    assert "optimization_preset" in recommendations


def test_adjust_batch_size_for_memory():
    """Test batch size adjustment for memory."""
    optimizer = PerformanceOptimizer()
    
    # Low memory
    batch_size = optimizer._adjust_batch_size_for_memory(64, 128, 4.0, "cpu")
    assert batch_size == 64
    
    # High memory
    batch_size = optimizer._adjust_batch_size_for_memory(64, 128, 32.0, "cpu")
    assert batch_size >= 128


def test_singleton_pattern():
    """Test singleton pattern."""
    optimizer1 = get_performance_optimizer()
    optimizer2 = get_performance_optimizer()
    assert optimizer1 is optimizer2

