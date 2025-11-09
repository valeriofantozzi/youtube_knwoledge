"""
Unit tests for hardware detector module.
"""

import pytest
from pathlib import Path
from src.utils.hardware_detector import HardwareDetector, get_hardware_detector


def test_hardware_detector_initialization():
    """Test hardware detector initialization."""
    detector = HardwareDetector()
    assert detector is not None
    assert detector.logger is not None


def test_detect_hardware():
    """Test hardware detection."""
    detector = HardwareDetector()
    profile = detector.detect_hardware(use_cache=False)
    
    # Check required keys
    assert "cpu" in profile
    assert "memory" in profile
    assert "gpu" in profile
    assert "system" in profile
    
    # Check CPU info
    cpu = profile["cpu"]
    assert "cores_physical" in cpu
    assert "cores_logical" in cpu
    assert cpu["cores_logical"] >= cpu["cores_physical"]
    
    # Check memory info
    memory = profile["memory"]
    assert "total_gb" in memory
    assert "available_gb" in memory
    assert memory["total_gb"] > 0
    
    # Check GPU info
    gpu = profile["gpu"]
    assert "cuda_available" in gpu
    assert "mps_available" in gpu
    assert "device" in gpu
    assert gpu["device"] in ["cpu", "cuda", "mps"]


def test_get_recommended_device():
    """Test device recommendation."""
    detector = HardwareDetector()
    device = detector.get_recommended_device()
    assert device in ["cpu", "cuda", "mps"]


def test_get_cpu_cores():
    """Test CPU cores detection."""
    detector = HardwareDetector()
    cores = detector.get_cpu_cores()
    assert cores >= 1


def test_get_available_memory_gb():
    """Test available memory detection."""
    detector = HardwareDetector()
    memory = detector.get_available_memory_gb()
    assert memory > 0


def test_singleton_pattern():
    """Test singleton pattern."""
    detector1 = get_hardware_detector()
    detector2 = get_hardware_detector()
    assert detector1 is detector2


def test_profile_caching(tmp_path):
    """Test profile caching."""
    detector = HardwareDetector(cache_dir=tmp_path)
    
    # First detection
    profile1 = detector.detect_hardware(use_cache=False)
    
    # Second detection with cache
    profile2 = detector.detect_hardware(use_cache=True)
    
    # Should be the same
    assert profile1["cpu"]["cores_logical"] == profile2["cpu"]["cores_logical"]


def test_clear_cache(tmp_path):
    """Test cache clearing."""
    detector = HardwareDetector(cache_dir=tmp_path)
    detector.detect_hardware(use_cache=False)
    assert detector.cache_file.exists()
    
    detector.clear_cache()
    assert not detector.cache_file.exists()

