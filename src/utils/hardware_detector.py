"""
Hardware Detection Module

Detects and profiles system hardware resources for performance optimization.
"""

import platform
import psutil
import torch
from typing import Dict, Optional, Literal
from pathlib import Path
import json
import time
from ..utils.logger import get_default_logger


class HardwareDetector:
    """Detects and profiles system hardware resources."""
    
    def __init__(self, cache_dir: Optional[Path] = None):
        """
        Initialize hardware detector.
        
        Args:
            cache_dir: Directory to cache hardware profile (default: project root)
        """
        # Use basic logging to avoid circular dependency
        import logging
        self.logger = logging.getLogger(__name__)
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            handler.setFormatter(logging.Formatter('%(levelname)s: %(message)s'))
            self.logger.addHandler(handler)
            self.logger.setLevel(logging.INFO)
        self.cache_dir = cache_dir or Path(__file__).parent.parent.parent / "data" / "cache"
        self.cache_file = self.cache_dir / "hardware_profile.json"
        self._profile: Optional[Dict] = None
    
    def detect_hardware(self, use_cache: bool = True) -> Dict:
        """
        Detect system hardware and return profile.
        
        Args:
            use_cache: If True, use cached profile if available
        
        Returns:
            Dictionary with hardware profile
        """
        # Try to load from cache
        if use_cache and self.cache_file.exists():
            try:
                with open(self.cache_file, 'r') as f:
                    cached_profile = json.load(f)
                    # Verify it's still valid (basic check)
                    if self._validate_profile(cached_profile):
                        self.logger.debug("Using cached hardware profile")
                        self._profile = cached_profile
                        return cached_profile
            except Exception as e:
                self.logger.warning(f"Failed to load cached hardware profile: {e}")
        
        # Detect hardware
        profile = {
            "cpu": self._detect_cpu(),
            "memory": self._detect_memory(),
            "gpu": self._detect_gpu(),
            "system": self._detect_system(),
            "detection_timestamp": time.time(),
            "hostname": platform.node()
        }
        
        # Cache the profile
        self._save_profile(profile)
        self._profile = profile
        
        return profile
    
    def _detect_cpu(self) -> Dict:
        """Detect CPU information."""
        cpu_info = {
            "cores_physical": psutil.cpu_count(logical=False) or 1,
            "cores_logical": psutil.cpu_count(logical=True) or 1,
            "frequency_mhz": psutil.cpu_freq().current if psutil.cpu_freq() else None,
            "architecture": platform.machine(),
            "processor": platform.processor(),
        }
        
        # Detect Apple Silicon chip type
        if platform.system() == "Darwin" and platform.machine() == "arm64":
            cpu_info["apple_silicon"] = self._detect_apple_silicon()
        else:
            cpu_info["apple_silicon"] = None
        
        return cpu_info
    
    def _detect_apple_silicon(self) -> Optional[Dict]:
        """Detect Apple Silicon chip type and capabilities."""
        try:
            import subprocess
            
            # Try to get chip info from sysctl
            result = subprocess.run(
                ["sysctl", "-n", "machdep.cpu.brand_string"],
                capture_output=True,
                text=True,
                timeout=2
            )
            
            if result.returncode == 0:
                brand = result.stdout.strip()
                chip_info = {
                    "brand": brand,
                    "type": self._parse_apple_chip_type(brand),
                    "is_pro": "Pro" in brand or "Max" in brand or "Ultra" in brand,
                }
                return chip_info
        except Exception as e:
            self.logger.debug(f"Could not detect Apple Silicon details: {e}")
        
        return {"type": "unknown", "is_pro": False}
    
    def _parse_apple_chip_type(self, brand: str) -> str:
        """Parse Apple Silicon chip type from brand string."""
        brand_lower = brand.lower()
        if "m3" in brand_lower:
            return "M3"
        elif "m2" in brand_lower:
            return "M2"
        elif "m1" in brand_lower:
            return "M1"
        return "unknown"
    
    def _detect_memory(self) -> Dict:
        """Detect memory information."""
        mem = psutil.virtual_memory()
        return {
            "total_gb": round(mem.total / (1024**3), 2),
            "available_gb": round(mem.available / (1024**3), 2),
            "used_gb": round(mem.used / (1024**3), 2),
            "percent_used": mem.percent,
        }
    
    def _detect_gpu(self) -> Dict:
        """Detect GPU availability and type."""
        gpu_info = {
            "cuda_available": False,
            "cuda_device_count": 0,
            "mps_available": False,
            "device": None,
        }
        
        # Check CUDA
        if torch.cuda.is_available():
            gpu_info["cuda_available"] = True
            gpu_info["cuda_device_count"] = torch.cuda.device_count()
            if torch.cuda.device_count() > 0:
                gpu_info["cuda_device_name"] = torch.cuda.get_device_name(0)
                gpu_info["cuda_memory_gb"] = round(
                    torch.cuda.get_device_properties(0).total_memory / (1024**3), 2
                )
        
        # Check MPS (Apple Silicon)
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            gpu_info["mps_available"] = True
        
        # Determine best device
        if gpu_info["cuda_available"]:
            gpu_info["device"] = "cuda"
        elif gpu_info["mps_available"]:
            gpu_info["device"] = "mps"
        else:
            gpu_info["device"] = "cpu"
        
        return gpu_info
    
    def _detect_system(self) -> Dict:
        """Detect system information."""
        return {
            "platform": platform.system(),
            "platform_release": platform.release(),
            "platform_version": platform.version(),
            "architecture": platform.machine(),
            "python_version": platform.python_version(),
            "pytorch_version": torch.__version__,
        }
    
    def get_profile(self) -> Dict:
        """
        Get hardware profile (detects if not already done).
        
        Returns:
            Hardware profile dictionary
        """
        if self._profile is None:
            return self.detect_hardware()
        return self._profile
    
    def get_recommended_device(self) -> str:
        """
        Get recommended device based on hardware.
        
        Returns:
            Recommended device: 'cuda', 'mps', or 'cpu'
        """
        profile = self.get_profile()
        gpu = profile.get("gpu", {})
        
        if gpu.get("cuda_available"):
            return "cuda"
        elif gpu.get("mps_available"):
            return "mps"
        else:
            return "cpu"
    
    def get_cpu_cores(self) -> int:
        """Get number of CPU cores (logical)."""
        profile = self.get_profile()
        return profile.get("cpu", {}).get("cores_logical", 1)
    
    def get_available_memory_gb(self) -> float:
        """Get available memory in GB."""
        profile = self.get_profile()
        return profile.get("memory", {}).get("available_gb", 8.0)
    
    def is_apple_silicon_pro(self) -> bool:
        """Check if running on Apple Silicon Pro/Max/Ultra."""
        profile = self.get_profile()
        apple_silicon = profile.get("cpu", {}).get("apple_silicon")
        if apple_silicon:
            return apple_silicon.get("is_pro", False)
        return False
    
    def _validate_profile(self, profile: Dict) -> bool:
        """Validate hardware profile structure."""
        required_keys = ["cpu", "memory", "gpu", "system"]
        return all(key in profile for key in required_keys)
    
    def _save_profile(self, profile: Dict) -> None:
        """Save hardware profile to cache."""
        try:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            with open(self.cache_file, 'w') as f:
                json.dump(profile, f, indent=2)
            self.logger.debug(f"Saved hardware profile to {self.cache_file}")
        except Exception as e:
            self.logger.warning(f"Failed to save hardware profile: {e}")
    
    def clear_cache(self) -> None:
        """Clear cached hardware profile."""
        if self.cache_file.exists():
            self.cache_file.unlink()
            self.logger.info("Cleared hardware profile cache")
        self._profile = None
    
    def print_profile(self) -> None:
        """Print hardware profile in readable format."""
        profile = self.get_profile()
        
        self.logger.info("=" * 60)
        self.logger.info("Hardware Profile")
        self.logger.info("=" * 60)
        
        # CPU
        cpu = profile.get("cpu", {})
        self.logger.info(f"CPU:")
        self.logger.info(f"  Physical cores: {cpu.get('cores_physical', 'N/A')}")
        self.logger.info(f"  Logical cores: {cpu.get('cores_logical', 'N/A')}")
        if cpu.get("apple_silicon"):
            apple = cpu["apple_silicon"]
            self.logger.info(f"  Apple Silicon: {apple.get('type', 'unknown')}")
            if apple.get("is_pro"):
                self.logger.info(f"  Pro/Max/Ultra: Yes")
        
        # Memory
        mem = profile.get("memory", {})
        self.logger.info(f"Memory:")
        self.logger.info(f"  Total: {mem.get('total_gb', 'N/A')} GB")
        self.logger.info(f"  Available: {mem.get('available_gb', 'N/A')} GB")
        
        # GPU
        gpu = profile.get("gpu", {})
        self.logger.info(f"GPU:")
        self.logger.info(f"  CUDA available: {gpu.get('cuda_available', False)}")
        self.logger.info(f"  MPS available: {gpu.get('mps_available', False)}")
        self.logger.info(f"  Recommended device: {gpu.get('device', 'cpu')}")
        if gpu.get("cuda_device_name"):
            self.logger.info(f"  CUDA device: {gpu.get('cuda_device_name')}")
        
        # System
        sys_info = profile.get("system", {})
        self.logger.info(f"System:")
        self.logger.info(f"  Platform: {sys_info.get('platform', 'N/A')}")
        self.logger.info(f"  Architecture: {sys_info.get('architecture', 'N/A')}")
        self.logger.info(f"  PyTorch version: {sys_info.get('pytorch_version', 'N/A')}")
        
        self.logger.info("=" * 60)


# Global hardware detector instance
_hardware_detector: Optional[HardwareDetector] = None


def get_hardware_detector(cache_dir: Optional[Path] = None) -> HardwareDetector:
    """
    Get global hardware detector instance (singleton pattern).
    
    Args:
        cache_dir: Cache directory (only used on first call)
    
    Returns:
        HardwareDetector instance
    """
    global _hardware_detector
    if _hardware_detector is None:
        _hardware_detector = HardwareDetector(cache_dir=cache_dir)
    return _hardware_detector

