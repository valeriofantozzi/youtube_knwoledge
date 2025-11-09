#!/usr/bin/env python3
"""
Script to display hardware profile and optimization recommendations.

Usage:
    python scripts/show_hardware_profile.py
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils.hardware_detector import get_hardware_detector
from src.utils.performance_optimizer import get_performance_optimizer
from rich.console import Console
from rich.table import Table
from rich.panel import Panel

def main():
    """Display hardware profile and recommendations."""
    console = Console()
    
    console.print(Panel.fit(
        "[bold blue]Hardware Profile & Optimization Recommendations[/bold blue]",
        border_style="blue"
    ))
    
    # Get hardware detector
    hardware_detector = get_hardware_detector()
    profile = hardware_detector.detect_hardware()
    
    # Get performance optimizer
    optimizer = get_performance_optimizer()
    recommendations = optimizer.get_recommendations()
    
    # Display hardware profile
    console.print("\n[bold cyan]Hardware Profile[/bold cyan]")
    hardware_detector.print_profile()
    
    # Display recommendations
    console.print("\n[bold cyan]Optimization Recommendations[/bold cyan]")
    
    table = Table(show_header=True, header_style="bold magenta")
    table.add_column("Setting", style="cyan")
    table.add_column("Value", style="green")
    
    table.add_row("Recommended Device", recommendations["device"].upper())
    table.add_row("Optimal Batch Size", str(recommendations["optimal_batch_size"]))
    table.add_row("Optimal Workers (CPU-bound)", str(recommendations["optimal_workers_cpu"]))
    table.add_row("Optimal Workers (I/O-bound)", str(recommendations["optimal_workers_io"]))
    table.add_row("Use Model Compilation", "Yes" if recommendations["use_compilation"] else "No")
    table.add_row("Memory Threshold (MB)", f"{recommendations['memory_threshold_mb']:.1f}")
    table.add_row("Cache Clear Interval", f"Every {recommendations['cache_clear_interval']} batches")
    table.add_row("Optimization Preset", recommendations["optimization_preset"].replace("_", " ").title())
    
    console.print(table)
    
    # Performance notes
    console.print("\n[bold yellow]Performance Notes[/bold yellow]")
    device = recommendations["device"]
    if device == "mps":
        console.print("  • Apple Silicon GPU (MPS) detected - expect 5-10x speedup vs CPU")
    elif device == "cuda":
        console.print("  • CUDA GPU detected - expect significant speedup vs CPU")
    else:
        console.print("  • Running on CPU - consider GPU acceleration for better performance")
    
    if recommendations["use_compilation"]:
        console.print("  • Model compilation enabled - expect 10-30% additional speedup")
    
    console.print(f"  • Batch size optimized for your hardware: {recommendations['optimal_batch_size']}")
    console.print(f"  • Using {recommendations['optimization_preset']} optimization preset")

if __name__ == "__main__":
    main()

