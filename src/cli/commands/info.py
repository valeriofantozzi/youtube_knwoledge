"""
CLI command: knowbase info

Displays system information, database statistics, and hardware profile.

Usage:
    knowbase info
    knowbase info --verbose
"""

import sys
from pathlib import Path
from typing import Optional, Dict, Any

import click
import psutil

from src.utils.config_manager import ConfigManager
from src.vector_store.chroma_manager import ChromaDBManager
from src.cli.utils.output import (
    console,
    print_error,
    print_panel,
    print_table,
    print_dict,
)


@click.command()
@click.option(
    "-v",
    "--verbose",
    is_flag=True,
    help="Show detailed information",
)
@click.option(
    "-c",
    "--config",
    type=click.Path(exists=True),
    help="Configuration file",
)
@click.pass_context
def info(
    ctx: click.Context,
    verbose: bool,
    config: Optional[str],
) -> None:
    """
    Show KnowBase system information and statistics.

    Displays:
    - CLI version
    - Database location and statistics
    - Available embedding models
    - Hardware information (CPU, memory, GPU)
    - Configuration settings

    Examples:
        \b
        knowbase info
        knowbase info --verbose
    """
    try:
        # Get configuration
        try:
            if config:
                config_manager = ConfigManager(config_file=Path(config))
                complete_config = config_manager.config
            else:
                # Use default config - just create from file
                config_manager = ConfigManager()
                complete_config = config_manager.config

            vs_cfg = complete_config.vector_store
            emb_cfg = complete_config.embedding
        except Exception as e:
            console.print(f"[yellow]Warning: Could not load configuration: {e}[/yellow]")
            vs_cfg = None
            emb_cfg = None

        # Get database info
        console.print("\n[bold cyan]📊 KnowBase System Information[/bold cyan]\n")

        # CLI Info
        from src.cli.main import __version__

        cli_info = {
            "CLI Version": __version__,
            "Status": "Ready",
        }

        print_panel(
            "\n".join(f"[cyan]{k}:[/cyan] {v}" for k, v in cli_info.items()),
            title="CLI Information",
            border_style="blue",
        )
        console.print()

        # Database Info
        try:
            if vs_cfg is None:
                # Use default paths
                db_path = "./data/vector_db"
                collection_name = "documents"
            else:
                db_path = vs_cfg.db_path
                collection_name = vs_cfg.collection_name

            chroma_manager = ChromaDBManager(
                db_path=db_path,
                collection_name=collection_name,
            )
            collection = chroma_manager.get_or_create_collection()

            db_size = collection.count()
            collection_path = Path(db_path)

            # Get directory size
            if collection_path.exists():
                dir_size_bytes = sum(
                    f.stat().st_size for f in collection_path.rglob("*") if f.is_file()
                )
                dir_size_mb = dir_size_bytes / (1024 * 1024)
            else:
                dir_size_mb = 0

            db_info = {
                "Location": str(collection_path),
                "Collection": collection_name,
                "Documents": str(db_size),
                "Size": f"{dir_size_mb:.1f} MB",
            }

            print_panel(
                "\n".join(f"[cyan]{k}:[/cyan] {v}" for k, v in db_info.items()),
                title="Vector Database",
                border_style="blue",
            )
            console.print()

        except Exception as e:
            console.print(f"[yellow]⚠ Could not access vector database: {e}[/yellow]\n")

        # Embedding Models Info
        try:
            if emb_cfg:
                model_info = {
                    "Model Name": emb_cfg.model_name,
                    "Device": emb_cfg.device,
                    "Batch Size": str(emb_cfg.batch_size),
                    "Precision": emb_cfg.precision,
                }
            else:
                model_info = {
                    "Model Name": "BAAI/bge-large-en-v1.5",
                    "Device": "auto",
                    "Batch Size": "32",
                    "Precision": "fp32",
                }

            console.print("[bold cyan]🤖 Embedding Configuration[/bold cyan]")
            for k, v in model_info.items():
                console.print(f"  [cyan]{k}:[/cyan] {v}")
            console.print()

        except Exception as e:
            console.print(f"[yellow]⚠ Could not display model info: {e}[/yellow]\n")

        # Hardware Info
        try:
            # CPU info
            cpu_count = psutil.cpu_count(logical=False) or 1
            cpu_count_logical = psutil.cpu_count(logical=True) or 1
            cpu_percent = psutil.cpu_percent(interval=0.5)

            # Memory info
            memory = psutil.virtual_memory()
            memory_gb = memory.total / (1024**3)
            memory_used_gb = memory.used / (1024**3)
            memory_percent = memory.percent

            # GPU info (if available)
            gpu_info = "Not detected"
            try:
                import torch

                if torch.cuda.is_available():
                    gpu_name = torch.cuda.get_device_name(0)
                    gpu_memory = torch.cuda.get_device_properties(0).total_memory / (
                        1024**3
                    )
                    gpu_info = f"{gpu_name} ({gpu_memory:.1f} GB)"
                elif torch.backends.mps.is_available():
                    gpu_info = "Metal Performance Shaders (Apple Silicon)"
            except Exception:
                pass

            hw_info = {
                "CPU Cores": f"{cpu_count} physical, {cpu_count_logical} logical",
                "CPU Usage": f"{cpu_percent}%",
                "Memory": f"{memory_used_gb:.1f} / {memory_gb:.1f} GB ({memory_percent}%)",
                "GPU": gpu_info,
                "Device Config": emb_cfg.device,
            }

            print_panel(
                "\n".join(f"[cyan]{k}:[/cyan] {v}" for k, v in hw_info.items()),
                title="Hardware",
                border_style="blue",
            )
            console.print()

        except Exception as e:
            console.print(f"[yellow]⚠ Could not retrieve hardware info: {e}[/yellow]\n")

        # Summary
        console.print(
            "[bold green]✓[/bold green] System information displayed successfully\n"
        )

    except Exception as e:
        print_error(f"Failed to retrieve system information: {e}")
