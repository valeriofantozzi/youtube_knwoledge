#!/usr/bin/env python3
"""
Main script for processing subtitle files.

Processes all SRT files, generates embeddings, and indexes them in the vector store.
"""

import argparse
import sys
from pathlib import Path
from typing import List, Optional
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TimeElapsedColumn, TaskID
from rich.table import Table
from rich.panel import Panel

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.preprocessing.pipeline import PreprocessingPipeline, ProcessedVideo
from src.embeddings.pipeline import EmbeddingPipeline
from src.vector_store.pipeline import VectorStorePipeline
from src.utils.logger import get_default_logger
from src.utils.config import get_config


def find_srt_files(input_dir: Path) -> List[Path]:
    """Find all SRT files in the input directory."""
    srt_files = list(input_dir.rglob("*.srt"))
    return sorted(srt_files)


def process_pipeline(
    input_dir: Path,
    skip_processed: bool = True,
    parallel_preprocessing: bool = True,
    max_workers: Optional[int] = None,
    batch_size: Optional[int] = None,
    show_progress: bool = True
) -> dict:
    """
    Run the complete pipeline: preprocessing -> embeddings -> indexing.
    
    Args:
        input_dir: Directory containing SRT files
        skip_processed: Skip videos that are already indexed
        parallel_preprocessing: Use parallel processing for preprocessing
        max_workers: Max workers for parallel processing
        batch_size: Batch size for embedding generation
        show_progress: Show progress bars
    
    Returns:
        Dictionary with processing statistics
    """
    console = Console()
    logger = get_default_logger()
    config = get_config()
    
    # Initialize pipelines
    console.print("[bold blue]Initializing pipelines...[/bold blue]")
    preprocessing_pipeline = PreprocessingPipeline()
    embedding_pipeline = EmbeddingPipeline(
        batch_size=batch_size,
        enable_optimizations=True  # Enable hardware-aware optimizations (Task 4.5)
    )
    vector_store_pipeline = VectorStorePipeline()
    
    # Find SRT files
    console.print(f"[bold blue]Scanning for SRT files in {input_dir}...[/bold blue]")
    srt_files = find_srt_files(input_dir)
    
    if not srt_files:
        console.print("[bold red]No SRT files found![/bold red]")
        return {
            "total_files": 0,
            "processed": 0,
            "skipped": 0,
            "failed": 0,
            "total_chunks": 0,
            "total_indexed": 0
        }
    
    console.print(f"[green]Found {len(srt_files)} SRT files[/green]")
    
    # Check for already processed videos if skip_processed is True
    processed_video_ids = set()
    if skip_processed:
        try:
            stats = vector_store_pipeline.get_index_statistics()
            # Get all video IDs from the index
            collection = vector_store_pipeline.chroma_manager.get_or_create_collection()
            # Sample some documents to check video IDs
            sample = collection.get(limit=1000)
            if sample.get("metadatas"):
                for metadata in sample["metadatas"]:
                    if metadata and "video_id" in metadata:
                        processed_video_ids.add(metadata["video_id"])
            console.print(f"[yellow]Found {len(processed_video_ids)} already processed videos[/yellow]")
        except Exception as e:
            logger.warning(f"Could not check for processed videos: {e}")
    
    # Filter out already processed files
    files_to_process = []
    skipped_count = 0
    
    if skip_processed and processed_video_ids:
        for srt_file in srt_files:
            # Extract video ID from filename (simple heuristic)
            filename = srt_file.stem
            # Try to find video ID in filename (11 character alphanumeric)
            video_id = None
            for i in range(len(filename) - 10):
                candidate = filename[i:i+11]
                if candidate.isalnum() and len(candidate) == 11:
                    video_id = candidate
                    break
            
            if video_id and video_id in processed_video_ids:
                skipped_count += 1
                continue
            
            files_to_process.append(srt_file)
    else:
        files_to_process = srt_files
    
    if not files_to_process:
        console.print("[yellow]All files are already processed![/yellow]")
        return {
            "total_files": len(srt_files),
            "processed": 0,
            "skipped": skipped_count,
            "failed": 0,
            "total_chunks": 0,
            "total_indexed": 0
        }
    
    console.print(f"[green]Processing {len(files_to_process)} files[/green]")
    if skipped_count > 0:
        console.print(f"[yellow]Skipping {skipped_count} already processed files[/yellow]")
    
    # Statistics
    stats = {
        "total_files": len(srt_files),
        "processed": 0,
        "skipped": skipped_count,
        "failed": 0,
        "total_chunks": 0,
        "total_indexed": 0,
        "errors": []
    }
    
    # Process files in batches to manage memory
    batch_size_processing = getattr(config, "PROCESSING_BATCH_SIZE", 10)
    
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        TimeElapsedColumn(),
        console=console
    ) as progress:
        
        # Phase 1: Preprocessing
        task_preprocess = progress.add_task(
            "[cyan]Preprocessing SRT files...",
            total=len(files_to_process)
        )
        
        processed_videos = []
        for i in range(0, len(files_to_process), batch_size_processing):
            batch_files = files_to_process[i:i+batch_size_processing]
            
            try:
                # Use None to let the pipeline auto-detect from config
                workers = max_workers if max_workers is not None else None
                batch_processed = preprocessing_pipeline.process_multiple_files(
                    batch_files,
                    skip_errors=True,
                    parallel=parallel_preprocessing,
                    max_workers=workers
                )
                processed_videos.extend(batch_processed)
                progress.update(task_preprocess, advance=len(batch_files))
            except Exception as e:
                logger.error(f"Error preprocessing batch: {e}", exc_info=True)
                stats["failed"] += len(batch_files)
                progress.update(task_preprocess, advance=len(batch_files))
        
        # Filter out None results
        processed_videos = [v for v in processed_videos if v is not None]
        stats["processed"] = len(processed_videos)
        stats["total_chunks"] = sum(len(v.chunks) for v in processed_videos)
        
        if not processed_videos:
            console.print("[bold red]No videos were successfully processed![/bold red]")
            return stats
        
        console.print(f"[green]✓ Preprocessed {len(processed_videos)} videos, {stats['total_chunks']} total chunks[/green]")
        
        # Phase 2: Embedding Generation
        task_embeddings = progress.add_task(
            "[cyan]Generating embeddings...",
            total=len(processed_videos)
        )
        
        embeddings_list = []
        for processed_video in processed_videos:
            try:
                embeddings, metadata = embedding_pipeline.generate_embeddings_with_checkpointing(
                    processed_video,
                    show_progress=False  # We have our own progress bar
                )
                embeddings_list.append(embeddings)
                progress.update(task_embeddings, advance=1)
            except Exception as e:
                logger.error(
                    f"Error generating embeddings for {processed_video.metadata.filename}: {e}",
                    exc_info=True
                )
                stats["failed"] += 1
                stats["errors"].append(f"Embedding generation failed for {processed_video.metadata.filename}: {str(e)}")
                # Add None to keep alignment
                embeddings_list.append(None)
                progress.update(task_embeddings, advance=1)
        
        # Filter out None embeddings
        valid_pairs = [(v, e) for v, e in zip(processed_videos, embeddings_list) if e is not None]
        processed_videos = [v for v, e in valid_pairs]
        embeddings_list = [e for v, e in valid_pairs]
        
        console.print(f"[green]✓ Generated embeddings for {len(processed_videos)} videos[/green]")
        
        # Phase 3: Indexing
        task_indexing = progress.add_task(
            "[cyan]Indexing in vector store...",
            total=len(processed_videos)
        )
        
        indexed_counts = {}
        for processed_video, embeddings in zip(processed_videos, embeddings_list):
            try:
                indexed_count = vector_store_pipeline.index_processed_video(
                    processed_video,
                    embeddings,
                    skip_duplicates=True,
                    show_progress=False
                )
                indexed_counts[processed_video.metadata.video_id] = indexed_count
                stats["total_indexed"] += indexed_count
                progress.update(task_indexing, advance=1)
            except Exception as e:
                logger.error(
                    f"Error indexing {processed_video.metadata.filename}: {e}",
                    exc_info=True
                )
                stats["failed"] += 1
                stats["errors"].append(f"Indexing failed for {processed_video.metadata.filename}: {str(e)}")
                progress.update(task_indexing, advance=1)
        
        console.print(f"[green]✓ Indexed {stats['total_indexed']} chunks[/green]")
    
    return stats


def print_summary(stats: dict, console: Console):
    """Print processing summary."""
    table = Table(title="Processing Summary", show_header=True, header_style="bold magenta")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="green", justify="right")
    
    table.add_row("Total SRT files", str(stats["total_files"]))
    table.add_row("Successfully processed", str(stats["processed"]))
    table.add_row("Skipped (already processed)", str(stats["skipped"]))
    table.add_row("Failed", str(stats["failed"]))
    table.add_row("Total chunks created", str(stats["total_chunks"]))
    table.add_row("Total chunks indexed", str(stats["total_indexed"]))
    
    console.print("\n")
    console.print(table)
    
    if stats["errors"]:
        console.print("\n[bold red]Errors encountered:[/bold red]")
        for error in stats["errors"][:10]:  # Show first 10 errors
            console.print(f"  • {error}")
        if len(stats["errors"]) > 10:
            console.print(f"  ... and {len(stats['errors']) - 10} more errors")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Process subtitle files and index them in vector store"
    )
    parser.add_argument(
        "input_dir",
        type=Path,
        help="Directory containing SRT files"
    )
    parser.add_argument(
        "--no-skip",
        action="store_true",
        help="Don't skip already processed videos"
    )
    parser.add_argument(
        "--no-parallel",
        action="store_true",
        help="Disable parallel preprocessing"
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        default=None,
        help="Maximum number of workers for parallel processing (default: auto-detect from CPU count)"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Batch size for embedding generation"
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress progress bars"
    )
    
    args = parser.parse_args()
    
    if not args.input_dir.exists():
        print(f"Error: Input directory does not exist: {args.input_dir}")
        sys.exit(1)
    
    console = Console()
    
    console.print(Panel.fit(
        "[bold blue]Subtitle Embedding & Retrieval System[/bold blue]\n"
        "[cyan]Processing Pipeline[/cyan]",
        border_style="blue"
    ))
    
    try:
        stats = process_pipeline(
            input_dir=args.input_dir,
            skip_processed=not args.no_skip,
            parallel_preprocessing=not args.no_parallel,
            max_workers=args.max_workers,
            batch_size=args.batch_size,
            show_progress=not args.quiet
        )
        
        print_summary(stats, console)
        
        # Get final index statistics
        try:
            vector_store_pipeline = VectorStorePipeline()
            index_stats = vector_store_pipeline.get_index_statistics()
            console.print("\n[bold blue]Vector Store Statistics:[/bold blue]")
            console.print(f"  Total documents: {index_stats.get('total_documents', 'N/A')}")
            console.print(f"  Total videos: {index_stats.get('total_videos', 'N/A')}")
        except Exception as e:
            console.print(f"[yellow]Could not retrieve index statistics: {e}[/yellow]")
        
    except KeyboardInterrupt:
        console.print("\n[yellow]Processing interrupted by user[/yellow]")
        sys.exit(1)
    except Exception as e:
        console.print(f"[bold red]Fatal error: {e}[/bold red]")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
