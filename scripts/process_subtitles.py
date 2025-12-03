#!/usr/bin/env python3
"""
Main script for processing document files.

Processes document files (SRT, TXT, MD), generates embeddings, and indexes them in the vector store.
"""

import argparse
import sys
import glob
from pathlib import Path
from typing import List, Optional, Set
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TimeElapsedColumn, TaskID
from rich.table import Table
from rich.panel import Panel

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.preprocessing.pipeline import PreprocessingPipeline, ProcessedDocument
from src.embeddings.pipeline import EmbeddingPipeline
from src.vector_store.pipeline import VectorStorePipeline
from src.utils.logger import get_default_logger
from src.utils.config import get_config


# Supported file extensions
SUPPORTED_EXTENSIONS = {'.srt', '.sub', '.txt', '.text', '.md', '.markdown', '.mdown'}


def find_document_files(input_dir: Path, max_files: Optional[int] = None, extensions: Optional[Set[str]] = None) -> List[Path]:
    """Find all supported document files in the input directory."""
    if extensions is None:
        extensions = SUPPORTED_EXTENSIONS
    
    doc_files = []
    for ext in extensions:
        doc_files.extend(input_dir.rglob(f"*{ext}"))
        doc_files.extend(input_dir.rglob(f"*{ext.upper()}"))
    
    doc_files = sorted(set(doc_files))
    if max_files is not None:
        doc_files = doc_files[:max_files]
    return doc_files


# Backward compatibility alias
find_srt_files = find_document_files


def process_pipeline(
    input_dir: Path,
    skip_processed: bool = True,
    parallel_preprocessing: bool = True,
    max_workers: Optional[int] = None,
    batch_size: Optional[int] = None,
    show_progress: bool = True,
    model_name: Optional[str] = None,
    max_files: Optional[int] = None,
    extensions: Optional[Set[str]] = None
) -> dict:
    """
    Run the complete pipeline: preprocessing -> embeddings -> indexing.

    Args:
        input_dir: Directory containing document files
        skip_processed: Skip documents that are already indexed
        parallel_preprocessing: Use parallel processing for preprocessing
        max_workers: Max workers for parallel processing
        batch_size: Batch size for embedding generation
        show_progress: Show progress bars
        model_name: Embedding model name (None uses config default)
        max_files: Maximum number of files to process
        extensions: Set of file extensions to process (None = all supported)

    Returns:
        Dictionary with processing statistics
    """
    console = Console()
    logger = get_default_logger()
    config = get_config()
    
    # Initialize pipelines
    console.print("[bold blue]Initializing pipelines...[/bold blue]")

    # Display model information
    if model_name:
        console.print(f"[cyan]Using embedding model: {model_name}[/cyan]")
        # Try to get model metadata
        try:
            from ..embeddings.model_registry import get_model_registry
            registry = get_model_registry()
            metadata = registry.get_model_metadata(model_name)
            if metadata:
                console.print(f"[cyan]Model info: {metadata.embedding_dimension}D, adapter: {metadata.adapter_class.__name__}[/cyan]")
        except Exception as e:
            logger.debug(f"Could not retrieve model metadata: {e}")

    preprocessing_pipeline = PreprocessingPipeline()
    embedding_pipeline = EmbeddingPipeline(
        batch_size=batch_size,
        enable_optimizations=True,  # Enable hardware-aware optimizations (Task 4.5)
        model_name=model_name  # Pass model name to embedding pipeline
    )
    vector_store_pipeline = VectorStorePipeline(model_name=model_name)
    
    # Find document files
    ext_str = ", ".join(extensions) if extensions else "all supported types"
    console.print(f"[bold blue]Scanning for document files ({ext_str}) in {input_dir}...[/bold blue]")
    doc_files = find_document_files(input_dir, max_files, extensions)
    
    if not doc_files:
        console.print("[bold red]No document files found![/bold red]")
        return {
            "total_files": 0,
            "processed": 0,
            "skipped": 0,
            "failed": 0,
            "total_chunks": 0,
            "total_indexed": 0
        }
    
    console.print(f"[green]Found {len(doc_files)} document files[/green]")
    
    # Check for already processed documents if skip_processed is True
    processed_source_ids = set()
    if skip_processed:
        try:
            stats = vector_store_pipeline.get_index_statistics()
            # Get all source IDs from the index
            collection = vector_store_pipeline.chroma_manager.get_or_create_collection()
            # Sample some documents to check source IDs
            sample = collection.get(limit=1000)
            if sample.get("metadatas"):
                for metadata in sample["metadatas"]:
                    if metadata:
                        # Support both source_id and legacy video_id
                        sid = metadata.get("source_id") or metadata.get("video_id")
                        if sid:
                            processed_source_ids.add(sid)
            console.print(f"[yellow]Found {len(processed_source_ids)} already processed documents[/yellow]")
        except Exception as e:
            logger.warning(f"Could not check for processed documents: {e}")
    
    # Filter out already processed files
    files_to_process = []
    skipped_count = 0
    
    if skip_processed and processed_source_ids:
        for doc_file in doc_files:
            # Extract source ID from filename
            filename = doc_file.stem
            # Try to find source ID in filename (11 character alphanumeric for YouTube videos)
            source_id = None
            for i in range(len(filename) - 10):
                candidate = filename[i:i+11]
                if candidate.isalnum() and len(candidate) == 11:
                    source_id = candidate
                    break
            
            # If no YouTube-style ID found, use the full filename stem as source ID
            if source_id is None:
                source_id = filename
            
            if source_id in processed_source_ids:
                skipped_count += 1
                continue
            
            files_to_process.append(doc_file)
    else:
        files_to_process = doc_files
    
    if not files_to_process:
        console.print("[yellow]All files are already processed![/yellow]")
        return {
            "total_files": len(doc_files),
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
        "total_files": len(doc_files),
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
            "[cyan]Preprocessing document files...",
            total=len(files_to_process)
        )
        
        processed_documents = []
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
                processed_documents.extend(batch_processed)
                progress.update(task_preprocess, advance=len(batch_files))
            except Exception as e:
                logger.error(f"Error preprocessing batch: {e}", exc_info=True)
                stats["failed"] += len(batch_files)
                progress.update(task_preprocess, advance=len(batch_files))
        
        # Filter out None results
        processed_documents = [d for d in processed_documents if d is not None]
        stats["processed"] = len(processed_documents)
        stats["total_chunks"] = sum(len(d.chunks) for d in processed_documents)
        
        if not processed_documents:
            console.print("[bold red]No documents were successfully processed![/bold red]")
            return stats
        
        console.print(f"[green]✓ Preprocessed {len(processed_documents)} documents, {stats['total_chunks']} total chunks[/green]")
        
        # Phase 2: Embedding Generation
        task_embeddings = progress.add_task(
            "[cyan]Generating embeddings...",
            total=len(processed_documents)
        )
        
        embeddings_list = []
        actual_model_names = []
        for processed_document in processed_documents:
            try:
                embeddings, metadata, actual_model_name = embedding_pipeline.generate_embeddings_with_checkpointing(
                    processed_document,
                    show_progress=False  # We have our own progress bar
                )
                embeddings_list.append(embeddings)
                actual_model_names.append(actual_model_name)
                progress.update(task_embeddings, advance=1)
            except Exception as e:
                logger.error(
                    f"Error generating embeddings for {processed_document.metadata.filename}: {e}",
                    exc_info=True
                )
                stats["failed"] += 1
                stats["errors"].append(f"Embedding generation failed for {processed_document.metadata.filename}: {str(e)}")
                # Add None to keep alignment
                embeddings_list.append(None)
                progress.update(task_embeddings, advance=1)
        
        # Filter out None embeddings
        valid_pairs = [(d, e) for d, e in zip(processed_documents, embeddings_list) if e is not None]
        processed_documents = [d for d, e in valid_pairs]
        embeddings_list = [e for d, e in valid_pairs]
        
        console.print(f"[green]✓ Generated embeddings for {len(processed_documents)} documents[/green]")
        
        # Phase 3: Indexing
        task_indexing = progress.add_task(
            "[cyan]Indexing in vector store...",
            total=len(processed_documents)
        )
        
        indexed_counts = {}
        for processed_document, embeddings, actual_model_name in zip(processed_documents, embeddings_list, actual_model_names):
            try:
                indexed_count = vector_store_pipeline.index_processed_document(
                    processed_document,
                    embeddings,
                    skip_duplicates=True,
                    show_progress=False,
                    model_name=actual_model_name
                )
                indexed_counts[processed_document.metadata.source_id] = indexed_count
                stats["total_indexed"] += indexed_count
                progress.update(task_indexing, advance=1)
            except Exception as e:
                logger.error(
                    f"Error indexing {processed_document.metadata.filename}: {e}",
                    exc_info=True
                )
                stats["failed"] += 1
                stats["errors"].append(f"Indexing failed for {processed_document.metadata.filename}: {str(e)}")
                progress.update(task_indexing, advance=1)
        
        console.print(f"[green]✓ Indexed {stats['total_indexed']} chunks[/green]")
    
    return stats


def print_summary(stats: dict, console: Console):
    """Print processing summary."""
    table = Table(title="Processing Summary", show_header=True, header_style="bold magenta")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="green", justify="right")
    
    table.add_row("Total document files", str(stats["total_files"]))
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
        description="Process document files and index them in vector store"
    )
    parser.add_argument(
        "input_dir",
        type=Path,
        help="Directory containing document files (SRT, TXT, MD)"
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
        "--model",
        type=str,
        default=None,
        help="Embedding model to use (default: from config or BAAI/bge-large-en-v1.5). "
             "Examples: 'BAAI/bge-large-en-v1.5', 'google/embeddinggemma-300m'"
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress progress bars"
    )
    parser.add_argument(
        "--max-files",
        type=int,
        default=None,
        help="Maximum number of SRT files to process (useful for testing)"
    )

    args = parser.parse_args()
    
    if not args.input_dir.exists():
        print(f"Error: Input directory does not exist: {args.input_dir}")
        sys.exit(1)
    
    console = Console()
    
    # Clear checkpoint files before processing
    checkpoint_dir = Path(__file__).parent.parent / "data" / "checkpoints"
    if checkpoint_dir.exists():
        checkpoint_files = glob.glob(str(checkpoint_dir / "*.pkl"))
        if checkpoint_files:
            console.print(f"[yellow]Removing {len(checkpoint_files)} checkpoint file(s)...[/yellow]")
            for checkpoint_file in checkpoint_files:
                try:
                    Path(checkpoint_file).unlink()
                except Exception as e:
                    console.print(f"[red]Warning: Could not remove {checkpoint_file}: {e}[/red]")
            console.print("[green]✓ Checkpoints cleared[/green]")
    
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
            show_progress=not args.quiet,
            model_name=args.model,
            max_files=args.max_files
        )
        
        print_summary(stats, console)
        
        # Get final index statistics
        try:
            vector_store_pipeline = VectorStorePipeline(model_name=args.model)
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
