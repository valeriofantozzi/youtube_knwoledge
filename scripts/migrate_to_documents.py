#!/usr/bin/env python3
"""
Migration script: Migrate existing vector database from video_id to source_id.

This script updates the existing ChromaDB collection to use the new field names:
- video_id -> source_id
- Adds content_type field for existing data

Run this script once to migrate existing data to the new document-centric schema.
"""

import argparse
import sys
from pathlib import Path
from typing import Dict, Any, List, Optional
from rich.console import Console  # type: ignore[import-untyped]
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskID  # type: ignore[import-untyped]
from rich.table import Table  # type: ignore[import-untyped]

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.vector_store.chroma_manager import ChromaDBManager
from src.utils.logger import get_default_logger
from src.utils.config import get_config


def get_collection_statistics(collection) -> Dict[str, Any]:
    """Get statistics about the collection."""
    count = collection.count()
    
    if count == 0:
        return {
            "total_documents": 0,
            "with_video_id": 0,
            "with_source_id": 0,
            "with_content_type": 0,
            "unique_sources": 0
        }
    
    # Sample documents to analyze metadata
    sample_size = min(count, 10000)
    sample = collection.get(
        limit=sample_size,
        include=["metadatas"]
    )
    
    with_video_id = 0
    with_source_id = 0
    with_content_type = 0
    unique_sources = set()
    
    for metadata in sample.get("metadatas", []):
        if metadata:
            if "video_id" in metadata:
                with_video_id += 1
                unique_sources.add(metadata["video_id"])
            if "source_id" in metadata:
                with_source_id += 1
                unique_sources.add(metadata["source_id"])
            if "content_type" in metadata:
                with_content_type += 1
    
    return {
        "total_documents": count,
        "sampled": sample_size,
        "with_video_id": with_video_id,
        "with_source_id": with_source_id,
        "with_content_type": with_content_type,
        "unique_sources": len(unique_sources)
    }


def migrate_metadata(
    collection,
    batch_size: int = 1000,
    dry_run: bool = False,
    console: Optional[Console] = None
) -> Dict[str, int]:
    """
    Migrate metadata from video_id to source_id schema.
    
    Args:
        collection: ChromaDB collection
        batch_size: Number of documents to process per batch
        dry_run: If True, show what would be migrated without making changes
        console: Rich console for output
    
    Returns:
        Dictionary with migration statistics
    """
    if console is None:
        console = Console()
    
    _console: Console = console  # Ensure type checker knows it's not None
    logger = get_default_logger()
    
    total_count = collection.count()
    migrated = 0
    skipped = 0
    errors = 0
    
    if total_count == 0:
        _console.print("[yellow]Collection is empty, nothing to migrate.[/yellow]")
        return {"migrated": 0, "skipped": 0, "errors": 0}
    
    _console.print(f"[cyan]Processing {total_count} documents in batches of {batch_size}...[/cyan]")
    
    # Process in batches
    offset = 0
    
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        console=_console
    ) as progress:
        task = progress.add_task(
            "[cyan]Migrating metadata...",
            total=total_count
        )
        
        while offset < total_count:
            # Get batch
            batch = collection.get(
                limit=batch_size,
                offset=offset,
                include=["metadatas"]
            )
            
            ids = batch.get("ids", [])
            metadatas = batch.get("metadatas", [])
            
            if not ids:
                break
            
            # Process batch
            ids_to_update = []
            new_metadatas = []
            
            for doc_id, metadata in zip(ids, metadatas):
                if not metadata:
                    skipped += 1
                    continue
                
                # Check if migration needed
                needs_migration = False
                new_metadata = dict(metadata)
                
                # Migrate video_id -> source_id
                if "video_id" in metadata and "source_id" not in metadata:
                    new_metadata["source_id"] = metadata["video_id"]
                    needs_migration = True
                
                # Add content_type for SRT files (inferred from existing data)
                if "content_type" not in metadata:
                    # Infer from filename if available
                    filename = metadata.get("filename", "")
                    if filename.lower().endswith(".srt") or filename.lower().endswith(".sub"):
                        new_metadata["content_type"] = "srt"
                    else:
                        # Default to srt for legacy data
                        new_metadata["content_type"] = "srt"
                    needs_migration = True
                
                if needs_migration:
                    ids_to_update.append(doc_id)
                    new_metadatas.append(new_metadata)
                else:
                    skipped += 1
            
            # Apply updates
            if ids_to_update and not dry_run:
                try:
                    collection.update(
                        ids=ids_to_update,
                        metadatas=new_metadatas
                    )
                    migrated += len(ids_to_update)
                except Exception as e:
                    logger.error(f"Error updating batch: {e}")
                    errors += len(ids_to_update)
            elif ids_to_update:
                # Dry run - just count
                migrated += len(ids_to_update)
            
            offset += batch_size
            progress.update(task, advance=len(ids))
    
    return {
        "migrated": migrated,
        "skipped": skipped,
        "errors": errors
    }


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Migrate existing vector database to document-centric schema"
    )
    
    parser.add_argument(
        "--db-path",
        type=str,
        help="Path to ChromaDB directory (default: from config)"
    )
    
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1000,
        help="Batch size for processing (default: 1000)"
    )
    
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be migrated without making changes"
    )
    
    parser.add_argument(
        "--force",
        action="store_true",
        help="Skip confirmation prompt"
    )
    
    args = parser.parse_args()
    
    console = Console()
    logger = get_default_logger()
    config = get_config()
    
    # Initialize ChromaDB manager
    db_path = args.db_path or getattr(config, "CHROMA_DB_PATH", "./data/vector_db")
    
    console.print(f"[bold blue]Vector Database Migration Tool[/bold blue]")
    console.print(f"Database path: {db_path}")
    console.print()
    
    try:
        manager = ChromaDBManager(db_path=db_path)
        collection = manager.get_or_create_collection()
    except Exception as e:
        console.print(f"[bold red]Error initializing database: {e}[/bold red]")
        return 1
    
    # Show current statistics
    console.print("[cyan]Analyzing current collection...[/cyan]")
    stats = get_collection_statistics(collection)
    
    table = Table(title="Collection Statistics", show_header=True, header_style="bold magenta")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="green", justify="right")
    
    table.add_row("Total documents", str(stats["total_documents"]))
    table.add_row("With video_id", str(stats["with_video_id"]))
    table.add_row("With source_id", str(stats["with_source_id"]))
    table.add_row("With content_type", str(stats["with_content_type"]))
    table.add_row("Unique sources", str(stats["unique_sources"]))
    
    console.print(table)
    console.print()
    
    # Check if migration needed
    if stats["with_source_id"] == stats["total_documents"] and stats["with_content_type"] == stats["total_documents"]:
        console.print("[green]✓ Collection is already migrated. No changes needed.[/green]")
        return 0
    
    # Estimate documents to migrate
    to_migrate = stats["total_documents"] - min(stats["with_source_id"], stats["with_content_type"])
    console.print(f"[yellow]Estimated documents to migrate: {to_migrate}[/yellow]")
    console.print()
    
    if args.dry_run:
        console.print("[yellow]DRY RUN MODE - No changes will be made[/yellow]")
    
    # Confirmation
    if not args.force and not args.dry_run:
        response = input("Proceed with migration? (y/N): ")
        if response.lower() != 'y':
            console.print("Migration cancelled.")
            return 0
    
    # Run migration
    console.print()
    result = migrate_metadata(
        collection,
        batch_size=args.batch_size,
        dry_run=args.dry_run,
        console=console
    )
    
    # Show results
    console.print()
    
    result_table = Table(title="Migration Results", show_header=True, header_style="bold magenta")
    result_table.add_column("Metric", style="cyan")
    result_table.add_column("Value", style="green", justify="right")
    
    result_table.add_row("Documents migrated", str(result["migrated"]))
    result_table.add_row("Documents skipped", str(result["skipped"]))
    result_table.add_row("Errors", str(result["errors"]))
    
    console.print(result_table)
    
    if args.dry_run:
        console.print()
        console.print("[yellow]This was a dry run. Run without --dry-run to apply changes.[/yellow]")
    elif result["errors"] == 0:
        console.print()
        console.print("[bold green]✓ Migration completed successfully![/bold green]")
        logger.info(f"Migration completed: {result['migrated']} migrated, {result['skipped']} skipped")
    else:
        console.print()
        console.print(f"[bold yellow]Migration completed with {result['errors']} errors.[/bold yellow]")
        logger.warning(f"Migration completed with errors: {result}")
    
    return 0 if result["errors"] == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
