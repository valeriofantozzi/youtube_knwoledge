#!/usr/bin/env python3
"""Check for duplicate chunks in the vector database."""

import sys
from pathlib import Path
from collections import defaultdict
from rich.console import Console
from rich.table import Table
from rich import box

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.vector_store.pipeline import VectorStorePipeline


def check_duplicates():
    """Detect duplicate chunks based on text content."""
    console = Console()
    
    console.print("[bold blue]Checking for duplicate chunks...[/bold blue]\n")
    
    try:
        pipeline = VectorStorePipeline()
        collection = pipeline.chroma_manager.get_or_create_collection()
        
        # Get all documents
        all_docs = collection.get(
            limit=100000,
            include=["documents", "metadatas"]
        )
        
        total_chunks = len(all_docs['ids'])
        console.print(f"Total chunks in database: {total_chunks}\n")
        
        if total_chunks == 0:
            console.print("[yellow]Database is empty[/yellow]")
            return 0
        
        # Group by text content
        text_to_docs = defaultdict(list)
        
        for doc_id, text, meta in zip(
            all_docs["ids"],
            all_docs["documents"],
            all_docs["metadatas"]
        ):
            text_to_docs[text].append({
                "id": doc_id,
                "source_id": meta.get("source_id", "N/A"),
                "chunk_index": meta.get("chunk_index", "N/A"),
                "filename": meta.get("filename", "N/A")
            })
        
        # Find duplicates (text appearing more than once)
        duplicates = {
            text: docs
            for text, docs in text_to_docs.items()
            if len(docs) > 1
        }
        
        if not duplicates:
            console.print("[bold green]✅ No duplicates found! Database is clean.[/bold green]")
            return 0
        
        # Report duplicates
        total_duplicate_chunks = sum(len(docs) for docs in duplicates.values())
        unique_texts_duplicated = len(duplicates)
        
        console.print(f"[bold red]❌ Found duplicates:[/bold red]")
        console.print(f"  • {unique_texts_duplicated} unique texts are duplicated")
        console.print(f"  • {total_duplicate_chunks} total duplicate chunks")
        console.print(f"  • {total_duplicate_chunks - unique_texts_duplicated} unnecessary duplicates\n")
        
        # Show sample duplicates
        console.print("[bold yellow]Sample Duplicates (first 5):[/bold yellow]\n")
        
        for i, (text, docs) in enumerate(list(duplicates.items())[:5], 1):
            table = Table(
                title=f"Duplicate #{i} ({len(docs)} copies)",
                box=box.ROUNDED,
                show_header=True
            )
            table.add_column("Source ID", style="cyan", width=30)
            table.add_column("Chunk", style="yellow", justify="center")
            table.add_column("Filename", style="white", width=40)
            
            for doc in docs:
                source_id = doc['source_id']
                if len(source_id) > 27:
                    source_id = source_id[:27] + "..."
                
                filename = doc['filename']
                if len(filename) > 37:
                    filename = filename[:37] + "..."
                
                table.add_row(
                    source_id,
                    str(doc['chunk_index']),
                    filename
                )
            
            console.print(table)
            text_preview = text[:100]
            if len(text) > 100:
                text_preview += "..."
            console.print(f"[dim]Text preview: {text_preview}[/dim]\n")
        
        return len(duplicates)
    
    except Exception as e:
        console.print(f"[bold red]Error:[/bold red] {e}")
        import traceback
        traceback.print_exc()
        return -1


if __name__ == "__main__":
    dup_count = check_duplicates()
    if dup_count > 0:
        print(f"\n⚠️  Found {dup_count} groups of duplicates. Re-index the database to remove them.")
        sys.exit(1)
    elif dup_count == 0:
        print("\n✅ Database is clean!")
        sys.exit(0)
    else:
        print("\n❌ Error checking duplicates")
        sys.exit(2)
