#!/usr/bin/env python3
"""
Script to explore and visualize the vector database.

Shows statistics, sample documents, and allows interactive exploration.
"""

import argparse
import sys
from pathlib import Path
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich import box
from collections import Counter

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.vector_store.pipeline import VectorStorePipeline
from src.vector_store.chroma_manager import ChromaDBManager


def show_statistics(pipeline: VectorStorePipeline, console: Console):
    """Show database statistics."""
    collection = pipeline.chroma_manager.get_or_create_collection()
    total_docs = collection.count()
    
    # Get sample to analyze
    sample_size = min(1000, total_docs)
    sample = collection.get(limit=sample_size)
    
    # Analyze metadata
    video_ids = []
    dates = []
    titles = []
    
    if sample.get('metadatas'):
        for meta in sample['metadatas']:
            if meta:
                if 'video_id' in meta:
                    video_ids.append(meta['video_id'])
                if 'date' in meta:
                    dates.append(meta['date'])
                if 'title' in meta:
                    titles.append(meta['title'])
    
    unique_videos = len(set(video_ids)) if video_ids else 0
    avg_chunks_per_video = total_docs / unique_videos if unique_videos > 0 else 0
    
    # Create statistics table
    stats_table = Table(title="📊 Vector Database Statistics", box=box.ROUNDED)
    stats_table.add_column("Metric", style="cyan", width=30)
    stats_table.add_column("Value", style="green", justify="right")
    
    stats_table.add_row("Total Documents", str(total_docs))
    stats_table.add_row("Unique Videos", str(unique_videos))
    stats_table.add_row("Avg Chunks/Video", f"{avg_chunks_per_video:.1f}")
    
    if sample.get('embeddings') and sample['embeddings']:
        emb_dim = len(sample['embeddings'][0])
        stats_table.add_row("Embedding Dimension", str(emb_dim))
    
    console.print(stats_table)
    
    # Show date distribution
    if dates:
        date_counter = Counter(dates)
        date_table = Table(title="📅 Date Distribution (Top 10)", box=box.ROUNDED)
        date_table.add_column("Date", style="cyan")
        date_table.add_column("Videos", style="green", justify="right")
        
        for date, count in date_counter.most_common(10):
            date_table.add_row(date, str(count))
        
        console.print("\n")
        console.print(date_table)


def show_sample_documents(pipeline: VectorStorePipeline, console: Console, num_samples: int = 10):
    """Show sample documents from the database."""
    collection = pipeline.chroma_manager.get_or_create_collection()
    sample = collection.get(limit=num_samples)
    
    if not sample.get('ids'):
        console.print("[yellow]No documents found in database[/yellow]")
        return
    
    console.print(f"\n[bold blue]📄 Sample Documents (showing {len(sample['ids'])} of {collection.count()})[/bold blue]\n")
    
    for i, (doc_id, doc_text, metadata) in enumerate(zip(
        sample['ids'],
        sample.get('documents', [''] * len(sample['ids'])),
        sample.get('metadatas', [{}] * len(sample['ids']))
    ), 1):
        panel_content = []
        
        if metadata:
            if 'video_id' in metadata:
                panel_content.append(f"[cyan]Video ID:[/cyan] {metadata['video_id']}")
            if 'title' in metadata:
                panel_content.append(f"[cyan]Title:[/cyan] {metadata['title']}")
            if 'date' in metadata:
                panel_content.append(f"[cyan]Date:[/cyan] {metadata['date']}")
            if 'chunk_index' in metadata:
                panel_content.append(f"[cyan]Chunk Index:[/cyan] {metadata['chunk_index']}")
        
        panel_content.append(f"\n[white]{doc_text[:200]}{'...' if len(doc_text) > 200 else ''}[/white]")
        
        console.print(Panel(
            "\n".join(panel_content),
            title=f"Document {i}",
            border_style="blue"
        ))
        console.print()


def show_video_list(pipeline: VectorStorePipeline, console: Console, limit: int = 20):
    """Show list of videos in the database."""
    collection = pipeline.chroma_manager.get_or_create_collection()
    
    # Get all documents to extract unique videos
    all_data = collection.get()
    
    videos = {}
    if all_data.get('metadatas'):
        for meta in all_data['metadatas']:
            if meta and 'video_id' in meta:
                vid_id = meta['video_id']
                if vid_id not in videos:
                    videos[vid_id] = {
                        'title': meta.get('title', 'N/A'),
                        'date': meta.get('date', 'N/A'),
                        'chunks': 0
                    }
                videos[vid_id]['chunks'] += 1
    
    # Sort by chunks (descending)
    sorted_videos = sorted(videos.items(), key=lambda x: x[1]['chunks'], reverse=True)
    
    video_table = Table(title=f"📹 Videos in Database (showing {min(limit, len(sorted_videos))} of {len(sorted_videos)})", box=box.ROUNDED)
    video_table.add_column("#", style="dim", width=4)
    video_table.add_column("Video ID", style="cyan", width=15)
    video_table.add_column("Title", style="white", width=50)
    video_table.add_column("Date", style="yellow", width=12)
    video_table.add_column("Chunks", style="green", justify="right", width=8)
    
    for i, (vid_id, info) in enumerate(sorted_videos[:limit], 1):
        title = info['title'][:47] + '...' if len(info['title']) > 50 else info['title']
        video_table.add_row(
            str(i),
            vid_id,
            title,
            info['date'],
            str(info['chunks'])
        )
    
    console.print(video_table)


def search_documents(pipeline: VectorStorePipeline, console: Console, query: str, top_k: int = 5):
    """Search documents using semantic search."""
    from src.embeddings.embedder import Embedder
    
    console.print(f"\n[bold blue]🔍 Searching for: '{query}'[/bold blue]\n")
    
    try:
        # Generate query embedding
        embedder = Embedder()
        query_embedding = embedder.encode([query], is_query=True)[0]
        
        # Search in collection
        collection = pipeline.chroma_manager.get_or_create_collection()
        results = collection.query(
            query_embeddings=[query_embedding.tolist()],
            n_results=top_k,
            include=['documents', 'metadatas', 'distances']
        )
        
        if not results.get('ids') or not results['ids'][0]:
            console.print("[yellow]No results found[/yellow]")
            return
        
        # Display results
        for i, (doc_id, doc_text, metadata, distance) in enumerate(zip(
            results['ids'][0],
            results['documents'][0],
            results['metadatas'][0],
            results['distances'][0]
        ), 1):
            similarity = 1 - distance  # Convert distance to similarity
            
            panel_content = []
            if metadata:
                if 'video_id' in metadata:
                    panel_content.append(f"[cyan]Video ID:[/cyan] {metadata['video_id']}")
                if 'title' in metadata:
                    panel_content.append(f"[cyan]Title:[/cyan] {metadata['title']}")
                if 'date' in metadata:
                    panel_content.append(f"[cyan]Date:[/cyan] {metadata['date']}")
            
            panel_content.append(f"\n[green]Similarity: {similarity:.3f}[/green]")
            panel_content.append(f"\n[white]{doc_text[:300]}{'...' if len(doc_text) > 300 else ''}[/white]")
            
            console.print(Panel(
                "\n".join(panel_content),
                title=f"Result {i}",
                border_style="green"
            ))
            console.print()
    
    except Exception as e:
        console.print(f"[red]Error during search: {e}[/red]")
        import traceback
        traceback.print_exc()


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Explore and visualize the vector database"
    )
    parser.add_argument(
        "--stats",
        action="store_true",
        help="Show database statistics"
    )
    parser.add_argument(
        "--samples",
        type=int,
        default=10,
        help="Number of sample documents to show (default: 10)"
    )
    parser.add_argument(
        "--videos",
        type=int,
        default=20,
        help="Number of videos to list (default: 20)"
    )
    parser.add_argument(
        "--search",
        type=str,
        help="Search query (semantic search)"
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="Number of results for search (default: 5)"
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Show all information (stats, samples, videos)"
    )
    
    args = parser.parse_args()
    
    console = Console()
    
    # Initialize pipeline
    console.print("[bold blue]Initializing vector store...[/bold blue]")
    pipeline = VectorStorePipeline()
    
    # Show header
    console.print(Panel.fit(
        "[bold blue]Vector Database Explorer[/bold blue]\n"
        "[cyan]ChromaDB Visualization Tool[/cyan]",
        border_style="blue"
    ))
    console.print()
    
    # Execute requested actions
    if args.all or args.stats:
        show_statistics(pipeline, console)
        console.print()
    
    if args.all or args.samples:
        show_sample_documents(pipeline, console, args.samples)
        console.print()
    
    if args.all or args.videos:
        show_video_list(pipeline, console, args.videos)
        console.print()
    
    if args.search:
        search_documents(pipeline, console, args.search, args.top_k)
    
    # If no arguments, show default view
    if not any([args.stats, args.samples, args.videos, args.search, args.all]):
        show_statistics(pipeline, console)
        console.print()
        show_sample_documents(pipeline, console, 5)
        console.print()
        show_video_list(pipeline, console, 10)
        console.print()
        console.print("[dim]Use --help to see all options[/dim]")
        console.print("[dim]Example: --search 'orchid care' --top-k 10[/dim]")


if __name__ == "__main__":
    main()

