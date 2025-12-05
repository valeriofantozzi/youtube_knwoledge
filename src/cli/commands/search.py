"""
CLI command: knowbase search

Performs semantic search over indexed documents in the vector database.

Usage:
    knowbase search --query "how to grow orchids"
    knowbase search --query "orchid care" --top-k 10 --output-format json
    knowbase search --query "tips" --model google/embeddinggemma-300m --format csv > results.csv
"""

import sys
import json
from pathlib import Path
from typing import Optional, List, Dict, Any

import click

from src.embeddings.embedder import Embedder
from src.vector_store.chroma_manager import ChromaDBManager
from src.utils.config_manager import ConfigManager, get_preset_config
from src.utils.config import get_config
from src.cli.utils.output import (
    console,
    print_success,
    print_error,
    print_panel,
    print_table,
)
from src.cli.utils.formatters import get_formatter
from src.cli.utils.validators import SearchCommandInput


@click.command()
@click.option(
    "-q",
    "--query",
    required=True,
    help="Search query text",
)
@click.option(
    "-k",
    "--top-k",
    type=int,
    default=5,
    help="Number of results to return",
)
@click.option(
    "-m",
    "--model",
    default="google/embeddinggemma-300m",
    help="Embedding model to use",
)
@click.option(
    "-t",
    "--threshold",
    type=float,
    default=0.0,
    help="Similarity score threshold (0.0-1.0)",
)
@click.option(
    "-f",
    "--format",
    type=click.Choice(["text", "json", "csv", "table"]),
    default="text",
    help="Output format",
)
@click.option(
    "-c",
    "--config",
    type=click.Path(exists=True),
    help="Configuration file",
)
@click.option(
    "-v",
    "--verbose",
    is_flag=True,
    help="Enable verbose output",
)
@click.pass_context
def search(
    ctx: click.Context,
    query: str,
    top_k: int,
    model: str,
    threshold: float,
    format: str,
    config: Optional[str],
    verbose: bool,
) -> None:
    """
    Perform semantic search over indexed documents.

    Searches the vector database using semantic similarity
    and returns the most relevant documents.

    Examples:
        \b
        knowbase search --query "machine learning"
        knowbase search --query "orchid care" --top-k 10
        knowbase search --query "tips" --format json > results.json
        knowbase search --query "test" --threshold 0.7
    """
    try:
        # Validate input
        validator = SearchCommandInput(
            query=query,
            top_k=top_k,
            model=model,
            threshold=threshold,
            output_format=format,
        )

        # Load configuration
        if config:
            config_manager = ConfigManager(config_file=Path(config))
            complete_config = config_manager.config
            if verbose:
                print_success(f"Loaded configuration from {config}")
        else:
            # Use system default config, not Pydantic defaults
            system_config = get_config()
            complete_config = get_preset_config("search_only")
            
            # Get active database path
            try:
                from src.utils.db_manager import get_db_manager
                db_manager = get_db_manager()
                complete_config.vector_store.db_path = str(db_manager.get_db_path())
            except Exception as e:
                if verbose:
                    print_warning(f"Could not resolve active database path: {e}")
                complete_config.vector_store.db_path = system_config.VECTOR_DB_PATH
            
            complete_config.vector_store.collection_name = system_config.COLLECTION_NAME
            
            # Auto-detect model from database if not explicitly specified
            if model == "google/embeddinggemma-300m":  # Default value, user didn't specify
                try:
                    temp_chroma = ChromaDBManager(db_path=complete_config.vector_store.db_path)
                    detected_model = temp_chroma.auto_detect_model()
                    if detected_model:
                        if verbose:
                            print_success(f"Auto-detected embedding model: {detected_model}")
                        complete_config.embedding.model_name = detected_model
                    else:
                        # No model detected, use default
                        complete_config.embedding.model_name = model
                except Exception as e:
                    if verbose:
                        print_warning(f"Could not auto-detect model: {e}, using default")
                    complete_config.embedding.model_name = model
            else:
                # User explicitly specified a model
                complete_config.embedding.model_name = model

        # Get configurations
        emb_cfg = complete_config.embedding
        vs_cfg = complete_config.vector_store

        if verbose:
            console.print(f"\n[cyan]Configuration:[/cyan]")
            console.print(f"  Model: {emb_cfg.model_name}")
            console.print(f"  Database: {vs_cfg.db_path}")
            console.print(f"  Collection: {vs_cfg.collection_name}\n")

        # Initialize embedder and vector store
        console.print("[bold blue]🔍 Performing search...[/bold blue]\n")

        # Initialize embedder with configured model
        from src.embeddings.model_loader import get_model_loader
        model_loader = get_model_loader(
            model_name=emb_cfg.model_name,
            device=emb_cfg.device
        )
        embedder = Embedder(model_loader=model_loader)
        chroma_manager = ChromaDBManager(
            db_path=vs_cfg.db_path,
            collection_name=vs_cfg.collection_name,
        )
        
        # Generate model-specific collection name (same as what load used)
        actual_collection_name = chroma_manager._generate_collection_name(
            model_name=emb_cfg.model_name,
            base_name=vs_cfg.collection_name
        )
        chroma_manager.collection_name = actual_collection_name

        # Get collection
        try:
            collection = chroma_manager.get_or_create_collection()
        except Exception as e:
            print_error(f"Failed to access vector database: {e}")
            sys.exit(1)

        # Check if collection has documents
        collection_size = collection.count()
        if collection_size == 0:
            print_error("Vector database is empty. Please run 'knowbase load' first.")
            sys.exit(1)

        if verbose:
            console.print(f"[cyan]Database has {collection_size} documents[/cyan]\n")

        # Generate query embedding
        try:
            query_embedding = embedder.encode([query], is_query=True)[0]
        except Exception as e:
            print_error(f"Failed to generate query embedding: {e}")
            sys.exit(1)

        # Perform search
        try:
            results = collection.query(
                query_embeddings=[query_embedding.tolist()],
                n_results=top_k,
                include=["documents", "metadatas", "distances"],
            )
        except Exception as e:
            print_error(f"Search failed: {e}")
            sys.exit(1)

        # Process results
        if not results.get("ids") or not results["ids"][0]:
            console.print("[yellow]No results found[/yellow]")
            sys.exit(0)

        # Format results
        result_list = []
        for i, (doc_id, doc_text, metadata, distance) in enumerate(
            zip(
                results["ids"][0],
                results["documents"][0],
                results["metadatas"][0],
                results["distances"][0],
            ),
            1,
        ):
            # ChromaDB returns distances, convert to similarity scores
            # For cosine distance: similarity = 1 - distance
            similarity = 1 - distance if distance is not None else 0

            # Filter by threshold
            if similarity < threshold:
                continue

            result_item = {
                "rank": i,
                "score": round(similarity, 4),
                "document": doc_text[:100] + "..." if len(doc_text) > 100 else doc_text,
                "source": metadata.get("source_id", "unknown") if metadata else "unknown",
                "chunk_id": doc_id,
            }
            result_list.append(result_item)

        if not result_list:
            console.print(
                f"[yellow]No results found above threshold {threshold}[/yellow]"
            )
            sys.exit(0)

        # Display results
        formatter = get_formatter(format)

        if format == "text":
            # Pretty print text format
            print_panel(
                f"Query: [bold cyan]{query}[/bold cyan]\n"
                f"Found: [green]{len(result_list)} result(s)[/green]",
                border_style="blue",
            )

            for result in result_list:
                console.print(
                    f"\n[bold yellow]#{result['rank']}[/bold yellow] "
                    f"[green]Score: {result['score']:.4f}[/green]"
                )
                console.print(f"[cyan]Source:[/cyan] {result['source']}")
                console.print(f"[cyan]Document:[/cyan] {result['document']}")

            console.print()

        elif format == "table":
            # Table format
            table_rows = [
                [
                    str(r["rank"]),
                    f"{r['score']:.4f}",
                    r["document"][:50] + "..." if len(r["document"]) > 50 else r["document"],
                    r["source"],
                ]
                for r in result_list
            ]
            print_table(
                table_rows,
                ["Rank", "Score", "Document", "Source"],
                title=f"Search Results for: {query}",
            )

        else:
            # JSON or CSV format
            output = formatter.format(result_list)
            click.echo(output)

        # Summary
        if verbose:
            console.print(
                f"\n[cyan]Returned {len(result_list)} of {collection_size} documents[/cyan]"
            )

    except Exception as e:
        print_error(f"Search failed: {e}")
        if verbose:
            import traceback

            traceback.print_exc()
        sys.exit(1)
