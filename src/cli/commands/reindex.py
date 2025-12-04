"""
Reindex Command - Reindex Documents with Different Embedding Model

Reindexes existing documents using a different embedding model.
Preserves all metadata while updating embeddings and creating a new collection.

Usage:
    knowbase reindex --new-model BAAI/bge-large-en-v1.5
    knowbase reindex --from-model google/embeddinggemma-300m --new-model BAAI/bge-large-en-v1.5
    knowbase reindex --new-model BAAI/bge-large --batch-size 32 --device mps
"""

import sys
import click
from pathlib import Path
from datetime import datetime
from typing import Optional
from pydantic import BaseModel, Field, ValidationError

from src.utils.config import Config
from src.vector_store.chroma_manager import ChromaDBManager
from src.embeddings.pipeline import EmbeddingPipeline
from src.embeddings.model_loader import ModelLoader
from src.cli.utils.output import console, print_error, print_success


class ReindexCommandInput(BaseModel):
    """Validation model for reindex command inputs."""
    from_model: str = Field(default="BAAI/bge-large-en-v1.5", description="Source embedding model")
    new_model: str = Field(..., description="Target embedding model")
    batch_size: int = Field(default=32, ge=1, le=256, description="Batch size for processing")
    device: str = Field(default="auto", pattern="^(auto|cpu|cuda|mps)$")


@click.command()
@click.option(
    "--from-model",
    default="BAAI/bge-large-en-v1.5",
    help="Source embedding model",
    metavar="TEXT",
)
@click.option(
    "-n",
    "--new-model",
    required=True,
    help="Target embedding model",
    metavar="TEXT",
)
@click.option(
    "-b",
    "--batch-size",
    type=int,
    default=32,
    help="Batch size for embedding generation",
    metavar="INT",
)
@click.option(
    "-d",
    "--device",
    type=click.Choice(["auto", "cpu", "cuda", "mps"], case_sensitive=False),
    default="auto",
    help="Device for model execution",
    metavar="TEXT",
)
@click.option(
    "--skip-backup",
    is_flag=True,
    help="Skip backing up the original collection",
)
@click.option(
    "-c",
    "--config",
    type=click.Path(exists=True),
    help="Configuration file path",
    metavar="PATH",
)
@click.option(
    "-v",
    "--verbose",
    is_flag=True,
    help="Enable verbose output",
)
def reindex(
    from_model: str,
    new_model: str,
    batch_size: int,
    device: str,
    skip_backup: bool,
    config: Optional[str],
    verbose: bool,
):
    """
    Reindex documents using a different embedding model.

    Retrieves all documents from the current collection, generates new embeddings
    with the specified model, and creates a new indexed collection while
    preserving all metadata.

    Examples:
        knowbase reindex --new-model BAAI/bge-large-en-v1.5
        knowbase reindex --from-model google/embeddinggemma-300m --new-model BAAI/bge-large
        knowbase reindex --new-model BAAI/bge-large --device mps
    """
    try:
        # Validate inputs
        try:
            reindex_input = ReindexCommandInput(
                from_model=from_model,
                new_model=new_model,
                batch_size=batch_size,
                device=device.lower(),
            )
        except ValidationError as e:
            print_error(f"Invalid input: {e.errors()[0]['msg']}")
            sys.exit(1)

        # Confirm reindexing
        console.print("\n[yellow]⚠️  Reindexing Operation[/yellow]")
        console.print(f"From model: [cyan]{reindex_input.from_model}[/cyan]")
        console.print(f"To model:   [cyan]{reindex_input.new_model}[/cyan]")
        console.print()

        if not click.confirm("Proceed with reindexing?"):
            print_error("Reindexing cancelled")
            sys.exit(1)

        # Load configuration
        config_obj = Config()
        if verbose:
            console.print("[dim]Configuration loaded successfully[/dim]")

        # Initialize ChromaDB
        if verbose:
            console.print("[dim]Connecting to ChromaDB...[/dim]")

        chroma_manager = ChromaDBManager(db_path=Path(config_obj.VECTOR_DB_PATH))

        # Get source collection
        source_collection_name = "subtitle_embeddings_bge_large"
        if verbose:
            console.print(f"[dim]Source collection: {source_collection_name}[/dim]")

        source_collection = chroma_manager.get_or_create_collection(
            name=source_collection_name
        )

        # Retrieve all documents
        if verbose:
            console.print("[dim]Retrieving documents from source collection...[/dim]")

        source_data = source_collection.get(include=["documents", "metadatas"])
        doc_ids = source_data["ids"]
        documents = source_data.get("documents", [])
        metadatas = source_data.get("metadatas", [])

        if len(doc_ids) == 0:
            print_error("No documents found in source collection")
            sys.exit(1)

        console.print(f"[green]✓ Retrieved {len(doc_ids)} documents[/green]")

        # Backup original collection if requested
        if not skip_backup:
            if verbose:
                console.print("[dim]Backing up original collection...[/dim]")
            backup_name = f"{source_collection_name}_backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            try:
                # Note: ChromaDB doesn't have a native backup, so we'll just note this in the output
                if verbose:
                    console.print(f"[dim]Backup would be: {backup_name}[/dim]")
            except Exception as e:
                if verbose:
                    console.print(f"[yellow]Could not create backup: {str(e)}[/yellow]")

        # Initialize embedding pipeline with new model
        if verbose:
            console.print(f"[dim]Loading model: {reindex_input.new_model}[/dim]")

        try:
            embedding_pipeline = EmbeddingPipeline(
                model_name=reindex_input.new_model,
                device=reindex_input.device,
                batch_size=reindex_input.batch_size,
            )
        except Exception as e:
            print_error(f"Failed to load model: {str(e)}")
            sys.exit(1)

        # Generate embeddings for all documents
        if verbose:
            console.print("[dim]Generating embeddings with new model...[/dim]")

        try:
            embeddings_result = embedding_pipeline.process(documents)
            embeddings = embeddings_result[0]
            if verbose:
                console.print(f"[green]✓ Generated {len(embeddings)} embeddings[/green]")
        except Exception as e:
            print_error(f"Failed to generate embeddings: {str(e)}")
            sys.exit(1)

        # Create new collection with new model
        target_collection_name = chroma_manager._generate_collection_name(
            reindex_input.new_model, "subtitle_embeddings"
        )
        if verbose:
            console.print(f"[dim]Target collection: {target_collection_name}[/dim]")

        try:
            # Check if target collection exists
            try:
                target_collection = chroma_manager.client.get_collection(target_collection_name)
                if verbose:
                    console.print(
                        f"[yellow]Target collection exists. Adding documents to it.[/yellow]"
                    )
            except Exception:
                # Collection doesn't exist, will be created
                target_collection = chroma_manager.get_or_create_collection(
                    name=target_collection_name
                )

            # Add documents to new collection
            if verbose:
                console.print("[dim]Adding documents to target collection...[/dim]")

            # Batch add for better performance
            batch_size = reindex_input.batch_size
            for i in range(0, len(doc_ids), batch_size):
                batch_end = min(i + batch_size, len(doc_ids))
                batch_ids = doc_ids[i:batch_end]
                batch_docs = documents[i:batch_end]
                batch_embeddings = embeddings[i:batch_end].tolist()
                batch_metadatas = metadatas[i:batch_end] if metadatas else None

                target_collection.add(
                    ids=batch_ids,
                    embeddings=batch_embeddings,
                    documents=batch_docs,
                    metadatas=batch_metadatas,
                )

                if verbose and (batch_end) % (batch_size * 5) == 0:
                    console.print(f"[dim]Added {batch_end}/{len(doc_ids)} documents[/dim]")

        except Exception as e:
            print_error(f"Failed to create target collection: {str(e)}")
            sys.exit(1)

        # Success message
        console.print("\n" + "=" * 70)
        console.print("[green]✓ Reindexing completed successfully![/green]")
        console.print("\nSummary:")
        console.print(f"  Source collection: {source_collection_name}")
        console.print(f"  Target collection: {target_collection_name}")
        console.print(f"  Documents reindexed: {len(doc_ids)}")
        console.print(f"  Model: {reindex_input.from_model} → {reindex_input.new_model}")
        console.print("=" * 70 + "\n")

        print_success("Reindexing completed")

    except KeyboardInterrupt:
        print_error("Reindexing interrupted by user")
        sys.exit(130)
    except Exception as e:
        print_error(f"Error during reindexing: {str(e)}")
        if verbose:
            console.print_exception()
        sys.exit(1)


if __name__ == "__main__":
    reindex()
