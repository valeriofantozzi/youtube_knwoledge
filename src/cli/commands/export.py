"""
Export Command - Export Collections to Multiple Formats

Exports documents and embeddings from ChromaDB to JSON, CSV, or other formats.
Supports filtering, chunking for large datasets, and optional embedding inclusion.

Usage:
    knowbase export --output documents.json
    knowbase export --output data.csv --model BAAI/bge-large-en-v1.5
    knowbase export --output large.json --batch-size 100 --format json
    knowbase export --output with-embeddings.json --include-embeddings
"""

import sys
import json
import csv
from typing import Optional, Dict, List, Any, Iterator
from pathlib import Path
import click
from pydantic import BaseModel, Field, ValidationError

from src.utils.config import Config
from src.vector_store.chroma_manager import ChromaDBManager
from src.cli.utils.output import console, print_error, print_success


class ExportCommandInput(BaseModel):
    """Validation model for export command inputs."""
    output_file: Path = Field(..., description="Output file path")
    model: str = Field(default="BAAI/bge-large-en-v1.5", description="Embedding model")
    output_format: str = Field(default="json", pattern="^(json|csv)$")
    include_embeddings: bool = Field(default=False, description="Include embeddings in export")
    batch_size: int = Field(default=100, ge=10, le=1000, description="Batch size for export")


@click.command()
@click.option(
    "-o",
    "--output",
    type=click.Path(),
    required=True,
    help="Output file path",
    metavar="PATH",
)
@click.option(
    "-m",
    "--model",
    default="BAAI/bge-large-en-v1.5",
    help="Embedding model to export",
    metavar="TEXT",
)
@click.option(
    "-f",
    "--format",
    type=click.Choice(["json", "csv"], case_sensitive=False),
    default="json",
    help="Output format",
    metavar="TEXT",
)
@click.option(
    "--include-embeddings",
    is_flag=True,
    help="Include embeddings in export (JSON only, large file size)",
)
@click.option(
    "--batch-size",
    type=int,
    default=100,
    help="Batch size for streaming export",
    metavar="INT",
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
def export(
    output: str,
    model: str,
    format: str,
    include_embeddings: bool,
    batch_size: int,
    config: Optional[str],
    verbose: bool,
):
    """
    Export documents from ChromaDB to JSON or CSV format.

    Supports large dataset exports via streaming/batching to avoid memory issues.
    Optional embedding inclusion for JSON exports (warning: large file sizes).

    Examples:
        knowbase export --output data.json
        knowbase export --output data.csv --format csv
        knowbase export --output all.json --include-embeddings
        knowbase export --output batch.json --batch-size 50
    """
    try:
        # Validate inputs
        try:
            export_input = ExportCommandInput(
                output_file=Path(output),
                model=model,
                output_format=format.lower(),
                include_embeddings=include_embeddings,
                batch_size=batch_size,
            )
        except ValidationError as e:
            print_error(f"Invalid input: {e.errors()[0]['msg']}")
            sys.exit(1)

        # Check if output file already exists
        if export_input.output_file.exists() and not click.confirm(
            f"File {export_input.output_file} already exists. Overwrite?"
        ):
            print_error("Export cancelled")
            sys.exit(1)

        # Load configuration
        config_obj = Config()
        if verbose:
            console.print("[dim]Configuration loaded successfully[/dim]")

        # Initialize ChromaDB
        if verbose:
            console.print("[dim]Connecting to ChromaDB...[/dim]")

        chroma_manager = ChromaDBManager(db_path=Path(config_obj.VECTOR_DB_PATH))
        collection = chroma_manager.get_or_create_collection(
            name="subtitle_embeddings_bge_large"
        )

        # Get data from collection
        if verbose:
            console.print("[dim]Retrieving documents from ChromaDB...[/dim]")

        all_data = collection.get(
            include=["embeddings", "metadatas", "documents"] if include_embeddings else ["metadatas", "documents"]
        )

        total_docs = len(all_data["ids"])
        if total_docs == 0:
            print_error("No documents found in database")
            sys.exit(1)

        if verbose:
            console.print(f"[dim]Found {total_docs} documents to export[/dim]")

        # Export based on format
        if export_input.output_format == "json":
            _export_json(
                all_data,
                export_input.output_file,
                include_embeddings,
                batch_size,
                verbose,
                total_docs,
            )
        else:
            _export_csv(
                all_data,
                export_input.output_file,
                batch_size,
                verbose,
                total_docs,
            )

        print_success(f"Export completed: {export_input.output_file}")

        if verbose:
            file_size = export_input.output_file.stat().st_size / (1024 * 1024)
            console.print(f"[dim]File size: {file_size:.2f} MB[/dim]")

    except KeyboardInterrupt:
        print_error("Export interrupted by user")
        sys.exit(130)
    except Exception as e:
        print_error(f"Error during export: {str(e)}")
        if verbose:
            console.print_exception()
        sys.exit(1)


def _export_json(
    data: Dict[str, Any],
    output_file: Path,
    include_embeddings: bool,
    batch_size: int,
    verbose: bool,
    total_docs: int,
) -> None:
    """Export data to JSON format with streaming support."""
    output_file.parent.mkdir(parents=True, exist_ok=True)

    with open(output_file, "w") as f:
        f.write("{\n  \"documents\": [\n")

        for idx, (doc_id, content, metadata, embedding) in enumerate(
            _batch_documents(data, batch_size, include_embeddings)
        ):
            doc = {
                "id": doc_id,
                "content": content,
                "metadata": metadata or {},
            }

            if include_embeddings and embedding is not None:
                doc["embedding"] = embedding

            # Write JSON with proper formatting
            json_str = json.dumps(doc, indent=2)
            lines = json_str.split("\n")
            f.write("    ")
            f.write(f"\n    ".join(lines))

            if idx < total_docs - 1:
                f.write(",\n")
            else:
                f.write("\n")

            if verbose and (idx + 1) % batch_size == 0:
                console.print(f"[dim]Exported {idx + 1}/{total_docs} documents[/dim]")

        f.write("  ]\n}")


def _export_csv(
    data: Dict[str, Any],
    output_file: Path,
    batch_size: int,
    verbose: bool,
    total_docs: int,
) -> None:
    """Export data to CSV format with streaming support."""
    output_file.parent.mkdir(parents=True, exist_ok=True)

    with open(output_file, "w", newline="") as f:
        fieldnames = ["id", "content", "metadata"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for idx, (doc_id, content, metadata, _) in enumerate(
            _batch_documents(data, batch_size, include_embeddings=False)
        ):
            # Flatten metadata to string
            metadata_str = json.dumps(metadata or {}) if metadata else ""

            writer.writerow({
                "id": doc_id,
                "content": content,
                "metadata": metadata_str,
            })

            if verbose and (idx + 1) % batch_size == 0:
                console.print(f"[dim]Exported {idx + 1}/{total_docs} documents[/dim]")


def _batch_documents(
    data: Dict[str, Any],
    batch_size: int,
    include_embeddings: bool = False,
) -> Iterator[tuple]:
    """Yield documents in batches."""
    ids = data["ids"]
    documents = data.get("documents", [])
    metadatas = data.get("metadatas", [])
    embeddings = data.get("embeddings", []) if include_embeddings else None

    for i in range(0, len(ids), batch_size):
        batch_end = min(i + batch_size, len(ids))
        for j in range(i, batch_end):
            embedding = embeddings[j] if embeddings and j < len(embeddings) else None
            yield (
                ids[j],
                documents[j] if j < len(documents) else "",
                metadatas[j] if j < len(metadatas) else None,
                embedding,
            )


if __name__ == "__main__":
    export()
