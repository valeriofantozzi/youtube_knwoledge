"""
CLI command: knowbase load

Loads documents, runs preprocessing, generates embeddings, and indexes into ChromaDB.

Usage:
    knowbase load --input ./documents
    knowbase load --input ./subtitles --model google/embeddinggemma-300m --device cuda
    knowbase load --input ./data --batch-size 64 --chunk-size 256
"""

import sys
import time
from pathlib import Path
from typing import Optional

import click

from src.preprocessing.pipeline import PreprocessingPipeline
from src.embeddings.pipeline import EmbeddingPipeline
from src.vector_store.pipeline import VectorStorePipeline
from src.utils.config_manager import ConfigManager, get_preset_config
from src.cli.utils.output import (
    console,
    print_success,
    print_error,
    print_warning,
    print_panel,
    print_dict,
)
from src.cli.utils.progress import get_progress_reporter
from src.cli.utils.validators import LoadCommandInput


@click.command()
@click.option(
    "-i",
    "--input",
    type=click.Path(exists=True),
    required=True,
    help="Input file or directory path",
)
@click.option(
    "-m",
    "--model",
    default="BAAI/bge-large-en-v1.5",
    help="Embedding model name",
)
@click.option(
    "-d",
    "--device",
    type=click.Choice(["auto", "cpu", "cuda", "mps"]),
    default="auto",
    help="Device to use for embeddings",
)
@click.option(
    "-b",
    "--batch-size",
    type=int,
    default=32,
    help="Batch size for embedding generation",
)
@click.option(
    "--chunk-size",
    type=int,
    default=512,
    help="Text chunk size for preprocessing",
)
@click.option(
    "--chunk-overlap",
    type=int,
    default=50,
    help="Overlap between chunks",
)
@click.option(
    "--skip-duplicates",
    is_flag=True,
    default=True,
    help="Skip duplicate chunks",
)
@click.option(
    "-c",
    "--config",
    type=click.Path(exists=True),
    help="Configuration file (overrides CLI options)",
)
@click.option(
    "-v",
    "--verbose",
    is_flag=True,
    help="Enable verbose output",
)
@click.pass_context
def load(
    ctx: click.Context,
    input: str,
    model: str,
    device: str,
    batch_size: int,
    chunk_size: int,
    chunk_overlap: int,
    skip_duplicates: bool,
    config: Optional[str],
    verbose: bool,
) -> None:
    """
    Load documents, preprocess, generate embeddings, and index.

    This command:
    1. Parses document files (SRT, TXT, PDF, etc.)
    2. Preprocesses: chunking, cleaning, normalization
    3. Generates embeddings using selected model
    4. Indexes into ChromaDB vector database

    Examples:
        \b
        knowbase load --input ./subtitles
        knowbase load --input ./docs --model google/embeddinggemma-300m
        knowbase load --input ./data --batch-size 64 --skip-duplicates
    """
    start_time = time.time()

    try:
        # Validate input
        input_path = Path(input)
        if not input_path.exists():
            print_error(f"Input path does not exist: {input_path}")
            sys.exit(1)

        # Load or create configuration
        if config:
            config_manager = ConfigManager(config_file=Path(config))
            complete_config = config_manager.config
            if verbose:
                print_success(f"Loaded configuration from {config}")
        else:
            # Create config from CLI options
            complete_config = get_preset_config("full_pipeline")
            
            # Set active database path
            try:
                from src.utils.db_manager import get_db_manager
                db_manager = get_db_manager()
                complete_config.vector_store.db_path = str(db_manager.get_db_path())
            except Exception as e:
                if verbose:
                    print_warning(f"Could not resolve active database path: {e}")

            # Update with CLI options
            complete_config.embedding.model_name = model
            complete_config.embedding.device = device  # type: ignore
            complete_config.embedding.batch_size = batch_size
            complete_config.preprocessing.chunk_size = chunk_size
            complete_config.preprocessing.chunk_overlap = chunk_overlap

        # Get configurations
        prep_cfg = complete_config.preprocessing
        emb_cfg = complete_config.embedding
        vs_cfg = complete_config.vector_store
        pipe_cfg = complete_config.pipeline

        if verbose:
            console.print("\n[cyan]Configuration:[/cyan]")
            print_dict(
                {
                    "Model": emb_cfg.model_name,
                    "Device": emb_cfg.device,
                    "Batch Size": str(emb_cfg.batch_size),
                    "Chunk Size": str(prep_cfg.chunk_size),
                    "Chunk Overlap": str(prep_cfg.chunk_overlap),
                }
            )

        # Initialize pipelines
        console.print("\n[bold blue]🚀 Starting document load pipeline...[/bold blue]\n")

        preprocessing_pipeline = PreprocessingPipeline()
        embedding_pipeline = EmbeddingPipeline(
            model_name=emb_cfg.model_name,
            device=emb_cfg.device,
            batch_size=emb_cfg.batch_size,
            enable_optimizations=True,
        )
        vector_store_pipeline = VectorStorePipeline(
            db_path=str(vs_cfg.db_path) if hasattr(vs_cfg, 'db_path') else None,
            model_name=emb_cfg.model_name
        )

        # Get progress reporter
        progress = get_progress_reporter("rich")

        # Phase 1: Find files
        console.print("[bold cyan]📂 Phase 1: Discovering files[/bold cyan]")
        if input_path.is_dir():
            # Support common document extensions
            extensions = {".srt", ".txt", ".md", ".pdf", ".docx"}
            files = [
                f for f in input_path.rglob("*") if f.suffix.lower() in extensions
            ]
        else:
            files = [input_path]

        if not files:
            print_warning(f"No supported document files found in {input_path}")
            sys.exit(0)

        console.print(f"[green]✓ Found {len(files)} file(s)[/green]\n")

        # Phase 2: Preprocessing
        console.print("[bold cyan]📝 Phase 2: Preprocessing documents[/bold cyan]")
        progress.start("Processing files", total=len(files))

        processed_documents = []
        failed_files = []

        for i, file_path in enumerate(files):
            try:
                processed_doc = preprocessing_pipeline.process_file(file_path)
                if processed_doc and processed_doc.chunks:
                    processed_documents.append(processed_doc)
                    progress.update(1, description=f"Processed {file_path.name}")
                else:
                    failed_files.append((file_path.name, "No chunks generated"))
                    progress.update(1)
            except Exception as e:
                failed_files.append((file_path.name, str(e)))
                progress.update(1)

        progress.finish(f"Preprocessed {len(processed_documents)} documents")
        console.print()

        if failed_files:
            print_warning(f"Failed to process {len(failed_files)} file(s):")
            for fname, error in failed_files[:5]:  # Show first 5 errors
                console.print(f"  [yellow]⚠[/yellow] {fname}: {error}")
            if len(failed_files) > 5:
                console.print(f"  ... and {len(failed_files) - 5} more")
            console.print()

        if not processed_documents:
            print_error("No documents were successfully processed")
            sys.exit(1)

        # Phase 3: Embedding
        console.print("[bold cyan]🔗 Phase 3: Generating embeddings[/bold cyan]")
        progress.start("Generating embeddings", total=len(processed_documents))

        embeddings_list = []
        total_chunks = 0

        for i, processed_doc in enumerate(processed_documents):
            try:
                result = embedding_pipeline.generate_embeddings(
                    processed_doc, show_progress=False
                )
                # Handle both (embeddings, metadata) and (embeddings, metadata, model_name) returns
                if len(result) == 3:
                    embeddings, metadata, _ = result
                else:
                    embeddings, metadata = result
                embeddings_list.append(embeddings)
                total_chunks += len(embeddings)
                progress.update(
                    1, description=f"Embedded {processed_doc.metadata.filename}"
                )
            except Exception as e:
                print_error(f"Failed to generate embeddings: {e}")
                sys.exit(1)

        progress.finish(f"Generated {total_chunks} embeddings")
        console.print()

        # Phase 4: Indexing
        console.print("[bold cyan]📇 Phase 4: Indexing into vector store[/bold cyan]")
        progress.start("Indexing documents", total=len(processed_documents))

        indexed_count = 0

        for i, (processed_doc, embeddings) in enumerate(
            zip(processed_documents, embeddings_list)
        ):
            try:
                count = vector_store_pipeline.index_processed_document(
                    processed_doc,
                    embeddings,
                    skip_duplicates=skip_duplicates,
                    show_progress=False,
                    model_name=emb_cfg.model_name,
                )
                indexed_count += count
                progress.update(
                    1, description=f"Indexed {processed_doc.metadata.filename}"
                )
            except Exception as e:
                print_error(f"Failed to index document: {e}")
                sys.exit(1)

        progress.finish(f"Indexed {indexed_count} chunks")
        console.print()

        # Summary
        elapsed = time.time() - start_time
        stats = vector_store_pipeline.get_index_statistics()

        print_panel(
            f"""
[green]✓ Load completed successfully![/green]

[cyan]Summary:[/cyan]
  Documents processed: {len(processed_documents)}
  Total chunks: {total_chunks}
  Chunks indexed: {indexed_count}
  Elapsed time: {elapsed:.1f}s
  Average speed: {indexed_count / elapsed:.0f} chunks/sec

[cyan]Vector store:[/cyan]
  Model: {emb_cfg.model_name}
  Collection: {vs_cfg.collection_name}
  Total documents: {stats.get('total_documents', 'N/A')}
  Total chunks: {stats.get('total_chunks', 'N/A')}
            """,
            title="Load Summary",
            border_style="green",
        )

    except Exception as e:
        print_error(f"Load failed: {e}")
        if verbose:
            import traceback

            traceback.print_exc()
        sys.exit(1)
