#!/usr/bin/env python3
"""
Example: Using ConfigManager with CLI Commands

This example shows how to build CLI commands that use the ConfigManager
for modular, flexible pipeline execution.

NOTE: This is a pseudocode example. Actual API signatures may differ.
Refer to the actual pipeline classes in src/ for correct usage.
"""

from pathlib import Path
from typing import Optional
import click
from src.utils.config_manager import ConfigManager, get_preset_config


# ============================================================================
# EXAMPLE 1: Load Command with Configuration
# ============================================================================


@click.command()
@click.option(
    "--config",
    type=click.Path(exists=True),
    help="Path to config YAML file (uses full_pipeline.yaml if not specified)",
)
@click.option("--input", required=True, type=click.Path(exists=True), help="Input directory")
@click.option("--model", help="Override embedding model")
@click.option("--batch-size", type=int, help="Override batch size")
@click.option("--device", type=click.Choice(["auto", "cpu", "cuda", "mps"]), help="Override device")
def load_command(
    config: Optional[str], input: str, model: Optional[str], batch_size: Optional[int], device: Optional[str]
) -> None:
    """
    Load documents and generate embeddings.

    Example:
        # Full pipeline
        python cli_examples.py load_command --input ./subtitles

        # With custom config
        python cli_examples.py load_command --config config/my_config.yaml --input ./docs

        # With overrides
        python cli_examples.py load_command --input ./docs --batch-size 64 --device cuda
    """

    # Load configuration
    if config:
        mgr = ConfigManager(config_file=Path(config))
    else:
        # Use full_pipeline preset
        mgr = ConfigManager()
        mgr.config = get_preset_config("full_pipeline")

    # Apply CLI overrides
    overrides = {}
    if model:
        overrides.setdefault("embedding", {})["model_name"] = model
    if batch_size:
        overrides.setdefault("embedding", {})["batch_size"] = batch_size
    if device:
        overrides.setdefault("embedding", {})["device"] = device

    if overrides:
        mgr.merge_with_dict(overrides)

    click.echo("✓ Configuration loaded")
    click.echo(f"  Model: {mgr.config.embedding.model_name}")
    click.echo(f"  Batch size: {mgr.config.embedding.batch_size}")
    click.echo(f"  Device: {mgr.config.embedding.device}")

    # Check which pipelines should run
    pipe_cfg = mgr.get_pipeline_config()

    # Run preprocessing if enabled
    if pipe_cfg.run_preprocessing:
        click.echo("\n→ Running preprocessing...")
        # from src.preprocessing.pipeline import PreprocessingPipeline
        # prep = PreprocessingPipeline()
        # processed_docs = prep.process_multiple_files([Path(input)])
        click.echo(f"  ✓ Processed documents")

    # Run embedding if enabled
    if pipe_cfg.run_embedding:
        click.echo("\n→ Generating embeddings...")
        # from src.embeddings.pipeline import EmbeddingPipeline
        # emb = EmbeddingPipeline()
        # embeddings = emb.generate_embeddings(processed_docs)
        click.echo(f"  ✓ Generated embeddings")

    # Run indexing if enabled
    if pipe_cfg.run_indexing:
        click.echo("\n→ Indexing into vector store...")
        # from src.vector_store.pipeline import VectorStorePipeline
        # vs = VectorStorePipeline()
        # vs.index(processed_docs, embeddings)
        click.echo(f"  ✓ Indexed documents")

    click.echo("\n✓ Load command completed successfully!")


# ============================================================================
# EXAMPLE 2: Search Command with Configuration
# ============================================================================


@click.command()
@click.option(
    "--config",
    type=click.Path(exists=True),
    help="Path to config YAML file (uses search_only.yaml if not specified)",
)
@click.option("--query", required=True, help="Search query")
@click.option("--top-k", type=int, help="Number of results")
@click.option("--threshold", type=float, help="Minimum similarity threshold")
@click.option("--output-format", type=click.Choice(["text", "json"]), default="text")
def search_command(
    config: Optional[str],
    query: str,
    top_k: Optional[int],
    threshold: Optional[float],
    output_format: str,
) -> None:
    """
    Search indexed documents.

    Example:
        # Basic search
        python cli_examples.py search_command --query "orchid care"

        # With custom config and overrides
        python cli_examples.py search_command \
            --config config/my_config.yaml \
            --query "orchid care" \
            --top-k 10 \
            --threshold 0.5

        # JSON output
        python cli_examples.py search_command \
            --query "orchid care" \
            --output-format json
    """

    # Load configuration
    if config:
        mgr = ConfigManager(config_file=Path(config))
    else:
        # Use search_only preset
        mgr = ConfigManager()
        mgr.config = get_preset_config("search_only")

    # Apply CLI overrides
    overrides = {}
    if top_k:
        overrides.setdefault("retrieval", {})["top_k"] = top_k
    if threshold:
        overrides.setdefault("retrieval", {})["similarity_threshold"] = threshold

    if overrides:
        mgr.merge_with_dict(overrides)

    click.echo(f"✓ Configuration loaded")
    click.echo(f"  Query: {query}")
    click.echo(f"  Top K: {mgr.config.retrieval.top_k}")
    click.echo(f"  Threshold: {mgr.config.retrieval.similarity_threshold}")

    # Run retrieval
    click.echo("\n→ Searching...")
    # from src.retrieval.pipeline import RetrievalPipeline
    # retrieval = RetrievalPipeline()
    # results = retrieval.search(query)

    # Output results
    if output_format == "json":
        import json
        output = {
            "query": query,
            "results": [
                # {"score": result.score, "text": result.text, ...}
            ],
        }
        click.echo(json.dumps(output, indent=2))
    else:
        click.echo(f"\n✓ Search completed")


# ============================================================================
# EXAMPLE 3: RAG Command with Configuration
# ============================================================================


@click.command()
@click.option(
    "--config",
    type=click.Path(exists=True),
    help="Path to config YAML file (uses rag_only.yaml if not specified)",
)
@click.option("--question", required=True, help="Question to ask")
@click.option("--llm-provider", help="Override LLM provider")
@click.option("--llm-model", help="Override LLM model")
@click.option("--llm-temperature", type=float, help="Override LLM temperature")
@click.option("--show-thinking", is_flag=True, help="Show thinking/reasoning")
def ask_command(
    config: Optional[str],
    question: str,
    llm_provider: Optional[str],
    llm_model: Optional[str],
    llm_temperature: Optional[float],
    show_thinking: bool,
) -> None:
    """
    Ask a question using RAG/LLM.

    Example:
        # Basic RAG
        python cli_examples.py ask_command --question "How to grow orchids?"

        # With custom LLM
        python cli_examples.py ask_command \
            --question "How to grow orchids?" \
            --llm-provider anthropic \
            --llm-model claude-3-opus

        # Show reasoning
        python cli_examples.py ask_command \
            --question "How to grow orchids?" \
            --show-thinking
    """

    # Load configuration
    if config:
        mgr = ConfigManager(config_file=Path(config))
    else:
        # Use rag_only preset
        mgr = ConfigManager()
        mgr.config = get_preset_config("rag_only")

    # Apply CLI overrides
    overrides = {}
    if llm_provider:
        overrides.setdefault("ai_search", {})["llm_provider"] = llm_provider
    if llm_model:
        overrides.setdefault("ai_search", {})["llm_model"] = llm_model
    if llm_temperature:
        overrides.setdefault("ai_search", {})["llm_temperature"] = llm_temperature
    if show_thinking:
        overrides.setdefault("ai_search", {})["show_thinking"] = True

    if overrides:
        mgr.merge_with_dict(overrides)

    click.echo("✓ Configuration loaded")
    click.echo(f"  Question: {question}")
    click.echo(f"  LLM Provider: {mgr.config.ai_search.llm_provider}")
    click.echo(f"  LLM Model: {mgr.config.ai_search.llm_model}")
    click.echo(f"  Show thinking: {mgr.config.ai_search.show_thinking}")

    # Run RAG
    click.echo("\n→ Processing with RAG...")
    # from src.ai_search.graph import build_graph
    # graph = build_graph()
    # answer = graph.run(question)

    click.echo(f"\n✓ RAG Processing complete!")


# ============================================================================
# EXAMPLE 4: Flexible Pipeline (All-in-One)
# ============================================================================


def example_flexible_pipeline(
    config_file: Optional[Path] = None,
    input_path: Optional[str] = None,
    query: Optional[str] = None,
    question: Optional[str] = None,
    **kwargs,
) -> None:
    """
    A single flexible command that can do preprocessing, embedding, search, or RAG
    based on configuration and inputs.

    Example:
        # Just load
        example_flexible_pipeline(
            config_file=Path("config/full_pipeline.yaml"),
            input_path="./subtitles"
        )

        # Just search
        example_flexible_pipeline(
            config_file=Path("config/search_only.yaml"),
            query="orchid care"
        )

        # RAG
        example_flexible_pipeline(
            config_file=Path("config/rag_only.yaml"),
            question="How to grow orchids?"
        )
    """

    # Load config
    if config_file:
        mgr = ConfigManager(config_file=config_file)
    else:
        mgr = ConfigManager()

    pipe_cfg = mgr.get_pipeline_config()

    # Processing pipeline
    if pipe_cfg.run_preprocessing and input_path:
        print("→ Preprocessing...")
        # from src.preprocessing.pipeline import PreprocessingPipeline
        # prep = PreprocessingPipeline()
        # docs = prep.process_multiple_files([Path(input_path)])
        print(f"  ✓ Preprocessed documents")

    if pipe_cfg.run_embedding and input_path:
        print("→ Embedding...")
        # from src.embeddings.pipeline import EmbeddingPipeline
        # emb = EmbeddingPipeline()
        # embeddings = emb.generate_embeddings(docs)
        print(f"  ✓ Generated embeddings")

    if pipe_cfg.run_indexing and input_path:
        print("→ Indexing...")
        # from src.vector_store.pipeline import VectorStorePipeline
        # vs = VectorStorePipeline()
        # vs.index(docs, embeddings)
        print(f"  ✓ Indexed documents")

    # Search
    if pipe_cfg.run_retrieval and query:
        print(f"\n→ Searching: {query}")
        # from src.retrieval.pipeline import RetrievalPipeline
        # retrieval = RetrievalPipeline()
        # results = retrieval.search(query)
        print(f"  ✓ Search complete")

    # RAG
    if pipe_cfg.run_ai_search and question:
        print(f"\n→ Answering: {question}")
        # from src.ai_search.graph import build_graph
        # graph = build_graph()
        # answer = graph.run(question)
        print(f"  ✓ Answer generated")


if __name__ == "__main__":
    # For demonstration, show how to use the commands:
    print("""
    This file demonstrates how to build CLI commands using ConfigManager.

    Run individual commands with Click:
        python cli_examples.py load_command --help
        python cli_examples.py search_command --help
        python cli_examples.py ask_command --help

    Or call the flexible pipeline directly:
        from cli_examples import example_flexible_pipeline
        example_flexible_pipeline(
            config_file=Path("config/presets/full_pipeline.yaml"),
            input_path="./subtitles"
        )
    """)
