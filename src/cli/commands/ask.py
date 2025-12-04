"""
Ask Command - Conversational RAG Queries

Provides RAG (Retrieval-Augmented Generation) interface for asking
questions about the knowledge base with thinking display and multi-provider LLM support.

Usage:
    knowbase ask "What are the best practices for orchid care?"
    knowbase ask "How to grow orchids?" --llm-provider openai --show-thinking
    knowbase ask "..." --top-k 10 --temperature 0.5 --format json
"""

import sys
import click
from pathlib import Path
from typing import Optional
from rich.console import Console
from rich.panel import Panel
from rich.markdown import Markdown
from pydantic import BaseModel, Field, ValidationError
import json

from src.utils.config import Config
from src.ai_search.graph import build_graph
from src.ai_search.thinking import ThinkingUpdate, ThinkingStatus
from src.vector_store.chroma_manager import ChromaDBManager
from src.cli.utils.output import console, print_error, print_success


class AskCommandInput(BaseModel):
    """Validation model for ask command inputs."""
    question: str = Field(..., min_length=1, max_length=2000, description="Question to ask")
    model: str = Field(default="BAAI/bge-large-en-v1.5", description="Embedding model")
    top_k: int = Field(default=5, ge=1, le=50, description="Top K results to retrieve")
    llm_provider: str = Field(default="openai", pattern="^(openai|anthropic|groq|azure|ollama)$")
    temperature: float = Field(default=0.7, ge=0.0, le=2.0, description="LLM temperature")
    show_thinking: bool = Field(default=True, description="Display thinking process")
    stream: bool = Field(default=False, description="Stream output in real-time")


@click.command()
@click.argument("question", required=True)
@click.option(
    "-m",
    "--model",
    default="BAAI/bge-large-en-v1.5",
    help="Embedding model to use for retrieval",
    metavar="TEXT",
)
@click.option(
    "-k",
    "--top-k",
    type=int,
    default=5,
    help="Number of documents to retrieve",
    metavar="INT",
)
@click.option(
    "-p",
    "--llm-provider",
    type=click.Choice(["openai", "anthropic", "groq", "azure", "ollama"], case_sensitive=False),
    default="openai",
    help="LLM provider to use for generation",
    metavar="TEXT",
)
@click.option(
    "-t",
    "--temperature",
    type=float,
    default=0.7,
    help="Temperature for LLM generation (0.0-2.0)",
    metavar="FLOAT",
)
@click.option(
    "--show-thinking/--no-thinking",
    default=True,
    help="Display agent thinking process",
)
@click.option(
    "--stream/--no-stream",
    default=False,
    help="Stream output in real-time (experimental)",
)
@click.option(
    "-f",
    "--format",
    type=click.Choice(["text", "json"], case_sensitive=False),
    default="text",
    help="Output format",
    metavar="TEXT",
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
def ask(
    question: str,
    model: str,
    top_k: int,
    llm_provider: str,
    temperature: float,
    show_thinking: bool,
    stream: bool,
    format: str,
    config: Optional[str],
    verbose: bool,
):
    """
    Ask a question about the knowledge base using RAG.

    Searches the vector database for relevant documents and uses an LLM
    to generate a comprehensive answer based on retrieved context.

    Examples:
        knowbase ask "What are best practices for orchid care?"
        knowbase ask "How to grow orchids fast?" --top-k 10
        knowbase ask "..." --llm-provider anthropic --show-thinking
        knowbase ask "..." --format json > answer.json
    """
    try:
        # Validate inputs
        try:
            ask_input = AskCommandInput(
                question=question,
                model=model,
                top_k=top_k,
                llm_provider=llm_provider.lower(),
                temperature=temperature,
                show_thinking=show_thinking,
                stream=stream,
            )
        except ValidationError as e:
            print_error(f"Invalid input: {e.errors()[0]['msg']}")
            sys.exit(1)

        # Load configuration
        config_obj = Config()
        if verbose:
            console.print(
                f"[dim]Configuration loaded successfully[/dim]"
            )

        # Display thinking header if enabled
        if show_thinking:
            console.print("\n[cyan]💭 Thinking Process:[/cyan]\n")

        # Build the RAG graph
        if verbose:
            console.print("[dim]Building RAG graph...[/dim]")

        graph = build_graph()

        # Prepare initial state
        initial_state = {
            "question": ask_input.question,
            "messages": [],
            "thinking_updates": [],
            "retrieved_docs": [],
        }

        # Run the graph
        if verbose:
            console.print("[dim]Running RAG agent...[/dim]")

        try:
            result = graph.invoke(initial_state)  # type: ignore
        except Exception as e:
            if "Could not find matching key" in str(e):
                # Graph returned partial state or missing fields
                result = initial_state
            else:
                raise

        # Extract thinking updates if available
        thinking_updates = result.get("thinking_updates", [])
        answer = result.get("generation", result.get("answer", ""))
        sources = result.get("documents", [])

        # Display thinking process if enabled
        if show_thinking and thinking_updates:
            _display_thinking_process(thinking_updates)

        # Display answer based on format
        if format.lower() == "json":
            _output_json_format(
                ask_input.question, answer, sources, thinking_updates
            )
        else:
            _output_text_format(ask_input.question, answer, sources, verbose)

        if verbose:
            console.print("\n[green]✓ Query completed successfully[/green]")

    except KeyboardInterrupt:
        print_error("Query interrupted by user")
        sys.exit(130)
    except Exception as e:
        print_error(f"Error during query: {str(e)}")
        if verbose:
            console.print_exception()
        sys.exit(1)


def _display_thinking_process(thinking_updates: list) -> None:
    """Display the thinking process from RAG agent."""
    for update in thinking_updates:
        if isinstance(update, dict):
            update_obj = update
        elif isinstance(update, ThinkingUpdate):
            update_obj = update.to_dict()
        else:
            update_obj = update

        status = update_obj.get("status", "processing")
        phase = update_obj.get("phase_title", "Processing")
        details = update_obj.get("details", "")
        progress = update_obj.get("progress", 0.0)

        # Format status with emoji
        status_emoji = {
            "analyzing": "🔍",
            "processing": "⚙️",
            "retrieving": "📚",
            "generating": "✍️",
            "reasoning": "🧠",
            "complete": "✓",
            "error": "❌",
        }.get(status, "•")

        # Display with progress bar if available
        if progress > 0:
            bar_length = 20
            filled = int(bar_length * progress)
            bar = "█" * filled + "░" * (bar_length - filled)
            console.print(
                f"{status_emoji} {phase} [{bar}] {int(progress * 100)}%"
            )
        else:
            console.print(f"{status_emoji} {phase}")

        if details:
            console.print(f"  [dim]{details}[/dim]")


def _output_text_format(
    question: str, answer: str, sources: list, verbose: bool
) -> None:
    """Output answer in text format with sources."""
    console.print("\n" + "=" * 70)
    console.print(f"[bold cyan]Question:[/bold cyan] {question}\n")

    # Display answer
    if answer:
        console.print(f"[bold green]Answer:[/bold green]\n")
        console.print(Markdown(answer))
    else:
        console.print("[yellow]No answer generated[/yellow]")

    # Display sources
    if sources:
        console.print("\n[bold]Sources:[/bold]")
        for i, source in enumerate(sources, 1):
            if isinstance(source, dict):
                source_name = source.get("metadata", {}).get(
                    "filename", source.get("source", "Unknown")
                )
            else:
                source_name = str(source)
            console.print(f"  {i}. {source_name}")
    else:
        if verbose:
            console.print("\n[dim]No sources retrieved[/dim]")

    console.print("=" * 70 + "\n")


def _output_json_format(
    question: str, answer: str, sources: list, thinking_updates: list
) -> None:
    """Output answer in JSON format."""
    output = {
        "question": question,
        "answer": answer,
        "sources": [
            str(s.get("metadata", {}).get("filename", s.get("source", str(s))))
            if isinstance(s, dict)
            else str(s)
            for s in sources
        ],
        "thinking_updates": [
            u.to_dict() if isinstance(u, ThinkingUpdate) else u
            for u in thinking_updates
        ],
    }
    console.print(json.dumps(output, indent=2))


if __name__ == "__main__":
    ask()
