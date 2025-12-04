"""
KnowBase CLI - Command-line interface for the knowledge base system.

This module provides the main CLI entry point using Click framework.
All commands are accessible via the 'knowbase' command.

Examples:
    $ knowbase --help
    $ knowbase load --input ./documents
    $ knowbase search --query "how to grow orchids"
    $ knowbase ask "What are the secrets to orchid care?"
    $ knowbase info
"""

import sys
from pathlib import Path
from typing import Optional

import click
from rich.console import Console

from src.utils.config_manager import ConfigManager
from src.cli.utils.output import (
    console,
    print_success,
    print_error,
    print_warning,
)
from src.cli.commands.load import load
from src.cli.commands.search import search
from src.cli.commands.info import info
from src.cli.commands.ask import ask
from src.cli.commands.cluster import cluster
from src.cli.commands.export import export
from src.cli.commands.reindex import reindex

__version__ = "0.1.0"


class CLIContext:
    """Context object for CLI commands."""

    def __init__(
        self,
        verbose: bool = False,
        config_file: Optional[Path] = None,
        output_format: str = "text",
    ):
        """
        Initialize CLI context.

        Args:
            verbose: Enable verbose output
            config_file: Optional configuration file path
            output_format: Default output format (text, json, csv, table)
        """
        self.verbose = verbose
        self.config_file = config_file
        self.output_format = output_format
        self.config_manager: Optional[ConfigManager] = None

        # Load configuration if provided
        if config_file:
            try:
                self.config_manager = ConfigManager(config_file=config_file)
                if verbose:
                    print_success(f"Loaded configuration from {config_file}")
            except Exception as e:
                print_error(f"Failed to load configuration: {e}")
                sys.exit(1)


@click.group(invoke_without_command=True)
@click.pass_context
@click.option(
    "--version",
    is_flag=True,
    help="Show version information",
)
@click.option(
    "-v",
    "--verbose",
    is_flag=True,
    help="Enable verbose output",
)
@click.option(
    "-c",
    "--config",
    type=click.Path(exists=True),
    help="Path to configuration file",
)
@click.option(
    "-f",
    "--format",
    type=click.Choice(["text", "json", "csv", "table"]),
    default="text",
    help="Default output format",
)
def cli(
    ctx: click.Context,
    version: bool,
    verbose: bool,
    config: Optional[str],
    format: str,
) -> None:
    """
    KnowBase - Knowledge Base Management System.

    A command-line interface for semantic search, RAG, clustering, and more.

    Examples:
        \b
        knowbase load --input ./documents
        knowbase search --query "machine learning"
        knowbase ask "What is transformers?"
        knowbase info
    """
    # Show version
    if version:
        console.print(f"KnowBase CLI version {__version__}")
        ctx.exit(0)

    # Initialize context
    config_file = Path(config) if config else None
    ctx.obj = CLIContext(
        verbose=verbose,
        config_file=config_file,
        output_format=format,
    )

    # Show help if no command provided
    if ctx.invoked_subcommand is None:
        console.print(ctx.get_help())


# Command registration happens in separate command files
# For now, we define a simple "hello" command to test the CLI works
@cli.command()
@click.pass_obj
def hello(ctx: CLIContext) -> None:
    """Test command to verify CLI is working."""
    print_success(
        f"KnowBase CLI is ready! Version {__version__}"
    )
    if ctx.config_manager:
        print_success(f"Configuration loaded from {ctx.config_file}")


# Register command handlers
cli.add_command(load)
cli.add_command(search)
cli.add_command(info)
cli.add_command(ask)
cli.add_command(cluster)
cli.add_command(export)
cli.add_command(reindex)


def main() -> None:
    """Main entry point for the CLI."""
    try:
        cli()
    except KeyboardInterrupt:
        print_warning("Operation cancelled by user")
        sys.exit(130)
    except Exception as e:
        print_error(f"An error occurred: {e}")
        if cli.obj and cli.obj.verbose:  # type: ignore
            import traceback

            traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
