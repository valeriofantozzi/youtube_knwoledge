"""
Rich console utilities and helper functions for beautiful terminal output.

This module provides a shared Rich console instance and convenience functions
for consistent styling across all CLI commands.
"""

from typing import Any, Optional

from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.syntax import Syntax
from rich.align import Align

# Global console instance
console = Console()


def print_success(message: str) -> None:
    """Print a success message with green checkmark."""
    console.print(f"[green]✓[/green] {message}")


def print_error(message: str, exit_code: int = 1) -> None:
    """Print an error message with red X and optionally exit."""
    console.print(f"[red]✗[/red] {message}")


def print_warning(message: str) -> None:
    """Print a warning message with yellow exclamation."""
    console.print(f"[yellow]⚠[/yellow] {message}")


def print_info(message: str) -> None:
    """Print an info message with blue info icon."""
    console.print(f"[blue]ℹ[/blue] {message}")


def print_table(
    rows: list[list[str]],
    headers: list[str],
    title: Optional[str] = None,
    caption: Optional[str] = None,
) -> None:
    """
    Print a formatted table.

    Args:
        rows: List of rows, each row is a list of strings
        headers: List of column headers
        title: Optional table title
        caption: Optional table caption
    """
    table = Table(title=title, caption=caption)

    for header in headers:
        table.add_column(header, style="cyan")

    for row in rows:
        table.add_row(*row)

    console.print(table)


def print_panel(
    content: str,
    title: Optional[str] = None,
    border_style: str = "blue",
    expand: bool = False,
) -> None:
    """
    Print content in a bordered panel.

    Args:
        content: Panel content
        title: Optional panel title
        border_style: Style for the border (color name)
        expand: Whether to expand panel to console width
    """
    panel = Panel(
        content,
        title=title,
        border_style=border_style,
        expand=expand,
    )
    console.print(panel)


def print_code(
    code: str,
    language: str = "python",
    title: Optional[str] = None,
) -> None:
    """
    Print syntax-highlighted code.

    Args:
        code: Code to display
        language: Programming language for syntax highlighting
        title: Optional code block title
    """
    syntax = Syntax(code, language, theme="monokai", line_numbers=True)
    panel = Panel(syntax, title=title)
    console.print(panel)


def print_dict(data: dict[str, Any], title: Optional[str] = None) -> None:
    """
    Print a dictionary in a formatted way.

    Args:
        data: Dictionary to print
        title: Optional title
    """
    lines = [f"[cyan]{k}[/cyan]: {v}" for k, v in data.items()]
    content = "\n".join(lines)
    print_panel(content, title=title, border_style="blue")


def print_centered(message: str) -> None:
    """Print a message centered on the console."""
    console.print(Align.center(message))


def print_json(data: dict[str, Any]) -> None:
    """Pretty-print JSON data with syntax highlighting."""
    import json

    json_str = json.dumps(data, indent=2)
    print_code(json_str, language="json", title="JSON Output")


__all__ = [
    "console",
    "print_success",
    "print_error",
    "print_warning",
    "print_info",
    "print_table",
    "print_panel",
    "print_code",
    "print_dict",
    "print_centered",
    "print_json",
]
