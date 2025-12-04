"""
Output formatters for different output formats (text, JSON, CSV).

This module provides a flexible formatter system to convert command results
into different output formats.
"""

import json
import csv
from abc import ABC, abstractmethod
from io import StringIO
from typing import Any, Optional, Union
from pathlib import Path
from dataclasses import asdict, is_dataclass

from .output import console


class BaseFormatter(ABC):
    """Base class for output formatters."""

    @abstractmethod
    def format(self, data: Any) -> str:
        """Format data into output string."""
        pass

    def write_to_file(self, data: Any, filepath: Path) -> None:
        """Write formatted output to file."""
        output = self.format(data)
        filepath.write_text(output)


class TextFormatter(BaseFormatter):
    """Format output as human-readable text."""

    def format(self, data: Any) -> str:
        """Convert data to formatted text."""
        if isinstance(data, dict):
            lines = []
            for key, value in data.items():
                if isinstance(value, (list, dict)):
                    lines.append(f"{key}:")
                    lines.append(f"  {self._format_value(value)}")
                else:
                    lines.append(f"{key}: {value}")
            return "\n".join(lines)
        elif isinstance(data, list):
            return "\n".join(str(item) for item in data)
        else:
            return str(data)

    def _format_value(self, value: Any, indent: int = 2) -> str:
        """Format nested values with indentation."""
        if isinstance(value, dict):
            lines = [f"{k}: {v}" for k, v in value.items()]
            return "\n".join(" " * indent + line for line in lines)
        elif isinstance(value, list):
            return "\n".join(" " * indent + str(item) for item in value)
        return str(value)


class JSONFormatter(BaseFormatter):
    """Format output as JSON."""

    def __init__(self, indent: int = 2, sort_keys: bool = True):
        """
        Initialize JSON formatter.

        Args:
            indent: JSON indentation level
            sort_keys: Whether to sort dictionary keys
        """
        self.indent = indent
        self.sort_keys = sort_keys

    def format(self, data: Any) -> str:
        """Convert data to JSON string."""
        return json.dumps(
            self._make_serializable(data),
            indent=self.indent,
            sort_keys=self.sort_keys,
        )

    @staticmethod
    def _make_serializable(obj: Any) -> Any:
        """Convert non-serializable objects to serializable form."""
        if is_dataclass(obj) and not isinstance(obj, type):
            return asdict(obj)  # type: ignore
        elif isinstance(obj, Path):
            return str(obj)
        elif isinstance(obj, dict):
            return {k: JSONFormatter._make_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, (list, tuple)):
            return [JSONFormatter._make_serializable(item) for item in obj]
        else:
            return obj


class CSVFormatter(BaseFormatter):
    """Format output as CSV."""

    def format(self, data: Any) -> str:
        """Convert data to CSV string."""
        if not isinstance(data, list):
            raise ValueError("CSV formatter requires list of dictionaries")

        if not data:
            return ""

        output = StringIO()
        if isinstance(data[0], dict):
            fieldnames = data[0].keys()
            writer = csv.DictWriter(output, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(data)
        else:
            # If list of non-dict items, create simple CSV
            writer = csv.writer(output)
            for item in data:
                writer.writerow([item])

        return output.getvalue()


class TableFormatter(BaseFormatter):
    """Format output as a text table using Rich."""

    def format(self, data: Any) -> str:
        """Convert data to table string."""
        from rich.table import Table
        from io import StringIO
        from rich.console import Console

        if not isinstance(data, list):
            raise ValueError("Table formatter requires list of dictionaries")

        if not data:
            return ""

        # Create table
        table = Table(show_header=True, header_style="bold cyan")

        if isinstance(data[0], dict):
            # Add columns
            for key in data[0].keys():
                table.add_column(str(key))

            # Add rows
            for row in data:
                table.add_row(*[str(v) for v in row.values()])
        else:
            # Simple table with single column
            table.add_column("Value")
            for item in data:
                table.add_row(str(item))

        # Capture output
        output = StringIO()
        console_temp = Console(file=output, width=100)
        console_temp.print(table)
        return output.getvalue()


def get_formatter(format_name: str) -> BaseFormatter:
    """
    Get formatter instance by name.

    Args:
        format_name: One of 'text', 'json', 'csv', 'table'

    Returns:
        Formatter instance

    Raises:
        ValueError: If format_name is unknown
    """
    formatters = {
        "text": TextFormatter(),
        "json": JSONFormatter(),
        "csv": CSVFormatter(),
        "table": TableFormatter(),
    }

    if format_name not in formatters:
        raise ValueError(
            f"Unknown format '{format_name}'. "
            f"Available: {', '.join(formatters.keys())}"
        )

    return formatters[format_name]


__all__ = [
    "BaseFormatter",
    "TextFormatter",
    "JSONFormatter",
    "CSVFormatter",
    "TableFormatter",
    "get_formatter",
]
