"""
Tests for CLI framework and utilities.

This test suite validates:
- CLI routing and command discovery
- Output formatters
- Input validators
- Progress reporters
"""

import json
from pathlib import Path
from io import StringIO

import pytest
from click.testing import CliRunner

from src.cli.main import cli, CLIContext
from src.cli.utils.formatters import (
    TextFormatter,
    JSONFormatter,
    CSVFormatter,
    get_formatter,
)
from src.cli.utils.validators import (
    LoadCommandInput,
    SearchCommandInput,
    AskCommandInput,
)
from src.cli.utils.progress import (
    RichProgress,
    SimpleProgress,
    get_progress_reporter,
)


class TestCLIMain:
    """Tests for main CLI functionality."""

    def test_cli_help(self):
        """Test that CLI help works."""
        runner = CliRunner()
        result = runner.invoke(cli, ["--help"])
        assert result.exit_code == 0
        assert "KnowBase" in result.output
        assert "Usage:" in result.output

    def test_cli_version(self):
        """Test version flag."""
        runner = CliRunner()
        result = runner.invoke(cli, ["--version"])
        assert result.exit_code == 0
        assert "0.1.0" in result.output

    def test_cli_hello_command(self):
        """Test hello command."""
        runner = CliRunner()
        result = runner.invoke(cli, ["hello"])
        assert result.exit_code == 0
        assert "ready" in result.output.lower()

    def test_cli_info_command(self):
        """Test info command."""
        runner = CliRunner()
        result = runner.invoke(cli, ["info"])
        assert result.exit_code == 0
        assert "System Information" in result.output

    def test_cli_verbose_flag(self):
        """Test verbose flag is accepted."""
        runner = CliRunner()
        result = runner.invoke(cli, ["-v", "hello"])
        assert result.exit_code == 0

    def test_cli_format_option(self):
        """Test format option is accepted."""
        runner = CliRunner()
        result = runner.invoke(cli, ["-f", "json", "hello"])
        assert result.exit_code == 0


class TestCLIContext:
    """Tests for CLI context object."""

    def test_context_init_defaults(self):
        """Test CLIContext initialization with defaults."""
        ctx = CLIContext()
        assert ctx.verbose is False
        assert ctx.config_file is None
        assert ctx.output_format == "text"
        assert ctx.config_manager is None

    def test_context_init_with_params(self):
        """Test CLIContext initialization with parameters."""
        ctx = CLIContext(
            verbose=True,
            config_file=Path("test.yaml"),
            output_format="json",
        )
        assert ctx.verbose is True
        assert ctx.config_file == Path("test.yaml")
        assert ctx.output_format == "json"


class TestTextFormatter:
    """Tests for text formatter."""

    def test_format_dict(self):
        """Test formatting a dictionary."""
        formatter = TextFormatter()
        data = {"name": "test", "value": 123}
        output = formatter.format(data)
        assert "name" in output
        assert "test" in output

    def test_format_list(self):
        """Test formatting a list."""
        formatter = TextFormatter()
        data = ["item1", "item2", "item3"]
        output = formatter.format(data)
        assert "item1" in output
        assert "item2" in output

    def test_format_nested_dict(self):
        """Test formatting nested dictionary."""
        formatter = TextFormatter()
        data = {
            "level1": {
                "level2": "value",
            }
        }
        output = formatter.format(data)
        assert "level1" in output
        assert "level2" in output


class TestJSONFormatter:
    """Tests for JSON formatter."""

    def test_format_dict(self):
        """Test formatting to JSON."""
        formatter = JSONFormatter()
        data = {"name": "test", "value": 123}
        output = formatter.format(data)
        parsed = json.loads(output)
        assert parsed["name"] == "test"
        assert parsed["value"] == 123

    def test_format_with_indent(self):
        """Test JSON formatting with custom indent."""
        formatter = JSONFormatter(indent=4)
        data = {"key": "value"}
        output = formatter.format(data)
        assert "    " in output  # 4-space indent

    def test_format_path(self):
        """Test formatting with Path objects."""
        formatter = JSONFormatter()
        data = {"path": Path("/tmp/test")}
        output = formatter.format(data)
        parsed = json.loads(output)
        assert parsed["path"] == "/tmp/test"


class TestCSVFormatter:
    """Tests for CSV formatter."""

    def test_format_list_of_dicts(self):
        """Test formatting list of dictionaries to CSV."""
        formatter = CSVFormatter()
        data = [
            {"name": "Alice", "age": 30},
            {"name": "Bob", "age": 25},
        ]
        output = formatter.format(data)
        assert "name" in output
        assert "Alice" in output
        assert "Bob" in output

    def test_format_empty_list(self):
        """Test formatting empty list."""
        formatter = CSVFormatter()
        output = formatter.format([])
        assert output == ""

    def test_format_invalid_type(self):
        """Test that non-list input raises error."""
        formatter = CSVFormatter()
        with pytest.raises(ValueError):
            formatter.format({"not": "a list"})


class TestGetFormatter:
    """Tests for formatter factory function."""

    def test_get_text_formatter(self):
        """Test getting text formatter."""
        formatter = get_formatter("text")
        assert isinstance(formatter, TextFormatter)

    def test_get_json_formatter(self):
        """Test getting JSON formatter."""
        formatter = get_formatter("json")
        assert isinstance(formatter, JSONFormatter)

    def test_get_csv_formatter(self):
        """Test getting CSV formatter."""
        formatter = get_formatter("csv")
        assert isinstance(formatter, CSVFormatter)

    def test_get_invalid_formatter(self):
        """Test that invalid format raises error."""
        with pytest.raises(ValueError):
            get_formatter("invalid_format")


class TestLoadCommandValidator:
    """Tests for LoadCommandInput validator."""

    def test_valid_input(self, tmp_path):
        """Test valid load command input."""
        input_file = tmp_path / "test.txt"
        input_file.write_text("test")

        validator = LoadCommandInput(
            input_path=input_file,
            model="BAAI/bge-large-en-v1.5",
            device="cpu",
        )
        assert validator.input_path == input_file
        assert validator.model == "BAAI/bge-large-en-v1.5"

    def test_invalid_path(self):
        """Test that non-existent path raises error."""
        with pytest.raises(ValueError):
            LoadCommandInput(input_path=Path("/nonexistent/path"))

    def test_invalid_device(self, tmp_path):
        """Test that invalid device raises error."""
        input_file = tmp_path / "test.txt"
        input_file.write_text("test")

        with pytest.raises(ValueError):
            LoadCommandInput(
                input_path=input_file,
                device="gpu",  # invalid
            )

    def test_valid_devices(self, tmp_path):
        """Test all valid device values."""
        input_file = tmp_path / "test.txt"
        input_file.write_text("test")

        for device in ["auto", "cpu", "cuda", "mps"]:
            validator = LoadCommandInput(
                input_path=input_file,
                device=device,
            )
            assert validator.device == device


class TestSearchCommandValidator:
    """Tests for SearchCommandInput validator."""

    def test_valid_input(self):
        """Test valid search command input."""
        validator = SearchCommandInput(
            query="test query",
            model="BAAI/bge-large-en-v1.5",
            top_k=5,
        )
        assert validator.query == "test query"
        assert validator.top_k == 5

    def test_empty_query(self):
        """Test that empty query raises error."""
        with pytest.raises(ValueError):
            SearchCommandInput(query="")

    def test_long_query(self):
        """Test that overly long query raises error."""
        with pytest.raises(ValueError):
            SearchCommandInput(query="x" * 3000)

    def test_invalid_format(self):
        """Test that invalid format raises error."""
        with pytest.raises(ValueError):
            SearchCommandInput(
                query="test",
                output_format="invalid",
            )

    def test_valid_formats(self):
        """Test all valid output formats."""
        for fmt in ["text", "json", "csv", "table"]:
            validator = SearchCommandInput(
                query="test",
                output_format=fmt,
            )
            assert validator.output_format == fmt


class TestAskCommandValidator:
    """Tests for AskCommandInput validator."""

    def test_valid_input(self):
        """Test valid ask command input."""
        validator = AskCommandInput(
            question="What is machine learning?",
            llm_provider="openai",
        )
        assert validator.question == "What is machine learning?"
        assert validator.llm_provider == "openai"

    def test_invalid_llm_provider(self):
        """Test that invalid LLM provider raises error."""
        with pytest.raises(ValueError):
            AskCommandInput(
                question="test",
                llm_provider="invalid_provider",
            )

    def test_valid_providers(self):
        """Test all valid LLM providers."""
        for provider in ["openai", "anthropic", "ollama"]:
            validator = AskCommandInput(
                question="test",
                llm_provider=provider,
            )
            assert validator.llm_provider == provider

    def test_temperature_range(self):
        """Test temperature value validation."""
        # Valid temperature
        validator = AskCommandInput(
            question="test",
            temperature=0.7,
        )
        assert validator.temperature == 0.7

        # Temperature too high
        with pytest.raises(ValueError):
            AskCommandInput(
                question="test",
                temperature=3.0,
            )


class TestProgressReporters:
    """Tests for progress reporter implementations."""

    def test_simple_progress(self):
        """Test SimpleProgress reporter."""
        reporter = SimpleProgress()
        reporter.start("Processing", total=100)
        reporter.update(50)
        reporter.finish("Done")
        # Should complete without error

    def test_rich_progress(self):
        """Test RichProgress reporter."""
        reporter = RichProgress()
        reporter.start("Processing", total=100)
        reporter.update(50)
        reporter.finish("Done")
        # Should complete without error

    def test_get_progress_reporter(self):
        """Test progress reporter factory."""
        for style in ["rich", "tqdm", "simple", "none"]:
            reporter = get_progress_reporter(style)
            assert reporter is not None

    def test_get_invalid_progress_style(self):
        """Test that invalid style raises error."""
        with pytest.raises(ValueError):
            get_progress_reporter("invalid_style")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
