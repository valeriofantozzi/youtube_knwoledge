"""
Progress tracking utilities for long-running operations.

Provides progress bar and spinner implementations using Rich and tqdm.
"""

from abc import ABC, abstractmethod
from typing import Optional, Iterator, Any

from rich.progress import (
    Progress,
    SpinnerColumn,
    BarColumn,
    TaskProgressColumn,
    TimeRemainingColumn,
    TransferSpeedColumn,
    TextColumn,
    DownloadColumn,
)
from rich.console import Console


class ProgressReporter(ABC):
    """Base class for progress reporting."""

    @abstractmethod
    def start(self, description: str, total: Optional[int] = None) -> None:
        """Start progress tracking."""
        pass

    @abstractmethod
    def update(self, advance: int = 1, description: Optional[str] = None) -> None:
        """Update progress."""
        pass

    @abstractmethod
    def finish(self, description: Optional[str] = None) -> None:
        """Finish progress tracking."""
        pass


class RichProgress(ProgressReporter):
    """Progress reporter using Rich library."""

    def __init__(self, console: Optional[Console] = None):
        """
        Initialize Rich progress reporter.

        Args:
            console: Optional Rich Console instance
        """
        self.console = console
        self.progress: Optional[Progress] = None
        self.task_id: Optional[int] = None

    def start(self, description: str, total: Optional[int] = None) -> None:
        """Start a progress task."""
        if self.progress is None:
            self.progress = Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                BarColumn(),
                TaskProgressColumn(),
                TimeRemainingColumn(),
                console=self.console,
            )
            self.progress.start()

        self.task_id = self.progress.add_task(description, total=total)

    def update(
        self, advance: int = 1, description: Optional[str] = None
    ) -> None:
        """Update the current task."""
        if self.progress is None or self.task_id is None:
            return

        self.progress.update(self.task_id, advance=advance, description=description)  # type: ignore

    def finish(self, description: Optional[str] = None) -> None:
        """Finish the current task."""
        if self.progress is None or self.task_id is None:
            return

        if description:
            self.progress.update(self.task_id, description=description)  # type: ignore

        self.progress.stop()
        self.progress = None
        self.task_id = None


class SimpleProgress(ProgressReporter):
    """Simple progress reporter for minimal output."""

    def __init__(self, console: Optional[Console] = None):
        """
        Initialize simple progress reporter.

        Args:
            console: Optional Rich Console instance
        """
        self.console = console or Console()
        self.current = 0
        self.total: Optional[int] = None
        self.description = ""

    def start(self, description: str, total: Optional[int] = None) -> None:
        """Start progress tracking."""
        self.description = description
        self.total = total
        self.current = 0
        self.console.print(f"[cyan]{description}...[/cyan]")

    def update(
        self, advance: int = 1, description: Optional[str] = None
    ) -> None:
        """Update progress."""
        self.current += advance
        if description:
            self.description = description

    def finish(self, description: Optional[str] = None) -> None:
        """Finish progress tracking."""
        if description:
            self.console.print(f"[green]✓[/green] {description}")
        else:
            self.console.print(f"[green]✓[/green] {self.description}")


class TqdmProgress(ProgressReporter):
    """Progress reporter using tqdm library."""

    def __init__(self):
        """Initialize tqdm progress reporter."""
        self.pbar: Optional[Any] = None

    def start(self, description: str, total: Optional[int] = None) -> None:
        """Start a progress bar with tqdm."""
        from tqdm import tqdm

        self.pbar = tqdm(
            total=total,
            desc=description,
            unit="it",
            unit_scale=True,
            leave=True,
        )

    def update(
        self, advance: int = 1, description: Optional[str] = None
    ) -> None:
        """Update the progress bar."""
        if self.pbar is None:
            return

        self.pbar.update(advance)
        if description:
            self.pbar.set_description(description)

    def finish(self, description: Optional[str] = None) -> None:
        """Close the progress bar."""
        if self.pbar:
            self.pbar.close()
            self.pbar = None


class NoProgress(ProgressReporter):
    """Silent progress reporter that does nothing."""

    def start(self, description: str, total: Optional[int] = None) -> None:
        """Do nothing."""
        pass

    def update(
        self, advance: int = 1, description: Optional[str] = None
    ) -> None:
        """Do nothing."""
        pass

    def finish(self, description: Optional[str] = None) -> None:
        """Do nothing."""
        pass


def get_progress_reporter(
    style: str = "rich",
    console: Optional[Console] = None,
) -> ProgressReporter:
    """
    Get a progress reporter instance.

    Args:
        style: Type of reporter ('rich', 'tqdm', 'simple', 'none')
        console: Optional Rich Console instance

    Returns:
        ProgressReporter instance

    Raises:
        ValueError: If style is unknown
    """
    reporters = {
        "rich": RichProgress(console),
        "tqdm": TqdmProgress(),
        "simple": SimpleProgress(console),
        "none": NoProgress(),
    }

    if style not in reporters:
        raise ValueError(f"Unknown progress style: {style}")

    return reporters[style]


__all__ = [
    "ProgressReporter",
    "RichProgress",
    "SimpleProgress",
    "TqdmProgress",
    "NoProgress",
    "get_progress_reporter",
]
