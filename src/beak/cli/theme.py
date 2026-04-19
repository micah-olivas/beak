"""Centralized Rich theme for the BEAK CLI."""

from rich.console import Console
from rich.theme import Theme

BEAK_BLUE = "#2E86AB"

BEAK_THEME = Theme({
    "brand": f"bold {BEAK_BLUE}",
    "brand.plain": BEAK_BLUE,
})

BORDER_STYLE = BEAK_BLUE

STATUS_STYLES = {
    "COMPLETED": "bold green",
    "RUNNING": "bold yellow",
    "SUBMITTED": "bold cyan",
    "FAILED": "bold red",
    "CANCELLED": "dim",
    "UNKNOWN": "dim",
    "PENDING": "dim",
}

STAGE_ICONS = {
    "done": "[green]\u2713[/green]",
    "active": "[yellow bold]\u25b6[/yellow bold]",
    "pending": "[dim]\u25cb[/dim]",
}


def get_console() -> Console:
    """Create a Console pre-configured with the BEAK theme."""
    return Console(theme=BEAK_THEME)
