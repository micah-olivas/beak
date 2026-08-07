"""Centralized Rich theme for the BEAK CLI."""

from rich.console import Console
from rich.theme import Theme

BEAK_BLUE = "#2E86AB"

# Matplotlib's tab10 — picked because it's designed for categorical encoding
# and stays legible on both light and dark terminal backgrounds. Rich
# downsamples these to 256- or 16-color automatically where truecolor isn't
# available. Two members are deliberately unused as category accents:
# tab10 red reads as an error next to the red MISSING status, and tab10 gray
# is reserved for the dim//background role.
TAB10 = {
    "blue": "#1f77b4",
    "orange": "#ff7f0e",
    "green": "#2ca02c",
    "red": "#d62728",
    "purple": "#9467bd",
    "brown": "#8c564b",
    "pink": "#e377c2",
    "gray": "#7f7f7f",
    "olive": "#bcbd22",
    "cyan": "#17becf",
}

# One accent per category, shared by the tools table and the databases table
# so a domain keeps its color across both — profile/HMM is purple whether
# you're looking at `hmmscan` or at Pfam-A, sequence search is blue whether
# it's `mmseqs` or UniRef90. Insertion order is the display order.
CATEGORY_STYLES = {
    "search": TAB10["blue"],
    "sequence.protein": TAB10["blue"],
    "sequence.nucleotide": TAB10["olive"],
    "align": TAB10["orange"],
    "tree": TAB10["brown"],
    "profile": TAB10["purple"],
    "structure": TAB10["green"],
    "embeddings": TAB10["cyan"],
    "utility": TAB10["gray"],
    "other": TAB10["gray"],
}

# Section headings. Keys match CATEGORY_STYLES; the tools table renders the
# `search`…`other` subset and the databases table the `sequence.*` subset,
# with `profile` and `structure` appearing in both.
CATEGORY_LABELS = {
    "search": "Search",
    "sequence.protein": "Sequence · protein",
    "sequence.nucleotide": "Sequence · nucleotide",
    "align": "Alignment",
    "tree": "Phylogenetics",
    "profile": "Profile · HMM",
    "structure": "Structure",
    "embeddings": "Embeddings",
    "utility": "Utility",
    "other": "Other",
}

BEAK_THEME = Theme({
    "brand": f"bold {BEAK_BLUE}",
    "brand.plain": BEAK_BLUE,
    **{f"cat.{k}": f"bold {v}" for k, v in CATEGORY_STYLES.items()},
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
