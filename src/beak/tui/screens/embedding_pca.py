"""PCA scatter plot of the project's embeddings, rendered with Unicode dots.

Loads `embeddings/**/mean_embeddings.pkl`, runs a 2-component PCA, and
renders each sequence as a colored dot. Sequences can be colored by any
taxonomic rank present in `taxonomy.parquet` (domain through species);
the target sequence is highlighted in red with a star marker.
"""

from typing import Dict, List, Optional, Tuple

import numpy as np
from textual.app import ComposeResult
from textual.binding import Binding
from textual.screen import Screen
from textual.widgets import Footer, Header, Static


_AXIS_W = 8        # cells reserved on the left for the y-axis label/ticks
_AXIS_H = 3        # cells reserved at the bottom for the x-axis label
_DEFAULT_COLOR = "#65CBF3"
_TARGET_COLOR = "#FF6B6B"
_UNKNOWN_COLOR = "#6E7681"

# Distinct, high-saturation colors for taxonomic groups. Sequenced for
# good legibility on dark terminals (no two adjacent are similar in hue).
_PALETTE = [
    "#65CBF3",  # cyan
    "#FFA62B",  # amber
    "#A66DD4",  # violet
    "#7DD87D",  # green
    "#FF6B9D",  # pink
    "#3DCFD4",  # teal
    "#F7E26B",  # yellow
    "#FF8C5A",  # orange
    "#9CA8FF",  # periwinkle
    "#5AD1A0",  # mint
    "#E08CFF",  # magenta
    "#C2B280",  # olive
]

# Coloring cycles through "off" plus whichever taxonomic ranks are
# actually populated in this project's taxonomy.parquet — broadest →
# narrowest matches the way most users think about clades.
_RANK_ORDER: List[str] = [
    "domain",
    "superkingdom",
    "kingdom",
    "phylum",
    "class",
    "order",
    "family",
    "genus",
    "species",
]


class _PCABody(Static):
    def __init__(self, parent_view: "EmbeddingPCAScreen", **kwargs) -> None:
        super().__init__("", **kwargs)
        self._pv = parent_view

    def render(self):
        if self._pv._error:
            return f"[red]{self._pv._error}[/red]"
        if self._pv._pcs is None:
            return "[dim]Loading PCA…[/dim]"
        return self._pv._render_scatter(self.size.width, self.size.height)

    def on_resize(self, event) -> None:
        self.refresh()


class EmbeddingPCAScreen(Screen):
    """Full-screen PCA scatter."""

    BINDINGS = [
        Binding("escape", "back", "Back"),
        Binding("c", "cycle_color", "Color"),
    ]

    DEFAULT_CSS = """
    EmbeddingPCAScreen _PCABody { height: 1fr; padding: 1 2; }
    """

    def __init__(self, project) -> None:
        super().__init__()
        self._project = project
        self._pcs = None
        self._var_ratio = None
        self._target_idx: Optional[int] = None
        self._error: Optional[str] = None
        self._n_dropped: int = 0
        # Per-rank label arrays, all the same length as `_pcs`. Keys are
        # whichever ranks the taxonomy.parquet has any non-null values
        # for — broader keys (domain) live alongside narrower (species).
        self._tax_levels: Dict[str, List[Optional[str]]] = {}
        # Available cycle for the `c` keybinding: ["off"] + ranks with data.
        self._color_modes: List[str] = ["off"]
        self._color_mode: str = "off"
        # Group → color, computed lazily per mode.
        self._color_cache: Dict[str, Dict[str, str]] = {}

    def compose(self) -> ComposeResult:
        yield Header()
        yield _PCABody(self, id="pca-body")
        yield Footer()

    def on_mount(self) -> None:
        self._load()
        # Default to "domain" coloring when taxonomy is available — the
        # broadest single group axis is the most informative starting view
        # on most projects.
        if "domain" in self._tax_levels:
            self._color_mode = "domain"
        elif len(self._color_modes) > 1:
            self._color_mode = self._color_modes[1]
        self._update_title()

    def action_back(self) -> None:
        self.app.sub_title = ""
        self.app.pop_screen()

    def action_cycle_color(self) -> None:
        if len(self._color_modes) <= 1:
            self.notify(
                "No taxonomy available — coloring stays off.", timeout=4
            )
            return
        idx = (
            self._color_modes.index(self._color_mode)
            if self._color_mode in self._color_modes else 0
        )
        self._color_mode = self._color_modes[(idx + 1) % len(self._color_modes)]
        self._update_title()
        self.query_one("#pca-body", _PCABody).refresh()

    def _update_title(self) -> None:
        self.app.title = "beak"
        if self._error:
            self.app.sub_title = f"PCA · {self._project.name} · error"
            return
        n = len(self._pcs) if self._pcs is not None else 0
        var = ""
        if self._var_ratio is not None and len(self._var_ratio) >= 2:
            v1, v2 = self._var_ratio[0] * 100, self._var_ratio[1] * 100
            var = f"  ·  PC1 {v1:.0f}%  PC2 {v2:.0f}%"
        dropped = f"  ·  dropped {self._n_dropped} NaN" if self._n_dropped else ""
        color = (
            f"  ·  color: {self._color_mode}" if self._color_mode != "off"
            else ""
        )
        self.app.sub_title = f"PCA · {self._project.name} ({n} seqs){var}{dropped}{color}"

    def _load(self) -> None:
        embed_dir = self._project.path / "embeddings"
        pkls = list(embed_dir.rglob("mean_embeddings.pkl"))
        if not pkls:
            self._error = (
                "No mean_embeddings.pkl in embeddings/ — submit an "
                "embedding job (with mean pooling) first."
            )
            return
        try:
            from ...embeddings import load_mean_embeddings
            df = load_mean_embeddings(pkls[0])
        except Exception as e:  # noqa: BLE001
            self._error = f"Could not read embeddings: {e}"
            return
        if df.empty:
            self._error = "Empty embeddings file."
            return

        n_before = len(df)
        df = df.dropna()
        n_dropped = n_before - len(df)
        if df.empty:
            self._error = "All embedding rows contained NaN."
            return

        try:
            n_comp = min(2, df.shape[0], df.shape[1])
            if n_comp < 2:
                self._error = "Need at least 2 sequences with valid embeddings for PCA."
                return
            from sklearn.decomposition import PCA
            model = PCA(n_components=2, random_state=0)
            coords = model.fit_transform(df.values)
            self._pcs = coords
            self._var_ratio = model.explained_variance_ratio_
            self._n_dropped = n_dropped
        except Exception as e:  # noqa: BLE001
            self._error = f"PCA failed: {e}"
            return

        # Match each PCA row against its taxonomy row by sequence_id, so
        # the per-point colors line up regardless of FASTA ordering.
        seq_ids = [str(s) for s in df.index]
        self._tax_levels = self._load_taxonomy(seq_ids)
        self._color_modes = ["off"] + [
            rank for rank in _RANK_ORDER if rank in self._tax_levels
        ]

        target_id = (
            self._project.manifest().get("target", {}).get("uniprot_id") or ""
        )
        if target_id:
            for i, seq_id in enumerate(seq_ids):
                if target_id in seq_id:
                    self._target_idx = i
                    break

    def _load_taxonomy(self, seq_ids: List[str]) -> Dict[str, List[Optional[str]]]:
        """Build per-rank label arrays aligned to `seq_ids`.

        Returns a dict keyed by rank name ("domain", "phylum", ...);
        only ranks with at least one non-null value across the project
        are included, so the cycle in the legend never offers a level
        with empty data.
        """
        tax_path = self._project.active_homologs_dir() / "taxonomy.parquet"
        if not tax_path.exists():
            return {}
        try:
            import pandas as pd
            tax = pd.read_parquet(tax_path)
        except Exception:
            return {}

        # Build a per-rank table indexed by both sequence_id and
        # uniprot_id so we can match rows regardless of the embedding's
        # FASTA-id flavor.
        per_rank: Dict[str, Dict[str, str]] = {}
        for rank in _RANK_ORDER:
            if rank not in tax.columns:
                continue
            lookup: Dict[str, str] = {}
            for _, row in tax.iterrows():
                sid = row.get("sequence_id")
                uid = row.get("uniprot_id")
                val = row.get(rank)
                if val is None or (isinstance(val, float) and np.isnan(val)):
                    continue
                sval = str(val)
                for k in (sid, uid):
                    if k and isinstance(k, str):
                        lookup.setdefault(k, sval)
            if lookup:
                per_rank[rank] = lookup

        def resolve(seq_id: str, table: Dict[str, str]) -> Optional[str]:
            if seq_id in table:
                return table[seq_id]
            for pfx in ("UniRef90_", "UniRef100_", "UniRef50_", "sp|", "tr|"):
                if seq_id.startswith(pfx):
                    candidate = seq_id[len(pfx):].split("|", 1)[0].split(" ", 1)[0]
                    if candidate in table:
                        return table[candidate]
            return None

        out: Dict[str, List[Optional[str]]] = {}
        for rank, lookup in per_rank.items():
            labels = [resolve(s, lookup) for s in seq_ids]
            if any(labels):
                out[rank] = labels
        return out

    def _color_for_idx(self, i: int) -> str:
        if i == self._target_idx:
            return _TARGET_COLOR
        if self._color_mode == "off" or self._color_mode not in self._tax_levels:
            return _DEFAULT_COLOR
        labels = self._tax_levels[self._color_mode]
        label = labels[i] if i < len(labels) else None
        if not label:
            return _UNKNOWN_COLOR
        cache = self._color_cache.setdefault(self._color_mode, {})
        if label not in cache:
            cache[label] = _PALETTE[len(cache) % len(_PALETTE)]
        return cache[label]

    # ---- rendering ----

    def _render_scatter(self, width: int, height: int) -> str:
        if self._pcs is None or len(self._pcs) == 0:
            return ""

        plot_w = max(20, width - _AXIS_W)
        # Reserve extra rows for the legend below the plot.
        legend_rows = self._legend_height()
        plot_h = max(8, height - _AXIS_H - legend_rows)

        xs = self._pcs[:, 0]
        ys = self._pcs[:, 1]
        x_min, x_max = float(xs.min()), float(xs.max())
        y_min, y_max = float(ys.min()), float(ys.max())
        x_range = max(x_max - x_min, 1e-6)
        y_range = max(y_max - y_min, 1e-6)

        # One braille cell is the smallest unit; we now plot a full cell
        # per point (using ●) instead of one sub-pixel — points read as
        # bigger and the color is unambiguous.
        col = ((xs - x_min) / x_range * (plot_w - 1)).astype(int)
        row = ((y_max - ys) / y_range * (plot_h - 1)).astype(int)

        # Per-cell winner: target outranks all others; otherwise the
        # most recently drawn color wins (cheap, deterministic).
        grid: List[List[Optional[Tuple[str, bool]]]] = [
            [None] * plot_w for _ in range(plot_h)
        ]
        for i in range(len(xs)):
            r, c = int(row[i]), int(col[i])
            if not (0 <= r < plot_h and 0 <= c < plot_w):
                continue
            is_target = i == self._target_idx
            current = grid[r][c]
            if current is not None and current[1] and not is_target:
                continue  # don't overwrite the target dot
            color = self._color_for_idx(i)
            grid[r][c] = (color, is_target)

        lines = []
        for cell_row in range(plot_h):
            if cell_row == 0:
                axis = f"{y_max:6.2f} "
            elif cell_row == plot_h - 1:
                axis = f"{y_min:6.2f} "
            elif cell_row == plot_h // 2:
                axis = f"{(y_min + y_max) / 2:6.2f} "
            else:
                axis = "       "
            line_parts = [f"[dim]{axis}[/dim]│"]

            for cell_col in range(plot_w):
                cell = grid[cell_row][cell_col]
                if cell is None:
                    line_parts.append(" ")
                    continue
                color, is_target = cell
                if is_target:
                    line_parts.append(f"[bold {color}]★[/bold {color}]")
                else:
                    line_parts.append(f"[{color}]●[/{color}]")
            lines.append("".join(line_parts))

        x_label_line = (
            "        " + "─" * plot_w + "\n"
            f"        [dim]{x_min:.2f}[/dim]"
            + " " * max(0, plot_w // 2 - 8)
            + f"[dim]{(x_min + x_max) / 2:.2f}[/dim]"
            + " " * max(0, plot_w // 2 - 6)
            + f"[dim]{x_max:.2f}[/dim]"
        )
        lines.append(x_label_line)
        lines.append(self._legend_text())
        return "\n".join(lines)

    def _legend_height(self) -> int:
        # A compact legend: one row for target/default, one row for the
        # taxonomy palette when active. Reserve 3 rows so the plot doesn't
        # jitter when the legend wraps to two lines.
        return 3

    def _legend_text(self) -> str:
        parts = []
        if self._target_idx is not None:
            parts.append(
                f"[bold {_TARGET_COLOR}]★[/bold {_TARGET_COLOR}] [dim]target[/dim]"
            )
        if self._color_mode == "off" or self._color_mode not in self._tax_levels:
            parts.append(
                f"[{_DEFAULT_COLOR}]●[/{_DEFAULT_COLOR}] [dim]hits[/dim]"
            )
        else:
            cache = self._color_cache.get(self._color_mode, {})
            labels = self._tax_levels[self._color_mode]
            # Show up to 8 groups in the legend, ranked by frequency —
            # at narrow ranks (genus/species) there can be hundreds, so
            # we deliberately truncate and surface the count.
            counts: Dict[str, int] = {}
            for lbl in labels:
                if lbl:
                    counts[lbl] = counts.get(lbl, 0) + 1
            ranked = sorted(counts.items(), key=lambda kv: kv[1], reverse=True)
            top = ranked[:8]
            for lbl, n in top:
                color = cache.get(lbl, _UNKNOWN_COLOR)
                parts.append(
                    f"[{color}]●[/{color}] [dim]{lbl} ({n})[/dim]"
                )
            n_more = len(ranked) - len(top)
            if n_more > 0:
                parts.append(f"[dim]+{n_more} more[/dim]")
            n_unk = sum(1 for l in labels if l is None)
            if n_unk:
                parts.append(
                    f"[{_UNKNOWN_COLOR}]●[/{_UNKNOWN_COLOR}] "
                    f"[dim]unknown ({n_unk})[/dim]"
                )
        prefix = "\n        "
        return prefix + "  ".join(parts)
