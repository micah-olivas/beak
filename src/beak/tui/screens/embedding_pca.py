"""PCA scatter plot of the project's embeddings, rendered with Unicode dots.

Loads `embeddings/**/mean_embeddings.pkl`, runs a 2-component PCA, and
renders each sequence as a colored dot. Sequences can be colored by any
taxonomic rank present in `taxonomy.parquet` (domain through species);
the target sequence is highlighted in red with a star marker.
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from textual import work
from textual.app import ComposeResult
from textual.binding import Binding
from textual.screen import Screen
from textual.widgets import Footer, Header, Static


_AXIS_W = 8        # cells reserved on the left for the y-axis label/ticks
_AXIS_H = 3        # cells reserved at the bottom for the x-axis label
_DEFAULT_COLOR = "#65CBF3"
_TARGET_COLOR = "#FF6B6B"
_UNKNOWN_COLOR = "#6E7681"

# Subsample budget for the scatter renderer. With tens of thousands of
# points crammed into a few thousand cells, the picture is dominated by
# overdraw — capping at ~3× the cell budget leaves the visual density
# essentially unchanged while dropping the per-render work several-fold.
# `_MIN_RENDER_POINTS` is the floor we never subsample below, so small
# projects still render every point.
_OVERSAMPLE = 3
_MIN_RENDER_POINTS = 5000

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
        Binding("m", "cycle_model", "Model"),
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
        # Multi-model selection. Populated in _load from the active
        # homolog set's embeddings entries; `m` cycles the active one.
        self._available_models: List[str] = []
        self._active_model: Optional[str] = None

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

    def action_cycle_model(self) -> None:
        """Cycle through embeddings computed for the active homolog set."""
        if len(self._available_models) <= 1:
            self.notify(
                "Only one model embedded for this set — submit another "
                "via the embeddings layer to compare.",
                timeout=4,
            )
            return
        idx = (
            self._available_models.index(self._active_model)
            if self._active_model in self._available_models else 0
        )
        self._active_model = self._available_models[
            (idx + 1) % len(self._available_models)
        ]
        # Recompute PCA + reset taxonomy/legend caches — vectors changed.
        self._pcs = None
        self._var_ratio = None
        self._target_idx = None
        self._tax_levels = {}
        self._color_cache = {}
        self._error = None
        self._n_dropped = 0
        self._load()
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
        # Show which model is active (and `m` to switch) when more than
        # one is available for this set.
        model_seg = ""
        if self._active_model:
            short = self._active_model.split("/")[-1]
            n_models = len(self._available_models)
            if n_models > 1:
                model_seg = f"  ·  model: {short} (m to switch · {n_models})"
            else:
                model_seg = f"  ·  model: {short}"
        self.app.sub_title = (
            f"PCA · {self._project.name} ({n} seqs)"
            f"{var}{dropped}{model_seg}{color}"
        )

    def _load(self) -> None:
        # Refresh the available-models list every load — the user may
        # have submitted another model since the last time.
        active_set = self._project.active_set_name()
        entries = [
            e for e in self._project.embeddings_models_for_set(active_set)
            if e.get("model") and e.get("n_embeddings")
        ]
        # Sort smallest-pkl-first so the foreground PCA loads on the
        # cheapest model. The other models get warmed in the background
        # by `_warm_pca_caches` after this returns, so cycling to them
        # later via `m` lands on a fresh sidecar instead of triggering
        # another 10-second fit.
        models_with_size = []
        for e in entries:
            model = e["model"]
            pkl = self._pkl_for_model(model)
            size = pkl.stat().st_size if pkl is not None else 0
            models_with_size.append((size, model))
        models_with_size.sort(key=lambda kv: kv[0])
        self._available_models = [m for _, m in models_with_size]
        if not self._active_model and self._available_models:
            # Default to the smallest model so first-paint is fast even
            # when the user has both a 35M-param and a 650M-param model
            # embedded — the bigger one finishes warming in the bg and
            # is one keystroke away.
            self._active_model = self._available_models[0]

        # Scope to the active (homolog set, model) embeddings dir.
        # Older projects pre-multi-model laid files at the per-set
        # root; we let `active_embeddings_dir(model=None)` resolve to
        # that legacy path if no model is selected (e.g. when the
        # manifest hasn't recorded a model yet).
        embed_dir = self._project.active_embeddings_dir(
            model=self._active_model
        )
        pkls = (
            list(embed_dir.rglob("mean_embeddings.pkl"))
            if embed_dir.exists() else []
        )
        if not pkls:
            self._error = (
                f"No mean_embeddings.pkl under "
                f"embeddings/{self._project.active_set_name()}/"
                f"{self._active_model or ''} — "
                "submit an embedding job (with mean pooling) for "
                "this set first."
            )
            return
        # Use the cached PCA path: returns immediately when a fresh
        # `<pkl>.pca.npz` sidecar exists, otherwise deserializes the
        # pickle, fits, and writes the sidecar. Any rewrite of the
        # pickle (model swap, fresh nseqs, re-pull) bumps its mtime and
        # invalidates the cache automatically.
        try:
            from ...embeddings import load_or_compute_pca_2d
            result = load_or_compute_pca_2d(pkls[0])
        except Exception as e:  # noqa: BLE001
            self._error = f"PCA failed: {e}"
            return

        if result is None:
            self._error = (
                "Need at least 2 sequences with valid embeddings for PCA."
            )
            return

        coords, var_ratio, seq_ids, n_dropped = result
        self._pcs = coords
        self._var_ratio = var_ratio
        self._n_dropped = n_dropped

        # Match each PCA row against its taxonomy row by sequence_id, so
        # the per-point colors line up regardless of FASTA ordering.
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

        # Foreground load done — warm the other models' PCA caches in
        # the background so a model switch is instant instead of paying
        # another fit.
        if len(self._available_models) > 1:
            self._warm_pca_caches()

    def _pkl_for_model(self, model_name: Optional[str]) -> Optional[Path]:
        """Resolve a model's `mean_embeddings.pkl`, or None if missing."""
        embed_dir = self._project.active_embeddings_dir(model=model_name)
        if not embed_dir.exists():
            return None
        pkls = list(embed_dir.rglob("mean_embeddings.pkl"))
        return pkls[0] if pkls else None

    @work(thread=True, exclusive=True, group="pca-warm")
    def _warm_pca_caches(self) -> None:
        """Build PCA sidecars for the non-active models, sequentially.

        Sequential rather than parallel: sklearn's PCA is multi-threaded
        already, so two concurrent fits compete for the same cores and
        each one runs slower than if they took turns. The cache writer
        is atomic, so the user can cycle to a model mid-warm and pick
        up whichever sidecar finished first; on a partial cache miss
        the cycle path simply re-fits.
        """
        from ...embeddings import load_or_compute_pca_2d

        for model in self._available_models:
            if model == self._active_model:
                continue
            pkl = self._pkl_for_model(model)
            if pkl is None:
                continue
            try:
                load_or_compute_pca_2d(pkl)
            except Exception:
                # Don't let a corrupt or unreadable pkl block warming
                # the rest — the user will see the error if they
                # actually cycle to that model.
                continue

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

    def _compute_cell_mode_colors(
        self,
        flat: np.ndarray,
        row: np.ndarray,
        col: np.ndarray,
        orig_idx: np.ndarray,
        plot_w: int,
    ):
        """Per-cell mode coloring, fully vectorized.

        Returns parallel arrays ``(rows, cols, colors)`` listing one
        entry per occupied cell, where ``colors[k]`` is the most-
        common label's color among the points that fell into that cell.
        Ties resolve deterministically (lexsort on (cell, -count) — the
        first label encountered wins among equally-frequent ones).

        When the active color mode is ``"off"`` or no taxonomy is
        loaded, every cell gets `_DEFAULT_COLOR` and we fall back to
        cheap first-occurrence-per-cell.
        """
        if (
            self._color_mode == "off"
            or self._color_mode not in self._tax_levels
        ):
            _, first = np.unique(flat, return_index=True)
            colors = [_DEFAULT_COLOR] * len(first)
            return row[first], col[first], colors

        # Build a label array aligned to `orig_idx`. Missing or empty
        # labels collapse to the sentinel "(unknown)" string so they
        # share the unknown-color cell.
        full_labels = self._tax_levels[self._color_mode]
        labels = np.empty(len(orig_idx), dtype=object)
        for k, idx in enumerate(orig_idx):
            ii = int(idx)
            lab = full_labels[ii] if ii < len(full_labels) else None
            labels[k] = str(lab) if lab else "(unknown)"

        # Encode labels as integer codes so we can fold them into the
        # cell-id key for a single np.unique pass that simultaneously
        # groups by cell *and* counts label occurrences.
        unique_labels, codes = np.unique(labels, return_inverse=True)
        n_labels = len(unique_labels)
        if n_labels == 0:
            _, first = np.unique(flat, return_index=True)
            return (
                row[first], col[first],
                [_DEFAULT_COLOR] * len(first),
            )

        combined = flat * n_labels + codes.astype(np.int64)
        combo_unique, counts = np.unique(combined, return_counts=True)
        combo_cells = combo_unique // n_labels
        combo_codes = combo_unique % n_labels

        # Sort by (cell ascending, count descending). The first row per
        # unique cell after this sort is the modal (cell, label) pair.
        order = np.lexsort([-counts, combo_cells])
        sorted_cells = combo_cells[order]
        sorted_codes = combo_codes[order]
        _, first_per_cell = np.unique(sorted_cells, return_index=True)
        mode_cells = sorted_cells[first_per_cell]
        mode_codes = sorted_codes[first_per_cell]

        # Translate cell ids back to (row, col) and look up the color
        # for each modal label via the per-mode color cache.
        rows_out = (mode_cells // plot_w).astype(np.int32)
        cols_out = (mode_cells % plot_w).astype(np.int32)
        cache = self._color_cache.setdefault(self._color_mode, {})
        colors_out: List[str] = []
        for code in mode_codes:
            label = str(unique_labels[int(code)])
            if label == "(unknown)":
                colors_out.append(_UNKNOWN_COLOR)
                continue
            cached = cache.get(label)
            if cached is None:
                cached = _PALETTE[len(cache) % len(_PALETTE)]
                cache[label] = cached
            colors_out.append(cached)
        return rows_out, cols_out, colors_out

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
        n_total = len(xs)

        # Subsample for big sets. With 25k+ points crammed into a few
        # thousand cells, the picture is dominated by overdraw — every
        # cell after the first ~plot_cells * _OVERSAMPLE points only
        # changes which color "wins" a cell. Capping at that level
        # drops the per-render work ~5× while leaving the visual
        # density essentially unchanged. Deterministic seed so cycling
        # models doesn't reshuffle the points the user just saw.
        plot_cells = plot_w * plot_h
        max_points = max(plot_cells * _OVERSAMPLE, _MIN_RENDER_POINTS)
        if n_total > max_points:
            rng = np.random.default_rng(0)
            sample_idx = rng.choice(n_total, size=max_points, replace=False)
            # Always preserve the target — it's the focal point of the
            # plot and must paint regardless of where it lands in the
            # random sample.
            if (self._target_idx is not None
                    and self._target_idx not in sample_idx):
                sample_idx = np.concatenate(
                    [sample_idx, np.array([self._target_idx])]
                )
            xs_s = xs[sample_idx]
            ys_s = ys[sample_idx]
            orig_idx = sample_idx
        else:
            xs_s = xs
            ys_s = ys
            orig_idx = np.arange(n_total)

        col = ((xs_s - x_min) / x_range * (plot_w - 1)).astype(np.int32)
        row = ((y_max - ys_s) / y_range * (plot_h - 1)).astype(np.int32)

        # In-bounds clamp via mask — handles edge points that round to
        # exactly plot_w/plot_h.
        in_bounds = (
            (col >= 0) & (col < plot_w) & (row >= 0) & (row < plot_h)
        )
        col = col[in_bounds]
        row = row[in_bounds]
        orig_idx = orig_idx[in_bounds]

        # Mode-per-cell coloring: when many points overlap a single
        # cell (eukaryotes + bacteria projecting to the same PC1×PC2
        # square), show the *majority* label rather than whichever
        # point np.unique happened to pick first. The whole pass stays
        # vectorized — the inner Python loop runs once per occupied
        # cell (a few thousand at most), not once per point.
        grid: List[List[Optional[Tuple[str, bool]]]] = [
            [None] * plot_w for _ in range(plot_h)
        ]

        if col.size:
            flat = row.astype(np.int64) * plot_w + col.astype(np.int64)
            occupied_rows, occupied_cols, occupied_colors = (
                self._compute_cell_mode_colors(flat, row, col, orig_idx, plot_w)
            )
            for k in range(len(occupied_rows)):
                grid[int(occupied_rows[k])][int(occupied_cols[k])] = (
                    occupied_colors[k], False,
                )

        # Target is overlaid last so the cell it lands in always shows
        # ★ regardless of how many other points share the cell or what
        # the mode color says.
        if self._target_idx is not None and 0 <= self._target_idx < n_total:
            tx = float(xs[self._target_idx])
            ty = float(ys[self._target_idx])
            tc = int((tx - x_min) / x_range * (plot_w - 1))
            tr = int((y_max - ty) / y_range * (plot_h - 1))
            if 0 <= tr < plot_h and 0 <= tc < plot_w:
                grid[tr][tc] = (_TARGET_COLOR, True)

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
