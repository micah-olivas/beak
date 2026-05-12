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
from textual.screen import ModalScreen, Screen
from textual.containers import Horizontal, Vertical
from textual.widgets import Button, Footer, Header, Label, Select, Static


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

# Continuous-color modes — rendered with a gradient + min/max legend
# instead of discrete chips. Sequences with no value get a muted grey
# so the colored points stand out.
_TEMP_MODE = "growth_temp"
_IDENTITY_MODE = "identity"
_TRAIT_MODE = "trait"
_TEMP_NONE_COLOR = "#3A3F4A"
_TRAIT_TRUE_COLOR = "#7DD87D"   # palette green
_TRAIT_FALSE_COLOR = "#FF6B6B"  # palette red


def _classify_trait(raw_values):
    """Decide if a trait column is binary / numeric / categorical.

    Returns `(kind, normalized_values, extras)` where:
        - `kind` is "binary", "numeric", or "categorical" (None to skip).
        - `normalized_values` is a list aligned to `raw_values`:
            binary → bool / None
            numeric → float / None
            categorical → str / None
        - `extras` carries `{"min", "max"}` for numeric or
          `{"categories"}` (frequency-ordered) for categorical.

    A column is "binary" iff every non-null value parses to exactly
    True or False (Python bool, the strings "true"/"false" in any
    case, or 0/1). It's "numeric" iff at least 50% of non-nulls parse
    to a float. Otherwise "categorical".
    """
    non_null = [v for v in raw_values if v is not None]
    if not non_null:
        return None, [None] * len(raw_values), {}

    # Try binary first — the most useful coloring for traits like
    # gram_positive / motile / aerotolerant.
    def to_bool(v):
        if isinstance(v, bool):
            return v
        if isinstance(v, (int, float)):
            if v == 0:
                return False
            if v == 1:
                return True
        if isinstance(v, str):
            s = v.strip().lower()
            if s == "true":
                return True
            if s == "false":
                return False
        return None

    bools = [to_bool(v) for v in non_null]
    if all(b is not None for b in bools):
        normalized = [
            (to_bool(v) if v is not None else None) for v in raw_values
        ]
        return "binary", normalized, {}

    # Numeric: at least 50% non-null values parse to float.
    def to_float(v):
        if isinstance(v, bool):
            return 1.0 if v else 0.0
        if isinstance(v, (int, float)):
            try:
                f = float(v)
                return None if (f != f) else f  # NaN check
            except (TypeError, ValueError):
                return None
        if isinstance(v, str):
            try:
                return float(v.strip())
            except ValueError:
                return None
        return None

    floats = [to_float(v) for v in non_null]
    n_numeric = sum(1 for f in floats if f is not None)
    if n_numeric / len(non_null) > 0.5:
        normalized = [
            (to_float(v) if v is not None else None) for v in raw_values
        ]
        finite = [f for f in normalized if f is not None]
        if not finite:
            return None, [None] * len(raw_values), {}
        return "numeric", normalized, {
            "min": float(min(finite)),
            "max": float(max(finite)),
        }

    # Categorical fallback — coerce to string, frequency-rank.
    str_values = [
        (str(v) if v is not None else None) for v in raw_values
    ]
    counts: Dict[str, int] = {}
    for v in str_values:
        if v is not None:
            counts[v] = counts.get(v, 0) + 1
    categories = [k for k, _ in sorted(
        counts.items(), key=lambda kv: kv[1], reverse=True
    )]
    return "categorical", str_values, {"categories": categories}


class _TaxonFilterModal(ModalScreen):
    """Two-pane list picker: rank list (left) + value list (right).

    Dismisses with:
      ("apply", rank, value) — apply this filter and recompute
      ("clear",)             — remove the active filter
      None                   — cancelled, no change
    """

    BINDINGS = [
        Binding("escape", "cancel", "Cancel"),
        Binding("left", "focus_ranks", "← Ranks", show=False),
    ]

    DEFAULT_CSS = """
    _TaxonFilterModal {
        align: center middle;
    }
    _TaxonFilterModal #filter-panel {
        width: 64;
        height: 22;
        padding: 1 2;
        background: $panel;
        border: solid #2E86AB;
    }
    _TaxonFilterModal #filter-title {
        text-align: center;
        margin-bottom: 1;
    }
    _TaxonFilterModal #lists-row {
        height: 1fr;
    }
    _TaxonFilterModal #rank-col {
        width: 20;
        margin-right: 2;
    }
    _TaxonFilterModal #rank-list {
        height: 1fr;
        border: solid $primary-darken-2;
    }
    _TaxonFilterModal #rank-list:focus-within {
        border: solid $primary;
    }
    _TaxonFilterModal #value-col {
        width: 1fr;
    }
    _TaxonFilterModal #value-list {
        height: 1fr;
        border: solid $primary-darken-2;
    }
    _TaxonFilterModal #value-list:focus-within {
        border: solid $primary;
    }
    _TaxonFilterModal .col-label {
        color: $text-muted;
        height: 1;
        margin-bottom: 0;
    }
    _TaxonFilterModal #hint-line {
        text-align: center;
        color: $text-muted;
        height: 1;
        margin-top: 1;
    }
    _TaxonFilterModal #filter-buttons {
        height: 3;
        align: center middle;
    }
    _TaxonFilterModal Button {
        margin: 0 1;
    }
    """

    def __init__(
        self,
        available_ranks: List[str],
        full_tax_levels: Dict[str, List[Optional[str]]],
        current_rank: Optional[str] = None,
        current_value: Optional[str] = None,
    ) -> None:
        super().__init__()
        self._available_ranks = available_ranks
        self._initial_rank = current_rank
        self._initial_value = current_value
        # Sorted unique non-null values per rank from the full (unfiltered)
        # taxonomy so the picker always shows all options.
        self._rank_values: Dict[str, List[str]] = {}
        for rank in available_ranks:
            labels = full_tax_levels.get(rank, [])
            vals = sorted({l for l in labels if l})
            self._rank_values[rank] = vals
        # Sequence count per (rank, value) for the info suffix.
        self._rank_counts: Dict[str, Dict[str, int]] = {}
        for rank in available_ranks:
            labels = full_tax_levels.get(rank, [])
            counts: Dict[str, int] = {}
            for lbl in labels:
                if lbl:
                    counts[lbl] = counts.get(lbl, 0) + 1
            self._rank_counts[rank] = counts
        # Per-rank saved value so navigating back to a rank restores where
        # the user left off.
        self._saved_value: Dict[str, str] = {}
        if current_rank and current_value:
            self._saved_value[current_rank] = current_value
        self._has_existing = current_rank is not None and current_value is not None

    def compose(self) -> ComposeResult:
        from textual.widgets import ListView, ListItem
        with Vertical(id="filter-panel"):
            yield Label("[bold]Filter PCA by taxon[/bold]", id="filter-title")
            with Horizontal(id="lists-row"):
                with Vertical(id="rank-col"):
                    yield Label("rank", classes="col-label")
                    yield ListView(
                        *[
                            ListItem(Label(rank))
                            for rank in self._available_ranks
                        ],
                        id="rank-list",
                    )
                with Vertical(id="value-col"):
                    yield Label("value", classes="col-label", id="value-col-label")
                    yield ListView(id="value-list")
            yield Static(
                "[dim]↑↓ navigate  ·  ↵ rank→values→apply  ·  ← back[/dim]",
                id="hint-line",
            )
            with Horizontal(id="filter-buttons"):
                yield Button("Cancel", id="cancel-btn")
                if self._has_existing:
                    yield Button("Clear filter", id="clear-btn", variant="warning")
                yield Button("Apply", id="apply-btn", variant="primary")

    def on_mount(self) -> None:
        from textual.widgets import ListView
        rank_list = self.query_one("#rank-list", ListView)
        rank_list.focus()
        # Set initial rank cursor and populate value list.
        start_rank_idx = 0
        if self._initial_rank in self._available_ranks:
            start_rank_idx = self._available_ranks.index(self._initial_rank)
        # call_after_refresh so the ListView has rendered before we move.
        self.call_after_refresh(self._init_cursors, start_rank_idx)

    def _init_cursors(self, rank_idx: int) -> None:
        from textual.widgets import ListView
        rank_list = self.query_one("#rank-list", ListView)
        rank_list.index = rank_idx
        # Populate value list for this rank. Setting index fires Highlighted
        # which calls _repopulate_value_list, but call directly here too
        # to be safe on mount before the first render cycle.
        rank = self._available_ranks[rank_idx] if rank_idx < len(self._available_ranks) else None
        if rank:
            self._repopulate_value_list(rank)

    # ---- helpers ----

    def _active_rank(self) -> Optional[str]:
        from textual.widgets import ListView
        try:
            rank_list = self.query_one("#rank-list", ListView)
            idx = rank_list.index
            if idx is not None and 0 <= idx < len(self._available_ranks):
                return self._available_ranks[idx]
        except Exception:
            pass
        return None

    def _active_value(self) -> Optional[str]:
        from textual.widgets import ListView
        rank = self._active_rank()
        if rank is None:
            return None
        try:
            value_list = self.query_one("#value-list", ListView)
            idx = value_list.index
            vals = self._rank_values.get(rank, [])
            if idx is not None and 0 <= idx < len(vals):
                return vals[idx]
        except Exception:
            pass
        return None

    def _repopulate_value_list(self, rank: str) -> None:
        from textual.widgets import ListView, ListItem
        try:
            value_list = self.query_one("#value-list", ListView)
            label = self.query_one("#value-col-label", Label)
        except Exception:
            return
        vals = self._rank_values.get(rank, [])
        counts = self._rank_counts.get(rank, {})
        value_list.clear()
        for val in vals:
            n = counts.get(val, 0)
            value_list.append(ListItem(Label(f"{val}  [dim]({n})[/dim]")))
        label.update(f"value  [dim]({len(vals)})[/dim]")
        # Restore saved position for this rank if available.
        saved = self._saved_value.get(rank)
        if saved and saved in vals:
            value_list.index = vals.index(saved)

    # ---- events ----

    def on_list_view_highlighted(self, event) -> None:
        from textual.widgets import ListView
        if event.list_view.id == "rank-list":
            rank = self._active_rank()
            if rank is not None:
                self._repopulate_value_list(rank)
        elif event.list_view.id == "value-list":
            # Save position so navigating away and back restores it.
            rank = self._active_rank()
            value = self._active_value()
            if rank and value:
                self._saved_value[rank] = value

    def on_list_view_selected(self, event) -> None:
        from textual.widgets import ListView
        if event.list_view.id == "rank-list":
            # Enter on rank → move focus to value list.
            self.query_one("#value-list", ListView).focus()
        elif event.list_view.id == "value-list":
            # Enter on value → apply filter.
            self._do_apply()

    # ---- actions ----

    def action_cancel(self) -> None:
        self.dismiss(None)

    def action_focus_ranks(self) -> None:
        from textual.widgets import ListView
        self.query_one("#rank-list", ListView).focus()

    def _do_apply(self) -> None:
        rank = self._active_rank()
        value = self._active_value()
        if rank and value:
            self.dismiss(("apply", rank, value))
        else:
            self.dismiss(None)

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "cancel-btn":
            self.dismiss(None)
        elif event.button.id == "clear-btn":
            self.dismiss(("clear",))
        elif event.button.id == "apply-btn":
            self._do_apply()


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

    def on_mouse_move(self, event) -> None:
        # `event.x/y` is widget-content-local (used for the data
        # lookup); `screen_x/screen_y` is absolute (used to position
        # the floating tooltip). Pass both — the parent picks the
        # nearest plotted point AND positions the tooltip in one
        # call.
        self._pv._on_hover(
            event.x, event.y,
            int(getattr(event, "screen_x", event.x)),
            int(getattr(event, "screen_y", event.y)),
        )

    def on_leave(self, event) -> None:
        self._pv._on_hover_leave()


class EmbeddingPCAScreen(Screen):
    """Full-screen PCA scatter."""

    BINDINGS = [
        Binding("escape", "back", "Back"),
        # `c` is a 3-position toggle (off / tax / growth_temp). The
        # taxonomic-rank choice within tax mode is handled separately
        # via `[` / `]` — earlier this was a single 9-position cycle
        # that meant reaching `growth_temp` cost 9 keypresses.
        Binding("c", "cycle_color", "Color"),
        Binding("left_square_bracket", "rank_prev", "Rank ←"),
        Binding("right_square_bracket", "rank_next", "Rank →"),
        Binding("f", "open_filter", "Filter"),
        # Model selection used to cycle on `m`; replaced with a Select
        # dropdown above the canvas because (a) discoverability is
        # better when the available models are visible at once, and
        # (b) random-access beats N-presses-to-cycle when a project
        # accumulates 3+ embeddings (esm2 + esmc-300m + esmc-600m, …).
    ]

    # Stack the floating tooltip on a layer above the base plot so its
    # `offset` positions it absolutely without disturbing the row's
    # horizontal layout. Without this the tooltip would either get
    # clipped by the body or shift the sidebar around when shown.
    LAYERS = ("base", "tooltip")

    DEFAULT_CSS = """
    /* Horizontal split: plot on the left taking all remaining width,
       sidebar on the right at fixed cell-width. `layout: horizontal`
       is the default for `Horizontal`, but stating it explicitly
       guards against a default-shift in a future Textual version. */
    EmbeddingPCAScreen #pca-row {
        layout: horizontal;
        width: 100%;
        height: 1fr;
    }
    EmbeddingPCAScreen #pca-body {
        width: 1fr;
        height: 100%;
        padding: 1 2;
    }
    /* Right sidebar — model dropdown + legend. Width 30 = enough for
       "Model" + a 28-cell Select with a 1-cell border-left; legend
       chips wrap onto multiple lines underneath. The earlier bottom
       legend ate 3 rows on every plot; pulling it into a dedicated
       column gives the plot back its full vertical span. */
    EmbeddingPCAScreen #pca-sidebar {
        width: 30;
        height: 100%;
        padding: 1 1;
        background: $surface;
        border-left: solid #2E86AB;
    }
    EmbeddingPCAScreen #pca-sidebar Label {
        color: $text-muted;
        margin-top: 1;
    }
    EmbeddingPCAScreen #pca-model-select {
        width: 28;
        margin-bottom: 1;
    }
    EmbeddingPCAScreen #pca-legend {
        height: auto;
        padding: 1 0 0 0;
    }
    /* Floating per-point tooltip. `layer: tooltip` puts it above the
       plot so it doesn't reflow anything; `offset` is set
       imperatively in `_update_hover_tooltip` to the current mouse
       position (screen-absolute). The widget renders its own
       vertical-spine callout (left edge `│` with a `◄` at the
       pointer row), so we don't ask Textual for a border — that
       would draw a uniform rectangle and we want one cell of the
       left edge swapped for an arrow. Hidden by default. */
    EmbeddingPCAScreen #pca-hover-tooltip {
        layer: tooltip;
        dock: none;
        background: $panel;
        padding: 0 1;
        width: auto;
        height: auto;
        max-width: 50;
        display: none;
        color: $text;
    }
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
        # Per-sequence growth temperature (°C), aligned to `_pcs`. None
        # for sequences whose organism wasn't in the Enqvist dataset.
        # Populated alongside `_tax_levels` when the column is present.
        self._tax_temps: List[Optional[float]] = []
        # Per-sequence % identity to target (0–100), aligned to `_pcs`.
        # None for the target itself (it's excluded from its own color)
        # and for any homolog whose alignment is unavailable.
        self._identity: List[Optional[float]] = []
        # Per-trait data — `dict[col_name, {"kind", "values", ...}]`.
        # Built once per `_load` from the active set's traits.parquet.
        # `_available_traits` is the cycle list (coverage-ordered);
        # `_trait_col` is the currently selected one.
        self._trait_data: Dict[str, Dict] = {}
        self._available_traits: List[str] = []
        self._trait_col: Optional[str] = None
        # Top-level color mode. The `c` keybinding cycles
        # off → tax → growth_temp → identity → trait → off; modes whose
        # data isn't populated for this project drop out automatically.
        self._color_modes: List[str] = ["off"]
        self._color_mode: str = "off"
        # Within tax mode, which taxonomic rank to color by. Drives the
        # `[` / `]` rank-step bindings. Defaults to the first available
        # rank in `_RANK_ORDER` (typically "domain").
        self._available_ranks: List[str] = []
        self._tax_rank: Optional[str] = None
        # Group → color, computed lazily per mode.
        self._color_cache: Dict[str, Dict[str, str]] = {}
        # Multi-model selection. Populated in _load from the active
        # homolog set's embeddings entries; the sidebar dropdown drives
        # which one is rendered.
        self._available_models: List[str] = []
        self._active_model: Optional[str] = None
        # Sequence ids aligned to `_pcs` rows — populated by `_load`.
        # Used by the hover tooltip to look up which sequence the
        # cursor is over.
        self._seq_ids: List[str] = []
        # Hover-lookup cache: arrays of plotted (col, row, orig_idx)
        # in body-local cell coords. Built once per `_render_scatter`,
        # consumed by `_on_hover`. None when no plot has rendered yet.
        self._hover_cols: Optional[np.ndarray] = None
        self._hover_rows: Optional[np.ndarray] = None
        self._hover_orig_idx: Optional[np.ndarray] = None
        self._hover_origin: Tuple[int, int] = (_AXIS_W, 0)
        # Last hovered point index, so identical mouse-move events
        # don't churn the sub_title.
        self._last_hover_idx: Optional[int] = None
        # Active taxon filter. When set, `_load` runs PCA only on the
        # sequences whose taxonomy matches `_filter_value` at `_filter_rank`.
        # `_full_tax_levels` is the unfiltered taxonomy so the filter modal
        # always shows all available values regardless of what's active.
        self._filter_rank: Optional[str] = None
        self._filter_value: Optional[str] = None
        self._full_tax_levels: Dict[str, List[Optional[str]]] = {}
        # Set in `action_back` so post-pop worker callbacks (the
        # `_load_async_done` hop) skip touching unmounted widgets.
        #
        # NB: must NOT be named ``_closing`` — that collides with
        # ``MessagePump._closing`` and silently breaks ``post_message``,
        # wedging screen-pop teardown. See alignment_view.py for the
        # full incident write-up.
        self._user_closing: bool = False

    def compose(self) -> ComposeResult:
        yield Header()
        with Horizontal(id="pca-row"):
            yield _PCABody(self, id="pca-body")
            with Vertical(id="pca-sidebar"):
                yield Label("Model", id="pca-model-label")
                yield Select(
                    options=[],
                    id="pca-model-select",
                    allow_blank=True,
                    prompt="(loading…)",
                )
                yield Label("Legend", id="pca-legend-label")
                yield Static("", id="pca-legend")
        # Floating tooltip — yielded outside the row so its absolute
        # offset positioning isn't constrained by the horizontal flex.
        yield Static("", id="pca-hover-tooltip")
        yield Footer()

    def on_mount(self) -> None:
        # Defer the heavy load to a thread worker so the screen
        # frame paints immediately. The body widget shows
        # "Loading PCA…" until `_load_async_done` flips state.
        self._update_title()
        self._load_async()

    @work(thread=True, exclusive=True, group="pca-load")
    def _load_async(self) -> None:
        """Run the slow `_load` (parquet reads, PCA compute, identity
        scan) off the UI thread, then schedule a finalize step on the
        main loop.

        Doing this synchronously in `on_mount` was the source of the
        2–6 s stall users hit when first opening the PCA screen on
        anything bigger than a few hundred sequences. The first PCA
        fit + the per-row identity loop were the dominant costs.

        UI hop uses ``loop.call_soon_threadsafe`` (non-blocking)
        instead of ``app.call_from_thread`` (blocking) — see
        ``alignment_view._populate_caches_async`` for the deadlock
        vector this avoids on screen-pop teardown.
        """
        try:
            self._load()
        except Exception as e:  # noqa: BLE001
            self._error = f"Load failed: {e}"
        try:
            loop = self.app._loop
            if loop is not None:
                loop.call_soon_threadsafe(self._load_async_done)
        except Exception:  # noqa: BLE001
            pass

    def _load_async_done(self) -> None:
        # Worker hop landed on the main thread — bail if the user already
        # popped the screen. Otherwise we'd push the loading state into
        # widgets that no longer paint and potentially leave the parent
        # detail screen in a broken render state on resume.
        if self._user_closing:
            return
        # Default to tax/domain coloring when taxonomy is available —
        # the broadest single group axis is the most informative
        # starting view on most projects.
        if "tax" in self._color_modes:
            self._color_mode = "tax"
        elif _TEMP_MODE in self._color_modes:
            self._color_mode = _TEMP_MODE
        self._update_title()
        self._populate_model_select()
        self._refresh_legend()
        try:
            self.query_one("#pca-body", _PCABody).refresh()
        except Exception:
            pass

    def action_back(self) -> None:
        # Cancel our background workers before popping. Their
        # `app.call_from_thread` callbacks would otherwise land on the
        # main thread minutes later and mark unmounted PCA widgets
        # dirty — we've seen those stale refreshes leave the parent
        # detail screen unable to repaint.
        self._user_closing = True
        try:
            self.workers.cancel_group(self, "pca-load")
            self.workers.cancel_group(self, "pca-warm")
        except Exception:
            pass
        self.app.sub_title = ""
        self.app.pop_screen()

    def action_cycle_color(self) -> None:
        """Cycle the top-level color mode (off / tax / growth_temp)."""
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
        self.query_one("#pca-body", _PCABody).refresh(); self._refresh_legend()

    def action_rank_prev(self) -> None:
        self._step_rank(-1)

    def action_rank_next(self) -> None:
        self._step_rank(+1)

    def _step_rank(self, delta: int) -> None:
        """Advance the sub-axis within the active mode.

        - Tax mode: cycles taxonomic rank (domain → phylum → ...).
        - Trait mode: cycles trait column (most-populated first).
        - Other modes: no-op with a one-line hint pointing at `c`.
        """
        if self._color_mode == "tax":
            if not self._available_ranks:
                return
            if self._tax_rank not in self._available_ranks:
                self._tax_rank = self._available_ranks[0]
                return
            idx = self._available_ranks.index(self._tax_rank)
            self._tax_rank = self._available_ranks[
                (idx + delta) % len(self._available_ranks)
            ]
        elif self._color_mode == _TRAIT_MODE:
            if not self._available_traits:
                return
            if self._trait_col not in self._available_traits:
                self._trait_col = self._available_traits[0]
                return
            idx = self._available_traits.index(self._trait_col)
            self._trait_col = self._available_traits[
                (idx + delta) % len(self._available_traits)
            ]
        else:
            self.notify(
                "[ / ] cycles ranks (in tax mode) or traits (in "
                "trait mode). Press `c` to switch.",
                timeout=3,
            )
            return
        self._update_title()
        self.query_one("#pca-body", _PCABody).refresh(); self._refresh_legend()

    def action_open_filter(self) -> None:
        if not self._full_tax_levels:
            self.notify(
                "No taxonomy data — load taxonomy first.", timeout=4
            )
            return
        available = [r for r in _RANK_ORDER if r in self._full_tax_levels]
        if not available:
            self.notify("No taxonomy ranks available.", timeout=3)
            return
        modal = _TaxonFilterModal(
            available_ranks=available,
            full_tax_levels=self._full_tax_levels,
            current_rank=self._filter_rank,
            current_value=self._filter_value,
        )
        self.app.push_screen(modal, self._on_filter_result)

    def _on_filter_result(self, result) -> None:
        if result is None:
            return
        if result[0] == "clear":
            self._filter_rank = None
            self._filter_value = None
        else:
            _, rank, value = result
            self._filter_rank = rank
            self._filter_value = value
        self._pcs = None
        self._var_ratio = None
        self._target_idx = None
        self._tax_levels = {}
        self._color_cache = {}
        self._error = None
        self._n_dropped = 0
        self._update_title()
        try:
            self.query_one("#pca-body", _PCABody).refresh()
        except Exception:
            pass
        self._load_async()

    def _run_pca_on_subset(self, pkl_path: Path, keep_ids: set):
        """Fit 2-component PCA on a subset of the pickle's sequences.

        Not cached — intended for interactive filtered views. Returns the
        same ``(coords, var_ratio, seq_ids, n_dropped)`` tuple as
        ``load_or_compute_pca_2d``, or None when the subset is too small.
        """
        from sklearn.decomposition import PCA
        from ...embeddings import load_mean_embeddings

        df = load_mean_embeddings(pkl_path)
        if df.empty:
            return None
        df = df[df.index.isin(keep_ids)]
        if df.empty:
            return None
        n_before = len(df)
        df = df.dropna()
        n_dropped = n_before - len(df)
        if df.shape[0] < 2 or df.shape[1] < 2:
            return None
        model = PCA(n_components=2, random_state=0)
        coords = model.fit_transform(df.values).astype("float32")
        var_ratio = model.explained_variance_ratio_.astype("float32")
        seq_ids = [str(s) for s in df.index]
        return coords, var_ratio, seq_ids, int(n_dropped)

    def _set_model(self, name: str) -> None:
        """Switch active embedding model and recompute PCA.

        Reset path: we drop every cache that's keyed by sequence
        (PCs, target index, taxonomy alignments, color palette) and
        re-run `_load`. Cheap on the parquet side; the dominant cost
        is `load_or_compute_pca_2d` which itself caches mtime-keyed
        sidecar npz files, so swapping back to a previously-seen
        model is near-instant after the first compute.
        """
        if name == self._active_model:
            return
        if name not in self._available_models:
            return
        self._active_model = name
        self._pcs = None
        self._var_ratio = None
        self._target_idx = None
        self._tax_levels = {}
        self._full_tax_levels = {}
        self._color_cache = {}
        self._error = None
        self._n_dropped = 0
        self._update_title()
        # Body re-renders to show the "Loading PCA…" placeholder
        # while the swap-to-new-model worker runs in the background.
        try:
            self.query_one("#pca-body", _PCABody).refresh()
        except Exception:
            pass
        self._load_async()

    def _populate_model_select(self) -> None:
        """Sync the Select widget's options with `_available_models`.

        Called after `_load` discovers what's on disk. When only one
        model exists the row stays visible (the user still sees what
        they're looking at) but the Select is disabled so it reads as
        an info readout rather than a useless control.
        """
        try:
            select = self.query_one("#pca-model-select", Select)
            label = self.query_one("#pca-model-label", Label)
        except Exception:
            return
        if not self._available_models:
            select.set_options([])
            select.prompt = "(no models)"
            select.disabled = True
            return
        select.set_options([(m, m) for m in self._available_models])
        # `value` setter would fire Select.Changed; suppress so the
        # initial state-sync doesn't trigger a re-load.
        if self._active_model in self._available_models:
            with select.prevent(Select.Changed):
                select.value = self._active_model
        # Single-model projects: keep the row but make the field
        # read-only (it's information, not a choice).
        only_one = len(self._available_models) <= 1
        select.disabled = only_one
        label.update("Model" if not only_one else "Model (only)")

    def on_select_changed(self, event: Select.Changed) -> None:
        if event.select.id != "pca-model-select":
            return
        new_model = str(event.value) if event.value is not None else None
        if new_model:
            self._set_model(new_model)

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
        # Color label includes the active sub-axis when one applies
        # (tax rank, trait column) so the current `[` / `]` selection
        # is visible without consulting the legend.
        if self._color_mode == "tax" and self._tax_rank:
            color = (
                f"  ·  color: tax/{self._tax_rank}"
                + (
                    f" ([/] · {len(self._available_ranks)})"
                    if len(self._available_ranks) > 1 else ""
                )
            )
        elif self._color_mode == _TEMP_MODE:
            color = "  ·  color: growth temp"
        elif self._color_mode == _IDENTITY_MODE:
            color = "  ·  color: %ID to target"
        elif self._color_mode == _TRAIT_MODE and self._trait_col:
            short = self._trait_col.removeprefix("trait_")
            color = (
                f"  ·  color: trait/{short}"
                + (
                    f" ([/] · {len(self._available_traits)})"
                    if len(self._available_traits) > 1 else ""
                )
            )
        elif self._color_mode != "off":
            color = f"  ·  color: {self._color_mode}"
        else:
            color = ""
        # Active model is shown in the sidebar dropdown now; the title
        # just notes the model name for at-a-glance context. The old
        # "(m to switch · N)" hint was retired with the dropdown.
        model_seg = ""
        if self._active_model:
            short = self._active_model.split("/")[-1]
            model_seg = f"  ·  model: {short}"
        filter_seg = ""
        if self._filter_rank and self._filter_value:
            filter_seg = (
                f"  ·  [{self._filter_rank}: {self._filter_value}]"
            )
        # Source set lives at the front of the subtitle so it's the
        # first thing scanned. Switching homolog sets and re-entering
        # PCA visibly changes this string, which makes the
        # "embeddings come from THIS set" association obvious.
        active_set = self._project.active_set_name()
        self.app.sub_title = (
            f"PCA · {self._project.name}/{active_set} ({n} seqs)"
            f"{var}{dropped}{model_seg}{color}{filter_seg}"
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
            result_full = load_or_compute_pca_2d(pkls[0])
        except Exception as e:  # noqa: BLE001
            self._error = f"PCA failed: {e}"
            return

        if result_full is None:
            self._error = (
                "Need at least 2 sequences with valid embeddings for PCA."
            )
            return

        full_coords, full_var, all_seq_ids, full_n_dropped = result_full

        # Full taxonomy — always loaded from the complete sequence set so
        # the filter modal always offers every available taxon value.
        full_tax_levels, full_tax_temps = self._load_taxonomy(all_seq_ids)
        self._full_tax_levels = full_tax_levels

        # If a taxon filter is active, recompute PCA on the matching subset.
        if self._filter_rank and self._filter_value:
            labels = full_tax_levels.get(
                self._filter_rank, [None] * len(all_seq_ids)
            )
            keep_ids = {
                sid for sid, lbl in zip(all_seq_ids, labels)
                if lbl == self._filter_value
            }
            filtered = self._run_pca_on_subset(pkls[0], keep_ids)
            if filtered is None:
                self._error = (
                    f"Too few sequences in '{self._filter_value}' "
                    f"({len(keep_ids)} matching) for PCA."
                )
                return
            coords, var_ratio, seq_ids, n_dropped = filtered
            # Taxonomy for the filtered subset only.
            self._tax_levels, self._tax_temps = self._load_taxonomy(seq_ids)
        else:
            coords, var_ratio, seq_ids, n_dropped = (
                full_coords, full_var, list(all_seq_ids), full_n_dropped
            )
            self._tax_levels = full_tax_levels
            self._tax_temps = full_tax_temps

        self._pcs = coords
        self._var_ratio = var_ratio
        self._n_dropped = n_dropped
        self._seq_ids = list(seq_ids)
        self._available_ranks = [
            rank for rank in _RANK_ORDER if rank in self._tax_levels
        ]
        if self._available_ranks and self._tax_rank not in self._available_ranks:
            self._tax_rank = self._available_ranks[0]
        # %ID to target — derived from the active set's alignment if it's
        # on disk. None per row when the homolog isn't aligned.
        self._identity = self._load_identity(seq_ids)
        # Trait coloring — pulls every populated column out of the
        # active set's traits.parquet, classifies dtype, and stages a
        # per-row value array. Cycled by `[` / `]` while in trait mode.
        self._trait_data = self._load_traits(seq_ids)
        self._available_traits = list(self._trait_data.keys())
        if self._available_traits and self._trait_col not in self._available_traits:
            self._trait_col = self._available_traits[0]

        # Top-level mode cycle: off → tax → growth_temp → identity →
        # trait → off. Skips modes whose data isn't populated so `c`
        # never lands on a useless mode.
        modes = ["off"]
        if self._available_ranks:
            modes.append("tax")
        if any(t is not None for t in self._tax_temps):
            modes.append(_TEMP_MODE)
        if any(v is not None for v in self._identity):
            modes.append(_IDENTITY_MODE)
        if self._available_traits:
            modes.append(_TRAIT_MODE)
        self._color_modes = modes

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

    def _load_taxonomy(
        self, seq_ids: List[str],
    ) -> Tuple[Dict[str, List[Optional[str]]], List[Optional[float]]]:
        """Build per-rank label arrays + a parallel growth-temp array.

        Returns `(rank_labels, temps)`:
        - `rank_labels` is a dict keyed by rank name ("domain", ...);
          only ranks with at least one non-null value across the
          project are included, so the cycle never offers a level
          with empty data.
        - `temps` is a per-sequence list of `Optional[float]` °C
          values from the Enqvist annotation, aligned to `seq_ids`.
          All-None when the column is absent or fully unmapped, in
          which case the caller skips the temperature mode.
        """
        empty_temps: List[Optional[float]] = [None] * len(seq_ids)
        tax_path = self._project.active_homologs_dir() / "taxonomy.parquet"
        if not tax_path.exists():
            return {}, empty_temps
        try:
            import pandas as pd
            tax = pd.read_parquet(tax_path)
        except Exception:
            return {}, empty_temps

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

        # Growth-temperature lookup, same dual-key pattern.
        temp_lookup: Dict[str, float] = {}
        if "growth_temp" in tax.columns:
            for _, row in tax.iterrows():
                sid = row.get("sequence_id")
                uid = row.get("uniprot_id")
                val = row.get("growth_temp")
                if val is None or (isinstance(val, float) and np.isnan(val)):
                    continue
                try:
                    f = float(val)
                except (TypeError, ValueError):
                    continue
                for k in (sid, uid):
                    if k and isinstance(k, str):
                        temp_lookup.setdefault(k, f)

        def resolve(seq_id: str, table):
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

        temps: List[Optional[float]] = (
            [resolve(s, temp_lookup) for s in seq_ids]
            if temp_lookup else empty_temps
        )
        return out, temps

    def _load_identity(self, seq_ids: List[str]) -> List[Optional[float]]:
        """Per-sequence % identity to the target, derived from alignment.

        Returns an aligned list (same length as `seq_ids`) where each
        entry is the % of non-gap target columns at which the homolog
        agrees with the target. None for the target itself, for
        sequences not present in the alignment, and when the alignment
        file is missing entirely.

        Cheap: O(n_seqs * n_target_residues) char comparisons, fully
        vectorised through numpy. A 5k × 500 alignment runs in well
        under a second.
        """
        empty: List[Optional[float]] = [None] * len(seq_ids)
        aln_path = self._project.active_homologs_dir() / "alignment.fasta"
        if not aln_path.exists():
            return empty
        try:
            from Bio import SeqIO
            records = list(SeqIO.parse(str(aln_path), "fasta"))
        except Exception:
            return empty
        if len(records) < 2:
            return empty

        target_id = records[0].id
        target_seq = str(records[0].seq).upper()
        n = len(target_seq)
        # Mask of target columns that hold a real residue (non-gap);
        # %ID is computed only over these positions so terminal /
        # internal gap regions in the target don't dilute the score.
        target_mask = np.array(
            [c != "-" and c != "." for c in target_seq], dtype=bool,
        )
        n_target_resid = int(target_mask.sum())
        if n_target_resid == 0:
            return empty

        # Build an id → identity-fraction lookup once.
        target_arr = np.array(list(target_seq))
        ident_by_id: Dict[str, float] = {}
        for rec in records[1:]:
            seq = str(rec.seq).upper()
            if len(seq) != n:
                continue  # malformed alignment row — skip
            row = np.array(list(seq))
            matches = (row == target_arr) & target_mask
            ident_by_id[rec.id] = float(matches.sum()) / n_target_resid * 100.0

        def resolve(seq_id: str) -> Optional[float]:
            if seq_id == target_id:
                return None
            if seq_id in ident_by_id:
                return ident_by_id[seq_id]
            for pfx in ("UniRef90_", "UniRef100_", "UniRef50_", "sp|", "tr|"):
                if seq_id.startswith(pfx):
                    candidate = seq_id[len(pfx):].split("|", 1)[0].split(" ", 1)[0]
                    if candidate in ident_by_id:
                        return ident_by_id[candidate]
            return None

        return [resolve(s) for s in seq_ids]

    def _load_traits(
        self, seq_ids: List[str],
    ) -> Dict[str, Dict]:
        """Per-trait data classified by dtype.

        Returns a dict keyed by trait column name (most-populated
        first). Each value carries:
            kind:        "binary" | "numeric" | "categorical"
            values:      List aligned to `seq_ids` (Optional[bool|float|str])
            min, max:    floats — numeric only
            categories:  List[str] of values, freq-ordered — categorical only

        Coverage-ordered so the first cycle position is the trait the
        user is most likely to want. Empty when no traits.parquet is
        present or no column has any non-null value across the active
        set.
        """
        traits_path = self._project.active_homologs_dir() / "traits.parquet"
        if not traits_path.exists():
            return {}
        try:
            import pandas as pd
            df = pd.read_parquet(traits_path)
        except Exception:
            return {}
        cols = [
            c for c in df.columns
            if c.startswith("trait_") and c != "trait_match_level"
        ]
        if not cols:
            return {}

        # Build the dual-key sequence_id / uniprot_id lookup once for
        # value resolution — same pattern as the taxonomy loader.
        def make_lookup(col):
            lookup: Dict[str, object] = {}
            for _, row in df.iterrows():
                sid = row.get("sequence_id")
                uid = row.get("uniprot_id")
                v = row.get(col)
                if v is None or (isinstance(v, float) and np.isnan(v)):
                    continue
                for k in (sid, uid):
                    if k and isinstance(k, str):
                        lookup.setdefault(k, v)
            return lookup

        def resolve(seq_id: str, lookup: Dict[str, object]):
            if seq_id in lookup:
                return lookup[seq_id]
            for pfx in ("UniRef90_", "UniRef100_", "UniRef50_", "sp|", "tr|"):
                if seq_id.startswith(pfx):
                    candidate = seq_id[len(pfx):].split("|", 1)[0].split(" ", 1)[0]
                    if candidate in lookup:
                        return lookup[candidate]
            return None

        # Coverage-rank columns and classify dtype per column.
        ordered: List[Tuple[str, int]] = []
        for c in cols:
            cov = int(df[c].notna().sum())
            if cov > 0:
                ordered.append((c, cov))
        ordered.sort(key=lambda kv: kv[1], reverse=True)

        out: Dict[str, Dict] = {}
        for col, _cov in ordered:
            lookup = make_lookup(col)
            raw_values = [resolve(s, lookup) for s in seq_ids]

            kind, normalized, extras = _classify_trait(raw_values)
            if kind is None:
                continue
            entry = {"kind": kind, "values": normalized}
            entry.update(extras)
            out[col] = entry
        return out

    def _color_for_idx(self, i: int) -> str:
        if i == self._target_idx:
            return _TARGET_COLOR
        if self._color_mode == "off":
            return _DEFAULT_COLOR

        if self._color_mode == _TEMP_MODE:
            t = self._tax_temps[i] if i < len(self._tax_temps) else None
            if t is None:
                return _TEMP_NONE_COLOR
            from .taxonomy_view import _temp_color
            return _temp_color(float(t))

        if self._color_mode == _IDENTITY_MODE:
            v = self._identity[i] if i < len(self._identity) else None
            if v is None:
                return _TEMP_NONE_COLOR
            # Reuse the temp gradient — same cool→warm shape works
            # nicely for an identity axis (low identity = cool / far,
            # high identity = warm / near). Map [0, 100] %ID into the
            # gradient's natural [0, 105] domain by passing through.
            from .taxonomy_view import _temp_color
            return _temp_color(float(v))

        if self._color_mode == _TRAIT_MODE:
            entry = self._trait_data.get(self._trait_col or "")
            if entry is None:
                return _DEFAULT_COLOR
            kind = entry["kind"]
            values = entry["values"]
            v = values[i] if i < len(values) else None
            if v is None:
                return _TEMP_NONE_COLOR
            if kind == "binary":
                return _TRAIT_TRUE_COLOR if v else _TRAIT_FALSE_COLOR
            if kind == "numeric":
                lo = entry.get("min", 0.0)
                hi = entry.get("max", 1.0)
                # Project the [min, max] range into the temp gradient's
                # 0–105 domain so each trait spans the full palette
                # regardless of its native scale.
                span = max(hi - lo, 1e-9)
                scaled = (float(v) - lo) / span * 105.0
                from .taxonomy_view import _temp_color
                return _temp_color(scaled)
            # categorical
            cache_key = f"trait:{self._trait_col}"
            cache = self._color_cache.setdefault(cache_key, {})
            label = str(v)
            if label not in cache:
                cache[label] = _PALETTE[len(cache) % len(_PALETTE)]
            return cache[label]

        # Tax mode — color by the currently selected rank.
        if self._color_mode != "tax" or not self._tax_rank:
            return _DEFAULT_COLOR
        labels = self._tax_levels.get(self._tax_rank) or []
        label = labels[i] if i < len(labels) else None
        if not label:
            return _UNKNOWN_COLOR
        cache_key = f"tax:{self._tax_rank}"
        cache = self._color_cache.setdefault(cache_key, {})
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

        Strategy: ask `_color_for_idx` for each sampled point's color
        (it handles every mode — tax / growth_temp / identity / trait /
        off — including the rank-within-tax cycle), then group by cell
        and pick the most-common color per cell. Ties resolve
        deterministically via lexsort on (cell, -count).

        Delegating to `_color_for_idx` means new color modes don't
        need a parallel implementation here — they just need the
        existing per-point hook to know about them.
        """
        if self._color_mode == "off":
            # Single default color — cheap first-occurrence-per-cell.
            _, first = np.unique(flat, return_index=True)
            return row[first], col[first], [_DEFAULT_COLOR] * len(first)

        # Per-point color via the mode-aware hook. This is the line that
        # was previously missing for `tax` (since the lookup keyed off
        # `_color_mode == "tax"` instead of `_tax_rank`) and for the
        # continuous / trait modes.
        per_point_colors = np.array(
            [self._color_for_idx(int(i)) for i in orig_idx],
            dtype=object,
        )

        # Encode colors as integer codes so we can fold them into the
        # cell-id key for a single np.unique pass that groups by cell
        # *and* counts color occurrences in one shot.
        unique_colors, color_codes = np.unique(
            per_point_colors, return_inverse=True
        )
        n_colors = len(unique_colors)
        if n_colors == 0:
            _, first = np.unique(flat, return_index=True)
            return row[first], col[first], [_DEFAULT_COLOR] * len(first)

        combined = flat * n_colors + color_codes.astype(np.int64)
        combo_unique, counts = np.unique(combined, return_counts=True)
        combo_cells = combo_unique // n_colors
        combo_codes = combo_unique % n_colors

        # Sort by (cell ascending, count descending). The first row per
        # unique cell after this sort is the modal (cell, color) pair.
        order = np.lexsort([-counts, combo_cells])
        sorted_cells = combo_cells[order]
        sorted_codes = combo_codes[order]
        _, first_per_cell = np.unique(sorted_cells, return_index=True)
        mode_cells = sorted_cells[first_per_cell]
        mode_codes = sorted_codes[first_per_cell]

        rows_out = (mode_cells // plot_w).astype(np.int32)
        cols_out = (mode_cells % plot_w).astype(np.int32)
        colors_out = [str(unique_colors[int(c)]) for c in mode_codes]
        return rows_out, cols_out, colors_out

    # ---- rendering ----

    def _render_scatter(self, width: int, height: int) -> str:
        if self._pcs is None or len(self._pcs) == 0:
            return ""

        plot_w = max(20, width - _AXIS_W)
        # Legend lives in the sidebar now (`#pca-legend`), so the plot
        # body keeps its full vertical span minus only the x-axis row.
        # The earlier bottom-legend strip ate 3 rows per render; moving
        # it out gave back ~10 % of the canvas on a 30-row terminal.
        plot_h = max(8, height - _AXIS_H)

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
        # Stash the projected coords for hover lookup. The plot starts
        # at column `_AXIS_W` from the body's left edge (the y-axis
        # gutter); record that origin so the hover handler can offset
        # mouse coords correctly. Only one ruler row sits above the
        # plot — the body's top is the plot's top.
        self._hover_cols = col.copy()
        self._hover_rows = row.copy()
        self._hover_orig_idx = orig_idx.copy()
        self._hover_origin = (_AXIS_W, 0)

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
        # the mode color says. Painted with an 8-neighbour halo of
        # small dots in the target colour so the marker reads as a
        # ~3x3 starburst — much easier to find on a crowded plot
        # without bumping the grid resolution. Halo cells only
        # overwrite empty cells (`None`); homologs that happen to land
        # next to the target stay visible.
        if self._target_idx is not None and 0 <= self._target_idx < n_total:
            tx = float(xs[self._target_idx])
            ty = float(ys[self._target_idx])
            tc = int((tx - x_min) / x_range * (plot_w - 1))
            tr = int((y_max - ty) / y_range * (plot_h - 1))
            for dr in (-1, 0, 1):
                for dc in (-1, 0, 1):
                    if dr == 0 and dc == 0:
                        continue
                    rr = tr + dr
                    cc = tc + dc
                    if (
                        0 <= rr < plot_h
                        and 0 <= cc < plot_w
                        and grid[rr][cc] is None
                    ):
                        grid[rr][cc] = (_TARGET_COLOR, "halo")
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
                color, marker = cell
                if marker is True:
                    # Center of the target marker — bold + filled star.
                    line_parts.append(f"[bold {color}]★[/bold {color}]")
                elif marker == "halo":
                    # Cardinal / diagonal halo cells around the target.
                    # Small middle-dot in the same color — visible but
                    # unobtrusive, builds a 3x3 starburst around the ★.
                    line_parts.append(f"[{color}]·[/{color}]")
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
        # Legend was previously appended here; now it's rendered into
        # the sidebar's `#pca-legend` Static (see `_refresh_legend`).
        return "\n".join(lines)

    def _on_hover(self, x: int, y: int, screen_x: int, screen_y: int) -> None:
        """Translate a body-local mouse position to a sequence id and
        pop a floating callout next to it.

        ``x, y`` are widget-content cell coords (used for the data
        lookup); ``screen_x, screen_y`` are absolute screen coords
        (used to position the tooltip widget via its `offset` style).
        Snap radius is Manhattan ≤1: cursor must be on a dot or in
        the four-neighbour cells around it. Anything wider made the
        tooltip pop in genuinely empty regions of the plot adjacent
        to clusters — a 3-cell radius bridges to the next dot column
        too easily on dense scatters.
        """
        if (
            self._hover_cols is None or len(self._hover_cols) == 0
            or self._pcs is None
        ):
            return
        ox, oy = self._hover_origin
        col = x - ox
        row = y - oy
        if col < 0 or row < 0:
            self._on_hover_leave()
            return
        d = np.abs(self._hover_cols - col) + np.abs(self._hover_rows - row)
        nearest_local = int(np.argmin(d))
        if d[nearest_local] > 1:
            self._on_hover_leave()
            return
        seq_idx = int(self._hover_orig_idx[nearest_local])
        if seq_idx == self._last_hover_idx:
            return
        self._last_hover_idx = seq_idx
        self._show_hover_tooltip(seq_idx, screen_x, screen_y)
        # Mirror the same info into the title bar — useful when the
        # callout overlaps the actual point and you want a more stable
        # reference at the top of the screen.
        self._update_hover_subtitle(seq_idx)

    def _on_hover_leave(self) -> None:
        if self._last_hover_idx is None:
            return
        self._last_hover_idx = None
        # Restore the standard sub_title (project / set / model).
        self._update_title()
        # Hide the floating callout.
        try:
            tip = self.query_one("#pca-hover-tooltip", Static)
            tip.display = False
        except Exception:
            pass

    def _show_hover_tooltip(
        self, seq_idx: int, screen_x: int, screen_y: int,
    ) -> None:
        """Render the per-point callout and position it next to the
        cursor.

        Layout — a vertical "spine" with one cell swapped for an arrow
        on the row aligned with the point:

            │ sp|P07311|ACYP1_HUMAN
            │ domain: Eukaryota
            ◄ 37 °C
            │ 100% id

        The arrow row is the geometric middle of the rendered text;
        the tooltip's screen position is then chosen so that
        arrow-row sits at `screen_y`, with the tooltip to the right
        of `screen_x` (a 2-cell gap so the arrow doesn't overlap the
        point itself). Falls back to placing the tooltip to the LEFT
        of the cursor when it would otherwise spill past the body's
        right edge — keeps it visible on points near the sidebar.
        """
        if seq_idx < 0 or seq_idx >= len(self._seq_ids):
            return
        try:
            tip = self.query_one("#pca-hover-tooltip", Static)
        except Exception:
            return

        # Build the text lines (no markup beyond simple coloring on
        # the seq_id header — the spine chars are added below).
        seq_id = self._seq_ids[seq_idx]
        body_lines: List[str] = [f"[bold]{seq_id}[/bold]"]
        if self._tax_rank and self._tax_rank in self._tax_levels:
            v = self._tax_levels[self._tax_rank][seq_idx]
            if v:
                body_lines.append(f"[dim]{self._tax_rank}:[/dim] {v}")
        if (
            seq_idx < len(self._tax_temps)
            and self._tax_temps[seq_idx] is not None
        ):
            body_lines.append(
                f"[dim]temp:[/dim] {self._tax_temps[seq_idx]:.0f} °C"
            )
        if (
            seq_idx < len(self._identity)
            and self._identity[seq_idx] is not None
        ):
            body_lines.append(
                f"[dim]%ID:[/dim] {self._identity[seq_idx]:.0f}%"
            )
        if (
            self._color_mode == _TRAIT_MODE and self._trait_col
            and self._trait_col in self._trait_data
        ):
            entry = self._trait_data[self._trait_col]
            values = entry.get("values") or []
            if seq_idx < len(values) and values[seq_idx] is not None:
                short = self._trait_col.removeprefix("trait_")
                body_lines.append(f"[dim]{short}:[/dim] {values[seq_idx]}")

        # Spine: one `│` per body row, with the middle row swapped to
        # an arrow. `◄` points at the cursor (which is to the LEFT of
        # the tooltip). Cyan blue ties it visually to the panel border.
        n = len(body_lines)
        arrow_row = n // 2
        spine_color = "#2E86AB"
        rendered_lines = []
        for i, content in enumerate(body_lines):
            glyph = "◄" if i == arrow_row else "│"
            rendered_lines.append(f"[{spine_color}]{glyph}[/{spine_color}] {content}")
        tip.update("\n".join(rendered_lines))

        # Position: arrow row aligned to cursor y; tooltip starts a
        # couple of cells right of the cursor. If that would overflow
        # the screen, flip to the LEFT of the cursor (and use `►` for
        # the arrow instead — but we'd have rendered `◄` already, so
        # for now we just clamp position; arrow direction stays the
        # same to keep this routine simple).
        screen_w = self.app.size.width if self.app else 200
        # Estimated tooltip width. Auto-width Static doesn't expose its
        # natural size before render, so we approximate from longest
        # body line. Ample padding prevents off-screen clipping.
        from rich.text import Text as _RT
        tip_w = max(
            _RT.from_markup(line).cell_len for line in rendered_lines
        ) + 2  # +2 padding
        tip_x = screen_x + 2
        if tip_x + tip_w > screen_w:
            tip_x = max(0, screen_x - tip_w - 2)
        tip_y = max(0, screen_y - arrow_row)
        tip.styles.offset = (tip_x, tip_y)
        tip.display = True

    def _update_hover_subtitle(self, seq_idx: int) -> None:
        """Build a one-liner about the hovered sequence and set
        `app.sub_title` to it. Includes whatever the active color mode
        encodes — taxonomy at the chosen rank, growth temp, %ID, or
        trait — so the user gets context for the colored point under
        the cursor.
        """
        if seq_idx < 0 or seq_idx >= len(self._seq_ids):
            return
        seq_id = self._seq_ids[seq_idx]
        bits = [seq_id]
        # Taxonomy at the active rank, when populated.
        if self._tax_rank and self._tax_rank in self._tax_levels:
            v = self._tax_levels[self._tax_rank][seq_idx]
            if v:
                bits.append(f"{self._tax_rank}: {v}")
        # Growth temp.
        if seq_idx < len(self._tax_temps) and self._tax_temps[seq_idx] is not None:
            bits.append(f"{self._tax_temps[seq_idx]:.0f} °C")
        # %ID to target.
        if seq_idx < len(self._identity) and self._identity[seq_idx] is not None:
            bits.append(f"{self._identity[seq_idx]:.0f}% id")
        # Active trait if any.
        if (
            self._color_mode == _TRAIT_MODE and self._trait_col
            and self._trait_col in self._trait_data
        ):
            entry = self._trait_data[self._trait_col]
            values = entry.get("values") or []
            if seq_idx < len(values) and values[seq_idx] is not None:
                short = self._trait_col.removeprefix("trait_")
                bits.append(f"{short}: {values[seq_idx]}")
        self.app.sub_title = "  ·  ".join(bits)

    def _refresh_legend(self) -> None:
        """Update the sidebar's #pca-legend Static.

        The color cache for tax mode is normally populated lazily by
        `_color_for_idx` during render. The legend used to live inside
        the render output, so it always saw a populated cache; with
        the legend in the sidebar the cache might be empty when this
        runs (the body's render is queued, not synchronous). Prime
        the cache here so the chip dots show their actual scatter
        colors instead of all-gray fallbacks.
        """
        try:
            target = self.query_one("#pca-legend", Static)
        except Exception:
            return
        self._prime_color_cache_for_legend()
        # `_legend_text` joins chips with two spaces, not `·`, so we
        # render its output verbatim — the sidebar wraps wide enough
        # for typical chip lengths and Rich handles line wrapping.
        text = self._legend_text()
        mode_label = {
            "off": "no coloring",
            "tax": f"taxonomy · {self._tax_rank or 'rank'}",
            _TEMP_MODE: "growth temp (°C)",
            _IDENTITY_MODE: "% identity to target",
            _TRAIT_MODE: f"trait · {self._trait_col or '?'}",
        }.get(self._color_mode, self._color_mode)
        header = f"[bold]{mode_label}[/bold]"
        target.update(header + "\n\n" + text.lstrip())

    def _prime_color_cache_for_legend(self) -> None:
        """Pre-fill the color cache so legend chips read the same
        palette the scatter will paint.

        Tax mode: assign palette colors to the top-N most frequent
        labels of the active rank. Ordering matches what
        `_color_for_idx` would produce on a left-to-right walk —
        first-seen-first-color — but using the *frequency* ordering
        the legend itself uses, so the most-common group always lands
        on the first palette slot regardless of where it sits in the
        FASTA.

        Trait mode: `_legend_text` already primes its own cache via
        the `cache[cat] = ...` write inside the categorical branch,
        so we only need to handle tax here.
        """
        if self._color_mode != "tax" or not self._tax_rank:
            return
        labels = self._tax_levels.get(self._tax_rank) or []
        if not labels:
            return
        cache_key = f"tax:{self._tax_rank}"
        cache = self._color_cache.setdefault(cache_key, {})
        # Frequency-ordered (matches the order chips appear in the
        # legend, so the user can pair color → row by reading down).
        counts: Dict[str, int] = {}
        for lbl in labels:
            if lbl:
                counts[lbl] = counts.get(lbl, 0) + 1
        ranked = sorted(counts.items(), key=lambda kv: kv[1], reverse=True)
        for lbl, _ in ranked:
            if lbl not in cache:
                cache[lbl] = _PALETTE[len(cache) % len(_PALETTE)]

    def _legend_height(self) -> int:
        # A compact legend: one row for target/default, one row for the
        # taxonomy palette when active. Reserve 3 rows so the plot doesn't
        # jitter when the legend wraps to two lines.
        return 3

    def _legend_text(self) -> str:
        parts: List[str] = []
        if self._target_idx is not None:
            parts.append(
                f"[bold {_TARGET_COLOR}]★[/bold {_TARGET_COLOR}] [dim]target[/dim]"
            )

        if self._color_mode == _TEMP_MODE:
            # Continuous gradient legend: min/max + a small bar.
            temps = [t for t in self._tax_temps if t is not None]
            n_unk = len(self._tax_temps) - len(temps)
            if temps:
                from .taxonomy_view import _temp_color
                lo = min(temps)
                hi = max(temps)
                bar_cells = 14
                bar = "".join(
                    f"[{_temp_color(lo + (hi - lo) * (i / max(bar_cells - 1, 1)))}]"
                    f"█[/]"
                    for i in range(bar_cells)
                )
                parts.append(
                    f"[dim]temp[/dim] {bar} "
                    f"[dim]{lo:.0f}–{hi:.0f} °C  ({len(temps)})[/dim]"
                )
            if n_unk:
                parts.append(
                    f"[{_TEMP_NONE_COLOR}]●[/{_TEMP_NONE_COLOR}] "
                    f"[dim]no temp ({n_unk})[/dim]"
                )
        elif self._color_mode == _IDENTITY_MODE:
            ids = [v for v in self._identity if v is not None]
            n_unk = len(self._identity) - len(ids)
            if ids:
                from .taxonomy_view import _temp_color
                lo, hi = min(ids), max(ids)
                bar_cells = 14
                bar = "".join(
                    f"[{_temp_color(lo + (hi - lo) * (i / max(bar_cells - 1, 1)))}]█[/]"
                    for i in range(bar_cells)
                )
                parts.append(
                    f"[dim]%ID[/dim] {bar} "
                    f"[dim]{lo:.0f}–{hi:.0f}%  ({len(ids)})[/dim]"
                )
            if n_unk:
                parts.append(
                    f"[{_TEMP_NONE_COLOR}]●[/{_TEMP_NONE_COLOR}] "
                    f"[dim]unaligned ({n_unk})[/dim]"
                )
        elif self._color_mode == _TRAIT_MODE and self._trait_col:
            entry = self._trait_data.get(self._trait_col, {})
            kind = entry.get("kind")
            values = entry.get("values") or []
            n_unk = sum(1 for v in values if v is None)
            short = self._trait_col.removeprefix("trait_")
            if kind == "binary":
                n_true = sum(1 for v in values if v is True)
                n_false = sum(1 for v in values if v is False)
                parts.append(
                    f"[{_TRAIT_TRUE_COLOR}]●[/{_TRAIT_TRUE_COLOR}] "
                    f"[dim]{short}=true ({n_true})[/dim]"
                )
                parts.append(
                    f"[{_TRAIT_FALSE_COLOR}]●[/{_TRAIT_FALSE_COLOR}] "
                    f"[dim]{short}=false ({n_false})[/dim]"
                )
            elif kind == "numeric":
                from .taxonomy_view import _temp_color
                lo = entry.get("min", 0.0)
                hi = entry.get("max", 1.0)
                bar_cells = 14
                bar = "".join(
                    f"[{_temp_color(0 + 105.0 * (i / max(bar_cells - 1, 1)))}]█[/]"
                    for i in range(bar_cells)
                )
                parts.append(
                    f"[dim]{short}[/dim] {bar} "
                    f"[dim]{lo:g}–{hi:g}  ({len(values) - n_unk})[/dim]"
                )
            elif kind == "categorical":
                cache = self._color_cache.get(f"trait:{self._trait_col}", {})
                cats = entry.get("categories") or []
                shown = 0
                for cat in cats:
                    if shown >= 6:
                        break
                    color = cache.get(cat)
                    if color is None:
                        # Pre-prime so the legend shows the same color
                        # the scatter will render.
                        color = _PALETTE[len(cache) % len(_PALETTE)]
                        cache[cat] = color
                        self._color_cache[f"trait:{self._trait_col}"] = cache
                    n = sum(1 for v in values if v == cat)
                    parts.append(
                        f"[{color}]●[/{color}] [dim]{cat} ({n})[/dim]"
                    )
                    shown += 1
                if len(cats) > shown:
                    parts.append(f"[dim]+{len(cats) - shown} more[/dim]")
            if n_unk:
                parts.append(
                    f"[{_TEMP_NONE_COLOR}]●[/{_TEMP_NONE_COLOR}] "
                    f"[dim]no value ({n_unk})[/dim]"
                )
        elif self._color_mode == "tax" and self._tax_rank:
            cache = self._color_cache.get(f"tax:{self._tax_rank}", {})
            labels = self._tax_levels.get(self._tax_rank) or []
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
        else:
            parts.append(
                f"[{_DEFAULT_COLOR}]●[/{_DEFAULT_COLOR}] [dim]hits[/dim]"
            )

        # One chip per line so the user can read down the legend like
        # a list — at narrow ranks (genus/species) eight chips on one
        # wrapped line is unreadable, and the sidebar already reserves
        # vertical space for it (`#pca-legend { height: auto }`).
        return "\n".join(parts)
