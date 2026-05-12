"""Vertical alignment viewer.

Shows every homolog sequence stacked: identifier on the left, sequence
to the right. Reads `homologs/alignment.fasta` if present (so columns
line up); otherwise reads `homologs/sequences.fasta` (unaligned, every
sequence left-justified).

Color modes (top dropdown):
    biochem       — Clustal-X-style palette: hydrophobic / charged /
                    polar / aromatic / special (G, P, C)
    conservation  — per-column identity fraction → white→red gradient
    none          — plain text

Keys: ↑↓ scroll rows, ←→ pan along sequence, PgUp/PgDn jump rows,
home/end jump to first/last column.
"""

from collections import Counter
from typing import Dict, List, Optional, Tuple

import numpy as np
from rich.text import Text
from textual import work
from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal
from textual.screen import Screen
from textual.widgets import Footer, Header, Static
from ..widgets.colorbar import Colorbar


_ID_COL_WIDTH = 22
_PAN_STEP = 10
_PAGE_ROWS = 10
# Width of the per-row identity bar drawn to the right of the %ID text.
_BAR_W = 10
# Cells reserved at the right of each row for the per-sequence
# %identity-to-target suffix: 2-space gutter + 5-char number with one
# decimal + `%` + 1 separator + _BAR_W bar chars.
_IDENT_SUFFIX_W = 9 + _BAR_W
# Cells of top padding inside `_AlignmentBody` (mirrors the CSS
# `padding: 1 2` declaration). Used by the click-handler to translate
# event.y from the widget's outer-box origin to content-row origin.
_BODY_PADDING_TOP = 1
_NEFF_MAX_PAIRS_SEQS = 600  # cap O(N^2) work; subsample beyond this
_LOGO_HEIGHT = 2            # rows in the logo: 1 bar row + 1 letter row
_BAR_GLYPHS = " ▁▂▃▄▅▆▇█"   # 8 levels for the bar row

# Phylum palette — stable color per taxonomic group, picked by hash so
# the same phylum gets the same hue across runs.
_PHYLUM_PALETTE = [
    "#FF6B6B", "#FFD93D", "#6BCF7F", "#4D96FF", "#A66DD4",
    "#FF9F45", "#3DCFD4", "#FF6B9D", "#7DD87D", "#65CBF3",
    "#FFA62B", "#9D7DD8",
]


def _identity_bar_color(pct: float) -> str:
    if pct < 30:
        return "#FF6B6B"
    if pct < 60:
        return "#FFD93D"
    return "#6BCF7F"


def _stack_seqs_to_bytes(
    sequences: List[Tuple[str, str]],
    max_len: Optional[int] = None,
) -> np.ndarray:
    """Stack ``[(id, seq), …]`` into an ``(N, L)`` ``uint8`` matrix.

    Sequences are padded on the right with ``-`` to ``max_len`` (or
    the longest input if not given). Used by every per-column
    aggregation in this module — vectorising once into ASCII byte
    codes lets the per-column statistics avoid an N×L Python loop.

    Non-ASCII / non-printable characters in the input are replaced
    with ``?`` (ord 63) by `bytes.encode("ascii", "replace")`. They
    won't match any standard amino-acid letter, so they fall through
    the gap-or-real distinction harmlessly.
    """
    if not sequences:
        return np.zeros((0, 0), dtype=np.uint8)
    if max_len is None:
        max_len = max(len(s) for _, s in sequences)
    n = len(sequences)
    if max_len == 0:
        return np.zeros((n, 0), dtype=np.uint8)
    # Build one contiguous buffer then frombuffer in a single call —
    # ~4× faster than per-row frombuffer + slice-assign on 20k seqs.
    raw = b"".join(
        s.upper().encode("ascii", "replace")[:max_len].ljust(max_len, b"-")
        for _, s in sequences
    )
    return np.frombuffer(raw, dtype=np.uint8).reshape(n, max_len).copy()


# Pre-computed alphabet bytes shared by the column-statistics loops.
_AA_BYTES = b"ACDEFGHIKLMNPQRSTVWY"
_GAP_CODES = (ord("-"), ord("."))


def _identity_per_row(
    sequences: List[Tuple[str, str]],
    arr: Optional[np.ndarray] = None,
) -> List[Optional[float]]:
    """Per-row %identity to the target row (sequences[0]).

    Returns one entry per element of `sequences`, aligned by index.
    `None` for the target itself (identity-to-self is meaningless to
    surface) and for rows with zero ungapped overlap. Values for
    homolog rows are 0–100.

    Identity is "matches over positions where both target and homolog
    have a non-gap residue", case-insensitive — matches the convention
    `SequenceDetailModal` uses so the per-row column on the alignment
    view and the per-pair stat in the modal agree to the decimal.

    Vectorised: the previous Python double-loop over (rows × target
    positions) was O(N·L) with constant-time inner work; for a 20k×500
    alignment that's ~5 s of Python overhead. The numpy version is
    one broadcast and a sum, ~30 ms on the same input.

    Pass ``arr`` (pre-built uint8 matrix) to skip the encode step when
    the caller already has the array from a prior operation.
    """
    if not sequences:
        return []
    n = len(sequences)
    if arr is None:
        arr = _stack_seqs_to_bytes(sequences)
    target = arr[0]
    target_valid = (target != _GAP_CODES[0]) & (target != _GAP_CODES[1])
    if not target_valid.any():
        return [None] * n
    rows_valid = (arr != _GAP_CODES[0]) & (arr != _GAP_CODES[1])
    both = target_valid[None, :] & rows_valid
    n_compared = both.sum(axis=1)
    n_match = ((arr == target[None, :]) & both).sum(axis=1)
    pct = np.where(n_compared > 0, n_match / np.maximum(n_compared, 1) * 100.0, 0.0)
    out: List[Optional[float]] = []
    for ridx in range(n):
        if ridx == 0 or n_compared[ridx] == 0:
            out.append(None)
        else:
            out.append(float(pct[ridx]))
    return out


def _mean_identity_to_target(sequences: List[Tuple[str, str]]) -> Optional[float]:
    """Mean per-residue identity of each homolog to the target.

    Identity is computed over target positions where both target and
    homolog have a non-gap residue. Returns percentage (0-100) or None.

    Vectorised: shares the uint8-stacking helper with
    `_identity_per_row` so the same N×L equality / valid-mask pair
    isn't recomputed twice on alignment load. The previous pure-Python
    triple-loop was the second-slowest part of the alignment-load
    pipeline (after Neff) on 16k-sequence MSAs.
    """
    if len(sequences) < 2:
        return None
    arr = _stack_seqs_to_bytes(sequences)
    if arr.size == 0:
        return None
    target = arr[0]
    target_valid = (target != _GAP_CODES[0]) & (target != _GAP_CODES[1])
    if not target_valid.any():
        return None
    homologs = arr[1:]
    rows_valid = (homologs != _GAP_CODES[0]) & (homologs != _GAP_CODES[1])
    both = target_valid[None, :] & rows_valid
    n_compared = both.sum(axis=1)
    n_match = ((homologs == target[None, :]) & both).sum(axis=1)
    # Per-row identity (skip rows with zero overlap), averaged.
    valid_rows = n_compared > 0
    if not valid_rows.any():
        return None
    pct = n_match[valid_rows] / n_compared[valid_rows]
    return float(pct.mean()) * 100.0


def _effective_n(
    sequences: List[Tuple[str, str]], threshold: float = 0.8
) -> Optional[int]:
    """Count of identity-based clusters — proxy for effective N.

    Single-link clustering at `threshold` pairwise identity. For large
    alignments we subsample to keep this O(N²) work bounded.
    """
    if not sequences:
        return None
    seqs = [s for _, s in sequences]
    if len(seqs) > _NEFF_MAX_PAIRS_SEQS:
        # Deterministic stride sample so the result is reproducible.
        step = len(seqs) // _NEFF_MAX_PAIRS_SEQS + 1
        seqs = seqs[::step]
    n = len(seqs)
    if n < 2:
        return n
    parent = list(range(n))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: int, b: int) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb

    arr = np.array([list(s) for s in seqs])
    gap_mask = (arr == "-") | (arr == ".")
    valid = ~gap_mask
    for i in range(n):
        ai, vi = arr[i], valid[i]
        for j in range(i + 1, n):
            both = vi & valid[j]
            denom = both.sum()
            if denom == 0:
                continue
            same = ((arr[j] == ai) & both).sum()
            if same / denom >= threshold:
                union(i, j)
    roots = {find(k) for k in range(n)}
    return len(roots)


# Clustal-X-flavored palette by amino-acid biochemistry.
_BIOCHEM_COLORS: Dict[str, str] = {
    # Hydrophobic
    "A": "#65CBF3", "I": "#65CBF3", "L": "#65CBF3", "M": "#65CBF3",
    "F": "#65CBF3", "W": "#65CBF3", "V": "#65CBF3",
    # Positive
    "K": "#FF6B6B", "R": "#FF6B6B",
    # Negative
    "D": "#A66DD4", "E": "#A66DD4",
    # Polar
    "N": "#7DD87D", "Q": "#7DD87D", "S": "#7DD87D", "T": "#7DD87D",
    # Cysteine
    "C": "#FFA62B",
    # Glycine, Proline (special backbone)
    "G": "#FFD93D", "P": "#FF7D45",
    # Aromatic basic / hydroxyl
    "H": "#3DCFD4", "Y": "#3DCFD4",
}


class _AlignmentBody(Static):
    """Inner Static that renders the visible window of the alignment."""

    def __init__(self, parent_view: "AlignmentViewerScreen", **kwargs) -> None:
        super().__init__("", **kwargs)
        self._pv = parent_view

    def render(self):
        return self._pv._render_body(self.size.width, self.size.height)

    def on_resize(self, event) -> None:
        self.refresh()

    def on_click(self, event) -> None:
        # Translate the click's local y-coord to a `_displayed` index
        # via the parent's geometry helper, then ask the parent to open
        # the row-detail modal. Clicks that land on the ruler / logo /
        # blank rows resolve to None and quietly no-op.
        idx = self._pv._row_idx_at_body_y(event.y)
        if idx is None:
            return
        self._pv._highlight_idx = idx
        self._pv._open_detail_for(idx)
        event.stop()


class AlignmentViewerScreen(Screen):
    """Full-screen alignment view."""

    # Skip auto-focus: the only focusable descendant is the Colorbar,
    # whose left/right bindings (midpoint shift) shadow this screen's
    # left/right pan bindings whenever it has focus. Same fix
    # ProjectDetailScreen uses (see detail.py).
    AUTO_FOCUS = ""

    BINDINGS = [
        Binding("escape",   "back",        "Back"),
        Binding("space",    "cycle_color", "Color"),
        Binding("g",        "cycle_gap",   "Ungap"),
        # Conservation midpoint shift — mirrors the structure/sequence
        # panel's left/right arrow on the inline colorbar, but those
        # keys are already pan here, so use brackets. (Textual's
        # binding parser splits on bare commas, hence `left_square_bracket`
        # / `right_square_bracket` rather than the literal characters.)
        Binding("left_square_bracket",  "midpoint_down", "Mid -"),
        Binding("right_square_bracket", "midpoint_up",   "Mid +"),
        Binding("up",       "row_up",      "Row ↑",  show=False),
        Binding("down",     "row_down",    "Row ↓",  show=False),
        Binding("left",     "pan_left",    "Pan ←",  show=False),
        Binding("right",    "pan_right",   "Pan →",  show=False),
        Binding("pageup",   "page_up",     "PgUp",   show=False),
        Binding("pagedown", "page_down",   "PgDn",   show=False),
        Binding("home",     "pan_home",    "Home",   show=False),
        Binding("end",      "pan_end",     "End",    show=False),
        # Row selection (vim-flavoured) + Enter to open the row's detail
        # modal. Click also opens the modal without needing to highlight
        # first; the highlight is only there for keyboard-only workflow.
        Binding("j",        "highlight_down", "Sel ↓", show=False),
        Binding("k",        "highlight_up",   "Sel ↑", show=False),
        Binding("enter",    "open_detail",    "Detail"),
    ]

    _MIDPOINT_STEP = 5.0
    _MIDPOINT_MIN = 5.0
    _MIDPOINT_MAX = 95.0

    DEFAULT_CSS = """
    AlignmentViewerScreen _AlignmentBody { height: 1fr; padding: 1 2; }
    /* Pin the phylum legend + status to the bottom of the viewport so
       it stays visible no matter how many homolog rows scroll past. */
    AlignmentViewerScreen #aln-legend {
        dock: bottom;
        height: auto;
        padding: 0 2 1 2;
        background: $surface;
        border-top: solid #2E86AB;
    }
    AlignmentViewerScreen #aln-phylum { width: 1fr; }
    AlignmentViewerScreen #aln-colorbar { display: none; height: 2; }
    """

    # `plddt` and `sasa` color each column by the *target's* per-residue
    # score (looked up via `_col_to_target_pos`); columns where the
    # target has a gap stay uncolored. The structure view's gradients
    # are reused via `color_for_mode` so a residue keeps its color
    # whether you're looking at the structure ribbon or the MSA row.
    _COLOR_MODES = ("biochem", "conservation", "plddt", "sasa", "none")
    # None = no ungap. Others = drop columns where gap fraction >= threshold.
    _GAP_THRESHOLDS = (None, 0.9, 0.8, 0.7)

    def __init__(self, project) -> None:
        super().__init__()
        self._project = project
        # Original parsed sequences (target row at index 0).
        self._sequences: List[Tuple[str, str]] = []
        # Active view: same as _sequences when not ungapped, otherwise filtered.
        self._displayed: List[Tuple[str, str]] = []
        # Loading flag drives the body's placeholder text — `True` while
        # `_load_async` is parsing / building per-column caches in the
        # background, `False` once the alignment is fully populated.
        self._loading: bool = True
        self._aligned: bool = False
        self._row_offset: int = 0
        self._col_offset: int = 0
        # Highlighted row for keyboard navigation. Index into
        # `_displayed` (so 0 = target). None until the user first
        # presses j / k or clicks a row. Click also sets it so the
        # selection is sticky after a mouse open.
        self._highlight_idx: Optional[int] = None
        self._color_mode: str = "biochem"
        self._gap_threshold: Optional[float] = 0.9
        # Midpoint for conservation coloring. Persisted in
        # `[view] aln_midpoint` so a user who shifted it on one project
        # doesn't have to re-shift on the next open. Only meaningful in
        # `conservation` mode — biochem ignores it.
        prefs = (project.manifest().get("view") or {})
        try:
            mid = float(prefs.get("aln_midpoint", 50.0))
        except (TypeError, ValueError):
            mid = 50.0
        self._midpoint: float = max(
            self._MIDPOINT_MIN, min(self._MIDPOINT_MAX, mid)
        )
        # Per-column conservation aligned to _displayed. Recomputed on ungap.
        self._col_conservation: Optional[np.ndarray] = None
        # Per-row %identity to target, aligned to `_displayed` indices.
        # `[0]` is the target row (always None — identity-to-self is
        # 100 by definition and adds no information). Recomputed in
        # `_rebuild_displayed`.
        self._row_identity_pct: List[Optional[float]] = []
        # Target's per-residue pLDDT / SASA loaded from the AlphaFold
        # CIF, plus a column→target-residue index map so the alignment
        # cells can look up the score for whichever target residue
        # sits at each column. All three are recomputed in `_load`
        # (they depend on which target row is in `_displayed` after
        # ungap filtering doesn't change which residue is at each col,
        # only conservation).
        self._target_plddt: Optional[np.ndarray] = None
        self._target_sasa: Optional[np.ndarray] = None
        self._col_to_target_pos: Optional[np.ndarray] = None
        # Per-column logo cache aligned to _displayed. Recomputed on
        # ungap; consumed cheaply by `_logo_lines_text` so scroll/pan
        # don't repeat O(n_seqs * n_visible_cols) work.
        self._logo_top: List[str] = []
        self._logo_level: List[int] = []
        self._logo_freq: List[float] = []
        # Diversity metrics — computed once at load.
        self._mean_identity: Optional[float] = None
        self._neff: Optional[int] = None
        # Taxonomy: seq_id → phylum (None when unresolved).
        self._tax_by_id: dict = {}
        self._phylum_legend: list = []  # ordered (phylum, color, count)
        # Pre-cached parquet rows for the sequence-detail modal — keyed
        # on `sequence_id`. Populated by `_warm_modal_caches` after
        # mount so the first modal open doesn't pay parquet read cost.
        # `None` = warm not finished (modal will fall back to its own
        # disk read); `{}` = warm finished but file missing or empty.
        self._tax_rows_by_id: Optional[dict] = None
        self._traits_rows_by_id: Optional[dict] = None
        # Close-flag: set in `action_back` before the heavy state
        # cleanup so any thread workers in flight (the modal-warm
        # parquet loader) bail early instead of doing more parquet
        # reads + dict builds that the now-popped screen will never
        # use. Pure short-circuit; the existing try/except already
        # prevents crashes.
        #
        # NB: this attribute MUST NOT be named ``_closing`` — that
        # name collides with ``MessagePump._closing`` (Textual's
        # internal flag). Setting ``self._closing = True`` makes
        # ``post_message`` return False silently for every subsequent
        # message, which prevents Prune from being delivered when the
        # screen is popped, which leaves the screen's pump alive
        # forever, which wedges ``screen.remove() → do_pop`` and
        # freezes the entire app. Symptoms: input dies after Esc,
        # mouse clicks ignored, structure pane visually frozen mid-
        # rotation, but watchdog reports loop-alive. Cost us a long
        # debugging session in May 2026 — keep the underscore prefix
        # but never the bare name.
        self._user_closing: bool = False
        # Per-row colored ``Text`` cache. Vertical scroll (j/k, page,
        # row offset) doesn't change any individual row's content — only
        # which rows are visible — so caching at this granularity makes
        # vertical scroll a pure dict-lookup over visible indices.
        # Invalidated whenever ``_row_cache_key`` would change (column
        # offset, viewport width, color mode, midpoint).
        self._row_cache: Dict[int, Text] = {}
        self._row_cache_key: Optional[tuple] = None

    def compose(self) -> ComposeResult:
        yield Header()
        yield _AlignmentBody(self, id="aln-body")
        with Horizontal(id="aln-legend"):
            yield Static("", id="aln-phylum")
            yield Colorbar(id="aln-colorbar")
        yield Footer()

    def on_mount(self) -> None:
        # Two-stage load: parse the FASTA synchronously (the parquet
        # cache makes this tens of ms even for 20k-sequence MSAs), so
        # the body paints with all sequences on first frame; then kick
        # off a worker for the heavier post-parse derivations (per-
        # column logo + conservation, per-row identity, mean ID,
        # Neff (O(N²) inside the 600-seq cap), taxonomy parquet,
        # target-structure scalars). Cold-cache projects (very first
        # open) pay the FASTA parse on the UI thread once — that's a
        # one-time cost; all subsequent opens go through the cache.
        homologs_dir = self._project.active_homologs_dir()
        target_seq = self._project.target_sequence() or ""
        target_id = (
            self._project.manifest().get("target", {}).get("uniprot_id")
            or self._project.name
        )

        aln = homologs_dir / "alignment.fasta"
        hits = homologs_dir / "sequences.fasta"
        path = aln if aln.exists() else (hits if hits.exists() else None)
        self._aligned = path == aln

        records: List[Tuple[str, str]] = []
        if path is not None:
            try:
                from ...alignments.cache import load_alignment_records
                records = load_alignment_records(path)
            except Exception:
                records = []

        if records:
            self._sequences = records
            if self._gap_threshold is not None and self._aligned:
                self._displayed, _ = self._ungap(records, self._gap_threshold)
            else:
                self._displayed = list(records)
        elif target_seq:
            self._sequences = [(target_id, target_seq)]
            self._displayed = list(self._sequences)
        self._loading = False  # body has real content now

        self._update_title()
        try:
            self.query_one("#aln-phylum", Static).update(self._render_legend())
        except Exception:
            pass
        try:
            cb = self.query_one("#aln-colorbar", Colorbar)
            cb.set_mode("conservation")
            cb.set_midpoint(self._midpoint)
            cb.display = (self._color_mode == "conservation")
        except Exception:
            pass
        # Pre-load BLOSUM62 + parquet rows in the background so the
        # first sequence-detail modal open doesn't pay parquet read +
        # Biopython init on the UI thread (~300-500 ms total otherwise).
        self._warm_modal_caches()
        # Worker handles the slow post-parse derivations (column logo
        # cache, per-row identity, conservation, Neff, taxonomy,
        # structure scalars). Body re-renders as each phase lands.
        if records:
            self._populate_caches_async(target_seq)

    @work(thread=True, exclusive=True, group="modal-warm")
    def _warm_modal_caches(self) -> None:
        # Bail early at every stage if the screen is closing —
        # parquet reads + dict builds the popped screen will never
        # use are pure waste. The check between stages is enough; we
        # don't need it inside the inner loops because the parquet
        # read is the slow part.
        if self._user_closing:
            return
        # 1. BLOSUM62 (~50-500 ms first call).
        try:
            from .sequence_detail import _blosum62_score
            _blosum62_score("A", "A")
        except Exception:
            pass
        if self._user_closing:
            return

        # 2. Taxonomy + traits parquets, transposed to seq_id → row
        # dict for O(1) lookup from the modal. The DataFrames live in
        # memory as long as the alignment screen does, so we pay this
        # once per project view rather than once per modal open.
        try:
            import pandas as pd
            hd = self._project.active_homologs_dir()
            tax_path = hd / "taxonomy.parquet"
            if tax_path.exists():
                df = pd.read_parquet(tax_path)
                if "sequence_id" in df.columns:
                    self._tax_rows_by_id = {
                        str(r["sequence_id"]): r.to_dict()
                        for _, r in df.iterrows()
                    }
                else:
                    self._tax_rows_by_id = {}
            else:
                self._tax_rows_by_id = {}
        except Exception:
            self._tax_rows_by_id = {}
        if self._user_closing:
            return

        try:
            import pandas as pd
            hd = self._project.active_homologs_dir()
            tr_path = hd / "traits.parquet"
            if tr_path.exists():
                df = pd.read_parquet(tr_path)
                if "sequence_id" in df.columns:
                    # Drop columns that are entirely null up front —
                    # `traits.parquet` typically has 2k+ columns and
                    # most are sparsely populated.
                    df = df.dropna(axis=1, how="all")
                    self._traits_rows_by_id = {
                        str(r["sequence_id"]): r.to_dict()
                        for _, r in df.iterrows()
                    }
                else:
                    self._traits_rows_by_id = {}
            else:
                self._traits_rows_by_id = {}
        except Exception:
            self._traits_rows_by_id = {}

    def _update_title(self) -> None:
        self.app.title = "beak"
        n = len(self._sequences)
        kind = "alignment" if self._aligned else "sequences"
        gap = (
            f"  ·  ungap≥{self._gap_threshold:.1f}"
            if self._gap_threshold is not None else ""
        )
        diversity = ""
        if self._mean_identity is not None:
            diversity += f"  ·  ident {self._mean_identity:.0f}%"
        if self._neff is not None:
            diversity += f"  ·  Neff@80 {self._neff}"
        # Show the midpoint only in conservation mode — it has no
        # effect in biochem / none, so surfacing it elsewhere is noise.
        color_str = self._color_mode
        if self._color_mode == "conservation":
            color_str += f" mid={int(self._midpoint)}"
        self.app.sub_title = (
            f"{kind} · {self._project.name} ({n})  ·  "
            f"color: {color_str}{gap}{diversity}"
        )

    @work(thread=True, exclusive=True, group="aln-load")
    def _populate_caches_async(self, target_seq: str) -> None:
        """Background work that runs *after* the alignment is on screen.

        Two phases, both kept off the UI thread so cycling ungap or
        scrolling isn't blocked while the heavy stats settle:

        Phase A — per-column / per-row caches (logo, conservation,
                  identity-per-row). All vectorised so even a
                  20k-sequence MSA finishes in well under 100 ms.
        Phase B — diversity stats (mean identity, Neff (O(N²) within
                  the 600-seq cap)), taxonomy table, target-structure
                  scalars. Title + legend re-render once they land.

        ALL UI-thread hops use ``_schedule_ui`` (non-blocking) instead
        of ``self.app.call_from_thread`` (blocking). The blocking
        variant deadlocks against screen teardown: if the worker is
        in ``call_from_thread`` when the user pops the screen via Esc,
        the asyncio loop is busy in ``do_pop → _replace_screen →
        screen.remove() → gather(child._task)``, and the worker thread
        is parked on ``future.result()`` waiting for the callback to
        run. While interleaving normally lets the callback run, screen
        teardown holds widget references that the worker thread blocks
        from being released — alignment's pump never terminates,
        ``screen.remove()`` never returns, and the App's pump task
        wedges. ``loop.call_soon_threadsafe`` schedules the callback
        without blocking the worker, so the worker thread can return
        even if the callback never runs (it's a no-op anyway when the
        screen is closing).
        """
        # ---- Phase A: per-column / per-row caches ----
        # Build the uint8 matrix once; share it across all three helpers
        # to avoid three separate GIL-holding encode loops (~200 ms on
        # a 20k-seq alignment).
        _phase_a_arr: Optional[np.ndarray] = None
        if self._displayed:
            try:
                _phase_a_arr = _stack_seqs_to_bytes(self._displayed)
            except Exception:
                pass
        try:
            # Single pass for conservation + logo (same 20-AA count matrix).
            self._rebuild_column_caches(arr=_phase_a_arr)
        except Exception:
            self._col_conservation = None
        try:
            self._row_identity_pct = (
                _identity_per_row(self._displayed, arr=_phase_a_arr)
                if self._displayed else []
            )
        except Exception:
            self._row_identity_pct = []
        _phase_a_arr = None  # release early; the large arr is no longer needed
        self._schedule_ui(self._refresh_body)

        # ---- Phase B: diversity stats + taxonomy + structure scalars ----
        try:
            self._mean_identity = _mean_identity_to_target(self._sequences)
        except Exception:
            self._mean_identity = None
        try:
            self._neff = _effective_n(self._sequences, threshold=0.8)
        except Exception:
            self._neff = None
        self._load_taxonomy()
        self._load_target_structure_scalars(target_seq)
        self._schedule_ui(self._refresh_legend)
        self._schedule_ui(self._refresh_body)

    def _schedule_ui(self, callback) -> None:
        """Non-blocking variant of ``app.call_from_thread``.

        Schedules ``callback`` on the asyncio loop without blocking
        the worker thread. Critical for any UI-update hop from a
        worker that runs while the screen might be torn down — see
        ``_populate_caches_async`` docstring for the deadlock vector.
        """
        try:
            loop = self.app._loop
            if loop is not None:
                loop.call_soon_threadsafe(callback)
        except Exception:  # noqa: BLE001
            pass

    def _load_taxonomy(self) -> None:
        """Load homologs/taxonomy.parquet and build (seq_id → phylum) +
        a counted legend ordered by frequency."""
        tax_path = self._project.active_homologs_dir() / "taxonomy.parquet"
        if not tax_path.exists():
            return
        try:
            import pandas as pd
            df = pd.read_parquet(tax_path)
        except Exception:
            return
        if "sequence_id" not in df or "phylum" not in df:
            return
        self._tax_by_id = dict(
            zip(df["sequence_id"].astype(str), df["phylum"].astype(object))
        )
        # Match seq_ids in our alignment to the table — UniProt-derived ids
        # may not match the alignment's record ids exactly, so try a few
        # accession-extraction passes.
        from ..taxonomy import _accession_from_seq_id
        if self._sequences:
            for seq_id, _ in self._sequences:
                if seq_id in self._tax_by_id:
                    continue
                acc = _accession_from_seq_id(seq_id)
                if acc:
                    # Match by accession-suffix
                    for tid, phy in self._tax_by_id.items():
                        if tid.endswith(acc) or acc in tid:
                            self._tax_by_id[seq_id] = phy
                            break

        # Legend: top phyla by count, in the alignment.
        from collections import Counter
        present = [
            self._tax_by_id.get(sid)
            for sid, _ in self._sequences
            if self._tax_by_id.get(sid)
        ]
        counts = Counter(p for p in present if p)
        self._phylum_legend = [
            (p, _PHYLUM_PALETTE[hash(p) % len(_PHYLUM_PALETTE)], n)
            for p, n in counts.most_common(8)
        ]

    def _phylum_color(self, phylum) -> Optional[str]:
        if not phylum:
            return None
        return _PHYLUM_PALETTE[hash(phylum) % len(_PHYLUM_PALETTE)]

    def _load_target_structure_scalars(self, target_seq: str) -> None:
        """Pull the target's per-residue pLDDT + SASA off disk, plus the
        column→target-residue index map needed to project them back
        onto the alignment.

        Silently leaves the scalars as None when no AlphaFold CIF is
        cached locally — `plddt` and `sasa` color modes then act like
        `none` (nothing to look up). The same loaders the structure
        view uses are reused so a residue keeps its color whether you
        look at the ribbon or the MSA row.
        """
        target_uniprot = (
            (self._project.manifest().get("target") or {}).get("uniprot_id")
        )
        if not target_uniprot:
            return
        # Use whatever cached structure exists — prefers PDB, falls
        # through to AlphaFold. Either source has a per-CA B-factor
        # column (pLDDT for AF, crystallographic B for PDB) that
        # `load_ca_coords` reads via `ca.b_iso`, so the same scalar
        # plumbing works for both.
        from ..structure import cached_structure_path
        cif_path = cached_structure_path(
            target_uniprot, self._project.path / "structures",
        )
        if cif_path is None:
            return

        try:
            from ..structure import load_ca_coords, load_sasa
            _, plddt = load_ca_coords(cif_path)
            if len(plddt) == len(target_seq):
                self._target_plddt = np.asarray(plddt, dtype=np.float32)
            sasa = load_sasa(cif_path, len(target_seq))
            if sasa is not None and len(sasa) == len(target_seq):
                self._target_sasa = np.asarray(sasa, dtype=np.float32)
        except Exception:
            # Defensive — if the CIF is malformed we still want the
            # other color modes to work.
            return

        # PDB structures often have partial coverage so the length check
        # above fails for pLDDT. Fall back to the AlphaFold CIF for
        # pLDDT scores when the primary structure is a PDB hit — AF's
        # B-factor column carries real pLDDT confidence values and always
        # covers the full sequence.
        if self._target_plddt is None:
            af_cif = (
                self._project.path / "structures"
                / f"{target_uniprot}_AF.cif"
            )
            if af_cif.exists():
                try:
                    from ..structure import load_ca_coords
                    _, plddt_af = load_ca_coords(af_cif)
                    if len(plddt_af) == len(target_seq):
                        self._target_plddt = np.asarray(
                            plddt_af, dtype=np.float32
                        )
                except Exception:
                    pass

        self._col_to_target_pos = self._build_col_to_target_pos(target_seq)

    def _build_col_to_target_pos(
        self, target_seq: str
    ) -> Optional[np.ndarray]:
        """Return an int array of length n_cols mapping each alignment
        column to a 0-based residue index in the target sequence, or
        -1 where the target has a gap.

        Identifies the target row by ungapped-sequence equality with
        `target_seq` rather than assuming row 0 — third-party tools
        sometimes reorder MSAs and we'd otherwise color homolog
        positions with the target's pLDDT.
        """
        # Use _displayed (the current ungapped view) so column indices in
        # the rendered output match the col_map. Building from _sequences
        # (original alignment) produces an N_original-length map, but
        # rendering passes ungapped column indices 0..N_ungapped-1 —
        # these hit the wrong (often all-gap) original columns and return
        # pos=-1 for every cell, silently dropping all pLDDT/SASA color.
        sequences = self._displayed or self._sequences
        if not sequences:
            return None
        target_upper = target_seq.upper()
        target_aln: Optional[str] = None
        for _, seq in sequences:
            if seq.upper().replace("-", "").replace(".", "") == target_upper:
                target_aln = seq
                break
        if target_aln is None:
            # Fall back to row 0 — the convention beak's own pipeline
            # follows. Better than refusing to color when an external
            # tool kept the target first but didn't match exactly
            # (e.g. case folding by an aligner).
            target_aln = sequences[0][1]

        n_cols = len(target_aln)
        out = np.full(n_cols, -1, dtype=np.int32)
        target_pos = 0
        for col, ch in enumerate(target_aln):
            if ch in "-.":
                continue
            if target_pos < len(target_seq):
                out[col] = target_pos
            target_pos += 1
        return out

    def _rebuild_displayed(self) -> None:
        """Refresh `_displayed` (and column conservation) from `_sequences`
        and the current gap threshold. Cheap when no ungap is active."""
        if self._gap_threshold is None or not self._aligned:
            self._displayed = list(self._sequences)
            # Build the array once; reused by all three stat helpers below.
            arr: np.ndarray = (
                _stack_seqs_to_bytes(self._sequences) if self._sequences
                else np.zeros((0, 0), dtype=np.uint8)
            )
        else:
            self._displayed, arr = self._ungap(self._sequences, self._gap_threshold)
        # `_displayed` changed under us — drop the per-row Text cache
        # so the next paint rebuilds rows from the new column space.
        # Without this, ungap appears to no-op because the body keeps
        # serving Texts built from the pre-ungap `_displayed`.
        self._row_cache = {}
        self._row_cache_key = None
        # Single combined pass for conservation + logo: both need the
        # same 20-AA count matrix — computing it once halves that cost.
        self._rebuild_column_caches(arr=arr)
        # Per-row %identity to target — precomputed once per
        # `_displayed` change so the per-frame paint is just a list
        # lookup, not an O(L) loop. Cost: O(N·L) on rebuild, ~100 ms
        # for 25k rows × 1000 cols. Aligned to `_displayed` indices
        # (so [0] is target, None there).
        self._row_identity_pct: List[Optional[float]] = (
            _identity_per_row(self._displayed, arr=arr) if self._displayed else []
        )
        # Rebuild the column→target mapping so pLDDT/SASA colors use
        # ungapped column indices, not the original alignment's columns.
        if self._target_plddt is not None or self._target_sasa is not None:
            self._col_to_target_pos = self._build_col_to_target_pos(
                self._project.target_sequence() or ""
            )

    def _rebuild_column_caches(self, arr: Optional[np.ndarray] = None) -> None:
        """Recompute conservation + logo caches in a single 20-AA pass.

        Both ``_col_conservation`` and the logo arrays need the same
        ``(20, L)`` per-AA count matrix — computing it once halves the
        cost versus calling the two helpers independently.

        Pass ``arr`` (pre-built uint8 matrix for ``_displayed``) to also
        skip the encode step.
        """
        if not self._displayed:
            self._col_conservation = None if not self._aligned else self._col_conservation
            self._logo_top = []
            self._logo_level = []
            self._logo_freq = []
            return
        if arr is None:
            arr = _stack_seqs_to_bytes(self._displayed)
        n_seq, cols = arr.shape
        if cols == 0:
            self._col_conservation = (
                np.zeros(0, dtype=np.float32) if self._aligned else None
            )
            self._logo_top = []
            self._logo_level = []
            self._logo_freq = []
            return

        gap_mask = (arr == _GAP_CODES[0]) | (arr == _GAP_CODES[1])
        n_valid = (~gap_mask).sum(axis=0)

        # Single 20-AA pass shared by conservation and logo.
        counts = np.zeros((len(_AA_BYTES), cols), dtype=np.int32)
        for i, aa in enumerate(_AA_BYTES):
            counts[i] = (arr == aa).sum(axis=0)
        top_idx = counts.argmax(axis=0)
        top_count = counts.max(axis=0)
        freqs_arr = np.where(
            n_valid > 0, top_count.astype(np.float32) / np.maximum(n_valid, 1), 0.0,
        )

        if self._aligned:
            self._col_conservation = np.where(
                n_valid > 0,
                top_count.astype(np.float32) / np.maximum(n_valid, 1) * 100.0,
                0.0,
            ).astype(np.float32)
        else:
            self._col_conservation = None

        n_glyphs = len(_BAR_GLYPHS)
        levels_arr = np.minimum(
            n_glyphs - 1,
            np.round(freqs_arr * (n_glyphs - 1)).astype(int),
        )
        # `top` falls back to space for all-gap columns so the logo row
        # paints blank glyphs there rather than smuggling an arbitrary
        # AA through with `freq=0`.
        self._logo_top = [
            chr(int(_AA_BYTES[int(top_idx[c])])) if n_valid[c] > 0 else " "
            for c in range(cols)
        ]
        self._logo_level = levels_arr.tolist()
        self._logo_freq = freqs_arr.tolist()

    def _rebuild_logo_cache(self, arr: Optional[np.ndarray] = None) -> None:
        """Precompute (top-char, intensity-level) per alignment column.

        Without this the logo + consensus rows walk the full sequence
        list and run Counter on every column on every paint — which is
        the dominant cost when scrolling, even for a few-hundred-row
        alignment. Building it once per `_displayed` change collapses
        the per-frame work to a slice + markup join.

        Pass ``arr`` (pre-built uint8 matrix for ``_displayed``) to skip
        the encode step when the caller already holds the array.
        """
        if not self._displayed:
            self._logo_top: List[str] = []
            self._logo_level: List[int] = []
            self._logo_freq: List[float] = []
            return

        # Fully vectorised: stack into uint8, then for each of the 20
        # standard AAs run a single `(arr == aa).sum(axis=0)` to get
        # its per-column count. The dominant AA per column is just
        # `argmax` across the (20, L) count table; its frequency
        # divides by the per-column non-gap total. Replaces the prior
        # per-column Python loop with Counter — roughly 100× faster on
        # a 20k×500 alignment (~700 ms → ~7 ms).
        if arr is None:
            arr = _stack_seqs_to_bytes(self._displayed)
        n_seq, cols = arr.shape

        gap_mask = (arr == _GAP_CODES[0]) | (arr == _GAP_CODES[1])
        n_valid = (~gap_mask).sum(axis=0)

        counts = np.zeros((len(_AA_BYTES), cols), dtype=np.int32)
        for i, aa in enumerate(_AA_BYTES):
            counts[i] = (arr == aa).sum(axis=0)
        top_idx = counts.argmax(axis=0)
        top_count = counts.max(axis=0)
        freqs_arr = np.where(
            n_valid > 0, top_count.astype(np.float32) / np.maximum(n_valid, 1), 0.0,
        )

        n_glyphs = len(_BAR_GLYPHS)
        levels_arr = np.minimum(
            n_glyphs - 1,
            np.round(freqs_arr * (n_glyphs - 1)).astype(int),
        )

        # `top` falls back to space for all-gap columns so the logo row
        # paints blank glyphs there rather than smuggling an arbitrary
        # AA through with `freq=0`.
        top: List[str] = [
            chr(int(_AA_BYTES[int(top_idx[c])])) if n_valid[c] > 0 else " "
            for c in range(cols)
        ]
        self._logo_top = top
        self._logo_level = levels_arr.tolist()
        self._logo_freq = freqs_arr.tolist()

    def _ungap(
        self,
        sequences: List[Tuple[str, str]],
        threshold: float,
    ) -> Tuple[List[Tuple[str, str]], np.ndarray]:
        """Drop columns where gap-fraction ≥ `threshold`. Vectorised.

        Returns ``(filtered_sequences, filtered_arr)`` so callers can
        reuse the uint8 matrix for downstream stats without re-encoding.

        Encodes once into a uint8 ``(N, L)`` matrix via the shared
        helper (which is ~10× faster than building an N-length list
        of `list(s.ljust(...))` then handing it to ``np.array``), then
        slices and decodes the kept columns in one pass per row.
        """
        if not sequences:
            return [], np.zeros((0, 0), dtype=np.uint8)
        arr = _stack_seqs_to_bytes(sequences)
        n_seq, max_len = arr.shape
        empty = [(ident, "") for ident, _ in sequences]
        if max_len == 0:
            return empty, arr
        gap_mask = (arr == _GAP_CODES[0]) | (arr == _GAP_CODES[1])
        gap_frac = gap_mask.sum(axis=0) / max(n_seq, 1)
        keep = gap_frac < threshold
        if not keep.any():
            return empty, arr[:, :0]
        kept = arr[:, keep]
        seqs = [
            (ident, kept[i].tobytes().decode("ascii", "replace"))
            for i, (ident, _) in enumerate(sequences)
        ]
        return seqs, kept

    @staticmethod
    def _compute_column_conservation(
        seqs: List[Tuple[str, str]],
        arr: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Per-column identity fraction in [0, 100], skipping gap-only cols.

        Vectorised across all standard 20 AAs: for each AA, compute
        an ``(N, L)`` boolean equality and reduce on axis 0. The
        previous per-column Python+Counter version was the dominant
        cost when cycling ungap on a 20k-sequence MSA; this finishes
        in tens of milliseconds.

        Pass ``arr`` (pre-built uint8 matrix) to skip the encode step
        when the caller already holds the array.
        """
        if arr is None:
            if not seqs:
                return np.zeros(0, dtype=np.float32)
            arr = _stack_seqs_to_bytes(seqs)
        n_seq, max_len = arr.shape
        if max_len == 0:
            return np.zeros(0, dtype=np.float32)
        gap_mask = (arr == _GAP_CODES[0]) | (arr == _GAP_CODES[1])
        n_valid = (~gap_mask).sum(axis=0)
        # Per-AA per-column counts; max across AAs gives the dominant
        # residue's count. 20 numpy comparisons + a single max — the
        # whole table is reduced in one pass per AA.
        counts = np.zeros((len(_AA_BYTES), max_len), dtype=np.int32)
        for i, aa in enumerate(_AA_BYTES):
            counts[i] = (arr == aa).sum(axis=0)
        top = counts.max(axis=0)
        return np.where(
            n_valid > 0, top.astype(np.float32) / np.maximum(n_valid, 1) * 100.0, 0.0,
        ).astype(np.float32)

    # ---- key actions ----

    def action_back(self) -> None:
        # Mark closing first so any in-flight worker (the parquet
        # warm) bails at its next checkpoint instead of finishing a
        # read into a dict the popped screen will never consume.
        self._user_closing = True
        self.app.log.info("alignment_view.action_back: starting cleanup")
        # Cancel the worker groups owned by this screen so their
        # `app.call_from_thread(self._refresh_body)` calls don't land
        # on the main thread minutes after pop_screen — those refreshes
        # can mark unmounted widgets dirty and (we've observed) leave
        # the parent detail screen unable to repaint key handlers.
        try:
            self.workers.cancel_group(self, "aln-load")
            self.workers.cancel_group(self, "modal-warm")
            self.workers.cancel_group(self, "aln-rebuild")
        except Exception as e:  # noqa: BLE001
            self.app.log.warning(
                f"alignment_view.action_back: cancel_group failed: {e!r}"
            )
        # Drop heavy in-memory state before popping. The screen
        # accumulates a few non-trivial caches over its lifetime:
        #   * `_row_cache`: per-row Rich `Text` objects (colored
        #     residues + ID prefix + %ID suffix). On a 7k×3.8k MSA
        #     each row is hundreds of style spans, so the cache can
        #     hold tens of thousands of segment objects.
        #   * `_tax_rows_by_id` / `_traits_rows_by_id`: dicts of
        #     parquet rows pre-warmed for the sequence-detail modal.
        #     `traits.parquet` can be 2k+ columns wide, so the dict
        #     has ~7k × 2k entries on a big project.
        #   * `_displayed`, `_sequences`: parsed FASTA + alignment.
        # Letting these GC during the screen-pop animation made the
        # exit feel sluggish (Python freeing ~250k objects mid-frame
        # is hundreds of ms). Clearing them synchronously here turns
        # exit into the cheap path: the references drop, GC reclaims
        # them in the next idle moment, and `pop_screen` returns
        # immediately.
        self._row_cache.clear()
        self._row_cache_key = None
        self._tax_rows_by_id = None
        self._traits_rows_by_id = None
        self._displayed = []
        self._sequences = []
        self._logo_top = []
        self._logo_level = []
        self._logo_freq = []
        self._row_identity_pct = []
        self._tax_by_id = {}
        # Clear the alignment-specific sub-title so the project detail
        # screen doesn't inherit "alignment · ..." after we pop back.
        self.app.sub_title = ""
        self.app.log.info("alignment_view.action_back: about to pop_screen")
        self.app.pop_screen()
        self.app.log.info("alignment_view.action_back: pop_screen returned")

    def _max_row_offset(self) -> int:
        # Target stays pinned; offset paginates rows[1:].
        return max(0, len(self._displayed) - 2)

    def action_row_up(self) -> None:
        self._row_offset = max(0, self._row_offset - 1)
        self._refresh_body()

    def action_row_down(self) -> None:
        self._row_offset = min(self._max_row_offset(), self._row_offset + 1)
        self._refresh_body()

    # ---- row selection ----

    def action_highlight_up(self) -> None:
        n = len(self._displayed)
        if n == 0:
            return
        cur = self._highlight_idx if self._highlight_idx is not None else 1
        # 0 is the target row — keyboard selection should reach it too,
        # so users can pull up the target's own taxonomy/structure modal
        # the same way as a homolog row.
        self._highlight_idx = max(0, cur - 1)
        self._scroll_highlight_into_view()
        self._refresh_body()

    def action_highlight_down(self) -> None:
        n = len(self._displayed)
        if n == 0:
            return
        cur = self._highlight_idx if self._highlight_idx is not None else 0
        self._highlight_idx = min(n - 1, cur + 1)
        self._scroll_highlight_into_view()
        self._refresh_body()

    def action_open_detail(self) -> None:
        """Open the row-detail modal for the currently highlighted row,
        defaulting to the first homolog if nothing is selected yet."""
        if not self._displayed:
            return
        if self._highlight_idx is None:
            self._highlight_idx = 1 if len(self._displayed) > 1 else 0
        self._open_detail_for(self._highlight_idx)

    def _scroll_highlight_into_view(self) -> None:
        """If the highlight has moved off the visible homolog window,
        nudge `_row_offset` so it's just inside the viewport again.
        The target row is always pinned, so `idx == 0` is always
        visible and needs no scroll math."""
        idx = self._highlight_idx
        if idx is None or idx <= 0:
            return
        # Highlight position within the `others` list.
        homolog_idx = idx - 1
        try:
            visible_rows = self._visible_homolog_rows()
        except Exception:
            visible_rows = 8
        if homolog_idx < self._row_offset:
            self._row_offset = homolog_idx
        elif homolog_idx >= self._row_offset + visible_rows:
            self._row_offset = max(0, homolog_idx - visible_rows + 1)

    def _visible_homolog_rows(self) -> int:
        """Approximate count of homolog rows that fit in the body.

        The body's height includes the ruler (1) + blank (1) + logo
        rows (2) + target (1) + blank (1) + status (2 with leading
        blank), so subtract those. Used by the auto-scroll path; not
        load-bearing — overestimating just means slightly aggressive
        scroll, never a crash.
        """
        try:
            body = self.query_one(_AlignmentBody)
            return max(1, body.size.height - 8)
        except Exception:
            return 8

    def _row_idx_at_body_y(self, y: int) -> Optional[int]:
        """Translate a click y-coord (in body-local cells) into an
        index into `_displayed`, or None if the click landed on
        ruler / logo / blank / status / out of range.

        The widget has `padding: 1 2` (see `DEFAULT_CSS`), so Textual
        reports clicks against the widget's outer box — meaning the
        visible ruler is at event.y == 1, not 0. Subtract the top
        padding before mapping so a click lands on the row the user
        actually pointed at instead of the row below.

        Content layout after padding correction:
            0:    ruler
            1:    blank
            2-3:  logo (always 2 lines, even when no logo data)
            4:    target sequence
            5:    blank
            6+:   homolog rows starting at `_row_offset`
        """
        y = y - _BODY_PADDING_TOP
        if y < 0:
            return None
        if y == 4:
            return 0  # target
        if y < 6:
            return None
        homolog_idx = (y - 6) + self._row_offset
        others_len = max(0, len(self._displayed) - 1)
        if homolog_idx >= others_len:
            return None
        return homolog_idx + 1  # +1 to get index into _displayed

    def _open_detail_for(self, displayed_idx: int) -> None:
        if displayed_idx < 0 or displayed_idx >= len(self._displayed):
            return
        seq_id, aligned_seq = self._displayed[displayed_idx]
        target_id, target_aligned = self._displayed[0]
        from .sequence_detail import SequenceDetailModal
        # Hand the modal pre-warmed parquet rows (when the warm worker
        # has finished) so it can render its taxonomy / traits sections
        # without disk I/O. Pass `None` while the warm is still in
        # flight — the modal then falls back to its own read.
        self.app.push_screen(
            SequenceDetailModal(
                project=self._project,
                seq_id=seq_id,
                aligned_seq=aligned_seq,
                target_id=target_id,
                target_aligned=target_aligned,
                is_target=(displayed_idx == 0),
                tax_rows_by_id=self._tax_rows_by_id,
                traits_rows_by_id=self._traits_rows_by_id,
            ),
        )

    def action_page_up(self) -> None:
        self._row_offset = max(0, self._row_offset - _PAGE_ROWS)
        self._refresh_body()

    def action_page_down(self) -> None:
        self._row_offset = min(self._max_row_offset(), self._row_offset + _PAGE_ROWS)
        self._refresh_body()

    def action_pan_left(self) -> None:
        self._col_offset = max(0, self._col_offset - _PAN_STEP)
        self._refresh_body()

    def action_pan_right(self) -> None:
        self._col_offset += _PAN_STEP
        self._refresh_body()

    def action_pan_home(self) -> None:
        self._col_offset = 0
        self._refresh_body()

    def action_pan_end(self) -> None:
        if self._displayed:
            longest = max(len(s) for _, s in self._displayed)
            self._col_offset = max(0, longest - 40)
        self._refresh_body()

    def action_cycle_color(self) -> None:
        idx = self._COLOR_MODES.index(self._color_mode)
        self._color_mode = self._COLOR_MODES[(idx + 1) % len(self._COLOR_MODES)]
        self._update_title()
        self._refresh_body()
        try:
            self.query_one("#aln-colorbar", Colorbar).display = (
                self._color_mode == "conservation"
            )
        except Exception:
            pass

    def action_midpoint_down(self) -> None:
        self._shift_midpoint(-self._MIDPOINT_STEP)

    def action_midpoint_up(self) -> None:
        self._shift_midpoint(+self._MIDPOINT_STEP)

    def _shift_midpoint(self, delta: float) -> None:
        # No-op in any mode where the midpoint has no visual effect —
        # avoids the surprising case where the user mashes `,/.` while
        # in biochem mode and wonders why the title isn't changing.
        if self._color_mode != "conservation":
            try:
                self.notify(
                    "Midpoint only applies to conservation coloring "
                    "(press `space` to switch).",
                    timeout=3,
                )
            except Exception:
                pass
            return
        new_mid = max(
            self._MIDPOINT_MIN,
            min(self._MIDPOINT_MAX, self._midpoint + delta),
        )
        if new_mid == self._midpoint:
            return
        self._midpoint = new_mid
        try:
            with self._project.mutate() as m:
                m.setdefault("view", {})["aln_midpoint"] = new_mid
        except Exception:
            pass
        self._update_title()
        self._refresh_body()
        try:
            self.query_one("#aln-colorbar", Colorbar).set_midpoint(self._midpoint)
        except Exception:
            pass

    def on_colorbar_midpoint_changed(self, event: Colorbar.MidpointChanged) -> None:
        if event.midpoint == self._midpoint:
            return
        self._midpoint = event.midpoint
        try:
            with self._project.mutate() as m:
                m.setdefault("view", {})["aln_midpoint"] = self._midpoint
        except Exception:
            pass
        self._update_title()
        self._refresh_body()

    def action_cycle_gap(self) -> None:
        if not self._aligned:
            return  # ungap is meaningless on raw hits
        idx = self._GAP_THRESHOLDS.index(self._gap_threshold)
        self._gap_threshold = self._GAP_THRESHOLDS[(idx + 1) % len(self._GAP_THRESHOLDS)]
        # Reset offsets — column space changed.
        self._col_offset = 0
        # Title flips immediately so the user sees the threshold change;
        # the actual rebuild runs on a worker so the UI stays
        # responsive even on big alignments. The body shows the
        # previous (pre-ungap) view until the worker swaps in the new
        # one.
        self._update_title()
        self._rebuild_displayed_async()

    @work(thread=True, exclusive=True, group="aln-rebuild")
    def _rebuild_displayed_async(self) -> None:
        self._rebuild_displayed()
        self._schedule_ui(self._refresh_body)

    def _refresh_body(self) -> None:
        # Body is the only thing that changes on scroll; the phylum
        # legend is dock-pinned and only depends on the loaded
        # taxonomy, not on viewport offsets. Skip updating it here —
        # `_load` and any taxonomy-changing path can call
        # `_refresh_legend` explicitly when needed.
        # Bail when the screen has been popped — workers that landed on
        # the main thread after `action_back` would otherwise schedule
        # paints on unmounted widgets, which we've seen leave the parent
        # detail screen frozen with stale Footer artefacts on resume.
        if self._user_closing:
            return
        try:
            self.query_one("#aln-body", _AlignmentBody).refresh()
        except Exception:
            pass

    def _refresh_legend(self) -> None:
        if self._user_closing:
            return
        try:
            self.query_one("#aln-phylum", Static).update(self._render_legend())
        except Exception:
            pass

    # ---- rendering ----

    def _render_body(self, width: int, height: int):
        """Build the alignment body as a Rich ``Text``.

        Returns a Text rather than a markup string because the body
        emits ~10k color spans per refresh, and Rich's markup parser
        was the dominant scroll-latency cost at that span count
        (~48 ms per repaint on a 200×60 viewport). Direct ``Text.append``
        with a ``style`` kwarg builds the same colored output without
        invoking the parser.
        """
        if not self._displayed:
            if self._loading:
                return Text(
                    "Loading alignment…",
                    style="dim",
                )
            return Text("No sequences to show.", style="dim")

        # Reserve room on the right for a per-row %ID-to-target column
        # so each homolog can carry its identity score next to its
        # sequence; the ruler line surfaces the column header.
        seq_w = max(20, width - _ID_COL_WIDTH - 4 - _IDENT_SUFFIX_W)
        # Header rows: ruler + blank + logo (N rows) + target + blank divider,
        # plus the anchor line at the end. Account for them in visible_rows.
        visible_rows = max(1, height - (_LOGO_HEIGHT + 6))

        # Drop the per-row cache when anything that affects row content
        # changes. Highlight index is intentionally NOT in the key —
        # the marker glyph is applied outside the cached body so j/k
        # repaints stay free.
        cache_key = (
            self._col_offset,
            seq_w,
            self._color_mode,
            self._midpoint,
            len(self._row_identity_pct),
        )
        if cache_key != self._row_cache_key:
            self._row_cache = {}
            self._row_cache_key = cache_key

        target_id, target_seq = self._displayed[0]
        target_short = target_id[: _ID_COL_WIDTH - 2]
        target_visible = target_seq[self._col_offset : self._col_offset + seq_w]

        body = Text(no_wrap=True, overflow="ellipsis")

        # --- Ruler row: "  col  ............:....|...... %ID  % ID    "
        body.append(
            f"  {'col':>{_ID_COL_WIDTH - 2}}  ", style="dim"
        )
        body.append(self._ruler_text(self._col_offset, seq_w))
        body.append(
            f"  {'%ID':>6}{'% ID':^{1 + _BAR_W}}\n", style="dim"
        )
        body.append("\n")

        # --- Logo rows (sequence logo: bar + letter)
        for r, logo_row in enumerate(
            self._logo_lines_text(self._col_offset, self._col_offset + seq_w)
        ):
            label = "logo" if r == 0 else ""
            body.append(
                f"  {label:<{_ID_COL_WIDTH - 2}}  ", style="dim"
            )
            body.append(logo_row)
            body.append("\n")

        # --- Target row
        target_marker = "►" if self._highlight_idx == 0 else " "
        body.append(f"{target_marker} ")
        body.append(
            f"{target_short:<{_ID_COL_WIDTH - 2}}  ",
            style="bold #2E86AB",
        )
        body.append(self._color_segment_text(target_visible, self._col_offset))
        body.append("\n\n")

        # --- Homolog rows (cached per row index, marker applied live)
        others = self._displayed[1:]
        end = min(len(others), self._row_offset + visible_rows)
        for i in range(self._row_offset, end):
            displayed_idx = i + 1
            marker = "►" if self._highlight_idx == displayed_idx else " "
            body.append(f"{marker} ")
            body.append_text(self._homolog_row_text(i, seq_w))
            body.append("\n")

        # --- Status line
        body.append("\n")
        if others:
            body.append(
                f"rows {self._row_offset + 1}-{end} of {len(others)} "
                f"(target pinned)  ·  col {self._col_offset + 1}",
                style="dim",
            )
        else:
            body.append(
                f"target only  ·  col {self._col_offset + 1}",
                style="dim",
            )
        return body

    def _render_legend(self) -> str:
        if not self._phylum_legend:
            return ""
        chips = "  ".join(
            f"[{color}]{name}[/{color}] [dim]({n})[/dim]"
            for name, color, n in self._phylum_legend
        )
        return f"[dim]Phyla:[/dim]  {chips}"

    def _homolog_row_text(self, others_idx: int, seq_w: int) -> Text:
        """Cached per-row body Text (everything after the marker glyph).

        Cache hits when the user is just scrolling vertically; that's
        the common case and reduces a 50-row repaint to dict lookups
        plus N ``append_text`` calls.
        """
        cached = self._row_cache.get(others_idx)
        if cached is not None:
            return cached

        others = self._displayed[1:]
        ident, seq = others[others_idx]
        ident_short = ident[: _ID_COL_WIDTH - 2]
        visible_seq = seq[self._col_offset : self._col_offset + seq_w]
        ident_color = self._phylum_color(self._tax_by_id.get(ident))
        ident_style = ident_color or "dim"
        displayed_idx = others_idx + 1
        pct = (
            self._row_identity_pct[displayed_idx]
            if displayed_idx < len(self._row_identity_pct) else None
        )

        row = Text(no_wrap=True)
        row.append(
            f"{ident_short:<{_ID_COL_WIDTH - 2}}  ",
            style=ident_style,
        )
        row.append(self._color_segment_text(visible_seq, self._col_offset))
        if pct is not None:
            row.append(f"  {pct:5.1f}% ", style="dim")
            filled = max(0, min(_BAR_W, round(pct / 100.0 * _BAR_W)))
            if filled:
                row.append("█" * filled, style=_identity_bar_color(pct))
            if filled < _BAR_W:
                row.append("░" * (_BAR_W - filled), style="dim")
        else:
            row.append(f"  {'—':>6} {'░' * _BAR_W}", style="dim")
        self._row_cache[others_idx] = row
        return row

    def _color_segment_text(self, segment: str, start_col: int) -> Text:
        """Same coloring logic as ``_color_segment`` but emits a Rich
        ``Text`` directly. Avoids the markup-parser pass that dominated
        scroll latency at ~10k spans per refresh.
        """
        out = Text(no_wrap=True)
        if self._color_mode == "none":
            out.append(segment)
            return out

        if self._color_mode == "biochem":
            self._coalesce_text(
                out,
                segment,
                lambda ch, _col: (
                    None if ch in "-." else _BIOCHEM_COLORS.get(ch.upper())
                ),
                start_col,
            )
            return out

        if self._color_mode == "conservation" and self._col_conservation is not None:
            from ..structure import conservation_color
            cons = self._col_conservation
            cons_len = len(cons)
            midpoint = self._midpoint

            def cons_color(ch: str, col: int):
                if ch in "-." or col >= cons_len:
                    return None
                return conservation_color(float(cons[col]), midpoint)

            self._coalesce_text(out, segment, cons_color, start_col)
            return out

        if self._color_mode in ("plddt", "sasa"):
            scalar = (
                self._target_plddt if self._color_mode == "plddt"
                else self._target_sasa
            )
            col_map = self._col_to_target_pos
            if scalar is None or col_map is None:
                out.append(segment)
                return out
            from ..structure import color_for_mode
            scalar_len = len(scalar)
            mode = self._color_mode
            map_len = len(col_map)

            def target_color(ch: str, col: int):
                if ch in "-." or col >= map_len:
                    return None
                pos = int(col_map[col])
                if pos < 0 or pos >= scalar_len:
                    return None
                return color_for_mode(float(scalar[pos]), mode)

            self._coalesce_text(out, segment, target_color, start_col)
            return out

        out.append(segment)
        return out

    @staticmethod
    def _coalesce_text(out: Text, segment: str, color_fn, start_col: int) -> None:
        """Walk ``segment``, merge consecutive same-color cells, and
        append each run to ``out`` as one ``Text.append`` call.

        The N=200 viewport × ~60 rows path emits ~10k characters; one
        run per ~2 chars in biochem mode means ~5k ``append`` calls
        per refresh, vs ~10k spans through the markup parser. Direct
        appends are about 10x cheaper than parsing + rendering markup.
        """
        run_color: Optional[str] = None
        run_start = 0
        for offset, ch in enumerate(segment):
            color = color_fn(ch, start_col + offset)
            if color != run_color:
                if offset > run_start:
                    out.append(
                        segment[run_start:offset],
                        style=run_color or "",
                    )
                run_color = color
                run_start = offset
        if run_start < len(segment):
            out.append(
                segment[run_start:],
                style=run_color or "",
            )

    def _ruler_text(self, start: int, width: int) -> Text:
        cells = []
        for i in range(width):
            pos = start + i + 1
            if pos % 10 == 0:
                cells.append("|")
            elif pos % 5 == 0:
                cells.append(":")
            else:
                cells.append(".")
        return Text("".join(cells), style="dim", no_wrap=True)

    def _logo_lines_text(self, start: int, end_exc: int) -> List[Text]:
        """Same as ``_logo_lines`` but emits two Rich ``Text`` rows
        with ``append(style=...)`` instead of markup strings."""
        empty = [Text("", no_wrap=True), Text("", no_wrap=True)]
        if not self._logo_top:
            return empty
        cap = len(self._logo_top)
        end_exc = min(end_exc, cap)
        if start >= end_exc:
            return empty

        bar = Text(no_wrap=True)
        letters = Text(no_wrap=True)
        # Coalesce consecutive same-color cells into single appends
        # (logo rows are usually short — 100-300 cells — but the same
        # microopt applies).
        run_color: Optional[str] = None
        run_bar: List[str] = []
        run_letters: List[str] = []
        for col in range(start, end_exc):
            top = self._logo_top[col]
            if top == " ":
                color = None
                bch = " "
                lch = " "
            else:
                color = _BIOCHEM_COLORS.get(top.upper(), "white")
                bch = _BAR_GLYPHS[self._logo_level[col]]
                lch = top
            if color != run_color:
                if run_bar:
                    bar.append("".join(run_bar), style=run_color or "")
                    letters.append("".join(run_letters), style=run_color or "")
                run_color = color
                run_bar = [bch]
                run_letters = [lch]
            else:
                run_bar.append(bch)
                run_letters.append(lch)
        if run_bar:
            bar.append("".join(run_bar), style=run_color or "")
            letters.append("".join(run_letters), style=run_color or "")
        return [bar, letters]

