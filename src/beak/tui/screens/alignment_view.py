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
from textual.app import ComposeResult
from textual.binding import Binding
from textual.screen import Screen
from textual.widgets import Footer, Header, Static


_ID_COL_WIDTH = 22
_PAN_STEP = 10
_PAGE_ROWS = 10
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


def _mean_identity_to_target(sequences: List[Tuple[str, str]]) -> Optional[float]:
    """Mean per-residue identity of each homolog to the target.

    Identity is computed over target positions where both target and
    homolog have a non-gap residue. Returns percentage (0-100) or None.
    """
    if len(sequences) < 2:
        return None
    target = sequences[1 - 1][1]  # row 0 = target
    target_positions = [i for i, c in enumerate(target) if c not in "-."]
    if not target_positions:
        return None
    total = 0.0
    n = 0
    for _, seq in sequences[1:]:
        matches = 0
        compared = 0
        for i in target_positions:
            if i >= len(seq):
                break
            ha = seq[i]
            if ha in "-.":
                continue
            compared += 1
            if ha == target[i]:
                matches += 1
        if compared > 0:
            total += matches / compared
            n += 1
    if n == 0:
        return None
    return total / n * 100.0


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


class AlignmentViewerScreen(Screen):
    """Full-screen alignment view."""

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
    """

    _COLOR_MODES = ("biochem", "conservation", "none")
    # None = no ungap. Others = drop columns where gap fraction >= threshold.
    _GAP_THRESHOLDS = (None, 0.9, 0.8, 0.7)

    def __init__(self, project) -> None:
        super().__init__()
        self._project = project
        # Original parsed sequences (target row at index 0).
        self._sequences: List[Tuple[str, str]] = []
        # Active view: same as _sequences when not ungapped, otherwise filtered.
        self._displayed: List[Tuple[str, str]] = []
        self._aligned: bool = False
        self._row_offset: int = 0
        self._col_offset: int = 0
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
        # Per-column logo cache aligned to _displayed. Recomputed on
        # ungap; consumed cheaply by `_logo_lines` / `_consensus_line`
        # so scroll/pan don't repeat O(n_seqs * n_visible_cols) work.
        self._logo_top: List[str] = []
        self._logo_level: List[int] = []
        self._logo_freq: List[float] = []
        # Diversity metrics — computed once at load.
        self._mean_identity: Optional[float] = None
        self._neff: Optional[int] = None
        # Taxonomy: seq_id → phylum (None when unresolved).
        self._tax_by_id: dict = {}
        self._phylum_legend: list = []  # ordered (phylum, color, count)

    def compose(self) -> ComposeResult:
        yield Header()
        yield _AlignmentBody(self, id="aln-body")
        yield Static("", id="aln-legend")
        yield Footer()

    def on_mount(self) -> None:
        self._load()
        self._update_title()
        try:
            self.query_one("#aln-legend", Static).update(self._render_legend())
        except Exception:
            pass

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

    def _load(self) -> None:
        homologs_dir = self._project.active_homologs_dir()
        target_seq = self._project.target_sequence()
        target_id = (
            self._project.manifest().get("target", {}).get("uniprot_id")
            or self._project.name
        )

        aln = homologs_dir / "alignment.fasta"
        hits = homologs_dir / "sequences.fasta"

        path = None
        if aln.exists():
            path = aln
            self._aligned = True
        elif hits.exists():
            path = hits
            self._aligned = False

        if path is None:
            self._sequences = [(target_id, target_seq)]
            return

        try:
            from Bio import SeqIO
            records = [(r.id, str(r.seq)) for r in SeqIO.parse(str(path), "fasta")]
        except Exception:
            records = []

        if not records:
            self._sequences = [(target_id, target_seq)]
            return

        self._sequences = records
        self._rebuild_displayed()
        # Diversity metrics are nice-to-haves; don't let a bug in them
        # take down the whole alignment view.
        try:
            self._mean_identity = _mean_identity_to_target(self._sequences)
        except Exception:
            self._mean_identity = None
        try:
            self._neff = _effective_n(self._sequences, threshold=0.8)
        except Exception:
            self._neff = None
        self._load_taxonomy()

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

    def _rebuild_displayed(self) -> None:
        """Refresh `_displayed` (and column conservation) from `_sequences`
        and the current gap threshold. Cheap when no ungap is active."""
        if self._gap_threshold is None or not self._aligned:
            self._displayed = list(self._sequences)
        else:
            self._displayed = self._ungap(self._sequences, self._gap_threshold)
        self._col_conservation = (
            self._compute_column_conservation(self._displayed)
            if self._aligned else None
        )
        self._rebuild_logo_cache()

    def _rebuild_logo_cache(self) -> None:
        """Precompute (top-char, intensity-level) per alignment column.

        Without this the logo + consensus rows walk the full sequence
        list and run Counter on every column on every paint — which is
        the dominant cost when scrolling, even for a few-hundred-row
        alignment. Building it once per `_displayed` change collapses
        the per-frame work to a slice + markup join.
        """
        if not self._displayed:
            self._logo_top: List[str] = []
            self._logo_level: List[int] = []
            self._logo_freq: List[float] = []
            return

        # Stack into a 2-D character array once, then iterate columns
        # vectorised through Counter — way cheaper than the per-paint
        # Python-level scan that was here before.
        max_len = max(len(s) for _, s in self._displayed)
        cols = max_len
        seqs = [s.ljust(max_len, "-") for _, s in self._displayed]

        n_glyphs = len(_BAR_GLYPHS)
        top: List[str] = [" "] * cols
        level: List[int] = [0] * cols
        freqs: List[float] = [0.0] * cols
        for col in range(cols):
            chars = [s[col] for s in seqs if s[col] not in "-."]
            if not chars:
                continue
            t, count = Counter(chars).most_common(1)[0]
            f = count / len(chars)
            top[col] = t
            level[col] = min(n_glyphs - 1, int(round(f * (n_glyphs - 1))))
            freqs[col] = f

        self._logo_top = top
        self._logo_level = level
        self._logo_freq = freqs

    def _ungap(self, sequences: List[Tuple[str, str]], threshold: float) -> List[Tuple[str, str]]:
        """Drop columns where gap-fraction >= `threshold`. Vectorised."""
        if not sequences:
            return []
        max_len = max(len(s) for _, s in sequences)
        arr = np.array([
            list(s.ljust(max_len, "-")) for _, s in sequences
        ])
        gap_mask = (arr == "-") | (arr == ".")
        gap_frac = gap_mask.sum(axis=0) / arr.shape[0]
        keep = gap_frac < threshold
        return [
            (ident, "".join(arr[i, keep]))
            for i, (ident, _) in enumerate(sequences)
        ]

    @staticmethod
    def _compute_column_conservation(seqs: List[Tuple[str, str]]) -> np.ndarray:
        """Per-column identity fraction, [0, 100]. Skips gap-only columns."""
        if not seqs:
            return np.zeros(0, dtype=np.float32)
        max_len = max(len(s) for _, s in seqs)
        out = np.zeros(max_len, dtype=np.float32)
        for col in range(max_len):
            chars = [
                s[col] for _, s in seqs
                if col < len(s) and s[col] not in "-."
            ]
            if not chars:
                continue
            top, count = Counter(chars).most_common(1)[0]  # noqa: F841
            out[col] = count / len(chars) * 100.0
        return out

    # ---- key actions ----

    def action_back(self) -> None:
        # Clear the alignment-specific sub-title so the project detail
        # screen doesn't inherit "alignment · ..." after we pop back.
        self.app.sub_title = ""
        self.app.pop_screen()

    def _max_row_offset(self) -> int:
        # Target stays pinned; offset paginates rows[1:].
        return max(0, len(self._displayed) - 2)

    def action_row_up(self) -> None:
        self._row_offset = max(0, self._row_offset - 1)
        self._refresh_body()

    def action_row_down(self) -> None:
        self._row_offset = min(self._max_row_offset(), self._row_offset + 1)
        self._refresh_body()

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

    def action_cycle_gap(self) -> None:
        if not self._aligned:
            return  # ungap is meaningless on raw hits
        idx = self._GAP_THRESHOLDS.index(self._gap_threshold)
        self._gap_threshold = self._GAP_THRESHOLDS[(idx + 1) % len(self._GAP_THRESHOLDS)]
        # Reset offsets — column space changed.
        self._col_offset = 0
        self._rebuild_displayed()
        self._update_title()
        self._refresh_body()

    def _refresh_body(self) -> None:
        try:
            self.query_one("#aln-body", _AlignmentBody).refresh()
        except Exception:
            pass
        try:
            self.query_one("#aln-legend", Static).update(self._render_legend())
        except Exception:
            pass

    # ---- rendering ----

    def _render_body(self, width: int, height: int) -> str:
        if not self._displayed:
            return "[dim]No sequences to show.[/dim]"

        seq_w = max(20, width - _ID_COL_WIDTH - 4)
        ruler = self._ruler(self._col_offset, seq_w)
        n = len(self._displayed)

        # Header rows: ruler + blank + logo (N rows) + target + blank divider,
        # plus the anchor line at the end. Account for them in visible_rows.
        visible_rows = max(1, height - (_LOGO_HEIGHT + 6))

        target_id, target_seq = self._displayed[0]
        target_short = target_id[: _ID_COL_WIDTH - 2]
        target_visible = target_seq[self._col_offset : self._col_offset + seq_w]
        target_colored = self._color_segment(target_visible, self._col_offset)

        lines = [
            f"  [dim]{'col':>{_ID_COL_WIDTH - 2}}[/dim]  {ruler}",
            "",
        ]
        # Sequence logo: N rows where each row holds the next-most-frequent
        # AA. Letters span vertically in proportion to their column freq.
        for r, logo_row in enumerate(
            self._logo_lines(self._col_offset, self._col_offset + seq_w)
        ):
            label = "logo" if r == 0 else ""
            lines.append(
                f"  [dim]{label:<{_ID_COL_WIDTH - 2}}[/dim]  {logo_row}"
            )
        lines.append(
            f"  [bold #2E86AB]{target_short:<{_ID_COL_WIDTH - 2}}[/bold #2E86AB]  {target_colored}"
        )
        lines.append("")

        others = self._displayed[1:]
        end = min(len(others), self._row_offset + visible_rows)
        for i in range(self._row_offset, end):
            ident, seq = others[i]
            ident_short = ident[: _ID_COL_WIDTH - 2]
            visible_seq = seq[self._col_offset : self._col_offset + seq_w]
            colored = self._color_segment(visible_seq, self._col_offset)
            # Phylum-color the ID column when taxonomy is available.
            ident_color = self._phylum_color(self._tax_by_id.get(ident))
            ident_style = ident_color or "dim"
            lines.append(
                f"  [{ident_style}]{ident_short:<{_ID_COL_WIDTH - 2}}[/{ident_style}]  {colored}"
            )

        # Per-render status line (visible-window counters); the phylum
        # color key has moved out to the dock-pinned `#aln-legend` so it
        # stays put no matter how many rows scroll past.
        lines.append("")
        if others:
            lines.append(
                f"[dim]rows {self._row_offset + 1}-{end} of {len(others)} (target pinned)  ·  "
                f"col {self._col_offset + 1}[/dim]"
            )
        else:
            lines.append(f"[dim]target only  ·  col {self._col_offset + 1}[/dim]")
        return "\n".join(lines)

    def _render_legend(self) -> str:
        if not self._phylum_legend:
            return ""
        chips = "  ".join(
            f"[{color}]{name}[/{color}] [dim]({n})[/dim]"
            for name, color, n in self._phylum_legend
        )
        return f"[dim]Phyla:[/dim]  {chips}"

    def _color_segment(self, segment: str, start_col: int) -> str:
        """Apply the active color mode to a slice of one row.

        Runs of consecutive same-color cells are coalesced into a
        single `[color]...[/color]` span so a row of N residues emits
        ~10-20 markup spans instead of one per character. Big win
        for scroll latency on long visible windows.
        """
        if self._color_mode == "none":
            return segment

        if self._color_mode == "biochem":
            return self._coalesce(
                segment,
                lambda ch, _col: (
                    None if ch in "-." else _BIOCHEM_COLORS.get(ch.upper())
                ),
                start_col,
            )

        if self._color_mode == "conservation" and self._col_conservation is not None:
            from ..structure import conservation_color
            cons = self._col_conservation
            cons_len = len(cons)
            midpoint = self._midpoint

            def cons_color(ch: str, col: int):
                if ch in "-." or col >= cons_len:
                    return None
                return conservation_color(float(cons[col]), midpoint)

            return self._coalesce(segment, cons_color, start_col)

        return segment

    @staticmethod
    def _coalesce(segment: str, color_fn, start_col: int) -> str:
        """Walk `segment` and merge consecutive same-color cells into one span."""
        parts: List[str] = []
        run_color: Optional[str] = None
        run_chars: List[str] = []
        for offset, ch in enumerate(segment):
            color = color_fn(ch, start_col + offset)
            if color != run_color:
                if run_chars:
                    if run_color is None:
                        parts.append("".join(run_chars))
                    else:
                        parts.append(
                            f"[{run_color}]{''.join(run_chars)}[/{run_color}]"
                        )
                run_color = color
                run_chars = [ch]
            else:
                run_chars.append(ch)
        if run_chars:
            if run_color is None:
                parts.append("".join(run_chars))
            else:
                parts.append(
                    f"[{run_color}]{''.join(run_chars)}[/{run_color}]"
                )
        return "".join(parts)

    def _logo_lines(self, start: int, end_exc: int) -> list:
        """Two-row sequence logo using the precomputed per-column cache."""
        if not self._logo_top:
            return ["", ""]
        cap = len(self._logo_top)
        end_exc = min(end_exc, cap)
        if start >= end_exc:
            return ["", ""]

        bar_chars: List[str] = []
        letter_chars: List[str] = []
        for col in range(start, end_exc):
            top = self._logo_top[col]
            if top == " ":
                bar_chars.append(" ")
                letter_chars.append(" ")
                continue
            color = _BIOCHEM_COLORS.get(top.upper(), "white")
            glyph = _BAR_GLYPHS[self._logo_level[col]]
            bar_chars.append(f"[{color}]{glyph}[/{color}]")
            letter_chars.append(f"[{color}]{top}[/{color}]")
        return ["".join(bar_chars), "".join(letter_chars)]

    def _consensus_line(self, start: int, end_exc: int) -> str:
        """One-line consensus from the cache."""
        if not self._logo_top:
            return ""
        cap = len(self._logo_top)
        end_exc = min(end_exc, cap)
        if start >= end_exc:
            return ""

        parts: List[str] = []
        for col in range(start, end_exc):
            top = self._logo_top[col]
            if top == " ":
                parts.append(" ")
                continue
            freq = self._logo_freq[col]
            color = _BIOCHEM_COLORS.get(top.upper(), "white")
            if freq > 0.7:
                parts.append(f"[bold {color}]{top}[/bold {color}]")
            elif freq < 0.3:
                parts.append(f"[dim {color}]{top}[/dim {color}]")
            else:
                parts.append(f"[{color}]{top}[/{color}]")
        return "".join(parts)

    def _ruler(self, start: int, width: int) -> str:
        cells = []
        for i in range(width):
            pos = start + i + 1
            if pos % 10 == 0:
                cells.append("|")
            elif pos % 5 == 0:
                cells.append(":")
            else:
                cells.append(".")
        return "[dim]" + "".join(cells) + "[/dim]"
